import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for softplus/relu
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Project imports
from model_ddpm import ContextUnet
from diffusion import Diffusion
from dataset import SRDataset

# Added imports for Geometric Loss
from loss import GeometricLossSeparate, estimate_s_inv_from_dataset
from utils import load_emulator  # Assumes utils.py exists as in your UNet script

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN = config["TRAIN_METADATA_FILE"]
METADATA_VAL = config["VAL_METADATA_FILE"]
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 100
PATIENCE = 10
NUM_WORKERS = 4
EXPERIMENT_NAME = "DDPM_SR_Geometric"

# --- Geometric Loss Configuration ---
GEOMETRIC_START_EPOCH = 20  # Warm-up phase (pure MSE)
GEOMETRIC_WEIGHT = 0.1  # Scaling factor for auxiliary loss
GEOMETRIC_T_THRESHOLD = 250  # Only apply loss when t < 250 (cleaner signal)
TRUST_TAU = config.get("TRUST_TAU", 0.5)  # Trust gate decay rate
EMULATOR_PATH = config.get("EMULATOR_CHECKPOINT_PATH", "checkpoints/emulator_best.pth")


class DataDenormalizer:
    """
    Handles the inverse transformation of data from Model Space ([0,1])
    back to Physical Space (mm/h).
    """

    def __init__(self, stats_path):
        try:
            # Load the numpy array
            data = np.load(stats_path)
            self.max_val = float(data.item())
            print(f"Loaded Denormalizer. Max Val (Log Space): {self.max_val:.4f}")
        except FileNotFoundError:
            print(
                f"Warning: Scaling stats not found at {stats_path}. Defaulting to 1.0."
            )
            self.max_val = 1.0
        except Exception as e:
            print(
                f"Warning: Failed to load denormalizer stats ({e}). Defaulting to 1.0."
            )
            self.max_val = 1.0

    def unnormalize(self, x_norm):
        """
        Inverse Pipeline (Numpy):
        1. Scale up: x' = x_norm * max_val
        2. Inverse Log: x_phys = exp(x') - 1
        """
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()

        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        # Physical constraint: Precip >= 0
        return np.maximum(x_phys, 0.0)

    def unnormalize_torch(self, x_norm):
        """
        Inverse Pipeline (Differentiable Torch):
        Used for calculating geometric loss on the GPU.
        """
        # Ensure max_val is a tensor on the correct device
        if not isinstance(self.max_val, torch.Tensor):
            self.max_val = torch.tensor(
                self.max_val, device=x_norm.device, dtype=x_norm.dtype
            )

        x_scaled = x_norm * self.max_val
        x_phys = torch.expm1(x_scaled)
        # Use ReLU to enforce non-negativity in the graph
        return F.relu(x_phys)


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return True


def compute_physical_metrics(real_batch, gen_batch, drizzle_threshold=0.1):
    """
    Computes physical metrics.
    Expects INPUTS to be in PHYSICAL UNITS (mm/h).
    """
    # --- Apply Threshold ---
    # Zero out background noise < 0.1 mm/h to match SR evaluation standards
    real_batch = real_batch * (real_batch > drizzle_threshold).astype(float)
    gen_batch = gen_batch * (gen_batch > drizzle_threshold).astype(float)

    real_flat = real_batch.flatten()
    gen_flat = gen_batch.flatten()

    # 1. Wasserstein Distance
    wd = wasserstein_distance(real_flat, gen_flat)

    # 2. Max Intensity Error
    real_max = np.max(real_flat)
    gen_max = np.max(gen_flat)
    max_err = abs(real_max - gen_max)

    return {"wasserstein_dist": wd, "max_intensity_err": max_err}


def save_sample_images(model, diffusion, loader, device, out_dir, epoch, denormalizer):
    model.eval()
    try:
        X, Y, _ = next(iter(loader))
    except StopIteration:
        return

    X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

    n_samples = min(50, X.shape[0])
    X_sample = X[:n_samples]
    Y_sample = Y[:n_samples]

    # --- INFERENCE GATE LOGIC ---
    x_generated = torch.zeros(
        (n_samples, 1, diffusion.img_size, diffusion.img_size), device=device
    )

    # Channel 0 is Precip
    input_precip = X_sample[:, 0, :, :]
    is_wet_mask = input_precip.amax(dim=(1, 2)) > 1e-6
    wet_indices = torch.where(is_wet_mask)[0]
    n_wet = len(wet_indices)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            if n_wet > 0:
                X_wet = X_sample[wet_indices]
                gen_wet = diffusion.sample(model, n=n_wet, conditions=X_wet)
                x_generated[wet_indices] = gen_wet

    # --- VISUALIZATION (PHYSICAL UNITS) ---
    fig, axs = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axs = axs[None, :]

    # Move to CPU
    X_cpu = X_sample.float().cpu().numpy()
    Y_cpu = Y_sample.float().cpu().numpy()
    Gen_cpu = x_generated.float().cpu().numpy()

    # Denormalize Precip channels
    X_phys = denormalizer.unnormalize(X_cpu[:, 0])
    Y_phys = denormalizer.unnormalize(Y_cpu[:, 0])
    Gen_phys = denormalizer.unnormalize(Gen_cpu[:, 0])

    for i in range(n_samples):
        vmax = max(np.max(Y_phys[i]), np.max(Gen_phys[i]), 1.0)

        im1 = axs[i, 0].imshow(X_phys[i], cmap="Blues", origin="lower", vmax=vmax)
        axs[i, 0].set_title(f"LR Input (mm/h)\nMax: {np.max(X_phys[i]):.2f}")
        axs[i, 0].axis("off")
        plt.colorbar(im1, ax=axs[i, 0], fraction=0.046, pad=0.04)

        im2 = axs[i, 1].imshow(Y_phys[i], cmap="Blues", origin="lower", vmax=vmax)
        axs[i, 1].set_title(f"Ground Truth (mm/h)\nMax: {np.max(Y_phys[i]):.2f}")
        axs[i, 1].axis("off")
        plt.colorbar(im2, ax=axs[i, 1], fraction=0.046, pad=0.04)

        im3 = axs[i, 2].imshow(Gen_phys[i], cmap="Blues", origin="lower", vmax=vmax)
        axs[i, 2].set_title(f"Generated (mm/h)\nMax: {np.max(Gen_phys[i]):.2f}")
        axs[i, 2].axis("off")
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"))
    plt.close()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    print(f"Active Device: {torch.cuda.get_device_name(0)}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    # --- Data & Denormalizer ---
    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    # Initialize Denormalizer
    stats_path = os.path.join(
        PREPROCESSED_DATA_DIR, "log_transformed_precip_max_val.npy"
    )
    denormalizer = DataDenormalizer(stats_path)

    train_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, METADATA_TRAIN, DEM_DATA_DIR, dem_stats, split="train"
    )
    val_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, METADATA_VAL, DEM_DATA_DIR, dem_stats, split="validation"
    )

    # --- Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # --- Geometric Loss Setup ---
    print(f"Initializing Geometric Loss (Start Epoch: {GEOMETRIC_START_EPOCH})...")

    # 1. Estimate S_inv (Covariance) from training data
    # Note: This might take a minute.
    S_inv_tensors = estimate_s_inv_from_dataset(
        train_dataset, num_samples=2000, device=device
    )
    geometric_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
        device
    )

    # 2. Load Emulator
    # We freeze it to ensure we don't accidentally train the emulator
    print(f"Loading Emulator from {EMULATOR_PATH}...")
    emulator = load_emulator(EMULATOR_PATH, config, device)
    emulator.eval()
    for param in emulator.parameters():
        param.requires_grad = False

    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_geom": [],
        "avg_trust": [],
        "timestamp": [],
    }

    print("Starting DDPM Training (Mixed Precision with Geometric Fine-Tuning)...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0
        running_geom = 0.0
        running_trust = 0.0

        # Unpack Y_gamma (dataset returns: input, target, target_gamma)
        for X, Y, Y_gamma in pbar:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            Y_gamma = Y_gamma.to(device, non_blocking=True)  # [B, 3] Log-space targets

            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                predicted_noise = model(x_t, t, X)
                loss_mse = mse(noise, predicted_noise)

                # --- Geometric Loss Logic (Curriculum + Time Gated + Trust Gated) ---
                loss_geom = torch.tensor(0.0, device=device)
                avg_trust_val = 1.0

                if epoch >= GEOMETRIC_START_EPOCH:
                    # A. Analytic Reconstruction of x0
                    # x0 ≈ (x_t - sqrt(1-alpha_hat) * eps_pred) / sqrt(alpha_hat)
                    alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
                    sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
                    sqrt_one_minus = torch.sqrt(1 - alpha_hat_t)

                    # Prevent division by zero stability check
                    pred_x0 = (x_t - sqrt_one_minus * predicted_noise) / (
                        sqrt_alpha_hat + 1e-8
                    )

                    # B. Time-Gating
                    # Only apply geometric loss where signal is strong (t is small).
                    mask_time = (t < GEOMETRIC_T_THRESHOLD).float()

                    if mask_time.sum() > 0:
                        # --- C. TRUST GATE CALCULATION ---
                        with torch.no_grad():
                            # 1. Unnormalize Ground Truth Y to Physical Units
                            Y_phys = denormalizer.unnormalize_torch(Y)
                            Y_phys = (
                                Y_phys * (Y_phys > 0.1).float()
                            )  # Sparsity consistency

                            # 2. Emulator Prediction on Ground Truth
                            gamma_truth_phys = emulator(Y_phys)
                            gamma_truth_log_pred = torch.log1p(gamma_truth_phys)

                            # 3. Calculate Emulator Error
                            diff_trust = gamma_truth_log_pred - Y_gamma
                            emu_error_sq = diff_trust.pow(2).mean(dim=1)  # [B]

                            # 4. Compute Trust Weights
                            trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_sq)
                            avg_trust_val = trust_weights.mean().item()

                        # --- D. Loss Computation ---

                        # 1. Prepare Predicted x0 (Physical)
                        pred_x0_phys = denormalizer.unnormalize_torch(pred_x0)
                        pred_x0_phys = pred_x0_phys * (pred_x0_phys > 0.1).float()

                        # 2. Emulator Prediction on Predicted x0
                        pred_gamma_phys = emulator(pred_x0_phys)
                        pred_gamma_log = torch.log1p(pred_gamma_phys)

                        # 3. Compute Raw Geometric Loss (Per Sample)
                        raw_geom_loss = geometric_criterion(pred_gamma_log, Y_gamma)

                        # 4. Apply Trust Weights AND Time Mask
                        weighted_loss = raw_geom_loss * trust_weights * mask_time

                        # Normalize by number of valid samples
                        loss_geom = weighted_loss.sum() / (mask_time.sum() + 1e-8)

                # Total Loss
                total_loss = loss_mse + (GEOMETRIC_WEIGHT * loss_geom)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_mse.item()
            running_geom += loss_geom.item()
            running_trust += avg_trust_val

            pbar.set_postfix(
                MSE=f"{loss_mse.item():.4f}",
                Geom=f"{loss_geom.item():.4f}",
                Trust=f"{avg_trust_val:.2f}",
            )

        avg_loss = running_loss / len(train_loader)
        avg_geom = running_geom / len(train_loader)
        avg_trust = running_trust / len(train_loader)
        print(
            f"Epoch {epoch+1} Train MSE: {avg_loss:.6f} | Geom: {avg_geom:.6f} | Avg Trust: {avg_trust:.2f}"
        )

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y, _ in val_loader:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)

                with torch.amp.autocast("cuda"):
                    predicted_noise = model(x_t, t, X)
                    loss = mse(noise, predicted_noise)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val MSE: {avg_val_loss:.6f}")

        # --- Logging ---
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_geom"].append(avg_geom)
        history["avg_trust"].append(avg_trust)
        history["timestamp"].append(time.strftime("%H:%M:%S"))
        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, "loss_history.csv"), index=False
        )

        if (epoch + 1) % 5 == 0 or (epoch == 0):
            save_sample_images(
                model, diffusion, val_loader, device, out_dir, epoch + 1, denormalizer
            )

        torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_latest.pth"))
        if epoch > 0 and avg_val_loss < min(history["val_loss"][:-1]):
            torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_best.pth"))

        # --- Physical Validation ---
        if (epoch + 1) % 5 == 0:
            print(f"Running Physical Validation (Sampling over {10} batches)...")
            model.eval()

            # Storage for aggregating metrics
            val_metrics = {"wd": [], "max_err": []}
            NUM_PHYS_BATCHES = 10

            with torch.no_grad():
                for i, (X_val, Y_val, _) in enumerate(val_loader):
                    if i >= NUM_PHYS_BATCHES:
                        break

                    X_val, Y_val = X_val.to(device), Y_val.to(device)

                    # --- FILTER WET SAMPLES ---
                    input_precip = X_val[:, 0, :, :]
                    is_wet = input_precip.amax(dim=(1, 2)) > 1e-6
                    wet_indices = torch.where(is_wet)[0]

                    if len(wet_indices) == 0:
                        continue

                    X_wet = X_val[wet_indices]
                    Y_wet = Y_val[wet_indices]

                    # Sample from model
                    gen_wet = diffusion.sample(
                        model, n=len(wet_indices), conditions=X_wet
                    )

                    # --- PHYSICAL TRANSFORMATION ---
                    Y_cpu = Y_wet.cpu().numpy().squeeze()
                    Gen_cpu = gen_wet.cpu().numpy().squeeze()

                    Y_phys = denormalizer.unnormalize(Y_cpu)
                    Gen_phys = denormalizer.unnormalize(Gen_cpu)

                    # Compute metrics for this batch
                    batch_metrics = compute_physical_metrics(Y_phys, Gen_phys)

                    val_metrics["wd"].append(batch_metrics["wasserstein_dist"])
                    val_metrics["max_err"].append(batch_metrics["max_intensity_err"])

            # --- AGGREGATE RESULTS ---
            if len(val_metrics["wd"]) > 0:
                mean_wd = np.mean(val_metrics["wd"])
                mean_max_err = np.mean(val_metrics["max_err"])

                print(
                    f"Epoch {epoch+1} Physical Metrics (Avg over {NUM_PHYS_BATCHES} batches):"
                )
                print(f"  > Wasserstein Dist: {mean_wd:.4f}")
                print(f"  > Max Intensity Err: {mean_max_err:.4f}")

                # Pass the meaningful average to early stopper
                early_stopper(mean_wd)
            else:
                print(
                    "Warning: No wet samples found in validation subset. Skipping metrics."
                )


if __name__ == "__main__":
    main()
