import torch
import torch.nn as nn
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

from model_ddpm import ContextUnet
from diffusion import Diffusion
from dataset import SRDataset

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
PATIENCE = 20
NUM_WORKERS = 4
EXPERIMENT_NAME = "DDPM_SR"


class DataDenormalizer:
    """
    Handles the inverse transformation of data from Model Space ([0,1])
    back to Physical Space (mm/h).
    """

    def __init__(self, stats_path):
        try:
            # Load the numpy array
            data = np.load(stats_path)

            # FIX: Use .item() instead of [0].
            # .item() works for both 0-d scalars (array(5.0)) and 1-d arrays of size 1 (array([5.0]))
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
        Inverse Pipeline:
        1. Scale up: x' = x_norm * max_val
        2. Inverse Log: x_phys = exp(x') - 1
        """
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()

        x_scaled = x_norm * self.max_val
        x_phys = np.expm1(x_scaled)
        # Physical constraint: Precip >= 0
        return np.maximum(x_phys, 0.0)


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


def compute_physical_metrics(real_batch, gen_batch):
    """
    Computes physical metrics.
    Expects INPUTS to be in PHYSICAL UNITS (mm/h).
    """
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

    n_samples = min(10, X.shape[0])
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
    # X_cpu is [N, 2, H, W] -> index 0 is Precip
    X_phys = denormalizer.unnormalize(X_cpu[:, 0])
    Y_phys = denormalizer.unnormalize(Y_cpu[:, 0])
    Gen_phys = denormalizer.unnormalize(Gen_cpu[:, 0])

    for i in range(n_samples):
        # Plotting Physical Values (mm/h)
        # Use a fixed vmax for consistent visualization if desired, or auto-scale
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

    # --- Loaders (Fixed Shuffling) ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # ENABLED: Random shuffle for training
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # DISABLED: Deterministic order for validation
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mae = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda")
    early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "timestamp": []}

    print("Starting DDPM Training (Mixed Precision)...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        for X, Y, _ in pbar:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                predicted_noise = model(x_t, t, X)
                loss = mae(noise, predicted_noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(MAE=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train MAE: {avg_loss:.6f}")

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
                    loss = mae(noise, predicted_noise)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val MAE: {avg_val_loss:.6f}")

        # --- Logging ---
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
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
            print("Running Physical Validation (Sampling)...")
            model.eval()

            X_val, Y_val, _ = next(iter(val_loader))
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            # --- APPLY INFERENCE GATE TO METRICS ---
            # To get accurate metrics, we should only assess the model on wet inputs
            # Otherwise, the "Perfect Zeros" from dry inputs will skew the Wasserstein distance
            input_precip = X_val[:, 0, :, :]
            is_wet = input_precip.amax(dim=(1, 2)) > 1e-6
            wet_indices = torch.where(is_wet)[0]

            if len(wet_indices) == 0:
                print("No wet samples in validation probe batch. Skipping metrics.")
                continue

            X_wet = X_val[wet_indices]
            Y_wet = Y_val[wet_indices]

            with torch.no_grad():
                gen_wet = diffusion.sample(model, n=len(wet_indices), conditions=X_wet)

            # --- PHYSICAL TRANSFORMATION ---
            # Unnormalize to mm/h
            Y_cpu = Y_wet.cpu().numpy().squeeze()
            Gen_cpu = gen_wet.cpu().numpy().squeeze()

            Y_phys = denormalizer.unnormalize(Y_cpu)
            Gen_phys = denormalizer.unnormalize(Gen_cpu)

            metrics = compute_physical_metrics(Y_phys, Gen_phys)

            print(f"Epoch {epoch+1} Physical Metrics:")
            print(f"  > Wasserstein Dist: {metrics['wasserstein_dist']:.4f}")
            print(f"  > Max Intensity Err: {metrics['max_intensity_err']:.4f}")

            early_stopper(metrics["wasserstein_dist"])


if __name__ == "__main__":
    main()
