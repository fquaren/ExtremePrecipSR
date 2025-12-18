import torch
import torch.nn as nn
import torch.nn.functional as F
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
import argparse
from scipy.stats import wasserstein_distance
import copy
import matplotlib.colors as mcolors

# Project imports
from model_ddpm import ContextUnet
from diffusion import Diffusion
from dataset import SRDataset
from loss import GeometricLossSeparate, estimate_s_inv_from_dataset
from utils import load_emulator

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
PATIENCE = 5
NUM_WORKERS = 24
EXPERIMENT_NAME = "DDPM_SR_Geometric"

# --- Geometric Loss Configuration ---
GEOMETRIC_TARGET_WEIGHT = 1  # The final weight we want to reach
GEOMETRIC_WARMUP_EPOCHS = 5  # Curriculum: Ramp up over 5 epochs
GEOMETRIC_T_THRESHOLD = 250
TRUST_TAU = config.get("TRUST_TAU", 0.5)
EMULATOR_PATH = config.get("EMULATOR_CHECKPOINT_PATH", "checkpoints/emulator_best.pth")


class DataDenormalizer:
    def __init__(self, stats_path):
        try:
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
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.cpu().numpy()
        scale_val = (
            self.max_val.item() if hasattr(self.max_val, "item") else self.max_val
        )
        x_scaled = x_norm * scale_val
        x_phys = np.expm1(x_scaled)
        return np.maximum(x_phys, 0.0)

    def unnormalize_torch(self, x_norm):
        max_val_tensor = torch.tensor(
            self.max_val, device=x_norm.device, dtype=x_norm.dtype
        )
        x_scaled = x_norm * max_val_tensor
        x_phys = torch.expm1(x_scaled)
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
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            return False

    def reset(self):
        if self.verbose:
            print("Resetting Early Stopping counter and best score.")
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def state_dict(self):
        return {
            "patience": self.patience,
            "counter": self.counter,
            "best_score": self.best_score,
            "early_stop": self.early_stop,
            "val_loss_min": self.val_loss_min,
            "delta": self.delta,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.early_stop = state_dict["early_stop"]
        self.val_loss_min = state_dict["val_loss_min"]
        self.delta = state_dict["delta"]


def compute_physical_metrics(real_batch, gen_batch, drizzle_threshold=0.1):
    real_batch = real_batch * (real_batch > drizzle_threshold).astype(float)
    gen_batch = gen_batch * (gen_batch > drizzle_threshold).astype(float)
    real_flat = real_batch.flatten()
    gen_flat = gen_batch.flatten()

    if len(real_flat) == 0 or len(gen_flat) == 0:
        return {"wasserstein_dist": 0.0, "max_intensity_err": 0.0}

    wd = wasserstein_distance(real_flat, gen_flat)
    real_max = np.max(real_flat) if len(real_flat) > 0 else 0
    gen_max = np.max(gen_flat) if len(gen_flat) > 0 else 0
    max_err = abs(real_max - gen_max)
    return {"wasserstein_dist": wd, "max_intensity_err": max_err}


# --- NEW HELPER FUNCTION TO AVOID DUPLICATION ---
def compute_geometric_loss_component(
    diffusion,
    emulator,
    criterion,
    denormalizer,
    x_t,
    predicted_noise,
    t,
    Y,
    Y_gamma,
    compute_trust=True,
):
    """
    Computes the geometric loss component.
    Used in both Training (with gradients) and Validation (no_grad).
    """
    loss_geom = torch.tensor(0.0, device=x_t.device)
    avg_trust_val = 1.0

    # 1. Reconstruct x0 (Approximation)
    alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
    sqrt_alpha_hat = torch.sqrt(alpha_hat_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_hat_t)

    pred_x0 = (x_t - sqrt_one_minus * predicted_noise) / (sqrt_alpha_hat + 1e-8)

    # 2. Masking (Apply only for t < Threshold)
    mask_time = (t < GEOMETRIC_T_THRESHOLD).float()

    if mask_time.sum() > 0:
        trust_weights = torch.ones(x_t.shape[0], device=x_t.device)

        # 3. Trust Mechanism (If enabled)
        if compute_trust:
            # Note: We use torch.no_grad() for the 'trust' estimation part usually,
            # even during training, as we don't want to optimize the emulator here.
            with torch.no_grad():
                Y_phys = denormalizer.unnormalize_torch(Y)
                Y_phys = Y_phys * (Y_phys > 0.1).float()
                gamma_truth_phys = emulator(Y_phys)
                gamma_truth_log_pred = torch.log1p(gamma_truth_phys)

                diff_trust = gamma_truth_log_pred - Y_gamma
                emu_error_sq = diff_trust.pow(2).mean(dim=1)
                trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_sq)
                avg_trust_val = trust_weights.mean().item()

        # 4. Forward pass through Emulator for Gradient Flow
        pred_x0_phys = denormalizer.unnormalize_torch(pred_x0)
        pred_x0_phys = pred_x0_phys * (pred_x0_phys > 0.1).float()  # Drizzle threshold
        pred_gamma_phys = emulator(pred_x0_phys)
        pred_gamma_log = torch.log1p(pred_gamma_phys)

        raw_geom_loss = criterion(pred_gamma_log, Y_gamma)

        weight_factor = (trust_weights * mask_time).view(-1, 1)
        weighted_loss = raw_geom_loss * weight_factor
        loss_geom = weighted_loss.sum() / (mask_time.sum() + 1e-8)

    return loss_geom, avg_trust_val


def save_sample_images(model, diffusion, loader, device, out_dir, epoch, denormalizer):
    model.eval()
    try:
        X, Y, _ = next(iter(loader))
    except StopIteration:
        return

    X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
    n_samples = min(5, X.shape[0])
    X_sample = X[:n_samples]
    Y_sample = Y[:n_samples]

    x_generated = torch.zeros(
        (n_samples, 1, diffusion.img_size, diffusion.img_size), device=device
    )
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

    X_cpu = X_sample.float().cpu().numpy()
    Y_cpu = Y_sample.float().cpu().numpy()
    Gen_cpu = x_generated.float().cpu().numpy()

    X_phys = denormalizer.unnormalize(X_cpu[:, 0])
    Y_phys = denormalizer.unnormalize(Y_cpu[:, 0])
    Gen_phys = denormalizer.unnormalize(Gen_cpu[:, 0])

    precip_cmap = copy.copy(plt.get_cmap("Blues"))
    precip_cmap.set_bad(color="lightgrey", alpha=1.0)

    def mask_low_values(img, threshold=0.1):
        masked = img.copy()
        masked[masked <= threshold] = np.nan
        return masked

    _, axs = plt.subplots(n_samples, 3, figsize=(18, 5 * n_samples), squeeze=False)

    for i in range(n_samples):
        img_in = X_phys[i]
        img_target = Y_phys[i]
        img_gen = Gen_phys[i]

        local_max = np.nanmax(
            [np.nanmax(img_in), np.nanmax(img_target), np.nanmax(img_gen)]
        )
        vmax = max(local_max, 1.0)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        img_in_masked = mask_low_values(img_in)
        img_target_masked = mask_low_values(img_target)
        img_gen_masked = mask_low_values(img_gen)

        axs[i, 0].imshow(img_in_masked, cmap=precip_cmap, norm=norm, origin="lower")
        axs[i, 0].set_title(f"Input (LR) | Max: {np.nanmax(img_in):.2f}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(img_gen_masked, cmap=precip_cmap, norm=norm, origin="lower")
        axs[i, 1].set_title(f"Generated (SR) | Max: {np.nanmax(img_gen):.2f}")
        axs[i, 1].axis("off")

        im3 = axs[i, 2].imshow(
            img_target_masked, cmap=precip_cmap, norm=norm, origin="lower"
        )
        axs[i, 2].set_title(f"Target (HR) | Max: {np.nanmax(img_target):.2f}")
        axs[i, 2].axis("off")
        plt.colorbar(im3, ax=axs[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"),
        bbox_inches="tight",
        dpi=100,
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train DDPM with Curriculum Learning")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    denormalizer = DataDenormalizer(
        os.path.join(PREPROCESSED_DATA_DIR, "log_transformed_precip_max_val.npy")
    )

    train_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, METADATA_TRAIN, DEM_DATA_DIR, dem_stats, split="train"
    )
    val_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, METADATA_VAL, DEM_DATA_DIR, dem_stats, split="validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Initializing Geometric Loss...")
    S_inv_tensors = estimate_s_inv_from_dataset(
        train_dataset, num_samples=2000, device=device
    )
    geometric_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
        device
    )

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
        "val_geom": [],  # Added val_geom to history
        "avg_trust": [],
        "geom_weight": [],  # Added geom_weight to history
    }
    start_epoch = 0

    # State flags for Curriculum Learning
    geometric_phase = False
    geometric_start_epoch = None  # The epoch index when the phase switched

    # --- RESUME LOGIC ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"==> Resuming training from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"]

            geometric_phase = checkpoint.get("geometric_phase", False)
            geometric_start_epoch = checkpoint.get("geometric_start_epoch", None)

            if geometric_phase:
                print(
                    f"==> Resuming in Geometric Phase (Started at Epoch {geometric_start_epoch})."
                )

            if "early_stop_state" in checkpoint:
                early_stopper.load_state_dict(checkpoint["early_stop_state"])
            else:
                early_stopper.best_score = -checkpoint.get("best_val_loss", np.Inf)
                early_stopper.val_loss_min = checkpoint.get("best_val_loss", np.Inf)

            if "history" in checkpoint:
                history = checkpoint["history"]
            else:
                print("==> Warning: No history found. Starting fresh.")
        else:
            print(f"==> Error: Checkpoint not found at {args.resume}")
            return

    # --- Training Loop ---
    print(f"Starting Training from Epoch {start_epoch+1}...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        # --- CALCULATE CURRENT CURRICULUM WEIGHT ---
        current_geom_weight = 0.0
        if geometric_phase and geometric_start_epoch is not None:
            # Linear Warmup: Ramp from 0 to TARGET_WEIGHT over WARMUP_EPOCHS
            epochs_active = epoch - geometric_start_epoch
            # Clamp between 0 and 1
            progress = min(
                1.0, max(0.0, epochs_active / float(GEOMETRIC_WARMUP_EPOCHS))
            )
            current_geom_weight = GEOMETRIC_TARGET_WEIGHT * progress

        phase_desc = (
            f"GEOMETRIC (w={current_geom_weight:.4f})"
            if geometric_phase
            else "MSE-ONLY"
        )
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [{phase_desc}]")

        running_loss = 0.0
        running_geom = 0.0
        running_trust = 0.0

        for X, Y, Y_gamma in pbar:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            Y_gamma = Y_gamma.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                predicted_noise = model(x_t, t, X)
                loss_mse = mse(noise, predicted_noise)

                # Compute Geometric Component (always 0.0 if weight is 0.0, but we skip computation if weight is 0)
                loss_geom = torch.tensor(0.0, device=device)
                avg_trust_val = 1.0

                if current_geom_weight > 0.0:
                    loss_geom, avg_trust_val = compute_geometric_loss_component(
                        diffusion,
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        x_t,
                        predicted_noise,
                        t,
                        Y,
                        Y_gamma,
                        compute_trust=True,
                    )

                # Apply Dynamic Weight
                total_loss = loss_mse + (current_geom_weight * loss_geom)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_mse.item()
            running_geom += loss_geom.item()
            running_trust += avg_trust_val
            pbar.set_postfix(
                MSE=f"{loss_mse.item():.4f}",
                Geom=f"{loss_geom.item():.4f}",
                W=f"{current_geom_weight:.3f}",
            )

        avg_loss = running_loss / len(train_loader)
        avg_geom = running_geom / len(train_loader)

        # --- VALIDATION LOOP (Modified to include Geometric Loss) ---
        model.eval()
        val_mse_loss = 0.0
        val_geom_loss = 0.0

        with torch.no_grad():
            # Note: We now unpack Y_gamma in validation too!
            for X, Y, Y_gamma in val_loader:
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                Y_gamma = Y_gamma.to(device, non_blocking=True)

                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)

                with torch.amp.autocast("cuda"):
                    predicted_noise = model(x_t, t, X)
                    loss_m = mse(noise, predicted_noise)

                    # ALWAYS compute geometric loss in validation for monitoring
                    loss_g, _ = compute_geometric_loss_component(
                        diffusion,
                        emulator,
                        geometric_criterion,
                        denormalizer,
                        x_t,
                        predicted_noise,
                        t,
                        Y,
                        Y_gamma,
                        compute_trust=False,
                    )

                val_mse_loss += loss_m.item()
                val_geom_loss += loss_g.item()

        avg_val_mse = val_mse_loss / len(val_loader)
        avg_val_geom = val_geom_loss / len(val_loader)
        print(
            f"Epoch {epoch+1} Val MSE: {avg_val_mse:.6f} | Val Geom: {avg_val_geom:.6f}"
        )

        # Update History
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_mse)
        history["train_geom"].append(avg_geom)
        history["val_geom"].append(avg_val_geom)  # Log validation geometric loss
        history["avg_trust_val"].append(avg_trust_val)
        history["geom_weight"].append(current_geom_weight)  # Log the curriculum weight

        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, "loss_history.csv"), index=False
        )

        if (epoch + 1) % 5 == 0 or (epoch == 0):
            save_sample_images(
                model, diffusion, val_loader, device, out_dir, epoch + 1, denormalizer
            )

        # --- SAVE CHECKPOINT ---
        checkpoint_latest = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "early_stop_state": early_stopper.state_dict(),
            "history": history,
            "best_val_loss": early_stopper.val_loss_min,
            "geometric_phase": geometric_phase,
            "geometric_start_epoch": geometric_start_epoch,  # Save curriculum state
        }
        torch.save(checkpoint_latest, os.path.join(out_dir, "ddpm_latest.pth"))

        # Save best based on MSE (primary metric)
        if epoch > 0 and avg_val_mse < min(history["val_loss"][:-1]):
            torch.save(checkpoint_latest, os.path.join(out_dir, "ddpm_best.pth"))

        # --- Physical Validation & Trigger Logic ---
        if (epoch + 1) % 5 == 0:
            print("Running Physical Validation...")
            model.eval()
            val_metrics = {"wd": [], "max_err": []}
            NUM_PHYS_BATCHES = 10

            with torch.no_grad():
                for i, (X_val, Y_val, _) in enumerate(val_loader):
                    if i >= NUM_PHYS_BATCHES:
                        break
                    X_val, Y_val = X_val.to(device), Y_val.to(device)
                    # ... [Same physical validation logic as before] ...
                    input_precip = X_val[:, 0, :, :]
                    is_wet = input_precip.amax(dim=(1, 2)) > 1e-6
                    wet_indices = torch.where(is_wet)[0]
                    if len(wet_indices) == 0:
                        continue
                    X_wet, Y_wet = X_val[wet_indices], Y_val[wet_indices]
                    gen_wet = diffusion.sample(
                        model, n=len(wet_indices), conditions=X_wet
                    )
                    Y_phys = denormalizer.unnormalize(Y_wet.cpu().numpy().squeeze())
                    Gen_phys = denormalizer.unnormalize(gen_wet.cpu().numpy().squeeze())
                    batch_metrics = compute_physical_metrics(Y_phys, Gen_phys)
                    val_metrics["wd"].append(batch_metrics["wasserstein_dist"])
                    val_metrics["max_err"].append(batch_metrics["max_intensity_err"])

            if len(val_metrics["wd"]) > 0:
                mean_wd = np.mean(val_metrics["wd"])
                print(f"  > Wasserstein Dist: {mean_wd:.4f}")

                # Check Early Stopping
                should_trigger = early_stopper(mean_wd)

                if should_trigger:
                    if not geometric_phase:
                        print(
                            f"!!! Convergence Reached (WD={mean_wd:.4f}). triggering Geometric Curriculum !!!"
                        )
                        print(
                            f"!!! Ramping up Geometric Loss over next {GEOMETRIC_WARMUP_EPOCHS} epochs !!!"
                        )
                        geometric_phase = True
                        geometric_start_epoch = epoch + 1  # Start curriculum next epoch
                        early_stopper.reset()
                    else:
                        print(
                            "!!! Convergence Reached in Geometric Phase. Stopping training."
                        )
                        break


if __name__ == "__main__":
    main()
