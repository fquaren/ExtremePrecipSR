import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml
import os
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt  # Added for visual logging

from model_ddpm import ContextUnet
from diffusion import Diffusion
from dataset import SRDataset

# --- Config ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
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
NUM_WORKERS = 24
EXPERIMENT_NAME = "DDPM_SR"


def save_sample_images(model, diffusion, loader, device, out_dir, epoch):
    """
    Scientifically validates the model by running the Reverse Diffusion Process (Sampling).
    This is distinct from the Forward Process (Noise Prediction) used in training.
    """
    model.eval()

    # Grab a single batch
    try:
        X, Y, _ = next(iter(loader))
    except StopIteration:
        return

    X, Y = X.to(device), Y.to(device)

    # We only sample 4 images to save time
    n_samples = 4
    X_sample = X[:n_samples]
    Y_sample = Y[:n_samples]

    with torch.no_grad():
        # Sample using the reverse process
        # Input X_sample is the Condition (LowRes + DEM)
        # Output x_sampled is the Generated HighRes
        x_sampled = diffusion.sample(model, n=n_samples, conditions=X_sample)

    # Plotting
    fig, axs = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axs = axs[None, :]

    X_cpu = X_sample.cpu().numpy()
    Y_cpu = Y_sample.cpu().numpy()
    Gen_cpu = x_sampled.cpu().numpy()

    for i in range(n_samples):
        # Column 1: Condition (Precipitation Channel only, index 0)
        axs[i, 0].imshow(X_cpu[i, 0], cmap="Blues", origin="lower")
        axs[i, 0].set_title("Condition (LR Precip)")
        axs[i, 0].axis("off")

        # Column 2: Ground Truth
        axs[i, 1].imshow(Y_cpu[i, 0], cmap="Blues", origin="lower")
        axs[i, 1].set_title("Ground Truth (HR)")
        axs[i, 1].axis("off")

        # Column 3: Generated (DDPM)
        axs[i, 2].imshow(Gen_cpu[i, 0], cmap="Blues", origin="lower")
        axs[i, 2].set_title("Generated (DDPM)")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:03d}.png"))
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Setup Output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    out_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save a copy of config for reproducibility
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    # --- Data ---
    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    train_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, METADATA_TRAIN, DEM_DATA_DIR, dem_stats, split="train"
    )
    val_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, METADATA_VAL, DEM_DATA_DIR, dem_stats, split="validation"
    )

    # Sampler
    sampler = None
    try:
        meta_df = pd.read_csv(METADATA_TRAIN, sep=" ")
        target_col = next((c for c in meta_df.columns if "max" in c.lower()), None)
        if target_col:
            max_vals = meta_df[target_col].values
            labels = np.zeros_like(max_vals, dtype=int)
            labels[max_vals > 0.1] = 1
            labels[max_vals > 12.0] = 2
            counts = np.bincount(labels)
            weights = 1.0 / np.maximum(counts, 1)
            sample_weights = torch.from_numpy(weights[labels]).float()
            sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )
            print("Sampler Active.")
    except Exception as e:
        print(f"Sampler init failed, defaulting to random shuffle: {e}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # --- Models ---
    # CORRECTED DIMENSIONS:
    # in_channels=1 (The noisy HR target image)
    # c_in_condition=2 (The stacked LR Precip + DEM from SRDataset)
    model = ContextUnet(in_channels=1, c_in_condition=2, device=device).to(device)
    diffusion = Diffusion(img_size=128, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mae = nn.L1Loss()

    # Logging Structures
    history = {"epoch": [], "train_loss": [], "val_loss": [], "timestamp": []}

    print("Starting DDPM Training...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        for X, Y, _ in pbar:
            # X: Condition [B, 2, H, W] (Precip + DEM)
            # Y: Target [B, 1, H, W] (High Res Precip)
            X, Y = X.to(device), Y.to(device)

            t = diffusion.sample_timesteps(X.shape[0])
            x_t, noise = diffusion.noise_images(Y, t)  # Diffuse the Target Y

            # Predict noise given x_t and condition X
            # Note: Model forward must accept X with 2 channels
            predicted_noise = model(x_t, t, X)

            loss = mae(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(MAE=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train MAE: {avg_loss:.6f}")

        # --- Validation (One Step noise prediction) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y, _ in val_loader:
                X, Y = X.to(device), Y.to(device)
                t = diffusion.sample_timesteps(X.shape[0])
                x_t, noise = diffusion.noise_images(Y, t)
                predicted_noise = model(x_t, t, X)
                loss = mae(noise, predicted_noise)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val MAE: {avg_val_loss:.6f}")

        # --- Update Logs ---
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
        history["timestamp"].append(time.strftime("%H:%M:%S"))

        # Save logs to CSV (Overwrites every epoch for safety)
        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, "loss_history.csv"), index=False
        )

        # --- Visual Sampling ---
        # We sample every 5 epochs (or first epoch) to check convergence
        if (epoch + 1) % 5 == 0 or (epoch == 0):
            save_sample_images(model, diffusion, val_loader, device, out_dir, epoch + 1)

        # Checkpoint
        torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_latest.pth"))

        # Optional: Save best model
        if epoch > 0 and avg_val_loss < min(history["val_loss"][:-1]):
            torch.save(model.state_dict(), os.path.join(out_dir, "ddpm_best.pth"))


if __name__ == "__main__":
    main()
