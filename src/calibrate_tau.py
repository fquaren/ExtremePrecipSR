import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
from tqdm import tqdm
import os
import sys

# Import your modules
from dataset import SRDataset
from utils import load_emulator


def calibrate_tau_log_space():
    """
    Calibrates tau using PHYSICAL inputs and LOG outputs.
    """

    # --- Config ---
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_path, "config.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Scaler ---
    scaler_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_transformed_precip_max_val.npy"
    )
    if not os.path.exists(scaler_path):
        print(f"CRITICAL ERROR: Scaler not found at {scaler_path}")
        return

    PHYSICAL_MAX_VAL = float(np.load(scaler_path))
    print(f"Loaded Physical Max (Log-Space) Scaling Factor: {PHYSICAL_MAX_VAL:.4f}")

    # Load Stats & Data
    with open(config["DEM_STATS"], "r") as f:
        stats = json.load(f)
    dem_stats = (float(stats["dem_mean"]), float(stats["dem_std"]))

    print("Initializing Dataset...")
    val_dataset = SRDataset(
        config["PREPROCESSED_DATA_DIR"],
        config["VAL_METADATA_FILE"],
        config["DEM_DATA_DIR"],
        dem_stats,
        split="validation",
    )

    # Use full dataset for accurate p99 statistics
    batch_size = config.get("BATCH_SIZE", 32)
    loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, shuffle=False
    )
    print(f"Data loaded. N_samples: {len(val_dataset)}")

    # Load Emulator
    print("Loading Emulator...")
    emu_path = config.get("EMULATOR_CHECKPOINT_PATH")
    emulator = load_emulator(emu_path, config, device)
    emulator.eval()

    errors_mean_spatial = []
    errors_max_spatial = []

    print("Computing Emulator Errors (Corrected I/O)...")

    with torch.no_grad():
        for _, Y, Y_gamma_log in tqdm(loader):
            Y = Y.to(device)
            Y_gamma_log = Y_gamma_log.to(device)

            # 1. Un-normalize Input
            Y_log_mag = Y * PHYSICAL_MAX_VAL
            Y_phys = torch.expm1(Y_log_mag)

            # 2. Emulator Prediction (Returns PHYSICAL Gamma)
            gamma_pred_phys = emulator(Y_phys)

            # --- CRITICAL FIX: Log-Transform Output ---
            # The dataset targets (Y_gamma_log) are already log1p transformed.
            # We must transform the emulator output to match.
            gamma_pred_log = torch.log1p(gamma_pred_phys)

            # 3. Compute MSE in Log Space
            mse_tensor = F.mse_loss(gamma_pred_log, Y_gamma_log, reduction="none")

            # Metrics
            mse_mean = mse_tensor.view(mse_tensor.size(0), -1).mean(dim=1)
            mse_max, _ = mse_tensor.view(mse_tensor.size(0), -1).max(dim=1)

            errors_mean_spatial.extend(mse_mean.cpu().numpy())
            errors_max_spatial.extend(mse_max.cpu().numpy())

    # Convert to numpy
    errors_mean_spatial = np.array(errors_mean_spatial)
    errors_max_spatial = np.array(errors_max_spatial)

    # --- STATISTICS ---
    print("\n" + "=" * 40)
    print("       ERROR STATISTICS (Final)      ")
    print("=" * 40)

    stats_mean = {
        "p50": np.percentile(errors_mean_spatial, 50),
        "p90": np.percentile(errors_mean_spatial, 90),
    }

    stats_max = {
        "p50": np.percentile(errors_max_spatial, 50),
        "p90": np.percentile(errors_max_spatial, 90),
    }

    print(f"{'Metric':<10} | {'Median (p50)':<12} | {'High (p90)':<12}")
    print("-" * 40)
    print(f"{'Mean MSE':<10} | {stats_mean['p50']:<12.5f} | {stats_mean['p90']:<12.5f}")
    print(f"{'Max MSE':<10} | {stats_max['p50']:<12.5f} | {stats_max['p90']:<12.5f}")

    # --- RECALCULATE TAU ---
    tau_std_aggressive = -np.log(0.1) / (stats_mean["p90"] + 1e-8)
    tau_stability = -np.log(0.5) / (stats_max["p90"] + 1e-8)

    print("\n" + "=" * 40)
    print("         RECOMMENDED TAU VALUES          ")
    print("=" * 40)

    print(f"Aggressive (p90=0.1):        {tau_std_aggressive:.6f}")
    print(f"Stability (Max_p90=0.5):     {tau_stability:.6f}")
    print("-" * 40)


if __name__ == "__main__":
    calibrate_tau_log_space()
