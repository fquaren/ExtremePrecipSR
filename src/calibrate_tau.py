import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
import argparse
from tqdm import tqdm

# Import your modules
from dataset import SRDataset
from utils import load_emulator


def calibrate_tau_log_space():
    # Load Config
    config_path = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Stats & Data
    with open(config["DEM_STATS"], "r") as f:
        stats = json.load(f)
    dem_stats = (float(stats["dem_mean"]), float(stats["dem_std"]))

    # Note: If you updated SRDataset to return Log values, Y_gamma will be log.
    # If you haven't updated it yet, Y_gamma is physical.
    # This script handles BOTH cases safely.
    val_dataset = SRDataset(
        config["PREPROCESSED_DATA_DIR"],
        config["VAL_METADATA_FILE"],
        config["DEM_DATA_DIR"],
        dem_stats,
        split="validation",
    )

    # Subset for speed
    val_dataset = torch.utils.data.Subset(
        val_dataset, range(min(len(val_dataset), 1000))
    )
    loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    # Load Emulator
    print("Loading Emulator...")
    emu_path = config.get("EMULATOR_CHECKPOINT_PATH")
    emulator = load_emulator(emu_path, config, device)
    emulator.eval()

    errors = []

    print("Computing Emulator Errors (Log Space) on Ground Truth...")
    with torch.no_grad():
        for _, Y, Y_gamma_log in tqdm(loader):
            Y = Y.to(device)
            Y_gamma_log = Y_gamma_log.to(device)

            # 1. Emulator Prediction (Always Log Space)
            gamma_pred_log = emulator(Y)

            # 3. Compute MSE per sample in Log Space
            mse = F.mse_loss(gamma_pred_log, Y_gamma_log, reduction="none")

            # Collapse (Batch, Features, Quantiles) -> (Batch)
            mse_per_sample = mse.view(mse.size(0), -1).mean(dim=1)

            errors.extend(mse_per_sample.cpu().numpy())

    errors = np.array(errors)
    p50 = np.percentile(errors, 50)  # Median
    p90 = np.percentile(errors, 90)  # Outlier boundary

    print("\n--- CORRECTED LOG-SPACE STATISTICS ---")
    print(f"Median Error (p50): {p50:.5f}")
    print(f"High Error   (p90): {p90:.5f}")

    # SUGGESTION LOGIC
    # Target Trust at Median = 0.8
    tau_conservative = -np.log(0.8) / p50

    # Target Trust at p90 = 0.1
    tau_aggressive = -np.log(0.1) / p90

    print("\n--- RECOMMENDATIONS ---")
    print(f"Conservative Tau (Trust median @ 0.8): {tau_conservative:.4f}")
    print(f"Aggressive Tau   (Reject p90 @ 0.1):   {tau_aggressive:.4f}")

    avg_tau = (tau_conservative + tau_aggressive) / 2
    print(f"Balanced Tau     (Average):            {avg_tau:.4f}")
    print(f"\n-> Update config.yaml with TRUST_TAU: {avg_tau:.4f}")


if __name__ == "__main__":
    calibrate_tau_log_space()
