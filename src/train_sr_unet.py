import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np
import warnings
import time
import argparse
import pandas as pd
import sys

# Project imports
from model import UNetSR
from dataset import SRDataset
from loss import (
    estimate_s_inv_from_dataset,
    GeometricLossSeparate,
)
from utils import load_emulator

# Suppress minor warnings for cleaner logs
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Config Load ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Constants & Config Extraction ---
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
METADATA_VAL_METADATA_FILE = config["VAL_METADATA_FILE"]

BATCH_SIZE = config.get("SR_BATCH_SIZE", 128)
LEARNING_RATE = config.get("SR_LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("SR_WEIGHT_DECAY", 1e-5)
NUM_EPOCHS = config.get("SR_NUM_EPOCHS", 50)
NUM_WORKERS = config.get("NUM_WORKERS", 32)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "SR_UNet_Baseline")

# Metric Loss Config
METRIC_LOSS_MODE = config.get("METRIC_LOSS_MODE", "none")
METRIC_LOSS_WEIGHT = config.get("METRIC_LOSS_WEIGHT", 0.1)
EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)
S_ESTIMATION_SAMPLES = config.get("S_ESTIMATION_SAMPLES", 2000)
METRIC_WARMUP_EPOCHS = config.get("METRIC_WARMUP_EPOCHS", 5)
METRIC_RAMP_EPOCHS = config.get("METRIC_RAMP_EPOCHS", 15)
TRUST_TAU = config.get("TRUST_TAU", 0.5)  # Updated based on your calibration

# Physical Constants
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)


def main():
    parser = argparse.ArgumentParser(description="Train a Super-Resolution UNet model.")
    parser.add_argument(
        "--metric_loss_mode",
        type=str,
        default=None,
        choices=["none", "train", "validate"],
        help="Override config METRIC_LOSS_MODE via CLI",
    )
    args = parser.parse_args()

    global METRIC_LOSS_MODE
    if args.metric_loss_mode is not None:
        METRIC_LOSS_MODE = args.metric_loss_mode
        print(f"--- CLI OVERRIDE: Using METRIC_LOSS_MODE='{METRIC_LOSS_MODE}' ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Physical Scalar ---
    # We load the scaler that represents the Max Value in Log Space (~5.017)
    scaler_path = os.path.join(
        PREPROCESSED_DATA_DIR, "log_transformed_precip_max_val.npy"
    )

    if not os.path.exists(scaler_path):
        # Check for the alternate name if the first doesn't exist
        scaler_path_alt = os.path.join(
            PREPROCESSED_DATA_DIR, "train", "precip_max_val.npy"
        )
        if os.path.exists(scaler_path_alt):
            scaler_path = scaler_path_alt

    if not os.path.exists(scaler_path):
        print(
            f"WARNING: Scaler not found at {scaler_path}. Defaulting to 1.0",
            file=sys.stderr,
        )
        PHYSICAL_MAX_VAL = 1.0
    else:
        PHYSICAL_MAX_VAL = float(np.load(scaler_path))
        print(f"Loaded Physical Max (Log-Space) Scaling Factor: {PHYSICAL_MAX_VAL:.4f}")

    # --- 2. Setup Experiment Output ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{METRIC_LOSS_MODE}_{timestamp}"
    output_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    # --- 3. Initialize Datasets ---
    print("\n--- Initializing Datasets ---")

    # TRAIN DATASET: By definition in your SRDataset class, this filters out dry patches!
    train_dataset = SRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN_METADATA_FILE,
        DEM_DATA_DIR,
        dem_stats,
        split="train",
    )

    # VALIDATION DATASET: Keeps dry patches to test realistic performance
    val_dataset = SRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL_METADATA_FILE,
        DEM_DATA_DIR,
        dem_stats,
        split="validation",
    )

    print(f"Training Samples (Wet Only): {len(train_dataset)}")
    print(f"Validation Samples (Full):   {len(val_dataset)}")

    # --- 4. Initialize Weighted Sampler ---
    sampler = None
    try:
        print("\nInitializing Balanced Sampler...")
        meta_df = pd.read_csv(METADATA_TRAIN_METADATA_FILE, sep=r"\s+")

        # CRITICAL: Align metadata indices with the dataset's filtered indices
        # This handles the "Wet Only" constraint so the sampler doesn't crash
        if hasattr(train_dataset, "valid_indices"):
            print(
                f"Aligning sampler metadata with {len(train_dataset)} filtered patches..."
            )
            meta_df = meta_df.iloc[train_dataset.valid_indices].reset_index(drop=True)

        target_col = next((c for c in meta_df.columns if "max" in c.lower()), None)

        if target_col:
            max_vals = meta_df[target_col].values

            # Create classes for balancing
            t1 = np.quantile(max_vals, 1.0 / 3.0)
            t2 = np.quantile(max_vals, 2.0 / 3.0)
            labels = np.zeros_like(max_vals, dtype=int)
            labels[max_vals <= t1] = 0
            labels[(max_vals > t1) & (max_vals <= t2)] = 1
            labels[max_vals > t2] = 2

            counts = np.bincount(labels)
            print(f"Class Balance (Low/Med/High): {counts}")

            weights = 1.0 / np.maximum(counts, 1)
            weights_tensor = torch.from_numpy(weights[labels]).float()

            assert len(weights_tensor) == len(train_dataset)

            sampler = WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(weights_tensor),
                replacement=True,
            )
            print("Sampler initialized successfully.")
        else:
            print("Warning: Could not find 'max_precip' column. Using random shuffle.")

    except Exception as e:
        print(f"Sampler init failed: {e}")
        print("Falling back to standard shuffling.")

    # --- 5. DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
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
    )

    # --- 6. Models & Optimizer ---
    sr_model = UNetSR(in_channels=2, out_channels=1).to(device)

    optimizer = torch.optim.Adam(
        sr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    pixel_criterion = nn.L1Loss()

    # --- 7. Metric/Emulator Setup ---
    emulator_model = None
    metric_criterion = None

    if METRIC_LOSS_MODE != "none":
        print(f"\nSetting up Metric Loss (Mode: {METRIC_LOSS_MODE})...")
        S_inv_tensors = estimate_s_inv_from_dataset(
            train_dataset, S_ESTIMATION_SAMPLES, device
        )
        metric_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
            device
        )

        if METRIC_LOSS_MODE == "train":
            if EMULATOR_CHECKPOINT_PATH is None:
                raise ValueError(
                    "METRIC_LOSS_MODE='train' requires EMULATOR_CHECKPOINT_PATH."
                )

            print(f"Loading Emulator from {EMULATOR_CHECKPOINT_PATH}...")
            emulator_model = load_emulator(EMULATOR_CHECKPOINT_PATH, config, device)
            emulator_model.eval()
            for param in emulator_model.parameters():
                param.requires_grad = False

    # --- 8. Logging Init ---
    log_file_path = os.path.join(output_dir, "sr_training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss_total,train_loss_mae,train_loss_metric,"
            "val_loss_total,val_loss_mae,val_loss_metric,"
            "val_consistency_gap,val_intrinsic_error,"
            "epoch_duration_sec,current_metric_weight,avg_trust_weight\n"
        )

    # --- 9. Training Loop ---
    print(f"\nStarting SR training for {NUM_EPOCHS} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    metric_loss_scaler = 1.0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        current_metric_weight = 0.0
        if METRIC_LOSS_MODE == "train":
            if epoch >= METRIC_WARMUP_EPOCHS:
                progress = (epoch - METRIC_WARMUP_EPOCHS + 1) / float(
                    METRIC_RAMP_EPOCHS
                )
                current_metric_weight = min(1.0, progress) * METRIC_LOSS_WEIGHT

        sr_model.train()
        logs = {"loss": 0.0, "mae": 0.0, "metric": 0.0, "trust": 0.0}

        pbar = tqdm(
            train_loader, desc=f"Ep {epoch+1} | W_metric={current_metric_weight:.4f}"
        )

        for X, Y, Y_gamma_log in pbar:
            X, Y, Y_gamma_log = X.to(device), Y.to(device), Y_gamma_log.to(device)
            optimizer.zero_grad()

            pred_X = sr_model(X)
            loss_mae = pixel_criterion(pred_X, Y)

            metric_term = torch.tensor(0.0, device=device)
            raw_metric_mean = torch.tensor(0.0, device=device)
            avg_trust = 1.0

            if METRIC_LOSS_MODE == "train" and current_metric_weight > 0:
                # --- INVERT LOG TRANSFORM ---

                # A. Enforce positivity
                pred_X_pos = F.softplus(pred_X, beta=5.0)

                # B. Scale to Log-Space Magnitude (0 to ~5.0)
                pred_X_log_space = pred_X_pos * PHYSICAL_MAX_VAL

                # C. Convert to Physical Units (0 to 150+ mm/h)
                pred_X_phys = torch.expm1(pred_X_log_space)

                # --- Sparsity ---
                # Remove background noise so the emulator sees clean shapes
                pred_X_phys = pred_X_phys * (pred_X_phys > DRIZZLE_THRESHOLD).float()

                # D. Emulator Prediction
                pred_gamma_phys = emulator_model(pred_X_phys)  # Output: km^2, km

                # --- Manifold Alignment ---
                # Transform Physical Output -> Log Space to match Dataset Targets
                pred_gamma_log = torch.log1p(pred_gamma_phys)

                # Trust Gating logic
                with torch.no_grad():
                    # Apply same transform to Ground Truth Y
                    Y_phys = torch.expm1(Y * PHYSICAL_MAX_VAL)

                    # Clean Ground Truth (should already be clean, but for consistency)
                    Y_phys = Y_phys * (Y_phys > DRIZZLE_THRESHOLD).float()

                    gamma_truth_phys = emulator_model(Y_phys)

                    # Transform Emulator prediction on GT to Log Space
                    gamma_truth_log_pred = torch.log1p(gamma_truth_phys)

                    # Now we compare Log vs Log. This is valid.
                    emu_error_matrix = F.mse_loss(
                        gamma_truth_log_pred, Y_gamma_log, reduction="none"
                    )
                    emu_error_scalar = emu_error_matrix.view(
                        emu_error_matrix.size(0), -1
                    ).mean(dim=1)

                    trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_scalar)
                    avg_trust = trust_weights.mean().item()

                loss_vec = metric_criterion(pred_gamma_log, Y_gamma_log)
                raw_metric_mean = loss_vec.mean()
                metric_term = (loss_vec * trust_weights).mean()

                total_loss = loss_mae + (
                    current_metric_weight * metric_loss_scaler * metric_term
                )
            else:
                total_loss = loss_mae

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sr_model.parameters(), max_norm=1.0)
            optimizer.step()

            logs["loss"] += total_loss.item()
            logs["mae"] += loss_mae.item()
            logs["metric"] += raw_metric_mean.item()
            logs["trust"] += avg_trust

            pbar.set_postfix(
                L=f"{total_loss.item():.3f}",
                MAE=f"{loss_mae.item():.3f}",
                Tr=f"{avg_trust:.2f}",
            )

        n_train = len(train_loader)
        avg_train = {k: v / n_train for k, v in logs.items()}
        print(
            f"Train Ep {epoch+1}: Loss={avg_train['loss']:.4f}, MAE={avg_train['mae']:.4f}, Trust={avg_train['trust']:.2f}"
        )

        # --- 10. Validation Loop ---
        sr_model.eval()
        val_logs = {"loss": 0.0, "mae": 0.0, "metric": 0.0, "gap": 0.0, "intr": 0.0}

        with torch.no_grad():
            for X, Y, Y_gamma_log in tqdm(val_loader, desc="Validation"):
                X, Y, Y_gamma_log = X.to(device), Y.to(device), Y_gamma_log.to(device)

                pred_X = sr_model(X)
                loss_mae = pixel_criterion(pred_X, Y)

                loss_metric_val = torch.tensor(0.0, device=device)
                gap = torch.tensor(0.0, device=device)
                intr = torch.tensor(0.0, device=device)

                if METRIC_LOSS_MODE != "none":
                    # --- VALIDATION TRANSFORM ---

                    # 1. Invert Log-Normal Transformation to get Physical Precip
                    pred_X_pos = F.softplus(pred_X, beta=5.0)
                    pred_X_phys = torch.expm1(pred_X_pos * PHYSICAL_MAX_VAL)
                    Y_phys = torch.expm1(Y * PHYSICAL_MAX_VAL)

                    # 2. Apply Sparsity / Drizzle Threshold
                    # Crucial: The emulator expects sparse storms, not continuous background noise.
                    pred_X_phys = (
                        pred_X_phys * (pred_X_phys > DRIZZLE_THRESHOLD).float()
                    )
                    # Apply to Y as well for consistency in "intrinsic" calculation
                    Y_phys = Y_phys * (Y_phys > DRIZZLE_THRESHOLD).float()

                    if METRIC_LOSS_MODE == "train":
                        # 3. Emulator Forward Pass (Outputs Physical Units: km^2, km)
                        pred_gamma_phys = emulator_model(pred_X_phys)
                        gamma_truth_phys = emulator_model(Y_phys)

                        # 4. Manifold Alignment: Physical -> Log Space
                        pred_gamma_log = torch.log1p(pred_gamma_phys)
                        gamma_truth_log_pred = torch.log1p(gamma_truth_phys)

                        # 5. Compute Metrics in Log Space

                        # A. Validation Metric Loss (Pred vs Ground Truth Dataset)
                        loss_vec = metric_criterion(pred_gamma_log, Y_gamma_log)
                        loss_metric_val = loss_vec.mean()

                        # B. Intrinsic Emulator Error (Emulator(GT) vs Ground Truth Dataset)
                        intr_vec = metric_criterion(gamma_truth_log_pred, Y_gamma_log)
                        intr = intr_vec.mean()

                        # C. Consistency Gap (Pred vs Emulator(GT))
                        # Represents the distance between "Predicted Topology" and "Topology of GT"
                        # as perceived by the emulator itself.
                        loss_perc_vec = metric_criterion(
                            pred_gamma_log, gamma_truth_log_pred
                        )
                        gap = loss_perc_vec.mean()  # Standardized definition of gap

                val_total_loss = loss_mae
                val_logs["loss"] += val_total_loss.item()
                val_logs["mae"] += loss_mae.item()
                val_logs["metric"] += loss_metric_val.item()
                val_logs["gap"] += gap.item()
                val_logs["intr"] += intr.item()

        n_val = len(val_loader)
        avg_val = {k: v / n_val for k, v in val_logs.items()}
        print(
            f"Val Ep {epoch+1}: MAE={avg_val['mae']:.4f}, Metric={avg_val['metric']:.4f}, Gap={avg_val['gap']:.4f}"
        )

        # --- 11. Auto-Calibration ---
        if METRIC_LOSS_MODE == "train" and epoch == METRIC_WARMUP_EPOCHS - 1:
            if avg_val["metric"] > 1e-6:
                metric_loss_scaler = avg_val["mae"] / avg_val["metric"]
                print(
                    f"WARMUP COMPLETE. Auto-Calibrated Scaler: {metric_loss_scaler:.4f}"
                )
                best_val_loss = float("inf")
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )

            # Save this model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": sr_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "scaler": metric_loss_scaler,
                },
                os.path.join(output_dir, "warmup_complete_model.pth"),
            )

        # --- 12. Checkpointing ---
        current_val_score = avg_val["mae"]
        scheduler.step(current_val_score)

        if current_val_score < best_val_loss:
            best_val_loss = current_val_score
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": sr_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "scaler": metric_loss_scaler,
                },
                os.path.join(output_dir, "best_sr_model.pth"),
            )
            print(">>> New Best Model Saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print("Early stopping triggered.")
                break

        with open(log_file_path, "a") as log:
            log.write(
                f"{epoch+1},{avg_train['loss']:.6f},{avg_train['mae']:.6f},{avg_train['metric']:.6f},"
                f"{avg_val['loss']:.6f},{avg_val['mae']:.6f},{avg_val['metric']:.6f},"
                f"{avg_val['gap']:.6f},{avg_val['intr']:.6f},"
                f"{time.time()-start_time:.2f},{current_metric_weight:.4f},{avg_train['trust']:.4f}\n"
            )

    print("SR Training Complete.")


if __name__ == "__main__":
    main()
