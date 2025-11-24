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

from model import UNetSR
from dataset import SRDataset

from loss import (
    estimate_s_inv_from_dataset,
    GeometricLossSeparate,
    compute_gamma_matrix_for_image,
)
from utils import load_emulator

warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Config Extraction ---
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
METRIC_WARMUP_EPOCHS = config.get("METRIC_WARMUP_EPOCHS", 10)
METRIC_RAMP_EPOCHS = config.get("METRIC_RAMP_EPOCHS", 20)
TRUST_TAU = config.get("TRUST_TAU", 2.0)  # Note: Check this value for Log Space errors

# Emulator Config
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)


def main():
    parser = argparse.ArgumentParser(description="Train a Super-Resolution UNet model.")
    parser.add_argument(
        "--metric_loss_mode",
        type=str,
        default=None,
        choices=["none", "train", "validate"],
    )
    args = parser.parse_args()

    global METRIC_LOSS_MODE
    if args.metric_loss_mode is not None:
        METRIC_LOSS_MODE = args.metric_loss_mode
        print(f"Overriding config: Using METRIC_LOSS_MODE='{METRIC_LOSS_MODE}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{METRIC_LOSS_MODE}_{timestamp}"
    output_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    # --- 1. Prepare Datasets ---
    # Dataset now returns LOG-TRANSFORMED gamma targets
    train_dataset = SRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN_METADATA_FILE,
        DEM_DATA_DIR,
        dem_stats,
        split="train",
    )
    val_dataset = SRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL_METADATA_FILE,
        DEM_DATA_DIR,
        dem_stats,
        split="validation",
    )

    # --- 1.5 Sampler Setup ---
    sampler = None
    try:
        meta_df = pd.read_csv(METADATA_TRAIN_METADATA_FILE, sep=r"\s+")
        target_col = next((c for c in meta_df.columns if "max" in c.lower()), None)

        if target_col:
            max_vals = meta_df[target_col].values
            wet_mask = max_vals > DRIZZLE_THRESHOLD
            wet_values = max_vals[wet_mask]

            if len(wet_values) > 0:
                t1 = np.quantile(wet_values, 1.0 / 3.0)
                t2 = np.quantile(wet_values, 2.0 / 3.0)
                labels = np.zeros_like(max_vals, dtype=int)
                labels[(max_vals > DRIZZLE_THRESHOLD) & (max_vals <= t1)] = 1
                labels[(max_vals > t1) & (max_vals <= t2)] = 2
                labels[max_vals > t2] = 3

                counts = np.bincount(labels)
                print(f"Class Balance: {counts}")
                weights = 1.0 / np.maximum(counts, 1)
                weights_tensor = torch.from_numpy(weights[labels]).float()
                sampler = WeightedRandomSampler(
                    weights=weights_tensor,
                    num_samples=len(weights_tensor),
                    replacement=True,
                )
                print("SR Sampler initialized.")
    except Exception as e:
        print(f"Sampler init failed: {e}")

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

    # --- 2. Initialize Models ---
    sr_model = UNetSR(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(
        sr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    pixel_criterion = nn.L1Loss()

    # --- Metric-based Loss Setup ---
    emulator_model = None
    metric_criterion = None

    if METRIC_LOSS_MODE != "none":
        print("Initializing GeometricLossSeparate...")
        # S_inv is estimated on LOG-TRANSFORMED data from the dataset
        S_inv_tensors = estimate_s_inv_from_dataset(
            train_dataset, S_ESTIMATION_SAMPLES, device
        )
        metric_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
            device
        )

        if METRIC_LOSS_MODE == "train":
            emulator_model = load_emulator(EMULATOR_CHECKPOINT_PATH, config, device)
            emulator_model.eval()
            for param in emulator_model.parameters():
                param.requires_grad = False

    # --- 3. Logging ---
    log_file = open(os.path.join(output_dir, "sr_training_log.csv"), "w")
    log_file.write(
        "epoch,train_loss_total,train_loss_mae,train_loss_metric,"
        "val_loss_total,val_loss_mae,val_loss_metric,"
        "val_consistency_gap,val_intrinsic_error,"
        "epoch_duration_sec,current_metric_weight,avg_trust_weight\n"
    )
    log_file.close()

    # --- 4. Training Loop ---
    print(f"Starting SR training... | Mode: {METRIC_LOSS_MODE}")
    best_val_loss = float("inf")
    patience_counter = 0
    metric_loss_scaler = 1.0

    for epoch in range(NUM_EPOCHS):
        start = time.time()

        # Weight Schedule
        current_metric_weight = 0.0
        if METRIC_LOSS_MODE == "train":
            if epoch >= METRIC_WARMUP_EPOCHS:
                progress = (epoch - METRIC_WARMUP_EPOCHS + 1) / METRIC_RAMP_EPOCHS
                current_metric_weight = min(1.0, progress) * METRIC_LOSS_WEIGHT

        sr_model.train()
        logs = {"loss": 0.0, "mae": 0.0, "metric": 0.0, "trust": 0.0}

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1} | W={current_metric_weight:.4f}"
        )

        for X, Y, Y_gamma_log in pbar:  # Y_gamma_log is from dataset
            X, Y, Y_gamma_log = X.to(device), Y.to(device), Y_gamma_log.to(device)
            optimizer.zero_grad()

            pred_X = sr_model(X)
            loss_mae = pixel_criterion(pred_X, Y)

            metric_term = torch.tensor(0.0, device=device)
            raw_metric_mean = torch.tensor(0.0, device=device)
            avg_trust = 1.0

            if METRIC_LOSS_MODE == "train" and current_metric_weight > 0:
                pred_X_phys = F.softplus(pred_X, beta=5)

                # Emulator predicts in LOG SPACE
                pred_gamma_log = emulator_model(pred_X_phys)

                # --- Soft Trust Gate (Log Space) ---
                with torch.no_grad():
                    gamma_truth_log_pred = emulator_model(Y)
                    # MSE in LOG SPACE: ||Emu(Y) - Y_gamma_log||^2
                    emu_error_matrix = F.mse_loss(
                        gamma_truth_log_pred, Y_gamma_log, reduction="none"
                    )
                    emu_error_scalar = emu_error_matrix.view(
                        emu_error_matrix.size(0), -1
                    ).mean(dim=1)

                    # Trust decay based on log-space error
                    trust_weights = torch.exp(-float(TRUST_TAU) * emu_error_scalar)
                    avg_trust = trust_weights.mean().item()

                # Metric Loss (Log Space)
                loss_vec = metric_criterion(pred_gamma_log, Y_gamma_log)
                raw_metric_mean = loss_vec.mean()

                # Weighted term
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
                loss=f"{total_loss.item():.4f}",
                mae=f"{loss_mae.item():.4f}",
                trust=f"{avg_trust:.2f}",
            )

        # Epoch Averages
        n = len(train_loader)
        avg_logs = {k: v / n for k, v in logs.items()}
        print(f"Epoch {epoch+1} Train: {avg_logs}")

        # --- Validation ---
        sr_model.eval()
        val_logs = {"loss": 0.0, "mae": 0.0, "metric": 0.0, "gap": 0.0, "intr": 0.0}

        with torch.no_grad():
            for X, Y, Y_gamma_log in tqdm(val_loader, desc="Val"):
                X, Y, Y_gamma_log = X.to(device), Y.to(device), Y_gamma_log.to(device)
                pred_X = sr_model(X)
                loss_mae = pixel_criterion(pred_X, Y)

                loss_metric_val = torch.tensor(0.0, device=device)
                gap = torch.tensor(0.0, device=device)
                intr = torch.tensor(0.0, device=device)
                metric_term_val = torch.tensor(0.0, device=device)

                if METRIC_LOSS_MODE != "none":
                    if METRIC_LOSS_MODE == "validate":
                        # Explicit TDA (Physical -> Log Transform -> Loss)
                        pred_X_np = pred_X.cpu().numpy()
                        batch_gammas_phys = []
                        for i in range(pred_X_np.shape[0]):
                            g = compute_gamma_matrix_for_image(
                                pred_X_np[i, 0], QUANTILE_LEVELS, PIXEL_SIZE_KM
                            )
                            batch_gammas_phys.append(g)

                        # TRANSFORM TO LOG SPACE
                        batch_gammas_log = np.log1p(np.array(batch_gammas_phys))
                        pred_gamma_log = (
                            torch.from_numpy(batch_gammas_log).float().to(device)
                        )

                        loss_vec = metric_criterion(pred_gamma_log, Y_gamma_log)
                        loss_metric_val = loss_vec.mean()

                    elif METRIC_LOSS_MODE == "train":
                        pred_X_phys = F.softplus(pred_X, beta=5)
                        pred_gamma_log = emulator_model(pred_X_phys)

                        # Trust (Validation)
                        gamma_truth_log = emulator_model(Y)
                        err_val = (
                            F.mse_loss(gamma_truth_log, Y_gamma_log, reduction="none")
                            .view(X.size(0), -1)
                            .mean(1)
                        )
                        trust_val = torch.exp(-float(TRUST_TAU) * err_val)

                        loss_vec = metric_criterion(pred_gamma_log, Y_gamma_log)
                        loss_metric_val = loss_vec.mean()
                        metric_term_val = (loss_vec * trust_val).mean()

                        # Consistency Stats (Log Space)
                        loss_perc_vec = metric_criterion(
                            pred_gamma_log, gamma_truth_log
                        )
                        intr_vec = metric_criterion(gamma_truth_log, Y_gamma_log)
                        gap = torch.abs(loss_metric_val - loss_perc_vec.mean())
                        intr = intr_vec.mean()

                total_loss = (
                    loss_mae
                    + (current_metric_weight * metric_loss_scaler * metric_term_val)
                    if METRIC_LOSS_MODE == "train"
                    else loss_mae
                )

                val_logs["loss"] += total_loss.item()
                val_logs["mae"] += loss_mae.item()
                val_logs["metric"] += loss_metric_val.item()
                val_logs["gap"] += gap.item()
                val_logs["intr"] += intr.item()

        n_val = len(val_loader)
        avg_val = {k: v / n_val for k, v in val_logs.items()}
        print(f"Epoch {epoch+1} Val: {avg_val}")

        # Dynamic Scaler Logic
        if METRIC_LOSS_MODE == "train" and epoch == METRIC_WARMUP_EPOCHS - 1:
            if avg_val["metric"] > 1e-6:
                metric_loss_scaler = avg_val["mae"] / avg_val["metric"]
                print(f"WARMUP COMPLETE: Scaler = {metric_loss_scaler:.4f}")
                best_val_loss = float("inf")  # Reset tracking
                # Reset LR
                for pg in optimizer.param_groups:
                    pg["lr"] = LEARNING_RATE
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5
                )

        scheduler.step(
            avg_val["metric"] if METRIC_LOSS_MODE == "validate" else avg_val["loss"]
        )

        # Checkpoint
        check_loss = (
            avg_val["metric"] if METRIC_LOSS_MODE == "validate" else avg_val["loss"]
        )
        if check_loss < best_val_loss:
            best_val_loss = check_loss
            torch.save(
                {
                    "model_state_dict": sr_model.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(output_dir, "best_sr_model.pth"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

        with open(os.path.join(output_dir, "sr_training_log.csv"), "a") as log:
            log.write(
                f"{epoch+1},{avg_logs['loss']},{avg_logs['mae']},{avg_logs['metric']},"
                f"{avg_val['loss']},{avg_val['mae']},{avg_val['metric']},"
                f"{avg_val['gap']},{avg_val['intr']},"
                f"{time.time()-start},{current_metric_weight},{avg_logs['trust']}\n"
            )

    print("SR Training Complete.")


if __name__ == "__main__":
    main()
