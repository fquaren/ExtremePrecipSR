import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np
import warnings
import time
import argparse

from model import UNetSR
from dataset import SRDataset
from loss import (
    ComponentWiseCDFLoss,
    compute_gamma_matrix_for_image,
    estimate_s_inv_from_dataset,
    GeometricLossSeparate,
)
from utils import load_emulator

# Suppress warnings
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- SR Task Config ---
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
METADATA_VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
GAMMA_TARGETS_DIR = config["PREPROCESSED_DATA_DIR"]
BATCH_SIZE = config.get("SR_BATCH_SIZE", 16)
LEARNING_RATE = config.get("SR_LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("SR_WEIGHT_DECAY", 1e-5)
NUM_EPOCHS = config.get("SR_NUM_EPOCHS", 50)
NUM_WORKERS = config.get("NUM_WORKERS", 16)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "SR_UNet_Baseline")

# --- METRIC Loss Config ---
METRIC_LOSS_MODE = config.get("METRIC_LOSS_MODE", "none")  # 'none', 'train', 'validate'
METRIC_LOSS_TYPE = config.get("METRIC_LOSS_TYPE", "geometric")  # 'cdf', 'geometric'
METRIC_LOSS_WEIGHT = config.get("METRIC_LOSS_WEIGHT", 0.1)
EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)
S_ESTIMATION_SAMPLES = config.get("S_ESTIMATION_SAMPLES", 1000)
METRIC_WARMUP_EPOCHS = config.get("METRIC_WARMUP_EPOCHS", 10)
METRIC_RAMP_EPOCHS = config.get("METRIC_RAMP_EPOCHS", 20)

# --- Emulator Model Config ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hybrid")  # For loading emulator


# --- Main Execution ---
def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description="Train a Super-Resolution UNet model.")
    parser.add_argument(
        "--metric_loss_mode",
        type=str,
        default=None,  # Default to None, so config is used if not provided
        choices=["none", "train", "validate"],
        help="Override the METRIC_LOSS_MODE from the config file.",
    )
    args = parser.parse_args()

    # --- Override config with command-line arg if provided ---
    global METRIC_LOSS_MODE  # Make it global so it's seen by the load_emulator helper
    if args.metric_loss_mode is not None:
        METRIC_LOSS_MODE = args.metric_loss_mode
        print(
            f"Overriding config: Using METRIC_LOSS_MODE='{METRIC_LOSS_MODE}' from command line."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{METRIC_LOSS_MODE}_{timestamp}"  # Add mode to name
    output_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving SR experiment artifacts to: {output_dir}")
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)  # Save the original config for reference

    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))
    print(f"Loaded DEM stats: mean={dem_stats[0]}, std={dem_stats[1]}")

    # --- 1. Prepare Datasets ---
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- 2. Initialize Models and Losses ---
    sr_model = UNetSR(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(
        sr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    mse_criterion = nn.MSELoss()

    # --- Metric-based Loss ---
    emulator_model = None
    metric_criterion = None

    if METRIC_LOSS_MODE != "none":
        if METRIC_LOSS_MODE == "train":
            if not EMULATOR_CHECKPOINT_PATH:
                raise ValueError(
                    "METRIC_LOSS_MODE is 'train', but EMULATOR_CHECKPOINT_PATH is not set."
                )
            emulator_model = load_emulator(EMULATOR_CHECKPOINT_PATH, device)

        if METRIC_LOSS_TYPE == "cdf":
            print("Using ComponentWiseCDFLoss (Log-Space) as metric loss.")
            metric_criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(
                device
            )
        elif METRIC_LOSS_TYPE == "geometric":
            print("Using GeometricLossSeparate (Physical-Space) as metric loss.")
            S_inv_tensors = estimate_s_inv_from_dataset(
                train_dataset, S_ESTIMATION_SAMPLES, device
            )
            metric_criterion = GeometricLossSeparate(S_inv_tensors).to(device)
        else:
            raise ValueError(f"Unknown METRIC_LOSS_TYPE: {METRIC_LOSS_TYPE}")

    # --- 3. Setup Logging ---
    log_file_path = os.path.join(output_dir, "sr_training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss_total,train_loss_mse,train_loss_metric,"
            "val_loss_total,val_loss_mse,val_loss_metric,epoch_duration_sec,current_metric_weight\n"
        )

    # --- 4. Training & Validation Loop ---
    print(
        f"Starting SR training... | Mode: {METRIC_LOSS_MODE} | Type: {METRIC_LOSS_TYPE}"
    )
    if METRIC_LOSS_MODE == "train":
        print(
            f"Metric loss schedule: Warmup epochs={METRIC_WARMUP_EPOCHS}, Ramp epochs={METRIC_RAMP_EPOCHS}, Max weight={METRIC_LOSS_WEIGHT}"
        )

    best_val_loss = float("inf")
    patience_counter = 0
    total_start_time = time.time()
    metric_loss_scaler = 1.0  # Default, will be updated after warmup

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        current_metric_weight = 0.0
        if METRIC_LOSS_MODE == "train":
            if epoch < METRIC_WARMUP_EPOCHS:
                current_metric_weight = 0.0
            elif epoch < METRIC_WARMUP_EPOCHS + METRIC_RAMP_EPOCHS:
                ramp_progress = (epoch - METRIC_WARMUP_EPOCHS + 1) / METRIC_RAMP_EPOCHS
                current_metric_weight = min(1.0, ramp_progress) * METRIC_LOSS_WEIGHT
            else:
                current_metric_weight = METRIC_LOSS_WEIGHT

        sr_model.train()
        running_loss, running_mse, running_metric = 0.0, 0.0, 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train) | Metric W: {current_metric_weight:.3f}",
        )

        for X, Y, Y_gamma in pbar:
            X, Y, Y_gamma = (X.to(device), Y.to(device), Y_gamma.to(device))
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            loss_mse = torch.tensor(0.0, device=device)
            loss_metric = torch.tensor(0.0, device=device)

            pred_X = sr_model(X)
            loss_mse = mse_criterion(pred_X, Y)

            if METRIC_LOSS_MODE == "train":
                pred_gamma_phys = emulator_model(F.relu(pred_X))
                if METRIC_LOSS_TYPE == "cdf":
                    pred_gamma_log = torch.log1p(pred_gamma_phys)
                    true_gamma_log = torch.log1p(Y_gamma)
                    loss_A, loss_P, loss_CC = metric_criterion(
                        pred_gamma_log, true_gamma_log
                    )
                    loss_metric = torch.mean(loss_A + loss_P + loss_CC)
                elif METRIC_LOSS_TYPE == "geometric":
                    loss_metric = metric_criterion(pred_gamma_phys, Y_gamma)

                total_loss = (
                    1 - current_metric_weight
                ) * loss_mse + current_metric_weight * (
                    metric_loss_scaler * loss_metric
                )
            else:
                total_loss = loss_mse

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sr_model.parameters(), max_norm=1.0
            )  # Keep clipping
            optimizer.step()

            running_loss += total_loss.item()
            running_mse += loss_mse.item()
            if METRIC_LOSS_MODE == "train":
                running_metric += loss_metric.item()
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                metric=f"{loss_metric.item():.4f}",
            )

        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches
        avg_train_mse = running_mse / num_batches
        avg_train_metric = running_metric / num_batches
        print(
            f"Epoch {epoch+1}\n"
            f"Train Loss: Total={avg_train_loss:.5f}, MSE={avg_train_mse:.5f}, Metric={avg_train_metric:.5f}"
        )

        # --- Validation ---
        sr_model.eval()
        val_running_loss, val_running_mse, val_running_metric = 0.0, 0.0, 0.0
        with torch.no_grad():
            for X, Y, Y_gamma in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                X, Y, Y_gamma = (X.to(device), Y.to(device), Y_gamma.to(device))
                loss_metric = torch.tensor(0.0, device=device)

                pred_X = sr_model(X)
                loss_mse = mse_criterion(pred_X, Y)

                if METRIC_LOSS_MODE != "none":
                    if METRIC_LOSS_MODE == "validate":
                        pred_X_np = pred_X.cpu().numpy()
                        pred_gamma_list = []
                        for i in range(pred_X_np.shape[0]):
                            pred_field = pred_X_np[i, 0, :, :]
                            gamma_matrix = compute_gamma_matrix_for_image(
                                pred_field, QUANTILE_LEVELS, PIXEL_SIZE_KM
                            )
                            pred_gamma_list.append(gamma_matrix)
                        pred_gamma_phys = torch.from_numpy(
                            np.array(pred_gamma_list)
                        ).to(device)
                    else:  # mode == "train"
                        pred_gamma_phys = emulator_model(F.relu(pred_X))

                    if METRIC_LOSS_TYPE == "cdf":
                        pred_gamma_log = torch.log1p(pred_gamma_phys)
                        true_gamma_log = torch.log1p(Y_gamma)
                        loss_A, loss_P, loss_CC = metric_criterion(
                            pred_gamma_log, true_gamma_log
                        )
                        loss_metric = torch.mean(loss_A + loss_P + loss_CC)
                    elif METRIC_LOSS_TYPE == "geometric":
                        loss_metric = metric_criterion(pred_gamma_phys, Y_gamma)

                if METRIC_LOSS_MODE == "train":
                    total_loss = (
                        1 - current_metric_weight
                    ) * loss_mse + current_metric_weight * (
                        metric_loss_scaler * loss_metric
                    )
                else:  # 'none' or 'validate'
                    total_loss = loss_mse

                val_running_loss += total_loss.item()
                val_running_mse += loss_mse.item()
                val_running_metric += loss_metric.item()

        num_val_batches = len(val_loader)
        avg_val_loss = val_running_loss / num_val_batches
        avg_val_mse = val_running_mse / num_val_batches
        avg_val_metric = val_running_metric / num_val_batches

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if METRIC_LOSS_MODE == "validate":
            val_loss_to_check = avg_val_metric
            print(
                f"Val Loss:   Total={avg_val_loss:.5f}, MSE={avg_val_mse:.5f}, Metric={avg_val_metric:.5f} (USING METRIC FOR SCHEDULER) | Epoch Time: {epoch_duration:.2f}s"
            )
        else:  # 'none' or 'train'
            val_loss_to_check = avg_val_loss
            print(
                f"Val Loss:   Total={avg_val_loss:.5f}, MSE={avg_val_mse:.5f}, Metric={avg_val_metric:.5f} (USING TOTAL FOR SCHEDULER) | Epoch Time: {epoch_duration:.2f}s"
            )

        # --- Handle Metric Loss Warmup Completion ---
        # Calculate scaler AND reset early stopping
        if METRIC_LOSS_MODE == "train" and epoch == METRIC_WARMUP_EPOCHS - 1:
            # This was the *last* warmup epoch. Calculate the scaler
            # using the just-computed validation loss averages.
            if avg_val_metric > 1e-6:  # Use validation data, avoid division by zero
                metric_loss_scaler = avg_val_mse / avg_val_metric
                print(f"\n--- WARMUP COMPLETE (Epoch {epoch+1}) ---")
                print(f"Setting metric loss scaler to: {metric_loss_scaler:.6f}")
                print(
                    f"(Based on val_mse: {avg_val_mse:.6f} / val_metric: {avg_val_metric:.6f})\n"
                )
            else:
                print(f"\n--- WARMUP COMPLETE (Epoch {epoch+1}) ---")
                print(
                    f"--- Warning: Could not set metric scaler (val metric loss {avg_val_metric:.6f} is too small) ---\n"
                )
                metric_loss_scaler = 1.0  # Keep default

            # --- 1. Reset Early Stopping ---
            print("Resetting early stopping baseline for composite loss phase.")
            best_val_loss = float("inf")
            patience_counter = 0

            # --- 2. Reset Optimizer's Learning Rate ---
            print(f"Resetting optimizer LR to initial: {LEARNING_RATE}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE

            # --- 3. Re-initialize the Scheduler ---
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            print("Re-initialized ReduceLROnPlateau scheduler.")
            print("--------------------------------------------------\n")

        scheduler.step(val_loss_to_check)

        if val_loss_to_check < best_val_loss:
            best_val_loss = val_loss_to_check
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": sr_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            model_save_path = os.path.join(output_dir, "best_sr_model.pth")
            torch.save(checkpoint, model_save_path)
            print(
                f"Validation loss ({METRIC_LOSS_MODE} mode) decreased to {best_val_loss:.6f}. Model saved."
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:  # Simple patience
                print(
                    f"Validation loss did not improve for {patience_counter} epochs. Stopping early."
                )
                break

        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train_loss:.6f},{avg_train_mse:.6f},{avg_train_metric:.6f},"
                f"{avg_val_loss:.6f},{avg_val_mse:.6f},{avg_val_metric:.6f},{epoch_duration:.2f},{current_metric_weight:.4f}\n"
            )

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"SR Training complete. Total time: {total_training_time / 60:.2f} minutes")


if __name__ == "__main__":
    main()
