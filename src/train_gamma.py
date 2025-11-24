import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from loss import (
    ComponentWiseCDFLoss,
    calculate_monotonicity_penalty,
    calculate_plausibility_penalty,
    calculate_bound_penalty,
    calculate_zero_penalty,
)
from dataset import PreprocessedNpzDataset
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorHierarchicalSoftGated,
    GammaPredictorHierarchicalHardGated,
)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Constants ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
N = N_QUANTILES * 3
PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 128)
LEARNING_RATE = config.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-4)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "Debugging")
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
MAX_DATASET_PRECIP = float(np.load(config["MAX_PRECIP_FILE"]))

# --- Constraint Configuration ---
CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hybrid")
LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.25)
LAMBDA_MONOTONICITY = config.get("LAMBDA_MONOTONICITY", 1.0)
LAMBDA_PLAUSIBILITY = config.get("LAMBDA_PLAUSIBILITY", 1.0)
PLAUSIBILITY_THRESHOLD = config.get("PLAUSIBILITY_THRESHOLD", 12.0)
WEIGHT_A = config.get("WEIGHT_A", 1.0)
WEIGHT_P = config.get("WEIGHT_P", 1.0)
WEIGHT_CC = config.get("WEIGHT_CC", 1.0)
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.1)
CONSTRAINT_WARMUP_EPOCHS = config.get("CONSTRAINT_WARMUP_EPOCHS", 5)

# Thresholds for Stratification
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)
HIGH_PREC_THRESHOLD = config.get("HIGH_PREC_THRESHOLD", 1.0)

# Sample Weights
SAMPLE_WEIGHTS = config.get(
    "SAMPLE_WEIGHTS", {"dry": 1.0, "normal": 1.0, "extreme": 1.0}
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GammaPredictor model."
    )
    parser.add_argument("--constraint_mode", type=str, required=False, default="hybrid")
    parser.add_argument("--arch", type=str, required=False, default="Vanilla")
    args = parser.parse_args()

    print("Overriding contraint mode ...")
    CONSTRAINT_MODE = args.constraint_mode
    print(f"Contraint mode: {CONSTRAINT_MODE}")

    current_arch = config.get("ARCHITECTURE", "Vanilla")
    if args.arch:
        print("Overriding architecture ...")
        current_arch = args.arch
    print(f"Architecture: {current_arch}")

    if current_arch == "Vanilla":
        HARD_EMULATOR = GammaPredictorSeparateHeadsHard
        SOFT_EMULATOR = GammaPredictorSeparateHeadsSoft
    elif current_arch == "Attention":
        HARD_EMULATOR = GammaPredictorHierarchicalHardGated
        SOFT_EMULATOR = GammaPredictorHierarchicalSoftGated

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Experiment Directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir = os.path.join("experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment artifacts to: {output_dir}")
    try:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
    except IOError as e:
        print(f"Error saving config file: {e}")

    # --- 1. Prepare Datasets ---
    train_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        augment=True,
    )

    val_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
        augment=False,
    )

    indices = torch.randperm(len(val_dataset_full))[
        : int(config["SUBSAMPLE_FRACTION"] * len(val_dataset_full))
    ]
    val_dataset = Subset(val_dataset_full, indices)

    # --- 2. SAMPLER SETUP ---
    print("Initializing Balanced Tercile Sampler...")
    try:
        # 1. Load Metadata
        meta_df = pd.read_csv(TRAIN_METADATA_FILE, sep=r"\s+")
        # Ensure we work with numpy for speed
        max_precip_vals = meta_df["max_precip"].values

        # 2. Identify "Wet" Data
        # We use the config threshold to define 'Zero/Dry' vs 'Wet'
        wet_mask = max_precip_vals > DRIZZLE_THRESHOLD
        wet_values = max_precip_vals[wet_mask]

        if len(wet_values) == 0:
            raise ValueError("No wet images found in dataset!")

        # 3. Calculate Terciles (33.3% and 66.6%) of the WET distribution
        tercile_1 = np.quantile(wet_values, 1.0 / 3.0)
        tercile_2 = np.quantile(wet_values, 2.0 / 3.0)

        print(
            f"Wet Precipitation Terciles: T1={tercile_1:.4f} mm/h, T2={tercile_2:.4f} mm/h"
        )

        # 4. Assign Classes (0=Dry, 1=Wet Low, 2=Wet Mid, 3=Wet High)
        labels = np.zeros_like(max_precip_vals, dtype=int)  # Default 0 (Dry)

        # Class 1: Low Wet ( Drizzle < x <= T1 )
        mask_c1 = (max_precip_vals > DRIZZLE_THRESHOLD) & (max_precip_vals <= tercile_1)
        labels[mask_c1] = 1

        # Class 2: Mid Wet ( T1 < x <= T2 )
        mask_c2 = (max_precip_vals > tercile_1) & (max_precip_vals <= tercile_2)
        labels[mask_c2] = 2

        # Class 3: High Wet ( x > T2 )
        mask_c3 = max_precip_vals > tercile_2
        labels[mask_c3] = 3

        # 5. Calculate Weights
        class_counts = np.bincount(labels)
        print("Dataset Counts:")
        print(f"  Class 0 (Dry, <= {DRIZZLE_THRESHOLD}): {class_counts[0]}")
        print(f"  Class 1 (Low, <= {tercile_1:.2f}):    {class_counts[1]}")
        print(f"  Class 2 (Mid, <= {tercile_2:.2f}):    {class_counts[2]}")
        print(f"  Class 3 (High, > {tercile_2:.2f}):    {class_counts[3]}")

        # Inverse frequency weights: W = 1 / Count
        # We add a small epsilon or max(1) to avoid division by zero if a class is empty
        class_weights = 1.0 / np.maximum(class_counts, 1)

        # Map weights to every sample
        sample_weights_vec = class_weights[labels]
        sample_weights_tensor = torch.from_numpy(sample_weights_vec).float()

        # 6. Create Sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )
        print("Sampler initialized successfully with 4-class balanced strategy.")

    except Exception as e:
        print(f"CRITICAL ERROR in Sampler: {e}")
        exit()

    # --- 3. DataLoaders ---
    train_loader = DataLoader(
        train_dataset_full,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.get("NUM_WORKERS", 16),
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 16),
        pin_memory=True,
    )

    # --- 4. Model Init ---
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if CONSTRAINT_MODE in ["soft", "none"]:
        model = SOFT_EMULATOR(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            max_precip_value=MAX_DATASET_PRECIP,
        ).to(device)
    elif CONSTRAINT_MODE in ["hybrid", "hard"]:
        model = HARD_EMULATOR(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
            max_precip_value=MAX_DATASET_PRECIP,
        ).to(device)
    else:
        raise ValueError(f"Unknown CONSTRAINT_MODE: {CONSTRAINT_MODE}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    # Logging
    log_file_path = os.path.join(output_dir, "training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss_total,train_loss_main,train_loss_A,train_loss_P,train_loss_CC,"
            "train_penalty_zero,train_penalty_mono,train_penalty_plaus,train_penalty_bound,"
            "val_loss_total,val_loss_main,val_loss_A,val_loss_P,val_loss_CC,"
            "val_penalty_zero,val_penalty_mono,val_penalty_plaus,val_penalty_bound\n"
        )

    # --- 5. Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        # --- CONSTRAINT WARMUP ---
        if epoch < CONSTRAINT_WARMUP_EPOCHS:
            constraint_weight = float(epoch) / float(CONSTRAINT_WARMUP_EPOCHS)
        else:
            constraint_weight = 1.0

        print(f"Epoch {epoch+1}: Constraint Penalty Weight = {constraint_weight:.2f}")

        # Accumulators
        running_loss_A, running_loss_P, running_loss_CC = 0.0, 0.0, 0.0
        (
            running_loss,
            running_main,
            running_pen_zero,
            running_pen_mono,
            running_pen_plaus,
            running_pen_bound,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")

        for input_data, log_target_gamma, _, _ in pbar:
            input_data = input_data.to(device)
            log_target_gamma = log_target_gamma.to(device)

            optimizer.zero_grad()

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            loss_A, loss_P, loss_CC = criterion(predicted_gamma_log, log_target_gamma)

            SUM_WEIGHTS = WEIGHT_A + WEIGHT_P + WEIGHT_CC
            weighted_main_loss = (
                (WEIGHT_A * loss_A) + (WEIGHT_P * loss_P) + (WEIGHT_CC * loss_CC)
            ) / SUM_WEIGHTS

            main_loss = torch.mean(weighted_main_loss)
            total_loss = main_loss

            # Penalties
            penalty_zero = torch.tensor(0.0, device=device)
            penalty_mono = torch.tensor(0.0, device=device)
            penalty_plaus = torch.tensor(0.0, device=device)
            penalty_bound = torch.tensor(0.0, device=device)

            pred_A_phys = predicted_gamma_phys[:, 0, :]
            pred_P_phys = predicted_gamma_phys[:, 1, :]
            pred_CC_phys = predicted_gamma_phys[:, 2, :]

            if CONSTRAINT_MODE == "soft":
                penalty_zero = torch.mean(
                    calculate_zero_penalty(input_data, predicted_gamma_phys)
                )
                penalty_mono = torch.mean(calculate_monotonicity_penalty(pred_A_phys))
                penalty_plaus = torch.mean(
                    calculate_plausibility_penalty(pred_A_phys, pred_P_phys)
                )
                penalty_bound = torch.mean(
                    calculate_bound_penalty(pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2)
                )

                total_loss = (
                    main_loss
                    + (LOSS_LAMBDA * constraint_weight * penalty_zero)
                    + (LAMBDA_MONOTONICITY * constraint_weight * penalty_mono)
                    + (LAMBDA_PLAUSIBILITY * constraint_weight * penalty_plaus)
                    + (LAMBDA_BOUND * constraint_weight * penalty_bound)
                )

            elif CONSTRAINT_MODE == "hybrid":
                penalty_bound = torch.mean(
                    calculate_bound_penalty(pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2)
                )
                total_loss = main_loss + (
                    LAMBDA_BOUND * constraint_weight * penalty_bound
                )

            total_loss.backward()
            # KEPT: Strict gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_loss += total_loss.item()
            running_main += main_loss.item()
            running_loss_A += torch.mean(loss_A).item()
            running_loss_P += torch.mean(loss_P).item()
            running_loss_CC += torch.mean(loss_CC).item()
            running_pen_zero += penalty_zero.item()
            running_pen_mono += penalty_mono.item()
            running_pen_plaus += penalty_plaus.item()
            running_pen_bound += penalty_bound.item()

            pbar.set_postfix(loss=f"{total_loss.item():.3f}")

        # Average stats
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_main_loss = running_main / num_batches if num_batches > 0 else 0
        avg_loss_A = running_loss_A / num_batches if num_batches > 0 else 0
        avg_loss_P = running_loss_P / num_batches if num_batches > 0 else 0
        avg_loss_CC = running_loss_CC / num_batches if num_batches > 0 else 0
        avg_pen_zero = running_pen_zero / num_batches if num_batches > 0 else 0
        avg_pen_mono = running_pen_mono / num_batches if num_batches > 0 else 0
        avg_pen_plaus = running_pen_plaus / num_batches if num_batches > 0 else 0
        avg_pen_bound = running_pen_bound / num_batches if num_batches > 0 else 0

        print(
            f"Epoch {epoch+1}\n"
            f"Train Loss: Total={avg_train_loss:.4f}, Main={avg_main_loss:.4f}\n"
            f"Train Penalties: Zero={avg_pen_zero:.4f}, Mono={avg_pen_mono:.4f}, Plaus={avg_pen_plaus:.4f}, Bound={avg_pen_bound:.4f}\n"
            f"Train Comps: A={avg_loss_A:.4f}, P={avg_loss_P:.4f}, CC={avg_loss_CC:.4f}"
        )

        # --- Validation ---
        model.eval()
        val_running_loss, val_running_main = 0.0, 0.0
        val_running_loss_A, val_running_loss_P, val_running_loss_CC = 0.0, 0.0, 0.0
        (
            val_running_pen_zero,
            val_running_pen_mono,
            val_running_pen_plaus,
            val_running_pen_bound,
        ) = (0.0, 0.0, 0.0, 0.0)

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in tqdm(val_loader, desc="Val"):
                input_data, log_target_gamma = input_data.to(
                    device
                ), log_target_gamma.to(device)
                predicted_gamma_phys = model(input_data)
                predicted_gamma_log = torch.log1p(predicted_gamma_phys)

                loss_A, loss_P, loss_CC = criterion(
                    predicted_gamma_log, log_target_gamma
                )
                main_loss = torch.mean(loss_A + loss_P + loss_CC)
                total_loss = main_loss

                penalty_zero = torch.tensor(0.0, device=device)
                penalty_mono = torch.tensor(0.0, device=device)
                penalty_plaus = torch.tensor(0.0, device=device)
                penalty_bound = torch.tensor(0.0, device=device)

                pred_A_phys = predicted_gamma_phys[:, 0, :]
                pred_P_phys = predicted_gamma_phys[:, 1, :]
                pred_CC_phys = predicted_gamma_phys[:, 2, :]

                if CONSTRAINT_MODE == "soft":
                    penalty_zero = torch.mean(
                        calculate_zero_penalty(input_data, predicted_gamma_phys)
                    )
                    penalty_mono = torch.mean(
                        calculate_monotonicity_penalty(pred_A_phys)
                    )
                    penalty_plaus = torch.mean(
                        calculate_plausibility_penalty(pred_A_phys, pred_P_phys)
                    )
                    penalty_bound = torch.mean(
                        calculate_bound_penalty(
                            pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2
                        )
                    )

                    total_loss = (
                        main_loss
                        + (LOSS_LAMBDA * constraint_weight * penalty_zero)
                        + (LAMBDA_MONOTONICITY * constraint_weight * penalty_mono)
                        + (LAMBDA_PLAUSIBILITY * constraint_weight * penalty_plaus)
                        + (LAMBDA_BOUND * constraint_weight * penalty_bound)
                    )

                elif CONSTRAINT_MODE == "hybrid":
                    penalty_bound = torch.mean(
                        calculate_bound_penalty(
                            pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2
                        )
                    )
                    total_loss = main_loss + (
                        LAMBDA_BOUND * constraint_weight * penalty_bound
                    )

                # Accumulate
                val_running_loss += total_loss.item()
                val_running_main += main_loss.item()
                val_running_loss_A += torch.mean(loss_A).item()
                val_running_loss_P += torch.mean(loss_P).item()
                val_running_loss_CC += torch.mean(loss_CC).item()
                val_running_pen_zero += penalty_zero.item()
                val_running_pen_mono += penalty_mono.item()
                val_running_pen_plaus += penalty_plaus.item()
                val_running_pen_bound += penalty_bound.item()

        # Calculate Validation Averages
        num_val_batches = len(val_loader)
        avg_val_loss = val_running_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_main_loss = (
            val_running_main / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_loss_A = (
            val_running_loss_A / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_loss_P = (
            val_running_loss_P / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_loss_CC = (
            val_running_loss_CC / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_pen_zero = (
            val_running_pen_zero / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_pen_mono = (
            val_running_pen_mono / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_pen_plaus = (
            val_running_pen_plaus / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_pen_bound = (
            val_running_pen_bound / num_val_batches if num_val_batches > 0 else 0
        )

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}\n"
            f"Val Loss: Total={avg_val_loss:.4f}, Main={avg_val_main_loss:.4f}\n"
            f"Val Penalties: Zero={avg_val_pen_zero:.4f}, Bound={avg_val_pen_bound:.4f}\n"
            f"Val Comps: A={avg_val_loss_A:.4f}, P={avg_val_loss_P:.4f}, CC={avg_val_loss_CC:.4f}"
        )

        # --- REVISED EARLY STOPPING LOGIC ---
        # 1. Warmup Reset: Once warmup is done, we reset the "best" baseline.
        # This prevents pre-constraint loss (which is naturally lower) from making the
        # post-constraint loss looks "bad", triggering premature stopping.
        if epoch == CONSTRAINT_WARMUP_EPOCHS:
            print(
                f"Warmup (Epoch 0-{CONSTRAINT_WARMUP_EPOCHS-1}) complete. Resetting best_val_loss baseline."
            )
            best_val_loss = float("inf")

        # 2. Checkpointing
        if avg_val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            model_save_path = os.path.join(output_dir, "best_model_checkpoint.pth")
            torch.save(checkpoint, model_save_path)
            print(
                f"Validation loss decreased to {best_val_loss:.6f}. Checkpoint saved."
            )
            # Reset patience if we found a better model
            patience_counter = 0
        else:
            # 3. Patience Counting: ONLY if we are past the warmup phase
            if epoch >= CONSTRAINT_WARMUP_EPOCHS:
                patience_counter += 1
                print(
                    f"No improvement (Post-Warmup). Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs."
                    )
                    break
            else:
                print(
                    f"Warmup phase (Epoch {epoch+1}). Patience counter not incremented."
                )

        # Write log
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train_loss:.6f},{avg_main_loss:.6f},"
                f"{avg_loss_A:.6f},{avg_loss_P:.6f},{avg_loss_CC:.6f},"
                f"{avg_pen_zero:.6f},{avg_pen_mono:.6f},{avg_pen_plaus:.6f},{avg_pen_bound:.6f},"
                f"{avg_val_loss:.6f},{avg_val_main_loss:.6f},"
                f"{avg_val_loss_A:.6f},{avg_val_loss_P:.6f},{avg_val_loss_CC:.6f},"
                f"{avg_val_pen_zero:.6f},{avg_val_pen_mono:.6f},{avg_val_pen_plaus:.6f},{avg_val_pen_bound:.6f}\n"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
