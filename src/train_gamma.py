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

# --- Updated Import ---
from dataset import PairedInterpolationDataset
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorHierarchicalSoftGated,  # The new IQ-CGN Soft
    GammaPredictorHierarchicalHardGated,  # The new IQ-CGN Hard
)

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Constants ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 128)
LEARNING_RATE = config.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-4)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
MAX_DATASET_PRECIP = float(np.load(config["MAX_PRECIP_FILE"]))

# --- Constraint Configuration ---
LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.25)
LAMBDA_MONOTONICITY = config.get("LAMBDA_MONOTONICITY", 1.0)
LAMBDA_PLAUSIBILITY = config.get("LAMBDA_PLAUSIBILITY", 1.0)
WEIGHT_A = config.get("WEIGHT_A", 1.0)
WEIGHT_P = config.get("WEIGHT_P", 1.0)
WEIGHT_CC = config.get("WEIGHT_CC", 1.0)
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.1)
CONSTRAINT_WARMUP_EPOCHS = config.get("CONSTRAINT_WARMUP_EPOCHS", 5)

# Thresholds
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)

EXPERIMENT_NAME = "GammaEmulator_PhysicsMixUp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--constraint_mode",
        type=str,
        default="hybrid",
        choices=["soft", "hard", "hybrid", "none"],
        help="Type of constraints to apply: 'soft' (penalties), 'hard' (architecture), 'hybrid' (hard + penalties).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="Vanilla",
        choices=["Vanilla", "Attention"],  # Attention = IQ-CGN
        help="Architecture backbone: 'Vanilla' (Flat Heads) or 'Attention' (Inter-Quantile Convolution).",
    )
    args = parser.parse_args()

    CONSTRAINT_MODE = args.constraint_mode
    CURRENT_ARCH = args.arch

    print(f"\n--- Configuration ---")
    print(f"Mode: {CONSTRAINT_MODE}")
    print(f"Architecture: {CURRENT_ARCH}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup Experiment Directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{CURRENT_ARCH}_{CONSTRAINT_MODE}_{timestamp}"
    output_dir = os.path.join("experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # --- 1. Prepare Datasets (PAIRED STRATEGY) ---
    print("\n--- Initializing Training Datasets ---")

    # Training: Uses MixUp, Noise, and Physics-Consistent Targets
    train_dataset = PairedInterpolationDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        quantile_levels=QUANTILE_LEVELS,
        pixel_size_km=PIXEL_SIZE_KM,
        augment=True,
        noise_std=0.05,
        mixup_alpha=0.4,
        mixup_prob=0.8,
        use_physics_consistent_mixup=True,
    )

    # Validation: Augment=False -> Disables MixUp/Noise -> Returns pure Real data
    val_dataset_full = PairedInterpolationDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
        quantile_levels=QUANTILE_LEVELS,
        pixel_size_km=PIXEL_SIZE_KM,
        augment=False,
    )

    indices = torch.randperm(len(val_dataset_full))[
        : int(config["SUBSAMPLE_FRACTION"] * len(val_dataset_full))
    ]
    val_dataset = Subset(val_dataset_full, indices)

    # --- 2. SAMPLER SETUP (Standard) ---
    print("\nInitializing Balanced Tercile Sampler...")
    try:
        # Load Metadata
        meta_df = pd.read_csv(TRAIN_METADATA_FILE, sep=r"\s+")
        max_precip_vals = meta_df["max_precip"].values

        # --- Calculate Weights ---
        wet_mask = max_precip_vals > DRIZZLE_THRESHOLD
        wet_values = max_precip_vals[wet_mask]
        tercile_1 = np.quantile(wet_values, 1.0 / 3.0)
        tercile_2 = np.quantile(wet_values, 2.0 / 3.0)

        labels = np.zeros_like(max_precip_vals, dtype=int)
        labels[
            (max_precip_vals > DRIZZLE_THRESHOLD) & (max_precip_vals <= tercile_1)
        ] = 1
        labels[(max_precip_vals > tercile_1) & (max_precip_vals <= tercile_2)] = 2
        labels[max_precip_vals > tercile_2] = 3

        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights_vec = class_weights[labels]
        sample_weights_tensor = torch.from_numpy(sample_weights_vec).float()

        print(f"Weights Shape: {sample_weights_tensor.shape}")

        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )
        print("Sampler initialized.")

    except Exception as e:
        print(f"CRITICAL ERROR in Sampler: {e}")
        exit()

    # --- 3. DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=config.get("NUM_WORKERS", 32),
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
    print(f"\nInitializing Model: {CURRENT_ARCH} ({CONSTRAINT_MODE})...")

    if CURRENT_ARCH == "Vanilla":
        # Baseline: Separate Heads (Flat)
        if CONSTRAINT_MODE in ["soft", "none"]:
            model = GammaPredictorSeparateHeadsSoft(
                input_shape=INPUT_SHAPE,
                n_quantiles=N_QUANTILES,
                activation_fn=nn.Mish(),
                max_precip_value=MAX_DATASET_PRECIP,
            )
        else:  # hard or hybrid
            model = GammaPredictorSeparateHeadsHard(
                input_shape=INPUT_SHAPE,
                n_quantiles=N_QUANTILES,
                activation_fn=nn.Mish(),
                quantile_levels=QUANTILE_LEVELS,
                pixel_area_km2=PIXEL_SIZE_KM**2,
                max_precip_value=MAX_DATASET_PRECIP,
            )

    elif CURRENT_ARCH == "Attention":
        # New: Inter-Quantile Convolutional Gated (IQ-CGN)
        Q_EMBEDDING_DIM = config.get("Q_EMBEDDING_DIM", 32)

        if CONSTRAINT_MODE in ["soft", "none"]:
            model = GammaPredictorHierarchicalSoftGated(
                input_shape=INPUT_SHAPE,
                n_quantiles=N_QUANTILES,
                activation_fn=nn.Mish(),
                max_precip_value=MAX_DATASET_PRECIP,
                q_embedding_dim=Q_EMBEDDING_DIM,
            )
        else:  # hard or hybrid
            model = GammaPredictorHierarchicalHardGated(
                input_shape=INPUT_SHAPE,
                n_quantiles=N_QUANTILES,
                activation_fn=nn.Mish(),
                quantile_levels=QUANTILE_LEVELS,
                pixel_area_km2=PIXEL_SIZE_KM**2,
                max_precip_value=MAX_DATASET_PRECIP,
                q_embedding_dim=Q_EMBEDDING_DIM,
            )

    model = model.to(device)

    # --- 5. Optimizer & Loss ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    # Logging setup
    log_file_path = os.path.join(output_dir, "training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss_total,train_loss_main,train_loss_A,train_loss_P,train_loss_CC,"
            "train_penalty_zero,train_penalty_mono,train_penalty_plaus,train_penalty_bound,"
            "val_loss_total,val_loss_main,val_loss_A,val_loss_P,val_loss_CC,"
            "val_penalty_zero,val_penalty_mono,val_penalty_plaus,val_penalty_bound\n"
        )

    # --- 6. Training Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        # Ramp-up constraints
        if epoch < CONSTRAINT_WARMUP_EPOCHS:
            constraint_weight = float(epoch) / float(CONSTRAINT_WARMUP_EPOCHS)
        else:
            constraint_weight = 1.0

        running_metrics = {
            k: 0.0
            for k in ["loss", "main", "A", "P", "CC", "zero", "mono", "plaus", "bound"]
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")

        for input_data, log_target_gamma, _, _ in pbar:
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )
            optimizer.zero_grad()

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            # --- Physical Loss ---
            loss_A, loss_P, loss_CC = criterion(predicted_gamma_log, log_target_gamma)

            SUM_WEIGHTS = WEIGHT_A + WEIGHT_P + WEIGHT_CC
            weighted_main_loss = (
                (WEIGHT_A * loss_A) + (WEIGHT_P * loss_P) + (WEIGHT_CC * loss_CC)
            ) / SUM_WEIGHTS
            main_loss = torch.mean(weighted_main_loss)

            # --- Penalties ---
            # Extract Components for Penalty Calculation
            pred_A = predicted_gamma_phys[:, 0, :]
            pred_P = predicted_gamma_phys[:, 1, :]
            pred_CC = predicted_gamma_phys[:, 2, :]

            # A. Bound Penalty (Apply to all modes, useful for CC limits)
            p_bound = torch.mean(
                calculate_bound_penalty(pred_A, pred_CC, PIXEL_SIZE_KM**2)
            )

            p_zero = p_mono = p_plaus = torch.tensor(0.0, device=device)

            if CONSTRAINT_MODE == "soft":
                # Soft Mode: Penalize non-zero output for dry inputs
                p_zero = torch.mean(
                    calculate_zero_penalty(input_data, predicted_gamma_phys)
                )
                # Soft Mode: Penalize non-monotonic Area
                p_mono = torch.mean(calculate_monotonicity_penalty(pred_A))
                # Soft Mode: Penalize P < circle(A)
                p_plaus = torch.mean(calculate_plausibility_penalty(pred_A, pred_P))

                total_loss = (
                    main_loss
                    + (LOSS_LAMBDA * constraint_weight * p_zero)
                    + (LAMBDA_MONOTONICITY * constraint_weight * p_mono)
                    + (LAMBDA_PLAUSIBILITY * constraint_weight * p_plaus)
                    + (LAMBDA_BOUND * constraint_weight * p_bound)
                )
            else:
                # Hard/Hybrid Mode: Monotonicity & Plausibility are structural.
                # Only apply bound penalty (CC vs Area) or auxiliary penalties if needed.
                total_loss = main_loss + (LAMBDA_BOUND * constraint_weight * p_bound)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Track Metrics
            running_metrics["loss"] += total_loss.item()
            running_metrics["main"] += main_loss.item()
            running_metrics["A"] += torch.mean(loss_A).item()
            running_metrics["P"] += torch.mean(loss_P).item()
            running_metrics["CC"] += torch.mean(loss_CC).item()
            running_metrics["zero"] += p_zero.item()
            running_metrics["mono"] += p_mono.item()
            running_metrics["plaus"] += p_plaus.item()
            running_metrics["bound"] += p_bound.item()

            pbar.set_postfix(loss=f"{total_loss.item():.3f}")

        # Compute Epoch Averages
        nb = len(train_loader)
        avg_train = {k: v / nb for k, v in running_metrics.items()}
        print(
            f"Epoch {epoch+1} Train: Loss={avg_train['loss']:.4f}, Main={avg_train['main']:.4f}"
        )

        # --- Validation Loop ---
        model.eval()
        val_metrics = {
            k: 0.0
            for k in ["loss", "main", "A", "P", "CC", "zero", "mono", "plaus", "bound"]
        }

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

                pred_A = predicted_gamma_phys[:, 0, :]
                pred_P = predicted_gamma_phys[:, 1, :]
                pred_CC = predicted_gamma_phys[:, 2, :]

                p_bound = torch.mean(
                    calculate_bound_penalty(pred_A, pred_CC, PIXEL_SIZE_KM**2)
                )

                if CONSTRAINT_MODE == "soft":
                    p_zero = torch.mean(
                        calculate_zero_penalty(input_data, predicted_gamma_phys)
                    )
                    total_loss = (
                        main_loss
                        + (LOSS_LAMBDA * constraint_weight * p_zero)
                        + (LAMBDA_BOUND * constraint_weight * p_bound)
                    )
                else:
                    total_loss = main_loss + (
                        LAMBDA_BOUND * constraint_weight * p_bound
                    )
                    p_zero = torch.tensor(0.0)

                val_metrics["loss"] += total_loss.item()
                val_metrics["main"] += main_loss.item()
                val_metrics["A"] += torch.mean(loss_A).item()
                val_metrics["P"] += torch.mean(loss_P).item()
                val_metrics["CC"] += torch.mean(loss_CC).item()
                val_metrics["bound"] += p_bound.item()
                val_metrics["zero"] += p_zero.item()

        nb_val = len(val_loader)
        avg_val = {k: v / nb_val for k, v in val_metrics.items()}
        scheduler.step(avg_val["loss"])

        print(
            f"Epoch {epoch+1} Val: Loss={avg_val['loss']:.4f}, A={avg_val['A']:.4f}, P={avg_val['P']:.4f}, CC={avg_val['CC']:.4f}"
        )

        # Early Stopping Logic
        if epoch == CONSTRAINT_WARMUP_EPOCHS:
            best_val_loss = float(
                "inf"
            )  # Reset best loss after warmup to avoid getting stuck in local minima

        if avg_val["loss"] < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,  # Save config with checkpoint for reproducibility
                },
                os.path.join(output_dir, "best_model_checkpoint.pth"),
            )
            patience_counter = 0
        elif epoch >= CONSTRAINT_WARMUP_EPOCHS:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Write Logs
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train['loss']:.6f},{avg_train['main']:.6f},"
                f"{avg_train['A']:.6f},{avg_train['P']:.6f},{avg_train['CC']:.6f},"
                f"{avg_train['zero']:.6f},{avg_train['mono']:.6f},{avg_train['plaus']:.6f},{avg_train['bound']:.6f},"
                f"{avg_val['loss']:.6f},{avg_val['main']:.6f},"
                f"{avg_val['A']:.6f},{avg_val['P']:.6f},{avg_val['CC']:.6f},"
                f"{avg_val['zero']:.6f},{0.0},{0.0},{avg_val['bound']:.6f}\n"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
