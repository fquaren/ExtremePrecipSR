import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import math

from loss import ComponentWiseCDFLoss
from dataset import PreprocessedNpzDataset, StratifiedBatchSampler
from gamma_predictors import (
    GammaPredictorSeparateHeadsHard,
    GammaPredictorSeparateHeadsSoft,
)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
    # "/home/fquareng/work/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
N = N_QUANTILES * 3
PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 16)
LEARNING_RATE = config.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-5)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "Debugging")
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
HARD_CONSTRAINED = config.get("HARD_CONSTRAINED", True)
LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.0)
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.0)


# --- Soft Constraint Penalty Functions ---
def calculate_monotonicity_penalty(pred_A):
    """
    Penalizes non-monotonic (increasing) values in the Area prediction.
    pred_A [B, NQ] should be monotonically decreasing.
    """
    # diffs shape: [B, NQ-1]
    diffs = pred_A[:, :-1] - pred_A[:, 1:]
    # We only care about *negative* diffs (where A_q_i < A_q_i+1)
    # F.relu(-diffs) will be > 0 only if diffs is negative.
    penalty = torch.mean(F.relu(-diffs))
    return penalty


def calculate_p_min_penalty(pred_A, pred_P):
    """
    Penalizes predictions where P < P_min.
    P_min = sqrt(4 * pi * A)
    """
    epsilon = 1e-6
    # We detach pred_A: backprop penalty only through P head.
    P_min = torch.sqrt(4 * math.pi * (pred_A.detach() + epsilon))

    # We only care about P_min > P
    penalty = torch.mean(F.relu(P_min - pred_P))
    return penalty


def calculate_zero_penalty(input_data, predicted_gamma_phys):
    """
    Penalizes non-zero predictions for dry (all-zero) input patches.
    """
    with torch.no_grad():
        # Find "dry" samples in the batch.
        is_dry_mask = input_data.sum(dim=(1, 2, 3)) <= 1e-6  # Shape [B]

    if not is_dry_mask.any():
        return torch.tensor(0.0, device=input_data.device)  # No dry samples, no penalty

    # Select the predictions for *only* the dry samples
    dry_predictions = predicted_gamma_phys[is_dry_mask]

    # The penalty is the mean sum of predicted values for these dry samples.
    penalty = torch.mean(dry_predictions.sum(dim=(1, 2)))
    return penalty


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Check for logical errors in config ---
    if not HARD_CONSTRAINED and (LOSS_LAMBDA == 0.0 and LAMBDA_BOUND == 0.0):
        print(
            "Warning: Running with SOFT constraints, but all penalty weights (LOSS_LAMBDA, LAMBDA_BOUND) are 0."
        )
    if HARD_CONSTRAINED and (LOSS_LAMBDA > 0.0 or LAMBDA_BOUND > 0.0):
        print(
            "Warning: Running with HARD constraints. Soft penalty weights will be ignored."
        )

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

    def subsample_dataset(dataset, fraction=0.1, seed=42):
        num_samples = int(fraction * len(dataset))
        if num_samples == 0 and len(dataset) > 0:
            num_samples = 1
        subset_indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )[:num_samples]
        return Subset(dataset, subset_indices)

    val_dataset = subsample_dataset(
        val_dataset_full, config["SUBSAMPLE_FRACTION"], seed=0
    )

    # --- Oversampling Strategy ---
    print("Stratifying dataset using MAX precipitation...")
    wet_event_metrics = [
        np.max(patch)
        for patch in train_dataset_full.original_patches
        if np.max(patch) > 1e-6
    ]
    if wet_event_metrics:
        extreme_threshold = np.percentile(wet_event_metrics, 95)
    else:
        extreme_threshold = float("inf")
    print(
        f"Data-driven threshold (95th percentile of MAX precip): {extreme_threshold:.4f} mm/hr"
    )
    indices_dry, indices_normal, indices_extreme = [], [], []
    for i, patch in enumerate(train_dataset_full.original_patches):
        metric = np.max(patch)
        if metric <= 1e-6:
            indices_dry.append(i)
        elif metric < extreme_threshold:
            indices_normal.append(i)
        else:
            indices_extreme.append(i)
    print(
        f"Stratification complete: {len(indices_dry)} \n"
        f"Dry, {len(indices_normal)} Normal, {len(indices_extreme)} Extreme."
    )
    batch_composition = {
        "dry": int(BATCH_SIZE * float(config["BATCH_COMPOSITION"]["dry"])),
        "normal": int(BATCH_SIZE * float(config["BATCH_COMPOSITION"]["normal"])),
        "extreme": int(BATCH_SIZE * float(config["BATCH_COMPOSITION"]["extreme"])),
    }
    stratified_sampler = StratifiedBatchSampler(
        indices_dry, indices_normal, indices_extreme, batch_composition
    )

    # --- Prepare DataLoaders ---
    train_loader = DataLoader(
        train_dataset_full,
        batch_sampler=stratified_sampler,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )

    # --- 3. Initialize Model, Optimizer, and Loss ---

    # Define the shared input shape
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if HARD_CONSTRAINED:
        print("Using GammaPredictor with hard constraints.")
        model = GammaPredictorSeparateHeadsHard(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        ).to(device)
    else:
        print("Using GammaPredictor with soft constraints.")
        model = GammaPredictorSeparateHeadsSoft(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    log_file_path = os.path.join(output_dir, "training_log.csv")
    try:
        with open(log_file_path, "w") as log_file:
            # Added columns for penalty logging
            log_file.write(
                "epoch,train_loss_total,train_loss_main,train_penalty_zero,train_penalty_bound,"
                "val_loss_total,val_loss_main,val_penalty_zero,val_penalty_bound\n"
            )
        print(f"Log file will be saved to {log_file_path}")
    except IOError as e:
        print(f"Error creating log file: {e}")
        exit()

    # --- 4. Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_penalty_zero = 0.0
        running_penalty_bound = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")
        for input_data, log_target_gamma, _, _ in pbar:
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )
            optimizer.zero_grad()

            # Model output is physical space
            predicted_gamma_phys = model(input_data)

            # Need log-space prediction for the main loss comparison
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            # --- Calculate Main Loss ---
            loss_A, loss_P, loss_CC = criterion(predicted_gamma_log, log_target_gamma)
            main_loss = loss_A + loss_P + loss_CC

            total_loss = main_loss
            penalty_zero = torch.tensor(0.0, device=device)
            penalty_bound = torch.tensor(0.0, device=device)

            # --- Add soft constraint penalties if NOT hard_constrained ---
            if not HARD_CONSTRAINED:
                if LAMBDA_BOUND > 0:
                    pred_A = predicted_gamma_phys[:, 0, :]
                    pred_P = predicted_gamma_phys[:, 1, :]
                    mono_penalty = calculate_monotonicity_penalty(pred_A)
                    p_min_penalty = calculate_p_min_penalty(pred_A, pred_P)
                    penalty_bound = mono_penalty + p_min_penalty
                    total_loss = (
                        1 - LAMBDA_BOUND
                    ) * total_loss + LAMBDA_BOUND * penalty_bound

                if LOSS_LAMBDA > 0:
                    penalty_zero = calculate_zero_penalty(
                        input_data, predicted_gamma_phys
                    )
                    total_loss = (
                        1 - LOSS_LAMBDA
                    ) * total_loss + LOSS_LAMBDA * penalty_zero

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Accumulate losses for logging ---
            running_loss += total_loss.item()
            running_main_loss += main_loss.item()
            running_penalty_zero += penalty_zero.item()
            running_penalty_bound += penalty_bound.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.3f}",
                main=f"{main_loss.item():.3f}",
                p_bound=f"{penalty_bound.item():.3f}",
                p_zero=f"{penalty_zero.item():.3f}",
            )

        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_train_main_loss = running_main_loss / num_batches if num_batches > 0 else 0
        avg_train_pen_zero = (
            running_penalty_zero / num_batches if num_batches > 0 else 0
        )
        avg_train_pen_bound = (
            running_penalty_bound / num_batches if num_batches > 0 else 0
        )

        print(
            f"Epoch {epoch+1}\n"
            f"Train Loss: Total={avg_train_loss:.4f}, "
            f"Main={avg_train_main_loss:.4f}, "
            f"ZeroPen={avg_train_pen_zero:.4f}, "
            f"BoundPen={avg_train_pen_bound:.4f}"
        )

        # --- Validation ---
        model.eval()
        val_running_loss = 0.0
        val_running_main_loss = 0.0
        val_running_penalty_zero = 0.0
        val_running_penalty_bound = 0.0

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                input_data, log_target_gamma = input_data.to(
                    device
                ), log_target_gamma.to(device)

                predicted_gamma_phys = model(input_data)
                predicted_gamma_log = torch.log1p(predicted_gamma_phys)

                # --- Calculate Main Loss ---
                loss_A, loss_P, loss_CC = criterion(
                    predicted_gamma_log, log_target_gamma
                )
                main_loss = loss_A + loss_P + loss_CC

                total_loss = main_loss
                penalty_zero = torch.tensor(0.0, device=device)
                penalty_bound = torch.tensor(0.0, device=device)

                # --- Calculate penalties for logging ---
                if not HARD_CONSTRAINED:
                    if LAMBDA_BOUND > 0:
                        pred_A = predicted_gamma_phys[:, 0, :]
                        pred_P = predicted_gamma_phys[:, 1, :]
                        mono_penalty = calculate_monotonicity_penalty(pred_A)
                        p_min_penalty = calculate_p_min_penalty(pred_A, pred_P)
                        penalty_bound = mono_penalty + p_min_penalty
                        total_loss = total_loss + LAMBDA_BOUND * penalty_bound

                    if LOSS_LAMBDA > 0:
                        penalty_zero = calculate_zero_penalty(
                            input_data, predicted_gamma_phys
                        )
                        total_loss = total_loss + LOSS_LAMBDA * penalty_zero

                val_running_loss += total_loss.item()
                val_running_main_loss += main_loss.item()
                val_running_penalty_zero += penalty_zero.item()
                val_running_penalty_bound += penalty_bound.item()

        # Calculate average validation losses
        num_val_batches = len(val_loader)
        avg_val_loss = val_running_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_main_loss = (
            val_running_main_loss / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_pen_zero = (
            val_running_penalty_zero / num_val_batches if num_val_batches > 0 else 0
        )
        avg_val_pen_bound = (
            val_running_penalty_bound / num_val_batches if num_val_batches > 0 else 0
        )

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}\n"
            f"Val Loss: Total={avg_val_loss:.4f}, "
            f"Main={avg_val_main_loss:.4f}, "
            f"ZeroPen={avg_val_pen_zero:.4f}, "
            f"BoundPen={avg_val_pen_bound:.4f}\n"
        )

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
                f"Validation loss decreased significantly to {best_val_loss:.6f}. Model checkpoint saved."
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No significant improvement for {patience_counter} epoch(s).")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs."
                )
                break

        try:
            with open(log_file_path, "a") as log_file:
                # Updated log write
                log_file.write(
                    f"{epoch+1},{avg_train_loss:.6f},{avg_train_main_loss:.6f},"
                    f"{avg_train_pen_zero:.6f},{avg_train_pen_bound:.6f},"
                    f"{avg_val_loss:.6f},{avg_val_main_loss:.6f},"
                    f"{avg_val_pen_zero:.6f},{avg_val_pen_bound:.6f}\n"
                )
        except IOError as e:
            print(f"Error writing to log file: {e}")

    print("Training complete.")


if __name__ == "__main__":
    main()
