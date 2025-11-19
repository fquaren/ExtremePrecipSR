import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
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
from dataset import PreprocessedNpzDataset, StratifiedBatchSampler
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorResNetSoftHierarchical,
    GammaPredictorResNetHardHierarchical,
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
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-4)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "Debugging")
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
MAX_DATASET_PRECIP = np.load(config["MAX_PRECIP_FILE"])[0]

# --- Constraint Configuration ---
CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hybrid")  # 'soft', 'hard', or 'hybrid'
LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.25)  # For soft zero penalty
LAMBDA_MONOTONICITY = config.get("LAMBDA_MONOTONICITY", 1.0)
LAMBDA_PLAUSIBILITY = config.get("LAMBDA_PLAUSIBILITY", 1.0)
PLAUSIBILITY_THRESHOLD = config.get("PLAUSIBILITY_THRESHOLD", 12.0)
WEIGHT_A = config.get("WEIGHT_A", 1.0)
WEIGHT_P = config.get("WEIGHT_P", 1.0)
WEIGHT_CC = config.get("WEIGHT_CC", 1.0)
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.1)

ARCHITECTURE = config.get("ARCHITECTURE", "CNN")
if ARCHITECTURE == "CNN":
    HARD_EMULATOR = GammaPredictorSeparateHeadsHard
    SOFT_EMULATOR = GammaPredictorSeparateHeadsSoft
elif ARCHITECTURE == "RESNET":
    HARD_EMULATOR = GammaPredictorResNetHardHierarchical
    SOFT_EMULATOR = GammaPredictorResNetSoftHierarchical


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained GammaPredictor model."
    )
    parser.add_argument(
        "--constraint_mode",
        type=str,
        required=False,
        default="hybrid",
        help="(Optional) Override constraint mode (none, soft, hybrid, hard).",
    )
    args = parser.parse_args()

    print("Overriding contraint mode ...")
    CONSTRAINT_MODE = args.contraint_mode
    print(f"Contraint mode: {CONSTRAINT_MODE}")

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

    # --- 1. Prepare Datasets & Sampler ---
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
        f"Stratification: {len(indices_dry)} Dry, {len(indices_normal)} Normal, {len(indices_extreme)} Extreme."
    )
    comp = config["BATCH_COMPOSITION"]
    batch_composition = {
        "dry": int(BATCH_SIZE * float(comp["dry"])),
        "normal": int(BATCH_SIZE * float(comp["normal"])),
    }
    batch_composition["extreme"] = (
        BATCH_SIZE - batch_composition["dry"] - batch_composition["normal"]
    )
    stratified_sampler = StratifiedBatchSampler(
        indices_dry, indices_normal, indices_extreme, batch_composition
    )

    # --- 2. Prepare DataLoaders ---
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
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    # Model selection now includes 'none'
    if CONSTRAINT_MODE == "soft" or CONSTRAINT_MODE == "none":
        if CONSTRAINT_MODE == "soft":
            print(
                "Using SOFT constraints model (GammaPredictorSeparateHeadsSoft) with soft penalties."
            )
        else:
            print(
                "Using NO constraints model (GammaPredictorSeparateHeadsSoft) with main loss only."
            )
        model = SOFT_EMULATOR(
            input_shape=INPUT_SHAPE, n_quantiles=N_QUANTILES, activation_fn=nn.Mish()
        ).to(device)

    elif CONSTRAINT_MODE == "hybrid" or CONSTRAINT_MODE == "hard":
        print("Using HYBRID constraints model (GammaPredictorResNetHardHierarchical).")
        model = HARD_EMULATOR(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
            max_precip_value=MAX_DATASET_PRECIP,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown CONSTRAINT_MODE: {CONSTRAINT_MODE}. Must be 'soft', 'hybrid', 'hard', or 'none'."
        )

    # Use manual weights, not homoscedastic uncertainty
    optimizer = torch.optim.Adam(
        list(model.parameters()),
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
            # Updated log file header
            log_file.write(
                "epoch,train_loss_total,train_loss_main,train_loss_A,train_loss_P,train_loss_CC,"
                "train_penalty_zero,train_penalty_mono,train_penalty_plaus,train_penalty_bound,"
                "val_loss_total,val_loss_main,val_loss_A,val_loss_P,val_loss_CC,"
                "val_penalty_zero,val_penalty_mono,val_penalty_plaus,val_penalty_bound\n"
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
        # Accumulators for component losses
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
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )
            optimizer.zero_grad()

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            # --- Calculate Main Weighted Loss ---
            loss_A, loss_P, loss_CC = criterion(predicted_gamma_log, log_target_gamma)
            # loss_A, loss_P, loss_CC are per-sample, shape [B]

            # Apply manual weights (per-sample)
            SUM_WEIGHTS = WEIGHT_A + WEIGHT_P + WEIGHT_CC
            main_loss_per_sample = (
                (WEIGHT_A / SUM_WEIGHTS * loss_A)
                + (WEIGHT_P / SUM_WEIGHTS * loss_P)
                + (WEIGHT_CC / SUM_WEIGHTS * loss_CC)
            )

            # --- REDUCE TO SCALAR (BATCH MEAN) ---
            main_loss = torch.mean(main_loss_per_sample)
            total_loss = main_loss  # total_loss is now a scalar

            # --- Initialize penalties ---
            penalty_zero = torch.tensor(0.0, device=device)
            penalty_mono = torch.tensor(0.0, device=device)
            penalty_plaus = torch.tensor(0.0, device=device)
            penalty_bound = torch.tensor(0.0, device=device)

            # --- Apply Penalties based on Mode ---
            if CONSTRAINT_MODE == "soft":
                pred_A_phys = predicted_gamma_phys[:, 0, :]
                pred_P_phys = predicted_gamma_phys[:, 1, :]
                pred_CC_phys = predicted_gamma_phys[:, 2, :]

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

                # Use simple weighted sum for loss
                total_loss = (
                    main_loss
                    + LOSS_LAMBDA * penalty_zero
                    + LAMBDA_MONOTONICITY * penalty_mono
                    + LAMBDA_PLAUSIBILITY * penalty_plaus
                    + LAMBDA_BOUND * penalty_bound
                )

            elif CONSTRAINT_MODE == "hybrid":
                pred_A_phys = predicted_gamma_phys[:, 0, :]
                pred_CC_phys = predicted_gamma_phys[:, 2, :]

                penalty_bound = torch.mean(
                    calculate_bound_penalty(pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2)
                )
                total_loss = main_loss + LAMBDA_BOUND * penalty_bound

            # If CONSTRAINT_MODE is 'none' or 'hard', no penalties are added
            # The 'hard' mode's constraints are built into the model's forward pass
            # The 'none' mode's logic is to have no penalties.

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Accumulate losses for logging ---
            running_loss_A += torch.mean(loss_A).item()
            running_loss_P += torch.mean(loss_P).item()
            running_loss_CC += torch.mean(loss_CC).item()
            running_loss += total_loss.item()
            running_main += main_loss.item()
            running_pen_zero += penalty_zero.item()
            running_pen_mono += penalty_mono.item()
            running_pen_plaus += penalty_plaus.item()
            running_pen_bound += penalty_bound.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.3f}", main=f"{main_loss.item():.3f}"
            )

        # Calculate average losses for the epoch
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
        val_running_loss_A, val_running_loss_P, val_running_loss_CC = 0.0, 0.0, 0.0
        val_running_loss, val_running_main = 0.0, 0.0
        (
            val_running_pen_zero,
            val_running_pen_mono,
            val_running_pen_plaus,
            val_running_pen_bound,
        ) = (0.0, 0.0, 0.0, 0.0)

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                input_data, log_target_gamma = input_data.to(
                    device
                ), log_target_gamma.to(device)
                predicted_gamma_phys = model(input_data)
                predicted_gamma_log = torch.log1p(predicted_gamma_phys)

                loss_A, loss_P, loss_CC = criterion(
                    predicted_gamma_log, log_target_gamma
                )
                # loss_A, loss_P, loss_CC are per-sample, shape [B]

                # Apply manual weights (per-sample)
                SUM_WEIGHTS = WEIGHT_A + WEIGHT_P + WEIGHT_CC
                main_loss_per_sample = (
                    (WEIGHT_A / SUM_WEIGHTS * loss_A)
                    + (WEIGHT_P / SUM_WEIGHTS * loss_P)
                    + (WEIGHT_CC / SUM_WEIGHTS * loss_CC)
                )

                # --- REDUCE TO SCALAR (BATCH MEAN) ---
                main_loss = torch.mean(main_loss_per_sample)
                total_loss = main_loss  # total_loss is now a scalar

                penalty_zero = torch.tensor(0.0, device=device)
                penalty_mono = torch.tensor(0.0, device=device)
                penalty_plaus = torch.tensor(0.0, device=device)
                penalty_bound = torch.tensor(0.0, device=device)

                pred_A_phys = predicted_gamma_phys[:, 0, :]
                pred_P_phys = predicted_gamma_phys[:, 1, :]
                pred_CC_phys = predicted_gamma_phys[:, 2, :]

                if CONSTRAINT_MODE == "soft":
                    penalty_zero = calculate_zero_penalty(
                        input_data, predicted_gamma_phys
                    )
                    penalty_mono = calculate_monotonicity_penalty(pred_A_phys)
                    penalty_plaus = calculate_plausibility_penalty(
                        pred_A_phys, pred_P_phys
                    )
                    penalty_bound = calculate_bound_penalty(
                        pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2
                    )
                    total_loss = (
                        main_loss
                        + LOSS_LAMBDA * penalty_zero
                        + LAMBDA_MONOTONICITY * penalty_mono
                        + LAMBDA_PLAUSIBILITY * penalty_plaus
                        + LAMBDA_BOUND * penalty_bound
                    )

                elif CONSTRAINT_MODE == "hybrid":
                    pred_A_phys = predicted_gamma_phys[:, 0, :]
                    pred_CC_phys = predicted_gamma_phys[:, 2, :]

                    penalty_bound = torch.mean(
                        calculate_bound_penalty(
                            pred_A_phys, pred_CC_phys, PIXEL_SIZE_KM**2
                        )
                    )
                    total_loss = main_loss + LAMBDA_BOUND * penalty_bound

                # If 'none' or 'hard', total_loss remains main_loss

                val_running_loss_A += torch.mean(loss_A).item()
                val_running_loss_P += torch.mean(loss_P).item()
                val_running_loss_CC += torch.mean(loss_CC).item()
                val_running_loss += total_loss.item()
                val_running_main += main_loss.item()
                val_running_pen_zero += penalty_zero.item()
                val_running_pen_mono += penalty_mono.item()
                val_running_pen_plaus += penalty_plaus.item()
                val_running_pen_bound += penalty_bound.item()

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
            f"Val Penalties: Zero={avg_val_pen_zero:.4f}, Mono={avg_val_pen_mono:.4f}, Plaus={avg_val_pen_plaus:.4f}, Bound={avg_val_pen_bound:.4f}\n"
            f"Val Comps: A={avg_val_loss_A:.4f}, P={avg_val_loss_P:.4f}, CC={avg_val_loss_CC:.4f}"
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
                f"Validation loss decreased to {best_val_loss:.6f}. Checkpoint saved."
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
                log_file.write(
                    f"{epoch+1},{avg_train_loss:.6f},{avg_main_loss:.6f},"
                    f"{avg_loss_A:.6f},{avg_loss_P:.6f},{avg_loss_CC:.6f},"
                    f"{avg_pen_zero:.6f},{avg_pen_mono:.6f},{avg_pen_plaus:.6f},{avg_pen_bound:.6f},"
                    f"{avg_val_loss:.6f},{avg_val_main_loss:.6f},"
                    f"{avg_val_loss_A:.6f},{avg_val_loss_P:.6f},{avg_val_loss_CC:.6f},"
                    f"{avg_val_pen_zero:.6f},{avg_val_pen_mono:.6f},{avg_val_pen_plaus:.6f},{avg_val_pen_bound:.6f}\n"
                )
        except IOError as e:
            print(f"Error writing to log file: {e}")

    print("Training complete.")


if __name__ == "__main__":
    main()
