import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import sys
import optuna
from datetime import datetime

# --- Local Imports ---
from loss import (
    WassersteinLogLoss,
    ComponentWiseCDFLoss,
    calculate_monotonicity_penalty,
    calculate_plausibility_penalty,
    calculate_bound_penalty,
    calculate_zero_penalty,
)
from dataset import PrecomputedMixupDataset
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorHierarchicalSoftGated_V2,
    GammaPredictorHierarchicalHardGated_V2,
)

# --- Config Setup ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")

# Load config once globally
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
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
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
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)

EXPERIMENT_NAME = "GammaEmulator_Optuna_Gated"


# --- Wrapper Class for Zero-Input Handling ---
class GatedEmulatorWrapper(nn.Module):
    """
    Wraps the base GammaPredictor to enforce f(0) = 0 using a differentiable gate.
    """

    def __init__(self, base_model, scaling_factor=100.0):
        super().__init__()
        self.base_model = base_model
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # 1. Forward pass through the backbone
        raw_output = self.base_model(x)

        # 2. Calculate the differentiable gate based on mean intensity
        input_intensity = x.mean(dim=(1, 2, 3))

        # Tanh gate: 0 -> 0, Large -> 1
        gate = torch.tanh(input_intensity * self.scaling_factor)

        # Reshape for broadcasting: [Batch, 1, 1]
        gate = gate.view(-1, 1, 1)

        # 3. Apply gate to all outputs (Area, Perimeter, CCs)
        return raw_output * gate


def get_data_loaders(data_fraction=1.0):
    """
    Initializes datasets using the internal subsetting logic of PrecomputedMixupDataset.
    Correctly aligns the WeightedRandomSampler with the subsetted indices.
    """
    print(f"\n--- Initializing Datasets (Subset Fraction: {data_fraction}) ---")

    # 1. Training Dataset (Internal Subsetting)
    train_dataset = PrecomputedMixupDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        augment=True,
        include_original=True,
        include_mixup=True,
        subset_fraction=data_fraction,
    )

    # 2. Validation Dataset
    val_dataset_full = PrecomputedMixupDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
        augment=False,
        include_original=True,
        include_mixup=False,
        subset_fraction=1.0,
    )

    indices = torch.randperm(len(val_dataset_full))[
        : int(config["SUBSAMPLE_FRACTION"] * len(val_dataset_full))
    ]
    val_dataset = Subset(val_dataset_full, indices)

    # 3. Sampler Setup
    print("Initializing Balanced Tercile Sampler for Subset...")
    try:
        # Load Metadata to calculate weights for the FULL original data
        meta_df = pd.read_csv(TRAIN_METADATA_FILE, sep=r"\s+")
        max_precip_vals = meta_df["max_precip"].values

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

        # Construct the FULL potential weight vector (Original + Mixup)
        if train_dataset.include_mixup:
            combined_weights_full = torch.cat(
                [sample_weights_tensor, sample_weights_tensor], dim=0
            )
        else:
            combined_weights_full = sample_weights_tensor

        # Slice weights to match the subsetted dataset
        if hasattr(train_dataset, "indices_map"):
            if train_dataset.indices_map.max() >= len(combined_weights_full):
                print(
                    "WARNING: Indices map exceeds calculated weights length. Check Mixup/Metadata alignment."
                )
            subset_weights = combined_weights_full[train_dataset.indices_map]
        else:
            print(
                "WARNING: 'indices_map' not found. Using full weights (Sample mismatch likely)."
            )
            subset_weights = combined_weights_full

        print(
            f"Sampler Weights Shape: {subset_weights.shape} | Dataset Length: {len(train_dataset)}"
        )

        sampler = WeightedRandomSampler(
            weights=subset_weights,
            num_samples=len(subset_weights),
            replacement=True,
        )
    except Exception as e:
        print(f"CRITICAL ERROR in Sampler: {e}")
        sys.exit(1)

    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=config.get("NUM_WORKERS", 8),
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 8),
        pin_memory=True,
    )

    return train_loader, val_loader


def run_training_session(hyperparams, train_loader, val_loader, args, trial=None):
    """
    Core training logic. Decoupled from Optuna trial object to allow standalone execution.

    Args:
        hyperparams (dict): Dictionary containing 'lr' and 'weight_decay'.
        train_loader, val_loader: PyTorch DataLoaders.
        args: Command line arguments namespace.
        trial (optuna.trial.Trial, optional): Optuna trial for pruning. Defaults to None.
    """
    learning_rate = hyperparams["lr"]
    weight_decay = hyperparams["weight_decay"]

    CONSTRAINT_MODE = args.constraint_mode
    current_arch = args.arch if args.arch else config.get("ARCHITECTURE", "Vanilla")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup Experiment Directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if trial is not None:
        run_id = f"T{trial.number}"
    else:
        run_id = "SingleRun"

    run_name = (
        f"{EXPERIMENT_NAME}_{CONSTRAINT_MODE}_{current_arch}_{run_id}_{timestamp}"
    )
    output_dir = os.path.join("experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting Run: {run_name}")
    print(f"Hyperparams: LR={learning_rate:.2e}, WD={weight_decay:.2e}")

    # Dump config for this run
    run_config = config.copy()
    run_config["LEARNING_RATE"] = learning_rate
    run_config["WEIGHT_DECAY"] = weight_decay
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(run_config, f)

    # --- Model Init ---
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if current_arch == "Vanilla":
        HARD_EMULATOR = GammaPredictorSeparateHeadsHard
        SOFT_EMULATOR = GammaPredictorSeparateHeadsSoft
    elif current_arch == "Attention":
        HARD_EMULATOR = GammaPredictorHierarchicalHardGated_V2
        SOFT_EMULATOR = GammaPredictorHierarchicalSoftGated_V2

    if CONSTRAINT_MODE in ["soft", "none"]:
        base_model = SOFT_EMULATOR(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            max_precip_value=MAX_DATASET_PRECIP,
        )
    elif CONSTRAINT_MODE in ["hybrid", "hard"]:
        base_model = HARD_EMULATOR(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
            max_precip_value=MAX_DATASET_PRECIP,
        )

    model = GatedEmulatorWrapper(base_model, scaling_factor=100.0).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = WassersteinLogLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    # Logging setup
    log_file_path = os.path.join(output_dir, "training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss_total,train_loss_main,train_loss_A,train_loss_P,train_loss_CC,"
            "train_penalty_zero,train_penalty_mono,train_penalty_plaus,train_penalty_bound,"
            "val_loss_total,val_loss_main,val_loss_A,val_loss_P,val_loss_CC,"
            "val_penalty_zero,val_penalty_mono,val_penalty_plaus,val_penalty_bound\n"
        )

    # --- Training Loop ---
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()

        if epoch < CONSTRAINT_WARMUP_EPOCHS:
            constraint_weight = float(epoch) / float(CONSTRAINT_WARMUP_EPOCHS)
        else:
            constraint_weight = 1.0

        running_metrics = {
            k: 0.0
            for k in ["loss", "main", "A", "P", "CC", "zero", "mono", "plaus", "bound"]
        }

        # Training iteration
        for input_data, log_target_gamma, _, _ in train_loader:
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )
            optimizer.zero_grad()

            predicted_gamma_phys = model(input_data)
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            loss_A, loss_P, loss_CC = criterion(predicted_gamma_log, log_target_gamma)

            SUM_WEIGHTS = WEIGHT_A + WEIGHT_P + WEIGHT_CC
            weighted_main_loss = (
                (WEIGHT_A * loss_A) + (WEIGHT_P * loss_P) + (WEIGHT_CC * loss_CC)
            ) / SUM_WEIGHTS
            main_loss = torch.mean(weighted_main_loss)

            pred_A = predicted_gamma_phys[:, 0, :]
            pred_P = predicted_gamma_phys[:, 1, :]
            pred_CC = predicted_gamma_phys[:, 2, :]

            p_bound = torch.mean(
                calculate_bound_penalty(pred_A, pred_CC, PIXEL_SIZE_KM**2)
            )

            p_zero = p_mono = p_plaus = torch.tensor(0.0, device=device)

            if CONSTRAINT_MODE == "soft":
                p_zero = torch.mean(
                    calculate_zero_penalty(input_data, predicted_gamma_phys)
                )
                p_mono = torch.mean(calculate_monotonicity_penalty(pred_A))
                p_plaus = torch.mean(calculate_plausibility_penalty(pred_A, pred_P))

                total_loss = (
                    main_loss
                    + (LOSS_LAMBDA * constraint_weight * p_zero)
                    + (LAMBDA_MONOTONICITY * constraint_weight * p_mono)
                    + (LAMBDA_PLAUSIBILITY * constraint_weight * p_plaus)
                    + (LAMBDA_BOUND * constraint_weight * p_bound)
                )
            else:  # Hybrid/Hard
                total_loss = main_loss + (LAMBDA_BOUND * constraint_weight * p_bound)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_metrics["loss"] += total_loss.item()
            running_metrics["main"] += main_loss.item()
            running_metrics["A"] += torch.mean(loss_A).item()
            running_metrics["P"] += torch.mean(loss_P).item()
            running_metrics["CC"] += torch.mean(loss_CC).item()
            running_metrics["zero"] += p_zero.item()
            running_metrics["mono"] += p_mono.item()
            running_metrics["plaus"] += p_plaus.item()
            running_metrics["bound"] += p_bound.item()

        nb = len(train_loader)
        avg_train = {k: v / nb for k, v in running_metrics.items()}

        # --- Validation Loop ---
        model.eval()
        val_metrics = {
            k: 0.0
            for k in ["loss", "main", "A", "P", "CC", "zero", "mono", "plaus", "bound"]
        }

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in val_loader:
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

        # Scheduler Step (based on Val Loss)
        scheduler.step(avg_val["loss"])

        # Logging
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train['loss']:.6f},{avg_train['main']:.6f},"
                f"{avg_train['A']:.6f},{avg_train['P']:.6f},{avg_train['CC']:.6f},"
                f"{avg_train['zero']:.6f},{avg_train['mono']:.6f},{avg_train['plaus']:.6f},{avg_train['bound']:.6f},"
                f"{avg_val['loss']:.6f},{avg_val['main']:.6f},"
                f"{avg_val['A']:.6f},{avg_val['P']:.6f},{avg_val['CC']:.6f},"
                f"{avg_val['zero']:.6f},{0.0},{0.0},{avg_val['bound']:.6f}\n"
            )

        # Optuna Pruning (Only if trial is present)
        if trial is not None:
            trial.report(avg_val["loss"], epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.TrialPruned()

        # Checkpoint logic
        if avg_val["loss"] < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "hyperparameters": hyperparams,
                },
                os.path.join(output_dir, "best_model_checkpoint.pth"),
            )
            patience_counter = 0
        elif epoch >= CONSTRAINT_WARMUP_EPOCHS:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered (Epoch {epoch}).")
                break

    return best_val_loss


def optuna_objective(trial, train_loader, val_loader, args):
    """
    Optuna objective wrapper.
    """
    # Define Hyperparameter Search Space
    learning_rate = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    hyperparams = {"lr": learning_rate, "weight_decay": weight_decay}

    return run_training_session(hyperparams, train_loader, val_loader, args, trial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint_mode", type=str, default="hybrid")
    parser.add_argument("--arch", type=str, default="Vanilla")
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use per epoch (0.0 to 1.0)",
    )

    # --- Optimization Control ---
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="If set, runs Optuna hyperparameter optimization. If not set, runs a single training session.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Fallback Learning rate (used if --load_params is not set)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
        help="Fallback Weight decay (used if --load_params is not set)",
    )
    parser.add_argument(
        "--load_params",
        type=str,
        default=None,
        help="Path to a YAML file containing 'lr' and 'weight_decay' to override CLI defaults.",
    )

    args = parser.parse_args()

    print(
        f"Mode: {args.constraint_mode} | Architecture: {args.arch if args.arch else config.get('ARCHITECTURE', 'Vanilla')}"
    )
    print(f"Data Usage: {args.data_fraction * 100}% (Static Subset)")

    # Load Data Loaders ONCE
    train_loader, val_loader = get_data_loaders(data_fraction=args.data_fraction)

    if args.optimize:
        print(f"\n--- Starting Optuna Optimization ({args.n_trials} trials) ---")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        func = lambda trial: optuna_objective(trial, train_loader, val_loader, args)
        study.optimize(func, n_trials=args.n_trials)

        print("\n--- Optimization Complete ---")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (Val Loss): {trial.value}")

        # Serialization Logic
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_filename = f"best_params_{args.arch}_{timestamp}.yaml"

        print(f"\n[INFO] Saving best hyperparameters to {save_filename}...")
        with open(save_filename, "w") as f:
            yaml.dump(trial.params, f)

        print(
            f"[INFO] Save Complete. Run the final experiment using: --load_params {save_filename}"
        )

    else:
        print("\n--- Starting Single Training Run (Optimization OFF) ---")

        # [NEW] Ingestion Logic
        if args.load_params:
            if os.path.exists(args.load_params):
                print(f"[INFO] Loading hyperparameters from {args.load_params}")
                with open(args.load_params, "r") as f:
                    loaded_params = yaml.safe_load(f)

                # Check for keys and assign
                lr = loaded_params.get("lr", args.lr)
                weight_decay = loaded_params.get("weight_decay", args.wd)

                if "lr" not in loaded_params:
                    print(
                        f"[WARNING] 'lr' not found in {args.load_params}, using CLI default: {args.lr}"
                    )
                if "weight_decay" not in loaded_params:
                    print(
                        f"[WARNING] 'weight_decay' not found in {args.load_params}, using CLI default: {args.wd}"
                    )
            else:
                print(f"[ERROR] Config file {args.load_params} not found! Exiting.")
                sys.exit(1)
        else:
            # Fallback to CLI args
            lr = args.lr
            weight_decay = args.wd

        hyperparams = {"lr": lr, "weight_decay": weight_decay}
        print(f"Final Configuration: LR={lr}, WD={weight_decay}")

        final_loss = run_training_session(
            hyperparams,
            train_loader,
            val_loader,
            args,
            trial=None,  # No trial means no pruning and single run logging
        )
        print(f"\nSingle run complete. Best Validation Loss: {final_loss}")


if __name__ == "__main__":
    main()
