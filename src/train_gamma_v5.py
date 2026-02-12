import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import optuna
import random
from datetime import datetime

# --- Local Imports ---
from loss import (
    WassersteinLogLoss,
    calculate_bound_penalty,
    calculate_zero_penalty,
)
from dataset import PrecomputedMixupDataset

# [CHANGED] Import all architectures, including FNO
from gamma_predictors_v5 import BaselineCNN, IsometricCNN, ConstrainedIsometricCNN
from models_fno import ProbabilisticFNO

# --- Config Setup ---
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
NUM_EPOCHS = config.get("NUM_EPOCHS", 20)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 15)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)

# --- Constraint Configuration ---
LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.25)
LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.1)
CONSTRAINT_WARMUP_EPOCHS = config.get("CONSTRAINT_WARMUP_EPOCHS", 5)
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)

# Weights for the 3 components
WEIGHT_A = config.get("WEIGHT_A", 1.0)
WEIGHT_P = config.get("WEIGHT_P", 1.0)
WEIGHT_CC = config.get("WEIGHT_CC", 1.0)

EXPERIMENT_NAME = "GammaEmulator_v6"


# --- Gaussian NLL Loss for Probabilistic FNO ---
class GaussianNLLLoss(nn.Module):
    """
    Negative Log Likelihood for Gaussian Distribution.
    Returns scalar loss + component-wise breakdown for logging compatibility.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_mean, pred_var, target):
        # target: [B, 3, Q] (Log-Transformed Metrics)
        # pred_mean: [B, 3, Q]
        # pred_var: [B, 3, Q]

        # 1. Squared Error term (Mahalanobis-like)
        mse_term = (target - pred_mean) ** 2 / pred_var

        # 2. Log Variance term (Penalty for high uncertainty)
        log_var_term = torch.log(pred_var)

        # Element-wise NLL
        loss_elementwise = 0.5 * (log_var_term + mse_term)

        # Component breakdown (Average over Batch and Quantiles)
        loss_per_comp = loss_elementwise.mean(dim=(0, 2))  # Shape [3]

        loss_A = loss_per_comp[0]
        loss_P = loss_per_comp[1]
        loss_CC = loss_per_comp[2]

        return loss_A, loss_P, loss_CC


def set_seed(seed=42):
    """Ensures reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_fraction=1.0):
    print(f"\n--- Initializing Datasets (Train Fraction: {data_fraction}) ---")

    # [ADDED] Load scaler here to pass to dataset
    scaler_path = config.get(
        "MAX_LOG_PRECIP_FILE", os.path.join(PREPROCESSED_DATA_DIR, "precip_max_val.npy")
    )
    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path))
    else:
        scaler_val = 5.5  # Fallback

    # 1. Training Dataset
    train_dataset = PrecomputedMixupDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        scaler_val=scaler_val,  # <--- MUST PASS THIS
        augment=True,
        include_original=True,
        include_mixup=True,
        subset_fraction=data_fraction,
    )

    # 2. Validation Dataset
    val_subsample = data_fraction
    print(f"--- Initializing Validation (Subsample Fraction: {val_subsample}) ---")

    val_dataset = PrecomputedMixupDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
        scaler_val=scaler_val,  # <--- MUST PASS THIS
        augment=False,
        include_original=True,
        include_mixup=config.get("VAL_INCLUDE_MIXUP", True),
        subset_fraction=val_subsample,
    )

    # 3. Sampler Setup
    print("Initializing Balanced Tercile Sampler for Subset...")
    try:
        # Load Metadata
        meta_df = pd.read_csv(TRAIN_METADATA_FILE, sep=r"\s+")
        full_max_precip = meta_df["max_precip"].values
        n_original = len(full_max_precip)

        if hasattr(train_dataset, "indices_map"):
            active_indices = train_dataset.indices_map
            mapped_indices = active_indices % n_original
            subset_max_precip = full_max_precip[mapped_indices]
        else:
            print(
                "Warning: indices_map not found. Calculating weights on full metadata."
            )
            subset_max_precip = full_max_precip

        # Calculate Weights on the SUBSET only
        wet_mask = subset_max_precip > DRIZZLE_THRESHOLD
        wet_values = subset_max_precip[wet_mask]

        if len(wet_values) > 0:
            tercile_1 = np.quantile(wet_values, 1.0 / 3.0)
            tercile_2 = np.quantile(wet_values, 2.0 / 3.0)
        else:
            tercile_1, tercile_2 = 0, 0

        labels = np.zeros_like(subset_max_precip, dtype=int)
        labels[
            (subset_max_precip > DRIZZLE_THRESHOLD) & (subset_max_precip <= tercile_1)
        ] = 1
        labels[(subset_max_precip > tercile_1) & (subset_max_precip <= tercile_2)] = 2
        labels[subset_max_precip > tercile_2] = 3

        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        subset_weights_vec = class_weights[labels]
        subset_weights_tensor = torch.from_numpy(subset_weights_vec).float()

        print(
            f"Sampler Weights Shape: {subset_weights_tensor.shape} | Dataset Length: {len(train_dataset)}"
        )

        sampler = WeightedRandomSampler(
            weights=subset_weights_tensor,
            num_samples=len(subset_weights_tensor),
            replacement=True,
        )
    except Exception as e:
        print(f"CRITICAL ERROR in Sampler: {e}")
        sampler = None

    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
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
    Core training logic supporting Deterministic (Baseline/Iso/Constrained) and Probabilistic (FNO) models.
    """
    learning_rate = hyperparams["lr"]
    weight_decay = hyperparams["weight_decay"]

    current_arch = args.arch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup Experiment Directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"T{trial.number}" if trial is not None else "SingleRun"
    run_name = f"{EXPERIMENT_NAME}_{current_arch}_{run_id}_{timestamp}"
    # output_dir = os.path.join("experiment_runs", run_name)
    output_dir = os.path.join("final_experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load Scaler for Input Normalization
    scaler_path = config.get(
        "MAX_LOG_PRECIP_FILE", os.path.join(PREPROCESSED_DATA_DIR, "precip_max_val.npy")
    )
    if os.path.exists(scaler_path):
        max_input_val = float(np.load(scaler_path))
    else:
        print("Warning: Scaler file not found. Defaulting to 5.5")
        max_input_val = 5.5  # Fallback

    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    # --- Model Selection (MERGED & FIXED) ---
    if current_arch == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif current_arch == "Isometric":
        model = IsometricCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif current_arch == "Constrained":
        model = ConstrainedIsometricCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
            max_input_val=max_input_val,
        )
    elif current_arch == "FNO":
        print("Initializing Probabilistic FNO Emulator...")
        model = ProbabilisticFNO(n_quantiles=N_QUANTILES, modes=12, width=32)
    else:
        raise ValueError(f"Unknown architecture: {current_arch}")

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- Loss Selection ---
    if current_arch == "FNO":
        criterion = GaussianNLLLoss().to(device)
    else:
        criterion = WassersteinLogLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    # --- Logging ---
    log_file_path = os.path.join(output_dir, "training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,"
            "train_loss_total,train_loss_main,train_loss_A,train_loss_P,train_loss_CC,"
            "val_loss_total,val_loss_main,val_loss_A,val_loss_P,val_loss_CC,"
            "temperature\n"
        )

    best_val_loss = float("inf")
    patience_counter = 0

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):

        model.train()

        running_metrics = {
            "loss": 0.0,
            "main": 0.0,
            "loss_A": 0.0,
            "loss_P": 0.0,
            "loss_CC": 0.0,
        }

        for input_data, log_target_gamma, _, _ in train_loader:
            input_data = input_data.to(device)
            log_target_gamma = log_target_gamma.to(device)

            optimizer.zero_grad()

            # --- Forward Pass Switch ---
            if current_arch == "FNO":
                # FNO returns Mean and Variance
                mu, var = model(input_data)
                loss_A, loss_P, loss_CC = criterion(mu, var, log_target_gamma)

                # For FNO, we don't apply soft constraints, physics is learned via uncertainty
                predicted_gamma_phys = None  # Not needed for penalties
            else:
                # Deterministic Models return Prediction
                predicted_gamma_phys = model(input_data)
                predicted_gamma_log = torch.log1p(predicted_gamma_phys)
                loss_A, loss_P, loss_CC = criterion(
                    predicted_gamma_log, log_target_gamma
                )

            if current_arch == "Constrained" or current_arch == "Isometric":
                WEIGHT_A = 3.0
                WEIGHT_P = 1.0
                WEIGHT_CC = 1.5
            else:
                WEIGHT_A = 1.0
                WEIGHT_P = 1.0
                WEIGHT_CC = 1.0
            weighted_main_loss = (
                (WEIGHT_A * loss_A) + (WEIGHT_P * loss_P) + (WEIGHT_CC * loss_CC)
            ) / (WEIGHT_A + WEIGHT_P + WEIGHT_CC)
            main_loss = torch.mean(weighted_main_loss)

            total_loss = main_loss

            # [FIX 4] Apply Penalties correctly
            # We apply Zero Penalty to Constrained model to fix bias drift
            if predicted_gamma_phys is not None:
                p_bound = 0.0

                # Only apply bound penalty to unconstrained models
                if current_arch in ["Baseline", "Isometric"]:
                    pred_A = predicted_gamma_phys[:, 0, :]
                    pred_CC = predicted_gamma_phys[:, 2, :]
                    p_bound = torch.mean(
                        calculate_bound_penalty(pred_A, pred_CC, PIXEL_SIZE_KM**2)
                    )

                # Apply Zero Penalty to ALL deterministic models
                p_zero = torch.mean(
                    calculate_zero_penalty(input_data, predicted_gamma_phys)
                )

                warmup_w = min(1.0, epoch / CONSTRAINT_WARMUP_EPOCHS)
                total_loss += warmup_w * (LAMBDA_BOUND * p_bound + LOSS_LAMBDA * p_zero)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Accumulate metrics
            running_metrics["loss"] += total_loss.item()
            running_metrics["main"] += main_loss.item()
            running_metrics["loss_A"] += torch.mean(loss_A).item()
            running_metrics["loss_P"] += torch.mean(loss_P).item()
            running_metrics["loss_CC"] += torch.mean(loss_CC).item()

        avg_train = {k: v / len(train_loader) for k, v in running_metrics.items()}

        # --- Validation Loop ---
        model.eval()
        val_metrics = {
            "loss": 0.0,
            "main": 0.0,
            "loss_A": 0.0,
            "loss_P": 0.0,
            "loss_CC": 0.0,
        }

        with torch.no_grad():
            for input_data, log_target_gamma, _, _ in val_loader:
                input_data = input_data.to(device)
                log_target_gamma = log_target_gamma.to(device)

                if current_arch == "FNO":
                    mu, var = model(input_data)
                    loss_A, loss_P, loss_CC = criterion(mu, var, log_target_gamma)
                else:
                    predicted_gamma_phys = model(input_data)
                    predicted_gamma_log = torch.log1p(predicted_gamma_phys)
                    loss_A, loss_P, loss_CC = criterion(
                        predicted_gamma_log, log_target_gamma
                    )

                # Validation Loss (Main)
                weighted_val_loss = (
                    (WEIGHT_A * loss_A) + (WEIGHT_P * loss_P) + (WEIGHT_CC * loss_CC)
                ) / (WEIGHT_A + WEIGHT_P + WEIGHT_CC)
                main_loss = torch.mean(weighted_val_loss)

                val_metrics["main"] += main_loss.item()
                val_metrics["loss"] += main_loss.item()
                val_metrics["loss_A"] += torch.mean(loss_A).item()
                val_metrics["loss_P"] += torch.mean(loss_P).item()
                val_metrics["loss_CC"] += torch.mean(loss_CC).item()

        avg_val = {k: v / len(val_loader) for k, v in val_metrics.items()}

        scheduler.step(avg_val["loss"])

        # Logging
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},"
                f"{avg_train['loss']:.6f},{avg_train['main']:.6f},{avg_train['loss_A']:.6f},{avg_train['loss_P']:.6f},{avg_train['loss_CC']:.6f},"
                f"{avg_val['loss']:.6f},{avg_val['main']:.6f},{avg_val['loss_A']:.6f},{avg_val['loss_P']:.6f},{avg_val['loss_CC']:.6f}\n"
            )

        # Optuna Pruning
        if trial is not None:
            trial.report(avg_val["loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Checkpoint
        if avg_val["loss"] < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "hyperparameters": hyperparams,
                    "arch": current_arch,
                },
                os.path.join(output_dir, "best_model_checkpoint.pth"),
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if trial is None:
                    print(f"Early stopping triggered (Epoch {epoch}).")
                break

    return best_val_loss


def optuna_objective(trial, train_loader, val_loader, args):
    # Search Space
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hyperparams = {"lr": learning_rate, "weight_decay": weight_decay}
    return run_training_session(hyperparams, train_loader, val_loader, args, trial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="Baseline",
        choices=["Baseline", "Isometric", "Constrained", "FNO"],
        help="Architecture: Baseline, Isometric, Constrained, or FNO",
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--data_fraction", type=float, default=0.1)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--load_params", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"--- Emulator Training: {args.arch} ---")

    train_loader, val_loader = get_data_loaders(data_fraction=args.data_fraction)

    if args.optimize:
        print(f"Starting Optuna for {args.arch}...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(
            lambda t: optuna_objective(t, train_loader, val_loader, args),
            n_trials=args.n_trials,
        )

        print("Best params:", study.best_trial.params)
        save_filename = os.path.join(parent_path, f"best_params_{args.arch}.yaml")
        with open(save_filename, "w") as f:
            yaml.dump(study.best_trial.params, f)
        print(f"Saved to {save_filename}")

        print("\n[INFO] Starting final training with best parameters...")
        best_params = study.best_trial.params
        run_training_session(
            {"lr": best_params["lr"], "weight_decay": best_params["weight_decay"]},
            train_loader,
            val_loader,
            args,
        )
    else:
        # Load params if provided
        lr, wd = args.lr, args.wd
        if args.load_params and os.path.exists(args.load_params):
            with open(args.load_params, "r") as f:
                p = yaml.safe_load(f)
                lr = p.get("lr", lr)
                wd = p.get("weight_decay", wd)
            print(f"Loaded params: LR={lr}, WD={wd}")

        run_training_session(
            {"lr": lr, "weight_decay": wd}, train_loader, val_loader, args
        )


if __name__ == "__main__":
    main()
