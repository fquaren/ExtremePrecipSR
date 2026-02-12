import yaml
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from gamma_predictors_v5 import BaselineCNN, IsometricCNN, ConstrainedIsometricCNN
from models_fno import ProbabilisticFNO

from loss import estimate_s_inv_from_dataset
from dataset import PrecomputedMixupDataset


def setup_evaluation(run_dir):
    """
    Loads config, sets up device, and retrieves the normalization scalar used during training.
    """
    print(f"Setting up evaluation for: {run_dir}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{run_dir}'")

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CRITICAL: Load the scalar used for input normalization ---
    # This must match the training script's logic exactly.
    data_dir = config["PREPROCESSED_DATA_DIR"]
    scaler_path = config.get(
        "MAX_LOG_PRECIP_FILE", os.path.join(data_dir, "precip_max_val.npy")
    )

    if os.path.exists(scaler_path):
        scaler_val = float(np.load(scaler_path))
        print(f"Loaded normalization scalar from disk: {scaler_val:.4f}")
    else:
        print(
            "Warning: Normalization scalar file not found. Defaulting to 5.5 (Check your paths!)"
        )
        scaler_val = 5.5

    return config, device, scaler_val


def load_model(
    config,
    device,
    run_dir,
    scaler_val,
    architecture_type=None,
):
    """
    Loads the specified architecture and restores weights.
    Now correctly passes scaler_val to Constrained architectures.
    """
    print(f"\nLoading Model Architecture: {architecture_type}")

    # Extract config parameters
    PATCH_SIZE = config["PATCH_SIZE"]
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
    N_QUANTILES = len(config["QUANTILE_LEVELS"])
    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)

    # Instantiate the correct class
    if architecture_type == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Isometric":
        model = IsometricCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Constrained":
        # CRITICAL FIX: Pass max_input_val to match training
        model = ConstrainedIsometricCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
            max_input_val=scaler_val,
        )
    elif architecture_type == "FNO":
        print("Initializing Probabilistic FNO...")
        model = ProbabilisticFNO(n_quantiles=N_QUANTILES, modes=12, width=32)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture_type}. Choose: Baseline, Isometric, Constrained, FNO"
        )

    model = model.to(device)

    # Load Weights
    checkpoint_path = os.path.join(run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Error: Checkpoint file not found: '{checkpoint_path}'"
        )

    print(f"Restoring weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    model.eval()
    return model


def load_data(config, scaler_val):
    """Loads test data loader with correct normalization."""
    # Validation: Real Data + Precomputed MixUp Data
    test_dataset = PrecomputedMixupDataset(
        preprocessed_data_dir=os.path.join(config["PREPROCESSED_DATA_DIR"], "test"),
        metadata_file=config["TEST_METADATA_FILE"],
        scaler_val=scaler_val,  # CRITICAL: Pass the scalar
        augment=False,
        include_original=True,
        include_mixup=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("BATCH_SIZE", 32),
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )
    print(f"Loaded {len(test_dataset)} samples for evaluation.")
    return test_loader


def load_s_inv(config, device, scaler_val):
    """Loads train dataset to compute S_inv for geometric loss."""
    print("Loading train dataset to compute S_inv...")
    train_dataset_for_s_inv = PrecomputedMixupDataset(
        preprocessed_data_dir=os.path.join(config["PREPROCESSED_DATA_DIR"], "train"),
        metadata_file=config["TRAIN_METADATA_FILE"],
        scaler_val=scaler_val,  # CRITICAL: Pass the scalar
        augment=True,
        include_original=True,
        include_mixup=True,
    )
    S_inv_tensors = estimate_s_inv_from_dataset(
        train_dataset_for_s_inv, config.get("S_ESTIMATION_SAMPLES", 1000), device
    )
    return S_inv_tensors


def save_metrics_text(
    output_dir, global_group_metrics, sample_wise_group_metrics, per_feature_metrics
):
    """
    Saves Global metrics, Sample-Wise metrics, and Per-feature matrices to text.
    Includes MSE and Variance in the report.
    """
    txt_path = os.path.join(output_dir, "evaluation_metrics.txt")
    csv_global_path = os.path.join(output_dir, "metrics_group_global.csv")
    csv_sample_path = os.path.join(output_dir, "metrics_group_sample_wise.csv")

    print(f"\nSaving evaluation metrics to {txt_path}...")

    # 1. Save CSVs
    global_group_metrics.to_csv(csv_global_path)
    sample_wise_group_metrics.to_csv(csv_sample_path)

    # 2. Save Readable Text Report
    try:
        with open(txt_path, "w") as f:
            f.write("======================================================\n")
            f.write("             MODEL EVALUATION REPORT                  \n")
            f.write("======================================================\n\n")

            # --- Section 1: Global Metrics ---
            f.write("1. GLOBAL METRICS (Component-Wise Aggregation)\n")
            f.write("-" * 46 + "\n")
            f.write(
                "Metrics calculated on concatenated vectors of all samples in group.\n"
            )
            f.write("Includes Target Variance (Var) to contextualize MSE and R2.\n\n")

            f.write(
                global_group_metrics.to_string(
                    float_format=lambda x: "{:.4g}".format(x)
                )
            )
            f.write("\n\n")

            # --- Section 2: Sample-Wise Metrics ---
            f.write("2. SAMPLE-WISE METRICS (Average of Individual Scores)\n")
            f.write("-" * 46 + "\n")
            f.write("Arithmetic mean of metrics calculated per sample.\n\n")

            f.write(
                sample_wise_group_metrics.to_string(
                    float_format=lambda x: "{:.4g}".format(x)
                )
            )
            f.write("\n\n")

            # --- Section 3: Per-Feature Metrics ---
            f.write("3. PER-FEATURE METRICS\n")
            f.write("-" * 46 + "\n")
            f.write("A. Averaged over Quantiles:\n")
            f.write(
                per_feature_metrics["mean_by_component"].to_string(float_format="%.4g")
            )
            f.write("\n\n")

            f.write("B. MSE Matrix (Component vs Quantile):\n")
            f.write(per_feature_metrics["mse_matrix"].to_string(float_format="%.3e"))
            f.write("\n\n")

            f.write("C. Variance Matrix (Component vs Quantile):\n")
            f.write(per_feature_metrics["var_matrix"].to_string(float_format="%.3e"))
            f.write("\n\n")

            f.write("D. R2 Matrix (Component vs Quantile):\n")
            f.write(per_feature_metrics["r2_matrix"].to_string(float_format="%.4f"))
            f.write("\n\n")

        print("Metrics saved successfully.")

    except IOError as e:
        print(f"Error saving metrics file: {e}")
