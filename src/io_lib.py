import yaml
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

# Import your existing modules
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorHierarchicalSoftGated,
    GammaPredictorHierarchicalHardGated,
)
from loss import estimate_s_inv_from_dataset
from dataset import PreprocessedNpzDataset


def setup_evaluation(run_dir):
    """Loads config and sets up device."""
    print(f"Setting up evaluation for: {run_dir}")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{run_dir}'")

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return config, device


def load_model(config, device, run_dir, constraint_mode_override=None):
    """Loads the specified model and checkpoint."""
    N_QUANTILES = len(config["QUANTILE_LEVELS"])
    PATCH_SIZE = config["PATCH_SIZE"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
    ARCHITECTURE = config.get("ARCHITECTURE", "Vanilla")

    # Default overrides for current experiments
    ARCHITECTURE = "Vanilla"

    if ARCHITECTURE == "Vanilla":
        HARD_EMULATOR = GammaPredictorSeparateHeadsHard
        SOFT_EMULATOR = GammaPredictorSeparateHeadsSoft
    elif ARCHITECTURE == "Attention":
        HARD_EMULATOR = GammaPredictorHierarchicalHardGated
        SOFT_EMULATOR = GammaPredictorHierarchicalSoftGated

    CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hybrid")
    if constraint_mode_override:
        CONSTRAINT_MODE = constraint_mode_override
        print(f"Overriding constraint mode to: {CONSTRAINT_MODE}")

    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if CONSTRAINT_MODE == "soft" or CONSTRAINT_MODE == "none":
        print("Using SOFT constraints model (GammaPredictorSeparateHeadsSoft).")
        model = SOFT_EMULATOR(
            input_shape=INPUT_SHAPE, n_quantiles=N_QUANTILES, activation_fn=nn.Mish()
        ).to(device)
    elif CONSTRAINT_MODE == "hybrid" or CONSTRAINT_MODE == "hard":
        print("Using HYBRID constraints model (GammaPredictorSeparateHeadsHard).")
        model = HARD_EMULATOR(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=config["QUANTILE_LEVELS"],
            pixel_area_km2=PIXEL_SIZE_KM**2,
        ).to(device)
    else:
        raise ValueError(f"Unknown CONSTRAINT_MODE: {CONSTRAINT_MODE}")

    checkpoint_path = os.path.join(run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Error: Checkpoint file not found: '{checkpoint_path}'"
        )

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")
    return model


def load_data(config):
    """Loads test data loader."""
    test_dataset = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(config["PREPROCESSED_DATA_DIR"], "test"),
        metadata_file=config["TEST_METADATA_FILE"],
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


def load_s_inv(config, device):
    """Loads train dataset to compute S_inv for geometric loss."""
    print("Loading train dataset to compute S_inv...")
    train_dataset_for_s_inv = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(config["PREPROCESSED_DATA_DIR"], "train"),
        metadata_file=config["TRAIN_METADATA_FILE"],
        augment=False,
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

            # Use scientific notation for MSE/Var, standard for R2
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
