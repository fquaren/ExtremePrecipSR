import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import argparse
import pandas as pd
from sklearn.metrics import r2_score
from gamma_predictors import (
    GammaPredictorSeparateHeadsHard,
    GammaPredictorSeparateHeadsSoft,
)
from loss import (
    SimpleCDFLossMetric,
    calculate_monotonicity_penalty,
    calculate_p_min_penalty,
    calculate_zero_penalty,
)
from dataset import PreprocessedNpzDataset


# --- Plotting Functions ---
def _plot_single_gamma_comparison(
    sample_idx,
    all_preds_phys,
    all_targets_phys,
    all_images,
    all_losses,
    quantiles,
    title_prefix,
    sub_folder,
    output_dir,
):
    pred_gamma = all_preds_phys[sample_idx]
    target_gamma = all_targets_phys[sample_idx]
    target_image = all_images[sample_idx]
    loss = all_losses[sample_idx]
    mean_precip = np.mean(target_image)
    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, wspace=0.4)
    ax_img = fig.add_subplot(gs[0, 0])
    im = ax_img.imshow(target_image, cmap="Blues", origin="lower", vmin=0)
    ax_img.set_title(f"Target Image (Mean: {mean_precip:.2f})")
    fig.colorbar(im, ax=ax_img, shrink=0.7, label="Precipitation (mm/hr)")
    for j in range(3):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.plot(quantiles, target_gamma[j], "o-", label="Target", color="royalblue")
        ax.plot(quantiles, pred_gamma[j], "x--", label="Prediction", color="salmon")
        ax.set_title(gamma_types[j])
        ax.set_xlabel("Precip. Threshold (mm/hr)")
        ax.grid(True, linestyle="--", alpha=0.6)
        if j == 0:
            ax.legend()
    fig.suptitle(
        f"{title_prefix} | Sample {sample_idx} | Total Loss: {loss:.4f}",
        fontsize=16,
        y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", sub_folder)
    os.makedirs(plot_save_dir, exist_ok=True)
    save_path = os.path.join(
        plot_save_dir,
        f"{title_prefix.replace(' ', '_').lower()}_sample_{sample_idx}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_performance_by_quantile(
    predictions_phys,
    targets_gamma_phys,
    target_images,
    losses_log,
    quantiles,
    output_dir,
    n_samples=10,
):
    print("\nGenerating plots for best and worst samples based on total loss...")
    all_means = np.mean(target_images, axis=(1, 2))
    sorted_indices_by_mean = np.argsort(all_means)
    n_total = len(target_images)
    quantile_groups = {
        "Low_Precip (0-33%)": sorted_indices_by_mean[: int(n_total * 0.33)],
        "Mid_Precip (33-67%)": sorted_indices_by_mean[
            int(n_total * 0.33) : int(n_total * 0.67)
        ],
        "High_Precip (67-100%)": sorted_indices_by_mean[int(n_total * 0.67) :],
    }
    for group_name, candidate_indices in quantile_groups.items():
        print(f"\n--- Processing Group: {group_name} ---")
        if len(candidate_indices) == 0:
            continue
        candidate_losses = losses_log[candidate_indices]
        sorted_loss_indices_in_group = np.argsort(candidate_losses)
        best_in_group_indices = candidate_indices[
            sorted_loss_indices_in_group[:n_samples]
        ]
        worst_in_group_indices = candidate_indices[
            sorted_loss_indices_in_group[-n_samples:]
        ]
        print(f"Plotting {len(best_in_group_indices)} best samples...")
        for rank, sample_idx in enumerate(best_in_group_indices):
            _plot_single_gamma_comparison(
                sample_idx,
                predictions_phys,
                targets_gamma_phys,
                target_images,
                losses_log,
                quantiles,
                f"Best Sample #{rank+1}",
                group_name,
                output_dir,
            )
        print(f"Plotting {len(worst_in_group_indices)} worst samples...")
        for rank, sample_idx in enumerate(worst_in_group_indices):
            _plot_single_gamma_comparison(
                sample_idx,
                predictions_phys,
                targets_gamma_phys,
                target_images,
                losses_log,
                quantiles,
                f"Worst Sample #{rank+1}",
                group_name,
                output_dir,
            )
        r2_scores_in_group = []
        for idx in candidate_indices:
            pred_flat = predictions_phys[idx].flatten()
            target_flat = targets_gamma_phys[idx].flatten()
            mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
            if np.sum(mask) == 0:
                r2_scores_in_group.append(-np.inf)
            else:
                r2 = r2_score(target_flat[mask], pred_flat[mask])
                r2_scores_in_group.append(r2)
        r2_scores_in_group = np.array(r2_scores_in_group)
        mean_r2 = np.mean(r2_scores_in_group[r2_scores_in_group > -np.inf])
        print(f"Mean R² Score for group '{group_name}': {mean_r2:.4f}")


# --- Updated plot function to match new log format ---
def plot_training_log(log_path, output_dir):
    if not os.path.exists(log_path):
        print(
            f"\nWarning: Log file not found at {log_path}. Skipping training history plot."
        )
        return
    print("\nGenerating training history plot...")
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading log file with pandas: {e}. Skipping plot.")
        return

    # Define columns based on the new log file
    required_cols = [
        "epoch",
        "train_loss_total",
        "train_loss_main",
        "train_penalty_zero",
        "train_penalty_bound",
        "val_loss_total",
        "val_loss_main",
        "val_penalty_zero",
        "val_penalty_bound",
    ]
    if not all(col in df.columns for col in required_cols):
        print("Warning: Log file columns mismatch. Expected new format. Skipping plot.")
        print(f"Missing: {[c for c in required_cols if c not in df.columns]}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Training History", fontsize=16)

    # --- Plot 1: Total & Main Loss ---
    ax1.plot(
        df["epoch"],
        df["train_loss_total"],
        "o-",
        label="Train Total Loss",
        color="C0",
    )
    ax1.plot(
        df["epoch"],
        df["val_loss_total"],
        "o-",
        label="Val Total Loss",
        color="C1",
    )
    ax1.plot(
        df["epoch"],
        df["train_loss_main"],
        "x--",
        label="Train Main Loss (CDF)",
        color="C0",
        alpha=0.6,
    )
    ax1.plot(
        df["epoch"],
        df["val_loss_main"],
        "x--",
        label="Val Main Loss (CDF)",
        color="C1",
        alpha=0.6,
    )
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Total Training & Validation Loss")
    ax1.legend(ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")

    # --- Plot 2: Zero Penalty ---
    ax2.plot(
        df["epoch"],
        df["train_penalty_zero"],
        "o-",
        label="Train Zero Penalty",
        color="C2",
    )
    ax2.plot(
        df["epoch"],
        df["val_penalty_zero"],
        "o-",
        label="Val Zero Penalty",
        color="C3",
    )
    ax2.set_ylabel("Penalty Value")
    ax2.set_title("Zero (Dry Patch) Penalty")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")

    # --- Plot 3: Bound Penalties ---
    ax3.plot(
        df["epoch"],
        df["train_penalty_bound"],
        "o-",
        label="Train Bound Penalty (Mono + P_min)",
        color="C4",
    )
    ax3.plot(
        df["epoch"],
        df["val_penalty_bound"],
        "o-",
        label="Val Bound Penalty (Mono + P_min)",
        color="C5",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Penalty Value")
    ax3.set_title("Physical Bound Penalties (Monotonicity + Isoperimetric)")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training history plot to: {save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GammaPredictor model."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped experiment run directory.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{args.run_dir}'")
    print(f"Evaluating experiment from: {args.run_dir}")

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # --- Load all relevant config values ---
    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    N_QUANTILES = len(QUANTILE_LEVELS)
    PATCH_SIZE = config["PATCH_SIZE"]
    PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
    TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
    BATCH_SIZE = config.get("BATCH_SIZE", 32)
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
    HARD_CONSTRAINED = config.get("HARD_CONSTRAINED", True)
    LOSS_LAMBDA = config.get("LOSS_LAMBDA", 0.0)  # For soft constraints
    LAMBDA_BOUND = config.get("LAMBDA_BOUND", 0.0)  # For soft constraints

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dynamic Model Instantiation ---
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if HARD_CONSTRAINED:
        print("Loading GammaPredictor with hard constraints.")
        model = GammaPredictorSeparateHeadsHard(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        ).to(device)
    else:
        print("Loading GammaPredictor with soft constraints.")
        model = GammaPredictorSeparateHeadsSoft(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
        ).to(device)

    checkpoint_path = os.path.join(args.run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Error: Checkpoint file not found: '{checkpoint_path}'"
        )

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # --- Remove sigma/log_var loading ---
    # (The new checkpoint doesn't contain them)

    # --- Use the new SimpleCDFLossMetric ---
    # We use this to calculate the "main_loss" component
    evaluation_metric = SimpleCDFLossMetric(quantile_levels=QUANTILE_LEVELS).to(device)

    # Dataset now returns log_target and original physical target
    test_dataset = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
        metadata_file=TEST_METADATA_FILE,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )
    print(f"Loaded {len(test_dataset)} samples for evaluation.")

    # Store predictions and targets in PHYSICAL space for plotting
    all_preds_phys, all_targets_phys = [], []
    all_original_images, all_total_losses = [], []

    with torch.no_grad():
        for input_data, log_target_gamma, original_precip, target_gamma_phys in tqdm(
            test_loader, desc="Generating predictions and calculating losses"
        ):
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )

            # Model prediction is in PHYSICAL space
            predicted_gamma_phys = model(input_data)

            # Transform prediction to LOG-SPACE for loss calculation
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            # --- Calculate per-sample losses ---
            # 1. Main CDF Loss (per-sample)
            main_loss_per_sample = evaluation_metric(
                predicted_gamma_log, log_target_gamma
            )

            total_loss_per_sample = main_loss_per_sample

            # 2. Add soft penalties if in that mode
            if not HARD_CONSTRAINED:
                if LAMBDA_BOUND > 0:
                    pred_A = predicted_gamma_phys[:, 0, :]
                    pred_P = predicted_gamma_phys[:, 1, :]
                    mono_penalty = calculate_monotonicity_penalty(pred_A)
                    p_min_penalty = calculate_p_min_penalty(pred_A, pred_P)
                    total_loss_per_sample = total_loss_per_sample + LAMBDA_BOUND * (
                        mono_penalty + p_min_penalty
                    )

                if LOSS_LAMBDA > 0:
                    penalty_zero = calculate_zero_penalty(
                        input_data, predicted_gamma_phys
                    )
                    total_loss_per_sample = (
                        total_loss_per_sample + LOSS_LAMBDA * penalty_zero
                    )

            # Store values
            all_total_losses.append(total_loss_per_sample.cpu().numpy())
            all_preds_phys.append(predicted_gamma_phys.cpu().numpy())
            all_targets_phys.append(target_gamma_phys.cpu().numpy())
            all_original_images.append(original_precip.squeeze(1).cpu().numpy())

    # Concatenate results
    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_original_images = np.concatenate(all_original_images, axis=0)
    all_total_losses = np.concatenate(all_total_losses, axis=0)
    print(f"Generated predictions and losses for {all_preds_phys.shape[0]} samples.")
    print(f"Mean Total Evaluation Loss: {np.mean(all_total_losses):.4f}")

    # Calculate and print R^2 scores
    print("\nCalculating R^2 scores (coefficient of determination)...")
    n_samples = all_preds_phys.shape[0]
    n_features = 3 * N_QUANTILES
    preds_flat = all_preds_phys.reshape(n_samples, n_features)
    targets_flat = all_targets_phys.reshape(n_samples, n_features)

    # Handle NaNs/Infs in targets just in case
    mask = np.isfinite(targets_flat).all(axis=1) & np.isfinite(preds_flat).all(axis=1)
    if not np.all(mask):
        print(
            f"Warning: Filtering {np.sum(~mask)} samples with NaNs/Infs before R^2 calculation."
        )

    r2_scores_raw = r2_score(
        targets_flat[mask], preds_flat[mask], multioutput="raw_values"
    )
    r2_scores_matrix = r2_scores_raw.reshape(3, N_QUANTILES)
    avg_r2_A = np.mean(r2_scores_matrix[0, :])
    avg_r2_P = np.mean(r2_scores_matrix[1, :])
    avg_r2_CC = np.mean(r2_scores_matrix[2, :])
    print(f"Average R^2 Score - Area:      {avg_r2_A:.4f}")
    print(f"Average R^2 Score - Perimeter: {avg_r2_P:.4f}")
    print(f"Average R^2 Score - CC:        {avg_r2_CC:.4f}")
    r2_save_path = os.path.join(args.run_dir, "r2_scores.npz")
    np.savez_compressed(
        r2_save_path, r2_matrix=r2_scores_matrix, quantiles=QUANTILE_LEVELS
    )
    print(f"Detailed R^2 scores saved to: {r2_save_path}")

    # --- Plotting ---
    plot_gamma_performance_by_quantile(
        predictions_phys=all_preds_phys,
        targets_gamma_phys=all_targets_phys,
        target_images=all_original_images,
        losses_log=all_total_losses,
        quantiles=QUANTILE_LEVELS,
        output_dir=args.run_dir,
        n_samples=15,
    )

    log_file_path = os.path.join(args.run_dir, "training_log.csv")
    plot_training_log(log_file_path, args.run_dir)

    print("\n✅ Evaluation script finished.")
