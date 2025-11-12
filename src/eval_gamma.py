import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import pandas as pd
from sklearn.metrics import r2_score

from gamma_predictors import (
    GammaPredictorSeparateHeadsHard,
    GammaPredictorSeparateHeadsSoft,
)
from loss import (
    TotalErrorMetric,
    GeometricLossSeparate,
    estimate_s_inv_from_dataset,
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
    losses_total,
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
        candidate_losses = losses_total[candidate_indices]
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
                losses_total,
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
                losses_total,
                quantiles,
                f"Worst Sample #{rank+1}",
                group_name,
                output_dir,
            )
        # R2 calculation in this function seems to be for the *group*,
        # let's keep it as is.
        r2_scores_in_group = []
        for idx in candidate_indices:
            pred_flat = predictions_phys[idx].flatten()
            target_flat = targets_gamma_phys[idx].flatten()
            mask = np.isfinite(pred_flat) & np.isfinite(target_flat)
            if np.sum(mask) < 2:
                r2_scores_in_group.append(np.nan)
            else:
                r2 = r2_score(target_flat[mask], pred_flat[mask])
                r2_scores_in_group.append(r2)
        r2_scores_in_group = np.array(r2_scores_in_group)
        mean_r2 = np.nanmean(r2_scores_in_group)
        print(f"Mean Per-Sample R² Score for group '{group_name}': {mean_r2:.4f}")


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

    # Check for main loss columns, handle if missing
    if "train_loss_main" not in df.columns:
        df["train_loss_main"] = (
            df["train_loss_A"] + df["train_loss_P"] + df["train_loss_CC"]
        )
    if "val_loss_main" not in df.columns:
        df["val_loss_main"] = df["val_loss_A"] + df["val_loss_P"] + df["val_loss_CC"]

    required_cols = [
        "epoch",
        "train_loss_total",
        "val_loss_total",
        "train_loss_A",
        "train_loss_P",
        "train_loss_CC",
        "val_loss_A",
        "val_loss_P",
        "val_loss_CC",
        "train_loss_main",
        "val_loss_main",
        "train_penalty_zero",
        "train_penalty_mono",
        "train_penalty_plaus",
        "train_penalty_bound",
        "val_penalty_zero",
        "val_penalty_mono",
        "val_penalty_plaus",
        "val_penalty_bound",
    ]

    if not all(col in df.columns for col in required_cols):
        print("Warning: Log file columns mismatch. Skipping training history plot.")
        print(f"Missing: {[c for c in required_cols if c not in df.columns]}")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle("Training History", fontsize=16)
    ax1.plot(df["epoch"], df["train_loss_total"], "o-", label="Train Total", color="C0")
    ax1.plot(df["epoch"], df["val_loss_total"], "o-", label="Val Total", color="C1")
    ax1.plot(
        df["epoch"],
        df["train_loss_main"],
        "x--",
        label="Train Main",
        color="C0",
        alpha=0.6,
    )
    ax1.plot(
        df["epoch"], df["val_loss_main"], "x--", label="Val Main", color="C1", alpha=0.6
    )
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Total & Main Loss")
    ax1.legend(ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")
    ax2.plot(
        df["epoch"],
        df["train_loss_A"],
        "s--",
        label="Train Loss A",
        color="lightblue",
        alpha=0.8,
    )
    ax2.plot(
        df["epoch"], df["val_loss_A"], "s-", label="Val Loss A", color="blue", alpha=0.8
    )
    ax2.plot(
        df["epoch"],
        df["train_loss_P"],
        "x--",
        label="Train Loss P",
        color="lightgreen",
        alpha=0.8,
    )
    ax2.plot(
        df["epoch"],
        df["val_loss_P"],
        "x-",
        label="Val Loss P",
        color="green",
        alpha=0.8,
    )
    ax2.plot(
        df["epoch"],
        df["train_loss_CC"],
        "d--",
        label="Train Loss CC",
        color="wheat",
        alpha=0.8,
    )
    ax2.plot(
        df["epoch"],
        df["val_loss_CC"],
        "d-",
        label="Val Loss CC",
        color="orange",
        alpha=0.8,
    )
    ax2.set_ylabel("Component Loss")
    ax2.set_title("Main Loss Components (in Log-Space)")
    ax2.legend(ncol=3)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")
    ax3.plot(
        df["epoch"],
        df["train_penalty_zero"],
        "p:",
        label="Train Zero Pen.",
        color="black",
        alpha=0.6,
    )
    ax3.plot(
        df["epoch"], df["val_penalty_zero"], "p-", label="Val Zero Pen.", color="black"
    )
    ax3.plot(
        df["epoch"],
        df["train_penalty_mono"],
        "s:",
        label="Train Mono Pen.",
        color="cyan",
        alpha=0.6,
    )
    ax3.plot(
        df["epoch"], df["val_penalty_mono"], "s-", label="Val Mono Pen.", color="cyan"
    )
    ax3.plot(
        df["epoch"],
        df["train_penalty_plaus"],
        "x:",
        label="Train Plaus Pen.",
        color="lime",
        alpha=0.6,
    )
    ax3.plot(
        df["epoch"], df["val_penalty_plaus"], "x-", label="Val Plaus Pen.", color="lime"
    )
    ax3.plot(
        df["epoch"],
        df["train_penalty_bound"],
        "d:",
        label="Train Bound Pen.",
        color="magenta",
        alpha=0.6,
    )
    ax3.plot(
        df["epoch"],
        df["val_penalty_bound"],
        "d-",
        label="Val Bound Pen.",
        color="magenta",
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Penalty Value")
    ax3.set_title("Soft Penalty Terms")
    ax3.legend(ncol=4)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.set_yscale("log")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training history plot to: {save_path}")


def plot_metric_distributions(
    total_losses,
    geom_losses,
    r2_A,
    r2_P,
    r2_CC,
    output_dir,
):
    """Generates box plots for the distributions of key evaluation metrics."""
    print("\nGenerating metric distribution box plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Evaluation Metric Distributions (Test Set)", fontsize=16, y=1.02)

    # Filter NaNs
    valid_total_losses = total_losses[np.isfinite(total_losses)]
    valid_geom_losses = geom_losses[np.isfinite(geom_losses)]
    valid_r2_A = r2_A[np.isfinite(r2_A)]
    valid_r2_P = r2_P[np.isfinite(r2_P)]
    valid_r2_CC = r2_CC[np.isfinite(r2_CC)]
    medianprops = dict(color="red", linewidth=1.5)

    # Plot 1: Total Loss
    ax1.boxplot(
        valid_total_losses,
        vert=True,
        patch_artist=True,
        labels=["Total Loss"],
        medianprops=medianprops,
    )
    ax1.set_title(f"Total Loss (Config) \nMean: {np.mean(valid_total_losses):.4f}")
    ax1.set_ylabel("Loss Value")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")

    # Plot 2: Geometric Loss
    ax2.boxplot(
        valid_geom_losses,
        vert=True,
        patch_artist=True,
        labels=["Geometric Loss"],
        medianprops=medianprops,
    )
    ax2.set_title(
        f"Geometric (Mahalanobis) Loss \nMean: {np.mean(valid_geom_losses):.4f}"
    )
    ax2.set_ylabel("Loss Value")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")

    # Plot 3: R^2 Score (per component)
    data_to_plot = [valid_r2_A, valid_r2_P, valid_r2_CC]
    labels = [
        f"Area (Mean: {np.mean(valid_r2_A):.3f})",
        f"Perim. (Mean: {np.mean(valid_r2_P):.3f})",
        f"CC (Mean: {np.mean(valid_r2_CC):.3f})",
    ]
    ax3.boxplot(
        data_to_plot,
        vert=True,
        patch_artist=True,
        labels=labels,
        medianprops=medianprops,
    )
    ax3.set_title("Per-Sample R² Score by Component")
    ax3.set_ylabel("R² Value")
    ax3.set_ylim(-1.05, 1.05)  # R2 can be negative
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, "evaluation_metric_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric distribution plots to: {save_path}")


def plot_gamma_mean_std_by_quantile(
    predictions_phys,
    targets_gamma_phys,
    target_images,
    quantiles,
    output_dir,
):
    """Plots the mean and std dev of Gamma functions, grouped by precipitation."""
    print("\nGenerating plots for mean/std performance by precip group...")
    all_means = np.mean(target_images, axis=(1, 2))
    sorted_indices_by_mean = np.argsort(all_means)
    n_total = len(target_images)

    # Define groups
    quantile_groups = {
        "Low_Precip (0-33%)": sorted_indices_by_mean[: int(n_total * 0.33)],
        "Mid_Precip (33-67%)": sorted_indices_by_mean[
            int(n_total * 0.33) : int(n_total * 0.67)
        ],
        "High_Precip (67-100%)": sorted_indices_by_mean[int(n_total * 0.67) :],
        "All_Samples (0-100%)": sorted_indices_by_mean,  # Add an overall plot
    }

    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", "mean_std_groups")
    os.makedirs(plot_save_dir, exist_ok=True)

    for group_name, indices in quantile_groups.items():
        print(f"--- Processing Group: {group_name} ---")
        if len(indices) == 0:
            print("Skipping group, no samples.")
            continue

        group_preds = predictions_phys[indices]
        group_targets = targets_gamma_phys[indices]

        # Calculate statistics, ignoring NaNs
        mean_preds = np.nanmean(group_preds, axis=0)
        std_preds = np.nanstd(group_preds, axis=0)
        mean_targets = np.nanmean(group_targets, axis=0)
        std_targets = np.nanstd(group_targets, axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

        for j in range(3):  # Loop over A, P, CC
            ax = axes[j]

            # Plot Target
            ax.plot(
                quantiles, mean_targets[j], "o-", label="Target Mean", color="royalblue"
            )
            ax.fill_between(
                quantiles,
                mean_targets[j] - std_targets[j],
                mean_targets[j] + std_targets[j],
                color="royalblue",
                alpha=0.2,
                label="Target ±1σ",
            )

            # Plot Prediction
            ax.plot(quantiles, mean_preds[j], "x--", label="Pred. Mean", color="salmon")
            ax.fill_between(
                quantiles,
                mean_preds[j] - std_preds[j],
                mean_preds[j] + std_preds[j],
                color="salmon",
                alpha=0.2,
                label="Pred. ±1σ",
            )

            ax.set_title(gamma_types[j])
            ax.set_xlabel("Precip. Threshold (mm/hr)")
            ax.grid(True, linestyle="--", alpha=0.6)
            if j == 0:
                ax.legend()
                ax.set_ylabel("Value")

            # Use log scale for A and P if their mean is large
            if j < 2 and np.nanmax(mean_targets[j]) > 100:
                ax.set_yscale("log")

        fig.suptitle(
            f"Mean Gamma Function Comparison (±1 Std. Dev.)\nGroup: {group_name} (N={len(indices)})",
            fontsize=16,
            y=1.05,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(
            plot_save_dir,
            f"mean_std_gamma_group_{group_name.split(' ')[0].lower()}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved mean/std plots to: {plot_save_dir}")


# --- Save_metrics function ---
def save_metrics_to_file(output_dir, metrics_dict):
    """Saves the key evaluation metrics to a text file."""
    file_path = os.path.join(output_dir, "evaluation_metrics.txt")
    print(f"\nSaving evaluation metrics to {file_path}...")

    try:
        with open(file_path, "w") as f:
            f.write("--- Overall Evaluation Metrics ---\n")
            f.write(
                f"Mean Total Evaluation Loss (from config): {metrics_dict['mean_total_loss']:.6f}\n"
            )
            f.write(
                f"Mean Geometric (Mahalanobis) Loss:      {metrics_dict['mean_geometric_loss']:.6f}\n"
            )

            f.write("\n--- Average R^2 Scores (per-feature, over all samples) ---\n")
            f.write(f"Average Gamma R^2 - Area:      {metrics_dict['avg_r2_A']:.4f}\n")
            f.write(f"Average Gamma R^2 - Perimeter: {metrics_dict['avg_r2_P']:.4f}\n")
            f.write(f"Average Gamma R^2 - CC:        {metrics_dict['avg_r2_CC']:.4f}\n")

            f.write("\n--- Mean Per-Sample R^2 Scores (averaged over samples) ---\n")
            f.write(
                f"Mean Per-Sample R^2 - Area:      {metrics_dict['mean_per_sample_r2_A']:.4f}\n"
            )
            f.write(
                f"Mean Per-Sample R^2 - Perimeter: {metrics_dict['mean_per_sample_r2_P']:.4f}\n"
            )
            f.write(
                f"Mean Per-Sample R^2 - CC:        {metrics_dict['mean_per_sample_r2_CC']:.4f}\n"
            )

        print("Metrics saved successfully.")
    except IOError as e:
        print(f"Error saving metrics file: {e}")


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
    parser.add_argument(
        "--constraint_mode",
        type=str,
        required=False,
        help="Constraint mode to use (none, soft, hybrid, hard).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"Error: Run directory not found at '{args.run_dir}'")
    print(f"Evaluating experiment from: {args.run_dir}")

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    N_QUANTILES = len(QUANTILE_LEVELS)
    PATCH_SIZE = config["PATCH_SIZE"]
    PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
    TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
    BATCH_SIZE = config.get("BATCH_SIZE", 32)
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
    if args.constraint_mode:
        CONSTRAINT_MODE = args.constraint_mode
        print(f"Overriding constraint mode to: {CONSTRAINT_MODE}")
    else:
        CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hybrid")
    S_ESTIMATION_SAMPLES = config.get("S_ESTIMATION_SAMPLES", 1000)
    TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dynamic Model Instantiation ---
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
    if CONSTRAINT_MODE == "soft" or CONSTRAINT_MODE == "none":
        print("Using SOFT constraints model (GammaPredictorSeparateHeadsSoft).")
        model = GammaPredictorSeparateHeadsSoft(
            input_shape=INPUT_SHAPE, n_quantiles=N_QUANTILES, activation_fn=nn.Mish()
        ).to(device)
    elif CONSTRAINT_MODE == "hybrid" or CONSTRAINT_MODE == "hard":
        print("Using HYBRID constraints model (GammaPredictorSeparateHeadsHard).")
        model = GammaPredictorSeparateHeadsHard(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
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

    # --- Load Test Dataset ---
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

    # --- Initialize Evaluation Metric ---
    evaluation_metric = TotalErrorMetric(
        quantile_levels=QUANTILE_LEVELS, config=config
    ).to(device)

    # --- Initialize Geometric Loss Metric ---
    print("Loading train dataset to compute S_inv for geometric loss...")
    train_dataset_for_s_inv = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        augment=False,  # No augment needed for this
    )
    S_inv_tensors = estimate_s_inv_from_dataset(
        train_dataset_for_s_inv, S_ESTIMATION_SAMPLES, device
    )

    print(
        "Note: Assuming 'GeometricLossSeparate' accepts 'reduction' argument ('mean' or 'none')."
    )
    # Original metric for mean batch loss
    geometric_metric_mean = GeometricLossSeparate(S_inv_tensors, reduction="mean").to(
        device
    )
    # New metric for per-sample loss
    geometric_metric_none = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
        device
    )

    all_preds_phys, all_targets_phys = [], []
    all_original_images, all_total_losses = [], []
    all_geom_losses = []  # for per-sample geometric loss
    total_geometric_loss = 0.0

    with torch.no_grad():
        for input_data, log_target_gamma, original_precip, target_gamma_phys in tqdm(
            test_loader, desc="Generating predictions and calculating losses"
        ):
            input_data, log_target_gamma, target_gamma_phys = (
                input_data.to(device),
                log_target_gamma.to(device),
                target_gamma_phys.to(device),
            )

            predicted_gamma_phys = model(input_data)

            # --- Calculate per-sample total loss (matching training logic) ---
            per_sample_losses = evaluation_metric(
                input_data, predicted_gamma_phys, log_target_gamma
            )

            # --- Calculate geometric loss for the batch (MODIFIED) ---
            # Calculate mean geometric loss for the batch (original)
            loss_geom_batch = geometric_metric_mean(
                predicted_gamma_phys, target_gamma_phys
            )
            total_geometric_loss += loss_geom_batch.item() * input_data.shape[0]

            # Calculate per-sample geometric loss (new)
            per_sample_geom_losses = geometric_metric_none(
                predicted_gamma_phys, target_gamma_phys
            )
            all_geom_losses.append(per_sample_geom_losses.cpu().numpy())

            all_total_losses.append(per_sample_losses.cpu().numpy())
            all_preds_phys.append(predicted_gamma_phys.cpu().numpy())
            all_targets_phys.append(target_gamma_phys.cpu().numpy())
            all_original_images.append(original_precip.squeeze(1).cpu().numpy())

    # Concatenate results
    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_original_images = np.concatenate(all_original_images, axis=0)
    all_total_losses = np.concatenate(all_total_losses, axis=0)
    all_geom_losses = np.concatenate(all_geom_losses, axis=0)

    mean_total_loss = np.nanmean(all_total_losses)
    mean_geometric_loss = np.nanmean(all_geom_losses)  # Use per-sample mean

    # Sanity check: ensure the manually accumulated mean is close
    mean_geometric_loss_accum = total_geometric_loss / len(test_dataset)
    print(f"Mean Geometric (Mahalanobis) Loss (per-sample): {mean_geometric_loss:.6f}")
    print(
        f"Mean Geometric (Mahalanobis) Loss (accum):    {mean_geometric_loss_accum:.6f}"
    )

    print(f"Generated predictions and losses for {all_preds_phys.shape[0]} samples.")
    print(f"Mean Total Evaluation Loss (from config): {mean_total_loss:.6f}")

    # --- Calculate per-feature R^2 scores (Original) ---
    print("\nCalculating R^2 scores (per-feature, over all samples)...")
    n_samples = all_preds_phys.shape[0]
    n_features = 3 * N_QUANTILES
    preds_flat = all_preds_phys.reshape(n_samples, n_features)
    targets_flat = all_targets_phys.reshape(n_samples, n_features)
    mask = np.isfinite(targets_flat).all(axis=1) & np.isfinite(preds_flat).all(axis=1)
    if not np.all(mask):
        print(f"Warning: Filtering {np.sum(~mask)} samples with NaNs/Infs.")

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
    r2_save_path = os.path.join(args.run_dir, "r2_scores_per_feature.npz")
    np.savez_compressed(
        r2_save_path, r2_matrix=r2_scores_matrix, quantiles=QUANTILE_LEVELS
    )
    print(f"Detailed per-feature R^2 scores saved to: {r2_save_path}")

    # --- Calculate per-sample R^2 scores ---
    print("\nCalculating per-sample R^2 scores (by component)...")
    all_r2_A, all_r2_P, all_r2_CC = [], [], []
    for i in range(n_samples):
        # Shape is (3, N_QUANTILES)
        pred_sample = all_preds_phys[i]
        target_sample = all_targets_phys[i]

        # R2 for Area (component 0)
        mask_A = np.isfinite(pred_sample[0]) & np.isfinite(target_sample[0])
        if np.sum(mask_A) < 2:
            all_r2_A.append(np.nan)
        else:
            all_r2_A.append(r2_score(target_sample[0][mask_A], pred_sample[0][mask_A]))

        # R2 for Perimeter (component 1)
        mask_P = np.isfinite(pred_sample[1]) & np.isfinite(target_sample[1])
        if np.sum(mask_P) < 2:
            all_r2_P.append(np.nan)
        else:
            all_r2_P.append(r2_score(target_sample[1][mask_P], pred_sample[1][mask_P]))

        # R2 for CC (component 2)
        mask_CC = np.isfinite(pred_sample[2]) & np.isfinite(target_sample[2])
        if np.sum(mask_CC) < 2:
            all_r2_CC.append(np.nan)
        else:
            all_r2_CC.append(
                r2_score(target_sample[2][mask_CC], pred_sample[2][mask_CC])
            )

    all_r2_A = np.array(all_r2_A)
    all_r2_P = np.array(all_r2_P)
    all_r2_CC = np.array(all_r2_CC)
    mean_per_sample_r2_A = np.nanmean(all_r2_A)
    mean_per_sample_r2_P = np.nanmean(all_r2_P)
    mean_per_sample_r2_CC = np.nanmean(all_r2_CC)
    print(f"Mean Per-Sample R^2 - Area:      {mean_per_sample_r2_A:.4f}")
    print(f"Mean Per-Sample R^2 - Perimeter: {mean_per_sample_r2_P:.4f}")
    print(f"Mean Per-Sample R^2 - CC:        {mean_per_sample_r2_CC:.4f}")

    # --- Save full metrics to NPZ ---
    full_metrics_save_path = os.path.join(args.run_dir, "full_evaluation_metrics.npz")
    print(f"\nSaving full metrics arrays to: {full_metrics_save_path}")
    np.savez_compressed(
        full_metrics_save_path,
        total_losses=all_total_losses,
        geometric_losses=all_geom_losses,
        r2_per_sample_Area=all_r2_A,
        r2_per_sample_Perimeter=all_r2_P,
        r2_per_sample_CC=all_r2_CC,
        r2_matrix_per_feature=r2_scores_matrix,
        quantiles=QUANTILE_LEVELS,
    )

    # --- Collect metrics and save to file ---
    metrics_to_save = {
        "mean_total_loss": mean_total_loss,
        "mean_geometric_loss": mean_geometric_loss,
        "avg_r2_A": avg_r2_A,
        "avg_r2_P": avg_r2_P,
        "avg_r2_CC": avg_r2_CC,
        "mean_per_sample_r2_A": mean_per_sample_r2_A,
        "mean_per_sample_r2_P": mean_per_sample_r2_P,
        "mean_per_sample_r2_CC": mean_per_sample_r2_CC,
    }
    save_metrics_to_file(args.run_dir, metrics_to_save)

    # --- Plotting ---
    plot_gamma_performance_by_quantile(
        predictions_phys=all_preds_phys,
        targets_gamma_phys=all_targets_phys,
        target_images=all_original_images,
        losses_total=all_total_losses,
        quantiles=QUANTILE_LEVELS,
        output_dir=args.run_dir,
        n_samples=15,
    )

    plot_metric_distributions(
        total_losses=all_total_losses,
        geom_losses=all_geom_losses,
        r2_A=all_r2_A,
        r2_P=all_r2_P,
        r2_CC=all_r2_CC,
        output_dir=args.run_dir,
    )

    plot_gamma_mean_std_by_quantile(
        predictions_phys=all_preds_phys,
        targets_gamma_phys=all_targets_phys,
        target_images=all_original_images,
        quantiles=QUANTILE_LEVELS,
        output_dir=args.run_dir,
    )

    # --- Original Training Log Plot ---
    log_file_path = os.path.join(args.run_dir, "training_log.csv")
    plot_training_log(log_file_path, args.run_dir)

    print("\n✅ Evaluation script finished.")
