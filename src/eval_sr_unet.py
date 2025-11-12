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
import warnings
import json

# --- Import from your local files ---
# (Ensure these files are in your src/ path)
from model import UNetSR
from dataset import SRDataset
from loss import (
    ComponentWiseCDFLoss,
    compute_gamma_matrix_for_image,
    estimate_s_inv_from_dataset,
    GeometricLossSeparate,
)
from metrics import compute_fss, compute_sal
from utils import load_emulator

# Suppress warnings
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


# --- Plotting Functions ---
def _plot_single_gamma_comparison(
    sample_idx,
    all_preds_phys,
    all_targets_phys,
    all_images,
    all_losses,
    title_prefix,
    sub_folder,
    output_dir,
    dem_stats,
):
    # --- Variables ---
    pred_phys = all_preds_phys[sample_idx]
    target_phys = all_targets_phys[sample_idx]
    input_stack = all_images[sample_idx]
    interp_precip = input_stack[0]
    dem_normalized = input_stack[1]
    loss = all_losses[sample_idx]

    # Un-normalize DEM
    dem_mean, dem_std = dem_stats
    dem_unnormalized = (dem_normalized * (dem_std + 1e-8)) + dem_mean

    # --- ADDED: Determine common color scale for precipitation ---
    vmin_precip = 0
    vmax_precip = np.max(
        [np.max(interp_precip), np.max(pred_phys), np.max(target_phys)]
    )

    if vmax_precip == 0:
        vmax_precip = 1.0

    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 4, wspace=0.3)

    ax_img1 = fig.add_subplot(gs[0, 0])
    im1 = ax_img1.imshow(
        interp_precip,
        cmap="Blues",
        origin="lower",
        vmin=vmin_precip,
        vmax=vmax_precip,
    )
    ax_img1.set_title(f"Input: Interp. Precip (Mean: {np.mean(interp_precip):.2f})")
    fig.colorbar(im1, ax=ax_img1, shrink=0.7, label="Precipitation (mm/hr)")

    ax_img2 = fig.add_subplot(gs[0, 1])
    im2 = ax_img2.imshow(dem_unnormalized, cmap="terrain", origin="lower")
    ax_img2.set_title("Input: DEM (Unnormalized)")
    fig.colorbar(im2, ax=ax_img2, shrink=0.7, label="Elevation (m)")

    ax_img3 = fig.add_subplot(gs[0, 2])
    im3 = ax_img3.imshow(
        pred_phys,
        cmap="Blues",
        origin="lower",
        vmin=vmin_precip,
        vmax=vmax_precip,
    )
    ax_img3.set_title(f"Prediction (Mean: {np.mean(pred_phys):.2f})")
    fig.colorbar(im3, ax=ax_img3, shrink=0.7, label="Precipitation (mm/hr)")

    ax_img4 = fig.add_subplot(gs[0, 3])
    im4 = ax_img4.imshow(
        target_phys,
        cmap="Blues",
        origin="lower",
        vmin=vmin_precip,
        vmax=vmax_precip,
    )
    ax_img4.set_title(f"Target (Mean: {np.mean(target_phys):.2f})")
    fig.colorbar(im4, ax=ax_img4, shrink=0.7, label="Precipitation (mm/hr)")

    fig.suptitle(
        f"{title_prefix} | Sample {sample_idx} | Ranking Loss: {loss:.4f}",
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


def plot_best_worst_samples(
    predictions_phys,
    targets_phys,
    inputs_phys,
    losses,
    output_dir,
    n_samples=10,
    dem_stats=None,
):
    print("\nGenerating plots for best and worst samples based on ranking loss...")
    sorted_indices = np.argsort(losses)
    best_indices = sorted_indices[:n_samples]
    worst_indices = sorted_indices[-n_samples:]

    print(f"Plotting {n_samples} best samples...")
    for rank, sample_idx in enumerate(best_indices):
        _plot_single_gamma_comparison(
            sample_idx,
            predictions_phys,
            targets_phys,
            inputs_phys,
            losses,
            f"Best Sample #{rank+1}",
            "best_samples",
            output_dir,
            dem_stats,
        )
    print(f"Plotting {n_samples} worst samples...")
    for rank, sample_idx in enumerate(worst_indices):
        _plot_single_gamma_comparison(
            sample_idx,
            predictions_phys,
            targets_phys,
            inputs_phys,
            losses,
            f"Worst Sample #{rank+1}",
            "worst_samples",
            output_dir,
            dem_stats,
        )


def plot_training_log(log_path, output_dir, config):
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

    # Load mode to customize plots
    metric_loss_mode = config.get("METRIC_LOSS_MODE", "none")

    required_cols = [
        "epoch",
        "train_loss_total",
        "val_loss_total",
        "train_loss_mse",
        "val_loss_mse",
        "train_loss_metric",
        "val_loss_metric",
    ]
    if "current_metric_weight" in config and "current_metric_weight" not in df.columns:
        required_cols.append("current_metric_weight")

    if not all(col in df.columns for col in required_cols):
        print("Warning: Log file columns mismatch. Skipping training history plot.")
        print(f"Missing: {[c for c in required_cols if c not in df.columns]}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"SR Training History (Mode: {metric_loss_mode})", fontsize=16)

    # --- Plot 1: Total Loss ---
    ax1.plot(
        df["epoch"],
        df["train_loss_total"],
        "o-",
        label="Train Total Loss",
        color="royalblue",
    )
    ax1.plot(
        df["epoch"], df["val_loss_total"], "o-", label="Val Total Loss", color="salmon"
    )

    plot1_title = "Total Training & Validation Loss"
    if metric_loss_mode == "validate":
        plot1_title += " (Note: Checkpointing used Val Metric)"
    elif metric_loss_mode == "train":
        # Add a second y-axis for the metric weight
        ax1b = ax1.twinx()
        ax1b.plot(
            df["epoch"],
            df["current_metric_weight"],
            "g--",
            label="Metric Weight",
            alpha=0.7,
        )
        ax1b.set_ylabel("Metric Loss Weight")
        ax1b.legend(loc="upper right")

    ax1.set_ylabel("Loss")
    ax1.set_title(plot1_title)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_yscale("log")

    # --- Plot 2: Loss Components ---
    ax2.plot(
        df["epoch"],
        df["train_loss_mse"],
        "s--",
        label="Train MSE",
        color="lightblue",
        alpha=0.8,
    )
    ax2.plot(
        df["epoch"], df["val_loss_mse"], "s-", label="Val MSE", color="blue", alpha=0.8
    )
    ax2.plot(
        df["epoch"],
        df["train_loss_metric"],
        "x--",
        label="Train Metric",
        color="lightgreen",
        alpha=0.8,
    )
    ax2.plot(
        df["epoch"],
        df["val_loss_metric"],
        "x-",
        label="Val Metric",
        color="green",
        alpha=0.8,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Component")
    ax2.set_title("Loss Components")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training history plot to: {save_path}")


# --- New function to save metrics ---
def save_metrics_to_file(output_dir, metrics_dict):
    """Saves the key evaluation metrics to a text file."""
    file_path = os.path.join(output_dir, "evaluation_metrics.txt")
    print(f"\nSaving evaluation metrics to {file_path}...")

    try:
        with open(file_path, "w") as f:
            f.write("--- Overall Evaluation Metrics ---\n")
            f.write(
                f"Mean Total Evaluation Loss: {metrics_dict['mean_total_loss']:.6f}\n"
            )
            f.write(f"Mean MSE:                   {metrics_dict['mean_mse']:.6f}\n")
            f.write(f"Mean FSS (1mm, 25km):       {metrics_dict['mean_fss']:.4f}\n")

            sal = metrics_dict["mean_sal"]
            f.write(
                f"Mean SAL (1mm):             S={sal[0]:.4f}, A={sal[1]:.4f}, L={sal[2]:.4f}\n"
            )

            f.write("\n--- Average Gamma R^2 Scores ---\n")
            f.write(f"Average Gamma R^2 - Area:      {metrics_dict['avg_r2_A']:.4f}\n")
            f.write(f"Average Gamma R^2 - Perimeter: {metrics_dict['avg_r2_P']:.4f}\n")
            f.write(f"Average Gamma R^2 - CC:        {metrics_dict['avg_r2_CC']:.4f}\n")

        print("Metrics saved successfully.")
    except IOError as e:
        print(f"Error saving metrics file: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet-SR model.")
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
    BATCH_SIZE = config.get("SR_BATCH_SIZE", 16)
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
    EVAL_MODE = config.get("EVAL_MODE", "none")
    SURROGATE_LOSS_TYPE = config.get("SURROGATE_LOSS_TYPE", "cdf")
    SURROGATE_LOSS_WEIGHT = config.get("SURROGATE_LOSS_WEIGHT", 0.1)
    CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hybrid")
    EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)
    S_ESTIMATION_SAMPLES = config.get("S_ESTIMATION_SAMPLES", 1000)
    DEM_DATA_DIR = config["DEM_DATA_DIR"]
    DEM_STATS = config["DEM_STATS"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model = UNetSR(in_channels=2, out_channels=1).to(device)
    checkpoint_path = os.path.join(args.run_dir, "best_sr_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Error: Checkpoint file not found: '{checkpoint_path}'"
        )
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # --- Load Emulator (if needed for ranking) ---
    emulator_model = None
    if EVAL_MODE == "train":
        if not EMULATOR_CHECKPOINT_PATH:
            raise ValueError("EMULATOR_CHECKPOINT_PATH not set in config.")
        emulator_model = load_emulator(EMULATOR_CHECKPOINT_PATH, device)

    # --- Load DEM Stats ---
    with open(DEM_STATS, "r") as f:
        stats_dict = json.load(f)
    dem_stats = (float(stats_dict["dem_mean"]), float(stats_dict["dem_std"]))

    # --- Load Losses/Metrics ---
    mse_criterion = nn.MSELoss(reduction="none")
    surrogate_criterion = None
    if SURROGATE_LOSS_TYPE == "cdf":
        surrogate_criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(
            device
        )
    elif SURROGATE_LOSS_TYPE == "geometric":
        print("Loading train dataset to compute S_inv for geometric loss...")
        train_dataset_for_s = SRDataset(
            PREPROCESSED_DATA_DIR,
            config["TRAIN_METADATA_FILE"],
            DEM_DATA_DIR,
            dem_stats,
            split="train",
        )
        S_inv_tensors = estimate_s_inv_from_dataset(
            train_dataset_for_s, S_ESTIMATION_SAMPLES, device
        )
        surrogate_criterion = GeometricLossSeparate(S_inv_tensors).to(device)

    # --- Load Test Dataset ---
    test_dataset = SRDataset(
        PREPROCESSED_DATA_DIR, TEST_METADATA_FILE, DEM_DATA_DIR, dem_stats, split="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )
    print(f"Loaded {len(test_dataset)} samples for evaluation.")

    all_preds_phys, all_targets_phys, all_inputs_phys = [], [], []
    all_losses_for_ranking = []

    total_mse = 0.0
    fss_scores = []
    sal_scores = []

    with torch.no_grad():
        for X, Y_true_phys, Y_gamma_phys in tqdm(
            test_loader, desc="Calculating Metrics"
        ):
            X, Y_true_phys, Y_gamma_phys = (
                X.to(device),
                Y_true_phys.to(device),
                Y_gamma_phys.to(device),
            )

            with torch.amp.autocast(device_type="cuda"):
                pred_X_phys = model(X)
                loss_mse_per_sample = mse_criterion(pred_X_phys, Y_true_phys).mean(
                    dim=(1, 2, 3)
                )
                total_mse += loss_mse_per_sample.sum().item()

                loss_surrogate_per_sample = torch.zeros_like(loss_mse_per_sample)
                if EVAL_MODE != "none":
                    pred_gamma_phys_analytic = []
                    pred_X_np = pred_X_phys.cpu().numpy()
                    for i in range(pred_X_np.shape[0]):
                        gamma_matrix = compute_gamma_matrix_for_image(
                            pred_X_np[i, 0], QUANTILE_LEVELS, PIXEL_SIZE_KM
                        )
                        pred_gamma_phys_analytic.append(gamma_matrix)
                    pred_gamma_phys = torch.from_numpy(
                        np.array(pred_gamma_phys_analytic)
                    ).to(device)

                    if SURROGATE_LOSS_TYPE == "cdf":
                        pred_gamma_log = torch.log1p(pred_gamma_phys)
                        true_gamma_log = torch.log1p(Y_gamma_phys)
                        loss_A, loss_P, loss_CC = surrogate_criterion(
                            pred_gamma_log, true_gamma_log
                        )
                        loss_surrogate_per_sample = loss_A + loss_P + loss_CC
                    elif SURROGATE_LOSS_TYPE == "geometric":
                        loss_surrogate_per_sample = surrogate_criterion(
                            pred_gamma_phys, Y_gamma_phys
                        )

                if EVAL_MODE == "validate":
                    all_losses_for_ranking.append(
                        loss_surrogate_per_sample.cpu().numpy()
                    )
                elif EVAL_MODE == "train":
                    total_loss = (
                        (1 - SURROGATE_LOSS_WEIGHT) * loss_mse_per_sample
                        + SURROGATE_LOSS_WEIGHT * loss_surrogate_per_sample
                    )
                    all_losses_for_ranking.append(total_loss.cpu().numpy())
                else:
                    all_losses_for_ranking.append(loss_mse_per_sample.cpu().numpy())

            pred_X_np = pred_X_phys.cpu().numpy()
            Y_true_np = Y_true_phys.cpu().numpy()
            for i in range(pred_X_np.shape[0]):
                pred_sample = pred_X_np[i, 0]
                target_sample = Y_true_np[i, 0]
                fss_scores.append(
                    compute_fss(
                        pred_sample, target_sample, window_size=12, threshold=1.0
                    )
                )
                structure, amplitude, location = compute_sal(
                    pred_sample,
                    target_sample,
                    threshold=1.0,
                    pixel_area_km2=PIXEL_SIZE_KM**2,
                )
                if not np.isnan(structure):
                    sal_scores.append([structure, amplitude, location])

            all_preds_phys.append(pred_X_np)
            all_targets_phys.append(Y_true_np)
            all_inputs_phys.append(X.cpu().numpy())

    # --- Consolidate Results ---
    all_preds_phys = np.concatenate(all_preds_phys, axis=0).squeeze()
    all_targets_phys = np.concatenate(all_targets_phys, axis=0).squeeze()
    all_inputs_phys = np.concatenate(all_inputs_phys, axis=0)
    all_losses_for_ranking = np.concatenate(all_losses_for_ranking, axis=0)

    mean_total_loss = np.mean(all_losses_for_ranking)
    print(f"Generated predictions and losses for {all_preds_phys.shape[0]} samples.")
    print(f"Mean Total Evaluation (Ranking) Loss: {mean_total_loss:.4f}")

    # --- Report Metrics ---
    mean_mse = total_mse / len(test_dataset)
    mean_fss = np.mean(fss_scores)
    mean_sal = np.nanmean(np.array(sal_scores), axis=0)
    print("\n--- Overall Test Metrics ---")
    print(f"Mean MSE:          {mean_mse:.6f}")
    print(f"Mean FSS (1mm, 25km): {mean_fss:.4f}")
    print(
        f"Mean SAL (1mm):    S={mean_sal[0]:.4f}, A={mean_sal[1]:.4f}, L={mean_sal[2]:.4f}"
    )

    # Calculate and print R^2 scores for gamma components
    print("\nCalculating R^2 scores for analytical gamma components...")
    all_pred_gamma, all_true_gamma = [], []
    for i in tqdm(range(all_preds_phys.shape[0]), desc="Calculating Gamma R^2"):
        all_pred_gamma.append(
            compute_gamma_matrix_for_image(
                all_preds_phys[i], QUANTILE_LEVELS, PIXEL_SIZE_KM
            )
        )
        all_true_gamma.append(
            compute_gamma_matrix_for_image(
                all_targets_phys[i], QUANTILE_LEVELS, PIXEL_SIZE_KM
            )
        )
    all_pred_gamma = np.array(all_pred_gamma)
    all_true_gamma = np.array(all_true_gamma)
    n_samples = all_pred_gamma.shape[0]
    n_features = 3 * N_QUANTILES
    preds_flat = all_pred_gamma.reshape(n_samples, n_features)
    targets_flat = all_true_gamma.reshape(n_samples, n_features)
    mask = np.isfinite(targets_flat).all(axis=1) & np.isfinite(preds_flat).all(axis=1)
    r2_scores_raw = r2_score(
        targets_flat[mask], preds_flat[mask], multioutput="raw_values"
    )
    r2_scores_matrix = r2_scores_raw.reshape(3, N_QUANTILES)
    avg_r2_A = np.mean(r2_scores_matrix[0, :])
    avg_r2_P = np.mean(r2_scores_matrix[1, :])
    avg_r2_CC = np.mean(r2_scores_matrix[2, :])
    print(f"Average Gamma R^2 - Area:      {avg_r2_A:.4f}")
    print(f"Average Gamma R^2 - Perimeter: {avg_r2_P:.4f}")
    print(f"Average Gamma R^2 - CC:        {avg_r2_CC:.4f}")
    r2_save_path = os.path.join(args.run_dir, "gamma_r2_scores.npz")
    np.savez_compressed(
        r2_save_path, r2_matrix=r2_scores_matrix, quantiles=QUANTILE_LEVELS
    )
    print(f"Detailed Gamma R^2 scores saved to: {r2_save_path}")

    # --- Collect metrics and save to file ---
    metrics_to_save = {
        "mean_total_loss": mean_total_loss,
        "mean_mse": mean_mse,
        "mean_fss": mean_fss,
        "mean_sal": mean_sal,
        "avg_r2_A": avg_r2_A,
        "avg_r2_P": avg_r2_P,
        "avg_r2_CC": avg_r2_CC,
    }
    save_metrics_to_file(args.run_dir, metrics_to_save)

    # --- Plotting ---
    plot_best_worst_samples(
        predictions_phys=all_preds_phys,
        targets_phys=all_targets_phys,
        inputs_phys=all_inputs_phys,
        losses=all_losses_for_ranking,
        output_dir=args.run_dir,
        n_samples=15,
        dem_stats=dem_stats,
    )

    log_file_path = os.path.join(args.run_dir, "sr_training_log.csv")
    plot_training_log(log_file_path, args.run_dir, config)

    print("\n✅ Evaluation script finished.")
