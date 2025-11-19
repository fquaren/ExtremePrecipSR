import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

# --- 1. Single Sample Plotting ---


def _plot_single_sr_comparison(
    sample_row: pd.Series,
    title: str,
    sub_folder: str,
    output_dir: str,
    dem_stats: tuple,
):
    """
    Plots a single sample's SR comparison, passed as a DataFrame row.

    'sample_row' contains all metrics and object columns:
    'pred_image', 'target_image', 'input_stack', 'total_loss', 'mse', 'fss',
    'sal_S', 'sal_A', 'sal_L', 'r2_A', 'r2_P', 'r2_CC', 'mean_precip'.
    """

    # --- Extract data from the row ---
    pred_phys = sample_row["pred_image"]
    target_phys = sample_row["target_image"]
    input_stack = sample_row["input_stack"]
    interp_precip = input_stack[0]
    dem_normalized = input_stack[1]
    sample_idx = sample_row.name

    # Un-normalize DEM
    dem_mean, dem_std = dem_stats
    dem_unnormalized = (dem_normalized * (dem_std + 1e-8)) + dem_mean

    # Extract metrics
    loss = sample_row["total_loss"]
    mse = sample_row["mse"]
    fss = sample_row["fss"]
    sal_S, sal_A, sal_L = sample_row["sal_S"], sample_row["sal_A"], sample_row["sal_L"]
    r2_A, r2_P, r2_CC = sample_row["r2_A"], sample_row["r2_P"], sample_row["r2_CC"]

    # --- Plotting Setup ---
    vmin_precip = 0
    vmax_precip = np.max(
        [np.max(interp_precip), np.max(pred_phys), np.max(target_phys)]
    )
    if vmax_precip == 0:
        vmax_precip = 1.0

    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 4, wspace=0.3)

    # Plot 1: Input Interp. Precip
    ax_img1 = fig.add_subplot(gs[0, 0])
    im1 = ax_img1.imshow(
        interp_precip, cmap="Blues", origin="lower", vmin=vmin_precip, vmax=vmax_precip
    )
    ax_img1.set_title(f"Input: Interp. Precip (Mean: {np.mean(interp_precip):.2f})")
    fig.colorbar(im1, ax=ax_img1, shrink=0.7, label="Precipitation (mm/hr)")

    # Plot 2: Input DEM
    ax_img2 = fig.add_subplot(gs[0, 1])
    im2 = ax_img2.imshow(dem_unnormalized, cmap="terrain", origin="lower")
    ax_img2.set_title("Input: DEM (Unnormalized)")
    fig.colorbar(im2, ax=ax_img2, shrink=0.7, label="Elevation (m)")

    # Plot 3: Prediction
    ax_img3 = fig.add_subplot(gs[0, 2])
    im3 = ax_img3.imshow(
        pred_phys, cmap="Blues", origin="lower", vmin=vmin_precip, vmax=vmax_precip
    )
    ax_img3.set_title(f"Prediction (Mean: {np.mean(pred_phys):.2f})")
    fig.colorbar(im3, ax=ax_img3, shrink=0.7, label="Precipitation (mm/hr)")

    # Plot 4: Target
    ax_img4 = fig.add_subplot(gs[0, 3])
    im4 = ax_img4.imshow(
        target_phys, cmap="Blues", origin="lower", vmin=vmin_precip, vmax=vmax_precip
    )
    ax_img4.set_title(f"Target (Mean: {np.mean(target_phys):.2f})")
    fig.colorbar(im4, ax=ax_img4, shrink=0.7, label="Precipitation (mm/hr)")

    # --- Updated Title ---
    metrics_str1 = f"Total Loss: {loss:.4f} | MSE: {mse:.4f} | FSS: {fss:.3f}"
    metrics_str2 = f"SAL (S/A/L): {sal_S:.3f} / {sal_A:.3f} / {sal_L:.3f} | Gamma R² (A/P/CC): {r2_A:.3f} / {r2_P:.3f} / {r2_CC:.3f}"

    fig.suptitle(
        f"{title} | Sample {sample_idx}\n{metrics_str1}\n{metrics_str2}",
        fontsize=14,
        y=1.12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Saving
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", sub_folder)
    os.makedirs(plot_save_dir, exist_ok=True)
    safe_title = title.replace(" ", "_").replace("#", "").lower()
    save_path = os.path.join(plot_save_dir, f"{safe_title}_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_comparisons(
    metrics_df: pd.DataFrame,
    output_dir: str,
    dem_stats: tuple,
    n_samples: int = 10,
):
    """Plots best, worst, and average-loss samples, grouped by precipitation."""
    print("\nGenerating plots for best, worst, and average samples (per group)...")

    grouped = metrics_df.groupby("precip_group")
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]

    for group_name in group_order:
        group_df = grouped.get_group(group_name)
        print(f"\n--- Processing Group: {group_name} (N={len(group_df)}) ---")

        if len(group_df) == 0:
            continue
        current_n = min(n_samples, len(group_df))

        # Best
        best_samples = group_df.nsmallest(current_n, "total_loss")
        print(f"Plotting {len(best_samples)} best samples...")
        for rank, (idx, row) in enumerate(best_samples.iterrows()):
            _plot_single_sr_comparison(
                row, f"Best Sample #{rank+1}", group_name, output_dir, dem_stats
            )

        # Worst
        worst_samples = group_df.nlargest(current_n, "total_loss")
        print(f"Plotting {len(worst_samples)} worst samples...")
        for rank, (idx, row) in enumerate(worst_samples.iloc[::-1].iterrows()):
            _plot_single_sr_comparison(
                row, f"Worst Sample #{rank+1}", group_name, output_dir, dem_stats
            )

        # Average-Loss
        mean_group_loss = group_df["total_loss"].mean()
        if np.isnan(mean_group_loss):
            continue

        group_df_copy = group_df.copy()
        group_df_copy["dist_to_mean"] = (
            group_df_copy["total_loss"] - mean_group_loss
        ).abs()
        avg_samples = group_df_copy.nsmallest(current_n, "dist_to_mean")

        print(f"Plotting {len(avg_samples)} average-loss samples...")
        for rank, (idx, row) in enumerate(avg_samples.iterrows()):
            _plot_single_sr_comparison(
                row, f"Average Loss Sample #{rank+1}", group_name, output_dir, dem_stats
            )


# --- 2. Distribution Plots ---


def plot_metric_distributions(metrics_df: pd.DataFrame, output_dir: str):
    """Generates box plots for the distributions of key evaluation metrics."""
    print("\nGenerating metric distribution box plots...")

    # Define metrics to plot
    metric_cols = {
        "total_loss": "Total Loss",
        "mse": "MSE",
        "fss": "FSS (1mm, 25km)",
        "sal_S": "SAL - Structure",
        "sal_A": "SAL - Amplitude",
        "sal_L": "SAL - Location",
        "r2_A": "Gamma R² Area",
        "r2_P": "Gamma R² Perim.",
        "r2_CC": "Gamma R² CC",
    }

    n_metrics = len(metric_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    medianprops = dict(color="red", linewidth=1.5)

    for i, (col, title) in enumerate(metric_cols.items()):
        ax = axes[i]
        valid_data = metrics_df[col].dropna()

        if valid_data.empty:
            ax.text(
                0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title)
            continue

        ax.boxplot(
            valid_data,
            vert=True,
            patch_artist=True,
            labels=[title],
            medianprops=medianprops,
        )
        mean_val = valid_data.mean()
        ax.set_title(f"{title}\nMean: {mean_val:.4f}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.6)

        if col in ["total_loss", "mse"] and mean_val > 1e-6:
            ax.set_yscale("log")
        elif "r2" in col or "sal" in col:
            if valid_data.min() < -1.1:
                ax.set_ylim(
                    bottom=max(-10, valid_data.min() * 1.1)
                )  # Cap extreme negatives
            else:
                ax.set_ylim(bottom=-1.1)
            if valid_data.max() > 1.1:
                ax.set_ylim(
                    top=min(10, valid_data.max() * 1.1)
                )  # Cap extreme positives
            else:
                ax.set_ylim(top=1.1)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        elif "fss" in col:
            ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Evaluation Metric Distributions (Test Set)", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(output_dir, "evaluation_metric_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric distribution plots to: {save_path}")


# --- 3. Grouped Mean/Std Plots (for Analytical Gammas) ---


def plot_gamma_mean_std_by_quantile(
    metrics_df: pd.DataFrame,
    group_metrics: pd.DataFrame,  # Pass the pre-computed group_metrics summary
    quantiles: list,
    output_dir: str,
):
    """
    Plots the mean and std dev of *analytical* Gamma functions, grouped by precipitation.
    Uses metrics_df to access the raw data for grouping.
    """
    print(
        "\nGenerating plots for mean/std analytical gamma performance by precip group..."
    )

    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    plot_save_dir = os.path.join(
        output_dir, "evaluation_plots", "mean_std_gamma_groups"
    )
    os.makedirs(plot_save_dir, exist_ok=True)

    grouped = metrics_df.groupby("precip_group")
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]
    group_order.append("All")

    for group_name in group_order:
        print(f"--- Plotting Group: {group_name} ---")

        if group_name == "All":
            group_df = metrics_df
            metrics = group_metrics.loc["All"]
            n_samples = len(group_df)
        else:
            group_df = grouped.get_group(group_name)
            metrics = group_metrics.loc[group_name]
            n_samples = int(metrics["n_samples"])

        if n_samples == 0:
            print("Skipping group, no samples.")
            continue

        # --- Stack arrays from the DataFrame rows ---
        group_preds = np.stack(group_df["pred_gamma"].values)
        group_targets = np.stack(group_df["target_gamma"].values)

        mean_preds = np.nanmean(group_preds, axis=0)
        std_preds = np.nanstd(group_preds, axis=0)
        mean_targets = np.nanmean(group_targets, axis=0)
        std_targets = np.nanstd(group_targets, axis=0)

        metric_str = f"Mean R² (A/P/CC): {metrics['r2_A']:.3f} / {metrics['r2_P']:.3f} / {metrics['r2_CC']:.3f}"

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
        for j in range(3):  # Loop over A, P, CC
            ax = axes[j]
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
            if j < 2 and np.nanmax(mean_targets[j]) > 100:
                ax.set_yscale("log")

        fig.suptitle(
            f"Mean Analytical Gamma Function Comparison (±1 Std. Dev.)\nGroup: {group_name} (N={n_samples})\n{metric_str}",
            fontsize=16,
            y=1.08,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = os.path.join(
            plot_save_dir, f"mean_std_gamma_group_{group_name.lower()}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved mean/std gamma plots to: {plot_save_dir}")


# --- 4. Training Log Plot ---


def plot_training_log(log_path, output_dir, config):
    """
    Plots the training history from the sr_training_log.csv file.
    (This function is moved from the original script)
    """
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

    if "current_metric_weight" not in df.columns and metric_loss_mode == "train":
        print(
            "Warning: 'current_metric_weight' missing from log. Skipping weight plot."
        )
        plot_weight = False
    elif metric_loss_mode == "train":
        plot_weight = True
    else:
        plot_weight = False

    if not all(col in df.columns for col in required_cols):
        print("Warning: Log file columns mismatch. Skipping training history plot.")
        print(f"Missing: {[c for c in required_cols if c not in df.columns]}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"SR Training History (Mode: {metric_loss_mode})", fontsize=16)

    # Plot 1: Total Loss
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

    if plot_weight:
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

    # Plot 2: Loss Components
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
