import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import seaborn as sns
import copy

# --- 1. Single Sample Plotting ---


def _plot_single_gamma_comparison(
    sample_row: pd.Series,
    quantiles: np.ndarray,
    title: str,
    sub_folder: str,
    output_dir: str,
):
    """
    Plots a single sample's comparison, passed as a DataFrame row.

    'sample_row' is a row from metrics_df, containing columns:
    'pred_gamma', 'target_gamma', 'target_image', 'total_loss',
    'geom_loss', 'r2_A', 'r2_P', 'r2_CC', 'mean_precip',
    and its name is the 'sample_idx'.
    """

    # --- Extract data from the row ---
    pred_gamma = sample_row["pred_gamma"]
    target_gamma = sample_row["target_gamma"]
    target_image = sample_row["target_image"]
    mean_precip = sample_row["mean_precip"]
    sample_idx = sample_row.name  # Get the original sample index

    # Extract metrics
    loss = sample_row["total_loss"]
    geom_loss = sample_row["geom_loss"]
    r2_A = sample_row["r2_A"]
    r2_P = sample_row["r2_P"]
    r2_CC = sample_row["r2_CC"]

    # --- Create Plot ---
    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, wspace=0.4)

    # Plot 1: Target Image
    ax_img = fig.add_subplot(gs[0, 0])
    # 1. Create a copy of the colormap to avoid global side effects
    cmap = copy.copy(plt.get_cmap("Blues"))
    # 2. Set the color for values below vmin (the 'under' values)
    # RGBA tuple: Red=1, Green=1, Blue=0 (Yellow), Alpha=0.5
    cmap.set_under(color=(1, 1, 0, 0.5))
    # 3. Plot strictly positive vmin is required to trigger the 'under' color for exact zeros.
    # 1e-5 is usually safe for precipitation mm/hr data.
    im = ax_img.imshow(target_image, cmap=cmap, origin="lower", vmin=1e-5)
    ax_img.set_title(f"Target Image (Mean: {mean_precip:.2f})")
    fig.colorbar(im, ax=ax_img, shrink=0.7, label="Precipitation (mm/hr)")

    # Plot 2-4: Gamma Functions
    for j in range(3):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.plot(quantiles, target_gamma[j], "o-", label="Target", color="royalblue")
        ax.plot(quantiles, pred_gamma[j], "x--", label="Prediction", color="salmon")
        ax.set_title(gamma_types[j])
        ax.set_xlabel("Precip. Threshold (mm/hr)")
        ax.grid(True, linestyle="--", alpha=0.6)
        if j == 0:
            ax.legend()

    # --- Updated Title ---
    # Create a second line for all the metrics
    metrics_str = (
        f"Total Loss: {loss:.4f} | Geom. Loss: {geom_loss:.4f} | "
        f"R² (A/P/CC): {r2_A:.3f} / {r2_P:.3f} / {r2_CC:.3f}"
    )

    fig.suptitle(
        f"{title} | Sample {sample_idx}\n{metrics_str}",
        fontsize=15,  # Slightly smaller font to fit two lines
        y=1.08,  # Increase y-position to make space for the new line
    )
    # Adjust tight_layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # --- Saving ---
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", sub_folder)
    os.makedirs(plot_save_dir, exist_ok=True)
    # Sanitize title for filename
    safe_title = title.replace(" ", "_").replace("#", "").lower()
    save_path = os.path.join(
        plot_save_dir,
        f"{safe_title}_sample_{sample_idx}.png",
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_comparisons(
    metrics_df,
    quantiles,
    output_dir,
    n_samples=10,
):
    """
    Plots best, worst, and average-loss samples, grouped by precipitation.
    This version uses the pandas DataFrame for clean data selection.
    """
    print("\nGenerating plots for best, worst, and average samples (per group)...")

    # Group by precipitation
    grouped = metrics_df.groupby("precip_group")

    # Re-order groups for logical plot output
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]

    for group_name in group_order:
        group_df = grouped.get_group(group_name)
        print(f"\n--- Processing Group: {group_name} (N={len(group_df)}) ---")

        if len(group_df) == 0:
            print("Skipping group, no samples.")
            continue

        current_n = min(n_samples, len(group_df))

        # --- Find Best (nsmallest) ---
        best_samples = group_df.nsmallest(current_n, "total_loss")
        print(f"Plotting {len(best_samples)} best samples...")
        for rank, (idx, row) in enumerate(best_samples.iterrows()):
            _plot_single_gamma_comparison(
                row, quantiles, f"Best Sample #{rank+1}", group_name, output_dir
            )

        # --- Find Worst (nlargest) ---
        worst_samples = group_df.nlargest(current_n, "total_loss")
        print(f"Plotting {len(worst_samples)} worst samples...")
        # Rank them from worst to "less worst"
        for rank, (idx, row) in enumerate(worst_samples.iloc[::-1].iterrows()):
            _plot_single_gamma_comparison(
                row, quantiles, f"Worst Sample #{rank+1}", group_name, output_dir
            )

        # --- Find Average-Loss ---
        mean_group_loss = group_df["total_loss"].mean()
        if np.isnan(mean_group_loss):
            print("Skipping average loss plots, mean group loss is NaN.")
            continue

        # Find closest to mean
        group_df_copy = group_df.copy()  # Avoid SettingWithCopyWarning
        group_df_copy["dist_to_mean"] = (
            group_df_copy["total_loss"] - mean_group_loss
        ).abs()
        avg_samples = group_df_copy.nsmallest(current_n, "dist_to_mean")

        print(f"Plotting {len(avg_samples)} average-loss samples...")
        for rank, (idx, row) in enumerate(avg_samples.iterrows()):
            _plot_single_gamma_comparison(
                row, quantiles, f"Average Loss Sample #{rank+1}", group_name, output_dir
            )


# --- 2. Distribution Plots ---


def plot_metric_distributions(
    metrics_df,
    output_dir,
):
    """
    Generates box plots for the distributions of key evaluation metrics,
    using the metrics_df DataFrame.
    """
    print("\nGenerating metric distribution box plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Evaluation Metric Distributions (Test Set)", fontsize=16, y=1.02)

    # Filter NaNs from the DataFrame columns
    valid_total_losses = metrics_df["total_loss"].dropna()
    valid_geom_losses = metrics_df["geom_loss"].dropna()
    valid_r2_A = metrics_df["r2_A"].dropna()
    valid_r2_P = metrics_df["r2_P"].dropna()
    valid_r2_CC = metrics_df["r2_CC"].dropna()

    medianprops = dict(color="red", linewidth=1.5)

    # Plot 1: Total Loss
    ax1.boxplot(
        valid_total_losses,
        vert=True,
        patch_artist=True,
        labels=["Total Loss"],
        medianprops=medianprops,
    )
    ax1.set_title(f"Total Loss (Config) \nMean: {valid_total_losses.mean():.4f}")
    ax1.set_ylabel("Loss Value")
    ax1.grid(True, linestyle="--", alpha=0.6)
    if not valid_total_losses.empty:
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
        f"Geometric (Mahalanobis) Loss \nMean: {valid_geom_losses.mean():.4f}"
    )
    ax2.set_ylabel("Loss Value")
    ax2.grid(True, linestyle="--", alpha=0.6)
    if not valid_geom_losses.empty:
        ax2.set_yscale("log")

    # Plot 3: R^2 Score (per component)
    data_to_plot = [valid_r2_A, valid_r2_P, valid_r2_CC]
    labels = [
        f"Area (Mean: {valid_r2_A.mean():.3f})",
        f"Perim. (Mean: {valid_r2_P.mean():.3f})",
        f"CC (Mean: {valid_r2_CC.mean():.3f})",
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


# --- 3. Grouped Mean/Std Plots ---


def plot_gamma_mean_std_by_quantile(
    metrics_df,
    group_metrics,  # Pass the pre-computed group_metrics summary
    quantiles,
    output_dir,
):
    """
    Plots the mean and std dev of Gamma functions, grouped by precipitation.
    Uses metrics_df to access the raw data for grouping.
    """
    print("\nGenerating plots for mean/std performance by precip group...")

    gamma_types = ["Area (km²)", "Perimeter (km)", "CCs"]
    plot_save_dir = os.path.join(output_dir, "evaluation_plots", "mean_std_groups")
    os.makedirs(plot_save_dir, exist_ok=True)

    grouped = metrics_df.groupby("precip_group")

    # Re-order groups + 'All'
    group_order = [g for g in ["Zero", "Low", "Mid", "High"] if g in grouped.groups]
    group_order.append("All")  # Add 'All' group

    for group_name in group_order:
        print(f"--- Plotting Group: {group_name} ---")

        if group_name == "All":
            group_df = metrics_df
            if group_name not in group_metrics.index:
                # Need to handle if 'All' wasn't pre-computed, but it should be
                print("Warning: 'All' group metrics not found in summary.")
                metrics = {k: np.nan for k in group_metrics.columns}
            else:
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

        # --- Calculate statistics for plotting (mean/std) ---
        mean_preds = np.nanmean(group_preds, axis=0)
        std_preds = np.nanstd(group_preds, axis=0)
        mean_targets = np.nanmean(group_targets, axis=0)
        std_targets = np.nanstd(group_targets, axis=0)

        # --- Use pre-computed metrics for title ---
        metric_str = (
            f"Mean Total Loss: {metrics['total_loss']:.4f} | "
            f"Mean R² (A/P/CC): {metrics['r2_A']:.3f} / {metrics['r2_P']:.3f} / {metrics['r2_CC']:.3f}"
        )

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
            f"Mean Gamma Function Comparison (±1 Std. Dev.)\n"
            f"Group: {group_name} (N={n_samples})\n"
            f"{metric_str}",
            fontsize=16,
            y=1.08,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = os.path.join(
            plot_save_dir,
            f"mean_std_gamma_group_{group_name.lower()}.png",
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved mean/std plots to: {plot_save_dir}")


def plot_per_feature_matrices(per_feature_metrics, output_dir):
    """
    Generates heatmaps for R2, MSE, and Variance matrices.

    Args:
        per_feature_metrics (dict): Dictionary containing 'r2_matrix',
                                    'mse_matrix', 'var_matrix' DataFrames.
        output_dir (str): Path to save the plots.
    """
    print("\nGenerating per-feature matrix heatmaps...")

    # Extract DataFrames
    r2_df = per_feature_metrics["r2_matrix"]
    mse_df = per_feature_metrics["mse_matrix"]
    var_df = per_feature_metrics["var_matrix"]

    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Per-Feature Metric Matrices (Component vs Quantile)", fontsize=16, y=1.05
    )

    # 1. R2 Heatmap
    # We use a diverging colormap centered generally around good values.
    # vmin=0 to visually distinguish negative/low R2 from good R2.
    sns.heatmap(
        r2_df,
        ax=axes[0],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "R² Score"},
    )
    axes[0].set_title("R² Score (Higher is Better)")
    axes[0].set_xlabel("Quantile (mm/hr)")

    # 2. MSE Heatmap
    # diverse magnitudes across rows, so we rely on annotations for exact values
    sns.heatmap(
        mse_df,
        ax=axes[1],
        cmap="viridis",
        annot=True,
        fmt=".2e",
        cbar_kws={"label": "Mean Squared Error"},
    )
    axes[1].set_title("MSE (Lower is Better)")
    axes[1].set_xlabel("Quantile (mm/hr)")

    # 3. Variance Heatmap
    sns.heatmap(
        var_df,
        ax=axes[2],
        cmap="magma",
        annot=True,
        fmt=".2e",
        cbar_kws={"label": "Target Variance"},
    )
    axes[2].set_title("Target Variance (Data Spread)")
    axes[2].set_xlabel("Quantile (mm/hr)")

    plt.tight_layout()

    save_path = os.path.join(output_dir, "evaluation_matrices.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved matrix heatmaps to: {save_path}")


# --- 4. Training Log Plot ---


def plot_training_log(log_path, output_dir):
    """
    Plots the training history from the training_log.csv file.
    (This function is unchanged from your original script as it
    does not depend on the new evaluation metrics_df.)
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

    # Check for main loss columns, handle if missing
    if "train_loss_main" not in df.columns:
        if all(
            c in df.columns for c in ["train_loss_A", "train_loss_P", "train_loss_CC"]
        ):
            df["train_loss_main"] = (
                df["train_loss_A"] + df["train_loss_P"] + df["train_loss_CC"]
            )
    if "val_loss_main" not in df.columns:
        if all(c in df.columns for c in ["val_loss_A", "val_loss_P", "val_loss_CC"]):
            df["val_loss_main"] = (
                df["val_loss_A"] + df["val_loss_P"] + df["val_loss_CC"]
            )

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
    ]

    # Check for optional penalty columns
    penalty_cols = [
        "train_penalty_zero",
        "train_penalty_mono",
        "train_penalty_plaus",
        "train_penalty_bound",
        "val_penalty_zero",
        "val_penalty_mono",
        "val_penalty_plaus",
        "val_penalty_bound",
    ]
    found_penalty_cols = [c for c in penalty_cols if c in df.columns]

    # Check if all *required* columns are present
    if not all(col in df.columns for col in required_cols):
        print("Warning: Log file columns mismatch. Skipping training history plot.")
        print(f"Missing: {[c for c in required_cols if c not in df.columns]}")
        return

    # Determine number of subplots
    n_subplots = 3 if found_penalty_cols else 2
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 6 * n_subplots), sharex=True)

    # --- Plot 1: Total & Main Loss ---
    ax1 = axes[0]
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

    # --- Plot 2: Main Loss Components ---
    ax2 = axes[1]
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

    # --- Plot 3: Penalty Terms (if they exist) ---
    if n_subplots == 3:
        ax3 = axes[2]

        # Plotting only the penalty columns that were found
        penalty_plot_map = {
            "train_penalty_zero": ("p:", "black", 0.6, "Train Zero Pen."),
            "val_penalty_zero": ("p-", "black", 1.0, "Val Zero Pen."),
            "train_penalty_mono": ("s:", "cyan", 0.6, "Train Mono Pen."),
            "val_penalty_mono": ("s-", "cyan", 1.0, "Val Mono Pen."),
            "train_penalty_plaus": ("x:", "lime", 0.6, "Train Plaus Pen."),
            "val_penalty_plaus": ("x-", "lime", 1.0, "Val Plaus Pen."),
            "train_penalty_bound": ("d:", "magenta", 0.6, "Train Bound Pen."),
            "val_penalty_bound": ("d-", "magenta", 1.0, "Val Bound Pen."),
        }

        for col_name, (fmt, color, alpha, label) in penalty_plot_map.items():
            if col_name in df.columns:
                ax3.plot(
                    df["epoch"],
                    df[col_name],
                    fmt,
                    label=label,
                    color=color,
                    alpha=alpha,
                )

        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Penalty Value")
        ax3.set_title("Soft Penalty Terms")
        ax3.legend(ncol=2)  # Adjusted to ncol=2 for fewer columns
        ax3.grid(True, linestyle="--", alpha=0.6)

        # Set y-scale to log only if there are values > 0
        if (df[found_penalty_cols] > 0).any().any():
            ax3.set_yscale("log")

    # Set X-label on the last axis
    axes[-1].set_xlabel("Epoch")

    fig.suptitle("Training History", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training history plot to: {save_path}")
