import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Scientific plotting style
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("ggplot")


def load_rapsd_stats(path):
    """
    Loads RAPSD results and computes statistics.
    Returns: freqs, mean, p05 (5th percentile), p95 (95th percentile)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")

    data = np.load(path)
    freqs = data["freqs"]
    rapsd = data["rapsd_gen"]

    # Check for NaNs or Infs
    if not np.all(np.isfinite(rapsd)):
        print(f"Warning: Non-finite values found in {path}. Replacing with 0.")
        rapsd = np.nan_to_num(rapsd)

    # Compute statistics across the batch dimension (axis 0)
    mean_rapsd = np.mean(rapsd, axis=0)
    p05_rapsd = np.percentile(rapsd, 5, axis=0)
    p95_rapsd = np.percentile(rapsd, 95, axis=0)

    return freqs, mean_rapsd, p05_rapsd, p95_rapsd, data["rapsd_gt"]


def setup_wavelength_axis(ax):
    """Adds a secondary x-axis on top showing Wavelength (km)."""

    def freq2wave(x):
        return 1.0 / (x + 1e-10)  # Avoid divide by zero

    def wave2freq(x):
        return 1.0 / (x + 1e-10)

    secax = ax.secondary_xaxis("top", functions=(freq2wave, wave2freq))
    secax.set_xlabel("Wavelength [km]")
    # Set specific ticks for readability
    ticks = [200, 100, 50, 20, 10, 4]
    secax.set_xticks(ticks)
    secax.set_xticklabels([str(t) for t in ticks])


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAPSD spectra of three models (Side-by-Side)."
    )
    parser.add_argument(
        "--baseline", type=str, required=True, help="Path to Baseline .npz"
    )
    parser.add_argument(
        "--isometric", type=str, required=True, help="Path to Isometric .npz"
    )
    parser.add_argument(
        "--constrained", type=str, required=True, help="Path to Constrained .npz"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rapsd_comparison_horizontal.png",
        help="Output filename",
    )
    args = parser.parse_args()

    # Dictionary to manage models and colors (IBM)
    models = {
        "CNN (Unconstr.)": {
            "path": args.baseline,
            "color": "#648fff",
            "style": "--",
        },  # Blue
        "Lip-CNN (Unconstr.)": {
            "path": args.isometric,
            "color": "#ffb000",
            "style": "-.",
        },  # Yellow
        "Lip-CNN (Constr.)": {
            "path": args.constrained,
            "color": "#dc2680",
            "style": ":",
        },  # Magenta
    }

    print("Loading data...")
    try:
        tmp_data = np.load(args.baseline)
        rapsd_gt_all = tmp_data["rapsd_gt"]
        freqs = tmp_data["freqs"]

        # Robust Mean calculation (ignoring NaNs just in case)
        gt_mean = np.nanmean(rapsd_gt_all, axis=0)

        # Sanity check
        if np.isnan(gt_mean).any():
            print(
                "Warning: Ground Truth mean still contains NaNs. Please run repair script."
            )

    except Exception as e:
        print(f"Error loading GT from baseline file: {e}")
        return

    # --- Initialize Plot: 1 Row, 2 Columns ---
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9, 3),  # Wide layout
        sharex=True,  # Zooming one zooms the other
        constrained_layout=True,  # Auto-adjust spacing
    )

    # ==========================================
    # LEFT PLOT: Absolute Power Spectra
    # ==========================================

    # 1. Plot Ground Truth (High zorder to stay on top)
    ax1.loglog(freqs, gt_mean, "k--", linewidth=2.5, label="Ground Truth", zorder=10)

    # 2. Iterate and plot models
    for name, config in models.items():
        try:
            _, mean, p05, p95, _ = load_rapsd_stats(config["path"])

            # Plot Mean
            ax1.loglog(
                freqs,
                mean,
                color=config["color"],
                linestyle=config["style"],
                linewidth=2,
                label=name,
                zorder=5,
            )

            # Plot Confidence Interval
            ax1.fill_between(
                freqs, p05, p95, color=config["color"], alpha=0.15, zorder=1
            )

            # Save mean for Ratio plot
            config["mean_data"] = mean

        except Exception as e:
            print(f"Failed to process {name}: {e}")

    ax1.set_ylabel(r"Power Spectral Density $[mm^2 h^{-2} km]$")
    ax1.set_xlabel(r"Frequency $[km^{-1}]$")
    # ax1.set_title("Absolute RAPSD")
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend(fontsize=9, loc="lower left")

    setup_wavelength_axis(ax1)

    # ==========================================
    # RIGHT PLOT: Spectral Ratio (Model / GT)
    # ==========================================

    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

    for name, config in models.items():
        if "mean_data" in config:
            # Ratio calculation
            ratio = config["mean_data"] / (gt_mean + 1e-8)

            ax2.semilogx(
                freqs,
                ratio,
                color=config["color"],
                linestyle=config["style"],
                linewidth=2,
                label=name,
            )

    ax2.set_xlabel(r"Frequency $[km^{-1}]$")
    ax2.set_ylabel("Ratio (Model / GT)")
    ax2.set_ylim(0.1, 2.0)
    ax2.set_yscale("log")
    # ax2.set_yticks([0.2, 0.5, 0.8, 1.0, 1.2, 1.5])
    # ax2.set_title("Spectral Ratio")
    ax2.grid(True, which="both", ls="-", alpha=0.4)

    # Annotate "Effective Resolution" zone
    ax2.text(
        0.95,
        0.05,
        "Under-resolved\n(Smoothing)",
        transform=ax2.transAxes,
        fontsize=9,
        alpha=0.6,
        ha="right",
        va="bottom",
    )

    setup_wavelength_axis(ax2)

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()
