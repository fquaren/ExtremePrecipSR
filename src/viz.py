import yaml
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import warnings

# Suppress potential warnings from fitting procedures on some data slices
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PREPROCESSED_DATA_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/extremes/OPERA/patches/precip"
# Get percentage from config, default to 100 if not specified
ANALYSIS_DATA_PERCENTAGE = config.get("ANALYSIS_DATA_PERCENTAGE", 100)

# Create a unique output directory based on the percentage used
ANALYSIS_OUTPUT_DIR = os.path.join(
    PREPROCESSED_DATA_DIR, f"analysis_plots_fit_{ANALYSIS_DATA_PERCENTAGE}pct"
)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)


# Custom Q-Q Plot for Discrete Distributions
def plot_discrete_qq(data, dist, sparams, ax, title="", xlabel="", ylabel=""):
    """
    Generates a Q-Q plot for discrete data against a specified discrete distribution.
    """
    data_sorted = np.sort(data)
    n = len(data_sorted)
    empirical_quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = dist.ppf(empirical_quantiles, *sparams)

    ax.plot(
        theoretical_quantiles,
        data_sorted,
        "o",
        color="darkblue",
        markersize=4,
        label="Data Quantiles",
    )

    min_val = min(theoretical_quantiles.min(), data_sorted.min())
    max_val = max(theoretical_quantiles.max(), data_sorted.max())
    padding = (max_val - min_val) * 0.05
    line_min, line_max = min_val - padding, max_val + padding
    ax.plot(
        [line_min, line_max], [line_min, line_max], "r--", lw=2, label="Reference Line"
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))


def analyze_and_plot_distributions(data_split):
    """
    Loads gamma targets, filters, subsamples, fits distributions, and plots.
    """
    print(f"--- Analyzing distributions for data split: {data_split} ---")
    print(f"--- Using {ANALYSIS_DATA_PERCENTAGE}% of the data ---")

    input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)
    gamma_path = os.path.join(input_dir, "gamma_targets.npz")
    plot_output_dir = os.path.join(ANALYSIS_OUTPUT_DIR, data_split)
    os.makedirs(plot_output_dir, exist_ok=True)

    if not os.path.exists(gamma_path):
        print(f"Gamma targets file not found at {gamma_path}. Skipping.")
        return

    print(f"Loading gamma targets from {gamma_path}...")
    gamma_data = np.load(gamma_path)["data"]

    initial_sample_count = gamma_data.shape[0]
    samples_to_keep_mask = np.sum(gamma_data, axis=(1, 2)) > 0
    gamma_data_filtered = gamma_data[samples_to_keep_mask]
    filtered_sample_count = gamma_data_filtered.shape[0]

    print(f"Original samples: {initial_sample_count}")
    print(
        f"Removed {initial_sample_count - filtered_sample_count} all-zero samples. {filtered_sample_count} remain."
    )

    # --- MODIFICATION: Subsample the dataset based on the percentage ---
    if ANALYSIS_DATA_PERCENTAGE < 100:
        # Use a fixed seed for reproducible random sampling
        rng = np.random.default_rng(42)
        subset_size = int(filtered_sample_count * (ANALYSIS_DATA_PERCENTAGE / 100.0))

        print(
            f"Subsampling to {ANALYSIS_DATA_PERCENTAGE}% of data -> {subset_size} samples."
        )

        subset_indices = rng.choice(
            filtered_sample_count, size=subset_size, replace=False
        )
        gamma_data_for_analysis = gamma_data_filtered[subset_indices]
    else:
        gamma_data_for_analysis = gamma_data_filtered

    if gamma_data_for_analysis.shape[0] == 0:
        print(f"No samples left to analyze in {data_split}. Skipping.")
        return

    components = ["Area", "Perimeter", "Connected_Components"]
    component_labels = [
        "Area ($km^2$)",
        "Perimeter (km)",
        r"Connected Components ($\chi$)",
    ]

    pbar = tqdm(
        total=len(QUANTILE_LEVELS) * len(components),
        desc=f"Generating plots for {data_split}",
    )

    for i, threshold in enumerate(QUANTILE_LEVELS):
        for j, comp_name in enumerate(components):
            data_slice = gamma_data_for_analysis[:, j, i]
            data_slice_positive = data_slice[data_slice > 0]

            thresh_str = str(threshold).replace(".", "p")
            filename = f"{data_split}_{comp_name}_thresh_{thresh_str}.png"
            output_path = os.path.join(plot_output_dir, filename)

            if data_slice_positive.size < 20:
                pbar.update(1)
                continue

            # --- Analysis for Area/Perimeter (Gamma Fit) ---
            if comp_name in ["Area", "Perimeter"]:
                shape, loc, scale = stats.gamma.fit(data_slice_positive, floc=0)
                ks_stat, p_value = stats.kstest(
                    data_slice_positive, "gamma", args=(shape, 0, scale)
                )

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                fig.suptitle(
                    f"Gamma Fit for {component_labels[j]} at {threshold} mm/hr\n"
                    f"({data_split.capitalize()} Set, {ANALYSIS_DATA_PERCENTAGE}% of Data)",
                    fontsize=16,
                )

                ax1.hist(
                    data_slice_positive,
                    bins=50,
                    density=True,
                    color="darkblue",
                    alpha=0.6,
                    label="Empirical Data",
                )
                x_fit = np.linspace(
                    data_slice_positive.min(), data_slice_positive.max(), 200
                )
                ax1.plot(
                    x_fit,
                    stats.gamma.pdf(x_fit, shape, 0, scale),
                    "r-",
                    lw=2,
                    label="Fitted Gamma PDF",
                )
                ax1.set_xlabel(component_labels[j])
                ax1.set_ylabel("Density")
                ax1.set_title("Histogram vs. Fitted PDF")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.6)

                stats_text = (
                    f"Fitted Gamma:\n  Shape (a): {shape:.2f}\n  Scale (θ): {scale:.2f}\n\n"
                    f"K-S Test:\n  Stat: {ks_stat:.3f}\n  p-val: {p_value:.3f}"
                )
                ax1.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax1.transAxes,
                    fontsize=10,
                    va="top",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
                )

                stats.probplot(
                    data_slice_positive, dist=stats.gamma, sparams=(shape,), plot=ax2
                )
                ax2.get_lines()[0].set_markerfacecolor("darkblue")
                ax2.get_lines()[1].set_color("red")
                ax2.set_title("Q-Q Plot vs. Gamma")
                ax2.set_xlabel("Theoretical Quantiles")
                ax2.set_ylabel("Sample Quantiles")

            # --- Analysis for Connected Components (Negative Binomial Fit) ---
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                fig.suptitle(
                    f"Neg. Binomial Fit for {component_labels[j]} at {threshold} mm/hr\n"
                    f"({data_split.capitalize()} Set, {ANALYSIS_DATA_PERCENTAGE}% of Data)",
                    fontsize=16,
                )

                mean_val, var_val = (
                    data_slice_positive.mean(),
                    data_slice_positive.var(),
                )
                fit_successful, n_fit, p_fit = False, np.nan, np.nan
                if var_val > mean_val and mean_val > 0:
                    p_fit = mean_val / var_val
                    n_fit = mean_val * p_fit / (1 - p_fit)
                    if n_fit > 0 and 0 < p_fit <= 1:
                        fit_successful = True

                bins = np.arange(
                    data_slice_positive.min() - 0.5, data_slice_positive.max() + 1.5
                )
                ax1.hist(
                    data_slice_positive,
                    bins=bins,
                    density=True,
                    color="darkblue",
                    alpha=0.7,
                    label="Empirical Data",
                )
                if fit_successful:
                    k_range = np.arange(0, int(data_slice_positive.max() + 5))
                    ax1.plot(
                        k_range,
                        stats.nbinom.pmf(k_range, n_fit, p_fit),
                        "ro-",
                        label="Fitted PMF",
                        markersize=4,
                    )

                ax1.set_xlabel(component_labels[j])
                ax1.set_ylabel("Density")
                ax1.set_title("Histogram vs. Fitted PMF")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.6)
                ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax1.set_xlim(left=-0.5)

                stats_text = (
                    f"Sample Stats:\n  Mean: {mean_val:.2f}\n  Var: {var_val:.2f}\n"
                )
                if fit_successful:
                    ks_stat, p_value = stats.kstest(
                        data_slice_positive, lambda x: stats.nbinom.cdf(x, n_fit, p_fit)
                    )
                    stats_text += (
                        f"Fitted Neg. Binom:\n  n: {n_fit:.2f}\n  p: {p_fit:.2f}\n\n"
                        f"K-S Test:\n  Stat: {ks_stat:.3f}\n  p-val: {p_value:.3f}"
                    )
                else:
                    stats_text += "\nFit Failed (Var <= Mean)"
                ax1.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=ax1.transAxes,
                    fontsize=10,
                    va="top",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
                )

                if fit_successful:
                    plot_discrete_qq(
                        data_slice_positive,
                        dist=stats.nbinom,
                        sparams=(n_fit, p_fit),
                        ax=ax2,
                        title="Q-Q Plot vs. Neg. Binomial",
                        xlabel="Theoretical Quantiles",
                        ylabel="Sample Quantiles",
                    )
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "Q-Q Plot not generated\n(Fit Failed)",
                        ha="center",
                        va="center",
                        fontsize=14,
                        transform=ax2.transAxes,
                    )
                    ax2.set_title("Q-Q Plot")
                    ax2.set_xlabel("Theoretical Quantiles")
                    ax2.set_ylabel("Sample Quantiles")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            pbar.update(1)

    pbar.close()
    print(f"Finished analysis for {data_split}. Plots saved to {plot_output_dir}")


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        analyze_and_plot_distributions(split)
    print("\nAll distribution analysis is complete.")
