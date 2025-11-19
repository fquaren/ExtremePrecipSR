import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import yaml

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
PERSISTENCE_THRESHOLD = config.get("PERSISTENCE_THRESHOLD", 0.05)

# --- Plotting Configuration ---
DATA_SPLIT_TO_PLOT = "validation"  # Use validation to pick hyperparameters!
USE_LOG_SCALE = False
BIN_GRID_SIZE = 100
CMAP = "inferno"
USE_LOG_COLOR = True


def plot_persistence_diagram(data_dir, data_split):
    npz_path = os.path.join(data_dir, data_split, "gamma_targets_persistence.npz")

    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return

    print(f"Loading persistence pairs from {npz_path}...")
    with np.load(npz_path) as data:
        pairs = data["persistence_pairs"]

    if pairs.shape[0] == 0:
        print("No pairs found.")
        return

    births = pairs[:, 0]
    deaths = pairs[:, 1]

    # --- Filter Logic ---
    # 1. Calculate Persistence
    #    Handle inf for calculation (Inf - Finite = Inf)
    persistence = np.zeros_like(births)
    finite_mask = np.isfinite(deaths)
    infinite_mask = ~finite_mask

    persistence[finite_mask] = births[finite_mask] - deaths[finite_mask]
    persistence[infinite_mask] = 999999  # Marker for infinite

    # 2. Get Valid Finite Persistence for Metrics (Ignore the infinite top line)
    #    And ignore negative noise (birth < death should not happen in standard TDA but numerical errors exist)
    valid_finite_mask = (finite_mask) & (persistence > 0)
    valid_persistence = persistence[valid_finite_mask]

    print(f"Total pairs: {len(pairs)}")
    print(f"Infinite (Essential) pairs: {np.sum(infinite_mask)}")
    print(f"Finite Noise/Signal pairs: {len(valid_persistence)}")

    # --- Metric 1: Histogram ---
    plt.figure(figsize=(10, 6))
    if len(valid_persistence) > 0:
        min_p = np.min(valid_persistence)
        max_p = np.max(valid_persistence)

        # Log bins
        bins = np.logspace(np.log10(max(min_p, 1e-4)), np.log10(max_p), 100)

        plt.hist(valid_persistence, bins=bins, color="darkcyan", label="Finite Pairs")
        plt.xscale("log")
        plt.yscale("log")
        plt.axvline(
            PERSISTENCE_THRESHOLD,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Config Threshold = {PERSISTENCE_THRESHOLD}",
        )

        # Heuristic: 99th percentile of the lower noise
        # This is a rough guess: many points are noise, the tail is signal
        # Let's just mark it for reference
        if len(valid_persistence) > 100:
            p90 = np.percentile(valid_persistence, 90)
            plt.axvline(
                p90, color="orange", linestyle=":", label=f"90th Percentile = {p90:.2f}"
            )

        plt.title(f"Metric 1: Persistence Histogram ({data_split})")
        plt.xlabel("Persistence (Birth - Death)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.savefig(f"persistence_histogram_{data_split}.png", dpi=300)
        print("Saved histogram.")

    # --- Metric 2: Elbow Plot ---
    plt.figure(figsize=(10, 6))
    if len(valid_persistence) > 0:
        sorted_p = np.sort(valid_persistence)[::-1]
        ranks = np.arange(1, len(sorted_p) + 1)

        plt.plot(ranks, sorted_p, "b-", alpha=0.8)
        plt.xscale("log")
        plt.yscale("log")
        plt.axhline(
            PERSISTENCE_THRESHOLD,
            color="r",
            linestyle="--",
            label=f"Config Threshold = {PERSISTENCE_THRESHOLD}",
        )
        plt.title(f"Metric 2: Elbow Plot ({data_split})")
        plt.xlabel("Rank")
        plt.ylabel("Persistence")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.savefig(f"persistence_elbow_plot_{data_split}.png", dpi=300)
        print("Saved elbow plot.")

    # --- Main Diagram ---
    # Transform infinite deaths for visualization
    # If death is infinite, place it at 1.1 * max_birth for visualization
    max_b = np.max(births[np.isfinite(births)]) if np.any(np.isfinite(births)) else 1.0

    deaths_plot = deaths.copy()
    deaths_plot[infinite_mask] = max_b * 1.1

    # Filter for log scale plotting (needs > 0)
    plot_mask = (births > 0) & (deaths_plot > 0)
    b_plot = births[plot_mask]
    d_plot = deaths_plot[plot_mask]

    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(
        b_plot,
        d_plot,
        gridsize=BIN_GRID_SIZE,
        cmap=CMAP,
        xscale="linear",
        yscale="linear",
        mincnt=1,
        norm=LogNorm() if USE_LOG_COLOR else None,
    )

    # Diagonal
    lims = [0, max_b * 1.2]
    plt.plot(lims, lims, "r--", alpha=0.5)

    # Threshold Line
    x_line = np.linspace(lims[0], lims[1], 100)
    y_line = x_line - PERSISTENCE_THRESHOLD
    plt.plot(
        x_line,
        y_line,
        "g-.",
        linewidth=1.5,
        label=f"Threshold = {PERSISTENCE_THRESHOLD}",
    )

    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect("equal")
    plt.colorbar(hb, label="Count (log)")
    plt.title(f"Persistence Diagram - {data_split}")
    plt.legend(loc="upper left")
    plt.savefig(f"persistence_diagram_{data_split}.png", dpi=300)
    print("Saved main diagram.")


if __name__ == "__main__":
    plot_persistence_diagram(PREPROCESSED_DATA_DIR, DATA_SPLIT_TO_PLOT)
