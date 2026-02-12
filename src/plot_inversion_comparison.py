import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from matplotlib.gridspec import GridSpec


def load_inversion_data(filepath):
    """Loads a single .npz inversion result file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath, allow_pickle=True)

    return {
        "final_image": data["final_image"],
        "initial_noise": data["initial_noise"],
        "target_vector": data["target_vector"],
        "ground_truth": data["ground_truth"] if "ground_truth" in data else None,
        "has_gt": "ground_truth" in data and data["ground_truth"].size > 0,
    }


def plot_comparison_figure(
    run_paths, run_names, output_path, quantile_levels, target_mode_label="Real Storm"
):
    """
    Generates the composite figure.
    """

    # 1. Load Data
    print(f"Loading data from {len(run_paths)} files...")
    results = [load_inversion_data(p) for p in run_paths]

    ref_target = results[0]["target_vector"]
    if ref_target.ndim == 3:
        ref_target = ref_target[0]

    phys_target = np.expm1(ref_target)  # Shape (3, Q)

    # 2. Setup Figure Layout
    # constrained_layout=True handles the spacing between subplots automatically
    # avoiding the need for tight_layout() which breaks add_axes()
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)

    # GridSpec:
    # Col 0: Targets (width 1)
    # Col 1: Noise (width 0.8)
    # Col 2: 2x2 Grid (width 1.2 to accomodate 2 images wide)
    # Col 3: Spacer for Colorbar (width 0.1)
    gs = GridSpec(
        1, 4, figure=fig, width_ratios=[0.7, 0.5, 1, 0.1], hspace=0.3, wspace=0.2
    )

    # --- Column 1: Target Statistics (Stacked Vertically) ---
    # We create a nested gridspec for the left column to stack the 3 plots
    gs_left = gs[0, 0].subgridspec(3, 1, hspace=0.05)

    ax_area = fig.add_subplot(gs_left[0, 0])
    ax_perim = fig.add_subplot(gs_left[1, 0], sharex=ax_area)
    ax_euler = fig.add_subplot(gs_left[2, 0], sharex=ax_area)

    thresholds = np.array(quantile_levels)

    # Plot Area
    ax_area.plot(thresholds, phys_target[0], "k-o", lw=1.5, markersize=4)
    ax_area.set_ylabel("Area (km²)")
    ax_area.grid(True, alpha=0.3)
    # Remove x-ticks for top plots
    plt.setp(ax_area.get_xticklabels(), visible=False)

    # Plot Perimeter
    ax_perim.plot(thresholds, phys_target[1], "k-o", lw=1.5, markersize=4)
    ax_perim.set_ylabel("Perimeter (km)")
    ax_perim.grid(True, alpha=0.3)
    plt.setp(ax_perim.get_xticklabels(), visible=False)

    # Plot Euler
    ax_euler.plot(thresholds, phys_target[2], "k-o", lw=1.5, markersize=4)
    ax_euler.set_ylabel("CC (count)")
    ax_euler.set_xlabel("Precip Threshold (quantile)")
    ax_euler.grid(True, alpha=0.3)

    # --- Column 2: Initial Noise (Centered Vertically) ---
    gs_mid = gs[0, 1].subgridspec(1, 1)
    ax_noise = fig.add_subplot(gs_mid[0, 0])

    noise_img = results[0]["initial_noise"]
    cmap = copy.copy(plt.get_cmap("cividis_r"))
    cmap.set_bad(color="lightgrey")

    vmax_noise = max(np.max(noise_img), 1e-6)

    # Anchor 'C' ensures it centers vertically if the aspect ratio differs
    im_noise = ax_noise.imshow(
        noise_img, cmap="viridis", origin="lower", vmin=0, vmax=vmax_noise
    )
    ax_noise.set_title("Initial Noise")
    ax_noise.axis("off")
    ax_noise.set_anchor("C")

    # Noise Colorbar (inside the subplot area at bottom)
    cb_noise = plt.colorbar(
        im_noise, ax=ax_noise, location="bottom", shrink=0.8, pad=0.02
    )
    cb_noise.set_label("Noise Amplitude (mm/hr)")

    # push up the noise axis to center it vertically
    box = ax_noise.get_position()
    ax_noise.set_position(
        [box.x0 - 0.01, box.y0 + box.height * 0.3, box.width, box.height]
    )

    # Adjust colorbar position accordingly
    cb_noise.ax.set_position(
        [
            (box.x0 - 0.01) + 0.1 * box.width,
            box.y0 + box.height * 0.3 - 0.08,
            0.8 * box.width,
            0.05,
        ]
    )

    # --- Column 3: The 2x2 Grid (Centered Vertically) ---
    # Nested gridspec for the 2x2 images
    gs_right = gs[0, 2].subgridspec(2, 2, wspace=0.05, hspace=0.05)

    ax_r1 = fig.add_subplot(gs_right[0, 0])
    ax_r2 = fig.add_subplot(gs_right[0, 1])
    ax_r3 = fig.add_subplot(gs_right[1, 0])
    ax_gt = fig.add_subplot(gs_right[1, 1])

    # Prepare Data
    imgs_to_plot = [r["final_image"] for r in results]
    if results[0]["has_gt"]:
        gt_img = results[0]["ground_truth"]
    else:
        gt_img = np.zeros_like(results[0]["final_image"])
    imgs_to_plot.append(gt_img)

    vmax_storm = max([np.max(img) for img in imgs_to_plot])
    vmax_storm = max(vmax_storm, 1.0)

    def plot_storm(ax, img, title):
        # Mask low values for visibility
        img_masked = img.copy()
        img_masked[img_masked <= 0.1] = np.nan

        im = ax.imshow(img_masked, cmap=cmap, origin="lower", vmin=0, vmax=vmax_storm)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        ax.set_anchor("C")  # Ensure images center in their slots
        return im

    # Plot images
    im_final = plot_storm(ax_r1, results[0]["final_image"], run_names[0])
    plot_storm(ax_r2, results[1]["final_image"], run_names[1])
    plot_storm(ax_r3, results[2]["final_image"], run_names[2])

    if results[0]["has_gt"]:
        plot_storm(ax_gt, gt_img, "Ground Truth")
    else:
        ax_gt.text(0.5, 0.5, "No Ground Truth", ha="center", va="center")
        ax_gt.axis("off")

    # --- Colorbar ---
    # We use add_axes relative to the figure to place it on the far right
    # [left, bottom, width, height]

    # Plot colorbar in the space reserved in the gridspec
    cbar_ax = fig.add_axes([0.97, 0.15, 0.015, 0.7])
    cb_storm = fig.colorbar(im_final, cax=cbar_ax)
    cb_storm.set_label("Precipitation (mm/hr)")

    print(f"Saving comparison figure to {output_path}...")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare inversion results from 3 runs."
    )
    parser.add_argument(
        "--files", nargs=3, required=True, help="Paths to the 3 .npz data files"
    )
    parser.add_argument(
        "--names",
        nargs=3,
        default=["CNN (Unconstr.)", "Lip-CNN (Unconstr.)", "Lip-CNN (Constr.)"],
        help="Display names for the models",
    )
    parser.add_argument("--output", type=str, default="inversion_comparison.pdf")

    parser.add_argument(
        "--quantile_levels",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0],
        help="Quantile thresholds used in the models (for x-axis)",
    )

    args = parser.parse_args()

    plot_comparison_figure(args.files, args.names, args.output, args.quantile_levels)
