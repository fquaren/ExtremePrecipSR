import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # Added explicit import
import copy
import yaml

# Import your custom libraries
import io_lib
from gamma_predictors_v5 import BaselineCNN, IsometricCNN, ConstrainedIsometricCNN

# ==========================================
# Plotting Logic
# ==========================================


def plot_comparative_panel(
    sample_best,
    sample_worst,
    predictions_best,
    predictions_worst,
    quantiles,
    best_idx,
    worst_idx,
    output_dir,
):
    """
    Generates a 2x4 grid using GridSpec:
    Row 0: Best Sample (Image + A/P/CC curves)
    Row 1: Worst Sample (Image + A/P/CC curves)
    Width Ratios: [1, 3, 3, 3] (Precipitation map is square-ish, curves are wide)
    """
    print(f"\nGenerating comparative plot for indices: {best_idx} and {worst_idx}")

    # Unpack Data
    rows_data = [
        (sample_best, predictions_best, best_idx),
        (sample_worst, predictions_worst, worst_idx),
    ]

    gamma_labels = ["Area (km²)", "Perimeter (km)", "CCs"]
    model_styles = {
        "Target": {"color": "black", "ls": "-", "marker": "o", "lw": 2, "alpha": 0.8},
        "CNN (Unconstr.)": {
            "color": "#648fff",  # "blue",
            "ls": "--",
            "marker": "x",
            "lw": 1,
        },
        "Lip-CNN (Unconstr.)": {
            "color": "#ffb000",  # "orange",
            "ls": "-.",
            "marker": "^",
            "lw": 1,
        },
        "Lip-CNN (Constr.)": {
            "color": "#dc2680",  # "magenta",
            "ls": ":",
            "marker": "s",
            "lw": 1,
        },
    }

    # 1. Initialize Figure
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)

    # 2. Define GridSpec with requested ratios
    # 2 rows, 4 columns
    # Col 0 (Precip) = 1 unit width
    # Cols 1-3 (Components) = 1 unit width
    gs = gridspec.GridSpec(2, 4, width_ratios=[0.5, 1, 1, 1], figure=fig)

    for row_idx, (sample, preds, sample_idx) in enumerate(rows_data):
        # Unpack sample
        _, _, orig_log_precip_tensor, target_phys_tensor = sample

        # Convert to numpy
        precip_img = np.expm1(orig_log_precip_tensor.squeeze().cpu().numpy())
        target_gamma = target_phys_tensor.cpu().numpy()

        # --- Col 0: Precipitation Image ---
        # Add subplot to the specific grid slot
        ax_img = fig.add_subplot(gs[row_idx, 0])

        # Mask zeros for visualization
        plot_data = precip_img.copy()
        plot_data[plot_data <= 0] = np.nan

        # Prepare Colormap
        cmap = copy.copy(plt.get_cmap("Blues"))
        cmap.set_bad(color="lightgrey", alpha=1.0)

        im = ax_img.imshow(plot_data, cmap=cmap, origin="lower")
        # No title for precip plot
        fig.colorbar(im, ax=ax_img, shrink=0.8, label="Precip. (mm/hr)")

        # --- Cols 1-3: Gamma Components ---
        for col_idx in range(3):
            # Add subplot to grid slot
            ax_gamma = fig.add_subplot(gs[row_idx, col_idx + 1])
            component_idx = col_idx

            # Plot Target
            ax_gamma.plot(
                quantiles,
                target_gamma[component_idx],
                label="Target",
                **model_styles["Target"],
            )

            # Plot Models
            for model_name, model_pred in preds.items():
                pred_curve = model_pred[component_idx]
                ax_gamma.plot(
                    quantiles, pred_curve, label=model_name, **model_styles[model_name]
                )

            ax_gamma.grid(True, linestyle="--", alpha=0.5)

            # Title only on the first row
            if row_idx == 0:
                ax_gamma.set_title(gamma_labels[component_idx], fontsize=14)

            # X-label only on the second row
            if row_idx == 1:
                ax_gamma.set_xlabel("Precip. Threshold (mm/hr)", fontsize=12)
            else:
                # Hide x-tick labels on the top row to reduce clutter
                ax_gamma.tick_params(labelbottom=False)

            # # Log scale for Area and Perimeter
            # if component_idx < 2:
            #     ax_gamma.set_yscale("log")

            # Legend only on the first gamma plot of the first row
            if row_idx == 0 and col_idx == 0:
                ax_gamma.legend()

    # Save
    filename = f"comparison_best_{best_idx}_worst_{worst_idx}.pdf"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


# ==========================================
# Helpers
# ==========================================


def get_prediction(model, input_tensor):
    """Run inference for a single sample."""
    model.eval()
    with torch.no_grad():
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch)
        if isinstance(output, tuple):
            output = output[0]
    return output.squeeze(0).cpu().numpy()


# ==========================================
# Main Execution
# ==========================================


def main(args):
    # 1. Setup Data & Config
    print("--- Setting up environment based on Baseline run ---")
    config, device, scaler_val = io_lib.setup_evaluation(args.dir_baseline)

    # Load dataset
    test_loader = io_lib.load_data(config, scaler_val)
    test_dataset = test_loader.dataset

    quantiles = config["QUANTILE_LEVELS"]

    # 2. Load Models
    # CRITIQUE: Map internal architecture names to Display Labels immediately here.
    # This ensures consistency downstream in the plotting logic.
    print("\n--- Loading Models ---")
    models = {}

    models["CNN (Unconstr.)"] = io_lib.load_model(
        config, device, args.dir_baseline, scaler_val, architecture_type="Baseline"
    )
    # assuming Isometric corresponds to the Unconstrained Lipschitz architecture in this context
    models["Lip-CNN (Unconstr.)"] = io_lib.load_model(
        config, device, args.dir_isometric, scaler_val, architecture_type="Isometric"
    )
    models["Lip-CNN (Constr.)"] = io_lib.load_model(
        config,
        device,
        args.dir_constrained,
        scaler_val,
        architecture_type="Constrained",
    )

    # 3. Retrieve Samples
    print(f"\n--- Retrieving Samples {args.best_idx} and {args.worst_idx} ---")

    if args.best_idx >= len(test_dataset) or args.worst_idx >= len(test_dataset):
        raise IndexError(f"Indices must be < {len(test_dataset)}")

    sample_best = test_dataset[args.best_idx]
    sample_worst = test_dataset[args.worst_idx]

    input_best = sample_best[0].to(device)
    input_worst = sample_worst[0].to(device)

    # 4. Run Inference
    preds_best = {}
    preds_worst = {}

    for name, model in models.items():
        preds_best[name] = get_prediction(model, input_best)
        preds_worst[name] = get_prediction(model, input_worst)

    # 5. Plot
    output_dir = "/home/fquareng/work/figures/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)

    plot_comparative_panel(
        sample_best,
        sample_worst,
        preds_best,
        preds_worst,
        quantiles,
        args.best_idx,
        args.worst_idx,
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Baseline, Isometric, and Constrained emulators."
    )

    parser.add_argument(
        "--dir_baseline",
        type=str,
        required=True,
        help="Run directory for Baseline model",
    )
    parser.add_argument(
        "--dir_isometric",
        type=str,
        required=True,
        help="Run directory for Isometric model",
    )
    parser.add_argument(
        "--dir_constrained",
        type=str,
        required=True,
        help="Run directory for Constrained model",
    )

    parser.add_argument(
        "--best_idx", type=int, required=True, help="Index of the 'Best' sample"
    )
    parser.add_argument(
        "--worst_idx", type=int, required=True, help="Index of the 'Worst' sample"
    )

    args = parser.parse_args()
    main(args)
