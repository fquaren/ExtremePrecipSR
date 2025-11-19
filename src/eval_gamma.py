import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import our new refactored libraries
import io_lib
import metrics_lib
import plotting_lib

# Import your existing modules
from loss import TotalErrorMetric, GeometricLossSeparate


def run_prediction_loop(model, loader, eval_metric, geom_metric_per_sample, device):
    """
    Runs the model over the test set and collects all results.
    """
    model.eval()
    all_preds_phys, all_targets_phys = [], []
    all_original_images, all_total_losses = [], []
    all_geom_losses = []

    with torch.no_grad():
        for input_data, log_target_gamma, original_precip, target_gamma_phys in tqdm(
            loader, desc="Running model inference"
        ):
            input_data, log_target_gamma, target_gamma_phys = (
                input_data.to(device),
                log_target_gamma.to(device),
                target_gamma_phys.to(device),
            )

            predicted_gamma_phys = model(input_data)

            # --- Calculate losses ---
            per_sample_total_losses = eval_metric(
                input_data, predicted_gamma_phys, log_target_gamma
            )
            per_sample_geom_losses = geom_metric_per_sample(
                predicted_gamma_phys, target_gamma_phys
            )

            # --- Collect results ---
            all_total_losses.append(per_sample_total_losses.cpu().numpy())
            all_geom_losses.append(per_sample_geom_losses.cpu().numpy())
            all_preds_phys.append(predicted_gamma_phys.cpu().numpy())
            all_targets_phys.append(target_gamma_phys.cpu().numpy())
            all_original_images.append(original_precip.squeeze(1).cpu().numpy())

    # Concatenate all results
    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_original_images = np.concatenate(all_original_images, axis=0)
    all_total_losses = np.concatenate(all_total_losses, axis=0)
    all_geom_losses = np.concatenate(all_geom_losses, axis=0)

    print(f"Inference complete. Processed {len(all_total_losses)} samples.")

    return (
        all_preds_phys,
        all_targets_phys,
        all_original_images,
        all_total_losses,
        all_geom_losses,
    )


def main(run_dir, constraint_mode):

    # --- 1. Setup ---
    config, device = io_lib.setup_evaluation(run_dir)
    model = io_lib.load_model(config, device, run_dir, constraint_mode)
    test_loader = io_lib.load_data(config)
    S_inv_tensors = io_lib.load_s_inv(config, device)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]

    # --- 2. Initialize Metric Objects ---
    evaluation_metric = TotalErrorMetric(
        quantile_levels=QUANTILE_LEVELS, config=config
    ).to(device)
    geometric_metric_per_sample = GeometricLossSeparate(
        S_inv_tensors, reduction="none"
    ).to(device)

    # --- 3. Run Inference ---
    (
        all_preds_phys,
        all_targets_phys,
        all_original_images,
        all_total_losses,
        all_geom_losses,
    ) = run_prediction_loop(
        model, test_loader, evaluation_metric, geometric_metric_per_sample, device
    )

    # --- 4. Compute All Metrics ---

    # Create the central DataFrame
    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_phys,
        all_targets_phys,
        all_original_images,
        all_total_losses,
        all_geom_losses,
    )

    # Calculate sample-averaged metrics
    group_metrics_sample_wise = metrics_lib.calculate_grouped_metrics(metrics_df)

    # Calculate global component-wise metrics
    group_metrics_global = metrics_lib.calculate_global_group_metrics(
        metrics_df, all_preds_phys, all_targets_phys
    )

    # Calculate per-feature metrics
    per_feature_metrics = metrics_lib.calculate_per_feature_metrics(
        all_preds_phys, all_targets_phys, QUANTILE_LEVELS
    )

    # --- 5. Save Results ---

    # Save metrics
    io_lib.save_metrics_text(
        run_dir,
        group_metrics_global,
        group_metrics_sample_wise,
        per_feature_metrics,
    )

    # --- 6. Generate Plots ---

    # Plot per-feature matrix heatmaps (R2, MSE, Var)
    plotting_lib.plot_per_feature_matrices(
        per_feature_metrics=per_feature_metrics, output_dir=run_dir
    )

    # Plot best/worst/average samples
    plotting_lib.plot_sample_comparisons(
        metrics_df=metrics_df,
        quantiles=QUANTILE_LEVELS,
        output_dir=run_dir,
        n_samples=15,
    )

    # Plot metric distributions
    plotting_lib.plot_metric_distributions(metrics_df=metrics_df, output_dir=run_dir)

    # Plot mean/std by group
    plotting_lib.plot_gamma_mean_std_by_quantile(
        metrics_df=metrics_df,
        group_metrics=group_metrics_sample_wise,  # Use sample-wise for standard deviation viz
        quantiles=QUANTILE_LEVELS,
        output_dir=run_dir,
    )

    # Plot training history
    log_file_path = os.path.join(run_dir, "training_log.csv")
    plotting_lib.plot_training_log(log_path=log_file_path, output_dir=run_dir)

    print("\nEvaluation script finished.")


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
        default=None,
        help="(Optional) Override constraint mode (none, soft, hybrid, hard).",
    )
    args = parser.parse_args()

    main(args.run_dir, args.constraint_mode)
