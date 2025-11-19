import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from tqdm import tqdm
import warnings

# Import our new refactored libraries
import io_lib_sr as io_lib
import metrics_lib_sr as metrics_lib
import plotting_lib_sr as plotting_lib

# Import your existing modules
from loss import (
    ComponentWiseCDFLoss,
    compute_gamma_matrix_for_image,
    GeometricLossSeparate,
)

# Suppress warnings
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def run_prediction_loop(
    model,
    loader,
    mse_criterion,
    surrogate_criterion,
    device,
    eval_mode,
    surrogate_weight,
    quantile_levels,
    pixel_size_km,
):
    """
    Runs the model over the test set and collects all raw results.
    """
    model.eval()

    all_preds_phys, all_targets_phys, all_inputs_phys = [], [], []
    all_total_losses, all_mse_losses, all_surrogate_losses = [], [], []

    with torch.no_grad():
        for X, Y_true_phys, Y_gamma_phys in tqdm(
            loader, desc="Running model inference"
        ):
            X, Y_true_phys, Y_gamma_phys = (
                X.to(device),
                Y_true_phys.to(device),
                Y_gamma_phys.to(device),
            )

            with torch.amp.autocast(device_type="cuda"):
                pred_X_phys = model(X)

                # --- Calculate Core Losses (per-sample) ---
                loss_mse_per_sample = mse_criterion(pred_X_phys, Y_true_phys).mean(
                    dim=(1, 2, 3)
                )
                loss_surrogate_per_sample = torch.zeros_like(loss_mse_per_sample)

                if eval_mode != "none" and surrogate_criterion is not None:
                    # Analytically compute gammas from predicted images
                    pred_gamma_phys_analytic = []
                    pred_X_np = pred_X_phys.cpu().numpy()
                    for i in range(pred_X_np.shape[0]):
                        gamma_matrix = compute_gamma_matrix_for_image(
                            pred_X_np[i, 0], quantile_levels, pixel_size_km
                        )
                        pred_gamma_phys_analytic.append(gamma_matrix)

                    pred_gamma_phys = torch.from_numpy(
                        np.array(pred_gamma_phys_analytic)
                    ).to(device)

                    if isinstance(surrogate_criterion, ComponentWiseCDFLoss):
                        pred_gamma_log = torch.log1p(pred_gamma_phys)
                        true_gamma_log = torch.log1p(Y_gamma_phys)
                        loss_A, loss_P, loss_CC = surrogate_criterion(
                            pred_gamma_log, true_gamma_log
                        )
                        loss_surrogate_per_sample = loss_A + loss_P + loss_CC
                    elif isinstance(surrogate_criterion, GeometricLossSeparate):
                        loss_surrogate_per_sample = surrogate_criterion(
                            pred_gamma_phys, Y_gamma_phys
                        )

                # --- Determine which loss to use for ranking ---
                if eval_mode == "validate":
                    loss_for_ranking = loss_surrogate_per_sample
                elif eval_mode == "train":
                    loss_for_ranking = (
                        (1 - surrogate_weight) * loss_mse_per_sample
                        + surrogate_weight * loss_surrogate_per_sample
                    )
                else:  # EVAL_MODE == "none" or default
                    loss_for_ranking = loss_mse_per_sample

            # --- Collect results ---
            all_preds_phys.append(pred_X_phys.squeeze(1).cpu().numpy())
            all_targets_phys.append(Y_true_phys.squeeze(1).cpu().numpy())
            all_inputs_phys.append(X.cpu().numpy())

            all_total_losses.append(loss_for_ranking.cpu().numpy())
            all_mse_losses.append(loss_mse_per_sample.cpu().numpy())
            all_surrogate_losses.append(loss_surrogate_per_sample.cpu().numpy())

    # Concatenate all results
    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_inputs_phys = np.concatenate(all_inputs_phys, axis=0)
    all_total_losses = np.concatenate(all_total_losses, axis=0)
    all_mse_losses = np.concatenate(all_mse_losses, axis=0)
    all_surrogate_losses = np.concatenate(all_surrogate_losses, axis=0)

    print(f"Inference complete. Processed {len(all_total_losses)} samples.")

    return (
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
    )


def main(run_dir):

    # --- 1. Setup ---
    config, device = io_lib.setup_evaluation(run_dir)
    model = io_lib.load_sr_model(config, device, run_dir)
    dem_stats = io_lib.load_dem_stats(config)
    test_loader = io_lib.load_data(config, dem_stats)

    # --- Config variables ---
    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
    EVAL_MODE = config.get("EVAL_MODE", "none")
    SURROGATE_LOSS_TYPE = config.get("SURROGATE_LOSS_TYPE", "cdf")
    SURROGATE_LOSS_WEIGHT = config.get("SURROGATE_LOSS_WEIGHT", 0.1)

    # --- 2. Initialize Metric Objects ---
    mse_criterion = nn.MSELoss(reduction="none")  # For per-sample MSE
    surrogate_criterion = None
    if EVAL_MODE != "none":
        if SURROGATE_LOSS_TYPE == "cdf":
            surrogate_criterion = ComponentWiseCDFLoss(
                quantile_levels=QUANTILE_LEVELS
            ).to(device)
        elif SURROGATE_LOSS_TYPE == "geometric":
            S_inv_tensors = io_lib.load_s_inv(config, dem_stats, device)
            # Use reduction='none' for per-sample loss
            surrogate_criterion = GeometricLossSeparate(
                S_inv_tensors, reduction="none"
            ).to(device)

    # --- 3. Run Inference ---
    (
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
    ) = run_prediction_loop(
        model,
        test_loader,
        mse_criterion,
        surrogate_criterion,
        device,
        EVAL_MODE,
        SURROGATE_LOSS_WEIGHT,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
    )

    # --- 4. Compute All Metrics (The heavy lifting) ---

    # Create the central DataFrame (computes FSS, SAL, Gamma R^2, etc.)
    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
    )

    # Calculate grouped metrics (by precip)
    group_metrics = metrics_lib.calculate_grouped_metrics(metrics_df)

    # Calculate per-feature analytical gamma metrics
    per_feature_gamma_metrics = metrics_lib.calculate_per_feature_gamma_metrics(
        metrics_df, QUANTILE_LEVELS
    )

    # --- 5. Save Results ---
    io_lib.save_metrics_text(run_dir, group_metrics, per_feature_gamma_metrics)
    io_lib.save_metrics_npz(run_dir, metrics_df, per_feature_gamma_metrics)

    # --- 6. Generate Plots ---

    # Plot best/worst/average samples
    plotting_lib.plot_sample_comparisons(
        metrics_df=metrics_df,
        output_dir=run_dir,
        dem_stats=dem_stats,
        n_samples=15,
    )

    # Plot metric distributions
    plotting_lib.plot_metric_distributions(metrics_df=metrics_df, output_dir=run_dir)

    # Plot mean/std of analytical gammas by group
    plotting_lib.plot_gamma_mean_std_by_quantile(
        metrics_df=metrics_df,
        group_metrics=group_metrics,
        quantiles=QUANTILE_LEVELS,
        output_dir=run_dir,
    )

    # Plot training history
    log_file_path = os.path.join(run_dir, "sr_training_log.csv")
    plotting_lib.plot_training_log(
        log_path=log_file_path, output_dir=run_dir, config=config
    )

    print("\n✅ Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet-SR model.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the timestamped experiment run directory.",
    )
    args = parser.parse_args()

    main(args.run_dir)
