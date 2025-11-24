import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import warnings

# Import shared libraries (Alignment with SR)
import io_lib_sr as io_lib
import metrics_lib_sr as metrics_lib
import plotting_lib_sr as plotting_lib

# Import DDPM specific modules
from model_ddpm import ContextUnet
from diffusion import Diffusion

# Import Physics Logic
from loss import (
    compute_gamma_matrix_for_image,
    GeometricLossSeparate,
)
from utils import load_emulator

warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)


def load_ddpm_model(config, device, run_dir):
    """
    Loads the DDPM ContextUnet and Diffusion scheduler.
    """
    checkpoint_path = os.path.join(run_dir, "ddpm_latest.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DDPM checkpoint not found at {checkpoint_path}")

    print(f"Loading DDPM model from: {checkpoint_path}")

    # Initialize Model (Parameters must match training config)
    # Assuming standard setup: 1 channel in, 1 channel condition
    model = ContextUnet(in_channels=1, c_in_condition=1, device=device).to(device)

    # Load Weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize Diffusion Scheduler
    # Ensure img_size matches your PATCH_SIZE config
    diffusion = Diffusion(
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=config["PATCH_SIZE"],
        device=device,
    )

    return model, diffusion


def run_ddpm_prediction_loop(
    model,
    diffusion,
    loader,
    audit_criterion,
    emulator,
    device,
    quantile_levels,
    pixel_size_km,
):
    """
    Iterates through the test set, sampling from the DDPM for each batch.
    Computes Analytic Topology and performs Emulator Audit.
    """
    model.eval()
    if emulator:
        emulator.eval()

    # Storage
    all_preds_gamma_analytic = []
    all_targets_gamma_analytic = []

    all_preds_phys, all_targets_phys, all_inputs_phys = [], [], []
    # For DDPM, "Total Loss" in this context is treated as MAE for ranking
    all_total_losses, all_mse_losses, all_surrogate_losses = [], [], []

    audit_results = {
        "L1_phys_err": [],
        "L2_perc_err": [],
        "L3_intr_err": [],
        "consistency_gap": [],
    }

    # Metrics for on-the-fly calculation
    mae_criterion = nn.L1Loss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    print("\n--- Starting DDPM Inference ---")
    print(
        "Note: Sampling is iterative (1000 steps). This will be slower than UNet inference."
    )

    with torch.no_grad():
        # We enumerate to show progress clearly as DDPM batches are slow
        for batch_idx, (X, Y_true_phys, Y_gamma_phys_analytic) in enumerate(loader):
            X = X.to(device)
            Y_true_phys = Y_true_phys.to(device)
            Y_gamma_phys_analytic = Y_gamma_phys_analytic.to(device)

            # --- 1. DDPM Sampling (The Generative Step) ---
            # X is the condition (Low Res / Upsamled Input)
            # diffusion.sample returns [B, 1, H, W]
            # We create a nested progress bar for the sampling steps if desired,
            # or let diffusion.sample handle it.
            pred_X_phys = diffusion.sample(model, n=X.shape[0], conditions=X)

            # Ensure Physics (Non-negative)
            # Diffusion usually clamps to [-1, 1] or [0, 1] depending on implementation.
            # Our data is [0, 1].
            pred_X_phys = pred_X_phys.clamp(0.0, 1.0)

            # --- 2. Compute Analytic Gamma for Predictions (CPU TDA) ---
            pred_X_np = pred_X_phys.cpu().numpy()
            batch_gammas = []
            for i in range(pred_X_np.shape[0]):
                g = compute_gamma_matrix_for_image(
                    pred_X_np[i, 0], quantile_levels, pixel_size_km
                )
                batch_gammas.append(g)
            batch_gammas = np.array(batch_gammas)

            all_preds_gamma_analytic.append(batch_gammas)
            all_targets_gamma_analytic.append(Y_gamma_phys_analytic.cpu().numpy())

            # --- 3. Calculate Pixel Losses ---
            # MAE is the deterministic equivalent of CRPS
            loss_mae_sample = mae_criterion(pred_X_phys, Y_true_phys).mean(
                dim=(1, 2, 3)
            )
            loss_mse_sample = mse_criterion(pred_X_phys, Y_true_phys).mean(
                dim=(1, 2, 3)
            )

            # Surrogate loss placeholder (we don't optimize this in DDPM inference usually)
            loss_surr_sample = torch.zeros_like(loss_mae_sample)

            if emulator and audit_criterion:
                # If we want to rank samples by topological fidelity
                pred_gamma_emu = emulator(pred_X_phys)
                loss_surr_sample = audit_criterion(
                    pred_gamma_emu, Y_gamma_phys_analytic
                )

            # --- 4. Emulator Consistency Audit ---
            if emulator and audit_criterion:
                gamma_pred_emu = emulator(pred_X_phys)
                gamma_true_emu = emulator(Y_true_phys)

                l1 = audit_criterion(gamma_pred_emu, Y_gamma_phys_analytic)
                l2 = audit_criterion(gamma_pred_emu, gamma_true_emu)
                l3 = audit_criterion(gamma_true_emu, Y_gamma_phys_analytic)
                gap = torch.abs(l1 - l2)

                audit_results["L1_phys_err"].append(l1.cpu().numpy())
                audit_results["L2_perc_err"].append(l2.cpu().numpy())
                audit_results["L3_intr_err"].append(l3.cpu().numpy())
                audit_results["consistency_gap"].append(gap.cpu().numpy())
            else:
                nan_vec = np.full(X.shape[0], np.nan)
                for k in audit_results:
                    audit_results[k].append(nan_vec)

            # Collect Results
            all_preds_phys.append(pred_X_phys.squeeze(1).cpu().numpy())
            all_targets_phys.append(Y_true_phys.squeeze(1).cpu().numpy())
            all_inputs_phys.append(X.cpu().numpy())

            all_total_losses.append(
                loss_mae_sample.cpu().numpy()
            )  # Using MAE as "Total" for ranking
            all_mse_losses.append(loss_mse_sample.cpu().numpy())
            all_surrogate_losses.append(loss_surr_sample.cpu().numpy())

            print(f"Batch {batch_idx+1} Processed. MAE: {loss_mae_sample.mean():.4f}")

    # Concatenate Results
    all_preds_gamma_analytic = np.concatenate(all_preds_gamma_analytic, axis=0)
    all_targets_gamma_analytic = np.concatenate(all_targets_gamma_analytic, axis=0)

    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_inputs_phys = np.concatenate(all_inputs_phys, axis=0)

    all_total_losses = np.concatenate(all_total_losses, axis=0)
    all_mse_losses = np.concatenate(all_mse_losses, axis=0)
    all_surrogate_losses = np.concatenate(all_surrogate_losses, axis=0)

    for k in audit_results:
        audit_results[k] = np.concatenate(audit_results[k], axis=0)

    print(f"Inference complete. Processed {len(all_total_losses)} samples.")

    return (
        all_preds_gamma_analytic,
        all_targets_gamma_analytic,
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        audit_results,
    )


def main(run_dir):
    # Reuse the Setup logic from SR lib
    config, device = io_lib.setup_evaluation(run_dir)

    # --- Load DDPM Specifics ---
    model, diffusion = load_ddpm_model(config, device, run_dir)

    dem_stats = io_lib.load_dem_stats(config)
    test_loader = io_lib.load_data(config, dem_stats)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
    EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)

    # --- Audit Setup ---
    emulator = None
    audit_criterion = None
    if EMULATOR_CHECKPOINT_PATH and os.path.exists(EMULATOR_CHECKPOINT_PATH):
        print(f"Loading emulator for audit: {EMULATOR_CHECKPOINT_PATH}")
        emulator = load_emulator(EMULATOR_CHECKPOINT_PATH, config, device)
        S_inv_tensors = io_lib.load_s_inv(config, dem_stats, device)
        # Use None reduction to get per-sample consistency
        audit_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
            device
        )

    # --- Inference ---
    (
        all_preds_gamma_analytic,
        all_targets_gamma_analytic,
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        audit_results,
    ) = run_ddpm_prediction_loop(
        model,
        diffusion,
        test_loader,
        audit_criterion,
        emulator,
        device,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
    )

    # --- Create DataFrame (Reuse SR Lib) ---
    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_gamma_analytic,
        all_targets_gamma_analytic,
        all_inputs_phys,
        all_targets_phys,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
    )

    # Add Audit Metrics
    metrics_df["L1_Physical_Error"] = audit_results["L1_phys_err"]
    metrics_df["Consistency_Flag"] = audit_results["consistency_gap"]

    # Calculate Metrics
    group_metrics = metrics_lib.calculate_grouped_metrics(metrics_df)
    per_feature_gamma_metrics = metrics_lib.calculate_per_feature_gamma_metrics(
        metrics_df, QUANTILE_LEVELS
    )

    # Save Results
    io_lib.save_metrics_text(run_dir, group_metrics, per_feature_gamma_metrics)
    io_lib.save_metrics_npz(run_dir, metrics_df, per_feature_gamma_metrics)

    # Print Audit Summary
    if emulator:
        pass_rate = (metrics_df["Consistency_Flag"] < 0.3).mean() * 100
        print("\n" + "=" * 40)
        print(f"EMULATOR AUDIT SUMMARY (N={len(metrics_df)})")
        print("=" * 40)
        print(f"Pass Rate (Flag < 0.3): {pass_rate:.2f}%")
        print(f"Avg Physical Error:     {metrics_df['L1_Physical_Error'].mean():.4f}")
        print("=" * 40 + "\n")

    # Generate Plots (Exact same plots as SR for comparison)
    plotting_lib.plot_sample_comparisons_fixed(metrics_df, QUANTILE_LEVELS, run_dir)
    plotting_lib.plot_per_feature_matrices(per_feature_gamma_metrics, run_dir)
    plotting_lib.plot_gamma_mean_std_by_quantile(
        metrics_df, group_metrics, QUANTILE_LEVELS, run_dir
    )
    plotting_lib.plot_metric_distributions(metrics_df, run_dir)

    # Note: Training log plot is skipped or needs separate handling as DDPM logs might differ structure,
    # but keeping it optional in library is fine.

    print("\nDDPM Evaluation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to DDPM run directory")
    args = parser.parse_args()
    main(args.run_dir)
