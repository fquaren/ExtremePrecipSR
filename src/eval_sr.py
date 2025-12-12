import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
import warnings
from metrics import compute_fss, compute_sal

# Import our libraries
import io_lib_sr as io_lib
import metrics_lib_sr as metrics_lib
import plotting_lib_sr as plotting_lib

# Import logic
from loss import (
    compute_gamma_matrix_for_image,
    GeometricLossSeparate,
)
from utils import load_emulator

warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)


def run_prediction_loop(
    model,
    loader,
    mse_criterion,
    surrogate_criterion,
    audit_criterion,
    emulator,
    device,
    eval_mode,
    surrogate_weight,
    quantile_levels,
    pixel_size_km,
    trust_tau,
    physical_max_val,
    drizzle_threshold=0.1,
):
    model.eval()
    if emulator:
        emulator.eval()

    # Storage
    all_preds_gamma_analytic = []
    all_targets_gamma_analytic = []
    all_preds_phys = []
    all_targets_phys = []
    all_inputs_phys = []
    all_dems = []
    all_total_losses, all_mse_losses, all_surrogate_losses = [], [], []
    all_trust_scores = []

    # --- NEW: Storage for SAL and FSS ---
    all_sal_S = []
    all_sal_A = []
    all_sal_L = []
    all_fss = []  # Can store FSS at a specific key threshold, e.g., drizzle or higher

    audit_results = {
        "L1_phys_err": [],
        "L2_perc_err": [],
        "L3_intr_err": [],
        "consistency_gap": [],
    }

    print("\n--- Starting SR Inference ---")

    # Define FSS/SAL parameters
    # Adjust window_size for FSS (e.g., 10-20km -> ~5-10 pixels)
    FSS_WINDOW = 5
    # Threshold for SAL/FSS computation (metric_threshold)
    # Using the drizzle threshold is standard to define "rain area"
    METRIC_THRESHOLD = drizzle_threshold

    with torch.no_grad():
        for X, Y_true, Y_gamma_log_target in tqdm(loader, desc="Inference"):
            X, Y_true, Y_gamma_log_target = (
                X.to(device),
                Y_true.to(device),
                Y_gamma_log_target.to(device),
            )

            # --- 1. RECONSTRUCT PHYSICAL PRECIPITATION ---
            pred_X_raw = model(X)

            # A. Inverse Transform
            pred_X_pos = F.softplus(pred_X_raw, beta=5.0)
            pred_X_phys = torch.expm1(pred_X_pos * physical_max_val)
            Y_true_phys = torch.expm1(Y_true * physical_max_val)

            # B. Apply Sparsity / Drizzle Threshold
            pred_X_phys = pred_X_phys * (pred_X_phys > drizzle_threshold).float()
            Y_true_phys = Y_true_phys * (Y_true_phys > drizzle_threshold).float()

            # --- Data Extraction ---
            input_precip_batch = X[:, 0, :, :].cpu().numpy()
            dem_batch = X[:, 1, :, :].cpu().numpy()
            all_inputs_phys.append(input_precip_batch)
            all_dems.append(dem_batch)

            # --- 2. Compute Analytic Metrics (CPU) ---
            pred_X_np = pred_X_phys.cpu().numpy()
            Y_true_np = Y_true_phys.cpu().numpy()

            batch_gammas = []

            # Per-Sample CPU Loop
            for i in range(pred_X_np.shape[0]):
                p_img = pred_X_np[i, 0]
                t_img = Y_true_np[i, 0]

                # A. Gamma Matrix
                g = compute_gamma_matrix_for_image(
                    p_img, quantile_levels, pixel_size_km
                )
                batch_gammas.append(g)

                # B. SAL & FSS
                s_val, a_val, l_val = compute_sal(
                    p_img, t_img, threshold=METRIC_THRESHOLD
                )
                fss_val = compute_fss(
                    p_img, t_img, window_size=FSS_WINDOW, threshold=METRIC_THRESHOLD
                )

                all_sal_S.append(s_val)
                all_sal_A.append(a_val)
                all_sal_L.append(l_val)
                all_fss.append(fss_val)

            all_preds_gamma_analytic.append(np.array(batch_gammas))

            # Inverse transform targets for analytic comparison
            Y_gamma_phys_target = np.expm1(Y_gamma_log_target.cpu().numpy())
            all_targets_gamma_analytic.append(Y_gamma_phys_target)

            # --- 3. Calculate Losses ---
            loss_mse = mse_criterion(pred_X_phys, Y_true_phys).mean(dim=(1, 2, 3))
            loss_surr = torch.zeros_like(loss_mse)
            trust_vec = torch.ones_like(loss_mse)

            if emulator:
                # Trust Calculation
                gamma_true_phys = emulator(Y_true_phys)
                gamma_true_log_rec = torch.log1p(gamma_true_phys)

                emu_error_matrix = F.mse_loss(
                    gamma_true_log_rec, Y_gamma_log_target, reduction="none"
                )
                emu_error = emu_error_matrix.view(emu_error_matrix.size(0), -1).mean(
                    dim=1
                )
                trust_vec = torch.exp(-trust_tau * emu_error)

                # Surrogate Loss
                if eval_mode != "none" and surrogate_criterion:
                    pred_gamma_phys = emulator(pred_X_phys)
                    pred_gamma_log = torch.log1p(pred_gamma_phys)
                    loss_surr = surrogate_criterion(pred_gamma_log, Y_gamma_log_target)

            if eval_mode == "train":
                total_loss = loss_mse + (surrogate_weight * trust_vec * loss_surr)
            else:
                total_loss = loss_mse

            # --- 4. Emulator Consistency Audit ---
            if emulator and audit_criterion:
                gamma_pred_phys = emulator(pred_X_phys)
                gamma_true_phys = emulator(Y_true_phys)

                gamma_pred_log = torch.log1p(gamma_pred_phys)
                gamma_true_log = torch.log1p(gamma_true_phys)

                l1 = audit_criterion(gamma_pred_log, Y_gamma_log_target)
                l2 = audit_criterion(gamma_pred_log, gamma_true_log)
                l3 = audit_criterion(gamma_true_log, Y_gamma_log_target)

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
            all_total_losses.append(total_loss.cpu().numpy())
            all_mse_losses.append(loss_mse.cpu().numpy())
            all_surrogate_losses.append(loss_surr.cpu().numpy())
            all_trust_scores.append(trust_vec.cpu().numpy())

    # Concatenate Results
    all_preds_gamma_analytic = np.concatenate(all_preds_gamma_analytic, axis=0)
    all_targets_gamma_analytic = np.concatenate(all_targets_gamma_analytic, axis=0)

    all_preds_phys = np.concatenate(all_preds_phys, axis=0)
    all_targets_phys = np.concatenate(all_targets_phys, axis=0)
    all_inputs_phys = np.concatenate(all_inputs_phys, axis=0)
    all_dems = np.concatenate(all_dems, axis=0)

    all_total_losses = np.concatenate(all_total_losses, axis=0)
    all_mse_losses = np.concatenate(all_mse_losses, axis=0)
    all_surrogate_losses = np.concatenate(all_surrogate_losses, axis=0)
    all_trust_scores = np.concatenate(all_trust_scores, axis=0)

    # NEW: Convert lists to numpy arrays
    all_sal_S = np.array(all_sal_S)
    all_sal_A = np.array(all_sal_A)
    all_sal_L = np.array(all_sal_L)
    all_fss = np.array(all_fss)

    for k in audit_results:
        audit_results[k] = np.concatenate(audit_results[k], axis=0)

    print(f"Inference complete. Processed {len(all_total_losses)} samples.")

    # Return expanded tuple
    return (
        all_preds_gamma_analytic,
        all_targets_gamma_analytic,
        all_preds_phys,
        all_targets_phys,
        all_inputs_phys,
        all_dems,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        all_trust_scores,
        audit_results,
        all_sal_S,
        all_sal_A,
        all_sal_L,
        all_fss,
    )


def main(run_dir):
    config, device = io_lib.setup_evaluation(run_dir)
    model = io_lib.load_sr_model(config, device, run_dir)
    dem_stats = io_lib.load_dem_stats(config)
    test_loader = io_lib.load_data(config, dem_stats)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
    EVAL_MODE = config.get("EVAL_MODE", "none")
    SURROGATE_LOSS_WEIGHT = config.get("METRIC_LOSS_WEIGHT", 0.1)
    EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)
    TRUST_TAU = config.get("TRUST_TAU", 2.0)

    # --- LOAD SCALER ---
    scaler_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_transformed_precip_max_val.npy"
    )
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(
            config["PREPROCESSED_DATA_DIR"], "train", "precip_max_val.npy"
        )

    if os.path.exists(scaler_path):
        PHYSICAL_MAX_VAL = float(np.load(scaler_path))
        print(f"Loaded Physical Max Scaler: {PHYSICAL_MAX_VAL:.4f}")
    else:
        print("Warning: Scaler not found. Defaulting to 1.0")
        PHYSICAL_MAX_VAL = 1.0

    mse_criterion = nn.L1Loss(reduction="none")

    # Surrogate (Training) Criterion
    surrogate_criterion = None
    if EVAL_MODE != "none":
        S_inv_tensors = io_lib.load_s_inv(config, dem_stats, device)
        surrogate_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
            device
        )

    # Audit Criterion
    emulator = None
    audit_criterion = None
    if EMULATOR_CHECKPOINT_PATH and os.path.exists(EMULATOR_CHECKPOINT_PATH):
        print(f"Loading emulator for audit: {EMULATOR_CHECKPOINT_PATH}")
        emulator = load_emulator(EMULATOR_CHECKPOINT_PATH, config, device)
        S_inv_tensors = io_lib.load_s_inv(config, dem_stats, device)
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
        all_dems,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        all_trust_scores,
        audit_results,
        all_sal_S,  # New
        all_sal_A,  # New
        all_sal_L,  # New
        all_fss,  # New
    ) = run_prediction_loop(
        model,
        test_loader,
        mse_criterion,
        surrogate_criterion,
        audit_criterion,
        emulator,
        device,
        EVAL_MODE,
        SURROGATE_LOSS_WEIGHT,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
        TRUST_TAU,
        physical_max_val=PHYSICAL_MAX_VAL,
        drizzle_threshold=config.get("DRIZZLE_THRESHOLD", 0.1),
    )

    # --- Create DataFrame ---
    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_gamma_analytic,
        all_targets_gamma_analytic,
        all_inputs_phys,
        all_targets_phys,
        all_preds_phys,
        all_dems,
        all_total_losses,
        all_mse_losses,
        all_surrogate_losses,
        QUANTILE_LEVELS,
        PIXEL_SIZE_KM,
    )

    # Add Audit & Trust
    metrics_df["L1_Physical_Error"] = audit_results["L1_phys_err"]
    metrics_df["Consistency_Flag"] = audit_results["consistency_gap"]
    metrics_df["Trust_Score"] = all_trust_scores

    # --- NEW: Add SAL and FSS to DataFrame ---
    metrics_df["SAL_S"] = all_sal_S
    metrics_df["SAL_A"] = all_sal_A
    metrics_df["SAL_L"] = all_sal_L
    metrics_df["FSS"] = all_fss

    # Metrics
    group_metrics = metrics_lib.calculate_grouped_metrics(metrics_df)
    per_feature_gamma_metrics = metrics_lib.calculate_per_feature_gamma_metrics(
        metrics_df, QUANTILE_LEVELS
    )

    # Save
    io_lib.save_metrics_text(run_dir, group_metrics, per_feature_gamma_metrics)
    io_lib.save_metrics_npz(run_dir, metrics_df, per_feature_gamma_metrics)

    # Print Summary
    print("\n" + "=" * 40)
    print("       SCIENTIFIC METRICS SUMMARY        ")
    print("=" * 40)
    # Using nanmean to handle potential NaNs in SAL calculation for empty fields
    print(f"Mean FSS:   {np.nanmean(metrics_df['FSS']):.4f}")
    print(f"Mean SAL_S: {np.nanmean(metrics_df['SAL_S']):.4f}")
    print(f"Mean SAL_A: {np.nanmean(metrics_df['SAL_A']):.4f}")
    print(f"Mean SAL_L: {np.nanmean(metrics_df['SAL_L']):.4f}")

    if emulator:
        pass_rate = (metrics_df["Consistency_Flag"] < 0.3).mean() * 100
        avg_trust = metrics_df["Trust_Score"].mean()
        print("-" * 40)
        print(f"Pass Rate (Flag < 0.3): {pass_rate:.2f}%")
        print(f"Avg Trust Score:        {avg_trust:.4f}")
        print("=" * 40 + "\n")

    # Plots
    plotting_lib.plot_sample_comparisons_fixed(
        metrics_df, QUANTILE_LEVELS, run_dir, n_samples=5
    )
    plotting_lib.plot_per_feature_matrices(per_feature_gamma_metrics, run_dir)
    plotting_lib.plot_gamma_mean_std_by_quantile(
        metrics_df, group_metrics, QUANTILE_LEVELS, run_dir
    )
    plotting_lib.plot_metric_distributions(metrics_df, run_dir)

    # NEW: Trust Analysis
    plotting_lib.plot_trust_analysis(metrics_df, run_dir)

    log_file_path = os.path.join(run_dir, "sr_training_log.csv")
    plotting_lib.plot_training_log(log_file_path, run_dir, config)

    print("\nEvaluation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    main(args.run_dir)
