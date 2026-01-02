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


def compute_rapsd(field_2d):
    """
    Computes Radially Averaged Power Spectral Density (RAPSD).
    field_2d: 2D numpy array [H, W]
    """
    H, W = field_2d.shape
    # FFT
    fft_output = np.fft.fftshift(np.fft.fft2(field_2d))
    psd = np.abs(fft_output) ** 2

    # Radial Profile setup
    center = (H // 2, W // 2)
    y, x = np.indices((H, W))
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    # Average PSD over rings of constant radius
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    # Avoid division by zero
    radial_profile = tbin / np.maximum(nr, 1)

    return radial_profile


def compute_spectral_distance(pred_phys, target_phys):
    """
    Computes Log-Spectral Distance between prediction and target.
    Input: Tensors or Arrays [H, W] (Single Image)
    Returns: Scalar float
    """
    if isinstance(pred_phys, torch.Tensor):
        pred_phys = pred_phys.cpu().numpy()
    if isinstance(target_phys, torch.Tensor):
        target_phys = target_phys.cpu().numpy()

    p_spec = compute_rapsd(pred_phys)
    t_spec = compute_rapsd(target_phys)

    # Log-Spectral Distance (LSD)
    # We use log10 and add epsilon to avoid log(0)
    p_spec_log = np.log10(p_spec + 1e-8)
    t_spec_log = np.log10(t_spec + 1e-8)

    # Calculate MSE in Log-Frequency domain
    dist = np.mean((p_spec_log - t_spec_log) ** 2)
    return dist


def run_prediction_loop(
    model,
    loader,
    mae_criterion,
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

    # --- NEW: Perception Metric Storage ---
    all_spectral_dists = []

    # Spatial Metrics
    all_sal_S, all_sal_A, all_sal_L, all_fss = [], [], [], []

    audit_results = {
        "L1_phys_err": [],
        "L2_perc_err": [],
        "L3_intr_err": [],
        "consistency_gap": [],
    }

    print("\n--- Starting SR Inference & Spectral Audit ---")

    FSS_WINDOW = 5
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

            # Loop over batch
            for i in range(pred_X_np.shape[0]):
                p_img = pred_X_np[i, 0]
                t_img = Y_true_np[i, 0]

                # Gamma
                g = compute_gamma_matrix_for_image(
                    p_img, quantile_levels, pixel_size_km
                )
                batch_gammas.append(g)

                # Standard Spatial Metrics
                s_val, a_val, l_val = compute_sal(
                    p_img, t_img, threshold=METRIC_THRESHOLD
                )
                fss_val = compute_fss(
                    p_img, t_img, window_size=FSS_WINDOW, threshold=METRIC_THRESHOLD
                )

                # --- Spectral Distance (Perception) ---
                spec_dist = compute_spectral_distance(p_img, t_img)
                all_spectral_dists.append(spec_dist)

                all_sal_S.append(s_val)
                all_sal_A.append(a_val)
                all_sal_L.append(l_val)
                all_fss.append(fss_val)

            all_preds_gamma_analytic.append(np.array(batch_gammas))
            Y_gamma_phys_target = np.expm1(Y_gamma_log_target.cpu().numpy())
            all_targets_gamma_analytic.append(Y_gamma_phys_target)

            # --- 3. Calculate Losses ---
            loss_mse = mae_criterion(pred_X_phys, Y_true_phys).mean(dim=(1, 2, 3))
            loss_surr = torch.zeros_like(loss_mse)
            trust_vec = torch.ones_like(loss_mse)

            if emulator:
                gamma_true_phys = emulator(Y_true_phys)
                gamma_true_log_rec = torch.log1p(gamma_true_phys)

                emu_error_matrix = F.mse_loss(
                    gamma_true_log_rec, Y_gamma_log_target, reduction="none"
                )
                emu_error = emu_error_matrix.view(emu_error_matrix.size(0), -1).mean(
                    dim=1
                )
                trust_vec = torch.exp(-trust_tau * emu_error)

                if eval_mode != "none" and surrogate_criterion:
                    pred_gamma_phys = emulator(pred_X_phys)
                    pred_gamma_log = torch.log1p(pred_gamma_phys)
                    loss_surr = surrogate_criterion(pred_gamma_log, Y_gamma_log_target)

            if eval_mode == "train":
                total_loss = loss_mse + (surrogate_weight * trust_vec * loss_surr)
            else:
                total_loss = loss_mse

            # Audit
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

            all_preds_phys.append(pred_X_phys.squeeze(1).cpu().numpy())
            all_targets_phys.append(Y_true_phys.squeeze(1).cpu().numpy())
            all_total_losses.append(total_loss.cpu().numpy())
            all_mse_losses.append(loss_mse.cpu().numpy())
            all_surrogate_losses.append(loss_surr.cpu().numpy())
            all_trust_scores.append(trust_vec.cpu().numpy())

    # Concatenation
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

    # Metrics to Arrays
    all_spectral_dists = np.array(all_spectral_dists)
    all_sal_S = np.array(all_sal_S)
    all_sal_A = np.array(all_sal_A)
    all_sal_L = np.array(all_sal_L)
    all_fss = np.array(all_fss)

    for k in audit_results:
        audit_results[k] = np.concatenate(audit_results[k], axis=0)

    print(f"Inference complete. Processed {len(all_total_losses)} samples.")

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
        all_spectral_dists,
    )


def evaluate_single_model(
    model,
    model_name,
    run_dir,
    output_subdir,
    test_loader,
    config,
    device,
    mae_criterion,
    surrogate_criterion,
    audit_criterion,
    emulator,
    PHYSICAL_MAX_VAL,
):
    """
    Helper to run evaluation for a single model checkpoint and save results to a subdirectory.
    """
    print(f"\n" + "=" * 50)
    print(f"   EVALUATING MODEL: {model_name} -> {output_subdir}")
    print("=" * 50)

    # Create subdirectory
    full_output_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(full_output_dir, exist_ok=True)

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
        all_sal_S,
        all_sal_A,
        all_sal_L,
        all_fss,
        all_spectral_dists,
    ) = run_prediction_loop(
        model,
        test_loader,
        mae_criterion,
        surrogate_criterion,
        audit_criterion,
        emulator,
        device,
        config.get("EVAL_MODE", "none"),
        config.get("METRIC_LOSS_WEIGHT", 0.1),
        config["QUANTILE_LEVELS"],
        config.get("PIXEL_SIZE_KM", 2.0),
        config.get("TRUST_TAU", 2.0),
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
        all_spectral_dists,
        config["QUANTILE_LEVELS"],
        config.get("PIXEL_SIZE_KM", 2.0),
    )

    metrics_df["L1_Physical_Error"] = audit_results["L1_phys_err"]
    metrics_df["Consistency_Flag"] = audit_results["consistency_gap"]
    metrics_df["Trust_Score"] = all_trust_scores
    metrics_df["SAL_S"] = all_sal_S
    metrics_df["SAL_A"] = all_sal_A
    metrics_df["SAL_L"] = all_sal_L
    metrics_df["FSS"] = all_fss

    group_metrics = metrics_lib.calculate_grouped_metrics(metrics_df)
    per_feature_gamma_metrics = metrics_lib.calculate_per_feature_gamma_metrics(
        metrics_df, config["QUANTILE_LEVELS"]
    )

    # Save to the specific subdirectory
    io_lib.save_metrics_text(full_output_dir, group_metrics, per_feature_gamma_metrics)
    io_lib.save_metrics_npz(full_output_dir, metrics_df, per_feature_gamma_metrics)

    print("\n" + "=" * 40)
    print(f"       SCIENTIFIC METRICS SUMMARY ({model_name})        ")
    print("=" * 40)
    print(
        f"Mean Spectral Dist (LSD): {np.nanmean(metrics_df['spectral_dist']):.4f} (Lower = Sharper)"
    )
    print(
        f"Mean MAE:                 {np.nanmean(metrics_df['mse_loss']):.4f} (Physical)"
    )
    print(f"Mean FSS:                 {np.nanmean(metrics_df['FSS']):.4f}")

    # Pass full_output_dir as run_dir to plotting lib so it saves in the right place
    plotting_lib.plot_sample_comparisons_fixed(
        metrics_df, config["QUANTILE_LEVELS"], full_output_dir, n_samples=5
    )
    plotting_lib.plot_per_feature_matrices(per_feature_gamma_metrics, full_output_dir)
    plotting_lib.plot_metric_distributions(metrics_df, full_output_dir)
    plotting_lib.plot_perception_distortion(metrics_df, full_output_dir)


def main(run_dir):
    config, device = io_lib.setup_evaluation(run_dir)

    # 1. Load Data & Constants (Shared)
    dem_stats = io_lib.load_dem_stats(config)
    test_loader = io_lib.load_data(config, dem_stats)

    EVAL_MODE = config.get("EVAL_MODE", "none")
    EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)

    scaler_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "log_transformed_precip_max_val.npy"
    )
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(
            config["PREPROCESSED_DATA_DIR"], "train", "precip_max_val.npy"
        )

    if os.path.exists(scaler_path):
        PHYSICAL_MAX_VAL = float(np.load(scaler_path))
    else:
        PHYSICAL_MAX_VAL = 1.0

    mae_criterion = nn.L1Loss(reduction="none")

    surrogate_criterion = None
    if EVAL_MODE != "none":
        S_inv_tensors = io_lib.load_s_inv(config, dem_stats, device)
        surrogate_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
            device
        )

    emulator = None
    audit_criterion = None
    if EMULATOR_CHECKPOINT_PATH and os.path.exists(EMULATOR_CHECKPOINT_PATH):
        emulator = load_emulator(EMULATOR_CHECKPOINT_PATH, config, device)
        S_inv_tensors = io_lib.load_s_inv(config, dem_stats, device)
        audit_criterion = GeometricLossSeparate(S_inv_tensors, reduction="none").to(
            device
        )

    # 2. Define Models to Evaluate
    models_to_evaluate = [
        ("Best", "best_sr_model.pth", "eval_best"),
        ("Last", "last_sr_model.pth", "eval_last"),
    ]

    # 3. Loop through models
    for model_friendly_name, model_file, output_subdir in models_to_evaluate:
        # Load Model
        model = io_lib.load_sr_model(config, device, run_dir, model_filename=model_file)

        if model is None:
            print(f"Skipping evaluation for {model_friendly_name} (File not found).")
            continue

        evaluate_single_model(
            model,
            model_friendly_name,
            run_dir,
            output_subdir,
            test_loader,
            config,
            device,
            mae_criterion,
            surrogate_criterion,
            audit_criterion,
            emulator,
            PHYSICAL_MAX_VAL,
        )

    # 4. Shared Plots (Training History)
    # This only needs to happen once as it comes from the log file, not the model weights
    log_file_path = os.path.join(run_dir, "training_log.csv")
    if os.path.exists(log_file_path):
        print(f"\nPlotting Training History from {log_file_path}...")
        plotting_lib.plot_training_log(log_file_path, run_dir, config)
    else:
        print(
            f"Warning: Training log not found at {log_file_path}. Skipping history plot."
        )

    print("\nEvaluation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    main(args.run_dir)
