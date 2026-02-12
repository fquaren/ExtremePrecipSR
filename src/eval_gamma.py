import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import our new refactored libraries
import io_lib
import metrics_lib

# import plotting_lib

# Import your existing modules
from loss import TotalErrorMetric, GeometricLossSeparate

# Import architectures
from gamma_predictors_v5 import BaselineCNN, IsometricCNN, ConstrainedIsometricCNN
from models_fno import ProbabilisticFNO

# ==========================================
# Helper: Model Loading
# ==========================================


def load_model_refactored(config, device, run_dir, architecture_type, scaler_val):
    """
    Loads the specific architecture (Baseline, Isometric, Constrained, FNO)
    and restores weights. Correctly passes normalization constants.
    """
    print(f"\nLoading Model Architecture: {architecture_type}")

    # Extract config parameters
    PATCH_SIZE = config["PATCH_SIZE"]
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
    N_QUANTILES = len(config["QUANTILE_LEVELS"])
    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)

    # Instantiate the correct class
    if architecture_type == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Isometric":
        model = IsometricCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Constrained":
        # Pass the scaler_val retrieved from setup
        model = ConstrainedIsometricCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
            max_input_val=scaler_val,
        )
    elif architecture_type == "FNO":
        print("Initializing Probabilistic FNO...")
        model = ProbabilisticFNO(n_quantiles=N_QUANTILES, modes=12, width=32)
    else:
        raise ValueError(
            f"Unknown architecture: {architecture_type}. Choose: Baseline, Isometric, Constrained, FNO"
        )

    model = model.to(device)

    # Load Weights
    checkpoint_path = os.path.join(run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Restoring weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    model.eval()
    return model


# ==========================================
# Analysis Helpers
# ==========================================


def generate_saliency_samples(model, loader, device, n_examples=10):
    """
    Selects specific examples and computes gradients.
    Handles Probabilistic FNO output tuple (mu, var).
    """
    print("\nGenerating Saliency Samples...")
    model.eval()
    samples = []
    found_dry = False
    found_wet = False
    count = 0

    for input_data, _, _, _ in loader:
        if count >= n_examples:
            break

        input_data = input_data.to(device)
        input_data.requires_grad_(True)

        # Forward
        output = model(input_data)

        # [HANDLE FNO TUPLE]
        if isinstance(output, tuple):
            output = output[0]  # Take Mean (mu) for saliency

        # Target: Sum of predicted Area (Index 0)
        target_scalar = output[:, 0, :].sum()

        grad = torch.autograd.grad(target_scalar, input_data, create_graph=False)[0]

        inp_np = input_data.detach().cpu().numpy()
        grad_np = grad.detach().cpu().numpy()

        for b in range(input_data.shape[0]):
            if len(samples) >= n_examples:
                break

            total_rain = inp_np[b].sum()
            img = inp_np[b, 0]
            g = grad_np[b, 0]

            label = ""
            if total_rain < 0.1 and not found_dry:
                label = "Dry Input"
                found_dry = True
            elif total_rain > 1000 and not found_wet:
                label = "Large Storm"
                found_wet = True
            elif len(samples) < n_examples:
                label = f"Sample {len(samples)}"

            if label:
                samples.append((img, g, label))

        model.zero_grad()
        count += 1

    return samples


def compute_jacobian_stats(model, loader, device, n_samples=100):
    """
    Computes Gradient Norms.
    Handles Probabilistic FNO output tuple (mu, var).
    """
    print(f"\nComputing Jacobian Spectrum on subset ({n_samples} samples)...")
    model.eval()
    norms = {"Area": [], "Perimeter": [], "CC": []}
    count = 0

    for input_data, _, _, _ in loader:
        if count >= n_samples:
            break

        input_data = input_data.to(device)
        input_data.requires_grad_(True)

        output = model(input_data)

        # [HANDLE FNO TUPLE]
        if isinstance(output, tuple):
            output = output[0]  # Use Mean for gradient stability check

        n_quantiles = output.shape[2]
        mid_idx = n_quantiles // 2

        # 1. Area Gradient
        target_A = output[:, 0, mid_idx].sum()
        grad_A = torch.autograd.grad(
            target_A, input_data, retain_graph=True, create_graph=False
        )[0]
        norm_A = grad_A.view(grad_A.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        norms["Area"].extend(norm_A)

        # 2. Perimeter Gradient
        target_P = output[:, 1, mid_idx].sum()
        grad_P = torch.autograd.grad(
            target_P, input_data, retain_graph=True, create_graph=False
        )[0]
        norm_P = grad_P.view(grad_P.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        norms["Perimeter"].extend(norm_P)

        # 3. CC Gradient
        target_CC = output[:, 2, mid_idx].sum()
        grad_CC = torch.autograd.grad(
            target_CC, input_data, retain_graph=False, create_graph=False
        )[0]
        norm_CC = (
            grad_CC.view(grad_CC.size(0), -1).norm(p=2, dim=1).detach().cpu().numpy()
        )
        norms["CC"].extend(norm_CC)

        model.zero_grad()
        input_data.grad = None
        count += input_data.size(0)

    return norms


def run_prediction_loop(model, loader, eval_metric, geom_metric_per_sample, device):
    """
    Standard inference loop.
    Handles Probabilistic FNO output tuple (mu, var).
    """
    model.eval()
    all_preds_phys, all_targets_phys = [], []
    all_original_images, all_total_losses = [], []
    all_geom_losses = []

    with torch.no_grad():
        for (
            input_data,
            log_target_gamma,
            original_log_precip,
            target_gamma_phys,
        ) in tqdm(loader, desc="Inference"):
            input_data = input_data.to(device)
            log_target_gamma = log_target_gamma.to(device)
            target_gamma_phys = target_gamma_phys.to(device)

            # Forward
            output = model(input_data)

            # [HANDLE FNO TUPLE]
            if isinstance(output, tuple):
                # FNO returns (Mean, Variance).
                # For standard metrics (RMSE, etc.), we evaluate the MEAN prediction.
                predicted_gamma_phys = output[0]
            else:
                predicted_gamma_phys = output

            # Loss Calculation
            # Eval metric expects Log Space inputs
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            per_sample_total_losses = eval_metric(
                input_data, predicted_gamma_log, log_target_gamma
            )

            # Geometric metric expects Phys Space inputs
            per_sample_geom_losses = geom_metric_per_sample(
                predicted_gamma_phys, target_gamma_phys
            )

            # original_log_precip is in log space
            original_precip = torch.expm1(original_log_precip)

            # Store
            all_total_losses.append(per_sample_total_losses.cpu().numpy())
            all_geom_losses.append(per_sample_geom_losses.cpu().numpy())
            all_preds_phys.append(predicted_gamma_phys.cpu().numpy())
            all_targets_phys.append(target_gamma_phys.cpu().numpy())
            all_original_images.append(original_precip.squeeze(1).cpu().numpy())

    return (
        np.concatenate(all_preds_phys, axis=0),
        np.concatenate(all_targets_phys, axis=0),
        np.concatenate(all_original_images, axis=0),
        np.concatenate(all_total_losses, axis=0),
        np.concatenate(all_geom_losses, axis=0),
    )


# ==========================================
# Main Execution
# ==========================================


def main(run_dir, architecture_type):

    # --- 1. Setup ---
    # Unpack scaler_val from setup
    config, device, scaler_val = io_lib.setup_evaluation(run_dir)

    # Pass scaler_val to model loader
    model = load_model_refactored(
        config, device, run_dir, architecture_type, scaler_val
    )

    # Pass scaler_val to data loaders
    test_loader = io_lib.load_data(config, scaler_val)
    S_inv_tensors = io_lib.load_s_inv(config, device, scaler_val)

    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]

    # --- 2. Initialize Metrics ---
    evaluation_metric = TotalErrorMetric(
        quantile_levels=QUANTILE_LEVELS, config=config
    ).to(device)
    geometric_metric_per_sample = GeometricLossSeparate(
        S_inv_tensors, reduction="none"
    ).to(device)

    # --- 3. Run Inference ---
    results = run_prediction_loop(
        model, test_loader, evaluation_metric, geometric_metric_per_sample, device
    )
    (
        all_preds_phys,
        all_targets_phys,
        all_original_images,
        all_total_losses,
        all_geom_losses,
    ) = results

    # --- 4. Gradient Analysis (Jacobian & Saliency) ---
    # jacobian_norms = compute_jacobian_stats(model, test_loader, device, n_samples=200)
    # saliency_data = generate_saliency_samples(model, test_loader, device, n_examples=5)

    # --- 5. Compute DataFrame Metrics ---
    metrics_df = metrics_lib.create_metrics_dataframe(
        all_preds_phys,
        all_targets_phys,
        all_original_images,
        all_total_losses,
        all_geom_losses,
    )

    group_metrics_sample_wise = metrics_lib.calculate_grouped_metrics(metrics_df)
    group_metrics_global = metrics_lib.calculate_global_group_metrics(
        metrics_df, all_preds_phys, all_targets_phys
    )
    per_feature_metrics = metrics_lib.calculate_per_feature_metrics(
        all_preds_phys, all_targets_phys, QUANTILE_LEVELS
    )

    # --- 6. Save Results ---
    io_lib.save_metrics_text(
        run_dir, group_metrics_global, group_metrics_sample_wise, per_feature_metrics
    )

    # --- 7. Plotting ---
    # print("\n--- Generating Plots ---")
    # plotting_lib.plot_isoperimetric_check(all_preds_phys, run_dir)
    # plotting_lib.plot_dry_input_error(all_preds_phys, all_original_images, run_dir)
    # plotting_lib.plot_saliency_maps(saliency_data, run_dir)
    # plotting_lib.plot_jacobian_spectrum(
    #     jacobian_data=jacobian_norms, output_dir=run_dir
    # )
    # plotting_lib.plot_per_feature_matrices(
    #     per_feature_metrics=per_feature_metrics, output_dir=run_dir
    # )
    # plotting_lib.plot_sample_comparisons(
    #     metrics_df=metrics_df,
    #     quantiles=QUANTILE_LEVELS,
    #     output_dir=run_dir,
    #     n_samples=15,
    # )
    # plotting_lib.plot_metric_distributions(metrics_df=metrics_df, output_dir=run_dir)
    # plotting_lib.plot_qq_summary(metrics_df=metrics_df, output_dir=run_dir)
    # plotting_lib.plot_gamma_mean_std_by_quantile(
    #     metrics_df=metrics_df,
    #     group_metrics=group_metrics_sample_wise,
    #     quantiles=QUANTILE_LEVELS,
    #     output_dir=run_dir,
    # )

    # Training Log
    # log_file_path = os.path.join(run_dir, "training_log.csv")
    # plotting_lib.plot_training_log(log_path=log_file_path, output_dir=run_dir)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained emulator (Baseline/Isometric/Constrained/FNO)."
    )
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to experiment run directory."
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["Baseline", "Isometric", "Constrained", "FNO"],
        help="Architecture type used in training.",
    )
    args = parser.parse_args()

    main(args.run_dir, args.arch)
