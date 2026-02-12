import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import copy

# Import your model loaders
from gamma_predictors_v5 import BaselineCNN, IsometricCNN, ConstrainedIsometricCNN
from models_fno import ProbabilisticFNO


def set_seed(seed):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_inversion(run_dir, architecture_type):
    """Loads config, sets up device, and loads the frozen emulator."""
    print(f"--- Setting up Inversion Test for: {architecture_type} ---")

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    PATCH_SIZE = config["PATCH_SIZE"]
    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
    N_QUANTILES = len(config["QUANTILE_LEVELS"])
    QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
    PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)

    if architecture_type == "Baseline":
        model = BaselineCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Isometric":
        model = IsometricCNN(n_quantiles=N_QUANTILES, input_shape=INPUT_SHAPE)
    elif architecture_type == "Constrained":
        model = ConstrainedIsometricCNN(
            n_quantiles=N_QUANTILES,
            input_shape=INPUT_SHAPE,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        )
    elif architecture_type == "FNO":
        model = ProbabilisticFNO(n_quantiles=N_QUANTILES, modes=12, width=32)
    else:
        raise ValueError(f"Unknown architecture: {architecture_type}")

    checkpoint_path = os.path.join(run_dir, "best_model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, config, device


def create_target_vector(mode, quantile_levels, device):
    """
    Creates synthetic physical targets y_target [1, 3, Q].
    Strictly enforces 0 Area/Perimeter for thresholds > 150 mm/h.
    """
    if isinstance(quantile_levels, list):
        quantile_levels = torch.tensor(quantile_levels, device=device)

    n_quantiles = len(quantile_levels)
    target = torch.zeros(1, 3, n_quantiles, device=device)

    # 1. Define Scale (Area/Perimeter in PIXELS)
    if mode == "large_storm":
        # Logspace Area: 10,000 px -> 10 px
        area_curve = torch.logspace(
            math.log10(10000), math.log10(2500), n_quantiles, device=device
        )
        perim_curve = torch.logspace(
            math.log10(3500), math.log10(1000), n_quantiles, device=device
        )
        euler_curve = torch.linspace(1, 20, n_quantiles, device=device)
    elif mode == "small_storm":
        area_curve = torch.linspace(2000, 10, n_quantiles, device=device)
        perim_curve = torch.linspace(600, 10, n_quantiles, device=device)
        euler_curve = torch.zeros(n_quantiles, device=device)
        euler_curve[0] = 1
    elif mode == "no_storm":
        area_curve = torch.zeros(n_quantiles, device=device)
        perim_curve = torch.zeros(n_quantiles, device=device)
        euler_curve = torch.zeros(n_quantiles, device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    target[0, 0, :] = area_curve
    target[0, 1, :] = perim_curve
    target[0, 2, :] = euler_curve

    # 2. Enforce Max Precipitation Cutoff
    max_precip = 150.0
    mask_zero = quantile_levels > max_precip

    target[0, 0, mask_zero] = 0.0
    target[0, 1, mask_zero] = 0.0
    target[0, 2, mask_zero] = 0.0

    return torch.log1p(target)


def plot_target_vector(run_id, log_target, quantile_levels, mode, output_dir):
    """
    Visualizes the target Minkowski Functionals against physical thresholds
    stacked vertically. Saves as PDF.
    """
    print(f"Plotting target vector for {mode}...")

    # Unlog the target to get physical units
    phys_target = torch.expm1(log_target).cpu().numpy()[0]  # [3, Q]
    thresholds = np.array(quantile_levels)

    # Changed layout to 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

    # 1. Area
    axes[0].plot(thresholds, phys_target[0], "k-o", label="Target Area")
    axes[0].set_title("Minkowski 0: Area")
    axes[0].set_ylabel("Area (pixels)")
    axes[0].grid(True, alpha=0.3)
    # Highlight the cutoff
    axes[0].legend()

    # 2. Perimeter
    axes[1].plot(thresholds, phys_target[1], "k-o", label="Target Perimeter")
    axes[1].set_title("Minkowski 1: Perimeter")
    axes[1].set_ylabel("Perimeter (pixels)")
    axes[1].grid(True, alpha=0.3)

    # 3. Euler
    axes[2].plot(thresholds, phys_target[2], "k-o", label="Target Euler")
    axes[2].set_title("Minkowski 2: Euler Characteristic")
    axes[2].set_xlabel("Precipitation Threshold (mm/h)")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)

    # plt.suptitle(f"Target Constraints: {mode}", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir, "inversion_test", f"target_vector_{mode}_{run_id}.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def run_inversion(model, log_target, config, device, steps=200, lr=0.1):
    """
    Optimization loop with UNet-like initialization.
    """
    PATCH_SIZE = config["PATCH_SIZE"]

    # 1. Start with a "Clear Sky" bias
    low_res_size = PATCH_SIZE // 4
    low_res_noise = torch.randn(1, 1, low_res_size, low_res_size, device=device)

    # Upsample to full size
    input_noise = F.interpolate(
        low_res_noise, size=PATCH_SIZE, mode="bilinear", align_corners=False
    )

    # Shift mean to -6.0.
    input_param = (input_noise * 2.0) - 6.0

    # Capture the physical initial state BEFORE optimization starts
    initial_image_phys = F.softplus(input_param).detach().cpu().numpy()[0, 0]

    input_param.requires_grad_(True)

    optimizer = torch.optim.Adam([input_param], lr=lr)
    history = []

    loss_stats = {"total": [], "mse": [], "tv": [], "l2": []}

    def smooth_gradients(grad_tensor):
        kernel = torch.tensor(
            [[[[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]]]
        ).to(device)
        return F.conv2d(grad_tensor, kernel, padding=1)

    iterator = tqdm(range(steps), desc="Dreaming")

    for i in iterator:
        optimizer.zero_grad()
        phys_input = F.softplus(input_param)
        output = model(phys_input)

        if isinstance(output, tuple):
            pred_phys = output[0]
        else:
            pred_phys = output

        pred_log = torch.log1p(pred_phys)
        mse_loss = F.mse_loss(pred_log, log_target)

        # TV Loss + L2 Decay
        tv_loss = torch.sum(
            torch.abs(phys_input[:, :, :, :-1] - phys_input[:, :, :, 1:])
        ) + torch.sum(torch.abs(phys_input[:, :, :-1, :] - phys_input[:, :, 1:, :]))
        l2_loss = torch.mean(phys_input**2)

        # Scaling factors
        tv_weight = 1e-5
        l2_weight = 1e-6

        total_loss = mse_loss + tv_weight * tv_loss + l2_weight * l2_loss

        total_loss.backward()

        # if input_param.grad is not None:
        #     input_param.grad = smooth_gradients(input_param.grad)

        optimizer.step()
        iterator.set_postfix(mse=mse_loss.item())

        loss_stats["total"].append(total_loss.item())
        loss_stats["mse"].append(mse_loss.item())
        loss_stats["tv"].append(tv_loss.item() * tv_weight)
        loss_stats["l2"].append(l2_loss.item() * l2_weight)

        if i % 50 == 0:
            history.append(phys_input.detach().cpu().numpy()[0, 0])

    final_image = F.softplus(input_param).detach().cpu().numpy()[0, 0]

    return initial_image_phys, final_image, history, loss_stats


def plot_inversion_results(
    run_id,
    initial_img_phys,
    final_img_phys,
    run_name,
    target_mode,
    output_dir,
    gt_img_phys=None,  # New optional argument
):
    """
    Visualizes Initial Noise vs Final Dreamt Storm vs Ground Truth (if available).
    """
    save_dir = os.path.join(output_dir, "inversion_test")
    os.makedirs(save_dir, exist_ok=True)

    # Determine layout
    has_gt = gt_img_phys is not None
    ncols = 3 if has_gt else 2
    figsize = (16, 5) if has_gt else (11, 5)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    # Prepare Colormap
    cmap = copy.copy(plt.get_cmap("cividis_r"))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    # 1. Initial Noise (Independent Scale)
    vmax_noise = max(np.max(initial_img_phys), 1e-6)
    im0 = axes[0].imshow(
        initial_img_phys, cmap=cmap, origin="lower", vmin=0, vmax=vmax_noise
    )
    axes[0].set_title("Initial Noise (Physical)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="mm/h")

    # Determine common scale for Prediction and GT
    vmax_storm = np.max(final_img_phys)
    if has_gt:
        vmax_storm = max(vmax_storm, np.max(gt_img_phys))

    # Ensure a reasonable minimum vmax to avoid black images on zero arrays
    vmax_storm = max(vmax_storm, 1.0)

    # 2. Final Result (Dreamed)
    im1 = axes[1].imshow(
        final_img_phys, cmap=cmap, origin="lower", vmin=0, vmax=vmax_storm
    )
    axes[1].set_title(f"{run_name}")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="mm/h")

    # 3. Ground Truth (Real Storm)
    if has_gt:
        im2 = axes[2].imshow(
            gt_img_phys, cmap=cmap, origin="lower", vmin=0, vmax=vmax_storm
        )
        axes[2].set_title("Ground Truth (Original)")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="mm/h")

    # plt.suptitle(f"Inversion: {run_name} | Target: {target_mode}", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            save_dir, f"inversion_NO_SMOOTHING_{target_mode}_{run_name}_{run_id}.pdf"
        )
    )
    plt.close()


def plot_loss_history(run_id, loss_stats, run_name, target_mode, output_dir):
    """
    Plots only the total optimization loss history. Saves as PDF.
    """
    save_dir = os.path.join(output_dir, "inversion_test")
    os.makedirs(save_dir, exist_ok=True)

    steps = range(len(loss_stats["total"]))

    plt.figure(figsize=(10, 5))

    # Plot Total Loss
    plt.plot(
        steps, loss_stats["total"], color="black", linewidth=1.5, label="Total Loss"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Total Loss (Log Scale)")
    plt.yscale("log")
    plt.title(f"Optimization History: {run_name} | {target_mode}")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir, f"history_NO_SMOOTHING_{target_mode}_{run_name}_{run_id}.pdf"
        )
    )
    plt.close()


def save_inversion_data(
    output_dir,
    run_id,
    run_name,
    target_mode,
    final_img,
    initial_noise,
    loss_stats,
    target_vector,
    gt_img=None,
):
    """
    Saves raw inversion results and stats to .npz for external plotting.
    """
    save_dir = os.path.join(output_dir, "inversion_test", "data")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"inversion_data_NO_SMOOTHING_{target_mode}_{run_name}_{run_id}.npz"
    filepath = os.path.join(save_dir, filename)

    # Flatten loss_stats dictionary (list -> array)
    # Keys become: 'total_loss', 'mse_loss', etc.
    data_dict = {f"{k}_loss": np.array(v) for k, v in loss_stats.items()}

    data_dict["final_image"] = final_img
    data_dict["initial_noise"] = initial_noise

    if torch.is_tensor(target_vector):
        data_dict["target_vector"] = target_vector.detach().cpu().numpy()
    else:
        data_dict["target_vector"] = target_vector

    # Use a safe sentinel if GT is missing
    if gt_img is not None:
        data_dict["ground_truth"] = gt_img
    else:
        data_dict["ground_truth"] = np.array([])  # Empty array

    np.savez(filepath, **data_dict)
    print(f"  -> Data saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["Baseline", "Isometric", "Constrained", "FNO"],
    )

    parser.add_argument("--target_idx", type=int, default=None)
    args = parser.parse_args()

    model, config, device = setup_inversion(args.run_dir, args.arch)

    target_idx = args.target_idx

    quantile_levels = config["QUANTILE_LEVELS"]
    preprocessed_data_dir = config.get("PREPROCESSED_DATA_DIR", "./preprocessed_data")

    # 1. Determine N, log_targets, and ground_truth
    if target_idx is None:
        N = 5
        print(f"Running inversion tests for {N} synthetic runs...")
        log_targets = [None] * N
        ground_truths = [None] * N  # No GT for synthetic
    else:
        # Wrap target_idx in list if it's a single integer for consistency
        # (Though argparse returns int, user logic implies we might want lists later)
        target_indices = [target_idx]
        print(f"Running inversion tests for predefined targets: {target_indices}...")

        # Load Targets
        log_targets_np = np.log1p(
            np.load(
                os.path.join(
                    preprocessed_data_dir, "test/gamma_targets_persistence.npz"
                ),
                mmap_mode="r",
            )["data"][target_indices]
        )
        # Fix dimensions: (N, 3, Q) -> list of (1, 3, Q)
        log_targets = [
            torch.tensor(t, device=device, dtype=torch.float32).unsqueeze(0)
            for t in log_targets_np
        ]

        # Load Ground Truth Images
        gt_path = os.path.join(preprocessed_data_dir, "test/physical_precip.npz")
        if os.path.exists(gt_path):
            print(f"Loading ground truth from {gt_path}...")
            # We load only the specific indices
            full_gt_data = np.load(gt_path, mmap_mode="r")["data"]
            ground_truths = []
            for idx in target_indices:
                img = full_gt_data[idx]
                if img.ndim == 3:
                    img = img[0]  # Handle (1, H, W)
                ground_truths.append(img)
        else:
            print(f"Warning: Ground truth file not found at {gt_path}")
            ground_truths = [None] * len(target_indices)

        N = len(target_indices)

    # 2. Initialize storage
    input_noises = [None] * N
    final_imgs = [None] * N
    history = [None] * N
    loss_histories = [None] * N

    for i in range(N):
        set_seed(42 + i)

        # Logic: If using real data, run only once. If synthetic, run 3 modes.
        modes_to_run = (
            ["large_storm", "small_storm", "no_storm"]
            if target_idx is None
            else [f"real_data_{target_idx}"]
        )

        for mode in modes_to_run:
            # 1. Create Target (only overrides if synthetic)
            if target_idx is None:
                log_targets[i] = create_target_vector(mode, quantile_levels, device)

            # 2. Plot Target
            plot_target_vector(i, log_targets[i], quantile_levels, mode, args.run_dir)

            # 3. Run Inversion
            res_noise, res_img, res_hist, res_stats = run_inversion(
                model, log_targets[i], config, device
            )

            # Store results
            input_noises[i] = res_noise
            final_imgs[i] = res_img
            history[i] = res_hist
            loss_histories[i] = res_stats

            # 4. Save Raw Data (New)
            save_inversion_data(
                output_dir=args.run_dir,
                run_id=i,
                run_name=args.arch,
                target_mode=mode,
                final_img=res_img,
                initial_noise=res_noise,
                loss_stats=res_stats,
                target_vector=log_targets[i],
                gt_img=ground_truths[i],  # Pass the correct GT from the list
            )

            # 5. Plot Results
            plot_inversion_results(
                run_id=i,
                initial_img_phys=res_noise,
                final_img_phys=res_img,
                run_name=args.arch,
                target_mode=mode,
                output_dir=args.run_dir,
                gt_img_phys=ground_truths[i],  # Pass the correct GT from the list
            )

            # 6. Plot History
            plot_loss_history(i, res_stats, args.arch, mode, args.run_dir)

    print("\nInversion test complete. Check 'inversion_test' folder in run directory.")
