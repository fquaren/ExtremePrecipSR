import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# Import your model loaders
from gamma_predictors_v5 import BaselineCNN, IsometricCNN, ConstrainedIsometricCNN
from models_fno import ProbabilisticFNO


def set_seed(seed):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_rapsd(image, fft_shift=True, pixel_size=2.0):
    """
    Computes the Radially Averaged Power Spectral Density (RAPSD) of a 2D image.
    Algorithm preserved exactly as requested.
    """
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    H, W = image.shape
    # Windowing
    w = np.hanning(H)
    window = np.outer(w, w)
    image_windowed = image * window
    scale_factor = 1.0 / np.mean(window**2)

    f_transform = np.fft.fft2(image_windowed)
    if fft_shift:
        f_transform = np.fft.fftshift(f_transform)

    magnitude_spectrum = (np.abs(f_transform) ** 2) / (H * W) ** 2
    magnitude_spectrum *= scale_factor

    y, x = np.indices((H, W))
    center = np.array([H // 2, W // 2])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    rapsd = tbin / np.maximum(nr, 1)

    freqs = np.fft.fftfreq(H, d=pixel_size)[: H // 2]
    freqs = freqs[freqs >= 0]
    limit = min(H, W) // 2
    return freqs[:limit], rapsd[:limit]


def setup_model(run_dir, architecture_type, device):
    """Loads config and frozen emulator."""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, config


def run_inversion_batched(model, log_targets, config, device, steps=200, lr=0.1):
    """
    Vectorized Optimization loop.
    Args:
        log_targets: Tensor of shape (B, 3, Q)
    Returns:
        final_images: Tensor of shape (B, H, W) on CPU
    """
    PATCH_SIZE = config["PATCH_SIZE"]
    batch_size = log_targets.shape[0]

    # 1. Initialization
    low_res_size = PATCH_SIZE // 4

    # Generate batch of noise
    low_res_noise = torch.randn(
        batch_size, 1, low_res_size, low_res_size, device=device
    )

    input_noise = F.interpolate(
        low_res_noise, size=PATCH_SIZE, mode="bilinear", align_corners=False
    )

    # Initialize param for the whole batch
    input_param = (input_noise * 2.0) - 6.0
    input_param.requires_grad_(True)

    optimizer = torch.optim.Adam([input_param], lr=lr)

    # Gradient smoothing kernel
    # smooth_kernel = torch.tensor(
    #     [[[[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]]]
    # ).to(device)

    for _ in range(steps):
        optimizer.zero_grad()
        phys_input = F.softplus(input_param)  # (B, 1, H, W)
        output = model(phys_input)

        if isinstance(output, tuple):
            pred_phys = output[0]
        else:
            pred_phys = output

        pred_log = torch.log1p(pred_phys)

        # MSE Loss: mean over all dims allows batch independence if gradients are separate
        mse_loss = F.mse_loss(pred_log, log_targets)

        # Regularization - Vectorized
        # We calculate per-sample TV, then take the mean over the batch to match MSE scaling
        tv_h = torch.abs(phys_input[:, :, :, :-1] - phys_input[:, :, :, 1:])
        tv_w = torch.abs(phys_input[:, :, :-1, :] - phys_input[:, :, 1:, :])

        # Sum over spatial dims (H, W), mean over batch (B) to keep lambda consistent
        tv_loss = (
            torch.sum(tv_h, dim=(1, 2, 3)) + torch.sum(tv_w, dim=(1, 2, 3))
        ).mean()

        l2_loss = torch.mean(phys_input**2)

        total_loss = mse_loss + (1e-5 * tv_loss) + (1e-6 * l2_loss)
        total_loss.backward()

        # if input_param.grad is not None:
        #     # Conv2d works natively on batches (B, C, H, W)
        #     input_param.grad = F.conv2d(input_param.grad, smooth_kernel, padding=1)

        optimizer.step()

    # Return physical space images, detached
    final_images = F.softplus(input_param).detach().cpu()  # (B, 1, H, W)
    return final_images.squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["Baseline", "Isometric", "Constrained"],
    )
    parser.add_argument("--output_name", type=str, default="rapsd_analysis_results.npz")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Optimization batch size"
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=0.1,
        help="Fraction of test set to use (0 < f <= 1.0). Default 0.1 (10%)",
    )
    args = parser.parse_args()

    # Basic input validation
    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be between 0.0 and 1.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = setup_model(args.run_dir, args.arch, device)

    preprocessed_data_dir = config.get("PREPROCESSED_DATA_DIR", "./preprocessed_data")

    # --- Load Test Data ---
    print("Loading test dataset...")

    target_path = os.path.join(
        preprocessed_data_dir, "test/gamma_targets_persistence.npz"
    )
    try:
        raw_targets = np.load(target_path, mmap_mode="r")["data"]
    except FileNotFoundError:
        print(f"Error: Could not find target file at {target_path}")
        return

    gt_path = os.path.join(preprocessed_data_dir, "test/physical_precip.npz")
    try:
        raw_gt = np.load(gt_path, mmap_mode="r")["data"]
    except FileNotFoundError:
        print(f"Error: Could not find ground truth file at {gt_path}")
        return

    if raw_targets.shape[0] != raw_gt.shape[0]:
        N = min(raw_targets.shape[0], raw_gt.shape[0])
    else:
        N = raw_targets.shape[0]

    # --- MODIFICATION: Subsample by Striding ---
    # Convert fraction to integer stride.
    # e.g., 0.1 -> 10 (take every 10th sample)
    # e.g., 1.0 -> 1 (take every sample)
    stride = int(1.0 / args.sample_fraction)
    if stride < 1:
        stride = 1

    subset_indices = slice(0, N, stride)

    # Apply slicing immediately during tensor creation.
    # This efficiently reads strided chunks from disk if mmap is supported/used.
    targets_t = torch.tensor(raw_targets[subset_indices], dtype=torch.float32)
    gt_t = torch.tensor(raw_gt[subset_indices], dtype=torch.float32)

    N_subset = targets_t.shape[0]
    print(
        f"Processing {N_subset} samples ({args.sample_fraction*100:.1f}% of {N}) "
        f"using Stride={stride} and Batch Size {args.batch_size}..."
    )
    # -------------------------------------------

    # Log-transform targets once here, globally
    targets_t = torch.log1p(targets_t)

    dataset = TensorDataset(targets_t, gt_t)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rapsd_gen_list = []
    rapsd_gt_list = []

    # Store freqs from the first iteration
    freqs = None
    pixel_size = config.get("PIXEL_SIZE_KM", 1.0)

    # Process Loop (Batched)
    set_seed(42)  # Global seed

    for batch_targets, batch_gt in tqdm(dataloader, desc="Inverting Test Set"):
        batch_targets = batch_targets.to(device)

        # 1. Run Inversion (Returns batch of CPU tensors)
        gen_imgs = run_inversion_batched(model, batch_targets, config, device)

        # 2. Process Batch for RAPSD
        # Note: RAPSD is still sequential per item in the batch because it's NumPy based.
        # However, the heavy lifting (Optimization) is now batched.

        batch_gt_np = batch_gt.numpy()
        gen_imgs_np = gen_imgs.numpy()

        for i in range(gen_imgs_np.shape[0]):
            gen_img_single = gen_imgs_np[i]
            gt_img_single = batch_gt_np[i]
            if gt_img_single.ndim == 3:
                gt_img_single = gt_img_single[0]

            gt_img_phys = np.maximum(gt_img_single, 0)

            # --- MODIFICATION: Log-Transform for Spectral Analysis ---
            # Transform physical R to log(1+R) to match training domain
            gen_img_log = np.log1p(gen_img_single)
            gt_img_log = np.log1p(gt_img_phys)

            curr_freqs, rapsd_gen = compute_rapsd(gen_img_log, pixel_size=pixel_size)
            _, rapsd_gt = compute_rapsd(gt_img_log, pixel_size=pixel_size)
            # ---------------------------------------------------------

            rapsd_gen_list.append(rapsd_gen)
            rapsd_gt_list.append(rapsd_gt)

            if freqs is None:
                freqs = curr_freqs

    rapsd_gen_arr = np.array(rapsd_gen_list)
    rapsd_gt_arr = np.array(rapsd_gt_list)

    save_path = os.path.join(args.run_dir, "inversion_test", args.output_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(save_path, freqs=freqs, rapsd_gen=rapsd_gen_arr, rapsd_gt=rapsd_gt_arr)
    print(f"RAPSD analysis saved to: {save_path}")


if __name__ == "__main__":
    main()
