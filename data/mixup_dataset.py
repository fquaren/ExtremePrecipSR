import yaml
import numpy as np
import os
from tqdm import tqdm
from functools import partial
import multiprocessing
import tempfile
from skimage import measure, morphology
from scipy.ndimage import label

# --- CONFIGURATION ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)
NUM_WORKERS = 32  # Adjust based on your GH200 node (usually has 64+ cores)

# --- HYPERPARAMETERS FOR MIXUP ---
MIXUP_ALPHA = 0.4
NOISE_STD = 0.05
# We generate 1 augmented sample per real sample.
# If you have plenty of storage and want more diversity, you can increase this multiplier.
AUGMENTATION_MULTIPLIER = 1


def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km
    structure = morphology.disk(1)
    _, num_features = label(mask, structure=structure)
    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km):
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


def _worker_mixup(
    indices,
    real_path,
    interp_path,
    temp_patch_path,
    temp_target_path,
    patch_shape,
    target_shape,
    quantiles,
    pixel_size,
    mixup_alpha,
    noise_std,
):
    # Open inputs (Read-Only)
    real_data = np.load(real_path, mmap_mode="r")["data"]
    interp_data = np.load(interp_path, mmap_mode="r")["data"]

    # Open outputs (Read-Write)
    out_patches = np.memmap(
        temp_patch_path, dtype=np.float32, mode="r+", shape=patch_shape
    )
    out_targets = np.memmap(
        temp_target_path, dtype=np.float32, mode="r+", shape=target_shape
    )

    for idx in indices:
        # Load patches
        real_patch = real_data[idx]
        interp_patch = interp_data[idx]

        # 1. Noise Injection (Simulate SR Artifacts)
        noise = np.random.normal(0, noise_std, interp_patch.shape)
        interp_noisy = np.clip(interp_patch + noise, 0.0, None)

        # 2. MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed_patch = lam * real_patch + (1 - lam) * interp_noisy

        # 3. Compute Physics-Consistent Topology
        # This is the heavy calculation we are offloading from the training loop
        gamma_matrix = compute_gamma_matrix_for_image(
            mixed_patch, quantiles, pixel_size
        )

        # 4. Save
        out_patches[idx] = mixed_patch.astype(np.float32)
        out_targets[idx] = gamma_matrix.astype(np.float32)

    out_patches.flush()
    out_targets.flush()
    return None


def main():
    split = "train"  # We only need this for training
    print(f"--- Generating Offline MixUp Dataset for {split.upper()} ---")

    input_dir = os.path.join(PREPROCESSED_DATA_DIR, split)
    real_path = os.path.join(input_dir, "physical_precip.npz")
    interp_path = os.path.join(input_dir, "interpolated_physical_precip.npz")

    # Check inputs
    if not os.path.exists(real_path) or not os.path.exists(interp_path):
        print("Input files missing.")
        return

    # Get Shapes
    N = np.load(real_path, mmap_mode="r")["data"].shape[0]
    H, W = np.load(real_path, mmap_mode="r")["data"].shape[1:]

    # Define Output Shapes
    out_patch_shape = (N, H, W)
    out_target_shape = (N, 3, len(QUANTILE_LEVELS))

    # Create Temp Files
    fd_p, temp_patch_path = tempfile.mkstemp(suffix=".npy", dir=input_dir)
    fd_t, temp_target_path = tempfile.mkstemp(suffix=".npy", dir=input_dir)
    os.close(fd_p)
    os.close(fd_t)

    # Initialize Memmaps
    p_mmap = np.memmap(
        temp_patch_path, dtype=np.float32, mode="w+", shape=out_patch_shape
    )
    t_mmap = np.memmap(
        temp_target_path, dtype=np.float32, mode="w+", shape=out_target_shape
    )
    del p_mmap, t_mmap

    # Prepare Workers
    indices = np.arange(N)
    chunks = np.array_split(indices, NUM_WORKERS)

    worker_func = partial(
        _worker_mixup,
        real_path=real_path,
        interp_path=interp_path,
        temp_patch_path=temp_patch_path,
        temp_target_path=temp_target_path,
        patch_shape=out_patch_shape,
        target_shape=out_target_shape,
        quantiles=QUANTILE_LEVELS,
        pixel_size=PIXEL_SIZE_KM,
        mixup_alpha=MIXUP_ALPHA,
        noise_std=NOISE_STD,
    )

    print(f"Starting processing with {NUM_WORKERS} workers...")
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(worker_func, chunks), total=len(chunks)))

    # Save Compressed
    print("Saving compressed datasets...")

    final_p = np.memmap(
        temp_patch_path, dtype=np.float32, mode="r", shape=out_patch_shape
    )
    np.savez_compressed(
        os.path.join(input_dir, "mixup_augmented_precip.npz"), data=final_p
    )

    final_t = np.memmap(
        temp_target_path, dtype=np.float32, mode="r", shape=out_target_shape
    )
    np.savez_compressed(
        os.path.join(input_dir, "mixup_augmented_targets.npz"), data=final_t
    )

    # Cleanup
    os.remove(temp_patch_path)
    os.remove(temp_target_path)
    print("Done.")


if __name__ == "__main__":
    main()
