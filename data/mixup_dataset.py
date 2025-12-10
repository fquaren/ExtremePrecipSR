import yaml
import numpy as np
import os
from tqdm import tqdm
from functools import partial
import multiprocessing
import tempfile
from skimage import measure
import gudhi as gd  # Added for consistent topological computation

# --- CONFIGURATION ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
QUANTILE_LEVELS = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
# Default to 0.05 if not in config, matching your production script default
PERSISTENCE_THRESHOLD = config.get("PERSISTENCE_THRESHOLD", 0.05)
NUM_WORKERS = 4

# --- HYPERPARAMETERS FOR MIXUP ---
MIXUP_ALPHA = 0.4
NOISE_STD = 0.05
AUGMENTATION_MULTIPLIER = 1


def compute_tda_persistence(prec_2d_np_clean):
    """
    Computes persistence diagram using Gudhi (Cubical Complex).
    """
    # Gudhi expects double precision for stability
    neg_prec_field = -prec_2d_np_clean.astype(np.float64)
    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    return cubical_complex.persistence()


def compute_gamma_matrix_consistent(
    prec_2d_data, thresholds_arr, pixel_size_km, persistence_threshold
):
    """
    Computes A, P, and Euler Characteristic (via TDA) consistent with the offline script.
    """
    gamma_matrix = np.zeros((3, len(thresholds_arr)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)

    # --- 1. Topologically Consistent Component Counting (Euler/CC) ---
    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)

    # Extract 0-dim features (connected components)
    # Pairs format: (dimension, (birth, death))
    pairs_d0 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 0], dtype=np.float64
    )

    if pairs_d0.shape[0] > 0:
        # Transform back to original intensity: birth = -val
        births = -pairs_d0[:, 0]
        deaths = -pairs_d0[:, 1]

        # Handle infinite death (global maxima)
        is_finite = deaths != -np.inf
        is_background = ~is_finite
        deaths[is_background] = np.inf

        persistence = births - deaths
        persistence[is_background] = np.inf

        # FILTER: Crucial step to remove noise artifacts
        is_significant = persistence > persistence_threshold

        # Apply filter
        births = births[is_significant]
        deaths = deaths[is_significant]
        is_finite = is_finite[is_significant]
        is_background = is_background[is_significant]

        # Vectorized Counting across all thresholds
        births_broadcast = births[:, np.newaxis]
        deaths_broadcast = deaths[:, np.newaxis]
        thresholds_broadcast = thresholds_arr[np.newaxis, :]

        # Count finite components: Birth >= T > Death
        mask_finite = (
            (births_broadcast >= thresholds_broadcast)
            & (deaths_broadcast < thresholds_broadcast)
            & is_finite[:, np.newaxis]
        )

        # Count background/infinite components: Birth >= T
        mask_background = (births_broadcast >= thresholds_broadcast) & is_background[
            :, np.newaxis
        ]

        gamma_matrix[2, :] = np.sum(mask_finite, axis=0) + np.sum(
            mask_background, axis=0
        )

    # --- 2. Area and Perimeter ---
    pixel_area_km2 = pixel_size_km**2

    # Broadcast for vectorized thresholding
    prec_broadcast = prec_2d_np_clean[..., np.newaxis]
    thresholds_broadcast_3d = thresholds_arr[np.newaxis, np.newaxis, :]
    masks_3d = prec_broadcast >= thresholds_broadcast_3d

    # Area: Sum pixels * pixel size
    gamma_matrix[0, :] = np.sum(masks_3d, axis=(0, 1)) * pixel_area_km2

    # Perimeter: Loop required for contour finding
    # Optimized to skip empty masks
    for i in range(len(thresholds_arr)):
        mask_t = masks_3d[:, :, i]
        if np.any(mask_t):
            contours = measure.find_contours(mask_t.astype(float), 0.5)
            # Vectorized Euclidean distance for perimeter sum
            perimeter_pixels = sum(
                np.sum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))
                for c in contours
            )
            gamma_matrix[1, i] = perimeter_pixels * pixel_size_km
        else:
            gamma_matrix[1, i] = 0.0

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
    persistence_thresh,  # Added argument
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
        # Using the corrected function with persistence thresholding
        gamma_matrix = compute_gamma_matrix_consistent(
            mixed_patch, quantiles, pixel_size, persistence_thresh
        )

        # 4. Save
        out_patches[idx] = mixed_patch.astype(np.float32)
        out_targets[idx] = gamma_matrix.astype(np.float32)

    out_patches.flush()
    out_targets.flush()
    return None


def main():
    split = ["train"]
    for s in split:
        print(f"--- Generating Offline MixUp Dataset for {s.upper()} ---")
        print(f"Using Persistence Threshold: {PERSISTENCE_THRESHOLD}")

        input_dir = os.path.join(PREPROCESSED_DATA_DIR, s)
        real_path = os.path.join(input_dir, "physical_precip.npz")
        interp_path = os.path.join(input_dir, "interpolated_physical_precip.npz")

        # Check inputs
        if not os.path.exists(real_path) or not os.path.exists(interp_path):
            print(f"Input files missing in {input_dir}.")
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
            persistence_thresh=PERSISTENCE_THRESHOLD,  # Pass the config value
        )

        print(f"Starting processing with {NUM_WORKERS} workers...")
        # Using spawn can be safer for TDA/Gudhi + Multiprocessing in some envs
        # If you encounter issues, uncomment the line below:
        # multiprocessing.set_start_method("spawn", force=True)

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
            os.path.join(input_dir, "mixup_augmented_targets_persistence.npz"),
            data=final_t,
        )

        # Cleanup
        os.remove(temp_patch_path)
        os.remove(temp_target_path)
    print("Done.")


if __name__ == "__main__":
    main()
