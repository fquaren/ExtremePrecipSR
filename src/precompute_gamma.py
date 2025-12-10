import yaml
import numpy as np
import os
from tqdm import tqdm
from skimage import measure
import warnings
import tempfile
import gudhi as gd
import multiprocessing
from functools import partial

# --- USER SETTINGS ---
# "diagnostic" = Extract ALL pairs (threshold=0), skip Gamma matrix calc. FAST. Use for plotting.
# "production" = Calculate Gamma matrix using config threshold. SLOW. Use for final training data.
MODE = "production"

# Suppress specific warning from scikit-image
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# Prevent numpy/MKL/OpenBLAS from using internal threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = np.array(config["QUANTILE_LEVELS"], dtype=np.float32)
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 2.0)
NUM_WORKERS = config.get("NUM_WORKERS", os.cpu_count())
WORKER_CHUNK_SIZE = config.get("WORKER_CHUNK_SIZE", 100)

# If in diagnostic mode, we capture EVERYTHING (Threshold ~ 0)
# If in production, we use the config threshold.
if MODE == "diagnostic":
    PERSISTENCE_THRESHOLD = 1e-9  # Effectively zero, avoids div by zero issues
    print(f"!!! DIAGNOSTIC MODE !!! Extracting ALL pairs > {PERSISTENCE_THRESHOLD}")
else:
    PERSISTENCE_THRESHOLD = config.get("PERSISTENCE_THRESHOLD", 0.05)
    print(
        f"--- PRODUCTION MODE --- Computing Gamma with Threshold = {PERSISTENCE_THRESHOLD}"
    )


# --- Core TDA Function ---
def compute_tda_persistence(prec_2d_np_clean):
    """Computes the persistence diagram once for a given image."""
    neg_prec_field = -prec_2d_np_clean.astype(np.float64)
    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    return cubical_complex.persistence()


# --- Fully Vectorized Gamma Matrix Calculation ---
def compute_gamma_matrix_for_image_optimized(
    prec_2d_data,
    thresholds_arr,
    pixel_size_km=1.0,
    persistence_threshold=PERSISTENCE_THRESHOLD,
    run_mode="production",
):
    """
    Computes the Gamma matrix.
    If run_mode == 'diagnostic': Skips Area/Perimeter, returns dummy matrix + ALL pairs.
    """
    gamma_matrix = np.zeros((3, len(thresholds_arr)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)

    # --- TDA Calculation (Needed for both modes) ---
    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)
    pairs_d0 = np.array(
        [p[1] for p in persistence_pairs if p[0] == 0], dtype=np.float64
    )

    significant_pairs = np.empty((0, 2), dtype=np.float64)

    if pairs_d0.shape[0] > 0:
        # Convert pairs (birth_neg, death_neg) to (birth, death, persistence)
        births = -pairs_d0[:, 0]
        deaths = -pairs_d0[:, 1]
        is_finite = deaths != -np.inf
        is_background = ~is_finite
        deaths[is_background] = np.inf
        persistence = births - deaths
        persistence[is_background] = np.inf

        # Filter pairs based on current threshold (0 if diagnostic, T if production)
        is_significant = persistence > persistence_threshold

        # Collect pairs for output
        significant_pairs = np.stack(
            [births[is_significant], deaths[is_significant]], axis=1
        )

        # --- If Production: Compute Euler Characteristic (Gamma Component 3) ---
        if run_mode == "production":
            pers_thresh_mask = is_significant[:, np.newaxis]
            births_broadcast = births[:, np.newaxis]
            deaths_broadcast = deaths[:, np.newaxis]
            thresholds_broadcast_1d = thresholds_arr[np.newaxis, :]

            birth_thresh_mask = births_broadcast >= thresholds_broadcast_1d
            death_thresh_mask = deaths_broadcast < thresholds_broadcast_1d

            finite_pass_mask = (
                pers_thresh_mask
                & birth_thresh_mask
                & death_thresh_mask
                & is_finite[:, np.newaxis]
            )
            finite_counts = np.sum(finite_pass_mask, axis=0)

            background_low_thresh_mask = thresholds_broadcast_1d <= 0.01
            background_pass_mask = (
                pers_thresh_mask
                & birth_thresh_mask
                & is_background[:, np.newaxis]
                & background_low_thresh_mask
            )
            background_counts = np.sum(background_pass_mask, axis=0)
            gamma_matrix[2, :] = finite_counts + background_counts

    # --- If Production: Compute Area and Perimeter ---
    if run_mode == "production":
        pixel_area_km2 = pixel_size_km**2
        prec_broadcast = prec_2d_np_clean[..., np.newaxis]
        thresholds_broadcast_3d = thresholds_arr[np.newaxis, np.newaxis, :]
        masks_3d = prec_broadcast >= thresholds_broadcast_3d

        # Area
        area_counts = np.sum(masks_3d, axis=(0, 1))
        gamma_matrix[0, :] = area_counts * pixel_area_km2

        # Perimeter (Loop)
        for i in range(len(thresholds_arr)):
            mask_t = masks_3d[:, :, i]
            contours = measure.find_contours(mask_t.astype(float), 0.5)
            perimeter_pixels = sum(
                np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
                for contour in contours
            )
            gamma_matrix[1, i] = perimeter_pixels * pixel_size_km

    return gamma_matrix, significant_pairs


# --- Worker function ---
def worker_compute_gamma_chunk(
    index_chunk,
    precip_path,
    output_mmap_path,
    output_shape,
    output_dtype,
    thresholds_arr,
    pixel_size_km,
    persistence_threshold,
    run_mode,
):
    precip_data = np.load(precip_path, mmap_mode="r")
    gamma_targets_array_mmap = np.memmap(
        output_mmap_path, dtype=output_dtype, mode="r+", shape=output_shape
    )
    chunk_persistence_pairs = []

    for index in index_chunk:
        prec_field = precip_data[index]
        gamma_matrix, significant_pairs = compute_gamma_matrix_for_image_optimized(
            prec_field, thresholds_arr, pixel_size_km, persistence_threshold, run_mode
        )
        gamma_targets_array_mmap[index] = gamma_matrix
        if significant_pairs.shape[0] > 0:
            chunk_persistence_pairs.append(significant_pairs)

    gamma_targets_array_mmap.flush()
    # Cleanup
    del gamma_targets_array_mmap
    del precip_data

    if chunk_persistence_pairs:
        return np.concatenate(chunk_persistence_pairs, axis=0)
    else:
        return np.empty((0, 2), dtype=np.float64)


# --- Main Processing Function ---
def process_and_save_gamma_targets(data_split):
    print(f"--- Processing data split: {data_split} | Mode: {MODE} ---")
    input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)
    precip_path = os.path.join(input_dir, "physical_precip.npy")
    output_path = os.path.join(input_dir, "gamma_targets_persistence.npz")

    # Temp file for memmap
    temp_fd, temp_output_path = tempfile.mkstemp(suffix=".npy", dir=input_dir)
    os.close(temp_fd)

    if not os.path.exists(precip_path):
        print(f"Precipitation file not found: {precip_path}")
        return

    precip_data_mmap = np.load(precip_path, mmap_mode="r")
    num_samples = precip_data_mmap.shape[0]
    del precip_data_mmap

    output_shape = (num_samples, 3, len(QUANTILE_LEVELS))
    output_dtype = np.float32

    # Initialize memmap
    gamma_targets_array_mmap = np.memmap(
        temp_output_path, dtype=output_dtype, mode="w+", shape=output_shape
    )
    del gamma_targets_array_mmap

    indices = range(num_samples)
    index_chunks = [
        indices[i : i + WORKER_CHUNK_SIZE]
        for i in range(0, num_samples, WORKER_CHUNK_SIZE)
    ]

    worker_func = partial(
        worker_compute_gamma_chunk,
        precip_path=precip_path,
        output_mmap_path=temp_output_path,
        output_shape=output_shape,
        output_dtype=output_dtype,
        thresholds_arr=QUANTILE_LEVELS,
        pixel_size_km=PIXEL_SIZE_KM,
        persistence_threshold=PERSISTENCE_THRESHOLD,
        run_mode=MODE,
    )

    all_persistence_pairs = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results_iterator = pool.imap_unordered(worker_func, index_chunks)
        for chunk_pairs in tqdm(
            results_iterator, total=len(index_chunks), desc=f"{MODE}: {data_split}"
        ):
            if chunk_pairs.shape[0] > 0:
                all_persistence_pairs.append(chunk_pairs)

    if all_persistence_pairs:
        final_pairs = np.concatenate(all_persistence_pairs, axis=0)
    else:
        final_pairs = np.empty((0, 2), dtype=np.float64)

    print(f"Aggregated {final_pairs.shape[0]} pairs.")

    final_data_mmap = np.memmap(
        temp_output_path, dtype=output_dtype, mode="r", shape=output_shape
    )

    np.savez_compressed(
        output_path,
        data=final_data_mmap,
        persistence_pairs=final_pairs,
    )

    del final_data_mmap
    os.remove(temp_output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    # Recommend running on validation first for threshold selection
    if MODE == "diagnostic":
        data_splits = ["validation"]
    else:
        data_splits = ["validation", "test", "train"]
    for split in data_splits:
        process_and_save_gamma_targets(split)
