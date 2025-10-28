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

# Suppress specific warning from scikit-image
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Configuration Loading ---
config_path = "/home/fquareng/work/ExtremePrecipSR/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
pixel_size_km = config.get("PIXEL_SIZE_KM", 1.0)  # Corrected config key
PERSISTENCE_THRESHOLD = config.get("PERSISTENCE_THRESHOLD", 0.05)
NUM_WORKERS = config.get("NUM_WORKERS", os.cpu_count())


# Separated TDA calculation from threshold-dependent parts
def compute_tda_persistence(prec_2d_np_clean):
    """Computes the persistence diagram once for a given image."""
    neg_prec_field = -prec_2d_np_clean.astype(np.float64)
    cubical_complex = gd.CubicalComplex(
        dimensions=neg_prec_field.shape, top_dimensional_cells=neg_prec_field.flatten()
    )
    # Return pairs AND the complex itself if needed for other TDA features later
    return cubical_complex.persistence()


def filter_persistence_for_cc(persistence_pairs, threshold, persistence_threshold):
    """Filters pre-computed persistence pairs for CC count at a given threshold."""
    num_features_tda = 0
    for dim, (birth_val_neg, death_val_neg) in persistence_pairs:
        if dim == 0:
            birth_prec = -birth_val_neg
            death_prec = -death_val_neg if death_val_neg != np.inf else np.inf
            persistence = (
                birth_prec - death_prec if death_prec != np.inf else np.inf
            )  # Handle infinite persistence

            # Count if significant and exists at or above the threshold
            if persistence > persistence_threshold and birth_prec >= threshold:
                # Exclude the background component unless threshold is very low
                if death_val_neg != np.inf:
                    num_features_tda += 1
                elif threshold <= 0.01:  # Count background only at low thresholds
                    num_features_tda += 1  # Ensure count is at least 1

    # Optional: ensure count is 1 if there is area > 0 but TDA count is 0
    # This depends on definition - keep raw TDA for now.
    return num_features_tda


# Function now only calculates Area and Perimeter (threshold-dependent)
def compute_A_P_single_threshold_numpy(prec_2d_np_clean, threshold, pixel_size_km=1.0):
    """Computes Area and Perimeter for a single threshold."""
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km
    return area_km2, perimeter_km, mask  # Return mask for potential reuse


# Optimized gamma matrix calculation
def compute_gamma_matrix_for_image_optimized(
    prec_2d_data,
    thresholds,
    pixel_size_km=1.0,
    persistence_threshold=PERSISTENCE_THRESHOLD,
):
    """Computes the Gamma matrix, calculating TDA persistence only once."""
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    prec_2d_np_clean = np.nan_to_num(prec_2d_data, nan=-1.0)  # Clean once

    # --- Compute TDA Persistence ONCE ---
    persistence_pairs = compute_tda_persistence(prec_2d_np_clean)
    # ------------------------------------

    for i, threshold_value in enumerate(thresholds):
        # Calculate Area and Perimeter for this threshold
        area_km2, perimeter_km, mask = compute_A_P_single_threshold_numpy(
            prec_2d_np_clean, threshold_value, pixel_size_km
        )

        # Filter pre-computed persistence for CC count at this threshold
        num_features_tda = filter_persistence_for_cc(
            persistence_pairs, threshold_value, persistence_threshold
        )

        # Optional: Adjust CC count based on Area
        # if area_km2 > 0 and num_features_tda == 0:
        #    num_features_tda = 1

        gamma_matrix[0, i] = area_km2
        gamma_matrix[1, i] = perimeter_km
        gamma_matrix[2, i] = num_features_tda

    return gamma_matrix


# Worker function for parallel processing
def worker_compute_gamma(
    index,
    precip_path,
    output_mmap_path,
    output_shape,
    output_dtype,
    thresholds,
    pixel_size_km,
    persistence_threshold,
):
    """Computes gamma for a single index and writes to memmap."""
    # Load the full data inside the worker, but only access the needed slice
    # This works efficiently with mmap_mode='r'
    precip_data = np.load(precip_path, mmap_mode="r")["data"]
    prec_field = precip_data[index]

    gamma_matrix = compute_gamma_matrix_for_image_optimized(
        prec_field, thresholds, pixel_size_km, persistence_threshold
    )

    # Re-open the memmap file in write mode within the worker
    gamma_targets_array_mmap = np.memmap(
        output_mmap_path, dtype=output_dtype, mode="r+", shape=output_shape
    )
    gamma_targets_array_mmap[index] = gamma_matrix
    gamma_targets_array_mmap.flush()  # Ensure write completes


# --- Main Processing Function (Optimized + Parallelized) ---
def process_and_save_gamma_targets(data_split):
    print(f"--- Processing data split: {data_split} ---")
    input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)
    precip_path = os.path.join(input_dir, "original_precip.npz")
    output_path = os.path.join(input_dir, "gamma_targets.npz")
    temp_fd, temp_output_path = tempfile.mkstemp(suffix=".npy", dir=input_dir)
    os.close(temp_fd)
    print(f"Using temporary memmap file at: {temp_output_path}")

    if not os.path.exists(precip_path):
        print(f"Precipitation file not found: {precip_path}. Skipping.")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return

    print(f"Loading precipitation data structure from {precip_path}...")
    # Load only shape initially
    with np.load(precip_path) as loader:
        num_samples = loader["data"].shape[0]

    output_shape = (num_samples, 3, len(QUANTILE_LEVELS))
    output_dtype = np.float32

    # Create the memmap file (zero-initialized)
    gamma_targets_array_mmap = np.memmap(
        temp_output_path, dtype=output_dtype, mode="w+", shape=output_shape
    )
    # IMPORTANT: Close the memmap file here so workers can open it in r+ mode
    del gamma_targets_array_mmap
    print("Memmap file initialized.")

    print(
        f"Computing Gamma matrices for {num_samples} samples using {NUM_WORKERS} workers..."
    )

    # Create a partial function with fixed arguments for the worker
    worker_func = partial(
        worker_compute_gamma,
        precip_path=precip_path,
        output_mmap_path=temp_output_path,
        output_shape=output_shape,
        output_dtype=output_dtype,
        thresholds=QUANTILE_LEVELS,
        pixel_size_km=pixel_size_km,
        persistence_threshold=PERSISTENCE_THRESHOLD,
    )

    # Use multiprocessing Pool
    indices = range(num_samples)
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Use tqdm to show progress with imap_unordered for efficiency
        list(
            tqdm(
                pool.imap_unordered(worker_func, indices),
                total=num_samples,
                desc=f"Calculating Gamma for {data_split}",
            )
        )

    print("All workers finished. Reloading from memmap and compressing...")
    final_data_mmap = np.load(temp_output_path, mmap_mode="r")
    np.savez_compressed(output_path, data=final_data_mmap)
    print(f"Saved compressed Gamma matrices to {output_path}")

    print(f"Cleaning up temporary file: {temp_output_path}")
    os.remove(temp_output_path)
    print(f"Finished processing for {data_split}. Shape of saved data: {output_shape}")


if __name__ == "__main__":
    data_splits = ["train", "validation", "test"]
    for split in data_splits:
        process_and_save_gamma_targets(split)
    print("\nAll pre-computation is complete.")
