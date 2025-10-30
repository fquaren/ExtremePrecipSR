import yaml
import numpy as np
import os
from tqdm import tqdm
from skimage import measure, morphology
from scipy.ndimage import label
import warnings
import tempfile

# Suppress specific warning from scikit-image
warnings.filterwarnings("ignore", message="No contour found", category=UserWarning)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
    # "/home/fquareng/work/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
pixel_size_km = config.get("PIXEL_SIZE_KM", 1.0)
PERSISTENCE_THRESHOLD = config.get("PERSISTENCE_THRESHOLD", 0.05)


# --- Core Geometric Calculation Functions ---
def compute_A_P_CC_single_threshold_numpy(
    prec_2d_np,
    threshold,
    pixel_size_km=1.0,
):
    """Computes Area, Perimeter using skimage, and Connected Components using TDA."""
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold

    # --- Area ---
    area_km2 = mask.sum() * (pixel_size_km**2)

    # --- Perimeter ---
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(contour, axis=0), axis=1).sum() for contour in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km

    # --- Connected Components ---
    structure = morphology.disk(1)  # Define connectivity (8-connectivity for disk(1))
    _, num_features = label(
        mask, structure=structure
    )  # Label connected regions in the mask

    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km=1.0):
    """Computes the 3xN_quantiles Gamma matrix for a single precipitation image."""
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


# --- Main Processing Function (Optimized) ---
def process_and_save_gamma_targets(data_split):
    """
    Loads precipitation data for a split (e.g., 'train'),
    computes all gamma matrices, and saves them to a new file
    using a memory-efficient memmap approach.
    """
    print(f"--- Processing data split: {data_split} ---")

    # Define paths
    input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)
    precip_path = os.path.join(input_dir, "original_precip.npz")
    output_path = os.path.join(input_dir, "gamma_targets.npz")

    # Create a temporary file path for the memmap
    # This file will hold the uncompressed results.
    temp_fd, temp_output_path = tempfile.mkstemp(suffix=".npy", dir=input_dir)
    os.close(temp_fd)  # Close the file handle, we just need the path

    print(f"Using temporary memmap file at: {temp_output_path}")

    if not os.path.exists(precip_path):
        print(f"Precipitation file not found at {precip_path}. Skipping.")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return

    print(f"Loading precipitation data from {precip_path}...")
    precip_data = np.load(precip_path, mmap_mode="r")["data"]
    num_samples = precip_data.shape[0]

    # 1. Define the shape of the final array
    output_shape = (num_samples, 3, len(QUANTILE_LEVELS))
    output_dtype = np.float32

    # 2. Create a memory-mapped array on disk in 'write' mode
    # This does NOT load into RAM.
    gamma_targets_array_mmap = np.memmap(
        temp_output_path, dtype=output_dtype, mode="w+", shape=output_shape
    )
    # --------------------------------

    print("Computing Gamma matrices for all samples (writing to memmap)...")
    for i in tqdm(range(num_samples), desc=f"Calculating Gamma for {data_split}"):
        prec_field = precip_data[i]
        gamma_matrix = compute_gamma_matrix_for_image(
            prec_field, QUANTILE_LEVELS, pixel_size_km=1.0
        )

        # 3. Write the result for this slice directly to the file
        gamma_targets_array_mmap[i] = gamma_matrix

    # 4. Flush changes to disk and close the memmap file
    print("Flushing computed data to disk...")
    gamma_targets_array_mmap.flush()
    del gamma_targets_array_mmap  # This closes the file mapping

    # --- Step 2: Compress the file ---
    # Now, load the temporary file (which is on disk) in read-only
    # mmap mode and save it using savez_compressed.
    # This streams the data from disk to the compressed file
    # without loading the whole thing into RAM.

    print(f"Reloading from memmap and compressing to {output_path}...")
    final_data_mmap = np.load(temp_output_path, mmap_mode="r", allow_pickle=True)

    np.savez_compressed(output_path, data=final_data_mmap)

    # 5. Clean up the temporary file
    print(f"Cleaning up temporary file: {temp_output_path}")
    os.remove(temp_output_path)

    print(f"Finished processing for {data_split}. Shape of saved data: {output_shape}")


if __name__ == "__main__":
    data_splits = ["train", "validation", "test"]
    for split in data_splits:
        process_and_save_gamma_targets(split)
    print("\nAll pre-computation is complete.")
