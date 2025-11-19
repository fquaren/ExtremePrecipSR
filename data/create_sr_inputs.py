import yaml
import numpy as np
import os
from tqdm import tqdm
import warnings
import tempfile
import multiprocessing
from functools import partial
from scipy.ndimage import zoom

# Suppress runtime warnings from np.load
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
PATCH_SIZE = config["PATCH_SIZE"]
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]
# Increased workers slightly, but kept conservative to avoid memory overflow
NUM_WORKERS = 1


# --- Image Processing Functions ---
def coarsen_image(img, factor):
    """
    Coarsens an image by a given factor using block averaging.
    Mathematically equivalent to Average Pooling.
    """
    if factor == 1:
        return img

    in_h, in_w = img.shape
    out_h, out_w = in_h // factor, in_w // factor

    if out_h == 0 or out_w == 0:
        raise ValueError(
            f"Downscaling factor {factor} is too large for shape {img.shape}"
        )

    # Crop the image to be divisible by the factor
    img_cropped = img[: out_h * factor, : out_w * factor]

    # Reshape and take the mean over the blocks
    coarse_img = img_cropped.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3))
    return coarse_img


def interpolate_image(img, target_shape):
    """
    Interpolates an image to a target shape using bilinear interpolation.
    """
    if img.shape == target_shape:
        return img

    in_h, in_w = img.shape
    target_h, target_w = target_shape

    if in_h == 0 or in_w == 0:
        return np.zeros(target_shape, dtype=img.dtype)

    # Calculate zoom factors
    zoom_h = target_h / in_h
    zoom_w = target_w / in_w

    # order=1 is for bilinear interpolation
    interp_img = zoom(img, zoom=(zoom_h, zoom_w), order=1)

    # --- Handle potential off-by-one pixel errors from zoom ---
    h, w = interp_img.shape
    if h > target_h or w > target_w:
        interp_img = interp_img[:target_h, :target_w]
    elif h < target_h or w < target_w:
        interp_img = np.pad(
            interp_img, ((0, target_h - h), (0, target_w - w)), mode="constant"
        )

    return interp_img


# --- Worker Function for Parallel Processing ---
def process_chunk(
    indices,
    input_path,
    temp_coarse_path,
    temp_interp_path,
    coarse_shape,
    interp_shape,
    dtype,
    factor,
):
    """
    Worker function to process a chunk of indices.
    Reads from disk (memmap), processes, and writes to temp disk (memmap).
    """
    # Open the large input file in memory-map mode (read-only)
    input_data = np.load(input_path, mmap_mode="r")["data"]

    # Re-open the memmap files in writeable mode (r+) inside the worker
    coarse_mmap = np.memmap(
        temp_coarse_path, dtype=dtype, mode="r+", shape=coarse_shape
    )
    interp_mmap = np.memmap(
        temp_interp_path, dtype=dtype, mode="r+", shape=interp_shape
    )

    try:
        for idx in indices:
            # 1. Read patch
            patch = input_data[idx]

            # 2. Coarsen (Average Pooling)
            coarse_patch = coarsen_image(patch, factor)

            # 3. Interpolate (Bilinear Upsampling)
            interpolated_patch = interpolate_image(coarse_patch, interp_shape[1:])

            # 4. Write to memmaps
            coarse_mmap[idx] = coarse_patch
            interp_mmap[idx] = interpolated_patch

        # Flush changes to disk
        coarse_mmap.flush()
        interp_mmap.flush()
        return None  # Success
    except Exception as e:
        return f"Error processing chunk starting at index {indices[0]}: {e}"


# --- Core Processing Logic (Encapsulated) ---
def process_dataset_variant(data_split, variant_name, input_dir):
    """
    Handles the processing for a specific variant (e.g., 'original' or 'physical')
    within a specific data split (e.g., 'train').
    """
    input_filename = f"{variant_name}_precip.npz"
    input_path = os.path.join(input_dir, input_filename)

    # Define output names based on the variant
    # e.g., coarse_original_precip.npz OR coarse_physical_precip.npz
    coarse_output_path = os.path.join(input_dir, f"coarse_{variant_name}_precip.npz")
    interp_output_path = os.path.join(
        input_dir, f"interpolated_{variant_name}_precip.npz"
    )

    if not os.path.exists(input_path):
        print(f"Warning: {input_filename} not found in {input_dir}. Skipping.")
        return

    print(f"\nProcessing {variant_name.upper()} data for {data_split}...")

    # Load metadata
    with np.load(input_path) as loader:
        num_samples = loader["data"].shape[0]
        input_shape = loader["data"].shape[1:]
        dtype = loader["data"].dtype

    if input_shape[0] != PATCH_SIZE or input_shape[1] != PATCH_SIZE:
        print(
            f"Warning: Patch size mismatch! Config: {PATCH_SIZE}, Data: {input_shape}"
        )

    # Define Output Shapes
    coarse_patch_h = input_shape[0] // DOWNSCALING_FACTOR
    coarse_patch_w = input_shape[1] // DOWNSCALING_FACTOR

    coarse_shape = (num_samples, coarse_patch_h, coarse_patch_w)
    interp_shape = (num_samples, input_shape[0], input_shape[1])

    # Create Temporary Memmap Files
    fd_c, temp_coarse_path = tempfile.mkstemp(
        suffix=f"_{variant_name}_coarse.npy", dir=input_dir
    )
    fd_i, temp_interp_path = tempfile.mkstemp(
        suffix=f"_{variant_name}_interp.npy", dir=input_dir
    )
    os.close(fd_c)
    os.close(fd_i)

    # Initialize Memmaps
    coarse_mmap = np.memmap(
        temp_coarse_path, dtype=dtype, mode="w+", shape=coarse_shape
    )
    interp_mmap = np.memmap(
        temp_interp_path, dtype=dtype, mode="w+", shape=interp_shape
    )
    del coarse_mmap, interp_mmap  # Release lock

    # Prepare Parallel Tasks
    all_indices = np.arange(num_samples)
    index_chunks = np.array_split(all_indices, NUM_WORKERS)

    worker_func = partial(
        process_chunk,
        input_path=input_path,
        temp_coarse_path=temp_coarse_path,
        temp_interp_path=temp_interp_path,
        coarse_shape=coarse_shape,
        interp_shape=interp_shape,
        dtype=dtype,
        factor=DOWNSCALING_FACTOR,
    )

    # Execute Parallel Processing
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker_func, index_chunks),
                total=len(index_chunks),
                desc=f"{variant_name} -> Coarse/Interp",
            )
        )
        for res in results:
            if res is not None:
                print(f"Worker Error: {res}")

    # Finalize: Save to Compressed NPZ
    print(f"Saving {variant_name} outputs...")

    # Read from memmap
    final_coarse_data = np.memmap(
        temp_coarse_path, dtype=dtype, mode="r", shape=coarse_shape
    )
    final_interp_data = np.memmap(
        temp_interp_path, dtype=dtype, mode="r", shape=interp_shape
    )

    np.savez_compressed(coarse_output_path, data=final_coarse_data)
    np.savez_compressed(interp_output_path, data=final_interp_data)

    # Cleanup
    os.remove(temp_coarse_path)
    os.remove(temp_interp_path)
    print(f"Finished {variant_name} for {data_split}.")


def main():
    """
    Main orchestration script.
    Iterates over splits and data variants (original/physical).
    """
    # The two types of files we saved in the previous step
    VARIANTS = ["original", "physical"]

    for data_split in ["train", "validation", "test"]:
        print(f"\n{'='*40}")
        print(f"--- DATA SPLIT: {data_split.upper()} ---")
        print(f"{'='*40}")

        input_dir = os.path.join(PREPROCESSED_DATA_DIR, data_split)

        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue

        # Process both 'original' (ML-ready) and 'physical' (Raw units)
        for variant in VARIANTS:
            process_dataset_variant(data_split, variant, input_dir)


if __name__ == "__main__":
    # Optimization flags for numpy/scipy
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Safety for multiprocessing with C-extensions
    multiprocessing.set_start_method("spawn", force=True)

    main()
