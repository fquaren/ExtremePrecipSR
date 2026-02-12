import os
import sys
import yaml
import glob
import numpy as np
import xarray as xr
import warnings
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.ndimage import zoom
from tqdm import tqdm

# Suppress runtime warnings from np.load and coincidental nans
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PATCH_SIZE = config["PATCH_SIZE"]
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)
RAW_OPERA_DATA_DIR = config["RAW_OPERA_DATA_DIR"]
PREPROCESSED_DATA_DIR = config.get("PREPROCESSED_DATA_DIR")
METADATA_DIR = config["METADATA_DIR"]
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]

# Worker configurations
NUM_WORKERS_EXTRACT = 8  # IO Bound (Zarr reading)
NUM_WORKERS_PROCESS = 4  # CPU/Memory Bound (Downscaling/Memmap)


# =============================================================================
# PART 1: ZARR EXTRACTION HELPERS
# =============================================================================

# Global cache for Zarr process workers
_process_zarr_cache = {}


def _initializer():
    global _process_zarr_cache
    _process_zarr_cache = {}


def _get_zarr_dataset(zarr_folder_path):
    global _process_zarr_cache
    if zarr_folder_path not in _process_zarr_cache:
        _process_zarr_cache[zarr_folder_path] = xr.open_zarr(
            zarr_folder_path, chunks={}, decode_times=True
        )
    return _process_zarr_cache[zarr_folder_path]


def extract_single_patch_worker(
    task_tuple,  # Changed to accept a tuple so we can unpack the index
    all_zarr_folder_info,
    precip_var_name,
    patch_size,
    drizzle_threshold,
):
    # Unpack the index we added in the submission loop
    index, (timestamp_str, y_start, x_start) = task_tuple

    try:
        if timestamp_str not in all_zarr_folder_info:
            return None

        zarr_folder_path, time_idx = all_zarr_folder_info[timestamp_str]
        ds = _get_zarr_dataset(zarr_folder_path)

        # Load data
        patch_data = (
            ds[precip_var_name]
            .isel(
                time=time_idx,
                y=slice(y_start, y_start + patch_size),
                x=slice(x_start, x_start + patch_size),
            )
            .load()
            .values.astype(np.float32)
        )

        # 1. Clean NaNs
        patch_data[np.isnan(patch_data)] = 0.0

        # 2. Denoise (Drizzle)
        if drizzle_threshold > 0:
            patch_data[patch_data < drizzle_threshold] = 0.0

        # RETURN THE INDEX WITH THE DATA
        return (index, patch_data)

    except Exception as e:
        print(f"Extraction Error {timestamp_str}: {e}", file=sys.stderr)
        return None


# =============================================================================
# PART 2: DOWNSCALING & TRANSFORMATION HELPERS
# =============================================================================


def coarsen_image(img, factor):
    """Block averaging (Physical mass conservation)."""
    if factor == 1:
        return img
    h, w = img.shape
    new_h, new_w = h // factor, w // factor

    # Crop to divisible size
    img = img[: new_h * factor, : new_w * factor]

    return img.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))


def interpolate_image(img, target_shape):
    """Bilinear interpolation."""
    if img.shape == target_shape:
        return img

    t_h, t_w = target_shape
    s_h, s_w = img.shape

    if s_h == 0 or s_w == 0:
        return np.zeros(target_shape, dtype=img.dtype)

    # Bilinear zoom
    zoom_factors = (t_h / s_h, t_w / s_w)
    out = zoom(img, zoom_factors, order=1)

    # Handle rounding mismatches
    h, w = out.shape
    if h > t_h or w > t_w:
        out = out[:t_h, :t_w]
    elif h < t_h or w < t_w:
        out = np.pad(out, ((0, t_h - h), (0, t_w - w)), mode="constant")

    return out


def generate_variants_worker(
    indices,
    input_path,
    output_paths,  # Dict of paths for memmaps
    shapes,  # Dict of shapes
    dtype,
    downscale_factor,
    scaler_val,
):
    """
    Reads Physical. Generates:
    1. Coarse Physical
    2. Interpolated Physical
    3. HR Original (Transformed)
    4. Coarse Original (Transformed)
    5. Interpolated Original (Transformed)
    """
    # Read-only input
    input_data = np.load(input_path, mmap_mode="r")["data"]

    # Open all output memmaps
    mmaps = {}
    for key, path in output_paths.items():
        mmaps[key] = np.memmap(path, dtype=dtype, mode="r+", shape=shapes[key])

    try:
        for idx in indices:
            # --- 1. Load Physical ---
            phys_patch = input_data[idx]

            # --- 2. Create Derived Physical ---
            # Coarsen the PHYSICAL data (Mass Conserved)
            phys_coarse = coarsen_image(phys_patch, downscale_factor)
            # Interpolate the PHYSICAL data
            phys_interp = interpolate_image(phys_coarse, phys_patch.shape)

            # Save Physicals
            mmaps["physical"][
                idx
            ] = phys_patch  # Redundant but keeps file consistent if we want separate memmap
            mmaps["coarse_physical"][idx] = phys_coarse
            mmaps["interpolated_physical"][idx] = phys_interp

            # --- 3. Create Transformed (Original) ---
            # Define transform closure
            def transform(arr):
                # Log1p -> Div by Max -> Clip
                x = np.log1p(arr)
                x = x / scaler_val
                return np.clip(x, 0.0, 1.0)

            mmaps["original"][idx] = transform(phys_patch)
            mmaps["coarse_original"][idx] = transform(phys_coarse)
            mmaps["interpolated_original"][idx] = transform(phys_interp)

        # Flush
        for m in mmaps.values():
            m.flush()
        return None

    except Exception as e:
        return f"Gen Error idx {indices[0]}: {e}"


# =============================================================================
# ORCHESTRATION
# =============================================================================


def step_1_extract_physical(split_name, metadata_path, output_dir, all_zarr_info):
    output_path = os.path.join(output_dir, "physical_precip.npz")
    if os.path.exists(output_path):
        print(f"[{split_name}] physical_precip.npz exists. Skipping extraction.")
        return output_path

    print(f"\n--- Extracting Physical Data: {split_name} ---")

    tasks = []
    with open(metadata_path, "r") as f:
        # Enumerate lines to track original order
        for i, line in enumerate(f):
            p = line.strip().split(",")
            if len(p) == 3:
                # Pass index 'i' along with the metadata
                tasks.append((i, (p[0], int(p[1]), int(p[2]))))

    patches_dict = {}  # Store in dict first

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS_EXTRACT, initializer=_initializer
    ) as pool:
        worker = partial(
            extract_single_patch_worker,
            all_zarr_folder_info=all_zarr_info,
            precip_var_name="TOT_PREC",
            patch_size=PATCH_SIZE,
            drizzle_threshold=DRIZZLE_THRESHOLD,
        )

        # We can still use as_completed for the progress bar,
        # but we must sort results later.
        futures = [pool.submit(worker, t) for t in tasks]

        for f in tqdm(
            as_completed(futures), total=len(futures), desc=f"Extracting {split_name}"
        ):
            res = f.result()
            if res is not None:
                idx, data = res
                patches_dict[idx] = data

    if not patches_dict:
        raise ValueError(f"No patches extracted for {split_name}!")

    # Reconstruct the list in the correct order (0 to N-1)
    # We filter out missing indices (failed extractions) while maintaining relative order
    # NOTE: If indices are missing, your metadata file lines must also be skipped
    # in the Dataset class. Ideally, you want to keep them aligned 1:1.

    sorted_indices = sorted(patches_dict.keys())
    ordered_patches = [patches_dict[i] for i in sorted_indices]

    # IMPORTANT: If extractions failed, your npz has fewer items than metadata lines.
    # You should probably save a 'clean' metadata file here corresponding to successful extracts,
    # or ensure extraction never fails silently.
    # For now, assuming standard success rate:

    print(f"Saving {len(ordered_patches)} patches to {output_path}...")
    full_arr = np.stack(ordered_patches, axis=0)
    np.savez_compressed(output_path, data=full_arr)

    return output_path


def step_2_calculate_scaler(train_physical_path, save_dir):
    """Calculates Max(Log1p(Physical)) from training data."""
    print("\n--- Calculating Global Scaler (Training) ---")
    scaler_path = os.path.join(save_dir, "precip_max_val.npy")

    # If exists, load (remove this check if you always want to recalc)
    if os.path.exists(scaler_path):
        val = np.load(scaler_path)[0]
        print(f"Loaded existing scaler: {val}")
        return val

    with np.load(train_physical_path) as loader:
        data = loader[
            "data"
        ]  # mmap not needed for max calc usually, but dataset might be huge
        # We can iterate if memory is tight, but assuming fits in RAM for now based on prev scripts
        # Use a safe max calculation
        print("Computing log1p max...")
        # Do not transform in place to avoid memory spikes, compute max on chunks if needed
        # Simple approach:
        current_max = 0.0
        # Iterate in chunks to be safe
        chunk_size = 1000
        for i in range(0, len(data), chunk_size):
            chunk = np.log1p(data[i : i + chunk_size])
            m = np.max(chunk)
            if m > current_max:
                current_max = m

    if current_max == 0:
        current_max = 1.0  # Safety

    np.save(scaler_path, np.array([current_max]))
    print(f"Saved new scaler: {current_max}")
    return current_max


def step_3_generate_derived(split_name, input_dir, scaler_val):
    """Generates coarse and original (transformed) datasets from physical."""
    print(f"\n--- Generating Derived Datasets: {split_name} ---")

    phys_path = os.path.join(input_dir, "physical_precip.npz")

    # Metadata scan
    with np.load(phys_path) as loader:
        n_samples = loader["data"].shape[0]
        h, w = loader["data"].shape[1:]
        dtype = loader["data"].dtype

    # Define Shapes
    h_lr, w_lr = h // DOWNSCALING_FACTOR, w // DOWNSCALING_FACTOR

    shapes = {
        "physical": (n_samples, h, w),
        "coarse_physical": (n_samples, h_lr, w_lr),
        "interpolated_physical": (n_samples, h, w),
        "original": (n_samples, h, w),
        "coarse_original": (n_samples, h_lr, w_lr),
        "interpolated_original": (n_samples, h, w),
    }

    # Prepare Temp Files and Paths
    temp_files = {}
    output_paths = {}
    final_paths = {}

    # keys to process
    keys = list(shapes.keys())
    # We skip "physical" in output generation if we don't want to duplicate it,
    # but for code simplicity we can treat it as a pass-through or skip.
    # Let's skip writing 'physical' back to itself to save IO.
    keys.remove("physical")

    for k in keys:
        fname = f"{k}_precip.npz"
        final_paths[k] = os.path.join(input_dir, fname)
        fd, tpath = tempfile.mkstemp(suffix=f"_{k}.npy", dir=input_dir)
        os.close(fd)
        temp_files[k] = tpath
        output_paths[k] = tpath

        # Init Memmap
        mm = np.memmap(tpath, dtype=dtype, mode="w+", shape=shapes[k])
        del mm

    # Worker Partial
    worker = partial(
        generate_variants_worker,
        input_path=phys_path,
        output_paths=output_paths,
        shapes=shapes,
        dtype=dtype,
        downscale_factor=DOWNSCALING_FACTOR,
        scaler_val=scaler_val,
    )

    # Parallel Execution
    indices = np.arange(n_samples)
    chunks = np.array_split(indices, NUM_WORKERS_PROCESS)

    with multiprocessing.Pool(NUM_WORKERS_PROCESS) as pool:
        res_iter = pool.imap_unordered(worker, chunks)
        for res in tqdm(res_iter, total=len(chunks), desc=f"Processing {split_name}"):
            if res:
                print(f"Error: {res}")

    # Save to Compressed NPZ & Cleanup
    print("Compressing and saving outputs...")
    for k in keys:
        # Read temp
        arr = np.memmap(temp_files[k], dtype=dtype, mode="r", shape=shapes[k])
        # Write compressed
        np.savez_compressed(final_paths[k], data=arr)
        # Delete temp
        os.remove(temp_files[k])

    print(f"Finished {split_name}.")


def main():
    # 1. Map Zarr Folders (Once)
    print("Mapping Zarr Data...")
    all_zarr_info = {}
    zarr_folders = sorted(glob.glob(os.path.join(RAW_OPERA_DATA_DIR, "[0-9]" * 8)))

    for f in tqdm(zarr_folders):
        try:
            # Quick open to get times
            with xr.open_zarr(f, chunks={}, decode_times=True) as ds:
                if "TOT_PREC" in ds and "time" in ds.coords:
                    times = ds.time.values
                    for i, t in enumerate(times):
                        t_str = (
                            t.astype("datetime64[s]").item().strftime("%Y%m%d%H%M%S")
                        )
                        all_zarr_info[t_str] = (f, i)
        except:
            pass

    print(f"Mapped {len(all_zarr_info)} timestamps.")

    # 2. Process Splits
    splits = ["train", "validation", "test"]
    scaler = None

    for split in splits:
        print(f"\n{'='*30}\nDATA SPLIT: {split.upper()}\n{'='*30}")

        # Paths
        meta_file = os.path.join(METADATA_DIR, f"{split}_patches_metadata.txt")
        split_dir = os.path.join(PREPROCESSED_DATA_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        # Step A: Extract Physical (if needed)
        phys_path = step_1_extract_physical(split, meta_file, split_dir, all_zarr_info)

        # Step B: Get Scaler (only once from Train)
        if split == "train":
            scaler = step_2_calculate_scaler(phys_path, PREPROCESSED_DATA_DIR)
        elif scaler is None:
            # Load if we started loop from val/test (unlikely but safe)
            scaler = np.load(os.path.join(PREPROCESSED_DATA_DIR, "precip_max_val.npy"))[
                0
            ]

        # Step C: Generate Derived Datasets
        step_3_generate_derived(split, split_dir, scaler)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
