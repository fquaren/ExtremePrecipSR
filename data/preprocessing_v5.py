import os
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

# Load configuration
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PATCH_SIZE = config["PATCH_SIZE"]
DRIZZLE_THRESHOLD = config.get("DRIZZLE_THRESHOLD", 0.1)
RAW_OPERA_DATA_DIR = config["RAW_OPERA_DATA_DIR"]
PREPROCESSED_DATA_DIR = config.get("PREPROCESSED_DATA_DIR")
METADATA_DIR = config["METADATA_DIR"]


# --- Global Zarr Cache for each process ---
_process_zarr_cache = {}


def _get_zarr_dataset_for_process(zarr_folder_path):
    global _process_zarr_cache
    if zarr_folder_path not in _process_zarr_cache:
        _process_zarr_cache[zarr_folder_path] = xr.open_zarr(
            zarr_folder_path, chunks={}, decode_times=True
        )
    return _process_zarr_cache[zarr_folder_path]


def _initializer():
    global _process_zarr_cache
    _process_zarr_cache = {}


def process_single_patch(
    patch_metadata,
    all_zarr_folder_info,
    precip_var_name,
    patch_size,
    drizzle_threshold,
):
    """
    Processes and returns the CLEANED PHYSICAL patch (mm/h).
    Note: Log-transform and Scaling are now deferred to the main process
    to allow saving both physical and processed versions.
    """
    timestamp_str, y_start, x_start = patch_metadata

    try:
        zarr_folder_path, time_idx_in_folder = all_zarr_folder_info[timestamp_str]
        ds = _get_zarr_dataset_for_process(zarr_folder_path)
        precip_data_array = ds[precip_var_name]

        # Load raw data
        output_precip = (
            precip_data_array.isel(
                time=time_idx_in_folder,
                y=slice(y_start, y_start + patch_size),
                x=slice(x_start, x_start + patch_size),
            )
            .load()
            .values.astype(np.float32)
        )

        # 1. Handle NaNs (Physical cleaning)
        output_precip[np.isnan(output_precip)] = 0.0

        # 2. Apply Drizzle Thresholding (Physical Denoising)
        if drizzle_threshold > 0:
            output_precip[output_precip < drizzle_threshold] = 0.0

        # Return physical units (mm/h)
        return output_precip

    except Exception as e:
        print(
            f"Error processing {timestamp_str}, Y:{y_start}, X:{x_start}: {e}",
            file=sys.stderr,
        )
        return None


def parallel_preprocess_and_save_data(
    metadata_file_path,
    all_zarr_folder_info,
    preprocessed_data_dir,
    precip_var_name="TOT_PREC",
    patch_size=PATCH_SIZE,
    drizzle_threshold=DRIZZLE_THRESHOLD,
    transform_input_precip=False,
    max_workers=32,
    global_max=None,
):
    """
    Preprocesses Zarr data.
    Saves:
      1. physical_precip.npz (Cleaned, raw units mm/h)
      2. original_precip.npz (Log-transformed and MinMax scaled)
    """
    metadata = []
    with open(metadata_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                timestamp_str, y_str, x_str = parts
                metadata.append((timestamp_str, int(y_str), int(x_str)))

    print(
        f"Starting preprocessing for {len(metadata)} entries from {metadata_file_path}"
    )

    physical_patches = []

    # Prepare arguments (removed transform_input_precip from worker args)
    tasks = [
        (
            patch,
            all_zarr_folder_info,
            precip_var_name,
            patch_size,
            drizzle_threshold,
        )
        for patch in metadata
    ]

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_initializer
    ) as executor:
        futures = [
            executor.submit(process_single_patch, *task_args) for task_args in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Parallel Preprocessing ({max_workers} workers)",
            leave=True,
        ):
            result = future.result()
            if result is None:
                continue
            physical_patches.append(result)

    if not physical_patches:
        print(f"Warning: No valid patches for {metadata_file_path}.")
        return None if global_max is None else None

    # Stack to create the Master Physical Array
    # Shape: (N, H, W)
    data_array = np.stack(physical_patches, axis=0)

    # --- SAVE 1: Physical Data (Cleaned, No Log, No Scale) ---
    output_file_phys = os.path.join(preprocessed_data_dir, "physical_precip.npz")
    np.savez_compressed(output_file_phys, data=data_array)
    print(f"Saved PHYSICAL data {data_array.shape} to {output_file_phys}")

    # Min-Max Scaling Logic
    if global_max is None:
        # Training Mode: Calculate Max from the (possibly logged) data
        current_max = np.max(data_array)
        if current_max == 0:
            print("Warning: Max is 0. Scaling skipped.", file=sys.stderr)
            current_max = 1.0
        print(f"Calculated new global max (on transformed data): {current_max}")
    else:
        # Val/Test Mode: Use provided Max
        print(f"Using provided global max for scaling: {global_max}")
        current_max = global_max

    # --- TRANSFORMATION PIPELINE ---

    # 1. Log Transformation (In-place to save RAM)
    # Mathematical formulation: x' = ln(1 + x)
    if transform_input_precip:
        print("Applying Log-Transform (log1p) to data...")
        np.log1p(data_array, out=data_array)

    # 2. Min-Max Scaling Logic (For the "original_precip.npz")
    # This calculates the max of the LOGGED data (e.g., 5.0)
    # We usually keep this for Super-Resolution models that work in log-space
    if global_max is None:
        log_space_max = np.max(data_array)
        if log_space_max == 0:
            log_space_max = 1.0
        current_max_for_scaling = log_space_max

        # Save the Log-Space scalar too (optional, if needed for other models)
        scaler_log_path = os.path.join(preprocessed_data_dir, "scaler_max_log.npy")
        np.save(scaler_log_path, np.array([log_space_max]))
    else:
        # IMPORTANT: If global_max is passed, it must be the LOG space max
        # if we are normalizing log data!
        current_max_for_scaling = global_max

    # 3. Apply Scaling (In-place)
    # Mathematical formulation: x'' = x' / max(x'_train)    # Note: We perform division in place.
    # We must cast to float32 explicitely if not already, though it should be.
    data_array /= current_max_for_scaling

    # 4. Clip to ensure [0, 1] bounds (numerical stability)
    np.clip(data_array, 0.0, 1.0, out=data_array)

    # --- SAVE 2: Processed Data (ML Ready) ---
    output_file_orig = os.path.join(preprocessed_data_dir, "original_precip.npz")
    np.savez_compressed(output_file_orig, data=data_array)
    print(f"Saved PROCESSED data {data_array.shape} to {output_file_orig}")

    if global_max is None:
        return current_max


def main():
    all_zarr_folder_info = {}
    precip_var_name = "TOT_PREC"

    # Regex loading of Zarr folders
    zarr_folders = sorted(glob.glob(os.path.join(RAW_OPERA_DATA_DIR, "[0-9]" * 8)))

    print(f"Found {len(zarr_folders)} Zarr folders.")
    if not zarr_folders:
        return

    # Map timestamps
    for folder_path in tqdm(zarr_folders, desc="Mapping Timestamps"):
        try:
            with xr.open_zarr(folder_path, chunks={}, decode_times=True) as ds:
                if "time" in ds.coords and precip_var_name in ds.data_vars:
                    # Using set/dictionary logic for O(1) access later
                    times = ds[precip_var_name]["time"].values
                    for t_idx, t_val in enumerate(times):
                        dt_obj = t_val.astype("datetime64[s]").item()
                        full_timestamp_str = dt_obj.strftime("%Y%m%d%H%M%S")
                        all_zarr_folder_info[full_timestamp_str] = (folder_path, t_idx)
        except Exception:
            continue

    print(f"Total timestamps mapped: {len(all_zarr_folder_info)}")

    # Setup Directories
    train_meta = os.path.join(METADATA_DIR, "train_patches_metadata.txt")
    val_meta = os.path.join(METADATA_DIR, "val_patches_metadata.txt")
    test_meta = os.path.join(METADATA_DIR, "test_patches_metadata.txt")

    base_dir = PREPROCESSED_DATA_DIR
    # Check if "transform_input_precip" logic is desired (assuming True for typical SR tasks)
    # You can toggle this to False if you want linear scaling only.
    DO_LOG_TRANSFORM = True

    # 1. Process Training Data
    print("\n--- Training Data ---")
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    train_max = parallel_preprocess_and_save_data(
        train_meta,
        all_zarr_folder_info,
        os.path.join(base_dir, "train"),
        transform_input_precip=DO_LOG_TRANSFORM,
        max_workers=32,
        global_max=None,
    )

    if train_max is None:
        print("Training preprocessing failed.")
        return

    # Save the scaler
    np.save(os.path.join(base_dir, "scaler_max_val.npy"), np.array([train_max]))

    # 2. Process Validation Data
    print("\n--- Validation Data ---")
    os.makedirs(os.path.join(base_dir, "validation"), exist_ok=True)
    parallel_preprocess_and_save_data(
        val_meta,
        all_zarr_folder_info,
        os.path.join(base_dir, "validation"),
        transform_input_precip=DO_LOG_TRANSFORM,
        max_workers=32,
        global_max=train_max,
    )

    # 3. Process Test Data
    print("\n--- Test Data ---")
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)
    parallel_preprocess_and_save_data(
        test_meta,
        all_zarr_folder_info,
        os.path.join(base_dir, "test"),
        transform_input_precip=DO_LOG_TRANSFORM,
        max_workers=32,
        global_max=train_max,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
