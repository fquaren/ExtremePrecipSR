import os
import glob
import random
import yaml
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_valid_patches_in_frame(frame, patch_size):
    """
    Uses an efficient striding method to find all non-NaN patches in a 2D array.
    (This function is unchanged)
    """
    shape = (
        frame.shape[0] - patch_size + 1,
        frame.shape[1] - patch_size + 1,
        patch_size,
        patch_size,
    )
    strides = frame.strides * 2
    patches = np.lib.stride_tricks.as_strided(frame, shape=shape, strides=strides)
    is_nan_in_patch = np.isnan(patches).any(axis=(2, 3))
    valid_y_x_indices = np.argwhere(~is_nan_in_patch)
    return [tuple(coord) for coord in valid_y_x_indices]


def save_metadata_to_file(metadata, filepath):
    """Saves a list of metadata tuples to a text file. (Unchanged)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for timestamp, y, x in metadata:
            f.write(f"{timestamp},{y},{x}\n")
    print(f"Saved {len(metadata)} entries to {filepath}")


def scan_zarr_folder_for_patches(folder_path, precip_var_name, patch_size):
    """
    **Worker function**: Scans a single daily Zarr folder and returns all valid patch coordinates.
    This function will be executed in a separate process.
    """
    local_valid_coords = []
    try:
        with xr.open_zarr(folder_path, consolidated=True) as ds:
            for t_idx in range(len(ds.time)):
                frame = ds[precip_var_name].isel(time=t_idx).load().values
                timestamp_dt_numpy = ds.time.isel(time=t_idx).values
                timestamp_str = (
                    np.datetime_as_string(timestamp_dt_numpy, unit="s")
                    .replace("-", "")
                    .replace("T", "")
                    .replace(":", "")
                )

                valid_patch_coords_in_frame = find_valid_patches_in_frame(
                    frame, patch_size
                )

                for y, x in valid_patch_coords_in_frame:
                    local_valid_coords.append((timestamp_str, y, x))
    except Exception as e:
        print(f"Warning: Worker process could not process {folder_path}. Error: {e}")
    return local_valid_coords


def main():
    """
    Main script, now refactored to orchestrate parallel processing of Zarr files.
    """
    # --- 1. Load Configuration ---
    config_path = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
    )
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    RAW_OPERA_DATA_DIR = config["RAW_OPERA_DATA_DIR"]
    METADATA_DIR = config["METADATA_DIR"]
    PATCH_SIZE = config["PATCH_SIZE"]
    SPLIT_RATIOS = config["SPLIT_RATIOS"]
    PRECIP_VAR_NAME = config["PRECIP_VAR_NAME"]
    MAX_WORKERS = config.get(
        "MAX_WORKERS", os.cpu_count()
    )  # Use config value or all available cores

    # --- 2. Find all daily Zarr directories ---
    zarr_folders = sorted(glob.glob(os.path.join(RAW_OPERA_DATA_DIR, "[0-9]" * 8)))
    if not zarr_folders:
        raise FileNotFoundError(f"No Zarr folders found in {RAW_OPERA_DATA_DIR}")
    print(
        f"Found {len(zarr_folders)} daily Zarr folders to scan using up to {MAX_WORKERS} workers."
    )

    # --- 3. Scan all files in PARALLEL ---
    all_valid_coords = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each folder to be processed as a separate job
        futures = [
            executor.submit(
                scan_zarr_folder_for_patches, folder, PRECIP_VAR_NAME, PATCH_SIZE
            )
            for folder in zarr_folders
        ]

        # Collect results as they complete, with a progress bar
        for future in tqdm(
            as_completed(futures), total=len(zarr_folders), desc="Scanning daily files"
        ):
            # result is the list of coordinates from one worker
            result = future.result()
            if result:
                all_valid_coords.extend(result)

    print(f"\nFound a total of {len(all_valid_coords)} valid patches across all files.")

    # --- 4. Shuffle and Split Data (Unchanged) ---
    random.shuffle(all_valid_coords)
    total_patches = len(all_valid_coords)
    train_end_idx = int(total_patches * SPLIT_RATIOS["train"])
    val_end_idx = train_end_idx + int(total_patches * SPLIT_RATIOS["validation"])
    train_meta = all_valid_coords[:train_end_idx]
    val_meta = all_valid_coords[train_end_idx:val_end_idx]
    test_meta = all_valid_coords[val_end_idx:]

    # --- 5. Save Metadata Files (Unchanged) ---
    save_metadata_to_file(
        train_meta, os.path.join(METADATA_DIR, "train_patches_metadata.txt")
    )
    save_metadata_to_file(
        val_meta, os.path.join(METADATA_DIR, "val_patches_metadata.txt")
    )
    save_metadata_to_file(
        test_meta, os.path.join(METADATA_DIR, "test_patches_metadata.txt")
    )


if __name__ == "__main__":
    main()
