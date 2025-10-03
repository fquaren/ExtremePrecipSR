import os
import glob
import yaml
import numpy as np
import xarray as xr
import zarr
from scipy.ndimage import zoom
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- Preprocessing Utility Functions ---
def declutter_precip(arr, threshold):
    arr_copy = arr.copy()
    arr_copy[arr_copy > threshold] = 0
    return arr_copy


def coarsen_array(arr, factor):
    m, n = arr.shape
    m_new, n_new = m // factor, n // factor
    return (
        arr[: m_new * factor, : n_new * factor]
        .reshape(m_new, factor, n_new, factor)
        .mean(axis=(1, 3))
    )


def interpolate_array(arr, factor, target_shape):
    interpolated = zoom(arr, zoom=factor, order=3)
    # Ensure exact target shape by cropping/padding if necessary
    h, w = interpolated.shape
    th, tw = target_shape
    if h > th or w > tw:
        interpolated = interpolated[:th, :tw]
    elif h < th or w < tw:
        interpolated = np.pad(interpolated, ((0, th - h), (0, tw - w)), mode="constant")
    return interpolated


# --- Worker Function ---
def process_and_write_patch(args):
    """
    Worker function to process a single patch and write it to the target Zarr store.
    """
    (
        idx,
        patch_meta,
        timestamp_map,
        output_zarr_path,
        group_name,
        config,
    ) = args

    try:
        timestamp_str, y_start, x_start = patch_meta
        patch_size = config["PATCH_SIZE"]

        # Find the source Zarr folder and time index for this patch
        source_folder, time_idx_in_folder = timestamp_map[timestamp_str]

        # Read the high-resolution patch from the raw data
        with xr.open_zarr(source_folder, consolidated=True) as ds:
            original_precip = (
                ds[config["PRECIP_VAR_NAME"]]
                .isel(
                    time=time_idx_in_folder,
                    y=slice(y_start, y_start + patch_size),
                    x=slice(x_start, x_start + patch_size),
                )
                .load()
                .values
            )

        # Handle any remaining NaNs (should be none, but for safety)
        original_precip[np.isnan(original_precip)] = 0.0

        # Perform transformations
        decluttered = declutter_precip(original_precip, config["DECLUTTER_THRESHOLD"])
        low_res = coarsen_array(decluttered, config["DOWNSCALING_FACTOR"])
        interpolated = interpolate_array(
            low_res, config["DOWNSCALING_FACTOR"], target_shape=(patch_size, patch_size)
        )

        # Write directly to the output Zarr store
        target_zarr = zarr.open(output_zarr_path, mode="r+")
        target_zarr[f"{group_name}/original_precip"][idx] = original_precip.astype(
            np.float32
        )
        target_zarr[f"{group_name}/interpolated_precip"][idx] = interpolated.astype(
            np.float32
        )
        target_zarr[f"{group_name}/coarse_precip"][idx] = low_res.astype(np.float32)

        return None  # Success
    except Exception as e:
        return f"Error processing patch {patch_meta} at index {idx}: {e}"


# --- Main Orchestration ---
def main():
    """
    Main script to preprocess data in parallel and save to a consolidated Zarr store.
    """
    # --- 1. Load Configuration ---
    config_path = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
    )
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # --- 2. Create a Timestamp -> File Map for efficient lookups ---
    print("Creating a map of timestamps to file locations...")
    timestamp_map = {}
    zarr_folders = sorted(
        glob.glob(os.path.join(config["RAW_OPERA_DATA_DIR"], "[0-9]" * 8))
    )
    for folder_path in tqdm(zarr_folders, desc="Mapping timestamps"):
        with xr.open_zarr(folder_path, consolidated=True) as ds:
            for t_idx, t_val in enumerate(ds.time.values):
                timestamp_str = (
                    np.datetime_as_string(t_val, unit="s")
                    .replace("-", "")
                    .replace("T", "")
                    .replace(":", "")
                )
                timestamp_map[timestamp_str] = (folder_path, t_idx)

    # --- 3. Load Metadata and Initialize Output Zarr Store ---
    metadata_paths = {
        "train": os.path.join(config["METADATA_DIR"], "train_patches_metadata.txt"),
        "validation": os.path.join(config["METADATA_DIR"], "val_patches_metadata.txt"),
        "test": os.path.join(config["METADATA_DIR"], "test_patches_metadata.txt"),
    }

    output_zarr_path = os.path.join(
        config["PREPROCESSED_DATA_DIR"], "preprocessed_dataset.zarr"
    )
    print(f"Initializing output Zarr store at: {output_zarr_path}")
    root = zarr.open(output_zarr_path, mode="w")

    all_tasks = {}
    patch_size = config["PATCH_SIZE"]
    coarse_patch_size = patch_size // config["DOWNSCALING_FACTOR"]

    for group_name, path in metadata_paths.items():
        with open(path, "r") as f:
            lines = f.readlines()

        num_patches = len(lines)
        group = root.create_group(group_name)
        group.create_array(
            "original_precip",
            shape=(num_patches, patch_size, patch_size),
            chunks=(1, patch_size, patch_size),
            dtype="float32",
        )
        group.create_array(
            "interpolated_precip",
            shape=(num_patches, patch_size, patch_size),
            chunks=(1, patch_size, patch_size),
            dtype="float32",
        )
        group.create_array(
            "coarse_precip",
            shape=(num_patches, coarse_patch_size, coarse_patch_size),
            chunks=(1, coarse_patch_size, coarse_patch_size),
            dtype="float32",
        )

        # Prepare tasks for this group
        tasks = []
        for i, line in enumerate(lines):
            timestamp_str, y_str, x_str = line.strip().split(",")
            patch_meta = (timestamp_str, int(y_str), int(x_str))
            tasks.append(
                (i, patch_meta, timestamp_map, output_zarr_path, group_name, config)
            )
        all_tasks[group_name] = tasks

    # --- 4. Execute Preprocessing in Parallel ---
    for group_name, tasks in all_tasks.items():
        print(
            f"\n--- Starting parallel processing for '{group_name}' set ({len(tasks)} patches) ---"
        )
        with ProcessPoolExecutor(max_workers=config["MAX_WORKERS"]) as executor:
            futures = [executor.submit(process_and_write_patch, task) for task in tasks]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {group_name}",
            ):
                result = future.result()
                if result:  # Log errors if any
                    print(result)

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
