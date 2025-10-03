import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- Parameters ---
# Size of the square patches (e.g., 128x128 pixels)
patch_size = 128
# Stride for patch extraction. If stride == patch_size, patches are non-overlapping.
# If stride < patch_size, patches will overlap.
stride = 128
# Minimum fraction of non-NaN pixels required for a precipitation patch to be considered valid.
# E.g., 0.9 means 90% of the patch must be valid data.
min_valid_fraction_precip = 1.0
# Minimum fraction of non-NaN pixels for a DEM patch.
# Typically 1.0 (100%) as DEMs are usually filled and complete.
min_valid_fraction_dem = 1.0
# Number of parallel processes to use for processing Zarr folders.
n_workers = 8

# --- Paths ---
# Base directory containing your Zarr precipitation data folders (e.g., 2017-01-01T00_00_00)
base_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/OPERA"
# Base directory where all processed patches (precipitation and DEM) will be saved.
output_base_dir = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches_v2"
)
# Directory where your preprocessed (reprojected) DEM data is located.
dem_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/dem"

# Full path to your reprojected DEM NetCDF file.
reprojected_dem_filename = "reproj_OPERA_1km_europe_dem.nc"
reprojected_dem_path = os.path.join(dem_data_dir, reprojected_dem_filename)

# Define output subdirectories for clear organization
PRECIP_PATCH_OUTPUT_DIR = os.path.join(output_base_dir, "precip")
DEM_PATCH_OUTPUT_DIR = os.path.join(output_base_dir, "dem")

# Create output directories if they don't exist
os.makedirs(PRECIP_PATCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEM_PATCH_OUTPUT_DIR, exist_ok=True)


def extract_valid_patches_with_coords(
    array_2d, patch_size, stride, min_valid_pixels_count
):
    """
    Extracts valid patches (as numpy arrays) and their top-left (y, x) coordinates from a 2D array.
    A patch is considered valid if it contains at least 'min_valid_pixels_count' non-NaN pixels.

    Args:
        array_2d (np.ndarray): The 2D input array (e.g., a single time slice of precipitation or DEM).
        patch_size (int): The height and width of the square patch.
        stride (int): The step size for moving the patch window.
        min_valid_pixels_count (int): Minimum number of non-NaN pixels for a patch to be valid.

    Returns:
        list: A list of tuples, where each tuple contains (patch_data, (start_y, start_x)).
    """
    y_dim, x_dim = array_2d.shape
    valid_patches_info = []

    # Iterate over the 2D array with the defined stride
    for i in range(0, y_dim - patch_size + 1, stride):
        for j in range(0, x_dim - patch_size + 1, stride):
            # Extract the patch
            patch = array_2d[i : i + patch_size, j : j + patch_size]
            # Count non-NaN pixels in the patch
            valid_pixels = np.count_nonzero(~np.isnan(patch))

            # If the patch meets the validity criteria, store its data and coordinates
            if valid_pixels >= min_valid_pixels_count:
                valid_patches_info.append((patch, (i, j)))
    return valid_patches_info


def process_single_zarr_folder(zarr_folder_name):
    """
    Processes a single Zarr folder containing precipitation data.
    It extracts valid patches for each time step, saves them to disk,
    and collects the unique (y, x) coordinates of all extracted patches within this folder.

    Args:
        zarr_folder_name (str): The name of the Zarr folder (e.g., "2017-01-01T00_00_00").

    Returns:
        list: A list of unique (y, x) coordinates found in this Zarr folder.
              These coordinates represent the top-left corner of valid patches.
    """
    folder_path = os.path.join(base_path, zarr_folder_name)
    all_valid_coords_in_folder = set()  # Use a set to store unique (y, x) coordinates
    precip_var_name = (
        "TOT_PREC"  # Assumed variable name for total precipitation in Zarr files
    )

    try:
        # Open the Zarr dataset. chunks={} loads the entire dataset into memory,
        # which is suitable for iterating through time steps for patch extraction.
        ds = xr.open_zarr(folder_path, chunks={})
        precip_data_array = ds[precip_var_name]

        print(
            f"Processing precipitation folder: {zarr_folder_name}, shape: {precip_data_array.shape}"
        )

        # Create a specific output directory for this Zarr folder's precipitation patches
        folder_output_path = os.path.join(PRECIP_PATCH_OUTPUT_DIR, zarr_folder_name)
        os.makedirs(folder_output_path, exist_ok=True)

        # Calculate the minimum number of valid pixels required for precipitation patches
        min_valid_pixels = int(min_valid_fraction_precip * patch_size * patch_size)

        # Iterate through each time step in the Zarr dataset
        for t in tqdm(
            range(precip_data_array.sizes["time"]),
            desc=f"Patches for {zarr_folder_name}",
        ):
            # Extract the 2D data slice for the current time step and ensure float32 type
            slice_2d = precip_data_array.isel(time=t).values.astype(np.float32)

            # Get a timestamp string from the xarray DataArray for naming output files
            # This handles datetime objects and provides a fallback for simpler indexing
            try:
                time_str = (
                    precip_data_array.time.isel(time=t)
                    .dt.strftime("%Y%m%d%H%M%S")
                    .item()
                )
            except AttributeError:
                time_str = f"time{t:04d}"  # Fallback if 'dt' accessor is not available

            # Extract valid patches along with their coordinates
            patches_info = extract_valid_patches_with_coords(
                slice_2d, patch_size, stride, min_valid_pixels
            )

            if patches_info:
                # Create a subdirectory for patches from this specific time step
                time_step_output_path = os.path.join(folder_output_path, time_str)
                os.makedirs(time_step_output_path, exist_ok=True)

                # Save each extracted precipitation patch individually
                for patch_data, (y_start, x_start) in patches_info:
                    patch_filename = f"patch_y{y_start:04d}_x{x_start:04d}.npy"
                    np.save(
                        os.path.join(time_step_output_path, patch_filename), patch_data
                    )
                    # Add the (y, x) coordinates to our set of unique locations
                    all_valid_coords_in_folder.add((y_start, x_start))

    except Exception as e:
        print(f"Error processing {zarr_folder_name}: {e}")

    # Return the list of unique coordinates found in this folder
    return list(all_valid_coords_in_folder)


def process_dem_data(
    dem_array_2d, all_unique_precip_coords, patch_size, min_valid_pixels_dem
):
    """
    Extracts and saves DEM patches only for the coordinates where valid precipitation
    patches were found. This ensures spatial alignment.

    Args:
        dem_array_2d (np.ndarray): The 2D numpy array of the DEM.
        all_unique_precip_coords (set): A set of unique (y, x) coordinates from
                                         all valid precipitation patches.
        patch_size (int): The size of the square patch.
        min_valid_pixels_dem (int): Minimum number of non-NaN pixels for a DEM patch to be valid.
    """
    print(
        f"\nExtracting and saving DEM patches for {len(all_unique_precip_coords)} unique locations..."
    )

    dem_y_dim, dem_x_dim = dem_array_2d.shape

    # Iterate through each unique coordinate collected from precipitation data
    for y_start, x_start in tqdm(all_unique_precip_coords, desc="Saving DEM patches"):
        # Check if the potential DEM patch falls entirely within the DEM array bounds
        if (y_start + patch_size <= dem_y_dim) and (x_start + patch_size <= dem_x_dim):
            # Extract the DEM patch and ensure float32 type
            dem_patch = dem_array_2d[
                y_start : y_start + patch_size, x_start : x_start + patch_size
            ].astype(np.float32)

            # Check for validity of the DEM patch (e.g., no NaNs if filled)
            valid_pixels_dem = np.count_nonzero(~np.isnan(dem_patch))
            if valid_pixels_dem >= min_valid_pixels_dem:
                # Save the DEM patch with a filename indicating its (y, x) coordinates
                patch_filename = f"dem_patch_y{y_start:04d}_x{x_start:04d}.npy"
                np.save(os.path.join(DEM_PATCH_OUTPUT_DIR, patch_filename), dem_patch)
        # else:
        # Optional: Print a message if a precipitation patch's coordinates fall outside DEM bounds
        # print(f"Skipping DEM patch at ({y_start}, {x_start}) as it's out of DEM bounds.")


def main():
    print(
        "Starting patch extraction for precipitation data and identifying valid patch locations..."
    )

    # Get a list of all Zarr folder names (each represents a time period or cluster)
    zarr_folders = sorted(
        [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    )

    all_precip_coords = (
        set()
    )  # Master set to collect all unique (y, x) coordinates from all precip folders

    # --- Pass 1: Process Precipitation Data in Parallel ---
    # Use a ProcessPoolExecutor to speed up processing of multiple Zarr folders
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit each Zarr folder to the executor and store the future objects
        futures = {
            executor.submit(process_single_zarr_folder, folder): folder
            for folder in zarr_folders
        }
        # Iterate over completed futures to collect results
        for future in as_completed(futures):
            folder = futures[future]
            try:
                coords_from_folder = future.result()
                if coords_from_folder:
                    all_precip_coords.update(
                        coords_from_folder
                    )  # Add new unique coordinates to the master set
            except Exception as e:
                print(f"Exception processing folder {folder}: {e}")

    # --- Pass 2: Load and Process DEM Data ---
    # This step runs sequentially after all precipitation data has been processed
    print(
        "\nAll precipitation patches processed. Now loading and processing DEM data..."
    )
    try:
        # Load the DEM DataArray from the specified path
        dem_da = xr.open_dataarray(reprojected_dem_path)

        # Ensure the y-axis (latitude) is oriented consistently (e.g., north-up).
        # xarray's .isel(y=slice(None, None, -1)) reverses the y-dimension if it's south-up.
        if dem_da.y.values[0] > dem_da.y.values[-1]:
            dem_da = dem_da.isel(y=slice(None, None, -1))

        # Get the 2D numpy array of the DEM data
        dem_array_2d = dem_da.values

        # Calculate the minimum number of valid pixels required for DEM patches
        min_valid_pixels_dem = int(min_valid_fraction_dem * patch_size * patch_size)

        # Call the function to extract and save DEM patches based on collected precipitation coordinates
        process_dem_data(
            dem_array_2d, all_precip_coords, patch_size, min_valid_pixels_dem
        )

    except FileNotFoundError:
        print(
            f"Error: DEM file not found at {reprojected_dem_path}. Cannot process DEM patches."
        )
    except Exception as e:
        print(f"Error loading or processing DEM: {e}")


if __name__ == "__main__":
    main()
    print(
        "\nAll done. Precipitation and DEM patches saved to corresponding subdirectories."
    )
