import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import rioxarray


# --- Parameters ---
patch_size = 128
stride = 128
min_valid_fraction = 1  # e.g., 50% of patch must be non-NaN. For DEM, likely 1 (no NaNs if filled).
n_workers = 8

# --- Paths ---
base_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/OPERA"
output_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches"
dem_data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/dem" # Directory where DEM is saved

# Specific path to the reprojected DEM
reprojected_dem_filename = 'reproj_OPERA_1km_europe_dem.nc'
reprojected_dem_path = os.path.join(dem_data_dir, reprojected_dem_filename)

os.makedirs(output_dir, exist_ok=True)


def extract_valid_patches(array_2d, patch_size, stride, min_valid):
    """
    Extracts valid patches (as numpy arrays) from a 2D array.
    """
    y_dim, x_dim = array_2d.shape
    patches = []

    for i in range(0, y_dim - patch_size + 1, stride):
        for j in range(0, x_dim - patch_size + 1, stride):
            patch = array_2d[i:i+patch_size, j:j+patch_size]
            valid_pixels = np.count_nonzero(~np.isnan(patch))
            if valid_pixels >= min_valid:
                patches.append(patch)
    
    return patches


def process_zarr_folder(zarr_folder):
    """
    Process a single Zarr folder (precipitation data): extract valid patches for each time step, save as .npz
    """
    folder_path = os.path.join(base_path, zarr_folder)
    try:
        # Explicitly set chunks={} to load as a single array if it's not too large
        # or manage chunks appropriately for larger datasets.
        ds = xr.open_zarr(folder_path, chunks={}) 
        var = ds["TOT_PREC"]

        print(f"Processing precipitation folder: {zarr_folder} with shape {var.shape}")

        # tqdm for outer loop (time steps)
        for t in tqdm(range(var.sizes["time"]), desc=f"Patches for {zarr_folder}"):
            slice_2d = var.isel(time=t).values  # 2D array for current time step
            min_valid = int(min_valid_fraction * patch_size * patch_size)
            patches = extract_valid_patches(slice_2d, patch_size, stride, min_valid)
            
            if patches:
                patches_np = np.stack(patches, axis=0)
                output_file = os.path.join(output_dir, f"{zarr_folder}_time{t:04d}.npz")
                np.savez_compressed(output_file, patches=patches_np)
                # print(f"Saved {patches_np.shape[0]} patches for {zarr_folder} time={t} to {output_file}")
            # else:
            #     print(f"No valid patches in {zarr_folder} time={t}")
    
    except Exception as e:
        print(f"Error processing {zarr_folder}: {e}")


def process_dem_patches(dem_path, patch_size, stride, min_valid_fraction):
    """
    Process the 2D elevation data: extract valid patches and save as .npz
    """
    print(f"\nProcessing DEM data from: {dem_path}")
    try:
        # Load the DataArray directly
        dem_da = xr.open_dataarray(dem_path)
        
        # Ensure y-axis is north-up if needed (should already be from previous steps)
        if dem_da.y.values[0] > dem_da.y.values[-1]:
            dem_da = dem_da.isel(y=slice(None, None, -1))
        
        dem_array_2d = dem_da.values # Get the 2D numpy array
        
        min_valid = int(min_valid_fraction * patch_size * patch_size)
        patches = extract_valid_patches(dem_array_2d, patch_size, stride, min_valid)
        
        if patches:
            patches_np = np.stack(patches, axis=0)
            output_file = os.path.join(output_dir, "DEM_patches.npz")
            np.savez_compressed(output_file, patches=patches_np)
            print(f"Saved {patches_np.shape[0]} DEM patches to {output_file}")
        else:
            print(f"No valid DEM patches found.")
            
    except FileNotFoundError:
        print(f"Error: DEM file not found at {dem_path}. Skipping DEM patching.")
    except Exception as e:
        print(f"Error processing DEM patches from {dem_path}: {e}")


def main():
    print("Starting patch extraction for precipitation data...")
    zarr_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_zarr_folder, folder): folder for folder in zarr_folders}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception in folder {folder}: {e}")

    # Process DEM patches after precipitation (or before, order doesn't strictly matter)
    print("\nStarting patch extraction for DEM data...")
    process_dem_patches(reprojected_dem_path, patch_size, stride, min_valid_fraction)


if __name__ == "__main__":
    main()
    print("\nAll done.")