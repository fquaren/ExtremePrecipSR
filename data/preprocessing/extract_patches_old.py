import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parameters
patch_size = 128
stride = 128
min_valid_fraction = 1  # e.g., 50% of patch must be non-NaN
n_workers = 8

# Paths
base_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/OPERA"
output_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches"
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
    Process a single Zarr folder: extract valid patches for each time step, save as .npz
    """
    folder_path = os.path.join(base_path, zarr_folder)
    try:
        ds = xr.open_zarr(folder_path, chunks={})
        var = ds["TOT_PREC"]

        for t in tqdm(range(var.sizes["time"])):
            slice_2d = var.isel(time=t).values  # 2D array
            min_valid = int(min_valid_fraction * patch_size * patch_size)
            patches = extract_valid_patches(slice_2d, patch_size, stride, min_valid)
            
            if patches:
                patches_np = np.stack(patches, axis=0)
                output_file = os.path.join(output_dir, f"{zarr_folder}_time{t:04d}.npz")
                np.savez_compressed(output_file, patches=patches_np)
                print(f"Saved {patches_np.shape[0]} patches to {output_file}")
            else:
                print(f"No valid patches in {zarr_folder} time={t}")
    
    except Exception as e:
        print(f"Error processing {zarr_folder}: {e}")


def main():
    zarr_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_zarr_folder, folder): folder for folder in zarr_folders}
        for future in as_completed(futures):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception in folder {folder}: {e}")


if __name__ == "__main__":
    main()
    print("All done.")