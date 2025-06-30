import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob  # To find all .npz files


class PrecipitationDataset(Dataset):
    def __init__(self, precip_npz_dir, dem_npy_dir, transform=None):
        """
        Args:
            precip_npz_dir (str): Directory containing preprocessed precipitation .npz files.
                                  Expected structure: precip_npz_dir/zarr_folder_name/time_str/patch_yYY_xXX.npz
            dem_npy_dir (str): Directory containing DEM .npy files.
                                  Expected structure: dem_npy_dir/dem_patch_yYY_xXX.npy
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.precip_npz_dir = precip_npz_dir
        self.dem_npy_dir = dem_npy_dir
        self.transform = transform

        # List all .npz files recursively within the precipitation directory
        self.npz_files = glob.glob(
            os.path.join(self.precip_npz_dir, "**", "*.npz"), recursive=True
        )

        if not self.npz_files:
            raise RuntimeError(
                f"No .npz files found in {self.precip_npz_dir}. Please ensure preprocessing script ran successfully."
            )

        print(f"Found {len(self.npz_files)} preprocessed precipitation patches.")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        # Path to the current precipitation NPZ file
        precip_npz_path = self.npz_files[idx]

        # Load the precipitation data
        data = np.load(precip_npz_path)
        coarsened_precip = data["coarsened"]
        interpolated_precip = data["interpolated"]
        # The target for training will be the high-resolution normalized precipitation
        target_normalized_precip = data["normalized"]

        # Extract y_start and x_start from the filename to find the corresponding DEM
        # Example filename: .../patch_y0000_x0000.npz
        filename = os.path.basename(precip_npz_path)
        parts = filename.replace(".npz", "").split("_")
        y_str = [p for p in parts if p.startswith("y")][0]  # e.g., 'y0000'
        x_str = [p for p in parts if p.startswith("x")][0]  # e.g., 'x0000'

        # Construct the DEM filename and path
        dem_filename = f"dem_patch_{y_str}_{x_str}.npy"
        dem_path = os.path.join(self.dem_npy_dir, dem_filename)

        if not os.path.exists(dem_path):
            raise FileNotFoundError(
                f"Corresponding DEM file not found: {dem_path} for {precip_npz_path}"
            )

        elevation_data = np.load(dem_path)

        # Convert numpy arrays to PyTorch tensors
        # Ensure float32 and add a channel dimension if it's not present (e.g., 2D -> 1, 2D)
        # UNet expects B, C, H, W. Our patches are H, W, so add C=1.
        coarsened_precip_tensor = (
            torch.from_numpy(coarsened_precip).float().unsqueeze(0)
        )
        interpolated_precip_tensor = (
            torch.from_numpy(interpolated_precip).float().unsqueeze(0)
        )
        elevation_tensor = torch.from_numpy(elevation_data).float().unsqueeze(0)
        target_normalized_precip_tensor = (
            torch.from_numpy(target_normalized_precip).float().unsqueeze(0)
        )

        # Apply transformations if any
        if self.transform:
            # You might want to define transforms that take multiple inputs
            # or apply them individually if suitable
            coarsened_precip_tensor = self.transform(coarsened_precip_tensor)
            interpolated_precip_tensor = self.transform(interpolated_precip_tensor)
            elevation_tensor = self.transform(elevation_tensor)
            target_normalized_precip_tensor = self.transform(
                target_normalized_precip_tensor
            )

        # Return the inputs for the UNet and the target
        return {
            "coarse_precip": coarsened_precip_tensor,
            "interpolated_precip": interpolated_precip_tensor,
            "elevation": elevation_tensor,
            "target_normalized_precip": target_normalized_precip_tensor,
        }


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """
    Creates a PyTorch DataLoader from a Dataset.
    """
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        # A common practice is to use fewer workers than CPU cores to avoid oversubscription
        # especially if each worker loads large files.
        num_workers = min(
            num_workers, 8
        )  # Cap at 8 or adjust based on your system/data size

    print(f"Using {num_workers} workers for DataLoader.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Optimizes data transfer to GPU
    )


# --- Example Usage ---
if __name__ == "__main__":
    # Define the base directory for your processed data
    output_base_dir = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches_v3"
    )
    precip_npz_base_dir = os.path.join(output_base_dir, "preprocessed_data")
    dem_npy_dir = os.path.join(output_base_dir, "dem")

    # The `main_merged_workflow` in the previous script generated
    # `train_files.txt`, `val_files.txt`, `test_files.txt`
    # within `precip_npz_base_dir`.
    # We will use these lists to create our datasets.

    # Function to read file paths from a list file
    def get_file_paths_from_list(list_filepath):
        with open(list_filepath, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # To create a dataset, we need to pass a list of *specific* NPZ files.
    # The current `PrecipitationDataset` constructor recursively finds all .npz files.
    # If you want to use the train/val/test split from `main_merged_workflow`,
    # you'd modify the dataset to take a list of explicit file paths instead of a directory.

    # Let's adapt `PrecipitationDataset` to take a list of specific NPZ files directly.
    class SplitPrecipitationDataset(PrecipitationDataset):
        def __init__(self, npz_filepaths, dem_npy_dir, transform=None):

            self.dem_npy_dir = dem_npy_dir
            self.transform = transform
            self.npz_files = npz_filepaths  # Use the provided list of files directly

            if not self.npz_files:
                raise RuntimeError("No NPZ file paths provided for the dataset.")

            print(
                f"Dataset initialized with {len(self.npz_files)} specific preprocessed precipitation patches."
            )

    print("\n--- Initializing Datasets and DataLoaders ---")

    # Paths to the lists of train/val/test NPZ files (from the previous script's output)
    train_npz_list_path = os.path.join(precip_npz_base_dir, "train_files.txt")
    val_npz_list_path = os.path.join(precip_npz_base_dir, "val_files.txt")
    test_npz_list_path = os.path.join(precip_npz_base_dir, "test_files.txt")

    try:
        train_npz_files = get_file_paths_from_list(train_npz_list_path)
        val_npz_files = get_file_paths_from_list(val_npz_list_path)
        test_npz_files = get_file_paths_from_list(test_npz_list_path)
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Make sure the preprocessing script has been run to generate file lists and NPZ files."
        )
        exit()

    # Create datasets
    train_dataset = SplitPrecipitationDataset(train_npz_files, dem_npy_dir)
    val_dataset = SplitPrecipitationDataset(val_npz_files, dem_npy_dir)
    test_dataset = SplitPrecipitationDataset(test_npz_files, dem_npy_dir)

    # Create DataLoaders
    batch_size = 16  # Adjust as per your GPU memory
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrain DataLoader has {len(train_loader)} batches of size {batch_size}")
    print(f"Validation DataLoader has {len(val_loader)} batches of size {batch_size}")
    print(f"Test DataLoader has {len(test_loader)} batches of size {batch_size}")

    # --- Test fetching a batch ---
    print("\n--- Testing DataLoader iteration ---")
    try:
        first_batch = next(iter(train_loader))
        print(f"Keys in the first batch: {first_batch.keys()}")
        print(f"Shape of coarse_precip in batch: {first_batch['coarse_precip'].shape}")
        print(
            f"Shape of interpolated_precip in batch: {first_batch['interpolated_precip'].shape}"
        )
        print(f"Shape of elevation in batch: {first_batch['elevation'].shape}")
        print(
            f"Shape of target_normalized_precip in batch: {first_batch['target_normalized_precip'].shape}"
        )

        # Verify data types are float32
        assert first_batch["coarse_precip"].dtype == torch.float32
        assert first_batch["interpolated_precip"].dtype == torch.float32
        assert first_batch["elevation"].dtype == torch.float32
        assert first_batch["target_normalized_precip"].dtype == torch.float32

        # Verify the dimensions (B, C, H, W) where C=1 for all inputs/targets
        assert first_batch["coarse_precip"].ndim == 4
        assert first_batch["interpolated_precip"].ndim == 4
        assert first_batch["elevation"].ndim == 4
        assert first_batch["target_normalized_precip"].ndim == 4

        # Check expected H, W dimensions for high-res data (128x128)
        assert first_batch["interpolated_precip"].shape[2:] == (128, 128)
        assert first_batch["elevation"].shape[2:] == (128, 128)
        assert first_batch["target_normalized_precip"].shape[2:] == (128, 128)

        # Check expected H, W dimensions for coarse data (128/6 = 21x21 approx, due to integer division)
        # The `coarsen_array` uses integer division, so 128 // 6 = 21.
        assert first_batch["coarse_precip"].shape[2:] == (21, 21)

        print("\nSuccessfully loaded and inspected a batch from the DataLoader!")

    except Exception as e:
        print(f"An error occurred during DataLoader test: {e}")


class SplitPrecipitationDataset(PrecipitationDataset):
    def __init__(self, npz_filepaths, dem_npy_dir, transform=None):
        self.precip_npz_dir = os.path.dirname(npz_filepaths[0]) if npz_filepaths else ""
        self.dem_npy_dir = dem_npy_dir
        self.transform = transform
        self.npz_files = npz_filepaths

        if not self.npz_files:
            raise RuntimeError("No NPZ file paths provided for the dataset.")

        print(
            f"Dataset initialized with {len(self.npz_files)} specific preprocessed precipitation patches."
        )
