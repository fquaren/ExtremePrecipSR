import os
import zarr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import random
import time
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ZarrPrecipitationUNetDataset(Dataset):
    def __init__(self, date_paths, patch_size=(128, 128)):
        super().__init__()
        self.date_paths = sorted(date_paths)
        self.patch_size = patch_size

        if not self.date_paths:
            raise ValueError("date_paths cannot be empty.")

        self.timesteps_per_day = 96
        self.patch_height, self.patch_width = self.patch_size

        # --- Open all Zarr stores once and store references ---
        self.zarr_stores = {path: zarr.open(path, mode="r") for path in self.date_paths}

        # --- Determine the full resolution and coordinate data from the first Zarr file ---
        first_store = self.zarr_stores[self.date_paths[0]]
        self.full_height = first_store["TOT_PREC"].shape[1]
        self.full_width = first_store["TOT_PREC"].shape[2]
        self.x_coords = first_store["x"][:]
        self.y_coords = first_store["y"][:]
        self.time_coords = first_store["time"][:]

        if self.patch_height > self.full_height or self.patch_width > self.full_width:
            raise ValueError(
                "Patch size cannot be larger than the full image resolution."
            )

        print(f"Initialized dataset with {len(self.date_paths)} days.")
        print(f"Full data resolution: ({self.full_height}, {self.full_width})")
        print(f"Using patch size: {self.patch_size}")

    def __len__(self):
        return len(self.date_paths) * self.timesteps_per_day

    def __getitem__(self, idx):
        day_index = idx // self.timesteps_per_day
        timestep_in_day = idx % self.timesteps_per_day
        day_path = self.date_paths[day_index]

        try:
            # --- Access the already-opened Zarr store ---
            daily_store = self.zarr_stores[day_path]
            precipitation_data = daily_store["TOT_PREC"]

            # --- Define the spatial slices (patching) ---
            h, w = self.patch_size
            top = random.randint(0, self.full_height - h)
            left = random.randint(0, self.full_width - w)

            # --- Extract the high-resolution patch ---
            high_res_np = precipitation_data[
                timestep_in_day, top : top + h, left : left + w
            ]
            high_res_target = (
                torch.from_numpy(np.array(high_res_np)).float().unsqueeze(0)
            )

            # --- Create the low-resolution input (as before) ---
            blurred_tensor = TF.gaussian_blur(high_res_target, kernel_size=5, sigma=1.0)
            low_res_temp = F.interpolate(
                blurred_tensor.unsqueeze(0),
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            )
            low_res_input = F.interpolate(
                low_res_temp, size=self.patch_size, mode="bilinear", align_corners=False
            ).squeeze(0)

            # --- Extract metadata for the patch ---
            x_patch = self.x_coords[left : left + w]
            y_patch = self.y_coords[top : top + h]
            time_patch = self.time_coords[timestep_in_day]

            # --- Return a dictionary of all data ---
            return {
                "low_res_input": low_res_input,
                "high_res_target": high_res_target,
                "x": torch.from_numpy(np.array(x_patch)).float(),
                "y": torch.from_numpy(np.array(y_patch)).float(),
                "time": torch.tensor(time_patch).float(),
            }

        except Exception as e:
            print(f"Error loading data for index {idx} at path {day_path}: {e}")
            return None


def collate_fn(batch):
    """
    Custom collate function to handle a list of dictionaries from the dataset.
    It filters out None values and stacks the tensors for each key.
    """
    # Filter out any None samples from the batch
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}  # Return an empty dict if the batch is empty

    # Stack the tensors for each key
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.utils.data.dataloader.default_collate(
            [item[key] for item in batch]
        )

    return collated_batch


if __name__ == "__main__":
    # --- Configuration ---
    ROOT_DATA_DIR = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/raw_data/OPERA"
    BATCH_SIZE = 32
    # Set num_workers to a value that best utilizes your GH200's CPU cores.
    # Start with a number of workers that is a multiple of the number of CPU cores.
    NUM_WORKERS = 0  # A good starting point for a high-core-count CPU.

    # --- Temporal Splitting Logic (unchanged) ---
    print("1. Determining date splits for training, validation, and testing...")
    train_val_start = pd.to_datetime("2023-08-01")
    train_val_end = pd.to_datetime("2024-07-31")
    test_start = pd.to_datetime("2024-08-01")
    test_end = pd.to_datetime("2024-10-30")

    all_train_val_dates = pd.date_range(start=train_val_start, end=train_val_end)
    training_dates = [date for i, date in enumerate(all_train_val_dates) if i % 7 != 0]
    validation_dates = [
        date for i, date in enumerate(all_train_val_dates) if i % 7 == 0
    ]
    test_dates = pd.date_range(start=test_start, end=test_end)

    print(f"   Training dates: {len(training_dates)}")
    print(f"   Validation dates: {len(validation_dates)}")
    print(f"   Testing dates: {len(test_dates)}")

    train_paths = [
        os.path.join(ROOT_DATA_DIR, d.strftime("%Y%m%d"))
        for d in training_dates
        if os.path.isdir(os.path.join(ROOT_DATA_DIR, d.strftime("%Y%m%d")))
    ]
    val_paths = [
        os.path.join(ROOT_DATA_DIR, d.strftime("%Y%m%d"))
        for d in validation_dates
        if os.path.isdir(os.path.join(ROOT_DATA_DIR, d.strftime("%Y%m%d")))
    ]
    test_paths = [
        os.path.join(ROOT_DATA_DIR, d.strftime("%Y%m%d"))
        for d in test_dates
        if os.path.isdir(os.path.join(ROOT_DATA_DIR, d.strftime("%Y%m%d")))
    ]

    print("\n2. Initializing PyTorch Datasets and DataLoaders...")

    PATCH_SIZE = (128, 128)
    train_dataset = ZarrPrecipitationUNetDataset(train_paths, patch_size=PATCH_SIZE)
    val_dataset = ZarrPrecipitationUNetDataset(val_paths, patch_size=PATCH_SIZE)
    test_dataset = ZarrPrecipitationUNetDataset(test_paths, patch_size=PATCH_SIZE)

    # --- Instantiate DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        # persistent_workers=True,  # Recommended for faster loading on subsequent epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        # persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        # persistent_workers=True,
    )

    print("\n3. Simulating a training loop...")
    print("\n--- Training Phase ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("Timing the fetch of the first batch...")
        start_time = time.time()
        # The loader now returns a dictionary
        batch_data = next(iter(train_loader))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Successfully fetched one training batch in {elapsed_time:.4f} seconds.")

        # --- Access the data from the dictionary ---
        low_res_inputs = batch_data["low_res_input"]
        high_res_targets = batch_data["high_res_target"]
        x_coords = batch_data["x"]
        y_coords = batch_data["y"]
        time_coords = batch_data["time"]

        print(f"Low-res input batch shape: {low_res_inputs.shape}")
        print(f"High-res target batch shape: {high_res_targets.shape}")
        print(f"x coordinates batch shape: {x_coords.shape}")
        print(f"y coordinates batch shape: {y_coords.shape}")
        print(f"time coordinates batch shape: {time_coords.shape}")

        # --- Move data to GPU for model training ---
        low_res_inputs = low_res_inputs.to(device)
        high_res_targets = high_res_targets.to(device)
        x_coords = x_coords.to(device)
        y_coords = y_coords.to(device)
        time_coords = time_coords.to(device)

        print(low_res_inputs)

        # Your model forward pass would look something like this:
        # output = your_model(low_res_inputs, x_coords, y_coords, time_coords)
        print("\nData is ready to be passed to the model on the GPU.")

    except StopIteration:
        print(
            "Could not fetch a batch. The dataset might be empty or all paths were invalid."
        )
    except Exception as e:
        print(f"An error occurred while fetching a batch: {e}")
