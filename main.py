import argparse
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import yaml
import random
import string
import pandas as pd

from src.logger import setup_logger
from src.dataset import ZarrPatchDataset
from src.models.unet import UNet
from src.train import train_model


# Generate random experiment ID
def generate_experiment_id(length=4):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml",
        help="Config path",
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    exp_id = generate_experiment_id()
    exp_path = os.path.join(config["EXPERIEMENTS_DIR"], exp_id)
    os.makedirs(exp_path, exist_ok=True)

    logger = setup_logger(exp_path, "experiment")
    start_time = pd.Timestamp.now()
    logger.info(f"Starting experiment (EXPERIMENT ID: {exp_id}, TIME: {start_time})")

    # Load the all_zarr_folder_info map
    METADATA_DIR = config["METADATA_DIR"]
    DEM_PATCH_DIR = config["DEM_PATCH_DIR"]
    zarr_info_map_path = os.path.join(METADATA_DIR, "zarr_info_map.json")
    with open(zarr_info_map_path, "r") as f:
        loaded_zarr_info = json.load(f)
        # Convert lists back to tuples if they were tuples originally
        loaded_zarr_info = {k: tuple(v) for k, v in loaded_zarr_info.items()}

    # Create datasets
    train_dataset = ZarrPatchDataset(
        metadata_file_path=os.path.join(METADATA_DIR, "train_patches_metadata.txt"),
        all_zarr_folder_info=loaded_zarr_info,
        dem_patch_dir=DEM_PATCH_DIR,
        patch_size=config["PATCH_SIZE"],
        downscaling_factor=config["DOWNSCALING_FACTOR"],
        declutter_threshold=config["DECLUTTER_THRESHOLD"],
        transform_input_precip=True,  # Set to False if your model expects raw HR input
    )
    val_dataset = ZarrPatchDataset(
        metadata_file_path=os.path.join(METADATA_DIR, "val_patches_metadata.txt"),
        all_zarr_folder_info=loaded_zarr_info,
        dem_patch_dir=DEM_PATCH_DIR,
        patch_size=config["PATCH_SIZE"],
        downscaling_factor=config["DOWNSCALING_FACTOR"],
        declutter_threshold=config["DECLUTTER_THRESHOLD"],
        transform_input_precip=True,
    )
    test_dataset = ZarrPatchDataset(
        metadata_file_path=os.path.join(METADATA_DIR, "test_patches_metadata.txt"),
        all_zarr_folder_info=loaded_zarr_info,
        dem_patch_dir=DEM_PATCH_DIR,
        patch_size=config["PATCH_SIZE"],
        downscaling_factor=config["DOWNSCALING_FACTOR"],
        declutter_threshold=config["DECLUTTER_THRESHOLD"],
        transform_input_precip=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["NUM_WORKERS"],
        pin_memory=True,
    )

    print(f"Train DataLoader has {len(train_loader)} batches.")
    print(f"Validation DataLoader has {len(val_loader)} batches.")
    print(f"Test DataLoader has {len(test_loader)} batches.")

    # --- Initialize Model, Optimizer, Loss Function ---
    print("\n--- Initializing Model, Optimizer, Loss Function ---")
    model = UNet(dropout_p=config["DROPOUT_PROB"])
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    loss_fn = nn.MSELoss()

    # --- Start Training ---
    print("\n--- Starting Training Process ---")
    _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        config["device"],
        config["NUM_EPOCHS"],
        config["PATIENCE"],
        exp_path,
        config["EARLY_STOPPING"],
    )

    print("\nTraining process complete.")


if __name__ == "__main__":
    main()
