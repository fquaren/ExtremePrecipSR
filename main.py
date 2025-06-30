import os
import torch
import torch.nn as nn
import torch.optim as optim


from src.dataset import SplitPrecipitationDataset
from src.dataloader import create_dataloader
from src.evaluate import evaluate_model
from src.models.unet import UNet
from src.train import train_model
from src.utils import get_file_paths_from_list


def main():
    # --- Configuration ---
    # Define the base directory for your processed data
    output_base_dir = (
        "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/OPERA/patches"
    )
    precip_npz_base_dir = os.path.join(output_base_dir, "preprocessed_data")
    dem_npy_dir = os.path.join(output_base_dir, "dem")
    checkpoint_dir = os.path.join(output_base_dir, "checkpoints")

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    DROPOUT_PROB = 0.3
    EARLY_STOPPING = True
    PATIENCE = 5

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths to the lists of train/val/test NPZ files (from the previous script's output)
    train_npz_list_path = os.path.join(precip_npz_base_dir, "train_files.txt")
    val_npz_list_path = os.path.join(precip_npz_base_dir, "val_files.txt")
    test_npz_list_path = os.path.join(precip_npz_base_dir, "test_files.txt")

    try:
        train_npz_files = get_file_paths_from_list(train_npz_list_path)
        val_npz_files = get_file_paths_from_list(val_npz_list_path)
        test_npz_files = get_file_paths_from_list(test_npz_list_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Cannot proceed without train/val file lists.")
        exit()

    # --- Create Datasets and DataLoaders ---
    print("\n--- Creating Datasets ---")
    train_dataset = SplitPrecipitationDataset(train_npz_files, dem_npy_dir)
    val_dataset = SplitPrecipitationDataset(val_npz_files, dem_npy_dir)
    test_dataset = SplitPrecipitationDataset(test_npz_files, dem_npy_dir)

    print("\n--- Creating DataLoaders ---")
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train DataLoader has {len(train_loader)} batches of size {BATCH_SIZE}")
    print(f"Validation DataLoader has {len(val_loader)} batches of size {BATCH_SIZE}")

    # --- Initialize Model, Optimizer, Loss Function ---
    print("\n--- Initializing Model, Optimizer, Loss Function ---")
    model = UNet(dropout_prob=DROPOUT_PROB)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # --- Start Training ---
    print("\n--- Starting Training Process ---")
    best_model_path = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        NUM_EPOCHS,
        PATIENCE,
        checkpoint_dir,
        EARLY_STOPPING,
    )

    print("\nTraining process complete.")

    print("\n--- Starting Evaluation Process ---")
    test_loss = evaluate_model(
        model,
        test_loader,
        loss_fn,
        device,
        model_path=best_model_path,
    )
    print(f"Evaluation complete. Final Test Loss: {test_loss:.6f}")
