import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from datetime import datetime
import json

from model import UNetSR
from dataset import SRDataset
from gamma_predictors import (
    GammaPredictorSeparateHeadsHard,
    GammaPredictorSeparateHeadsSoft,
)
from loss import ComponentWiseCDFLoss

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- SR Task Config ---
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_DATA_DIR = config["DEM_DATA_DIR"]
DEM_STATS = config["DEM_STATS"]
METADATA_TRAIN_METADATA_FILE = config["METADATA_TRAIN_METADATA_FILE"]
METADATA_VAL_METADATA_FILE = config["METADATA_VAL_METADATA_FILE"]

GAMMA_TARGETS_DIR = config["PREPROCESSED_DATA_DIR"]
BATCH_SIZE = config.get("SR_BATCH_SIZE", 16)
LEARNING_RATE = config.get("SR_LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("SR_WEIGHT_DECAY", 1e-5)
NUM_EPOCHS = config.get("SR_NUM_EPOCHS", 50)
NUM_WORKERS = config.get("NUM_WORKERS", 16)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "SR_UNet_Baseline")

# --- Surrogate Loss Config ---
USE_SURROGATE_LOSS = config.get("USE_SURROGATE_LOSS", False)
SURROGATE_LOSS_WEIGHT = config.get("SURROGATE_LOSS_WEIGHT", 0.1)
EMULATOR_CHECKPOINT_PATH = config.get("EMULATOR_CHECKPOINT_PATH", None)
EMULATOR_IS_HARD_CONSTRAINED = config.get("HARD_CONSTRAINED", True)

# --- Emulator Model Config (Needed to load the checkpoint) ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)


# --- Helper to load the emulator ---
def load_emulator(checkpoint_path, is_hard_constrained, device):
    print(f"Loading Gamma Emulator from: {checkpoint_path}")

    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)

    if is_hard_constrained:
        model = GammaPredictorSeparateHeadsHard(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=F.mish,
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        )
    else:
        model = GammaPredictorSeparateHeadsSoft(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=F.mish,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print("Emulator loaded, set to eval mode, and weights are frozen.")
    return model


# --- Main Execution ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir = os.path.join("sr_experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving SR experiment artifacts to: {output_dir}")
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    dem_stats = json.load(open(DEM_STATS, "r"))

    # --- 1. Prepare Datasets ---
    train_dataset = SRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_TRAIN_METADATA_FILE,
        DEM_DATA_DIR,
        dem_stats,
        split="train",
    )
    val_dataset = SRDataset(
        PREPROCESSED_DATA_DIR,
        METADATA_VAL_METADATA_FILE,
        DEM_DATA_DIR,
        dem_stats,
        split="validation",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- 2. Initialize Models and Losses ---

    # --- SR UNet (The model we are training) ---
    # We use the UNet from your model.py.
    sr_model = UNetSR(in_channels=1, out_channels=1).to(device)

    optimizer = torch.optim.Adam(
        sr_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --- Primary Loss: MSE ---
    mse_criterion = nn.MSELoss()

    # --- Surrogate Loss (Optional) ---
    emulator_model = None
    surrogate_criterion = None
    if USE_SURROGATE_LOSS:
        if not EMULATOR_CHECKPOINT_PATH:
            raise ValueError(
                "USE_SURROGATE_LOSS is True, but EMULATOR_CHECKPOINT_PATH is not set in config."
            )

        emulator_model = load_emulator(
            EMULATOR_CHECKPOINT_PATH, EMULATOR_IS_HARD_CONSTRAINED, device
        )
        # We use the CDF loss in log-space, just like the emulator was trained
        surrogate_criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(
            device
        )

    # --- 3. Setup Logging ---
    log_file_path = os.path.join(output_dir, "sr_training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            "epoch,train_loss_total,train_loss_mse,train_loss_surrogate,"
            "val_loss_total,val_loss_mse,val_loss_surrogate\n"
        )

    # --- 4. Training & Validation Loop ---
    print(f"Starting SR training... | Surrogate Loss: {USE_SURROGATE_LOSS}")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        sr_model.train()
        running_loss, running_mse, running_surr = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")
        for X, Y_residual, Y_original, Y_gamma in pbar:

            X, Y_residual, Y_original, Y_gamma = (
                X.to(device),
                Y_residual.to(device),
                Y_original.to(device),
                Y_gamma.to(device),
            )

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                # Predict the residual
                pred_residual = sr_model(X)

                # --- 1. Calculate Primary MSE Loss ---
                loss_mse = mse_criterion(pred_residual, Y_residual)

                total_loss = loss_mse
                loss_surrogate = torch.tensor(0.0, device=device)

                # --- 2. Calculate Surrogate Loss (if enabled) ---
                if USE_SURROGATE_LOSS:
                    # Reconstruct the full precipitation image
                    pred_image = X + pred_residual

                    # Pass both pred and target images through the emulator
                    # The emulator expects physical, non-negative images
                    pred_gamma_phys = emulator_model(F.relu(pred_image))

                    # We compare in log-space, just like the emulator was trained
                    pred_gamma_log = torch.log1p(pred_gamma_phys)
                    true_gamma_log = torch.log1p(Y_gamma)

                    # Calculate the CDF loss
                    loss_A, loss_P, loss_CC = surrogate_criterion(
                        pred_gamma_log, true_gamma_log
                    )
                    loss_surrogate = loss_A + loss_P + loss_CC

                    # Add to the total loss
                    total_loss = (
                        1 - SURROGATE_LOSS_WEIGHT
                    ) * loss_mse + SURROGATE_LOSS_WEIGHT * loss_surrogate

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # --- Accumulate losses for logging ---
            running_loss += total_loss.item()
            running_mse += loss_mse.item()
            running_surr += loss_surrogate.item()

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                surr=f"{loss_surrogate.item():.4f}",
            )

        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches
        avg_train_mse = running_mse / num_batches
        avg_train_surr = running_surr / num_batches

        print(
            f"Epoch {epoch+1}\n"
            f"Train Loss: Total={avg_train_loss:.5f}, MSE={avg_train_mse:.5f}, "
            f"Surrogate={avg_train_surr:.5f}"
        )

        # --- Validation ---
        sr_model.eval()
        val_running_loss, val_running_mse, val_running_surr = 0.0, 0.0, 0.0

        with torch.no_grad():
            for X, Y_residual, Y_original, Y_gamma in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                X, Y_residual, Y_original, Y_gamma = (
                    X.to(device),
                    Y_residual.to(device),
                    Y_original.to(device),
                    Y_gamma.to(device),
                )

                with torch.amp.autocast(device_type="cuda"):
                    pred_residual = sr_model(X)

                    # --- 1. Calculate Primary MSE Loss ---
                    loss_mse = mse_criterion(pred_residual, Y_residual)

                    total_loss = loss_mse
                    loss_surrogate = torch.tensor(0.0, device=device)

                    # --- 2. Calculate Surrogate Loss (if enabled) ---
                    if USE_SURROGATE_LOSS:
                        pred_image = X + pred_residual
                        pred_gamma_phys = emulator_model(F.relu(pred_image))

                        pred_gamma_log = torch.log1p(pred_gamma_phys)
                        true_gamma_log = torch.log1p(Y_gamma)

                        loss_A, loss_P, loss_CC = surrogate_criterion(
                            pred_gamma_log, true_gamma_log
                        )
                        loss_surrogate = loss_A + loss_P + loss_CC
                        total_loss = (
                            1 - SURROGATE_LOSS_WEIGHT
                        ) * loss_mse + SURROGATE_LOSS_WEIGHT * loss_surrogate

                val_running_loss += total_loss.item()
                val_running_mse += loss_mse.item()
                val_running_surr += loss_surrogate.item()

        num_val_batches = len(val_loader)
        avg_val_loss = val_running_loss / num_val_batches
        avg_val_mse = val_running_mse / num_val_batches
        avg_val_surr = val_running_surr / num_val_batches

        scheduler.step(avg_val_loss)
        print(
            f"Val Loss:   Total={avg_val_loss:.5f}, MSE={avg_val_mse:.5f}, "
            f"Surrogate={avg_val_surr:.5f}\n"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": sr_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            model_save_path = os.path.join(output_dir, "best_sr_model.pth")
            torch.save(checkpoint, model_save_path)
            print(f"Validation loss decreased to {best_val_loss:.6f}. Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:  # Simple patience
                print("Validation loss did not improve. Stopping early.")
                break

        # Log to file
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{epoch+1},{avg_train_loss:.6f},{avg_train_mse:.6f},{avg_train_surr:.6f},"
                f"{avg_val_loss:.6f},{avg_val_mse:.6f},{avg_val_surr:.6f}\n"
            )

    print("SR Training complete.")


if __name__ == "__main__":
    main()
