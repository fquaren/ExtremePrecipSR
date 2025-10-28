import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import random
from torch.utils.data import Sampler
import itertools
import math  # Needed for pi

# --- Configuration Loading ---
config_path = (
    # "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
    "/home/fquareng/work/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
N = N_QUANTILES * 3
PATCH_SIZE = config["PATCH_SIZE"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 16)
LEARNING_RATE = config.get("LEARNING_RATE", 1e-4)
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-5)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
EARLY_STOPPING_PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
EARLY_STOPPING_DELTA = config.get("EARLY_STOPPING_DELTA", 0.001)
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME", "Debugging")
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)


# Model now includes hard zero constraint
class GammaPredictorHardConstraints(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        num_output_features_flat=N,
        n_quantiles=N_QUANTILES,
        activation_fn=F.gelu,
        quantile_levels=QUANTILE_LEVELS,
        pixel_area_km2=PIXEL_SIZE_KM**2,
    ):
        super(GammaPredictorHardConstraints, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_input_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_output_features_flat)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        # --- Feature Extraction ---
        x_conv = self._forward_conv(x)
        x_flat = x_conv.view(-1, self.fc_input_size)
        x_fc = self.activation(self.fc1(x_flat))
        x_fc = self.dropout1(x_fc)
        x_fc = self.activation(self.fc2(x_fc))
        x_fc = self.dropout2(x_fc)
        raw_output = self.fc3(x_fc)  # Shape [B, 3 * NQ]

        # --- Reconstruct A and P with hard constraints ---
        raw_A_logits = raw_output[:, 0 * self.n_quantiles : 1 * self.n_quantiles]
        raw_P_logits = raw_output[:, 1 * self.n_quantiles : 2 * self.n_quantiles]
        raw_CC_pred = raw_output[:, 2 * self.n_quantiles : 3 * self.n_quantiles]

        with torch.no_grad():  # A_total calc doesn't need grad w.r.t input
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + 1e-6
            A_total = A_total.unsqueeze(1)  # [B, 1]

        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )  # [B, NQ]

        epsilon = 1e-6
        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        P_excess = F.relu(raw_P_logits)  # Or F.softplus
        pred_P = P_min + P_excess  # [B, NQ]

        pred_CC = F.relu(raw_CC_pred)  # [B, NQ]

        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)  # [B, 3, NQ]

        # --- Apply Hard Zero Constraint based on Input ---
        with torch.no_grad():
            is_dry_mask = x.sum(dim=(1, 2, 3)) <= 1e-6  # Shape [B]
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)  # Shape [B, 1, 1]

        final_output = constrained_output * wet_factor

        return final_output


# --- Data Handling & Augmentation (using log1p transform) ---
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.01):
        self.std, self.mean = std, mean

    def __call__(self, tensor):
        return (
            tensor
            + torch.randn(tensor.size(), device=tensor.device) * self.std
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


class PreprocessedNpzDataset(Dataset):
    def __init__(
        self, preprocessed_data_dir, metadata_file, augment=False, noise_std=0.01
    ):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            self.metadata = [line.strip().split(",") for line in f]
        precip_path = os.path.join(preprocessed_data_dir, "original_precip.npz")
        gamma_path = os.path.join(preprocessed_data_dir, "gamma_targets.npz")
        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]
        self.augment = augment
        if self.augment:
            rotations = T.RandomChoice(
                [
                    T.RandomRotation([0, 0]),
                    T.RandomRotation([90, 90]),
                    T.RandomRotation([180, 180]),
                    T.RandomRotation([270, 270]),
                ]
            )
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    rotations,
                    T.RandomApply([AddGaussianNoise(0.0, noise_std)], p=0.2),
                ]
            )
            print("Data augmentation is enabled for this dataset.")
        if len(self.metadata) != self.original_patches.shape[0]:
            raise ValueError("Metadata/patch count mismatch.")
        if self.original_patches.shape[0] != self.gamma_targets.shape[0]:
            raise ValueError("Precipitation/Gamma count mismatch.")
        print(f"Loaded {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        original_precip = self.original_patches[idx]
        target_gamma = self.gamma_targets[idx]
        input_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        target_gamma_tensor = torch.from_numpy(target_gamma).float()
        log_target_gamma_tensor = torch.log1p(
            target_gamma_tensor
        )  # Apply log transform
        if self.augment:
            input_tensor = self.transform(input_tensor)
        return input_tensor, log_target_gamma_tensor  # Return input and LOG TARGET


# --- Stratified Sampler ---
class StratifiedBatchSampler(Sampler):
    def __init__(self, indices_dry, indices_normal, indices_extreme, batch_composition):
        self.indices_dry, self.indices_normal, self.indices_extreme = (
            indices_dry,
            indices_normal,
            indices_extreme,
        )
        self.batch_composition = batch_composition
        self.batch_size = sum(batch_composition.values())
        if not self.indices_extreme or self.batch_composition.get("extreme", 0) == 0:
            self.num_batches = 0
        else:
            self.num_batches = (
                len(self.indices_extreme) // self.batch_composition["extreme"]
            )

    def __iter__(self):
        dry_iter = iter(
            itertools.cycle(random.sample(self.indices_dry, len(self.indices_dry)))
        )
        normal_iter = iter(
            itertools.cycle(
                random.sample(self.indices_normal, len(self.indices_normal))
            )
        )
        extreme_iter = iter(
            random.sample(self.indices_extreme, len(self.indices_extreme))
        )
        for _ in range(self.num_batches):
            batch = []
            try:
                batch.extend(
                    [
                        next(extreme_iter)
                        for _ in range(self.batch_composition["extreme"])
                    ]
                )
                batch.extend(
                    [next(normal_iter) for _ in range(self.batch_composition["normal"])]
                )
                batch.extend(
                    [next(dry_iter) for _ in range(self.batch_composition["dry"])]
                )
            except StopIteration:
                break
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# --- Loss Function ---
class ComponentWiseCDFLoss(nn.Module):
    def __init__(self, quantile_levels):
        super(ComponentWiseCDFLoss, self).__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantile_levels, dtype=torch.float32)
        )

    def forward(self, gamma_pred_3d, gamma_target_3d):  # Expects LOG SPACE values
        abs_diff_log = torch.abs(gamma_pred_3d - gamma_target_3d)
        integrand = abs_diff_log * self.quantiles
        integral_per_component = torch.trapezoid(integrand, self.quantiles, dim=2)
        return (
            torch.mean(integral_per_component[:, 0]),
            torch.mean(integral_per_component[:, 1]),
            torch.mean(integral_per_component[:, 2]),
        )


# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{EXPERIMENT_NAME}_{timestamp}"
    output_dir = os.path.join("experiment_runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving experiment artifacts to: {output_dir}")
    try:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
    except IOError as e:
        print(f"Error saving config file: {e}")

    # --- 1. Prepare Datasets ---
    train_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
        augment=True,
    )
    val_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
        augment=False,
    )

    def subsample_dataset(dataset, fraction=0.1, seed=42):
        num_samples = int(fraction * len(dataset))
        if num_samples == 0 and len(dataset) > 0:
            num_samples = 1
        subset_indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )[:num_samples]
        return Subset(dataset, subset_indices)

    val_dataset = subsample_dataset(
        val_dataset_full, config["SUBSAMPLE_FRACTION"], seed=0
    )

    # --- Oversampling Strategy ---
    print("Stratifying dataset using MAX precipitation...")
    wet_event_metrics = [
        np.max(patch)
        for patch in train_dataset_full.original_patches
        if np.max(patch) > 1e-6
    ]
    if wet_event_metrics:
        extreme_threshold = np.percentile(wet_event_metrics, 95)
    else:
        extreme_threshold = float("inf")
    print(
        f"Data-driven threshold (95th percentile of MAX precip): {extreme_threshold:.4f} mm/hr"
    )
    indices_dry, indices_normal, indices_extreme = [], [], []
    for i, patch in enumerate(train_dataset_full.original_patches):
        metric = np.max(patch)
        if metric <= 1e-6:
            indices_dry.append(i)
        elif metric < extreme_threshold:
            indices_normal.append(i)
        else:
            indices_extreme.append(i)
    print(
        f"Stratification complete: {len(indices_dry)} \n"
        f"Dry, {len(indices_normal)} Normal, {len(indices_extreme)} Extreme."
    )
    batch_composition = {
        "dry": int(BATCH_SIZE / 4),
        "normal": int(BATCH_SIZE / 2),
        "extreme": int(BATCH_SIZE / 4),
    }
    stratified_sampler = StratifiedBatchSampler(
        indices_dry, indices_normal, indices_extreme, batch_composition
    )

    # --- Prepare DataLoaders ---
    train_loader = DataLoader(
        train_dataset_full,
        batch_sampler=stratified_sampler,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=config.get("NUM_WORKERS", 0),
        pin_memory=True,
    )

    # --- 3. Initialize Model, Optimizer, and Loss ---
    log_var_A = nn.Parameter(torch.zeros((1,), device=device))
    log_var_P = nn.Parameter(torch.zeros((1,), device=device))
    log_var_CC = nn.Parameter(torch.zeros((1,), device=device))
    # Use the new model class with hard constraints
    model = GammaPredictorHardConstraints(
        activation_fn=F.mish,  # Ensure activation matches if needed
        quantile_levels=QUANTILE_LEVELS,
        pixel_area_km2=PIXEL_SIZE_KM**2,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_var_A, log_var_P, log_var_CC],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = ComponentWiseCDFLoss(quantile_levels=QUANTILE_LEVELS).to(device)

    log_file_path = os.path.join(output_dir, "training_log.csv")
    try:
        with open(log_file_path, "w") as log_file:
            # Update log header - remove zero penalty columns
            log_file.write(
                "epoch,train_loss_total,train_loss_main,"
                "train_penalty_bound,"  # Removed mono, plaus, zero
                "val_loss_total,val_loss_main,"
                "val_penalty_bound,"  # Removed mono, plaus, zero
                "sigma_A,sigma_P,sigma_CC\n"
            )
        print(f"Log file will be saved to {log_file_path}")
    except IOError as e:
        print(f"Error creating log file: {e}")
        exit()

    # --- 4. Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, running_main_loss = 0.0, 0.0

        for input_data, log_target_gamma in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"
        ):
            input_data, log_target_gamma = input_data.to(device), log_target_gamma.to(
                device
            )
            optimizer.zero_grad()
            # Model output is physical space due to hard constraints (including zero)
            predicted_gamma_phys = model(input_data)

            # Need log-space prediction for the main loss comparison
            predicted_gamma_log = torch.log1p(predicted_gamma_phys)

            # --- Calculate Main Homoscedastic Loss (in log-space) ---
            loss_A, loss_P, loss_CC = criterion(predicted_gamma_log, log_target_gamma)
            term_A = torch.exp(-log_var_A) * loss_A + log_var_A
            term_P = torch.exp(-log_var_P) * loss_P + log_var_P
            term_CC = torch.exp(-log_var_CC) * loss_CC + log_var_CC
            main_loss_homo = (term_A + term_P + term_CC) * 0.5

            # --- Combine ALL losses ---
            # Zero penalty removed from total loss
            total_loss = main_loss_homo

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # --- Accumulate losses for logging ---
            running_loss += total_loss.item()
            running_main_loss += main_loss_homo.item()

        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_main_loss = running_main_loss / num_batches if num_batches > 0 else 0

        # Update print statement
        print(
            f"Epoch {epoch+1}\n"
            f"Train Loss: Total={avg_train_loss:.4f} (Main={avg_main_loss:.4f})"
        )
        # --- Validation ---
        model.eval()
        # Remove zero penalty accumulator
        val_running_loss, val_running_main_loss = 0.0, 0.0

        with torch.no_grad():
            for input_data, log_target_gamma in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                input_data, log_target_gamma = input_data.to(
                    device
                ), log_target_gamma.to(device)
                predicted_gamma_phys = model(input_data)  # Physical space output
                predicted_gamma_log = torch.log1p(
                    predicted_gamma_phys
                )  # Log space for loss

                # --- Calculate Main Homoscedastic Loss (log-space)---
                loss_A, loss_P, loss_CC = criterion(
                    predicted_gamma_log, log_target_gamma
                )
                term_A = torch.exp(-log_var_A) * loss_A + log_var_A
                term_P = torch.exp(-log_var_P) * loss_P + log_var_P
                term_CC = torch.exp(-log_var_CC) * loss_CC + log_var_CC
                main_loss_homo = (term_A + term_P + term_CC) * 0.5

                # --- Combine ALL losses ---
                total_loss = main_loss_homo

                # --- Accumulate validation losses ---
                val_running_loss += total_loss.item()
                val_running_main_loss += main_loss_homo.item()

        # Calculate average validation losses
        num_val_batches = len(val_loader)
        avg_val_loss = val_running_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_main_loss = (
            val_running_main_loss / num_val_batches if num_val_batches > 0 else 0
        )

        scheduler.step(avg_val_loss)
        sigma_A = torch.sqrt(torch.exp(log_var_A)).item()
        sigma_P = torch.sqrt(torch.exp(log_var_P)).item()
        sigma_CC = torch.sqrt(torch.exp(log_var_CC)).item()

        # Update print statement
        print(
            f"Epoch {epoch+1}\n"
            f"Val Loss: Total={avg_val_loss:.4f} (Main={avg_val_main_loss:.4f})\n"
            f"Sigmas: A={sigma_A:.3f}, P={sigma_P:.3f}, CC={sigma_CC:.3f}"
        )
        if avg_val_loss < best_val_loss - EARLY_STOPPING_DELTA:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "log_var_A": log_var_A,
                "log_var_P": log_var_P,
                "log_var_CC": log_var_CC,
            }
            model_save_path = os.path.join(output_dir, "best_model_checkpoint.pth")
            torch.save(checkpoint, model_save_path)
            print(
                f"Validation loss decreased significantly to {best_val_loss:.6f}. Model checkpoint saved."
            )
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No significant improvement for {patience_counter} epoch(s).")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs."
                )
                break

        try:
            with open(log_file_path, "a") as log_file:
                # Update log record format
                log_file.write(
                    f"{epoch+1},{avg_train_loss:.6f},{avg_main_loss:.6f},"
                    f"{avg_val_loss:.6f},{avg_val_main_loss:.6f},"
                    f"{sigma_A:.6f},{sigma_P:.6f},{sigma_CC:.6f}\n"
                )
        except IOError as e:
            print(f"Error writing to log file: {e}")

    print("Training complete.")
