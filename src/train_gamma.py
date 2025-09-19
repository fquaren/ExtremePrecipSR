import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from skimage import measure, morphology
from scipy.ndimage import label
import torch.nn.functional as F
from torch.utils.data import Subset


# Load configuration
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Global configuration parameters
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
# The number of quantiles
N_QUANTILES = len(QUANTILE_LEVELS)
# The total number of output features is 3 (A, P, CC) * number of quantiles
N = N_QUANTILES * 3
PATCH_SIZE = config["PATCH_SIZE"]
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]
DECLUTTER_THRESHOLD = config["DECLUTTER_THRESHOLD"]
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_PATCH_DIR = config["DEM_PATCH_DIR"]
TRAIN_METADATA_FILE = config["TRAIN_METADATA_FILE"]
VAL_METADATA_FILE = config["VAL_METADATA_FILE"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 16)
LEARNING_RATE = config.get("LEARNING_RATE", 1e-4)
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
S_INV_ESTIMATION_SAMPLES = config.get("S_INV_ESTIMATION_SAMPLES", 100)


class GammaPredictor(nn.Module):
    def __init__(self, num_output_features_flat=N, n_quantiles=N_QUANTILES):
        super(GammaPredictor, self).__init__()
        self.n_quantiles = n_quantiles

        # Reduced number of filters in convolutional layers
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

        # Calculate the input size for the first FC layer
        self.fc_input_size = 64 * 16 * 16

        # Reduced number of neurons in fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        # Added dropout layer
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        # Added dropout layer
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_output_features_flat)

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the feature maps
        x = x.view(-1, self.fc_input_size)

        # Apply fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        # Reshape the output
        x = x.view(-1, 3, self.n_quantiles)
        return x


# Loss functions and helper functions
def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = 0
    for contour in contours:
        perimeter_pixels += np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
    perimeter_km = perimeter_pixels * pixel_size_km
    structure = morphology.disk(1)
    _, num_features = label(mask, structure=structure)
    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km=1.0):
    N_thresholds = len(thresholds)
    gamma_matrix = np.zeros((3, N_thresholds), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


def estimate_S_inv_from_dataset_separate(
    dataset_of_target_precip_fields,
    global_quantiles_as_thresholds,
    pixel_size_km,
    regularization_epsilon=1e-6,
):
    all_gamma_A = []
    all_gamma_P = []
    all_gamma_CC = []

    for i, prec_field in enumerate(dataset_of_target_precip_fields):
        gamma_matrix = compute_gamma_matrix_for_image(
            prec_field, global_quantiles_as_thresholds, pixel_size_km
        )
        all_gamma_A.append(gamma_matrix[0, :])
        all_gamma_P.append(gamma_matrix[1, :])
        all_gamma_CC.append(gamma_matrix[2, :])

    if not all_gamma_A:
        raise ValueError(
            "Dataset of target precipitation fields is empty. Cannot estimate S_inv."
        )

    # Convert to numpy arrays
    all_gamma_A_np = np.array(all_gamma_A)
    all_gamma_P_np = np.array(all_gamma_P)
    all_gamma_CC_np = np.array(all_gamma_CC)

    # Compute covariance and inverse for each component
    S_A = np.cov(all_gamma_A_np, rowvar=False)
    S_P = np.cov(all_gamma_P_np, rowvar=False)
    S_CC = np.cov(all_gamma_CC_np, rowvar=False)

    S_A += np.eye(S_A.shape[0]) * regularization_epsilon
    S_P += np.eye(S_P.shape[0]) * regularization_epsilon
    S_CC += np.eye(S_CC.shape[0]) * regularization_epsilon

    S_A_inv = np.linalg.inv(S_A)
    S_P_inv = np.linalg.inv(S_P)
    S_CC_inv = np.linalg.inv(S_CC)

    return S_A_inv, S_P_inv, S_CC_inv


class PreprocessedNpzDataset(Dataset):
    def __init__(self, preprocessed_data_dir, dem_patch_dir, metadata_file):
        print(f"Loading data from {preprocessed_data_dir}...")

        # 1. Read metadata to get the list of all patches
        self.metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                self.metadata.append((parts[0], int(parts[1]), int(parts[2])))

        # 2. Load all precipitation data
        # Only original_precip is needed now
        self.original_patches = np.load(
            os.path.join(preprocessed_data_dir, "original_precip.npz")
        )["data"]

        if len(self.metadata) != self.original_patches.shape[0]:
            raise ValueError(
                f"Number of metadata entries ({len(self.metadata)}) does not match "
                f"number of precipitation patches ({self.original_patches.shape[0]})"
            )

        # 3. DEM patches and other data are no longer needed, so we don't load them
        self.dem_patches = None

        print(f"Loaded {len(self.metadata)} precipitation patches from metadata.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Retrieve the original precipitation patch
        original_precip = self.original_patches[idx]

        # 2. Convert to a PyTorch tensor and add a channel dimension
        # (shape: [1, H, W])
        input_for_model = torch.from_numpy(original_precip).float().unsqueeze(0)

        # 3. The target is the same as the input
        output_precip = input_for_model.clone()

        return input_for_model, output_precip


class GeometricLossSeparate(nn.Module):
    def __init__(self, S_inv_tensors, quantile_levels, pixel_size_km=1.0):
        super(GeometricLossSeparate, self).__init__()
        self.S_A_inv, self.S_P_inv, self.S_CC_inv = S_inv_tensors
        self.quantile_levels = quantile_levels
        self.pixel_size_km = pixel_size_km

    def forward(self, gamma_pred_3d, prec_2d_target_batch):
        prec_2d_target_batch_np = prec_2d_target_batch.squeeze(1).cpu().numpy()
        gamma_target_batch_list = []
        for i in range(prec_2d_target_batch_np.shape[0]):
            gamma_matrix = compute_gamma_matrix_for_image(
                prec_2d_target_batch_np[i],
                self.quantile_levels,
                self.pixel_size_km,
            )
            gamma_target_batch_list.append(gamma_matrix)

        gamma_target_batch_3d = (
            torch.from_numpy(np.array(gamma_target_batch_list))
            .float()
            .to(self.S_A_inv.device)
        )

        # 1. Separate the predicted and target gamma vectors for each component
        pred_A = gamma_pred_3d[:, 0, :]  # Shape: (B, N_quantiles)
        pred_P = gamma_pred_3d[:, 1, :]
        pred_CC = gamma_pred_3d[:, 2, :]

        target_A = gamma_target_batch_3d[:, 0, :]
        target_P = gamma_target_batch_3d[:, 1, :]
        target_CC = gamma_target_batch_3d[:, 2, :]

        # 2. Compute the Mahalanobis distance for each component
        diff_A = pred_A - target_A
        loss_A_sq = torch.sum((diff_A @ self.S_A_inv) * diff_A, dim=1)

        diff_P = pred_P - target_P
        loss_P_sq = torch.sum((diff_P @ self.S_P_inv) * diff_P, dim=1)

        diff_CC = pred_CC - target_CC
        loss_CC_sq = torch.sum((diff_CC @ self.S_CC_inv) * diff_CC, dim=1)

        # 3. Sum the square roots and take the mean over the batch
        total_loss = torch.mean(
            torch.sqrt(loss_A_sq) + torch.sqrt(loss_P_sq) + torch.sqrt(loss_CC_sq)
        )

        return total_loss


# New Combined Loss Class
class CombinedLoss(nn.Module):
    def __init__(self, geometric_loss, normal_loss, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.geometric_loss = geometric_loss
        self.normal_loss = normal_loss
        self.alpha = alpha

    def forward(self, pred_gamma, target_gamma_precip):
        # Geometric loss on the 3D tensors (will internally compute target gamma)
        geo_loss = self.geometric_loss.forward(pred_gamma, target_gamma_precip)

        # Calculate the normal loss (L1/MAE)
        prec_2d_target_batch_np = target_gamma_precip.squeeze(1).cpu().numpy()
        gamma_target_batch_list = []
        for i in range(prec_2d_target_batch_np.shape[0]):
            gamma_matrix = compute_gamma_matrix_for_image(
                prec_2d_target_batch_np[i],
                self.geometric_loss.quantile_levels,
                self.geometric_loss.pixel_size_km,
            )
            gamma_target_batch_list.append(gamma_matrix)
        gamma_target_batch_3d = (
            torch.from_numpy(np.array(gamma_target_batch_list))
            .float()
            .to(pred_gamma.device)
        )

        # Flatten the predicted gamma and target gamma for L1 loss
        pred_gamma_flat = pred_gamma.view(pred_gamma.shape[0], -1)
        target_gamma_flat = gamma_target_batch_3d.view(
            gamma_target_batch_3d.shape[0], -1
        )

        normal_l1_loss = self.normal_loss(pred_gamma_flat, target_gamma_flat)

        # Return the weighted sum
        return geo_loss + self.alpha * normal_l1_loss, geo_loss, normal_l1_loss


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Estimate separate S_inv matrices from a subset of the training data ---
print("Estimating separate S_inv matrices from training data subset...")
temp_dataset_for_S_inv = PreprocessedNpzDataset(
    preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
    dem_patch_dir=DEM_PATCH_DIR,
    metadata_file=TRAIN_METADATA_FILE,
)

# Sample a subset of target precipitation fields to estimate S_inv
num_samples_for_S_inv = min(S_INV_ESTIMATION_SAMPLES, len(temp_dataset_for_S_inv))
indices = torch.randperm(len(temp_dataset_for_S_inv)).tolist()[:num_samples_for_S_inv]

target_precip_fields_for_S_inv = []
for i in tqdm(indices, desc="Collecting S_inv estimation samples"):
    _, target_precip = temp_dataset_for_S_inv[i]
    target_precip_fields_for_S_inv.append(target_precip.squeeze(0).numpy())

# The new function call returns three matrices
S_A_inv, S_P_inv, S_CC_inv = estimate_S_inv_from_dataset_separate(
    target_precip_fields_for_S_inv,
    global_quantiles_as_thresholds=QUANTILE_LEVELS,
    pixel_size_km=1.0,
)
print("Separate S_inv estimation complete.")

# Convert each numpy array to a PyTorch tensor
S_A_inv_torch = torch.from_numpy(S_A_inv).float().to(device)
S_P_inv_torch = torch.from_numpy(S_P_inv).float().to(device)
S_CC_inv_torch = torch.from_numpy(S_CC_inv).float().to(device)


# --- 2. Prepare Datasets and DataLoaders ---
train_dataset_full = PreprocessedNpzDataset(
    preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
    dem_patch_dir=DEM_PATCH_DIR,
    metadata_file=TRAIN_METADATA_FILE,
)

val_dataset_full = PreprocessedNpzDataset(
    preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
    dem_patch_dir=DEM_PATCH_DIR,
    metadata_file=VAL_METADATA_FILE,
)

test_dataset_full = PreprocessedNpzDataset(
    preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
    dem_patch_dir=DEM_PATCH_DIR,
    metadata_file=TEST_METADATA_FILE,
)


# Helper function to subsample 10% with fixed seed
def subsample_dataset(dataset, fraction=0.1, seed=42):
    dataset_size = len(dataset)
    subset_size = int(fraction * dataset_size)
    g = torch.Generator().manual_seed(seed)  # fixed seed
    subset_indices = torch.randperm(dataset_size, generator=g)[:subset_size]
    return Subset(dataset, subset_indices)


# Subsample train, val, and test consistently
train_dataset = subsample_dataset(train_dataset_full, 0.1, seed=42)
val_dataset = subsample_dataset(
    val_dataset_full, 0.1, seed=123
)  # different seed if you want different samples
test_dataset = subsample_dataset(test_dataset_full, 0.1, seed=456)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=config.get("NUM_WORKERS", os.cpu_count() // 2),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=config.get("NUM_WORKERS", os.cpu_count() // 2),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=config.get("NUM_WORKERS", os.cpu_count() // 2),
)

print(
    f"Training with {len(train_dataset)} samples (10% of train), "
    f"validating with {len(val_dataset)} samples (10% of val)."
    f"Testing with {len(test_dataset)} samples (10% of test)."
)
# --- 3. Initialize Model, Optimizer, and Loss Function ---
model = GammaPredictor(num_output_features_flat=N, n_quantiles=N_QUANTILES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

# Initialize the new loss classes with the three matrices
geometric_criterion = GeometricLossSeparate(
    S_inv_tensors=(S_A_inv_torch, S_P_inv_torch, S_CC_inv_torch),
    quantile_levels=QUANTILE_LEVELS,
).to(device)
normal_l1_criterion = nn.L1Loss().to(device)

# The new main criterion
criterion = CombinedLoss(geometric_criterion, normal_l1_criterion, alpha=0.5).to(device)

# --- 4. Training Loop ---
print("Starting training...")
best_val_loss = float("inf")
count_early_stop = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_geo_loss = 0.0
    running_l1_loss = 0.0
    for input_data, target_precip in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"
    ):
        input_data = input_data.to(device)
        target_precip = target_precip.to(device)

        optimizer.zero_grad()
        # The model's output is now (B, 3, N_quantiles)
        predicted_gamma_3d = model(input_data)

        # Unpack the three returned values from CombinedLoss
        # The criterion handles the reshaping internally
        loss, geo_loss, l1_loss = criterion(predicted_gamma_3d, target_precip)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_geo_loss += geo_loss.item()
        running_l1_loss += l1_loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_train_geo_loss = running_geo_loss / len(train_loader)
    avg_train_l1_loss = running_l1_loss / len(train_loader)

    print(
        f"Epoch {epoch+1} finished. Avg Total Loss: {avg_train_loss:.4f} | Avg Geometric Loss: {avg_train_geo_loss:.4f} | Avg L1 Loss: {avg_train_l1_loss:.4f}"
    )

    # --- 5. Validation Loop ---
    model.eval()
    val_running_loss = 0.0
    val_running_geo_loss = 0.0
    val_running_l1_loss = 0.0
    with torch.no_grad():
        for input_data, target_precip in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)"
        ):
            input_data = input_data.to(device)
            target_precip = target_precip.to(device)
            # The model's output is now (B, 3, N_quantiles)
            predicted_gamma_3d = model(input_data)

            # Unpack the three returned values from CombinedLoss
            # The criterion handles the reshaping internally
            loss, geo_loss, l1_loss = criterion(predicted_gamma_3d, target_precip)

            val_running_loss += loss.item()
            val_running_geo_loss += geo_loss.item()
            val_running_l1_loss += l1_loss.item()

    avg_val_loss = val_running_loss / len(val_loader)
    avg_val_geo_loss = val_running_geo_loss / len(val_loader)
    avg_val_l1_loss = val_running_l1_loss / len(val_loader)

    scheduler.step(avg_val_loss)
    print(
        f"Epoch {epoch+1} finished. Avg Val Total Loss: {avg_val_loss:.4f} | Avg Val Geometric Loss: {avg_val_geo_loss:.4f} | Avg Val L1 Loss: {avg_val_l1_loss:.4f}"
    )

    # Early stopping or model checkpointing can be added here if needed
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_gamma_predictor_model.pth")
        print("Model checkpoint saved.")
    else:
        print("No improvement in validation loss.")
        count_early_stop += 1

    if count_early_stop >= config.get("EARLY_STOPPING_PATIENCE", 10):
        print("Early stopping triggered.")
        break

print("Training complete.")

# --- 6. Save the trained model ---
model_save_path = "gamma_predictor_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the best model for evaluation
model.load_state_dict(torch.load("best_gamma_predictor_model.pth"))
model.eval()
# --- 7. Testing Loop (if you have a test set) ---
test_running_loss = 0.0
test_running_geo_loss = 0.0
test_running_l1_loss = 0.0
with torch.no_grad():
    for input_data, target_precip in tqdm(test_loader, desc="Testing"):
        input_data = input_data.to(device)
        target_precip = target_precip.to(device)
        # The model's output is now (B, 3, N_quantiles)
        predicted_gamma_3d = model(input_data)

        # The criterion handles the reshaping internally
        loss, geo_loss, l1_loss = criterion(predicted_gamma_3d, target_precip)

        test_running_loss += loss.item()
        test_running_geo_loss += geo_loss.item()
        test_running_l1_loss += l1_loss.item()

avg_test_loss = test_running_loss / len(test_loader)
avg_test_geo_loss = test_running_geo_loss / len(test_loader)
avg_test_l1_loss = test_running_l1_loss / len(test_loader)
print(
    f"Testing complete. Avg Test Total Loss: {avg_test_loss:.4f} | Avg Test Geometric Loss: {avg_test_geo_loss:.4f} | Avg Test L1 Loss: {avg_test_l1_loss:.4f}"
)
