import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
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
NUM_EPOCHS = config.get("NUM_EPOCHS", 10)
S_ESTIMATION_SAMPLES = config.get("S_ESTIMATION_SAMPLES", 100)


# --- Model Definition (with Robustness Improvement) ---
class GammaPredictor(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        num_output_features_flat=N,
        n_quantiles=N_QUANTILES,
    ):
        super(GammaPredictor, self).__init__()
        self.n_quantiles = n_quantiles

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

        # Dynamically calculate the flattened size for robustness.
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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 3, self.n_quantiles)
        return x


# --- Data Handling ---
class PreprocessedNpzDataset(Dataset):
    def __init__(self, preprocessed_data_dir, metadata_file):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            self.metadata = [line.strip().split(",") for line in f]

        precip_path = os.path.join(preprocessed_data_dir, "original_precip.npz")
        gamma_path = os.path.join(preprocessed_data_dir, "gamma_targets.npz")

        self.original_patches = np.load(precip_path, mmap_mode="r")["data"]
        self.gamma_targets = np.load(gamma_path, mmap_mode="r")["data"]

        if len(self.metadata) != self.original_patches.shape[0]:
            raise ValueError("Metadata count does not match precipitation patch count.")
        if self.original_patches.shape[0] != self.gamma_targets.shape[0]:
            raise ValueError("Precipitation patches and Gamma targets count mismatch.")

        print(
            f"Loaded {len(self.metadata)} samples (precipitation and pre-computed targets)."
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Directly load the pre-computed gamma target.
        original_precip = self.original_patches[idx]
        target_gamma = self.gamma_targets[idx]

        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(original_precip).float().unsqueeze(0)
        target_gamma_tensor = torch.from_numpy(target_gamma).float()

        return input_tensor, target_gamma_tensor


# --- S Matrix Estimation (using pre-computed targets) ---
def estimate_S_from_precomputed(gamma_targets_dataset):
    """Estimates covariance matrices S directly from pre-computed gamma targets."""
    # Stack all gamma targets into a single array
    all_gamma = np.stack(
        [gamma_targets_dataset[i][1].numpy() for i in range(len(gamma_targets_dataset))]
    )

    # Separate into A, P, CC
    all_gamma_A = all_gamma[:, 0, :]
    all_gamma_P = all_gamma[:, 1, :]
    all_gamma_CC = all_gamma[:, 2, :]

    # Compute covariance and apply regularization
    S_A = np.cov(all_gamma_A, rowvar=False) + np.eye(N_QUANTILES) * 1e-6
    S_P = np.cov(all_gamma_P, rowvar=False) + np.eye(N_QUANTILES) * 1e-6
    S_CC = np.cov(all_gamma_CC, rowvar=False) + np.eye(N_QUANTILES) * 1e-6

    return S_A, S_P, S_CC


# --- Visualization Function ---
def plot_3d_precipitation_surface(dataset, sample_index):
    """
    Plots a 3D surface of a precipitation patch from the dataset. 📈

    The elevation of the surface at each point corresponds to the
    precipitation intensity at that pixel.

    Args:
        dataset (Dataset): The dataset object containing precipitation data.
        sample_index (int): The index of the sample to retrieve and plot.
    """
    # --- 1. Retrieve and process data ---
    if not (0 <= sample_index < len(dataset)):
        print(
            f"Error: sample_index {sample_index} is out of bounds for the dataset (size: {len(dataset)})."
        )
        return

    # Get the original precipitation data tensor from the dataset
    precip_tensor, _ = dataset[sample_index]

    # Convert to a 2D numpy array for plotting: (C, H, W) -> (H, W)
    precip_data = precip_tensor.squeeze().cpu().numpy()

    # --- 2. Create grid for plotting ---
    height, width = precip_data.shape
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)

    # --- 3. Generate the 3D plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create the surface plot with a suitable colormap
    surf = ax.plot_surface(X, Y, precip_data, cmap="viridis", edgecolor="none")

    # --- 4. Customize the plot for clarity ---
    ax.set_title(f"3D Surface Plot of Precipitation - Sample {sample_index}")
    ax.set_xlabel("X Coordinate (pixels)")
    ax.set_ylabel("Y Coordinate (pixels)")
    ax.set_zlabel("Precipitation Intensity")

    # Add a color bar to map values to colors
    fig.colorbar(surf, shrink=0.6, aspect=10, label="Precipitation Intensity")

    # Adjust viewing angle for better perspective
    ax.view_init(elev=30, azim=-60)

    # --- 5. Show the plot ---
    plt.show()


# --- Loss Functions (Refactored) ---
class GeometricLossSeparate(nn.Module):
    def __init__(self, S_tensors):
        super(GeometricLossSeparate, self).__init__()
        self.S_A, self.S_P, self.S_CC = S_tensors

    def forward(self, gamma_pred_3d, gamma_target_batch_3d):
        # No on-the-fly calculation. The target is passed directly.
        pred_A, pred_P, pred_CC = (
            gamma_pred_3d[:, 0, :],
            gamma_pred_3d[:, 1, :],
            gamma_pred_3d[:, 2, :],
        )
        target_A, target_P, target_CC = (
            gamma_target_batch_3d[:, 0, :],
            gamma_target_batch_3d[:, 1, :],
            gamma_target_batch_3d[:, 2, :],
        )

        # Compute Mahalanobis distance for each component
        diff_A = pred_A - target_A
        solved_A = torch.linalg.solve(
            self.S_A.expand(diff_A.shape[0], -1, -1), diff_A.unsqueeze(-1)
        )
        loss_A_sq = torch.sum(diff_A.unsqueeze(-1) * solved_A, dim=(1, 2))

        diff_P = pred_P - target_P
        solved_P = torch.linalg.solve(
            self.S_P.expand(diff_P.shape[0], -1, -1), diff_P.unsqueeze(-1)
        )
        loss_P_sq = torch.sum(diff_P.unsqueeze(-1) * solved_P, dim=(1, 2))

        diff_CC = pred_CC - target_CC
        solved_CC = torch.linalg.solve(
            self.S_CC.expand(diff_CC.shape[0], -1, -1), diff_CC.unsqueeze(-1)
        )
        loss_CC_sq = torch.sum(diff_CC.unsqueeze(-1) * solved_CC, dim=(1, 2))

        # Sum the square roots and take the mean over the batch
        total_loss = torch.mean(
            torch.sqrt(loss_A_sq) + torch.sqrt(loss_P_sq) + torch.sqrt(loss_CC_sq)
        )
        return total_loss


# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Prepare Datasets ---
    train_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "train"),
        metadata_file=TRAIN_METADATA_FILE,
    )
    val_dataset_full = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "validation"),
        metadata_file=VAL_METADATA_FILE,
    )

    # Helper for subsampling
    def subsample_dataset(dataset, fraction=0.1, seed=42):
        subset_indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )[: int(fraction * len(dataset))]
        return Subset(dataset, subset_indices)

    train_dataset = subsample_dataset(train_dataset_full, config["SUBSAMPLE_FRACTION"])
    val_dataset = subsample_dataset(
        val_dataset_full, config["SUBSAMPLE_FRACTION"], seed=0
    )

    # --- 2. Estimate S matrices from a subset of the training data ---
    print("Estimating separate S matrices from training data subset...")
    s_estimation_subset = Subset(
        train_dataset_full,
        torch.randperm(len(train_dataset_full))[:S_ESTIMATION_SAMPLES].tolist(),
    )
    S_A, S_P, S_CC = estimate_S_from_precomputed(s_estimation_subset)

    S_A_torch = torch.from_numpy(S_A).float().to(device)
    S_P_torch = torch.from_numpy(S_P).float().to(device)
    S_CC_torch = torch.from_numpy(S_CC).float().to(device)
    print("Separate S estimation complete.")

    # --- 3. Prepare DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --- 4. Initialize Model, Optimizer, and Loss ---
    model = GammaPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Use GeometricLoss directly as the criterion
    criterion = GeometricLossSeparate(S_tensors=(S_A_torch, S_P_torch, S_CC_torch)).to(
        device
    )

    # --- 5. Training & Validation Loop ---
    print("Starting training...")
    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        # Loop now yields target_gamma directly.
        for input_data, target_gamma in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"
        ):
            input_data, target_gamma = input_data.to(device), target_gamma.to(device)

            optimizer.zero_grad()
            predicted_gamma_3d = model(input_data)

            # Loss calculated with pre-computed target tensor.
            loss = criterion(predicted_gamma_3d, target_gamma)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for input_data, target_gamma in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"
            ):
                input_data, target_gamma = input_data.to(device), target_gamma.to(
                    device
                )
                predicted_gamma_3d = model(input_data)
                loss = criterion(predicted_gamma_3d, target_gamma)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_gamma_predictor_model.pth")
            print("Model checkpoint saved.")

    print("Training complete.")

    # --- 6. Evaluation & Visualization ---
    print("\nLoading test dataset for visualization...")
    test_dataset = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
        metadata_file=TEST_METADATA_FILE,
    )

    # You can now use the plotting function on any sample from the test set
    print("Generating 3D plot for a test sample...")
    # Feel free to change the sample_index to view different images
    plot_3d_precipitation_surface(dataset=test_dataset, sample_index=42)
