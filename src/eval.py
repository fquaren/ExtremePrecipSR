import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import measure, morphology
from scipy.ndimage import label
import torch.nn.functional as F
from torch.utils.data import Dataset  # Ensure Dataset is imported

# Load configuration (needed for N, N_QUANTILES)
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
N = N_QUANTILES * 3  # Ensure N is defined globally for GammaPredictor
PATCH_SIZE = config[
    "PATCH_SIZE"
]  # Also needed for visualization, if you want original size


class GammaPredictor(nn.Module):
    def __init__(self, num_output_features_flat=N, n_quantiles=N_QUANTILES):
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

        # Calculate the input size for the first FC layer. Assuming a 64x64 input patch
        # is downscaled by 2^3 = 8 (due to 3 MaxPool2d layers with kernel_size=2, stride=2)
        # So, 64 / 8 = 8. Input feature map size will be 8x8.
        # But based on your original script, it looks like PATCH_SIZE / 8, so if PATCH_SIZE=128
        # then 128/8 = 16, resulting in 16x16. Let's make it robust to PATCH_SIZE.

        # We need to compute this dynamically or ensure PATCH_SIZE is consistent.
        # Let's assume input_size_after_convs is PATCH_SIZE / 8 for now.
        # If your patches are 64x64, then it should be 64 / 8 = 8, so 64 * 8 * 8.
        # Your original script assumes 16x16, which implies original input was 128x128.
        # Let's use the explicit PATCH_SIZE from config for calculation if available.
        # Otherwise, revert to 64 * 16 * 16 as in your original script if PATCH_SIZE is not provided or different.

        # Dynamic calculation based on PATCH_SIZE (assuming square patches)
        input_res = config.get("PATCH_SIZE", 128)  # Default to 128 if not in config
        pooled_res = input_res // (2**3)  # Three pooling layers
        self.fc_input_size = 64 * pooled_res * pooled_res

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_output_features_flat)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = x.view(-1, 3, self.n_quantiles)
        return x


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


class PreprocessedNpzDataset(Dataset):
    def __init__(self, preprocessed_data_dir, dem_patch_dir, metadata_file):
        print(f"Loading data from {preprocessed_data_dir}...")
        self.metadata = []
        with open(metadata_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                self.metadata.append((parts[0], int(parts[1]), int(parts[2])))
        self.original_patches = np.load(
            os.path.join(preprocessed_data_dir, "original_precip.npz")
        )["data"]
        if len(self.metadata) != self.original_patches.shape[0]:
            raise ValueError(
                f"Number of metadata entries ({len(self.metadata)}) does not match "
                f"number of precipitation patches ({self.original_patches.shape[0]})"
            )
        self.dem_patches = None  # Not used for this model
        print(f"Loaded {len(self.metadata)} precipitation patches from metadata.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        original_precip = self.original_patches[idx]
        input_for_model = torch.from_numpy(original_precip).float().unsqueeze(0)
        output_precip = input_for_model.clone()  # Target is the same as input
        return input_for_model, output_precip


def subsample_dataset(dataset, fraction=0.1, seed=42):
    dataset_size = len(dataset)
    subset_size = int(fraction * dataset_size)
    g = torch.Generator().manual_seed(seed)
    subset_indices = torch.randperm(dataset_size, generator=g)[:subset_size]
    return Subset(dataset, subset_indices)


# --- Main Evaluation Logic ---

# 1. Load configuration and setup
# (Already loaded QUANTILE_LEVELS, N_QUANTILES, N)
PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
DEM_PATCH_DIR = config["DEM_PATCH_DIR"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Initialize model and load trained weights
model = GammaPredictor(num_output_features_flat=N, n_quantiles=N_QUANTILES).to(device)
model_save_path = "best_gamma_predictor_model.pth"
if not os.path.exists(model_save_path):
    print(f"Error: Model file '{model_save_path}' not found.")
    exit()

print("Loading model weights...")
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()
print("Model loaded successfully.")

# 3. Prepare test data
test_dataset_full = PreprocessedNpzDataset(
    preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
    dem_patch_dir=DEM_PATCH_DIR,
    metadata_file=TEST_METADATA_FILE,
)
test_dataset = subsample_dataset(test_dataset_full, 0.1, seed=456)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=config.get("NUM_WORKERS", os.cpu_count() // 2),
)
print(f"Loaded {len(test_dataset)} samples for evaluation.")

# 4. Generate predictions and targets for plotting
all_preds = []
all_targets_gamma = []
all_target_images = []  # Store target images for plotting

with torch.no_grad():
    for input_data, target_precip in tqdm(test_loader, desc="Generating predictions"):
        input_data = input_data.to(device)
        predicted_gamma_3d = model(input_data)

        predictions_np = predicted_gamma_3d.cpu().numpy()

        target_precip_np = target_precip.squeeze(1).cpu().numpy()

        target_gamma_batch = []
        for i in range(target_precip_np.shape[0]):
            gamma_matrix = compute_gamma_matrix_for_image(
                target_precip_np[i], QUANTILE_LEVELS
            )
            target_gamma_batch.append(gamma_matrix)
        targets_gamma_np = np.stack(target_gamma_batch)

        all_preds.append(predictions_np)
        all_targets_gamma.append(targets_gamma_np)
        all_target_images.append(
            target_precip_np
        )  # Append the actual precipitation images

# Concatenate all batches
all_preds = np.concatenate(all_preds, axis=0)
all_targets_gamma = np.concatenate(all_targets_gamma, axis=0)
all_target_images = np.concatenate(all_target_images, axis=0)
print(f"Generated predictions for {all_preds.shape[0]} samples.")


# 5. Plotting function (Modified)
def plot_gamma_predictions_with_image(
    predictions, targets_gamma, target_images, quantiles, n_samples_to_plot=10
):
    """
    Plots the target precipitation image, and predicted vs. target gamma matrix elements for a few samples.
    """
    gamma_types = ["Area (km²)", "Perimeter (km)", "Number of Connected Components"]

    np.random.seed(42)
    plot_indices = np.random.choice(
        len(targets_gamma), size=n_samples_to_plot, replace=False
    )

    for i, sample_idx in enumerate(plot_indices):
        pred_gamma = predictions[sample_idx]
        target_gamma = targets_gamma[sample_idx]
        target_image = target_images[sample_idx]  # Get the target image

        # Create a new figure for each sample
        fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 4, wspace=0.3, hspace=0.3)

        # Plot the target precipitation image in the first column
        ax_img = fig.add_subplot(gs[0, 0])
        im = ax_img.imshow(
            target_image,
            cmap="Blues",
            origin="lower",
            vmin=0,
            vmax=np.percentile(target_image, 99),
        )  # Adjust vmax as needed
        ax_img.set_title("Target Image", fontsize=12)
        ax_img.set_xlabel("X-coordinate")
        ax_img.set_ylabel("Y-coordinate")
        cbar = fig.colorbar(im, ax=ax_img, shrink=0.7)  # Get colorbar object
        cbar.set_label("Precipitation (mm/hr)")  # Set the label for the colorbar

        # Plot gamma elements in the next three columns
        for j in range(3):
            ax = fig.add_subplot(gs[0, j + 1])  # Offset by 1 for the image column

            ax.plot(quantiles, target_gamma[j], "o-", label="Target", color="royalblue")
            ax.plot(quantiles, pred_gamma[j], "x--", label="Prediction", color="salmon")

            ax.set_title(f"{gamma_types[j]}", fontsize=12)
            ax.set_xlabel("Precipitation Threshold (mm/hr)")
            ax.set_ylabel(f"{gamma_types[j]}")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

        fig.suptitle(
            f"Sample {sample_idx+1} Predictions with Target Image",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()

        # Save each row as a single image with a unique filename
        plt.savefig(
            f"gamma_predictions/gamma_predictions_sample_{sample_idx+1}.png",
            dpi=1000,
            bbox_inches="tight",
        )
        plt.close(fig)  # Close the figure to free up memory


# # 6. Execute plotting with the new function
# plot_gamma_predictions_with_image(
#     all_preds, all_targets_gamma, all_target_images, QUANTILE_LEVELS
# )


# 7. Create a new function to plot the gamma scatter points
def plot_gamma_scatter_comparison(
    predictions, targets_gamma, n_samples_to_plot=500, seed=42
):
    """
    Creates a single figure with two rows of three scatter plots comparing
    different gamma matrix elements for predictions vs. targets. The top row
    shows the first threshold, and the bottom row shows the last.

    Args:
        predictions (np.ndarray): Predicted gamma values. Shape: (N, 3, N_QUANTILES).
        targets_gamma (np.ndarray): Target gamma values. Shape: (N, 3, N_QUANTILES).
        n_samples_to_plot (int): Number of random samples to plot.
        seed (int): Random seed for reproducibility.
    """
    gamma_types = ["Area (km²)", "Perimeter (km)", "Number of Connected Components"]

    np.random.seed(seed)
    # Select a random subset of samples
    plot_indices = np.random.choice(
        len(targets_gamma), size=n_samples_to_plot, replace=False
    )

    # Filter the data to the selected samples
    preds_subset = predictions[plot_indices]
    targets_subset = targets_gamma[plot_indices]

    # Select the first and last threshold points for each gamma component
    targets_first_thresh = targets_subset[:, :, 0]
    preds_first_thresh = preds_subset[:, :, 0]

    targets_last_thresh = targets_subset[:, :, -1]
    preds_last_thresh = preds_subset[:, :, -1]

    # Create the figure with 2 rows and 3 columns of subplots
    fig, axes = plt.subplots(2, 3, figsize=(21, 20))

    # Define the pairs to plot (Area-Perimeter, Perimeter-Components, Area-Components)
    pairs = [(0, 1), (1, 2), (0, 2)]

    # Plot the first threshold on the top row
    for j, (x_idx, y_idx) in enumerate(pairs):
        ax = axes[0, j]
        ax.scatter(
            targets_first_thresh[:, x_idx],
            targets_first_thresh[:, y_idx],
            alpha=0.6,
            label="Target",
            s=25,
            color="royalblue",
        )
        ax.scatter(
            preds_first_thresh[:, x_idx],
            preds_first_thresh[:, y_idx],
            alpha=0.6,
            label="Prediction",
            s=25,
            color="salmon",
        )
        ax.set_title(
            f"First Threshold: {gamma_types[x_idx]} vs. {gamma_types[y_idx]}",
            fontsize=14,
        )
        ax.set_xlabel(gamma_types[x_idx])
        ax.set_ylabel(gamma_types[y_idx])
        ax.grid(True, linestyle="--", alpha=0.6)
        if j == 0:
            ax.legend()

    # Plot the last threshold on the bottom row
    for j, (x_idx, y_idx) in enumerate(pairs):
        ax = axes[1, j]
        ax.scatter(
            targets_last_thresh[:, x_idx],
            targets_last_thresh[:, y_idx],
            alpha=0.6,
            label="Target",
            s=25,
            color="royalblue",
        )
        ax.scatter(
            preds_last_thresh[:, x_idx],
            preds_last_thresh[:, y_idx],
            alpha=0.6,
            label="Prediction",
            s=25,
            color="salmon",
        )
        ax.set_title(
            f"Last Threshold: {gamma_types[x_idx]} vs. {gamma_types[y_idx]}",
            fontsize=14,
        )
        ax.set_xlabel(gamma_types[x_idx])
        ax.set_ylabel(gamma_types[y_idx])
        ax.grid(True, linestyle="--", alpha=0.6)
        if j == 0:
            ax.legend()

    fig.suptitle(
        f"Gamma Matrix Element Comparison for {n_samples_to_plot} Test Samples",
        fontsize=18,
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig("gamma_scatter_comparison.png", dpi=1000, bbox_inches="tight")
    plt.show()


# 8. Create a new function to plot the gamma scatter points
def plot_gamma_target_scatter_comparison(targets_gamma, n_samples_to_plot=500, seed=42):
    """
    Creates a single figure with two rows of three scatter plots showing only
    the target gamma matrix elements. The top row shows the first threshold,
    and the bottom row shows the last.

    Args:
        targets_gamma (np.ndarray): Target gamma values. Shape: (N, 3, N_QUANTILES).
        n_samples_to_plot (int): Number of random samples to plot.
        seed (int): Random seed for reproducibility.
    """
    gamma_types = ["Area (km²)", "Perimeter (km)", "Number of Connected Components"]

    np.random.seed(seed)
    # Select a random subset of samples
    plot_indices = np.random.choice(
        len(targets_gamma), size=n_samples_to_plot, replace=False
    )

    # Filter the data to the selected samples
    targets_subset = targets_gamma[plot_indices]

    # Select the first and last threshold points for each gamma component
    targets_first_thresh = targets_subset[:, :, 0]
    targets_last_thresh = targets_subset[:, :, -1]

    # Create the figure with 2 rows and 3 columns of subplots
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    # Define the pairs to plot
    pairs = [(0, 1), (1, 2), (0, 2)]

    # Calculate global min and max for each axis across both thresholds
    # to ensure the same axis limits for each column
    max_vals = [
        np.max(np.concatenate((targets_first_thresh[:, i], targets_last_thresh[:, i])))
        for i in range(3)
    ]
    min_vals = [
        np.min(np.concatenate((targets_first_thresh[:, i], targets_last_thresh[:, i])))
        for i in range(3)
    ]

    # Plot the first threshold on the top row
    for j, (x_idx, y_idx) in enumerate(pairs):
        ax = axes[0, j]
        ax.scatter(
            targets_first_thresh[:, x_idx],
            targets_first_thresh[:, y_idx],
            alpha=0.6,
            s=25,
            color="royalblue",
        )
        ax.set_title(
            f"First Threshold: {gamma_types[x_idx]} vs. {gamma_types[y_idx]}",
            fontsize=14,
        )
        ax.set_xlabel(gamma_types[x_idx])
        ax.set_ylabel(gamma_types[y_idx])
        ax.grid(True, linestyle="--", alpha=0.6)

        # Set shared axis limits for the column
        ax.set_xlim(min_vals[x_idx], max_vals[x_idx] * 1.1)
        ax.set_ylim(min_vals[y_idx], max_vals[y_idx] * 1.1)

    # Plot the last threshold on the bottom row
    for j, (x_idx, y_idx) in enumerate(pairs):
        ax = axes[1, j]
        ax.scatter(
            targets_last_thresh[:, x_idx],
            targets_last_thresh[:, y_idx],
            alpha=0.6,
            s=25,
            color="royalblue",
        )
        ax.set_title(
            f"Last Threshold: {gamma_types[x_idx]} vs. {gamma_types[y_idx]}",
            fontsize=14,
        )
        ax.set_xlabel(gamma_types[x_idx])
        ax.set_ylabel(gamma_types[y_idx])
        ax.grid(True, linestyle="--", alpha=0.6)

        # Set shared axis limits for the column
        ax.set_xlim(min_vals[x_idx], max_vals[x_idx] * 1.1)
        ax.set_ylim(min_vals[y_idx], max_vals[y_idx] * 1.1)

    fig.suptitle(
        f"Target Gamma Matrix Element Distributions for {n_samples_to_plot} Test Samples",
        fontsize=18,
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig("gamma_target_scatter_comparison.png", dpi=1000, bbox_inches="tight")


# 7. Execute the new plotting function with the gathered data
print("Creating scatter plots for gamma element comparisons...")
plot_gamma_scatter_comparison(all_preds, all_targets_gamma, n_samples_to_plot=500)
print("Plot saved as gamma_scatter_comparison.png")


# 8. Execute the new function to plot only targets
print("Creating scatter plots for target gamma element distributions...")
plot_gamma_target_scatter_comparison(all_targets_gamma, n_samples_to_plot=500)
print("Plot saved as gamma_target_scatter_comparison.png")
