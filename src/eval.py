import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <-- ADDED IMPORT
import matplotlib.gridspec as gridspec
from skimage import measure, morphology
from scipy.ndimage import label
import torch.nn.functional as F
from torch.utils.data import Dataset

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
DEM_PATCH_DIR = config["DEM_PATCH_DIR"]
TEST_METADATA_FILE = config["TEST_METADATA_FILE"]
BATCH_SIZE = config.get("BATCH_SIZE", 16)


# --- Model Definition ---
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
        self.fc_input_size = self._get_conv_output_size(input_shape)
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_output_features_flat)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input_tensor = torch.rand(1, *shape)
            output = self._forward_conv(input_tensor)
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


# --- Data Handling and Gamma Computation ---
def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = sum(
        np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in contours
    )
    perimeter_km = perimeter_pixels * pixel_size_km
    structure = morphology.disk(1)
    _, num_features = label(mask, structure=structure)
    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km=1.0):
    gamma_matrix = np.zeros((3, len(thresholds)), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


class PreprocessedNpzDataset(Dataset):
    def __init__(self, preprocessed_data_dir, dem_patch_dir, metadata_file):
        print(f"Loading data from {preprocessed_data_dir}...")
        with open(metadata_file, "r") as f:
            self.metadata = [line.strip().split(",") for line in f]
        precip_path = os.path.join(preprocessed_data_dir, "original_precip.npz")
        self.original_patches = np.load(precip_path)["data"]
        if len(self.metadata) != self.original_patches.shape[0]:
            raise ValueError("Metadata and patch count mismatch.")
        print(f"Loaded {len(self.metadata)} precipitation patches.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        original_precip = self.original_patches[idx]
        input_for_model = torch.from_numpy(original_precip).float().unsqueeze(0)
        return input_for_model, input_for_model.clone()


def subsample_dataset(dataset, fraction=1.0, seed=42):
    if fraction >= 1.0:
        return dataset
    dataset_size = len(dataset)
    subset_size = int(fraction * dataset_size)
    g = torch.Generator().manual_seed(seed)
    subset_indices = torch.randperm(dataset_size, generator=g)[:subset_size]
    return Subset(dataset, subset_indices)


# --- Plotting Functions ---


# --- ADDED FUNCTION ---
def plot_3d_precipitation_surface(target_images, sample_index):
    """
    Plots a 3D surface of a precipitation patch from the collected data. 📈

    The elevation of the surface at each point corresponds to the
    precipitation intensity at that pixel.

    Args:
        target_images (np.ndarray): A numpy array of all target images.
        sample_index (int): The index of the sample to retrieve and plot.
    """
    # --- 1. Retrieve and process data ---
    if not (0 <= sample_index < len(target_images)):
        print(
            f"Error: sample_index {sample_index} is out of bounds for the target images array (size: {len(target_images)})."
        )
        return

    # Get the 2D precipitation data directly from the numpy array
    precip_data = target_images[sample_index]

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
    ax.set_zlabel("Precipitation Intensity (mm/hr)")

    # Add a color bar to map values to colors
    fig.colorbar(surf, shrink=0.6, aspect=10, label="Precipitation Intensity (mm/hr)")

    # Adjust viewing angle for better perspective
    ax.view_init(elev=30, azim=-60)

    # --- 5. Save and close the plot ---
    output_dir = "evaluation_plots/3d_surfaces"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"3d_surface_sample_{sample_index}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D surface plot: {save_path}")


# --- END OF ADDED FUNCTION ---


def plot_quantile_contours(sample_index, target_images, quantiles_to_plot=(10, 50, 90)):
    """
    Plots the precipitation image for a given sample index and overlays contours
    at specified percentile levels.
    """
    if not 0 <= sample_index < len(target_images):
        print(f"Error: Sample index {sample_index} is out of bounds.")
        return

    image = target_images[sample_index]
    non_zero_pixels = image[image > 0]

    if non_zero_pixels.size == 0:
        print(
            f"Warning: Sample {sample_index} has no non-zero precipitation. Skipping contour plot."
        )
        return

    percentile_values = np.percentile(non_zero_pixels, quantiles_to_plot)
    colors = ["cyan", "lime", "magenta"]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image, cmap="Blues", origin="lower")

    legend_elements = []
    for val, q, color in zip(percentile_values, quantiles_to_plot, colors):
        ax.contour(image, levels=[val], colors=[color], linewidths=2)
        legend_elements.append(
            plt.Line2D(
                [0], [0], color=color, lw=2, label=f"{q}th Quantile ({val:.2f} mm/hr)"
            )
        )

    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_title(f"Precipitation Contours for Sample {sample_index}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Precipitation (mm/hr)")

    output_dir = "evaluation_plots/contours"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"quantile_contours_sample_{sample_index}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contour plot: {save_path}")


def plot_gamma_for_dataset_quantiles(
    predictions, targets_gamma, target_images, quantiles, n_examples=3
):
    """
    Identifies samples representing low, medium, and high mean precipitation across
    the dataset and generates gamma plots for them.
    """
    print("\nGenerating plots for samples representing dataset-wide quantiles...")
    dataset_quantiles = [10, 50, 90]
    gamma_types = ["Area (km²)", "Perimeter (km)", "Number of Connected Components"]

    all_means = np.mean(target_images, axis=(1, 2))

    for dq in dataset_quantiles:
        target_mean = np.percentile(all_means, dq)
        closest_indices = np.argsort(np.abs(all_means - target_mean))[:n_examples]

        print(
            f"\n--- Plotting for Dataset {dq}th Quantile (Mean Precip ≈ {target_mean:.2f}) ---"
        )

        for i, sample_idx in enumerate(closest_indices):
            pred_gamma = predictions[sample_idx]
            target_gamma = targets_gamma[sample_idx]
            target_image = target_images[sample_idx]

            fig = plt.figure(figsize=(20, 5))
            gs = gridspec.GridSpec(1, 4, wspace=0.35)

            ax_img = fig.add_subplot(gs[0, 0])
            im = ax_img.imshow(target_image, cmap="Blues", origin="lower", vmin=0)
            ax_img.set_title(f"Target Image (Mean: {all_means[sample_idx]:.2f})")
            fig.colorbar(im, ax=ax_img, shrink=0.7, label="Precipitation (mm/hr)")

            for j in range(3):
                ax = fig.add_subplot(gs[0, j + 1])
                ax.plot(
                    quantiles, target_gamma[j], "o-", label="Target", color="royalblue"
                )
                ax.plot(
                    quantiles, pred_gamma[j], "x--", label="Prediction", color="salmon"
                )
                ax.set_title(gamma_types[j])
                ax.set_xlabel("Precipitation Threshold (mm/hr)")
                ax.grid(True, linestyle="--", alpha=0.6)
                if j == 0:
                    ax.legend()

            fig.suptitle(
                f"Dataset {dq}th Quantile (Example {i+1} / Sample {sample_idx})",
                fontsize=16,
                y=1.03,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            output_dir = f"evaluation_plots/dataset_quantile_{dq}"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(
                output_dir, f"gamma_plot_example_{i+1}_sample_{sample_idx}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {save_path}")


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize model and load trained weights
    model = GammaPredictor(num_output_features_flat=N, n_quantiles=N_QUANTILES).to(
        device
    )
    model_save_path = "best_gamma_predictor_model.pth"
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Error: Model file '{model_save_path}' not found.")

    print("Loading model weights...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 3. Prepare test data
    test_dataset = PreprocessedNpzDataset(
        preprocessed_data_dir=os.path.join(PREPROCESSED_DATA_DIR, "test"),
        dem_patch_dir=DEM_PATCH_DIR,
        metadata_file=TEST_METADATA_FILE,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(test_dataset)} samples for evaluation.")

    # 4. Generate predictions and compute target gamma values
    all_preds, all_targets_gamma, all_target_images = [], [], []

    with torch.no_grad():
        for input_data, target_precip in tqdm(
            test_loader, desc="Generating predictions"
        ):
            # input_data = input_data.to(device)
            # predicted_gamma_3d = model(input_data)

            target_precip_np = target_precip.squeeze(1).cpu().numpy()
            # target_gamma_batch = [
            #     compute_gamma_matrix_for_image(img, QUANTILE_LEVELS)
            #     for img in target_precip_np
            # ]

            # all_preds.append(predicted_gamma_3d.cpu().numpy())
            # all_targets_gamma.append(np.stack(target_gamma_batch))
            all_target_images.append(target_precip_np)

    # all_preds = np.concatenate(all_preds, axis=0)
    # all_targets_gamma = np.concatenate(all_targets_gamma, axis=0)
    all_target_images = np.concatenate(all_target_images, axis=0)
    # print(f"Generated predictions for {all_preds.shape[0]} samples.")

    # 5. Execute plotting functions
    # plot_quantile_contours(sample_index=150, target_images=all_target_images)

    # plot_gamma_for_dataset_quantiles(
    #     predictions=all_preds,
    #     targets_gamma=all_targets_gamma,
    #     target_images=all_target_images,
    #     quantiles=QUANTILE_LEVELS,
    # )

    # Generate a 3D surface plot for a specific sample
    for i in range(100):
        plot_3d_precipitation_surface(
            target_images=all_target_images, sample_index=i * 100
        )
        plot_quantile_contours(target_images=all_target_images, sample_index=i * 100)

    print("\n✅ Evaluation script finished.")
