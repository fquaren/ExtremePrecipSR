import yaml
import torch.nn as nn
import torch.nn.functional as F

# Load configuration
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Length of gamma
N = len(config["QUANTILE_LEVELS"]) * 3


# Define the Neural Network
class GammaPredictor(nn.Module):
    def __init__(self, num_output_features=N):
        super(GammaPredictor, self).__init__()
        # Convolutional Layers for feature extraction
        # We'll use the same architecture, but the pooling effects will differ with 128x128 input.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the input size for the first FC layer dynamically or by tracing
        # For a 128x128 image with the current architecture:
        # Initial: (128, 128)
        # After conv1 (3x3 kernel, 1 padding): output size remains (128, 128)
        # After pool (2x2 kernel, 2 stride): (64, 64)

        # After conv2 (3x3 kernel, 1 padding): output size remains (64, 64)
        # After pool (2x2 kernel, 2 stride): (32, 32)

        # After conv3 (3x3 kernel, 1 padding): output size remains (32, 32)
        # After pool (2x2 kernel, 2 stride): (16, 16)

        # So, the final feature map will have 128 channels and be 16x16 pixels.
        self.fc_input_size = 128 * 16 * 16  # 128 channels * 16x16 feature map size

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(
            256, num_output_features
        )  # Output is a 1D vector of length N

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128 -> 64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64 -> 32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32 -> 16

        # Flatten the feature maps
        x = x.view(-1, self.fc_input_size)  # -1 infers the batch size

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression output

        return x
