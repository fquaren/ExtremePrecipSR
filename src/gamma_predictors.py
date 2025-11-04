import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Load configuration
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Length of gamma
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N = len(QUANTILE_LEVELS) * 3
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PIXEL_SIZE_KM = config["PIXEL_SIZE_KM"]


class GammaPredictorSeparateHeadsHard(nn.Module):
    def __init__(
        self,
        input_shape,  # e.g., (1, PATCH_SIZE, PATCH_SIZE)
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],
        pixel_area_km2=1.0,
    ):
        """
        This model uses a shared CNN trunk and three separate
        fully-connected heads for Area, Perimeter, and CC.
        """
        super(GammaPredictorSeparateHeadsHard, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2

        # --- Shared CNN Trunk ---
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
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Calculate the flattened size after the CNN trunk
        self.fc_input_size = self._get_conv_output_size(input_shape)

        # --- Separate Regression Heads ---
        # Head for Area (A)
        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),  # Final output for Area
        )

        # Head for Perimeter (P)
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),  # Final output for Perimeter
        )

        # Head for Connected Components (CC)
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),  # Final output for CC
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        # This is the shared trunk
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        epsilon = 1e-6
        x_conv = self._forward_conv(x)
        x_flat = x_conv.view(-1, self.fc_input_size)

        raw_A_logits = self.head_A(x_flat)
        raw_P_logits = self.head_P(x_flat)
        raw_CC_logits = self.head_CC(x_flat)  # Renamed for clarity

        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + epsilon

        # --- 1. Constrain Area (Monotonicity) ---
        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )

        # --- 2. Constrain Perimeter (Plausibility via Ratio) ---
        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        R_P = 1.0 + F.softplus(raw_P_logits)
        pred_P = P_min * R_P

        # --- 3. Revert CC to simple non-negativity ---
        # The model's head is now free to learn the absolute (log-space) value.
        # Softplus ensures the output is always positive.
        pred_CC = F.softplus(raw_CC_logits)

        # --- Stack ---
        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        # --- 4. Apply Hard Zero Constraint ---
        with torch.no_grad():
            is_dry_mask = x.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        final_output = constrained_output * wet_factor

        return final_output


class GammaPredictorSeparateHeadsSoft(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        n_quantiles=N_QUANTILES,
        activation_fn=F.gelu,
    ):
        """
        MODIFIED: This model uses a shared CNN trunk and three separate
        heads to predict A, P, and CC directly.

        It does NOT enforce hard constraints (monotonicity, A_total, P_min).
        It only enforces non-negativity via a softplus activation.
        """
        super(GammaPredictorSeparateHeadsSoft, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn

        # --- Shared CNN Trunk ---
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
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Calculate the flattened size after the CNN trunk
        self.fc_input_size = self._get_conv_output_size(input_shape)

        # --- Separate Regression Heads ---
        # Head for Area (A)
        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),  # Final raw output for Area
        )

        # Head for Perimeter (P)
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),  # Final raw output for Perimeter
        )

        # Head for Connected Components (CC)
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),  # Final raw output for CC
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        # This is the shared trunk
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x):
        # --- Shared Trunk Feature Extraction ---
        x_conv = self._forward_conv(x)
        x_flat = x_conv.view(-1, self.fc_input_size)

        # --- Process Separate Heads ---
        raw_A = self.head_A(x_flat)  # Shape [B, NQ]
        raw_P = self.head_P(x_flat)  # Shape [B, NQ]
        raw_CC = self.head_CC(x_flat)  # Shape [B, NQ]

        # --- Apply Minimal "Soft" Constraint (Non-negativity) ---
        # We use softplus as it's a smooth version of ReLU
        # and doesn't "kill" gradients at zero.
        pred_A = F.softplus(raw_A)
        pred_P = F.softplus(raw_P)
        pred_CC = F.softplus(raw_CC)

        # --- Stack Components Back Together ---
        # The model's output is now the unconstrained physical values
        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)  # Shape [B, 3, NQ]

        return final_output
