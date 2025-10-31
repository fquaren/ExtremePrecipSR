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
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)


class GammaPredictorSeparateHeadsHard(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        n_quantiles=N_QUANTILES,
        activation_fn=F.gelu,
        quantile_levels=QUANTILE_LEVELS,
        pixel_area_km2=PIXEL_SIZE_KM**2,
    ):
        """
        This model uses a shared CNN trunk and three separate
        fully-connected heads for Area, Perimeter, and CC.

        The output size is now implicitly defined by the 3 heads.
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
        # --- Shared Trunk Feature Extraction ---
        x_conv = self._forward_conv(x)
        x_flat = x_conv.view(-1, self.fc_input_size)

        # --- Process Separate Heads ---
        raw_A_logits = self.head_A(x_flat)  # Shape [B, NQ]
        raw_P_logits = self.head_P(x_flat)  # Shape [B, NQ]
        raw_CC_pred = self.head_CC(x_flat)  # Shape [B, NQ]

        # --- Calculate A_total Directly from Input (Hard Constraint Logic) ---
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x, nan=-1.0) >= threshold
            A_total = (
                mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + 1e-6
            )  # Shape [B, 1]

        # --- Constrain Area (Monotonicity) ---
        probs_A = torch.softmax(raw_A_logits, dim=1)  # [B, NQ]
        # Broadcasting [B, NQ] * [B, 1] -> [B, NQ]
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )  # [B, NQ]

        # --- Constrain Perimeter (Plausibility) ---
        epsilon = 1e-6
        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        P_excess = F.relu(raw_P_logits)  # Or F.softplus(raw_P_logits)
        pred_P = P_min + P_excess  # [B, NQ]

        # --- Constrain CC (Non-negativity) ---
        pred_CC = F.relu(raw_CC_pred)  # [B, NQ]

        # --- Stack Components Back Together ---
        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)  # Shape [B, 3, NQ]

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


# # Model now includes hard zero constraint
# class GammaPredictorHardConstraints(nn.Module):
#     def __init__(
#         self,
#         input_shape=(1, PATCH_SIZE, PATCH_SIZE),
#         num_output_features_flat=N,
#         n_quantiles=N_QUANTILES,
#         activation_fn=F.gelu,
#         quantile_levels=QUANTILE_LEVELS,
#         pixel_area_km2=PIXEL_SIZE_KM**2,
#     ):
#         super(GammaPredictorHardConstraints, self).__init__()
#         self.n_quantiles = n_quantiles
#         self.activation = activation_fn
#         self.register_buffer(
#             "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
#         )
#         self.pixel_area_km2 = pixel_area_km2
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(
#             in_channels=16, out_channels=32, kernel_size=3, padding=1
#         )
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(
#             in_channels=32, out_channels=64, kernel_size=3, padding=1
#         )
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.fc_input_size = self._get_conv_output_size(input_shape)
#         self.fc1 = nn.Linear(self.fc_input_size, 256)
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(256, 128)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(128, num_output_features_flat)

#     def _get_conv_output_size(self, shape):
#         with torch.no_grad():
#             input = torch.rand(1, *shape)
#             output = self._forward_conv(input)
#             return int(np.prod(output.size()[1:]))

#     def _forward_conv(self, x):
#         x = self.pool(self.activation(self.bn1(self.conv1(x))))
#         x = self.pool(self.activation(self.bn2(self.conv2(x))))
#         x = self.pool(self.activation(self.bn3(self.conv3(x))))
#         return x

#     def forward(self, x):
#         # --- Feature Extraction ---
#         x_conv = self._forward_conv(x)
#         x_flat = x_conv.view(-1, self.fc_input_size)
#         x_fc = self.activation(self.fc1(x_flat))
#         x_fc = self.dropout1(x_fc)
#         x_fc = self.activation(self.fc2(x_fc))
#         x_fc = self.dropout2(x_fc)
#         raw_output = self.fc3(x_fc)  # Shape [B, 3 * NQ]

#         # --- Reconstruct A and P with hard constraints ---
#         raw_A_logits = raw_output[
#             :, 0 * self.n_quantiles : 1 * self.n_quantiles
#         ]  # [B, NQ]
#         raw_P_logits = raw_output[
#             :, 1 * self.n_quantiles : 2 * self.n_quantiles
#         ]  # [B, NQ]
#         raw_CC_pred = raw_output[
#             :, 2 * self.n_quantiles : 3 * self.n_quantiles
#         ]  # [B, NQ]

#         # --- Calculate A_total Directly from Input ---
#         with torch.no_grad():
#             threshold = self.quantile_levels_tensor[0]
#             mask = torch.nan_to_num(x, nan=-1.0) >= threshold
#             A_total = (
#                 mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + 1e-6
#             )  # Shape [B, 1]

#         # --- Constrain Area (Monotonicity) ---
#         probs_A = torch.softmax(raw_A_logits, dim=1)  # [B, NQ]
#         # Broadcasting [B, NQ] * [B, 1] -> [B, NQ]
#         scaled_probs_A = probs_A * A_total
#         pred_A = torch.flip(
#             torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
#         )  # [B, NQ]

#         # --- Constrain Perimeter (Plausibility) ---
#         epsilon = 1e-6
#         P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
#         P_excess = F.relu(raw_P_logits)  # Or F.softplus(raw_P_logits)
#         pred_P = P_min + P_excess  # [B, NQ]

#         # --- Constrain CC (Non-negativity) ---
#         pred_CC = F.relu(raw_CC_pred)  # [B, NQ]

#         # --- Stack Components Back Together ---
#         final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)  # Shape [B, 3, NQ]

#         return final_output
