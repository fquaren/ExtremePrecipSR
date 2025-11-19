import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models  # Required for ResNet
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


class InputNormalization(nn.Module):
    """
    Scales physical input (mm/hr) to [0, 1] range for numerical stability
    inside the network.
    """

    def __init__(self, max_precip_value):
        super().__init__()
        # Register as a buffer so it's saved with the state_dict
        # but is not a learnable parameter.
        self.register_buffer(
            "scale_factor", torch.tensor(max_precip_value, dtype=torch.float32)
        )

    def forward(self, x):
        # Safety: Avoid division by zero if config is wrong
        return x / (self.scale_factor + 1e-8)


class GammaPredictorResNetHardHierarchical(nn.Module):
    def __init__(
        self,
        input_shape,
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],
        pixel_area_km2=1.0,
        max_precip_value=150.0,  # Physical Normalization Constant
    ):
        super(GammaPredictorResNetHardHierarchical, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        # --- ResNet-18 Trunk ---
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_input_size = 512
        resnet.fc = nn.Identity()
        self.resnet_trunk = resnet

        # --- Hierarchical Regression Heads ---
        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size + self.n_quantiles, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size + 2 * self.n_quantiles, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )

    def _forward_resnet_trunk(self, x):
        x = self.resnet_trunk.conv1(x)
        x = self.resnet_trunk.bn1(x)
        x = self.resnet_trunk.relu(x)
        x = self.resnet_trunk.maxpool(x)
        x = self.resnet_trunk.layer1(x)
        x = self.resnet_trunk.layer2(x)
        x = self.resnet_trunk.layer3(x)
        x = self.resnet_trunk.layer4(x)
        x = self.resnet_trunk.avgpool(x)
        return x

    def forward(self, x_phys):
        """
        x_phys: Input tensor in PHYSICAL units (mm/hr).
        """
        epsilon = 1e-6

        # --- A. Normalize for Neural Network Stability ---
        x_norm = self.normalizer(x_phys)  # [0, 1] range

        # --- B. Forward pass (Using Normalized Data) ---
        x_pooled = self._forward_resnet_trunk(x_norm)
        x_flat = torch.flatten(x_pooled, 1)

        raw_A_logits = self.head_A(x_flat)
        x_flat_A = torch.cat([x_flat, raw_A_logits.detach()], dim=1)
        raw_P_logits = self.head_P(x_flat_A)
        x_flat_A_P = torch.cat(
            [x_flat, raw_A_logits.detach(), raw_P_logits.detach()], dim=1
        )
        raw_CC_logits = self.head_CC(x_flat_A_P)

        # --- C. Constraints (Using PHYSICAL Data) ---
        # We use x_phys because 'threshold' and 'epsilon' are physical values
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x_phys, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + epsilon

        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )

        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        R_P = 1.0 + F.softplus(raw_P_logits)
        pred_P = P_min * R_P

        pred_CC_continuous = F.softplus(raw_CC_logits)
        pred_CC = (
            pred_CC_continuous if self.training else torch.round(pred_CC_continuous)
        )

        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        # Zero Constraint on Physical Input
        with torch.no_grad():
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        final_output = constrained_output * wet_factor
        return final_output


class GammaPredictorResNetSoftHierarchical(nn.Module):
    def __init__(
        self,
        input_shape=(1, 128, 128),
        n_quantiles=9,
        activation_fn=F.gelu,
        max_precip_value=150.0,
    ):
        super(GammaPredictorResNetSoftHierarchical, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        # --- ResNet Trunk ---
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc_input_size = 512
        resnet.fc = nn.Identity()
        self.resnet_trunk = resnet

        # --- Hierarchical Regression Heads ---
        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size + self.n_quantiles, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size + 2 * self.n_quantiles, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )

    def _forward_resnet_trunk(self, x):
        x = self.resnet_trunk.conv1(x)
        x = self.resnet_trunk.bn1(x)
        x = self.resnet_trunk.relu(x)
        x = self.resnet_trunk.maxpool(x)
        x = self.resnet_trunk.layer1(x)
        x = self.resnet_trunk.layer2(x)
        x = self.resnet_trunk.layer3(x)
        x = self.resnet_trunk.layer4(x)
        x = self.resnet_trunk.avgpool(x)
        return x

    def forward(self, x_phys):
        # Normalize internally
        x_norm = self.normalizer(x_phys)

        x_pooled = self._forward_resnet_trunk(x_norm)
        x_flat = torch.flatten(x_pooled, 1)

        raw_A = self.head_A(x_flat)
        x_flat_A = torch.cat([x_flat, raw_A.detach()], dim=1)
        raw_P = self.head_P(x_flat_A)
        x_flat_A_P = torch.cat([x_flat, raw_A.detach(), raw_P.detach()], dim=1)
        raw_CC = self.head_CC(x_flat_A_P)

        pred_A = F.softplus(raw_A)
        pred_P = F.softplus(raw_P)
        pred_CC = F.softplus(raw_CC)

        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)
        return final_output


class GammaPredictorSeparateHeadsHard(nn.Module):
    def __init__(
        self,
        input_shape,
        n_quantiles,
        activation_fn=F.gelu,
        quantile_levels=[0.0],
        pixel_area_km2=1.0,
        max_precip_value=150.0,
    ):
        super(GammaPredictorSeparateHeadsHard, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn
        self.register_buffer(
            "quantile_levels_tensor", torch.tensor(quantile_levels, dtype=torch.float32)
        )
        self.pixel_area_km2 = pixel_area_km2

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        # --- Shared CNN Trunk ---
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)

        self.fc_input_size = self._get_conv_output_size(input_shape)

        # --- Separate Regression Heads ---
        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(128, self.n_quantiles),
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            # Normalizer doesn't change shape, but we call it for correctness
            input = self.normalizer(input)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x_phys):
        epsilon = 1e-6

        # Normalize for the Trunk
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = x_conv.view(-1, self.fc_input_size)

        raw_A_logits = self.head_A(x_flat)
        raw_P_logits = self.head_P(x_flat)
        raw_CC_logits = self.head_CC(x_flat)

        # Physical constraints use x_phys
        with torch.no_grad():
            threshold = self.quantile_levels_tensor[0]
            mask = torch.nan_to_num(x_phys, nan=-1.0) >= threshold
            A_total = mask.sum(dim=(2, 3)).float() * self.pixel_area_km2 + epsilon

        probs_A = torch.softmax(raw_A_logits, dim=1)
        scaled_probs_A = probs_A * A_total
        pred_A = torch.flip(
            torch.cumsum(torch.flip(scaled_probs_A, dims=[1]), dim=1), dims=[1]
        )

        P_min = torch.sqrt(4 * math.pi * (pred_A + epsilon))
        R_P = 1.0 + F.softplus(raw_P_logits)
        pred_P = P_min * R_P

        pred_CC = F.softplus(raw_CC_logits)

        constrained_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)

        with torch.no_grad():
            is_dry_mask = x_phys.sum(dim=(1, 2, 3)) <= epsilon
            wet_factor = (~is_dry_mask).float().view(-1, 1, 1)

        final_output = constrained_output * wet_factor
        return final_output


class GammaPredictorSeparateHeadsSoft(nn.Module):
    def __init__(
        self,
        input_shape=(1, PATCH_SIZE, PATCH_SIZE),
        n_quantiles=N_QUANTILES,
        activation_fn=F.gelu,
        max_precip_value=150.0,  # <--- NEW
    ):
        super(GammaPredictorSeparateHeadsSoft, self).__init__()
        self.n_quantiles = n_quantiles
        self.activation = activation_fn

        # --- 1. Internal Normalizer ---
        self.normalizer = InputNormalization(max_precip_value)

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)

        self.fc_input_size = self._get_conv_output_size(input_shape)

        self.head_A = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_P = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )
        self.head_CC = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, self.n_quantiles),
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            input = self.normalizer(input)
            output = self._forward_conv(input)
            return int(np.prod(output.size()[1:]))

    def _forward_conv(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        return x

    def forward(self, x_phys):
        x_norm = self.normalizer(x_phys)
        x_conv = self._forward_conv(x_norm)
        x_flat = x_conv.view(-1, self.fc_input_size)

        raw_A = self.head_A(x_flat)
        raw_P = self.head_P(x_flat)
        raw_CC = self.head_CC(x_flat)

        pred_A = F.softplus(raw_A)
        pred_P = F.softplus(raw_P)
        pred_CC = F.softplus(raw_CC)

        final_output = torch.stack([pred_A, pred_P, pred_CC], dim=1)
        return final_output
