import yaml
import torch
import torch.nn as nn
import numpy as np
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorHierarchicalSoftGated,
    GammaPredictorHierarchicalHardGated,
)

# --- Configuration Loading ---
config_path = (
    "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/config.yaml"
)
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# --- Surrogate Loss Config ---
CONSTRAINT_MODE = config.get("CONSTRAINT_MODE", "hard")

# --- Emulator Model Config (Needed to load the checkpoint) ---
QUANTILE_LEVELS = config["QUANTILE_LEVELS"]
N_QUANTILES = len(QUANTILE_LEVELS)
PATCH_SIZE = config["PATCH_SIZE"]
PIXEL_SIZE_KM = config.get("PIXEL_SIZE_KM", 1.0)

ARCHITECTURE = config.get("ARCHITECTURE", "Vanilla")
if ARCHITECTURE == "Vanilla":
    HARD_EMULATOR = GammaPredictorSeparateHeadsHard
    SOFT_EMULATOR = GammaPredictorSeparateHeadsSoft
elif ARCHITECTURE == "Attention":
    HARD_EMULATOR = GammaPredictorHierarchicalHardGated
    SOFT_EMULATOR = GammaPredictorHierarchicalSoftGated


def load_emulator(checkpoint_path, config, device):
    """
    Loads a trained Gamma Emulator model for use in the SR loop.

    Args:
        checkpoint_path (str): Path to the .pth file.
        config (dict): Configuration dictionary containing model hyperparameters.
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        nn.Module: The loaded, frozen emulator in eval mode.
    """
    print(f"--- Loading Gamma Emulator from: {checkpoint_path} ---")

    # 1. Extract necessary config parameters
    # Defaulting to values seen in your previous scripts if missing
    arch = config.get("ARCHITECTURE", "Vanilla")
    constraint_mode = config.get("CONSTRAINT_MODE", "hybrid")
    patch_size = config["PATCH_SIZE"]
    n_quantiles = len(config["QUANTILE_LEVELS"])
    pixel_size_km = config.get("PIXEL_SIZE_KM", 1.0)
    # Max precip is required for Soft/Hybrid scaling, ensure it matches training!
    max_dataset_precip = float(np.load(config["MAX_PRECIP_FILE"]))

    input_shape = (1, patch_size, patch_size)

    # 2. Select Architecture Class
    # This must match the logic used in train_gamma.py exactly
    if arch == "Vanilla":
        HardClass = GammaPredictorSeparateHeadsHard
        SoftClass = GammaPredictorSeparateHeadsSoft
    elif arch == "Attention":
        HardClass = GammaPredictorHierarchicalHardGated
        SoftClass = GammaPredictorHierarchicalSoftGated
    else:
        raise ValueError(f"Unknown Architecture type in config: {arch}")

    # 3. Instantiate Model
    if constraint_mode in ["soft", "none"]:
        print(f"Initializing {SoftClass.__name__} (Mode: {constraint_mode})")
        model = SoftClass(
            input_shape=input_shape,
            n_quantiles=n_quantiles,
            activation_fn=nn.Mish(),
            max_precip_value=max_dataset_precip,
        ).to(device)

    elif constraint_mode in ["hybrid", "hard"]:
        print(f"Initializing {HardClass.__name__} (Mode: {constraint_mode})")
        model = HardClass(
            input_shape=input_shape,
            n_quantiles=n_quantiles,
            activation_fn=nn.Mish(),
            quantile_levels=config["QUANTILE_LEVELS"],
            pixel_area_km2=pixel_size_km**2,
            max_precip_value=max_dataset_precip,
        ).to(device)
    else:
        raise ValueError(f"Unknown CONSTRAINT_MODE: {constraint_mode}")

    # 4. Load Weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle cases where checkpoint is nested or just state_dict
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"CRITICAL ERROR loading state dict: {e}")
        print(
            "Tip: Check if 'ARCHITECTURE' in config matches the checkpoint's architecture."
        )
        raise e

    # 5. Freeze and Eval
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Emulator successfully loaded, set to eval mode, and weights frozen.")
    return model
