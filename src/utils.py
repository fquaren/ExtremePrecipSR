import yaml
import torch
import torch.nn as nn
from gamma_predictors import (
    GammaPredictorSeparateHeadsSoft,
    GammaPredictorSeparateHeadsHard,
    GammaPredictorResNetSoftHierarchical,
    GammaPredictorResNetHardHierarchical,
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

ARCHITECTURE = config.get("ARCHITECTURE", "CNN")
if ARCHITECTURE == "CNN":
    HARD_EMULATOR = GammaPredictorSeparateHeadsHard
    SOFT_EMULATOR = GammaPredictorSeparateHeadsSoft
elif ARCHITECTURE == "RESNET":
    HARD_EMULATOR = GammaPredictorResNetHardHierarchical
    SOFT_EMULATOR = GammaPredictorResNetSoftHierarchical


# --- Helper to load the emulator ---
def load_emulator(checkpoint_path, device):
    print(f"Loading Gamma Emulator from: {checkpoint_path}")

    INPUT_SHAPE = (1, PATCH_SIZE, PATCH_SIZE)
    if CONSTRAINT_MODE == "soft" or CONSTRAINT_MODE is None:
        print("Using SOFT constraints model (GammaPredictorSeparateHeadsSoft).")
        model = GammaPredictorSeparateHeadsSoft(
            input_shape=INPUT_SHAPE, n_quantiles=N_QUANTILES, activation_fn=nn.Mish()
        ).to(device)
    elif CONSTRAINT_MODE == "hybrid" or CONSTRAINT_MODE == "hard":
        print("Using HYBRID constraints model (GammaPredictorSeparateHeadsHard).")
        model = GammaPredictorSeparateHeadsHard(
            input_shape=INPUT_SHAPE,
            n_quantiles=N_QUANTILES,
            activation_fn=nn.Mish(),
            quantile_levels=QUANTILE_LEVELS,
            pixel_area_km2=PIXEL_SIZE_KM**2,
        ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print("Emulator loaded, set to eval mode, and weights are frozen.")
    return model
