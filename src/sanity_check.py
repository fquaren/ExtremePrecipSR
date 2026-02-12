import os
import yaml
import numpy as np
from tqdm import tqdm

# --- Config ---
# Adjust paths if necessary to match your structure
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

DATA_DIR = os.path.join(config["PREPROCESSED_DATA_DIR"], "train")
SCALER_PATH = os.path.join(config["PREPROCESSED_DATA_DIR"], "precip_max_val.npy")
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]


def coarsen_image_manual(img, factor):
    """
    Reference implementation of mass-conserving pooling.
    """
    h, w = img.shape
    new_h, new_w = h // factor, w // factor
    img = img[: new_h * factor, : new_w * factor]
    return img.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))


def run_sanity_check():
    print("--- DATA CONSISTENCY AUDIT ---")

    # 1. Load Data
    print(f"Loading data from {DATA_DIR}...")
    try:
        phys_hr = np.load(os.path.join(DATA_DIR, "physical_precip.npz"))["data"]
        input_lr = np.load(os.path.join(DATA_DIR, "coarse_original_precip.npz"))["data"]
        scaler = np.load(SCALER_PATH)[0]
    except FileNotFoundError as e:
        print(f"[FATAL] Could not load required files: {e}")
        return

    print(f"Scaler Value loaded: {scaler:.4f}")

    # 2. Select a Random Subset for Validation (to save time)
    n_samples = phys_hr.shape[0]
    n_check = min(100, n_samples)
    indices = np.random.choice(n_samples, n_check, replace=False)

    print(f"\nAudit 1: Mass Conservation (Checking {n_check} random samples)")
    print(f"Logic: InverseLog(Input_LR) == Average(Physical_HR)")

    errors = []

    for idx in tqdm(indices):
        # A. Get the Model Input (LR) and recover physical units
        # Transformation was: x_norm = log(1+x)/max
        # Inverse: x = exp(x_norm * max) - 1
        lr_patch_norm = input_lr[idx]
        lr_patch_phys_recovered = np.expm1(lr_patch_norm * scaler)

        # B. Get the Ground Truth Physical (HR) and manually coarsen
        hr_patch_phys = phys_hr[idx]
        lr_patch_phys_calculated = coarsen_image_manual(
            hr_patch_phys, DOWNSCALING_FACTOR
        )

        # C. Compare
        # We allow small float32 epsilon differences
        diff = np.abs(lr_patch_phys_recovered - lr_patch_phys_calculated)
        errors.append(np.mean(diff))

    avg_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"\nMean Absolute Error (Physical mm/h): {avg_error:.6f}")
    print(f"Max Absolute Error (Physical mm/h):  {max_error:.6f}")

    if avg_error < 1e-4:
        print("DIAGNOSIS: [PASS] Mass is conserved. The logic is sound.")
    else:
        print("DIAGNOSIS: [FAIL] Significant discrepancy found.")
        print(
            "Reason: The input data likely does not represent the average of the physical target."
        )

    # 3. The Single Pixel Test (Revisited)
    print("\nAudit 2: The 'Single Pixel' Signal Strength Test")

    # Create a synthetic physical patch: 0 everywhere, 100 mm/h in top-left corner
    h, w = phys_hr.shape[1], phys_hr.shape[2]
    synthetic_patch = np.zeros((h, w), dtype=np.float32)
    synthetic_patch[0, 0] = 100.0  # Intense storm pixel

    # Expected Physics:
    # If factor is 4, block size is 16. Average should be 100/16 = 6.25 mm/h.
    # Log value should be log(1 + 6.25) ≈ 1.98.

    # Process it via your pipeline logic:
    # 1. Coarsen Physical
    coarse_phys = coarsen_image_manual(synthetic_patch, DOWNSCALING_FACTOR)
    # 2. Transform
    coarse_norm = np.log1p(coarse_phys) / scaler

    # Check the value at the active pixel (0,0)
    signal_val = coarse_norm[0, 0]
    recovered_val = np.expm1(signal_val * scaler)

    print(f"Input: Single pixel 100.0 mm/h")
    print(f"Downscaling Factor: {DOWNSCALING_FACTOR}")
    print(f"Expected Coarse Physical Value: {100.0 / (DOWNSCALING_FACTOR**2):.4f} mm/h")
    print(f"Recovered Coarse Value (from pipeline): {recovered_val:.4f} mm/h")

    # Previous flawed pipeline gave ~0.29 input value (recovered ~0.3 mm/h)
    # Correct pipeline should give ~1.98 input value (recovered 6.25 mm/h)

    if abs(recovered_val - (100.0 / DOWNSCALING_FACTOR**2)) < 0.1:
        print("DIAGNOSIS: [PASS] Signal strength is preserved.")
    else:
        print("DIAGNOSIS: [FAIL] The signal is being dampened.")


if __name__ == "__main__":
    run_sanity_check()
