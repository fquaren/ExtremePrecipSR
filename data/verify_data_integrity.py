import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_path, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

PREPROCESSED_DATA_DIR = config["PREPROCESSED_DATA_DIR"]
PATCH_SIZE = config["PATCH_SIZE"]
DOWNSCALING_FACTOR = config["DOWNSCALING_FACTOR"]

# Tolerance for floating point comparisons
# log1p and float32 operations can introduce slight deviations
ATOL = 1e-5


def check_statistical_integrity(
    name, data, min_val=None, max_val=None, allow_nans=False
):
    """
    Checks basic statistics and bounds of a dataset.
    """
    print(f"  Checking {name}...")
    print(f"    Shape: {data.shape}, Dtype: {data.dtype}")

    # 1. NaN Check
    num_nans = np.isnan(data).sum()
    if num_nans > 0 and not allow_nans:
        print(f"    [FAIL] Found {num_nans} NaNs!")
        return False
    elif num_nans > 0:
        print(f"    [WARN] Found {num_nans} NaNs (Allowed).")

    # 2. Min/Max Stats
    d_min, d_max, d_mean = np.nanmin(data), np.nanmax(data), np.nanmean(data)
    print(f"    Stats -> Min: {d_min:.4f}, Max: {d_max:.4f}, Mean: {d_mean:.4f}")

    # 3. Bound Checks
    valid = True
    if min_val is not None and d_min < min_val - ATOL:
        print(f"    [FAIL] Min value {d_min} is below allowed threshold {min_val}")
        valid = False
    if max_val is not None and d_max > max_val + ATOL:
        print(f"    [FAIL] Max value {d_max} is above allowed threshold {max_val}")
        valid = False

    return valid


def verify_transformation_logic(physical_data, original_data, scaler_max):
    """
    Verifies: Original == Clip( Log1p(Physical) / Scaler )
    """
    print("  Verifying Transformation Logic (Physical vs Original)...")

    # Replicate the forward transform
    # 1. Log
    reconstructed = np.log1p(physical_data)
    # 2. Scale
    reconstructed = reconstructed / scaler_max
    # 3. Clip (simulating the preprocessing step)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    # Compare
    # We use a slightly loose tolerance because of float32 precision accumulation
    diff = np.abs(original_data - reconstructed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    is_close = np.allclose(original_data, reconstructed, atol=ATOL)

    print(f"    Max absolute difference: {max_diff:.6f}")
    print(f"    Mean absolute difference: {mean_diff:.6f}")

    if is_close:
        print("    [PASS] Mathematical transformation is consistent.")
        return True
    else:
        print("    [FAIL] Mathematical transformation mismatch.")
        return False


def visualize_comparison(phys_set, orig_set, split_name, idx=0):
    """
    Saves a plot comparing Physical and Original/Scaled views of the same patch.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Physical (mm/h)
    im0 = axes[0, 0].imshow(phys_set["high"][idx], cmap="jet", origin="lower")
    axes[0, 0].set_title(
        f"Physical High-Res (mm/h)\nMax: {phys_set['high'][idx].max():.2f}"
    )
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(phys_set["coarse"][idx], cmap="jet", origin="lower")
    axes[0, 1].set_title(f"Physical Coarse\nMax: {phys_set['coarse'][idx].max():.2f}")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(phys_set["interp"][idx], cmap="jet", origin="lower")
    axes[0, 2].set_title(f"Physical Interp\nMax: {phys_set['interp'][idx].max():.2f}")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Original (Log + Scaled)
    im3 = axes[1, 0].imshow(
        orig_set["high"][idx], cmap="viridis", origin="lower", vmin=0, vmax=1
    )
    axes[1, 0].set_title(
        f"Original (Log+Scale)\nMax: {orig_set['high'][idx].max():.2f}"
    )
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(orig_set["coarse"][idx], cmap="viridis", origin="lower")
    axes[1, 1].set_title(f"Original Coarse\nMax: {orig_set['coarse'][idx].max():.2f}")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(orig_set["interp"][idx], cmap="viridis", origin="lower")
    axes[1, 2].set_title(f"Original Interp\nMax: {orig_set['interp'][idx].max():.2f}")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.suptitle(f"Data Verification Sample: {split_name} (Index {idx})")
    plt.tight_layout()

    out_path = os.path.join(
        PREPROCESSED_DATA_DIR, f"verification_plot_{split_name}.png"
    )
    plt.savefig(out_path)
    print(f"  [INFO] Comparison plot saved to {out_path}")
    plt.close()


def main():
    # Load Global Scaler
    scaler_path = os.path.join(PREPROCESSED_DATA_DIR, "scaler_max_val.npy")
    if not os.path.exists(scaler_path):
        print(f"[CRITICAL] Scaler not found at {scaler_path}. Cannot verify logic.")
        return

    scaler_max = np.load(scaler_path)[0]
    print(f"Global Max Scaler loaded: {scaler_max:.4f}")

    for split in ["train", "validation", "test"]:
        print(f"\n{'='*20} Verifying {split.upper()} {'='*20}")
        data_dir = os.path.join(PREPROCESSED_DATA_DIR, split)

        files = {
            "phys_high": "physical_precip.npz",
            "phys_coarse": "coarse_physical_precip.npz",
            "phys_interp": "interpolated_physical_precip.npz",
            "orig_high": "original_precip.npz",
            "orig_coarse": "coarse_original_precip.npz",
            "orig_interp": "interpolated_original_precip.npz",
        }

        # Load all data into memory (assuming they fit for verification purposes)
        # If strictly too large, one would load just the first 100 samples.
        loaded = {}
        try:
            for key, fname in files.items():
                fpath = os.path.join(data_dir, fname)
                if not os.path.exists(fpath):
                    print(f"[SKIP] Missing {fname}")
                    continue
                # Load only first 500 samples to save RAM for verification
                with np.load(fpath) as f:
                    loaded[key] = f["data"][:500]
        except Exception as e:
            print(f"[ERROR] Failed loading data: {e}")
            continue

        if len(loaded) != 6:
            print("[ERROR] Not all files were found. Skipping logic checks.")
            continue

        # --- 1. Statistical Checks ---
        # Physical: must be >= 0. No upper bound strict limit, but usually < 500mm/h
        check_statistical_integrity("Physical High", loaded["phys_high"], min_val=0.0)

        # Original: must be [0, 1]
        check_statistical_integrity(
            "Original High", loaded["orig_high"], min_val=0.0, max_val=1.0
        )

        # --- 2. Shape Checks ---
        N, H, W = loaded["phys_high"].shape
        print(f"  Base Dimensions: ({N}, {H}, {W})")

        # Coarse Shape Check
        cH, cW = H // DOWNSCALING_FACTOR, W // DOWNSCALING_FACTOR
        if loaded["phys_coarse"].shape[1:] != (cH, cW):
            print(
                f"  [FAIL] Coarse shape mismatch. Expected {cH}x{cW}, got {loaded['phys_coarse'].shape[1:]}"
            )
        else:
            print(f"  [PASS] Coarse dimensions correct ({cH}x{cW}).")

        # --- 3. Logic Checks ---
        verify_transformation_logic(
            loaded["phys_high"], loaded["orig_high"], scaler_max
        )

        # --- 4. Conservation of Mean (Approximate) ---
        # The mean of the coarse image should be very close to the mean of the high-res image
        # Note: It won't be exact due to cropping if dimensions aren't perfect multiples, but here they should be.
        phys_mean = np.mean(loaded["phys_high"])
        phys_coarse_mean = np.mean(loaded["phys_coarse"])
        diff_percent = abs(phys_mean - phys_coarse_mean) / (phys_mean + 1e-6) * 100
        print(
            f"  Mean Conservation Check: High({phys_mean:.4f}) vs Coarse({phys_coarse_mean:.4f}) -> Diff: {diff_percent:.4f}%"
        )
        if diff_percent < 1.0:
            print("  [PASS] Mass/Mean roughly conserved.")
        else:
            print("  [WARN] Mass difference > 1%. Check coarsening logic.")

        # --- 5. Visual Sample ---
        # Group for plotting
        phys_set = {
            "high": loaded["phys_high"],
            "coarse": loaded["phys_coarse"],
            "interp": loaded["phys_interp"],
        }
        orig_set = {
            "high": loaded["orig_high"],
            "coarse": loaded["orig_coarse"],
            "interp": loaded["orig_interp"],
        }

        # Find a sample with actual rain (max > 0)
        idx_to_plot = 0
        for i in range(len(loaded["phys_high"])):
            if np.max(loaded["phys_high"][i]) > 1.0:  # at least 1mm/h
                idx_to_plot = i
                break

        visualize_comparison(phys_set, orig_set, split, idx=idx_to_plot)

    print("\nVerification Complete.")


if __name__ == "__main__":
    main()
