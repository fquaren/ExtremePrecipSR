import numpy as np
from skimage import measure, morphology
from scipy.ndimage import label


# --- Helper Function for Single Threshold Gamma Calculation (unchanged) ---
def compute_A_P_CC_single_threshold_numpy(prec_2d_np, threshold, pixel_size_km=1.0):
    prec_2d_np_clean = np.nan_to_num(prec_2d_np, nan=-1.0)
    mask = prec_2d_np_clean >= threshold
    area_km2 = mask.sum() * (pixel_size_km**2)
    contours = measure.find_contours(mask.astype(float), 0.5)
    perimeter_pixels = 0
    for contour in contours:
        perimeter_pixels += np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
    perimeter_km = perimeter_pixels * pixel_size_km
    structure = morphology.disk(1)  # 4-connectivity
    _, num_features = label(mask, structure=structure)
    return np.array([area_km2, perimeter_km, num_features], dtype=np.float32)


# --- Function to Compute Gamma Matrix for an Image (unchanged) ---
def compute_gamma_matrix_for_image(prec_2d_data, thresholds, pixel_size_km=1.0):
    N_thresholds = len(thresholds)
    gamma_matrix = np.zeros((3, N_thresholds), dtype=np.float32)
    for i, threshold_value in enumerate(thresholds):
        gamma_matrix[:, i] = compute_A_P_CC_single_threshold_numpy(
            prec_2d_data, threshold_value, pixel_size_km
        )
    return gamma_matrix


# --- Function to Estimate S_inv (unchanged, but note it needs the loaded quantiles) ---
def estimate_S_inv_from_dataset(
    dataset_of_target_precip_fields,
    global_quantiles_as_thresholds,
    pixel_size_km,
    regularization_epsilon=1e-6,
):
    all_gamma_vectors_flat = []
    for i, prec_field in enumerate(dataset_of_target_precip_fields):
        gamma_matrix = compute_gamma_matrix_for_image(
            prec_field, global_quantiles_as_thresholds, pixel_size_km
        )
        all_gamma_vectors_flat.append(gamma_matrix.flatten())

    if not all_gamma_vectors_flat:
        raise ValueError(
            "Dataset of target precipitation fields is empty. Cannot estimate S_inv."
        )

    all_gamma_vectors_np = np.array(all_gamma_vectors_flat)

    S = np.cov(all_gamma_vectors_np, rowvar=False)
    S += np.eye(S.shape[0]) * regularization_epsilon
    S_inv = np.linalg.inv(S)
    return S_inv


# --- Function to Compute Mahalanobis Distance (unchanged) ---
def mahalanobis_distance(vec1, vec2, S_inv):
    diff = vec1 - vec2
    return np.sqrt(diff.T @ S_inv @ diff)


# --- Main Geometric Accuracy Metric Function (unchanged) ---
def geometric_accuracy_mahalanobis(
    prec_2d_target,
    prec_2d_pred,
    global_quantiles_as_thresholds,
    S_inv,
    pixel_size_km=1.0,
):
    thresholds = global_quantiles_as_thresholds
    gamma_target_matrix = compute_gamma_matrix_for_image(
        prec_2d_target, thresholds, pixel_size_km
    )
    gamma_pred_matrix = compute_gamma_matrix_for_image(
        prec_2d_pred, thresholds, pixel_size_km
    )
    gamma_target_flat = gamma_target_matrix.flatten()
    gamma_pred_flat = gamma_pred_matrix.flatten()

    expected_dim = len(thresholds) * 3
    if gamma_target_flat.shape[0] != expected_dim or S_inv.shape[0] != expected_dim:
        raise ValueError(
            f"Dimension mismatch: Flattened gamma vector has shape {gamma_target_flat.shape[0]}, "
            f"but expected {expected_dim} based on thresholds. S_inv has shape {S_inv.shape[0]}. "
            f"Ensure global_quantiles_as_thresholds used for S_inv calculation matches this function call."
        )
    dist = mahalanobis_distance(gamma_target_flat, gamma_pred_flat, S_inv)
    return dist
