import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

from loss import compute_gamma_matrix_for_image
from metrics import compute_fss, compute_sal


def _calculate_per_sample_gamma_r2(pred_gamma_list, target_gamma_list):
    """
    Helper to calculate R^2 for each sample's gamma vector (A, P, CC).
    Takes lists of (3, Q) numpy arrays.
    """
    n_samples = len(pred_gamma_list)
    r2_A, r2_P, r2_CC = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)

    for i in range(n_samples):
        preds = pred_gamma_list[i]
        targets = target_gamma_list[i]

        for j, arr_r2 in enumerate([r2_A, r2_P, r2_CC]):
            pred_sample = preds[j, :]
            target_sample = targets[j, :]

            mask = np.isfinite(pred_sample) & np.isfinite(target_sample)
            if np.sum(mask) < 2:
                arr_r2[i] = np.nan
                continue

            if np.var(target_sample[mask]) < 1e-9:
                arr_r2[i] = np.nan
            else:
                arr_r2[i] = r2_score(target_sample[mask], pred_sample[mask])

    return r2_A, r2_P, r2_CC


def _get_precipitation_groups(mean_precip_col):
    """Applies the robust precipitation grouping strategy."""
    groups = pd.Series("Zero", index=mean_precip_col.index, dtype=object)

    non_zero_mask = mean_precip_col > 0
    if non_zero_mask.any():
        non_zero_means = mean_precip_col[non_zero_mask]

        p33 = np.quantile(non_zero_means, 0.33)
        p67 = np.quantile(non_zero_means, 0.67)

        groups[non_zero_mask & (non_zero_means <= p33)] = "Low"
        groups[(non_zero_means > p33) & (non_zero_means <= p67)] = "Mid"
        groups[non_zero_means > p67] = "High"

    return groups


def create_metrics_dataframe(
    all_preds_phys,
    all_targets_phys,
    all_inputs_phys,
    all_total_losses,
    all_mse_losses,
    all_surrogate_losses,
    quantile_levels,
    pixel_size_km,
):
    """
    Combines all per-sample numpy arrays into a single, comprehensive
    pandas DataFrame, computing expensive metrics along the way.
    """
    print("\nCreating comprehensive metrics DataFrame...")
    print("This may take a moment (calculating FSS, SAL, and Gamma functions)...")

    n_samples = all_preds_phys.shape[0]

    # --- Compute expensive metrics in a loop ---
    fss_scores = []
    sal_S, sal_A, sal_L = [], [], []
    pred_gamma_list = []
    target_gamma_list = []

    for i in tqdm(range(n_samples), desc="Calculating per-sample metrics"):
        pred_img = all_preds_phys[i]
        target_img = all_targets_phys[i]

        # FSS
        fss_scores.append(
            compute_fss(pred_img, target_img, window_size=12, threshold=1.0)
        )

        # SAL
        s, a, l = compute_sal(
            pred_img, target_img, threshold=1.0, pixel_area_km2=pixel_size_km**2
        )
        sal_S.append(s)
        sal_A.append(a)
        sal_L.append(l)

        # Analytical Gammas
        pred_gamma_list.append(
            compute_gamma_matrix_for_image(pred_img, quantile_levels, pixel_size_km)
        )
        target_gamma_list.append(
            compute_gamma_matrix_for_image(target_img, quantile_levels, pixel_size_km)
        )

    # 1. Calculate per-sample Gamma R2 scores
    r2_A, r2_P, r2_CC = _calculate_per_sample_gamma_r2(
        pred_gamma_list, target_gamma_list
    )

    # 2. Calculate mean precipitation (from target)
    mean_precip = np.mean(all_targets_phys, axis=(1, 2))

    # 3. Build the DataFrame
    df = pd.DataFrame(
        {
            "total_loss": all_total_losses,
            "mse": all_mse_losses,
            "surrogate_loss": all_surrogate_losses,
            "mean_precip": mean_precip,
            "fss": fss_scores,
            "sal_S": sal_S,
            "sal_A": sal_A,
            "sal_L": sal_L,
            "r2_A": r2_A,
            "r2_P": r2_P,
            "r2_CC": r2_CC,
        }
    )

    # 4. Add precipitation group
    df["precip_group"] = _get_precipitation_groups(df["mean_precip"])

    # 5. Add original data as object columns (for plotting)
    df["pred_image"] = [img for img in all_preds_phys]
    df["target_image"] = [img for img in all_targets_phys]
    df["input_stack"] = [stack for stack in all_inputs_phys]
    df["pred_gamma"] = pred_gamma_list
    df["target_gamma"] = target_gamma_list

    print("DataFrame created.")
    print(df.info())

    return df


def calculate_grouped_metrics(metrics_df):
    """
    Calculates the mean of all numeric metrics, grouped by the
    'precip_group' column.
    """
    print("\nCalculating metrics by precipitation group...")

    # Define columns to aggregate
    metric_cols = [
        "total_loss",
        "mse",
        "surrogate_loss",
        "fss",
        "sal_S",
        "sal_A",
        "sal_L",
        "r2_A",
        "r2_P",
        "r2_CC",
    ]

    grouped = metrics_df.groupby("precip_group")

    # Calculate mean and count
    group_means = grouped[metric_cols].mean()
    group_counts = grouped.size().to_frame("n_samples")

    group_metrics = pd.concat([group_means, group_counts], axis=1)

    # Add 'All' group
    all_metrics = metrics_df[metric_cols].mean().to_frame("All").T
    all_metrics["n_samples"] = len(metrics_df)

    final_metrics = pd.concat([group_metrics, all_metrics])

    # Re-order for clarity
    group_order = ["Zero", "Low", "Mid", "High", "All"]
    final_metrics = final_metrics.reindex(group_order).dropna(how="all")

    print(final_metrics.to_string(float_format="%.6f"))

    return final_metrics


def calculate_per_feature_gamma_metrics(metrics_df, quantiles):
    """
    Calculates R^2, MSE, and Target Variance for each analytical gamma feature
    across all samples.
    """
    print("\nCalculating per-feature analytical gamma metrics (R^2, MSE, Var)...")

    # Stack the gamma matrices from the DataFrame
    try:
        all_preds_gamma = np.stack(metrics_df["pred_gamma"].values)
        all_targets_gamma = np.stack(metrics_df["target_gamma"].values)
    except ValueError as e:
        print(f"Error stacking gamma arrays: {e}. Check for empty/ragged lists.")
        return None  # Handle error gracefully

    n_samples, n_components, n_quantiles = all_preds_gamma.shape

    # Flatten from (N, 3, Q) to (N, 3*Q)
    preds_flat = all_preds_gamma.reshape(n_samples, -1)
    targets_flat = all_targets_gamma.reshape(n_samples, -1)

    # Filter samples with any NaNs/Infs
    mask = np.isfinite(targets_flat).all(axis=1) & np.isfinite(preds_flat).all(axis=1)

    n_valid = np.sum(mask)
    if n_valid < n_samples:
        print(
            f"Warning: Filtering {n_samples - n_valid} samples with NaNs/Infs in analytical gammas."
        )

    idx = pd.Index(["Area", "Perimeter", "CCs"], name="Component")
    cols = pd.Index(quantiles, name="Quantile (mm/hr)")

    if n_valid < 2:
        print("CRITICAL: Less than 2 valid samples. Returning NaN metrics.")
        r2_matrix = pd.DataFrame(np.nan, index=idx, columns=cols)
        mse_matrix = pd.DataFrame(np.nan, index=idx, columns=cols)
        var_matrix = pd.DataFrame(np.nan, index=idx, columns=cols)
        mean_by_component = pd.DataFrame(
            np.nan, index=idx, columns=["Avg_R2", "Avg_MSE", "Avg_Target_Var"]
        )

    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_raw = r2_score(
                targets_flat[mask], preds_flat[mask], multioutput="raw_values"
            )

        mse_raw = mean_squared_error(
            targets_flat[mask], preds_flat[mask], multioutput="raw_values"
        )
        var_raw = np.var(targets_flat[mask], axis=0)

        r2_matrix = pd.DataFrame(
            r2_raw.reshape(3, n_quantiles), index=idx, columns=cols
        )
        mse_matrix = pd.DataFrame(
            mse_raw.reshape(3, n_quantiles), index=idx, columns=cols
        )
        var_matrix = pd.DataFrame(
            var_raw.reshape(3, n_quantiles), index=idx, columns=cols
        )

        mean_by_component = pd.DataFrame(
            {
                "Avg_R2": r2_matrix.mean(axis=1),
                "Avg_MSE": mse_matrix.mean(axis=1),
                "Avg_Target_Var": var_matrix.mean(axis=1),
            }
        )

    print("Mean analytical gamma metrics by component (averaged over quantiles):")
    print(mean_by_component.to_string(float_format="%.4e"))

    return {
        "r2_matrix": r2_matrix,
        "mse_matrix": mse_matrix,
        "var_matrix": var_matrix,
        "mean_by_component": mean_by_component,
        "quantiles": quantiles,
    }
