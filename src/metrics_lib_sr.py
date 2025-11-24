import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def _calculate_per_sample_metrics(preds_gamma, targets_gamma):
    """
    Calculates R2, MSE, and Target Variance for each sample's vector (A, P, CC).
    preds_gamma: [N, 3, Q]
    targets_gamma: [N, 3, Q]
    """
    n_samples = preds_gamma.shape[0]
    components = ["A", "P", "CC"]
    metrics = {}
    for m in ["R2", "MSE", "Var"]:
        for c in components:
            metrics[f"{m}_{c}"] = np.zeros(n_samples)

    for i in range(n_samples):
        for j, comp in enumerate(components):
            pred_sample, target_sample = preds_gamma[i, j, :], targets_gamma[i, j, :]
            mask = np.isfinite(pred_sample) & np.isfinite(target_sample)

            # Need at least 2 points for R2/Variance
            if np.sum(mask) < 2:
                metrics[f"R2_{comp}"][i] = np.nan
                metrics[f"MSE_{comp}"][i] = np.nan
                metrics[f"Var_{comp}"][i] = np.nan
                continue

            t_clean, p_clean = target_sample[mask], pred_sample[mask]
            variance = np.var(t_clean)
            mse = mean_squared_error(t_clean, p_clean)
            metrics[f"Var_{comp}"][i] = variance
            metrics[f"MSE_{comp}"][i] = mse

            # If variance is effectively zero, R2 is undefined (or 0/1 depending on def).
            # We set to NaN to avoid skewing averages with "perfect" scores on empty images.
            if variance < 1e-9:
                metrics[f"R2_{comp}"][i] = np.nan
            else:
                metrics[f"R2_{comp}"][i] = r2_score(t_clean, p_clean)
    return metrics


def _get_precipitation_groups(mean_precip_col):
    """
    Categorizes samples into Zero, Low, Mid, High based on mean precipitation.
    """
    groups = pd.Series("Zero", index=mean_precip_col.index, dtype=object)
    non_zero_mask = mean_precip_col > 0
    if non_zero_mask.any():
        non_zero_means = mean_precip_col[non_zero_mask]
        p33, p67 = np.quantile(non_zero_means, 0.33), np.quantile(non_zero_means, 0.67)
        bins = [-np.inf, p33, p67, np.inf]
        labels = ["Low", "Mid", "High"]
        binned_data = pd.cut(non_zero_means, bins=bins, labels=labels)
        groups.loc[binned_data.index] = binned_data
    return groups


def create_metrics_dataframe(
    all_preds_gamma,
    all_targets_gamma,
    all_inputs_images,
    all_targets_images,
    all_preds_images,
    all_dems,
    all_total_losses,
    all_mse_losses,
    all_surrogate_losses,
    quantiles,
    pixel_size_km,
):
    print("\nCreating comprehensive metrics DataFrame...")

    # 1. Calculate scalar metrics based on Gamma curves
    sample_metrics = _calculate_per_sample_metrics(all_preds_gamma, all_targets_gamma)

    # 2. Calculate physical mean precipitation for grouping
    mean_precip = np.mean(all_targets_images, axis=(1, 2))

    # 3. Build Base DataFrame
    data = {
        "total_loss": all_total_losses,
        "mse_loss": all_mse_losses,
        "surrogate_loss": all_surrogate_losses,
        "mean_precip": mean_precip,
    }
    data.update(sample_metrics)

    df = pd.DataFrame(data)
    df["precip_group"] = _get_precipitation_groups(df["mean_precip"])

    # 4. Store High-Dimensional Arrays
    df["target_image"] = list(all_targets_images)
    df["pred_image"] = list(all_preds_images)
    df["input_image"] = list(all_inputs_images)
    df["dem_image"] = list(all_dems)
    df["pred_gamma"] = list(all_preds_gamma)
    df["target_gamma"] = list(all_targets_gamma)

    print("DataFrame created.")
    return df


def calculate_grouped_metrics(metrics_df):
    print("\nCalculating grouped metrics...")
    # Select only numeric columns for averaging (exclude images/arrays)
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns

    grouped = metrics_df.groupby("precip_group")
    group_means = grouped[numeric_cols].mean()
    group_counts = grouped.size().to_frame("n_samples")
    group_metrics = pd.concat([group_means, group_counts], axis=1)

    all_metrics = metrics_df[numeric_cols].mean().to_frame("All").T
    all_metrics["n_samples"] = len(metrics_df)
    final_metrics = pd.concat([group_metrics, all_metrics])

    order = ["Zero", "Low", "Mid", "High", "All"]
    return final_metrics.reindex([g for g in order if g in final_metrics.index])


def calculate_per_feature_gamma_metrics(metrics_df, quantiles):
    print("\nCalculating per-feature gamma metrics (Global R2/MSE matrices)...")
    all_preds = np.stack(metrics_df["pred_gamma"].values)
    all_targets = np.stack(metrics_df["target_gamma"].values)

    n_samples, n_components, n_quantiles = all_preds.shape
    preds_flat = all_preds.reshape(n_samples, -1)
    targets_flat = all_targets.reshape(n_samples, -1)

    mask = np.isfinite(targets_flat).all(axis=1) & np.isfinite(preds_flat).all(axis=1)
    valid_preds, valid_targets = preds_flat[mask], targets_flat[mask]

    if len(valid_preds) < 2:
        return {}

    with np.errstate(divide="ignore", invalid="ignore"):
        r2_raw = r2_score(valid_targets, valid_preds, multioutput="raw_values")
    mse_raw = mean_squared_error(valid_targets, valid_preds, multioutput="raw_values")
    var_raw = np.var(valid_targets, axis=0)

    idx = pd.Index(["Area", "Perimeter", "CCs"], name="Component")
    cols = pd.Index(quantiles, name="Quantile (mm/hr)")

    r2_matrix = pd.DataFrame(r2_raw.reshape(3, n_quantiles), index=idx, columns=cols)
    mse_matrix = pd.DataFrame(mse_raw.reshape(3, n_quantiles), index=idx, columns=cols)
    var_matrix = pd.DataFrame(var_raw.reshape(3, n_quantiles), index=idx, columns=cols)

    mean_by_component = pd.DataFrame(
        {
            "Avg_R2": r2_matrix.mean(axis=1),
            "Avg_MSE": mse_matrix.mean(axis=1),
            "Avg_Var": var_matrix.mean(axis=1),
        }
    )

    print("Mean metrics by component:")
    print(mean_by_component.to_string(float_format="%.4f"))

    return {
        "r2_matrix": r2_matrix,
        "mse_matrix": mse_matrix,
        "var_matrix": var_matrix,
        "mean_by_component": mean_by_component,
        "quantiles": quantiles,
    }
