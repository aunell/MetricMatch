"""
Predictor computation for variance-alignment scatter plots.

Provides two predictors that characterise how well inter-model (IM) variance
structure approximates human-model (HM) variance structure for a given
(dataset, axis, model) tuple:

    mean_shift   – (im_msb + im_mse) − (hm_msb + hm_mse) on the full dataset.
                   Captures the global offset between IM and HM variance.

    correlation  – Pearson r between (im_msb_s + im_mse_s) and
                   (hm_msb_s + hm_mse_s) across n_samples bootstrap samples of
                   sample_size text_ids.  Captures how well IM MS tracks HM MS
                   locally across different sub-populations of items.

These predictors are used as y-axes for scatter plots where the x-axis is the
error gap (vs random) for ICC, Krippendorff's Alpha, or MSE.
"""

import numpy as np
from functools import partial

from src.utils.reliability_metrics import compute_ms_components

# Methods compared against random in the predictor scatter plots.
PREDICTOR_COMPARISON_METHODS = [
    "variance_matched_combined",
    "variance_matched_combined_imc",
    "variance_matched_combined_tc_imc",
    "variance_matched_msb",
    "variance_matched_msb_imc",
    "variance_matched_msb_tc",
    "variance_matched_msb_tc_imc",
]


def compute_correlation_predictor(im_full_df, hm_full_df,
                                   n_samples=500, sample_size=10, seed=123,
                                   components="combined"):
    """
    Pearson correlation between IM and HM MS components across bootstrap samples.

    For each of *n_samples* independent draws of *sample_size* text_ids (drawn
    without replacement within each draw; items can recur across draws):
        components="combined": im_val = im_msb + im_mse, hm_val = hm_msb + hm_mse
        components="msb_only": im_val = im_msb,           hm_val = hm_msb

    Returns Pearson r across the n_samples pairs, or np.nan if data are insufficient.

    Args:
        im_full_df: Full inter-model DataFrame (text_id, model_name, evaluation_score).
        hm_full_df: Full human-model DataFrame (text_id, model_name, evaluation_score).
        n_samples: Number of bootstrap draws (default 500).
        sample_size: Text IDs per draw (default 10).
        seed: Random seed.
        components: Which MS components to correlate. "combined" uses msb+mse;
                    "msb_only" uses msb alone (default: "combined").

    Returns:
        float: Pearson correlation coefficient, or np.nan.
    """
    rng = np.random.RandomState(seed)
    fast_ms = partial(compute_ms_components, validate=False)

    im_ids = set(im_full_df["text_id"].unique())
    hm_ids = set(hm_full_df["text_id"].unique())
    shared_ids = np.array(sorted(im_ids & hm_ids))

    if len(shared_ids) < sample_size:
        return np.nan

    im_vals, hm_vals = [], []
    for _ in range(n_samples):
        sample_ids = rng.choice(shared_ids, size=sample_size, replace=False)

        im_ms = fast_ms(im_full_df[im_full_df["text_id"].isin(sample_ids)])
        hm_ms = fast_ms(hm_full_df[hm_full_df["text_id"].isin(sample_ids)])

        if (im_ms is None or hm_ms is None
                or not np.isfinite(im_ms.msb) or not np.isfinite(hm_ms.msb)):
            continue

        if components == "msb_only":
            im_vals.append(im_ms.msb)
            hm_vals.append(hm_ms.msb)
        else:  # "combined"
            if not np.isfinite(im_ms.mse) or not np.isfinite(hm_ms.mse):
                continue
            im_vals.append(im_ms.msb + im_ms.mse)
            hm_vals.append(hm_ms.msb + hm_ms.mse)

    if len(im_vals) < 5:
        return np.nan

    im_arr, hm_arr = np.array(im_vals), np.array(hm_vals)
    if np.std(im_arr) == 0 or np.std(hm_arr) == 0:
        return np.nan
    return float(np.corrcoef(im_arr, hm_arr)[0, 1])


def compute_predictor_records(axis_jobs, per_model_variance_by_axis,
                               icc_results_by_axis, im_df_builder,
                               n_samples=500, sample_size=10):
    """
    Build a list of predictor records for scatter plotting.

    Each record corresponds to one (axis, model) pair and contains:
        axis, model
        mean_shift          – (im_msb+im_mse) − (hm_msb+hm_mse) on the full dataset
        correlation         – Pearson r between im_ms and hm_ms across bootstrap samples
        icc_gap_<method>    – mean_ICC_error(method) − mean_ICC_error(random),
                              averaged across all budgets/trials

    Args:
        axis_jobs: List of (axis, axis_df) pairs from the experiment run.
        per_model_variance_by_axis: Dict mapping axis -> per_model_variance dict
            (each value has im_msb, im_mse, hm_msb, hm_mse keyed by model name,
            as returned by compute_variance_alignment).
        icc_results_by_axis: Dict mapping axis -> ICC results DataFrame
            (columns: model, budget, method, estimation_error).
        im_df_builder: Callable(axis_df, model) -> im_full_df.
            Builds the inter-model DataFrame for a given axis_df and model using
            whatever comparison_mode and ensemble_models are appropriate for the run.
            Defined in the calling experiment so no logic is duplicated here.
        n_samples: Bootstrap samples for the correlation predictor.
        sample_size: Items per bootstrap sample.

    Returns:
        List[dict]: One record per (axis, model) pair.
    """
    # Discover all non-random methods present across all axes
    all_methods: set = set()
    for axis, _ in axis_jobs:
        axis_icc_tmp = icc_results_by_axis.get(axis)
        if axis_icc_tmp is not None and len(axis_icc_tmp) > 0:
            all_methods.update(axis_icc_tmp["method"].unique())
    all_methods.discard("random")
    comparison_methods = sorted(all_methods)

    records = []

    for axis, axis_df in axis_jobs:
        per_model_var = per_model_variance_by_axis.get(axis, {})
        axis_icc = icc_results_by_axis.get(axis)
        if axis_icc is None or len(axis_icc) == 0:
            continue

        for model, var_info in per_model_var.items():
            im_msb = var_info.get("im_msb", np.nan)
            im_mse = var_info.get("im_mse", np.nan)
            hm_msb = var_info.get("hm_msb", np.nan)
            hm_mse = var_info.get("hm_mse", np.nan)

            mean_shift = (
                (im_msb + im_mse) - (hm_msb + hm_mse)
                if all(np.isfinite(v) for v in [im_msb, im_mse, hm_msb, hm_mse])
                else np.nan
            )

            print(f"  Correlation predictor: axis={axis}, model={model} ...")
            im_full_df = im_df_builder(axis_df, model)
            hm_full_df = axis_df[axis_df["model_name"].isin([model, "original"])]
            corr = compute_correlation_predictor(
                im_full_df, hm_full_df,
                n_samples=n_samples, sample_size=sample_size,
                components="combined"
            )
            corr_msb = compute_correlation_predictor(
                im_full_df, hm_full_df,
                n_samples=n_samples, sample_size=sample_size,
                components="msb_only"
            )

            model_icc = axis_icc[axis_icc["model"] == model]
            random_errors = model_icc[model_icc["method"] == "random"]["estimation_error"]
            random_mean = random_errors.mean() if len(random_errors) > 0 else np.nan

            record = {
                "axis": axis,
                "model": model,
                "mean_shift": float(mean_shift) if np.isfinite(mean_shift) else np.nan,
                "correlation": corr,
                "correlation_msb": corr_msb,
            }

            for method in comparison_methods:
                method_errors = model_icc[model_icc["method"] == method]["estimation_error"]
                if len(method_errors) == 0 or not np.isfinite(random_mean):
                    record[f"icc_gap_{method}"] = np.nan
                else:
                    record[f"icc_gap_{method}"] = float(method_errors.mean() - random_mean)

            records.append(record)

    return records
