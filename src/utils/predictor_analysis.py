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
    "variance_matched_weighted_.5",
    "variance_matched_weighted_.5_imc",
    "variance_matched_weighted_.7",
    "variance_matched_weighted_.7_imc",
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
        elif components == "mse_only":
            if not np.isfinite(im_ms.mse) or not np.isfinite(hm_ms.mse):
                continue
            im_vals.append(im_ms.mse)
            hm_vals.append(hm_ms.mse)
        elif components == "icc_only":
            if (not hasattr(im_ms, "icc") or not hasattr(hm_ms, "icc")
                    or not np.isfinite(im_ms.icc) or not np.isfinite(hm_ms.icc)):
                continue
            im_vals.append(im_ms.icc)
            hm_vals.append(hm_ms.icc)
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


def compute_ms_budget_samples(im_full_df, hm_full_df,
                               budgets=(10, 20, 30, 40, 50),
                               n_samples=200, seed=123):
    """
    For each budget, draw n_samples subsets of `budget` text_ids and compute
    (im_msb+im_mse, hm_msb+hm_mse) for each subset.

    Args:
        im_full_df: Inter-model DataFrame (text_id, model_name, evaluation_score).
        hm_full_df: Human-model DataFrame (text_id, model_name, evaluation_score).
        budgets: Sequence of sample sizes to evaluate.
        n_samples: Number of random draws per budget.
        seed: Random seed.

    Returns:
        dict mapping budget (int) -> list of (im_ms_val, hm_ms_val) tuples.
    """
    rng = np.random.RandomState(seed)
    fast_ms = partial(compute_ms_components, validate=False)

    im_ids = set(im_full_df["text_id"].unique())
    hm_ids = set(hm_full_df["text_id"].unique())
    shared_ids = np.array(sorted(im_ids & hm_ids))

    results = {}
    for budget in budgets:
        if len(shared_ids) < budget:
            results[budget] = []
            continue

        combined, msb_only, mse_only, icc_pairs = [], [], [], []
        for _ in range(n_samples):
            sample_ids = rng.choice(shared_ids, size=budget, replace=False)

            im_ms = fast_ms(im_full_df[im_full_df["text_id"].isin(sample_ids)])
            hm_ms = fast_ms(hm_full_df[hm_full_df["text_id"].isin(sample_ids)])

            if im_ms is None or hm_ms is None:
                continue

            if (np.isfinite(im_ms.msb) and np.isfinite(hm_ms.msb)
                    and np.isfinite(im_ms.mse) and np.isfinite(hm_ms.mse)):
                combined.append((im_ms.msb + im_ms.mse, hm_ms.msb + hm_ms.mse))

            if np.isfinite(im_ms.msb) and np.isfinite(hm_ms.msb):
                msb_only.append((im_ms.msb, hm_ms.msb))

            if np.isfinite(im_ms.mse) and np.isfinite(hm_ms.mse):
                mse_only.append((im_ms.mse, hm_ms.mse))

            if (hasattr(im_ms, "icc") and hasattr(hm_ms, "icc")
                    and np.isfinite(im_ms.icc) and np.isfinite(hm_ms.icc)):
                icc_pairs.append((im_ms.icc, hm_ms.icc))

        results[budget] = {"combined": combined, "msb": msb_only, "mse": mse_only,
                           "icc": icc_pairs}

    return results


def compute_predictor_records(axis_jobs, per_model_variance_by_axis,
                               icc_results_by_axis, im_df_builder,
                               alpha_results_by_axis=None,
                               mse_results_by_axis=None,
                               n_samples=500, sample_size=10,
                               pairwise_stats_builder=None):
    """
    Build a list of predictor records for scatter plotting.

    Each record corresponds to one (axis, model) pair and contains:
        axis, model
        mean_shift            – (im_msb+im_mse) − (hm_msb+hm_mse) on the full dataset
        correlation           – Pearson r between im_ms and hm_ms across bootstrap samples
        correlation_msb       – Pearson r between im_msb and hm_msb across bootstrap samples
        icc_gap_<method>      – mean_ICC_error(method) − mean_ICC_error(random)
        alpha_gap_<method>    – mean_Alpha_error(method) − mean_Alpha_error(random)
                                (only if alpha_results_by_axis is provided)
        mse_gap_<method>      – mean_MSE_error(method) − mean_MSE_error(random)
                                (only if mse_results_by_axis is provided)

    Args:
        axis_jobs: List of (axis, axis_df) pairs from the experiment run.
        per_model_variance_by_axis: Dict mapping axis -> per_model_variance dict
            (each value has im_msb, im_mse, hm_msb, hm_mse keyed by model name,
            as returned by compute_variance_alignment).
        icc_results_by_axis: Dict mapping axis -> ICC results DataFrame
            (columns: model, budget, method, estimation_error).
        im_df_builder: Callable(axis_df, model) -> im_full_df.
        alpha_results_by_axis: Optional dict mapping axis -> Alpha results DataFrame.
        mse_results_by_axis: Optional dict mapping axis -> MSE results DataFrame.
        n_samples: Bootstrap samples for the correlation predictor.
        sample_size: Items per bootstrap sample.

    Returns:
        List[dict]: One record per (axis, model) pair.
    """
    # Discover all non-random methods present across all axes (union across all metrics)
    all_methods: set = set()
    for results_by_axis in [icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis]:
        if results_by_axis is None:
            continue
        for axis, _ in axis_jobs:
            axis_res = results_by_axis.get(axis)
            if axis_res is not None and len(axis_res) > 0:
                all_methods.update(axis_res["method"].unique())
    all_methods.discard("random")
    comparison_methods = sorted(all_methods)

    records = []

    for axis, axis_df in axis_jobs:
        per_model_var = per_model_variance_by_axis.get(axis, {})
        axis_icc = icc_results_by_axis.get(axis)
        if axis_icc is None or len(axis_icc) == 0:
            continue

        axis_alpha = alpha_results_by_axis.get(axis) if alpha_results_by_axis else None
        axis_mse = mse_results_by_axis.get(axis) if mse_results_by_axis else None

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

            # Pairwise-averaged inter-model stats (target vs each other model, averaged)
            if pairwise_stats_builder is not None:
                pw = pairwise_stats_builder(axis_df, model)
                im_msb = pw.get("im_msb", im_msb)
                im_mse = pw.get("im_mse", im_mse)
                im_icc = pw.get("im_icc", np.nan)
            else:
                _full_im_ms = compute_ms_components(im_full_df, validate=False)
                im_icc = (
                    float(_full_im_ms.icc)
                    if (_full_im_ms is not None
                        and hasattr(_full_im_ms, "icc")
                        and np.isfinite(_full_im_ms.icc))
                    else np.nan
                )

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
            corr_mse = compute_correlation_predictor(
                im_full_df, hm_full_df,
                n_samples=n_samples, sample_size=sample_size,
                components="mse_only"
            )
            corr_icc = compute_correlation_predictor(
                im_full_df, hm_full_df,
                n_samples=n_samples, sample_size=sample_size,
                components="icc_only"
            )
            # Per-budget correlations (10 draws of `budget` items each)
            _budget_corrs: dict = {}
            for _b in sorted(set(
                int(c.split("_b")[-1])
                for results in [icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis]
                if results is not None
                for _ax_res in [results.get(axis)]
                if _ax_res is not None and "budget" in _ax_res.columns
                for c in [f"_b{int(bv)}" for bv in _ax_res["budget"].unique()]
            )):
                for _comp, _suffix in [("msb_only", "msb"), ("mse_only", "mse"), ("icc_only", "icc")]:
                    _budget_corrs[f"correlation_{_suffix}_b{_b}"] = compute_correlation_predictor(
                        im_full_df, hm_full_df,
                        n_samples=10, sample_size=_b,
                        components=_comp,
                    )

            record = {
                "axis": axis,
                "model": model,
                "mean_shift": float(mean_shift) if np.isfinite(mean_shift) else np.nan,
                "im_msb": float(im_msb) if np.isfinite(im_msb) else np.nan,
                "im_mse": float(im_mse) if np.isfinite(im_mse) else np.nan,
                "im_icc": im_icc,
                "hm_msb": float(hm_msb) if np.isfinite(hm_msb) else np.nan,
                "hm_mse": float(hm_mse) if np.isfinite(hm_mse) else np.nan,
                "correlation": corr,
                "correlation_msb": corr_msb,
                "correlation_mse": corr_mse,
                "correlation_icc": corr_icc,
                **_budget_corrs,
            }

            # Compute error gaps for each metric (across all budgets and per budget)
            metric_results = [
                ("icc",   axis_icc),
                ("alpha", axis_alpha),
                ("mse",   axis_mse),
            ]
            for metric_name, axis_res in metric_results:
                if axis_res is None or len(axis_res) == 0:
                    continue
                model_res = axis_res[axis_res["model"] == model]

                # Across-all-budgets gap
                random_mean = (
                    model_res[model_res["method"] == "random"]["estimation_error"].mean()
                    if len(model_res[model_res["method"] == "random"]) > 0
                    else np.nan
                )
                for method in comparison_methods:
                    method_errors = model_res[model_res["method"] == method]["estimation_error"]
                    if len(method_errors) == 0 or not np.isfinite(random_mean):
                        record[f"{metric_name}_gap_{method}"] = np.nan
                    else:
                        record[f"{metric_name}_gap_{method}"] = float(
                            method_errors.mean() - random_mean
                        )

                # Per-budget gap
                if "budget" in model_res.columns:
                    for budget in sorted(model_res["budget"].unique()):
                        budget_res = model_res[model_res["budget"] == budget]
                        random_budget_mean = (
                            budget_res[budget_res["method"] == "random"]["estimation_error"].mean()
                            if len(budget_res[budget_res["method"] == "random"]) > 0
                            else np.nan
                        )
                        for method in comparison_methods:
                            method_budget_errors = budget_res[budget_res["method"] == method]["estimation_error"]
                            if len(method_budget_errors) == 0 or not np.isfinite(random_budget_mean):
                                record[f"{metric_name}_gap_{method}_b{budget}"] = np.nan
                            else:
                                record[f"{metric_name}_gap_{method}_b{budget}"] = float(
                                    method_budget_errors.mean() - random_budget_mean
                                )

            records.append(record)

    return records
