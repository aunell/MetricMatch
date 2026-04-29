"""
Adds `im_metric_variance_sampled` to delta_ranked.csv.

For each row (dataset, model, axis, metric):
  - Load the inter-model axis_data for that axis
  - Sample 10 sets of 25 text_ids
  - For each sample: compute the target metric pairwise between target_model
    and each other ensemble model on those 25 items, then average across pairs
  - Record variance of the 10 averaged metric values → im_metric_variance_sampled

Then prints Pearson r of im_metric_variance_sampled with:
  - variance_delta
  - delta (estimation error delta)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/share/pi/nigam/users/aunell/SmartSample_local")
from src.utils.reliability_metrics import (
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_spearman_rho,
    compute_kendall_tau,
    compute_ms_components,
)

RESULTS_DIR = "/share/pi/nigam/users/aunell/SmartSample_local/results/04_22_add_metric_match"
CSV_PATH = os.path.join(RESULTS_DIR, "model_variance_analysis", "delta_ranked.csv")
N_ITEMS = 10
N_SAMPLES = 25
SEED = 42

METRIC_FN = {
    "icc":   compute_icc_pingouin,
    "alpha": compute_krippendorff_alpha,
    "rho":   compute_spearman_rho,
    "tau":   compute_kendall_tau,
}


def load_axis_data(dataset, axis):
    path = os.path.join(
        RESULTS_DIR, dataset, dataset, "dataframes", "predictor_inputs",
        f"axis_data_{axis}.csv",
    )
    return pd.read_csv(path) if os.path.exists(path) else None


def load_ensemble_models(dataset):
    path = os.path.join(
        RESULTS_DIR, dataset, dataset, "dataframes", "predictor_inputs",
        "predictor_config.json",
    )
    with open(path) as f:
        return json.load(f)["ensemble_models"]


def compute_sampled_msb(df, target_model, other_models, rng):
    """
    Same sampling scheme as compute_sampled_metric but computes MSB pairwise,
    averages across pairs, then returns variance across N_SAMPLES.
    """
    text_ids = df["text_id"].unique()
    if len(text_ids) < N_ITEMS:
        return np.nan

    sample_values = []
    for _ in range(N_SAMPLES):
        sampled_ids = rng.choice(text_ids, size=N_ITEMS, replace=False)
        sub = df[df["text_id"].isin(sampled_ids)]

        pair_msbs = []
        for other in other_models:
            pair_df = sub[sub["model_name"].isin([target_model, other])].copy()
            if pair_df["model_name"].nunique() < 2:
                continue
            icc_obj = compute_ms_components(pair_df, validate=True)
            if icc_obj is not None and np.isfinite(icc_obj.msb):
                pair_msbs.append(icc_obj.msb)

        if pair_msbs:
            sample_values.append(np.mean(pair_msbs))

    if len(sample_values) < 2:
        return np.nan
    return float(np.var(sample_values, ddof=1))


def compute_delta_im(df, ensemble_models, metric_fn, rng):
    """
    Sample N_ITEMS text_ids N_SAMPLES times.
    Each sample: compute metric across ALL ensemble models together → one value.
    Average the N_SAMPLES estimates, subtract the true population metric
    (computed on all items using all ensemble models).
    Returns: mean_subset_estimate - true_population_metric.
    """
    # restrict to items where every ensemble model has a score
    counts = df.groupby("text_id")["model_name"].nunique()
    valid_ids = counts[counts == len(ensemble_models)].index
    df = df[df["text_id"].isin(valid_ids)]

    text_ids = df["text_id"].unique()
    if len(text_ids) < N_ITEMS:
        return np.nan

    true_pop = metric_fn(df)
    if not np.isfinite(true_pop):
        return np.nan

    estimates = []
    for _ in range(N_SAMPLES):
        sampled_ids = rng.choice(text_ids, size=N_ITEMS, replace=False)
        sub = df[df["text_id"].isin(sampled_ids)]
        val = metric_fn(sub)
        if np.isfinite(val):
            estimates.append(val)

    if not estimates:
        return np.nan
    return float(np.mean(estimates)) - true_pop


def compute_sampled_metric(df, target_model, other_models, metric_fn, rng):
    """
    Sample N_ITEMS text_ids N_SAMPLES times.
    Each sample: compute metric pairwise (target vs. each other), average → one value.
    Return variance of the N_SAMPLES averaged values.
    """
    text_ids = df["text_id"].unique()
    if len(text_ids) < N_ITEMS:
        return np.nan

    sample_values = []
    for _ in range(N_SAMPLES):
        sampled_ids = rng.choice(text_ids, size=N_ITEMS, replace=False)
        sub = df[df["text_id"].isin(sampled_ids)]

        pair_metrics = []
        for other in other_models:
            pair_df = sub[sub["model_name"].isin([target_model, other])].copy()
            # ensure both models present
            if pair_df["model_name"].nunique() < 2:
                continue
            val = metric_fn(pair_df)
            if np.isfinite(val):
                pair_metrics.append(val)

        if pair_metrics:
            sample_values.append(np.mean(pair_metrics))

    if len(sample_values) < 2:
        return np.nan
    return float(np.var(sample_values, ddof=1))


def main():
    df = pd.read_csv(CSV_PATH)
    rng = np.random.RandomState(SEED)

    # Cache axis data per (dataset, axis) to avoid re-reading
    axis_cache = {}
    ensemble_cache = {}

    metric_results = []
    msb_results = []
    delta_im_results = []
    # cache true population metric per (dataset, axis, metric) for delta_im
    true_pop_cache = {}
    for _, row in df.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        axis = row["axis"]
        metric = row["metric"]

        key = (dataset, axis)
        if key not in axis_cache:
            axis_cache[key] = load_axis_data(dataset, axis)
        if dataset not in ensemble_cache:
            ensemble_cache[dataset] = load_ensemble_models(dataset)

        axis_df = axis_cache[key]
        ensemble = ensemble_cache[dataset]
        metric_fn = METRIC_FN.get(metric)

        if axis_df is None or metric_fn is None or model not in ensemble:
            metric_results.append(np.nan)
            msb_results.append(np.nan)
            delta_im_results.append(np.nan)
            continue

        im_df = axis_df[axis_df["model_name"].isin(ensemble)].copy()
        others = [m for m in ensemble if m != model]

        var_val = compute_sampled_metric(im_df, model, others, metric_fn, rng)
        msb_val = compute_sampled_msb(im_df, model, others, rng)

        pop_key = (dataset, axis, metric)
        if pop_key not in true_pop_cache:
            delta_im_val = compute_delta_im(im_df, ensemble, metric_fn, rng)
            true_pop_cache[pop_key] = delta_im_val
        delta_im_val = true_pop_cache[pop_key]

        metric_results.append(var_val)
        msb_results.append(msb_val)
        delta_im_results.append(delta_im_val)
        print(f"  {dataset}/{axis}/{model}/{metric}: metric_var={var_val:.6f}, msb_var={msb_val:.6f}, delta_im={delta_im_val:.6f}"
              if np.isfinite(var_val) and np.isfinite(msb_val) and np.isfinite(delta_im_val)
              else f"  {dataset}/{axis}/{model}/{metric}: NaN")

    df["im_metric_variance_sampled"] = metric_results
    df["im_msb_variance_sampled"] = msb_results
    df["delta_im"] = delta_im_results
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved updated CSV to {CSV_PATH}")

    # Report correlations
    valid = df.dropna(subset=["im_metric_variance_sampled", "im_msb_variance_sampled", "variance_delta", "delta"])
    print(f"\n{len(valid)} rows with valid values")

    for xcol, xlabel in [("im_metric_variance_sampled", "im_metric_variance_sampled"),
                         ("im_msb_variance_sampled",    "im_msb_variance_sampled")]:
        x = valid[xcol].values
        for col, label in [("variance_delta", "variance_delta"), ("delta", "estimation_error_delta")]:
            y = valid[col].values
            r, p = stats.pearsonr(x, y)
            print(f"\nCorrelation: {xlabel} vs {label}")
            print(f"  Pearson r = {r:.4f}, p = {p:.4f}")

    print("\n--- By metric ---")
    for metric in df["metric"].unique():
        sub = valid[valid["metric"] == metric]
        if len(sub) < 3:
            continue
        for xcol, xlabel in [("im_metric_variance_sampled", "im_metric_var"),
                              ("im_msb_variance_sampled",    "im_msb_var")]:
            x_m = sub[xcol].values
            for col, label in [("variance_delta", "variance_delta"), ("delta", "est_err_delta")]:
                y_m = sub[col].values
                r, p = stats.pearsonr(x_m, y_m)
                print(f"  [{metric}] {xlabel} vs {label}: r={r:.4f}, p={p:.4f}")

    # delta_im vs delta correlation
    valid_dim = df.dropna(subset=["delta_im", "delta"])
    print(f"\n--- delta_im vs delta (n={len(valid_dim)}) ---")
    r, p = stats.pearsonr(valid_dim["delta_im"].values, valid_dim["delta"].values)
    r_sp, p_sp = stats.spearmanr(valid_dim["delta_im"].values, valid_dim["delta"].values)
    print(f"  Pearson  r = {r:.4f}, p = {p:.4f}")
    print(f"  Spearman r = {r_sp:.4f}, p = {p_sp:.4f}")
    print("\n--- delta_im vs delta by metric ---")
    for metric in df["metric"].unique():
        sub = valid_dim[valid_dim["metric"] == metric]
        if len(sub) < 3:
            continue
        r, p = stats.pearsonr(sub["delta_im"].values, sub["delta"].values)
        print(f"  [{metric}] Pearson r = {r:.4f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
