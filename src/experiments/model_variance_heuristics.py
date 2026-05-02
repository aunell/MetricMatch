"""
Experiment 9: Predictors of Estimation Error Improvement over Random.

For each (dataset, axis, target_model, metric) we compute:
  y = mean(estimation_error_variance_matched_msb) − mean(estimation_error_random)
      averaged over budgets  (negative = our method beats random)

And four candidate x predictors:
  1. mean_msb      — mean inter-model MSB across the 4 pairwise comparisons
  2. msb_variance  — variance of those 4 pairwise MSBs
  3. im_hm_ratio   — im_msb / hm_msb  (proxy alignment with target)
  4. snr           — im_msb / im_mse  (signal-to-noise of inter-model structure)
  5. true_metric   — true reliability value at full sample (per metric)

Produces a 5×4 subplot grid (predictors × metrics) with R² in each title.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.reliability_metrics import (
    compute_ms_components,
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_spearman_rho,
    compute_kendall_tau,
)
from src.utils.match_metrics import compute_mean_sq_err

RESULTS_DIR = "/Users/alyssaunell/code/SmartSample_local/data/metric_matched_subsets"
DATASETS = ["hanna", "mslr", "summeval", "medval"]
METRICS = ["icc", "alpha", "spearman", "kendalltau", "mean_sq_error"]
OUR_METHOD = "metric_match"
BASELINE = "random"

# Mapping of datasets to their predictor input directories
PREDICTOR_DIRS = {
    "hanna": "/Users/alyssaunell/code/SmartSample_local/results/04_31_hanna/hanna/dataframes/predictor_inputs",
    "medval": "/Users/alyssaunell/code/SmartSample_local/results/04_30_medval/medval/dataframes/predictor_inputs",
    "mslr": "/Users/alyssaunell/code/SmartSample_local/results/03_17/mslr/mslr/dataframes/predictor_inputs",
    "summeval": "/Users/alyssaunell/code/SmartSample_local/results/03_17/summeval/summeval/dataframes/predictor_inputs",
}

DATASET_COLORS = {
    "hanna": "#1f77b4",
    "mslr": "#ff7f0e",
    "summeval": "#2ca02c",
    "medval": "#d62728",
}

PREDICTORS = [
    ("mean_msb",     "Mean Inter-Model MSB"),
    ("msb_variance", "Variance of Pairwise MSBs"),
    ("im_hm_ratio",  "IM-MSB / HM-MSB Ratio"),
    ("snr",          "MSB / MSE (Signal-to-Noise)"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_axes(dataset):
    """Get available axes for a dataset from its predictor inputs directory."""
    predictor_dir = PREDICTOR_DIRS.get(dataset)
    if not predictor_dir or not os.path.exists(predictor_dir):
        return []

    # Check for axes.json first
    axes_json = os.path.join(predictor_dir, "axes.json")
    if os.path.exists(axes_json):
        with open(axes_json) as f:
            return json.load(f)

    # Otherwise, infer from axis_data_*.csv files
    files = [f for f in os.listdir(predictor_dir) if f.startswith("axis_data_") and f.endswith(".csv")]
    return [f.replace("axis_data_", "").replace(".csv", "") for f in files]


def load_predictor_config(dataset):
    """Load predictor configuration for a dataset."""
    predictor_dir = PREDICTOR_DIRS.get(dataset)
    if not predictor_dir:
        return None

    config_path = os.path.join(predictor_dir, "predictor_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return None


def load_per_model_variance(dataset):
    """Load per-model variance data for a dataset."""
    predictor_dir = PREDICTOR_DIRS.get(dataset)
    if not predictor_dir:
        return {}

    var_path = os.path.join(predictor_dir, "per_model_variance.json")
    if os.path.exists(var_path):
        with open(var_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Predictor computation
# ---------------------------------------------------------------------------

def compute_msb_features(datasets):
    """
    For each (dataset, axis, target_model): compute
      - 4 pairwise MSBs via compute_ms_components → mean_msb, msb_variance
      - im_msb, hm_msb, im_mse from per_model_variance.json → im_hm_ratio, snr
    """
    records = []
    for dataset in datasets:
        config = load_predictor_config(dataset)
        if not config:
            print(f"  Warning: No predictor config found for {dataset}, skipping MSB features")
            continue

        ensemble_models = config["ensemble_models"]
        per_model_var = load_per_model_variance(dataset)
        axes = load_axes(dataset)

        if not axes:
            print(f"  Warning: No axes found for {dataset}, skipping")
            continue

        predictor_dir = PREDICTOR_DIRS.get(dataset)
        if not predictor_dir:
            print(f"  Warning: No predictor directory configured for {dataset}, skipping")
            continue

        for axis in axes:
            csv_path = os.path.join(predictor_dir, f"axis_data_{axis}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            df = df[df["model_name"].isin(ensemble_models)]

            for target_model in ensemble_models:
                others = [m for m in ensemble_models if m != target_model]
                msbs = []
                for other in others:
                    pair_df = df[df["model_name"].isin([target_model, other])]
                    icc_obj = compute_ms_components(pair_df)
                    if icc_obj is not None:
                        msbs.append(icc_obj.msb)

                if len(msbs) < 2:
                    continue

                # im_hm_ratio and snr from per_model_variance.json
                axis_var = per_model_var.get(axis, {}).get(target_model, {})
                im_msb = axis_var.get("im_msb", np.nan)
                hm_msb = axis_var.get("hm_msb", np.nan)
                im_mse = axis_var.get("im_mse", np.nan)

                hm_mse = axis_var.get("hm_mse", np.nan)

                im_hm_ratio = im_msb / hm_msb if hm_msb and hm_msb != 0 else np.nan
                snr = im_msb / im_mse if im_mse and im_mse != 0 else np.nan
                hm_snr = hm_msb / hm_mse if hm_mse and hm_mse != 0 else np.nan

                records.append({
                    "dataset": dataset,
                    "axis": axis,
                    "model": target_model,
                    "mean_msb": float(np.mean(msbs)),
                    "msb_variance": float(np.var(msbs, ddof=1)),
                    "im_hm_ratio": im_hm_ratio,
                    "snr": snr,
                    "hm_snr": hm_snr,
                    "n_pairs": len(msbs),
                })

    return pd.DataFrame(records)


def load_estimation_errors(results_dir, datasets, metrics):
    """
    Load estimation errors from the combined results CSV files.
    New format: model, budget, method, match_metric, est_metric, est_error, axis
    """
    frames = []
    for dataset in datasets:
        csv_path = os.path.join(results_dir, f"{dataset}_combined_results2.csv")
        if not os.path.exists(csv_path):
            print(f"  Missing: {dataset}_combined_results2.csv — skipping")
            continue

        df = pd.read_csv(csv_path, low_memory=False)

        # Filter for methods we care about
        available = [m for m in [OUR_METHOD, BASELINE] if m in df["method"].values]
        df = df[df["method"].isin(available)].copy()

        # Rename columns to match expected format
        df = df.rename(columns={
            "est_error": "estimation_error",
            "est_metric": "metric"
        })

        # Add dataset column
        df["dataset"] = dataset

        # Filter for metrics we care about
        df = df[df["metric"].isin(metrics)].copy()

        # Select relevant columns
        cols = ["model", "budget", "method", "estimation_error", "axis", "dataset", "metric"]
        df = df[cols].copy()

        frames.append(df)

    if not frames:
        raise RuntimeError("No estimation error data found.")
    return pd.concat(frames, ignore_index=True)


def compute_improvement(est_df):
    """
    Improvement = mean(our_err) − mean(random_err), averaged over budgets.
    """
    avg = (
        est_df.groupby(["dataset", "axis", "model", "metric", "method"])
        .agg(estimation_error=("estimation_error", "mean"))
        .reset_index()
    )
    ours = avg[avg["method"] == OUR_METHOD].rename(columns={"estimation_error": "our_err"}).drop(columns="method")
    base = avg[avg["method"] == BASELINE].rename(columns={"estimation_error": "random_err"}).drop(columns="method")
    merged = ours.merge(base[["dataset", "axis", "model", "metric", "random_err"]],
                        on=["dataset", "axis", "model", "metric"])
    merged["improvement"] = merged["our_err"] - merged["random_err"]
    return merged


# ---------------------------------------------------------------------------
# IM Metric Variance Computation
# ---------------------------------------------------------------------------

def compute_im_metric_variance(dataset, axis, models, n_samples=100, budget_sizes=[5, 10, 20, 30, 40, 50]):
    """
    Compute variance of IM metric predictions by randomly sampling subsets.

    Args:
        dataset: Dataset name
        axis: Evaluation axis
        models: List of model names for IM comparison
        n_samples: Number of random samples per budget
        budget_sizes: List of budget sizes to test

    Returns:
        DataFrame with variance values for each metric and budget
    """
    predictor_dir = PREDICTOR_DIRS.get(dataset)
    if not predictor_dir:
        print(f"No predictor directory for {dataset}")
        return None

    axis_path = os.path.join(predictor_dir, f"axis_data_{axis}.csv")
    if not os.path.exists(axis_path):
        print(f"No axis data for {dataset}/{axis}")
        return None

    df = pd.read_csv(axis_path)
    df = df[df["model_name"].isin(models)]

    # Get available text_ids that have all models
    text_id_counts = df.groupby("text_id")["model_name"].nunique()
    shared_ids = text_id_counts[text_id_counts == len(models)].index.values

    if len(shared_ids) == 0:
        print(f"No shared text_ids for {dataset}/{axis} with models {models}")
        return None

    results = []

    for budget in budget_sizes:
        if budget > len(shared_ids):
            continue

        metric_predictions = {
            "icc": [],
            "alpha": [],
            "spearman": [],
            "kendalltau": [],
            "mean_sq_error": []
        }

        for trial in range(n_samples):
            # Randomly sample text_ids
            np.random.seed(42 + trial)
            sampled_ids = np.random.choice(shared_ids, size=budget, replace=False)
            subset_df = df[df["text_id"].isin(sampled_ids)]

            # Compute each metric
            try:
                icc = compute_icc_pingouin(subset_df, models=models)
                if np.isfinite(icc):
                    metric_predictions["icc"].append(icc)
            except:
                pass

            try:
                alpha = compute_krippendorff_alpha(subset_df, models=models)
                if np.isfinite(alpha):
                    metric_predictions["alpha"].append(alpha)
            except:
                pass

            try:
                rho = compute_spearman_rho(subset_df, models=models)
                if np.isfinite(rho):
                    metric_predictions["spearman"].append(rho)
            except:
                pass

            try:
                tau = compute_kendall_tau(subset_df, models=models)
                if np.isfinite(tau):
                    metric_predictions["kendalltau"].append(tau)
            except:
                pass

            try:
                mse = compute_mean_sq_err(subset_df)
                if mse is not None and np.isfinite(mse):
                    metric_predictions["mean_sq_error"].append(mse)
            except:
                pass

        # Compute variance for each metric
        for metric, preds in metric_predictions.items():
            if len(preds) > 1:
                variance = np.var(preds, ddof=1)
                results.append({
                    "dataset": dataset,
                    "axis": axis,
                    "budget": budget,
                    "metric": metric,
                    "im_metric_variance": variance,
                    "n_valid_samples": len(preds)
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_panel(ax, x, y, plot_df, metric, predictor_col, predictor_label):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    sub_df = plot_df[mask]

    if len(x) < 3:
        ax.set_title(f"{metric.upper()} | {predictor_label}\n(insufficient data)", fontsize=9)
        return None, None, None

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    r2 = r_value ** 2

    for dataset in DATASETS:
        dsub = sub_df[sub_df["dataset"] == dataset]
        if dsub.empty:
            continue
        ax.scatter(dsub[predictor_col], dsub["improvement"],
                   color=DATASET_COLORS.get(dataset, "gray"),
                   label=dataset, s=35, alpha=0.7, zorder=3)

    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6, linewidth=1.2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_title(f"{metric.upper()} | {predictor_label}\nR²={r2:.3f}, r={r_value:.3f}, p={p_value:.3f}", fontsize=8)
    ax.grid(alpha=0.25)
    return r2, r_value, p_value


def plot_extremes(plot_df, output_dir, n=30):
    """
    For each metric, take the top-n and bottom-n rows by delta (= -improvement),
    then run the predictor regressions on just those 2n points.
    Markers distinguish top (▲ beats random most) from bottom (▼ random beats us most).
    """
    os.makedirs(output_dir, exist_ok=True)
    n_pred = len(PREDICTORS)
    n_met = len(METRICS)

    fig, axes = plt.subplots(n_pred, n_met, figsize=(4 * n_met, 3.5 * n_pred))

    all_results = {}
    for row, (pred_col, pred_label) in enumerate(PREDICTORS):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            mdf = plot_df[plot_df["metric"] == metric].copy()
            mdf["delta"] = -mdf["improvement"]

            top = mdf.nlargest(n, "delta")
            bot = mdf.nsmallest(n, "delta")
            sub = pd.concat([top, bot])
            sub["group"] = ["top"] * len(top) + ["bot"] * len(bot)

            x_vals = sub[pred_col].values
            y_vals = sub["improvement"].values

            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            if mask.sum() < 3:
                continue

            slope, intercept, r_value, p_value, _ = stats.linregress(x_vals[mask], y_vals[mask])
            r2 = r_value ** 2
            all_results[(pred_col, metric)] = (r2, r_value, p_value)

            for dataset in DATASETS:
                for grp, marker, size in [("top", "^", 55), ("bot", "v", 55)]:
                    dsub = sub[(sub["dataset"] == dataset) & (sub["group"] == grp)]
                    if dsub.empty:
                        continue
                    xp = dsub[pred_col].values
                    ax.scatter(xp, dsub["improvement"],
                               color=DATASET_COLORS.get(dataset, "gray"),
                               marker=marker, s=size, alpha=0.75, zorder=3)

            x_line = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 200)
            ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6, linewidth=1.2)
            ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.set_title(
                f"{metric.upper()} | {pred_label}\nR²={r2:.3f}, r={r_value:.3f}, p={p_value:.3f}",
                fontsize=8,
            )
            ax.grid(alpha=0.25)

            if col == 0:
                ax.set_ylabel("improvement (our − random)", fontsize=7)

    # Legend: datasets by color, groups by marker
    color_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=d)
                     for d, c in DATASET_COLORS.items()]
    group_handles = [
        plt.Line2D([0], [0], marker="^", color="gray", markersize=7, linestyle="", label=f"top {n} (we win most)"),
        plt.Line2D([0], [0], marker="v", color="gray", markersize=7, linestyle="", label=f"bot {n} (random wins most)"),
    ]
    fig.legend(handles=color_handles + group_handles, loc="lower center",
               ncol=len(DATASETS) + 2, bbox_to_anchor=(0.5, -0.02), fontsize=8)

    fig.suptitle(
        f"Predictors vs. Improvement — Extremes Only (top {n} + bottom {n} per metric)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    out_path = os.path.join(output_dir, "predictors_vs_improvement_extremes.jpg")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")
    return all_results


def _snr_figure_by_metric(plot_df, pred_col, pred_label, output_dir, filename):
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4), sharey=False)
    for col, metric in enumerate(METRICS):
        ax = axes[col]
        mdf = plot_df[plot_df["metric"] == metric].copy()
        x = mdf[pred_col].values
        y = mdf["improvement"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x_f, y_f, mdf_f = x[mask], y[mask], mdf[mask]

        for dataset in DATASETS:
            dsub = mdf_f[mdf_f["dataset"] == dataset]
            if dsub.empty:
                continue
            ax.scatter(dsub[pred_col], dsub["improvement"],
                       color=DATASET_COLORS[dataset], label=dataset,
                       s=35, alpha=0.7, zorder=3)

        if len(x_f) >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(x_f, y_f)
            x_line = np.linspace(x_f.min(), x_f.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6, linewidth=1.2)
            r2 = r_value ** 2
            ax.set_title(f"{metric.upper()}\nR²={r2:.3f}, r={r_value:.3f}, p={p_value:.3f}", fontsize=9)
        else:
            ax.set_title(f"{metric.upper()}\n(insufficient data)", fontsize=9)

        ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.set_xlabel(pred_label, fontsize=8)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("Estimation error delta\n(our − random)", fontsize=8)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=DATASET_COLORS[d], markersize=7, label=d)
               for d in DATASETS]
    fig.legend(handles=handles, title="Dataset", loc="lower center",
               ncol=len(DATASETS), bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(f"{pred_label} vs. Estimation Error Delta — by Metric", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def _snr_figure_by_dataset(plot_df, pred_col, pred_label, output_dir, filename):
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4.5 * n_datasets, 4), sharey=False)
    metric_colors = {m: f"C{i}" for i, m in enumerate(METRICS)}

    for col, dataset in enumerate(DATASETS):
        ax = axes[col]
        ddf = plot_df[plot_df["dataset"] == dataset].copy()
        x = ddf[pred_col].values
        y = ddf["improvement"].values
        mask = np.isfinite(x) & np.isfinite(y)
        x_f, y_f, ddf_f = x[mask], y[mask], ddf[mask]

        for metric in METRICS:
            msub = ddf_f[ddf_f["metric"] == metric]
            if msub.empty:
                continue
            ax.scatter(msub[pred_col], msub["improvement"],
                       color=metric_colors[metric], label=metric.upper(),
                       s=35, alpha=0.7, zorder=3)

        if len(x_f) >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(x_f, y_f)
            x_line = np.linspace(x_f.min(), x_f.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6, linewidth=1.2)
            r2 = r_value ** 2
            ax.set_title(f"{dataset}\nR²={r2:.3f}, r={r_value:.3f}, p={p_value:.3f}", fontsize=9)
        else:
            ax.set_title(f"{dataset}\n(insufficient data)", fontsize=9)

        ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.set_xlabel(pred_label, fontsize=8)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("Estimation error delta\n(our − random)", fontsize=8)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=metric_colors[m], markersize=7, label=m.upper())
               for m in METRICS]
    fig.legend(handles=handles, title="Metric", loc="lower center",
               ncol=len(METRICS), bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(f"{pred_label} vs. Estimation Error Delta — by Dataset", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def compute_all_im_metric_variances(datasets, n_samples=100):
    """Compute IM metric variance for all datasets/axes/models."""
    all_results = []

    for dataset in datasets:
        config = load_predictor_config(dataset)
        if not config:
            continue

        ensemble_models = config["ensemble_models"]
        axes = load_axes(dataset)

        for axis in axes:
            print(f"Computing IM metric variance for {dataset}/{axis}...")
            variance_df = compute_im_metric_variance(
                dataset, axis, ensemble_models,
                n_samples=n_samples,
                budget_sizes=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            )
            if variance_df is not None and len(variance_df) > 0:
                all_results.append(variance_df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def plot_im_variance_vs_delta(variance_df, improvement_df, output_dir):
    """Plot IM metric variance vs estimation error delta."""
    os.makedirs(output_dir, exist_ok=True)

    # Merge variance data with improvement data
    # Need to average variance across budgets for each dataset/axis/metric
    avg_variance = (
        variance_df.groupby(["dataset", "axis", "metric"])
        .agg(im_metric_variance=("im_metric_variance", "mean"))
        .reset_index()
    )

    # Merge with improvement data (averaged across models)
    avg_improvement = (
        improvement_df.groupby(["dataset", "axis", "metric"])
        .agg(improvement=("improvement", "mean"))
        .reset_index()
    )

    merged = avg_variance.merge(avg_improvement, on=["dataset", "axis", "metric"])

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4), sharey=True)

    all_results = {}
    for col, metric in enumerate(METRICS):
        ax = axes[col]
        mdf = merged[merged["metric"] == metric].copy()

        x = mdf["im_metric_variance"].values
        y = mdf["improvement"].values

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            ax.set_title(f"{metric.upper()}\n(insufficient data)", fontsize=9)
            continue

        x_f, y_f, mdf_f = x[mask], y[mask], mdf[mask]

        # Plot by dataset
        for dataset in DATASETS:
            dsub = mdf_f[mdf_f["dataset"] == dataset]
            if dsub.empty:
                continue
            ax.scatter(dsub["im_metric_variance"], dsub["improvement"],
                      color=DATASET_COLORS[dataset], label=dataset,
                      s=50, alpha=0.7, zorder=3)

        # Regression
        slope, intercept, r_value, p_value, _ = stats.linregress(x_f, y_f)
        r2 = r_value ** 2
        all_results[metric] = (r2, r_value, p_value)

        x_line = np.linspace(x_f.min(), x_f.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6, linewidth=1.2)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

        ax.set_title(f"{metric.upper()}\nR²={r2:.3f}, r={r_value:.3f}, p={p_value:.3f}", fontsize=9)
        ax.set_xlabel(f"IM {metric.upper()} Variance", fontsize=8)
        ax.grid(alpha=0.25)

        if col == 0:
            ax.set_ylabel("Estimation error delta\n(metric_match − random)", fontsize=8)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=DATASET_COLORS[d], markersize=7, label=d)
              for d in DATASETS]
    fig.legend(handles=handles, title="Dataset", loc="lower center",
              ncol=len(DATASETS), bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle("IM Metric Variance vs. Estimation Error Delta", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    path = os.path.join(output_dir, "im_variance_vs_delta.jpg")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")

    return all_results


def plot_snr_breakdown(plot_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    snr_variants = [
        ("snr",    "IM-MSB / IM-MSE (Signal-to-Noise)", "snr_by_metric.jpg",    "snr_by_dataset.jpg"),
        ("hm_snr", "HM-MSB / HM-MSE (Signal-to-Noise)", "hm_snr_by_metric.jpg", "hm_snr_by_dataset.jpg"),
    ]
    for pred_col, pred_label, fname_metric, fname_dataset in snr_variants:
        _snr_figure_by_metric(plot_df, pred_col, pred_label, output_dir, fname_metric)
        _snr_figure_by_dataset(plot_df, pred_col, pred_label, output_dir, fname_dataset)


def plot_and_analyze(plot_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics = METRICS
    n_pred = len(PREDICTORS)
    n_met = len(metrics)

    fig, axes = plt.subplots(n_pred, n_met, figsize=(4 * n_met, 3.5 * n_pred))

    all_results = {}
    for row, (pred_col, pred_label) in enumerate(PREDICTORS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            mdf = plot_df[plot_df["metric"] == metric].copy()

            x_vals = mdf[pred_col].values

            r2, r, p = _scatter_panel(ax, x_vals, mdf["improvement"].values, mdf, metric, pred_col, pred_label)

            if r2 is not None:
                all_results[(pred_col, metric)] = (r2, r, p)
                print(f"  [{pred_label} | {metric}] R²={r2:.4f}, r={r:.4f}, p={p:.4f}")

            if col == 0:
                ax.set_ylabel(f"{pred_label}\n(error improvement)", fontsize=8)
            if row == n_pred - 1:
                ax.set_xlabel(pred_label, fontsize=8)

    # Single shared legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7, label=d)
               for d, c in DATASET_COLORS.items()]
    fig.legend(handles=handles, title="Dataset", loc="lower center",
               ncol=len(DATASETS), bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle("Predictors of Estimation Error Improvement over Random (metric_match)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = os.path.join(output_dir, "predictors_vs_improvement.jpg")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir=RESULTS_DIR):
    output_dir = os.path.join(results_dir, "model_variance_analysis")

    print("Step 1: Computing MSB features per (dataset, axis, target_model)...")
    feat_df = compute_msb_features(DATASETS)
    print(f"  {len(feat_df)} triples")

    print("\nStep 2: Loading estimation errors...")
    est_df = load_estimation_errors(results_dir, DATASETS, METRICS)
    print(f"  Loaded {len(est_df)} rows")

    print("\nStep 3: Computing improvement over random...")
    improvement_df = compute_improvement(est_df)

    print("\nStep 4: Merging...")
    plot_df = feat_df.merge(improvement_df, on=["dataset", "axis", "model"])
    print(f"  {len(plot_df)} data points ({plot_df['metric'].nunique()} metrics × {len(feat_df)} triples)")

    print("\nStep 5: Plotting SNR vs. delta broken down by metric and dataset...")
    plot_snr_breakdown(plot_df, output_dir)

    print("\nStep 6: Plotting all predictors × metrics (all points)...")
    all_results = plot_and_analyze(plot_df, output_dir)

    print("\nStep 7: Plotting extremes (top 15 + bottom 15 per metric)...")
    ext_results = plot_extremes(plot_df, output_dir, n=15)

    print("\nStep 8: Computing IM metric variance for all datasets...")
    variance_df = compute_all_im_metric_variances(DATASETS, n_samples=100)
    if len(variance_df) > 0:
        print(f"  Computed variance for {len(variance_df)} (dataset, axis, budget, metric) combinations")

        print("\nStep 9: Plotting IM metric variance vs estimation error delta...")
        variance_results = plot_im_variance_vs_delta(variance_df, improvement_df, output_dir)

        print(f"\nIM Metric Variance vs Delta — R² summary:")
        for metric, (r2, r, p) in sorted(variance_results.items()):
            print(f"  {metric:<20}  R²={r2:.4f}  r={r:+.4f}  p={p:.4f}")

    print(f"\n{'='*60}")
    print("R² summary — all points:")
    for (pred_col, metric), (r2, r, p) in sorted(all_results.items()):
        print(f"  {pred_col:<15} | {metric:<6}  R²={r2:.4f}  r={r:+.4f}  p={p:.4f}")

    print(f"\nR² summary — extremes only:")
    for (pred_col, metric), (r2, r, p) in sorted(ext_results.items()):
        print(f"  {pred_col:<15} | {metric:<6}  R²={r2:.4f}  r={r:+.4f}  p={p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    main(results_dir=args.results_dir)
