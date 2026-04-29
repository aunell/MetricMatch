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
from src.utils.reliability_metrics import compute_ms_components

RESULTS_DIR = "/share/pi/nigam/users/aunell/SmartSample_local/results/04_22_add_metric_match"
DATASETS = ["hanna", "mslr", "summeval", "medval"]
METRICS = ["icc", "alpha", "rho", "tau"]
OUR_METHOD = "variance_matched_msb"
BASELINE = "random"

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
    ("true_metric",  "True Reliability Value"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_axes(dataset, results_dir):
    path = os.path.join(
        results_dir, dataset, dataset, "dataframes", "predictor_inputs", "axes.json"
    )
    with open(path) as f:
        return json.load(f)


def load_predictor_config(dataset, results_dir):
    path = os.path.join(
        results_dir, dataset, dataset, "dataframes", "predictor_inputs", "predictor_config.json"
    )
    with open(path) as f:
        return json.load(f)


def load_per_model_variance(dataset, results_dir):
    path = os.path.join(
        results_dir, dataset, dataset, "dataframes", "predictor_inputs", "per_model_variance.json"
    )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Predictor computation
# ---------------------------------------------------------------------------

def compute_msb_features(results_dir, datasets):
    """
    For each (dataset, axis, target_model): compute
      - 4 pairwise MSBs via compute_ms_components → mean_msb, msb_variance
      - im_msb, hm_msb, im_mse from per_model_variance.json → im_hm_ratio, snr
    """
    records = []
    for dataset in datasets:
        config = load_predictor_config(dataset, results_dir)
        ensemble_models = config["ensemble_models"]
        per_model_var = load_per_model_variance(dataset, results_dir)

        axes = load_axes(dataset, results_dir)
        if not axes:
            inputs_dir = os.path.join(
                results_dir, dataset, dataset, "dataframes", "predictor_inputs"
            )
            axes = [
                f.replace("axis_data_", "").replace(".csv", "")
                for f in os.listdir(inputs_dir)
                if f.startswith("axis_data_") and f.endswith(".csv")
            ]

        for axis in axes:
            csv_path = os.path.join(
                results_dir, dataset, dataset, "dataframes", "predictor_inputs",
                f"axis_data_{axis}.csv",
            )
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
    frames = []
    for dataset in datasets:
        for metric in metrics:
            csv_path = os.path.join(
                results_dir, dataset, dataset, "dataframes", f"{metric}_results.csv"
            )
            if not os.path.exists(csv_path):
                print(f"  Missing: {dataset}/{metric}_results.csv — skipping")
                continue
            df = pd.read_csv(csv_path)
            available = [m for m in [OUR_METHOD, BASELINE] if m in df["method"].values]
            cols = ["model", "budget", "method", "estimation_error", "axis", f"true_{metric}"]
            df = df[df["method"].isin(available)][cols].copy()
            df["dataset"] = dataset
            df["metric"] = metric
            df = df.rename(columns={f"true_{metric}": "true_metric"})
            frames.append(df)
    if not frames:
        raise RuntimeError("No estimation error data found.")
    return pd.concat(frames, ignore_index=True)


def compute_improvement(est_df):
    """
    Improvement = mean(our_err) − mean(random_err), averaged over budgets.
    Also extract true_metric (constant per dataset/axis/model/metric).
    """
    avg = (
        est_df.groupby(["dataset", "axis", "model", "metric", "method"])
        .agg(estimation_error=("estimation_error", "mean"),
             true_metric=("true_metric", "first"))
        .reset_index()
    )
    ours = avg[avg["method"] == OUR_METHOD].rename(columns={"estimation_error": "our_err"}).drop(columns="method")
    base = avg[avg["method"] == BASELINE].rename(columns={"estimation_error": "random_err"}).drop(columns="method")
    merged = ours.merge(base[["dataset", "axis", "model", "metric", "random_err"]],
                        on=["dataset", "axis", "model", "metric"])
    merged["improvement"] = merged["our_err"] - merged["random_err"]
    return merged


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

            x_vals = sub["true_metric"].values if pred_col == "true_metric" else sub[pred_col].values
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
                    xp = dsub["true_metric"].values if pred_col == "true_metric" else dsub[pred_col].values
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
    fig, axes = plt.subplots(1, 4, figsize=(4.5 * 4, 4), sharey=False)
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
    fig, axes = plt.subplots(1, 4, figsize=(4.5 * 4, 4), sharey=False)
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

            if pred_col == "true_metric":
                x_vals = mdf["true_metric"].values
            else:
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

    fig.suptitle("Predictors of Estimation Error Improvement over Random (variance_matched_msb)", fontsize=12)
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
    feat_df = compute_msb_features(results_dir, DATASETS)
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
