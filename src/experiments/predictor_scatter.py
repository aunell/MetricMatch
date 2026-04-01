#!/usr/bin/env python3
"""
Standalone predictor scatter plot analysis.

Loads predictor inputs saved by variance_selection_analysis.py from one or
more dataset run directories, computes predictor records, and generates
scatter plots with all datasets combined (points coloured by dataset).

Usage:
    python src/experiments/predictor_scatter.py \
        --results-dir results/03_13 \
        --datasets medval summeval hanna mslr \
        --output-dir results/03_13/predictor_scatter

The --results-dir must be the same path that was passed as --plots-dir to
variance_selection_analysis.py so that the expected
  results-dir/<dataset>/dataframes/<dataset>/predictor_inputs/
  results-dir/<dataset>/dataframes/<dataset>/icc_by_axis_*.csv
directory structure is found.

--output-dir defaults to results-dir/predictor_scatter.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.plotting import load_results_dataframes, load_predictor_inputs, plot_ms_budget_scatter
from src.utils.predictor_analysis import compute_predictor_records, PREDICTOR_COMPARISON_METHODS
from src.experiments.meta_correlation_analysis import run as run_meta_correlation_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_pairwise_im_stats(axis_df, target_model, ensemble_models):
    """
    For each other model in ensemble_models, compute MSB/MSE/ICC on the
    (target_model, other_model) pair, then return the average across pairs.

    Returns dict with keys im_msb, im_mse, im_icc.
    """
    from src.utils.reliability_metrics import compute_ms_components

    other_models = [m for m in ensemble_models if m != target_model]
    msb_vals, mse_vals, icc_vals = [], [], []

    for other in other_models:
        pair_df = axis_df[axis_df["model_name"].isin([target_model, other])]
        counts = pair_df.groupby("text_id")["model_name"].nunique()
        shared_ids = counts[counts == 2].index
        pair_df = pair_df[pair_df["text_id"].isin(shared_ids)]
        if len(pair_df) == 0:
            continue
        ms = compute_ms_components(pair_df, validate=False)
        if ms is None:
            continue
        if np.isfinite(ms.msb):
            msb_vals.append(ms.msb)
        if np.isfinite(ms.mse):
            mse_vals.append(ms.mse)
        if hasattr(ms, "icc") and np.isfinite(ms.icc):
            icc_vals.append(ms.icc)

    return {
        "im_msb": float(np.mean(msb_vals)) if msb_vals else np.nan,
        "im_mse": float(np.mean(mse_vals)) if mse_vals else np.nan,
        "im_icc": float(np.mean(icc_vals)) if icc_vals else np.nan,
    }


def _build_im_df(axis_df, model, ensemble_models, comparison_mode):
    """Reconstruct im_full_df from saved axis_df using the run's comparison mode."""
    eff_ensemble = [e for e in ensemble_models if e != model]
    im_model_set = [model] + eff_ensemble

    if comparison_mode in ("average_pairwise", "pairwise_average"):
        other_models = [x for x in im_model_set if x != model]
        im_subset = axis_df[axis_df["model_name"].isin(im_model_set)]
        model_rows = (
            im_subset[im_subset["model_name"] == model][["text_id", "evaluation_score"]]
            .copy()
        )
        model_rows["model_name"] = model
        avg_rows = (
            im_subset[im_subset["model_name"].isin(other_models)]
            .groupby("text_id")["evaluation_score"]
            .mean()
            .reset_index()
        )
        avg_rows["model_name"] = "avg_other"
        return pd.concat(
            [model_rows[["text_id", "model_name", "evaluation_score"]],
             avg_rows[["text_id", "model_name", "evaluation_score"]]],
            ignore_index=True,
        )
    else:
        im_subset = axis_df[axis_df["model_name"].isin(im_model_set)]
        counts = im_subset.groupby("text_id")["model_name"].nunique()
        shared = counts[counts == len(im_model_set)].index
        return im_subset[im_subset["text_id"].isin(shared)]


def _plot_predictor_scatter(df, output_dir, title_suffix="", filename_suffix=""):
    """Generate scatter plots with points coloured by dataset and shaped by model."""
    method_labels = {
        "variance_matched_combined":        "VM (combined)",
        "variance_matched_combined_imc":    "VM+IMC",
        "variance_matched_combined_tc":     "VM+TC",
        "variance_matched_combined_tc_imc": "VM+TC+IMC",
        "variance_matched_msb":             "VM (MSB)",
        "variance_matched_msb_imc":         "VM MSB+IMC",
        "variance_matched_msb_tc":          "VM MSB+TC",
        "variance_matched_msb_tc_imc":      "VM MSB+TC+IMC",
        "proxy_oracle":                     "Proxy Oracle",
        "proxy_oracle_imc":                 "Proxy Oracle+IMC",
        "oracle_msb_mse":                   "Oracle (MSB+MSE)",
        "oracle_msb":                       "Oracle (MSB)",
        "oracle_mse":                       "Oracle (MSE)",
        "oracle_icc":                       "Oracle (ICC)",
        "oracle_alpha":                     "Oracle (Alpha)",
        "oracle_mean_squared_error":        "Oracle (MSE)",
        "random_imc":                       "Random+IMC",
        "metric_matched_icc":               "Metric (ICC)",
        "metric_matched_alpha":             "Metric (Alpha)",
        "metric_matched_mse":               "Metric (MSE)",
        "variance_matched_weighted_.2":     "VM Weighted (.2/.8)",
        "variance_matched_weighted_.5":     "VM Weighted (.5/.5)",
        "variance_matched_weighted_.5_imc": "VM Weighted (.5/.5)+IMC",
        "variance_matched_weighted_.7":     "VM Weighted (.7/.3)",
        "variance_matched_weighted_.7_imc": "VM Weighted (.7/.3)+IMC",
        "variance_matched_weighted_.9":     "VM Weighted (.9/.1)",
    }
    predictors = [
        ("mean_shift",       "Mean Shift\n(im_msb+im_mse) − (hm_msb+hm_mse)"),
        ("correlation",      "Correlation\nr(im_ms, hm_ms) across bootstrap samples"),
        ("correlation_msb",  "MSB Correlation\nr(im_msb, hm_msb) across bootstrap samples"),
    ]

    # Discover which metrics have gap columns in the DataFrame
    metric_info = [
        ("icc",   "ICC",   "ICC error gap vs random\n(method − random; negative = better)"),
        ("alpha", "Alpha", "Alpha error gap vs random\n(method − random; negative = better)"),
        ("mse",   "MSE",   "MSE error gap vs random\n(method − random; negative = better)"),
    ]
    available_metrics = [
        (key, label, xlabel) for key, label, xlabel in metric_info
        if any(col.startswith(f"{key}_gap_") for col in df.columns)
    ]

    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    color_map = {d: colors[i % len(colors)] for i, d in enumerate(datasets)}
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    print(f"\n[Predictor scatter plots] {len(df)} total (axis, model) records "
          f"across {len(datasets)} dataset(s), {len(available_metrics)} metric(s)")

    import matplotlib.lines as mlines

    for metric_key, metric_label, x_label in available_metrics:
        comparison_methods = sorted(
            col[len(f"{metric_key}_gap_"):] for col in df.columns
            if col.startswith(f"{metric_key}_gap_")
        )
        n_methods = len(comparison_methods)
        n_cols = min(n_methods, 4)
        n_rows = int(np.ceil(n_methods / n_cols))

        for predictor_col, predictor_label in predictors:
            filename = os.path.join(
                output_dir, f"predictor_scatter_{metric_key}_{predictor_col}{filename_suffix}.jpg"
            )
            if os.path.exists(filename):
                print(f"  Skipping (exists): {filename}")
                continue

            fig, axes_grid = plt.subplots(n_rows, n_cols,
                                          figsize=(5 * n_cols, 4 * n_rows),
                                          sharey=True, squeeze=False)
            ax_flat = [axes_grid[r][c] for r in range(n_rows) for c in range(n_cols)]
            for ax in ax_flat[n_methods:]:
                ax.set_visible(False)

            for ax, method in zip(ax_flat, comparison_methods):
                gap_col = f"{metric_key}_gap_{method}"
                plot_df = df[[predictor_col, gap_col, "dataset", "model"]].dropna()

                if len(plot_df) == 0:
                    ax.set_title(method_labels.get(method, method))
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

                for ds in datasets:
                    for mdl in models:
                        sub = plot_df[(plot_df["dataset"] == ds) & (plot_df["model"] == mdl)]
                        if len(sub) == 0:
                            continue
                        ax.scatter(sub[gap_col], sub[predictor_col],
                                   alpha=0.7, edgecolors="k", linewidths=0.4, s=60,
                                   color=color_map[ds], marker=marker_map[mdl])

                ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_title(method_labels.get(method, method), fontsize=10)

                if len(plot_df) >= 3:
                    xs_pd = plot_df[gap_col].values
                    ys_pd = plot_df[predictor_col].values
                    r = np.corrcoef(xs_pd, ys_pd)[0, 1]
                    slope = np.polyfit(xs_pd, ys_pd, 1)[0]
                    ax.text(0.05, 0.95, f"r={r:.2f}, slope={slope:.2f}", transform=ax.transAxes,
                            fontsize=8, va="top")

                ax.grid(alpha=0.3)

            # Build two legend groups: color → dataset, marker → model
            dataset_handles = [
                mlines.Line2D([], [], color=color_map[d], marker="o", linestyle="None",
                              markersize=6, label=d)
                for d in datasets
            ]
            model_handles = [
                mlines.Line2D([], [], color="gray", marker=marker_map[m], linestyle="None",
                              markersize=6, label=m)
                for m in models
            ]
            for r in range(n_rows):
                axes_grid[r][0].set_ylabel(predictor_label, fontsize=9)
            ax_flat[n_methods - 1].legend(
                handles=dataset_handles + [mlines.Line2D([], [], linestyle="None")] + model_handles,
                labels=[d for d in datasets] + [""] + [m for m in models],
                title="Dataset / Model", fontsize=7, loc="upper left",
                bbox_to_anchor=(1.02, 1), borderaxespad=0,
            )
            fig.suptitle(
                f"Predictor vs {metric_label} error gap ({predictor_col}) — all datasets{title_suffix}",
                fontsize=11
            )
            fig.tight_layout()

            filename = os.path.join(
                output_dir, f"predictor_scatter_{metric_key}_{predictor_col}{filename_suffix}.jpg"
            )
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {filename}")


def _plot_predictor_scatter_by_budget(df, output_dir):
    """
    For each budget found in the per-budget gap columns, create the same predictor
    scatter plots as _plot_predictor_scatter but using only data for that budget.

    Per-budget gap columns are expected to be named {metric}_gap_{method}_b{budget}.
    Saves into output_dir/by_budget/.
    """
    import re
    budget_pattern = re.compile(r"_b(\d+)$")

    budgets = set()
    for col in df.columns:
        m = budget_pattern.search(col)
        if m:
            budgets.add(int(m.group(1)))
    budgets = sorted(budgets)

    if not budgets:
        print("No per-budget gap columns found; skipping by-budget plots.")
        return

    budget_dir = os.path.join(output_dir, "by_budget")
    os.makedirs(budget_dir, exist_ok=True)

    meta_cols = [c for c in ["axis", "model", "dataset", "mean_shift", "correlation", "correlation_msb"]
                 if c in df.columns]

    meta_col_set = set(meta_cols)

    for budget in budgets:
        # Build a df with only this budget's gap columns, renamed to generic form.
        # Skip budget columns whose stripped name collides with a meta column
        # (e.g. correlation_msb_b5 → correlation_msb conflicts with the predictor col).
        rename_map = {}
        for col in df.columns:
            m = budget_pattern.search(col)
            if m and int(m.group(1)) == budget:
                base = col[:col.rfind(f"_b{budget}")]
                if base not in meta_col_set:
                    rename_map[col] = base

        keep_cols = meta_cols + list(rename_map.keys())
        budget_df = df[keep_cols].rename(columns=rename_map)

        print(f"\n[Predictor scatter — budget={budget}]")
        _plot_predictor_scatter(
            budget_df, budget_dir,
            title_suffix=f" (budget={budget})",
            filename_suffix=f"_b{budget}",
        )


# ---------------------------------------------------------------------------
# Budget correlation overview (MSB / MSE / ICC / Alpha across budgets)
# ---------------------------------------------------------------------------

def _collect_budget_correlations(axis_jobs, im_df_builder, dataset,
                                  budgets=(10, 20, 30, 40, 50),
                                  n_samples=200, seed=123):
    """
    For each (axis, model) pair, collect individual bootstrap sample pairs of
    IM and HM values for each budget and metric (msb, mse, icc, alpha).

    Returns a list of dicts — one per bootstrap sample:
        {dataset, axis, model, budget, metric, im_val, hm_val}
    """
    from src.utils.predictor_analysis import compute_ms_budget_samples
    from src.utils.reliability_metrics import compute_krippendorff_alpha

    budgets = list(budgets)
    records = []

    for axis, axis_df in axis_jobs:
        models = [m for m in axis_df["model_name"].unique() if m != "original"]

        for model in models:
            im_full_df = im_df_builder(axis_df, model)
            hm_full_df = axis_df[axis_df["model_name"].isin([model, "original"])]

            # MSB / MSE / ICC via existing sampling utility
            budget_samples = compute_ms_budget_samples(
                im_full_df, hm_full_df,
                budgets=budgets, n_samples=n_samples, seed=seed,
            )

            for budget in budgets:
                bs = budget_samples.get(budget, {})
                if not isinstance(bs, dict):
                    continue
                for metric in ("msb", "mse", "icc"):
                    pairs = bs.get(metric, [])
                    for im_val, hm_val in pairs:
                        if np.isfinite(im_val) and np.isfinite(hm_val):
                            records.append({
                                "dataset": dataset, "axis": axis, "model": model,
                                "budget": budget, "metric": metric,
                                "im_val": float(im_val), "hm_val": float(hm_val),
                            })

            # Alpha — bootstrap manually (not in compute_ms_budget_samples)
            rng = np.random.RandomState(seed + 1)
            im_ids = set(im_full_df["text_id"].unique())
            hm_ids = set(hm_full_df["text_id"].unique())
            shared_ids = np.array(sorted(im_ids & hm_ids))

            for budget in budgets:
                if len(shared_ids) < budget:
                    continue
                for _ in range(n_samples):
                    sample_ids = rng.choice(shared_ids, size=budget, replace=False)
                    im_sub = im_full_df[im_full_df["text_id"].isin(sample_ids)]
                    hm_sub = hm_full_df[hm_full_df["text_id"].isin(sample_ids)]
                    try:
                        im_a = compute_krippendorff_alpha(im_sub)
                        hm_a = compute_krippendorff_alpha(hm_sub)
                        if isinstance(im_a, dict) or isinstance(hm_a, dict):
                            continue
                        if np.isfinite(im_a) and np.isfinite(hm_a):
                            records.append({
                                "dataset": dataset, "axis": axis, "model": model,
                                "budget": budget, "metric": "alpha",
                                "im_val": float(im_a), "hm_val": float(hm_a),
                            })
                    except Exception:
                        continue

            print(f"    Budget corr collected: axis={axis}, model={model}")

    return records


def _plot_budget_correlation_overview(all_budget_records, output_dir):
    """
    Four-subplot figure (MSB, MSE, ICC, Alpha), each showing a scatter of
    all individual bootstrap sample (IM, HM) pairs across all datasets,
    axes, models, and budgets.

    Each point = one bootstrap sample's IM value vs HM value.
    Points are coloured by dataset.  Overall Pearson r across all points
    is reported in each subplot title.
    """
    import matplotlib.lines as mlines

    if not all_budget_records:
        print("No budget correlation records; skipping overview plot.")
        return

    df = pd.DataFrame(all_budget_records)

    if "im_val" not in df.columns or "hm_val" not in df.columns:
        print("Budget correlation records are missing im_val/hm_val columns "
              "(cached from an older run). Delete budget_correlation_records.csv "
              "and re-run to regenerate.")
        return

    metric_labels = {
        "msb":   "MSB",
        "mse":   "MSE",
        "icc":   "ICC",
        "alpha": "Alpha (Krippendorff)",
    }
    metric_axis_labels = {
        "msb":   ("Model-Model MSB (IM)", "Model-Human MSB (HM)"),
        "mse":   ("Model-Model MSE (IM)", "Model-Human MSE (HM)"),
        "icc":   ("Model-Model ICC (IM)", "Model-Human ICC (HM)"),
        "alpha": ("Model-Model Alpha (IM)", "Model-Human Alpha (HM)"),
    }
    metrics = [m for m in ("msb", "mse", "icc", "alpha") if m in df["metric"].unique()]
    if not metrics:
        print("No metrics found in budget correlation records; skipping overview plot.")
        return

    datasets = sorted(df["dataset"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {d: colors[i % len(colors)] for i, d in enumerate(datasets)}

    n_metrics = len(metrics)
    n_cols = min(n_metrics, 2)
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes_grid = plt.subplots(n_rows, n_cols,
                                   figsize=(6 * n_cols, 5 * n_rows),
                                   squeeze=False)
    ax_flat = [axes_grid[r][c] for r in range(n_rows) for c in range(n_cols)]
    for ax in ax_flat[n_metrics:]:
        ax.set_visible(False)

    for ax, metric in zip(ax_flat, metrics):
        sub = df[df["metric"] == metric].dropna(subset=["im_val", "hm_val"])
        if sub.empty:
            ax.set_title(metric_labels.get(metric, metric))
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        for ds in datasets:
            pts = sub[sub["dataset"] == ds]
            if pts.empty:
                continue
            ax.scatter(pts["im_val"], pts["hm_val"],
                       color=color_map[ds], alpha=0.3,
                       edgecolors="none", s=10,
                       label=ds, zorder=2)

        xs = sub["im_val"].values
        ys = sub["hm_val"].values
        r = float(np.corrcoef(xs, ys)[0, 1])

        # Regression line
        slope, intercept = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                color="black", linewidth=1.5, alpha=0.8, zorder=3)

        # y = x reference
        lim_min = min(xs.min(), ys.min())
        lim_max = max(xs.max(), ys.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="red", linestyle="--", linewidth=1, alpha=0.5)

        x_label, y_label = metric_axis_labels.get(metric, ("IM", "HM"))
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(
            f"{metric_labels.get(metric, metric)}   r = {r:.3f}  (n={len(sub):,})",
            fontsize=11
        )
        ax.grid(alpha=0.3)

    # Legend (dataset colours only)
    dataset_handles = [
        mlines.Line2D([], [], color=color_map[d], marker="o", linestyle="None",
                      markersize=7, label=d)
        for d in datasets
    ]
    fig.legend(dataset_handles, datasets,
               title="Dataset", fontsize=9,
               loc="lower center",
               ncol=len(datasets),
               bbox_to_anchor=(0.5, -0.02),
               borderaxespad=0)

    fig.suptitle(
        "Model-Model vs Model-Human — all bootstrap samples across datasets, axes, models, and budgets",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    filename = os.path.join(output_dir, "budget_correlation_overview.jpg")
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True,
                        help="Base results directory (same as --plots-dir used in rollouts)")
    parser.add_argument("--datasets", nargs="+",
                        default=["medval", "summeval", "hanna", "mslr"],
                        help="Datasets to include")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save scatter plots "
                             "(default: results-dir/predictor_scatter)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Bootstrap samples for correlation predictor (default: 500)")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Items per bootstrap sample (default: 10)")
    parser.add_argument("--human-agreement", default=None,
                        help="Path to human_agreement.csv with columns: "
                             "dataset, axis, icc, krippendorff_alpha, mse")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "predictor_scatter")
    os.makedirs(output_dir, exist_ok=True)

    records_path = os.path.join(output_dir, "predictor_records.csv")
    budget_corr_path = os.path.join(output_dir, "budget_correlation_records.csv")
    ms_scatter_dir = os.path.join(output_dir, "ms_budget_scatter")

    # Load cached predictor records if available
    if os.path.exists(records_path):
        df_cached = pd.read_csv(records_path)
        cached_pred_datasets = set(df_cached["dataset"].unique())
        all_records = df_cached.to_dict("records")
        print(f"Loaded {len(all_records)} predictor records from cache "
              f"(datasets: {sorted(cached_pred_datasets)})")
    else:
        cached_pred_datasets = set()
        all_records = []

    # Load cached budget correlation records if available
    if os.path.exists(budget_corr_path):
        df_bcorr_cached = pd.read_csv(budget_corr_path)
        if "im_val" in df_bcorr_cached.columns and "hm_val" in df_bcorr_cached.columns:
            cached_bcorr_datasets = set(df_bcorr_cached["dataset"].unique())
            all_budget_records = df_bcorr_cached.to_dict("records")
            print(f"Loaded {len(all_budget_records)} budget-correlation records from cache "
                  f"(datasets: {sorted(cached_bcorr_datasets)})")
        else:
            print(f"budget_correlation_records.csv is in old format (mean_im/mean_hm); "
                  f"re-collecting all datasets.")
            cached_bcorr_datasets = set()
            all_budget_records = []
    else:
        cached_bcorr_datasets = set()
        all_budget_records = []

    for dataset in args.datasets:
        need_predictor = dataset not in cached_pred_datasets
        need_budget_corr = dataset not in cached_bcorr_datasets
        existing_ms = (
            [f for f in os.listdir(ms_scatter_dir) if f.startswith(f"{dataset}_")]
            if os.path.exists(ms_scatter_dir) else []
        )
        need_ms_scatter = len(existing_ms) == 0

        if not (need_predictor or need_budget_corr or need_ms_scatter):
            print(f"\nSkipping {dataset}: all outputs already cached.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 60}")

        try:
            axis_jobs, per_model_variance_by_axis, config = load_predictor_inputs(
                args.results_dir, dataset
            )
            (_, _, _,
             icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
             _, reliability_metadata_by_axis) = load_results_dataframes(args.results_dir, dataset)
        except FileNotFoundError as e:
            print(f"  Skipping {dataset}: {e}")
            continue

        comparison_mode = config["comparison_mode"]
        ensemble_models = config["ensemble_models"]

        def im_df_builder(axis_df, model,
                          _em=ensemble_models, _cm=comparison_mode):
            return _build_im_df(axis_df, model, _em, _cm)

        def pairwise_stats_builder(axis_df, model, _em=ensemble_models):
            return _compute_pairwise_im_stats(axis_df, model, _em)

        if need_predictor:
            records = compute_predictor_records(
                axis_jobs, per_model_variance_by_axis, icc_results_by_axis,
                im_df_builder,
                alpha_results_by_axis=alpha_results_by_axis,
                mse_results_by_axis=mse_results_by_axis,
                n_samples=args.n_samples, sample_size=args.sample_size,
                pairwise_stats_builder=pairwise_stats_builder,
            )
            for r in records:
                r["dataset"] = dataset
            all_records.extend(records)
            print(f"  Collected {len(records)} (axis, model) records for {dataset}")
        else:
            print(f"  Predictor records cached for {dataset}, skipping.")

        if need_ms_scatter:
            print(f"\n[MS budget scatter] {dataset}")
            plot_ms_budget_scatter(
                axis_jobs, im_df_builder, dataset, output_dir,
                budgets=(10, 20, 30, 40, 50), n_samples=args.n_samples,
            )
        else:
            print(f"\n[MS budget scatter] {dataset}: {len(existing_ms)} files cached, skipping.")

        if need_budget_corr:
            print(f"\n[Budget correlation collection] {dataset}")
            budget_records = _collect_budget_correlations(
                axis_jobs, im_df_builder, dataset,
                budgets=(10, 20, 30, 40, 50), n_samples=args.n_samples,
            )
            all_budget_records.extend(budget_records)
            print(f"  Collected {len(budget_records)} budget-correlation records for {dataset}")
        else:
            print(f"\n[Budget correlation collection] {dataset}: cached, skipping.")

    if not all_records:
        print("\nNo predictor records found across any dataset. Exiting.")
        return

    df = pd.DataFrame(all_records)
    df.to_csv(records_path, index=False)
    print(f"\nSaved predictor_records.csv ({len(df)} rows) → {records_path}")

    if all_budget_records:
        pd.DataFrame(all_budget_records).to_csv(budget_corr_path, index=False)
        print(f"Saved budget_correlation_records.csv ({len(all_budget_records)} rows) "
              f"→ {budget_corr_path}")

    meta_summary_path = os.path.join(output_dir, "predictor_meta_summary.csv")
    if not os.path.exists(meta_summary_path):
        run_meta_correlation_analysis(df, output_dir,
                                      human_agreement_path=args.human_agreement)
    else:
        print(f"\nMeta correlation analysis outputs found, skipping.")

    _plot_predictor_scatter(df, output_dir)
    _plot_predictor_scatter_by_budget(df, output_dir)

    print(f"\n[Budget correlation overview]")
    _plot_budget_correlation_overview(all_budget_records, output_dir)

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
