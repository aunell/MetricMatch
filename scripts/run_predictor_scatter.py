#!/usr/bin/env python3
"""
Standalone predictor scatter plot analysis.

Loads predictor inputs saved by variance_selection_analysis.py from one or
more dataset run directories, computes predictor records, and generates
scatter plots with all datasets combined (points coloured by dataset).

Usage:
    python scripts/run_predictor_scatter.py \
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

# Make the project importable when run from the repo root or the scripts/ dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.plotting import load_results_dataframes, load_predictor_inputs, plot_ms_budget_scatter
from src.utils.predictor_analysis import compute_predictor_records, PREDICTOR_COMPARISON_METHODS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        "oracle":                           "Oracle",
        "oracle_imc":                       "Oracle+IMC",
        "random_imc":                       "Random+IMC",
        "metric_matched_icc":               "Metric (ICC)",
        "metric_matched_alpha":             "Metric (Alpha)",
        "metric_matched_mse":               "Metric (MSE)",
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

    for budget in budgets:
        # Build a df with only this budget's gap columns, renamed to generic form
        rename_map = {}
        for col in df.columns:
            m = budget_pattern.search(col)
            if m and int(m.group(1)) == budget:
                base = col[:col.rfind(f"_b{budget}")]
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
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "predictor_scatter")
    os.makedirs(output_dir, exist_ok=True)

    all_records = []

    for dataset in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 60}")

        try:
            axis_jobs, per_model_variance_by_axis, config = load_predictor_inputs(
                args.results_dir, dataset
            )
            (_, _, _,
             icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
             _, _) = load_results_dataframes(args.results_dir, dataset)
        except FileNotFoundError as e:
            print(f"  Skipping {dataset}: {e}")
            continue

        comparison_mode = config["comparison_mode"]
        ensemble_models = config["ensemble_models"]

        def im_df_builder(axis_df, model,
                          _em=ensemble_models, _cm=comparison_mode):
            return _build_im_df(axis_df, model, _em, _cm)

        records = compute_predictor_records(
            axis_jobs, per_model_variance_by_axis, icc_results_by_axis,
            im_df_builder,
            alpha_results_by_axis=alpha_results_by_axis,
            mse_results_by_axis=mse_results_by_axis,
            n_samples=args.n_samples, sample_size=args.sample_size,
        )
        for r in records:
            r["dataset"] = dataset
        all_records.extend(records)
        print(f"  Collected {len(records)} (axis, model) records for {dataset}")

        print(f"\n[MS budget scatter] {dataset}")
        plot_ms_budget_scatter(
            axis_jobs, im_df_builder, dataset, output_dir,
            budgets=(10, 20, 30, 40, 50), n_samples=args.n_samples,
        )

    if not all_records:
        print("\nNo predictor records found across any dataset. Exiting.")
        return

    df = pd.DataFrame(all_records)
    records_path = os.path.join(output_dir, "predictor_records.csv")
    df.to_csv(records_path, index=False)
    print(f"\nSaved predictor_records.csv ({len(df)} rows) → {records_path}")

    _plot_predictor_scatter(df, output_dir)
    _plot_predictor_scatter_by_budget(df, output_dir)

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
