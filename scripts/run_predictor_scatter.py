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
  results-dir/<dataset>/dataframes/icc_by_axis_*.csv
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

from src.utils.plotting import load_results_dataframes, load_predictor_inputs
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


def _plot_predictor_scatter(df, output_dir):
    """Generate scatter plots with points coloured by dataset and shaped by model."""
    method_labels = {
        "variance_matched_combined":        "VM",
        "variance_matched_combined_imc":    "VM+IMC",
        "variance_matched_combined_tc_imc": "VM+TC+IMC",
    }
    predictors = [
        ("mean_shift",  "Mean Shift\n(im_msb+im_mse) − (hm_msb+hm_mse)"),
        ("correlation", "Correlation\nr(im_ms, hm_ms) across bootstrap samples"),
    ]

    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    color_map = {d: colors[i % len(colors)] for i, d in enumerate(datasets)}
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(models)}

    print(f"\n[Predictor scatter plots] {len(df)} total (axis, model) records "
          f"across {len(datasets)} dataset(s)")

    for predictor_col, predictor_label in predictors:
        n_methods = len(PREDICTOR_COMPARISON_METHODS)
        fig, axes_list = plt.subplots(1, n_methods,
                                      figsize=(5 * n_methods, 4),
                                      sharey=True)
        if n_methods == 1:
            axes_list = [axes_list]

        for ax, method in zip(axes_list, PREDICTOR_COMPARISON_METHODS):
            gap_col = f"icc_gap_{method}"
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
            ax.set_xlabel("ICC error gap vs random\n(method − random; negative = better)",
                          fontsize=9)
            ax.set_title(method_labels.get(method, method), fontsize=10)

            if len(plot_df) >= 3:
                r = np.corrcoef(plot_df[gap_col].values,
                                plot_df[predictor_col].values)[0, 1]
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, va="top")

            ax.grid(alpha=0.3)

        # Build two legend groups: color → dataset, marker → model
        import matplotlib.lines as mlines
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
        axes_list[0].set_ylabel(predictor_label, fontsize=9)
        axes_list[-1].legend(
            handles=dataset_handles + [mlines.Line2D([], [], linestyle="None")] + model_handles,
            labels=[d for d in datasets] + [""] + [m for m in models],
            title="Dataset / Model", fontsize=7, loc="upper left",
            bbox_to_anchor=(1.02, 1), borderaxespad=0,
        )
        fig.suptitle(f"Predictor vs ICC error gap ({predictor_col}) — all datasets",
                     fontsize=11)
        fig.tight_layout()

        filename = os.path.join(output_dir, f"predictor_scatter_{predictor_col}.jpg")
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
            _, _, _, icc_results_by_axis, _, _, _, _ = load_results_dataframes(
                args.results_dir, dataset
            )
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
            n_samples=args.n_samples, sample_size=args.sample_size,
        )
        for r in records:
            r["dataset"] = dataset
        all_records.extend(records)
        print(f"  Collected {len(records)} (axis, model) records for {dataset}")

    if not all_records:
        print("\nNo predictor records found across any dataset. Exiting.")
        return

    df = pd.DataFrame(all_records)
    records_path = os.path.join(output_dir, "predictor_records.csv")
    df.to_csv(records_path, index=False)
    print(f"\nSaved predictor_records.csv ({len(df)} rows) → {records_path}")

    _plot_predictor_scatter(df, output_dir)

    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
