"""
Estimation error plotting across methods and metrics.

Loads pre-computed results from a results directory and generates one
estimation error plot per metric (icc, rho, tau, alpha), with one line
per selected method, averaged across all datasets/models/axes with 95% CI.

Usage:
    python estimation_error_plotting.py [results_dir] [--output-dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = "/share/pi/nigam/users/aunell/SmartSample_local/results/04_16_new_baselines"

# Base methods always included (resolved per-metric below for metric_matched)
BASE_METHODS = [
    "random",
    "random_imc",
    "stratified",
    "variance_matched_msb",
    "variance_matched_weighted_.9",
]

# Per-metric variant of the metric_matched method
METRIC_MATCHED = {
    "icc": "metric_matched_icc",
    "alpha": "metric_matched_alpha",
    "rho": "metric_matched_rho",
    "tau": "metric_matched_tau",
}

METRIC_YLABELS = {
    "icc": "Absolute ICC Error",
    "alpha": "Absolute Alpha Error",
    "rho": "Absolute Rho Error",
    "tau": "Absolute Tau Error",
}

# Canonical display labels (metric_matched_* all display as "metric_matched")
METHOD_DISPLAY = {
    "random": "random",
    "random_imc": "random_imc",
    "stratified": "stratified",
    "metric_matched_icc": "metric_matched",
    "metric_matched_alpha": "metric_matched",
    "metric_matched_rho": "metric_matched",
    "metric_matched_tau": "metric_matched",
    "variance_matched_msb": "variance_matched_msb",
    "variance_matched_weighted_.9": "variance_matched_weighted_.9",
}

METHOD_COLORS = {
    "random": "#1f77b4",
    "random_imc": "#aec7e8",
    "stratified": "#ffbb78",
    "metric_matched": "#2ca02c",
    "variance_matched_msb": "#17becf",
    "variance_matched_weighted_.9": "#9467bd",
}


def find_datasets(results_dir):
    """Return list of (dataset_name, dataframes_path) for each dataset found."""
    datasets = []
    for name in sorted(os.listdir(results_dir)):
        candidate = os.path.join(results_dir, name, name, "dataframes")
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "icc_results.csv")):
            datasets.append((name, candidate))
    return datasets


def load_metric_df(df_dir, metric):
    path = os.path.join(df_dir, f"{metric}_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_bootstrap_cis(df, n_bootstrap=1000, seed=42):
    """Return DataFrame with mean and 95% bootstrap CI half-width per method/budget."""
    rng = np.random.default_rng(seed)
    records = []
    for method in df["method"].unique():
        for budget in sorted(df["budget"].unique()):
            errors = df[(df["method"] == method) & (df["budget"] == budget)]["estimation_error"].values
            if len(errors) == 0:
                continue
            mean = errors.mean()
            if len(errors) > 1:
                boots = [rng.choice(errors, size=len(errors), replace=True).mean()
                         for _ in range(n_bootstrap)]
                ci_hw = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5)) / 2
            else:
                ci_hw = 0.0
            records.append({"method": method, "budget": int(budget), "mean": mean, "ci_hw": ci_hw})
    return pd.DataFrame(records)


def plot_metric(all_df, metric, methods, output_dir, datasets_used):
    df = all_df[all_df["method"].isin(methods)].copy()
    if df.empty:
        print(f"  No data found, skipping {metric}")
        return

    ci_df = compute_bootstrap_cis(df)
    if ci_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        display = METHOD_DISPLAY.get(method, method)
        mdata = ci_df[ci_df["method"] == method].sort_values("budget")
        if mdata.empty:
            continue
        color = METHOD_COLORS.get(display)
        ax.errorbar(
            mdata["budget"], mdata["mean"], yerr=mdata["ci_hw"],
            marker="o", linewidth=2.5, capsize=5, capthick=2,
            label=display, color=color, alpha=0.85,
        )

    ax.set_xlabel("Human Annotation Budget", fontsize=13)
    ax.set_ylabel(METRIC_YLABELS.get(metric, f"Absolute {metric.upper()} Error"), fontsize=13)
    datasets_str = ", ".join(datasets_used)
    ax.set_title(
        f"{metric.upper()} Estimation Error — Selected Methods\n"
        f"Datasets: {datasets_str} | Averaged over all models & axes with 95% CI",
        fontsize=11,
    )
    ax.legend(title="Method", fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{metric}_estimation_error_selected_methods.jpg")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot estimation errors from precomputed results.")
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Root results directory (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: results_dir/estimation_error_plots)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "estimation_error_plots")
    os.makedirs(output_dir, exist_ok=True)

    datasets = find_datasets(results_dir)
    if not datasets:
        print(f"No datasets with dataframes found in {results_dir}")
        sys.exit(1)
    print(f"Found datasets: {[d[0] for d in datasets]}")

    for metric in ["icc", "alpha", "rho", "tau"]:
        print(f"\nProcessing: {metric}")
        frames = []
        for dataset_name, df_dir in datasets:
            df = load_metric_df(df_dir, metric)
            if not df.empty:
                df["dataset"] = dataset_name
                frames.append(df)
                print(f"  Loaded {len(df)} rows from {dataset_name}")

        if not frames:
            print(f"  No CSV data found for metric={metric}")
            continue

        all_df = pd.concat(frames, ignore_index=True)

        methods = list(BASE_METHODS)
        mm = METRIC_MATCHED.get(metric)
        if mm and mm in all_df["method"].values:
            methods.append(mm)
        methods = [m for m in methods if m in all_df["method"].values]
        missing = [m for m in BASE_METHODS + ([mm] if mm else []) if m not in all_df["method"].values]
        if missing:
            print(f"  Methods not found in data (skipped): {missing}")
        print(f"  Plotting methods: {methods}")

        plot_metric(all_df, metric, methods, output_dir, [d[0] for d in datasets])

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
