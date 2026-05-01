"""
Estimation error plotting across methods and metrics.

Loads pre-computed results from combined metric-matched CSVs and generates one
estimation error plot per metric (icc, spearman, kendalltau, alpha, mean_sq_error),
with one line per selected method, averaged across all datasets/models/axes with 95% CI.

Usage:
    python estimation_error_plotting.py [data_dir] [--output-dir DIR]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = "/Users/alyssaunell/code/SmartSample_local/data/metric_matched_subsets"

# Methods to include in plots
SELECTED_METHODS = [
    "random",
    "random_imc",
    "stratified",
    "metric_match_alpha",
    "metric_match_icc",
    "metric_match_spearman",
    "metric_match_kendalltau",
    "metric_match_mean_sq_error",
]

METRIC_YLABELS = {
    "icc": "Absolute ICC Error",
    "alpha": "Absolute Alpha Error",
    "spearman": "Absolute Spearman Error",
    "kendalltau": "Absolute Kendall Tau Error",
    "mean_sq_error": "Absolute MSE Error",
}

METRIC_DISPLAY_NAMES = {
    "icc": "ICC",
    "alpha": "Alpha",
    "spearman": "Spearman",
    "kendalltau": "Kendall Tau",
    "mean_sq_error": "MSE",
}

# Canonical display labels
METHOD_DISPLAY = {
    "random": "random",
    "random_imc": "random_imc",
    "stratified": "stratified",
    "metric_match_alpha": "metric_matched",
    "metric_match_icc": "metric_matched",
    "metric_match_spearman": "metric_matched",
    "metric_match_kendalltau": "metric_matched",
    "metric_match_mean_sq_error": "metric_matched",
}

METHOD_COLORS = {
    "random": "#1f77b4",
    "random_imc": "#aec7e8",
    "stratified": "#ffbb78",
    "metric_matched": "#2ca02c",
    "variance_matched_msb": "#17becf",
    "variance_matched_weighted_.9": "#9467bd",
}


def load_combined_csv(data_dir, dataset_name):
    """Load the combined CSV for a dataset."""
    path = os.path.join(data_dir, f"{dataset_name}_combined_results2.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Split metric_match into separate methods based on match_metric
    # This allows us to plot msb+msre matching separately from single-metric matching
    if 'metric_match' in df['method'].values:
        # Create a new method column that combines method and match_metric
        df['plot_method'] = df.apply(
            lambda row: f"metric_match_{row['match_metric']}" if row['method'] == 'metric_match' and pd.notna(row['match_metric']) and row['match_metric'] != ''
            else row['method'],
            axis=1
        )
    else:
        df['plot_method'] = df['method']

    return df


def compute_bootstrap_cis(df, n_bootstrap=1000, seed=42):
    """Return DataFrame with mean and 95% bootstrap CI half-width per method/budget."""
    rng = np.random.default_rng(seed)
    records = []
    for method in df["plot_method"].unique():
        for budget in sorted(df["budget"].unique()):
            errors = df[(df["plot_method"] == method) & (df["budget"] == budget)]["est_error"].values
            # Clip errors between 0 and 2
            errors = np.clip(errors, 0, 2)
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


def plot_metric(all_df, metric, methods, output_dir, datasets_used, dataset_filter=None):
    """Plot estimation error for a specific metric."""
    # Filter for the specific metric
    df = all_df[all_df["est_metric"] == metric].copy()

    # Filter for selected methods (using plot_method which may include match_metric info)
    df = df[df["plot_method"].isin(methods)].copy()

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter].copy()

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

    metric_display = METRIC_DISPLAY_NAMES.get(metric, metric.upper())

    if dataset_filter:
        datasets_str = dataset_filter
        title_suffix = f"Dataset: {datasets_str} | Averaged over all models & axes with 95% CI"
    else:
        datasets_str = ", ".join(datasets_used)
        title_suffix = f"Datasets: {datasets_str} | Averaged over all models & axes with 95% CI"

    ax.set_title(
        f"{metric_display} Estimation Error — Selected Methods\n{title_suffix}",
        fontsize=11,
    )
    ax.legend(title="Method", fontsize=10, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if dataset_filter:
        out_path = os.path.join(output_dir, f"{metric}_estimation_error_selected_methods_{dataset_filter}.jpg")
    else:
        out_path = os.path.join(output_dir, f"{metric}_estimation_error_selected_methods.jpg")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot estimation errors from combined CSV files.")
    parser.add_argument("data_dir", nargs="?", default=DEFAULT_DATA_DIR,
                        help="Directory containing combined CSV files (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: data_dir/plots)")
    parser.add_argument("--datasets", nargs="+", default=["hanna", "medval", "mslr", "summeval"],
                        help="Datasets to process (default: all)")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or os.path.join(data_dir, "plots2")
    os.makedirs(output_dir, exist_ok=True)

    # Load all datasets
    all_frames = []
    datasets_loaded = []

    for dataset_name in args.datasets:
        print(f"Loading {dataset_name}...")
        df = load_combined_csv(data_dir, dataset_name)
        if not df.empty:
            df["dataset"] = dataset_name
            all_frames.append(df)
            datasets_loaded.append(dataset_name)
            print(f"  Loaded {len(df)} rows")
        else:
            print(f"  No data found for {dataset_name}")

    if not all_frames:
        print("No data loaded!")
        sys.exit(1)

    # Combine all data
    all_df = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(all_df)}")
    print(f"Unique methods: {sorted(all_df['method'].unique())}")
    print(f"Unique metrics: {sorted(all_df['est_metric'].unique())}")

    # Process each metric
    for metric in ["icc", "alpha", "spearman", "kendalltau", "mean_sq_error"]:
        print(f"\n{'='*60}")
        print(f"Processing: {metric}")
        print(f"{'='*60}")

        # Filter to rows for this metric
        metric_df = all_df[all_df["est_metric"] == metric].copy()

        if metric_df.empty:
            print(f"  No data found for metric={metric}")
            continue

        print(f"  Found {len(metric_df)} rows for {metric}")
        print(f"  Methods available (raw): {sorted(metric_df['method'].unique())}")
        print(f"  Methods available (plot_method): {sorted(metric_df['plot_method'].unique())}")

        # Determine which methods to plot using plot_method
        available_methods = set(metric_df["plot_method"].unique())
        methods_to_plot = [m for m in SELECTED_METHODS if m in available_methods]

        if not methods_to_plot:
            print(f"  No selected methods found in data")
            continue

        print(f"  Plotting methods: {methods_to_plot}")

        # Plot averaged over all datasets
        plot_metric(metric_df, metric, methods_to_plot, output_dir, datasets_loaded)

        # For each dataset, create individual plots
        for dataset_name in datasets_loaded:
            dataset_metric_df = metric_df[metric_df["dataset"] == dataset_name]
            if not dataset_metric_df.empty:
                print(f"  Creating {dataset_name}-only plot for {metric}")
                plot_metric(metric_df, metric, methods_to_plot, output_dir,
                           datasets_loaded, dataset_filter=dataset_name)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
