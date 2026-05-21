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

BEST= "/Users/alyssaunell/code/SmartSample_local/results/05_03_VM_alyssa_40_pairwise_average"
DEFAULT_RESULTS_DIR = "/Users/alyssaunell/code/SmartSample_local/results/05_03_VM_alyssa_40_pairwise_average" #"/Users/alyssaunell/code/SmartSample_local/results/05_03_VM_small_20"
OUTPUT_PATH = f"{DEFAULT_RESULTS_DIR}/estimation_error_plots_0521"
SPLIT_ON = "alyssa"  # Used to identify where to inject dataset names in the path template
# Base methods always included (resolved per-metric below for metric_matched)
BASE_METHODS = [
    "random",
    "random_imc",
    "stratified",
    # "variance_matched_msb",
    # "variance_matched_weighted_.9",
]

# Per-metric variant of the metric_matched method
METRIC_MATCHED = {
    "icc": "metric_matched_icc",
    "alpha": "metric_matched_alpha",
    "rho": "metric_matched_rho",
    "tau": "metric_matched_tau",
    "mse": "metric_matched_mse",
}

METRIC_YLABELS = {
    "icc": "ICC Estimation Error",
    "alpha": "Alpha Estimation Error",
    "rho": "Rho Estimation Error",
    "tau": "Tau Estimation Error",
    "mse": "MSE Estimation Error",
}

# Canonical display labels (metric_matched_* all display as "metric_matched")
METHOD_DISPLAY = {
    "random": "Random",
    "random_imc": "Random_bc",
    "stratified": "Stratified",
    "metric_matched_icc": "Metric_matched",
    "metric_matched_alpha": "Metric_matched",
    "metric_matched_rho": "Metric_matched",
    "metric_matched_tau": "Metric_matched",
    "metric_matched_mse": "Metric_matched",
    "variance_matched_msb": "Variance_matched_msb",
    "variance_matched_weighted_.9": "Variance_matched",
}

TITLE =  {
    "icc": "ICC",
    "alpha": "Krippendorff's Alpha",
    "rho": "Spearman's Rho",
    "tau": "Kendall's Tau",
    "mse": "Mean Squared Error",
}

# METHOD_COLORS = {
#     "random": "#1f77b4",
#     "random_imc": "#aec7e8",
#     "stratified": "#ffbb78",
#     "metric_matched": "#2ca02c",
#     "variance_matched_msb": "#17becf",
#     "variance_matched_weighted_.9": "#9467bd",
# }


def find_datasets(results_dir):
    """Return list of (dataset_name, dataframes_path) using template path."""
    datasets = []
    dataset_names = ["medval", "mslr", "summeval", "hanna"]

    for name in dataset_names:
        # Inject dataset name into the path
        dataset_root = results_dir.replace(f"_{SPLIT_ON}", f"_{name}_{SPLIT_ON}")

        candidate = os.path.join(dataset_root, name, "dataframes")

        if os.path.isdir(candidate):
            if any(os.path.exists(os.path.join(candidate, f"{m}_results.csv"))
                   for m in ["icc", "alpha", "rho", "tau", "mse"]):
                datasets.append((name, candidate))
            else:
                print(f"  Found dir but no metric CSVs: {candidate}")
        else:
            print(f"  Missing dataset dir: {candidate}")

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


def compute_relative_improvement(df, metric_matched_method, n_bootstrap=1000, seed=42):
    """
    Calculate the average relative improvement of metric_matched over random with 95% CI.

    For each budget point:
        - Calculate average estimation error for random
        - Calculate average estimation error for metric_matched
        - Compute relative improvement: (random_error - metric_error) / random_error

    Returns a dict with 'mean', 'ci_lower', and 'ci_upper' for the average relative improvement,
    or None if data is missing.
    """
    if "random" not in df["method"].values or metric_matched_method not in df["method"].values:
        return None

    budgets = sorted(df["budget"].unique())

    # Collect all individual error pairs for bootstrap resampling
    random_errors_by_budget = []
    metric_errors_by_budget = []

    for budget in budgets:
        random_errors = df[(df["method"] == "random") & (df["budget"] == budget)]["estimation_error"].values
        metric_errors = df[(df["method"] == metric_matched_method) & (df["budget"] == budget)]["estimation_error"].values

        # Clip errors between 0 and 2 (same as in compute_bootstrap_cis)
        random_errors = np.clip(random_errors, 0, 2)
        metric_errors = np.clip(metric_errors, 0, 2)

        if len(random_errors) == 0 or len(metric_errors) == 0:
            continue

        random_errors_by_budget.append(random_errors)
        metric_errors_by_budget.append(metric_errors)

    if len(random_errors_by_budget) == 0:
        return None

    # Compute observed mean relative improvement
    relative_improvements = []
    for random_errors, metric_errors in zip(random_errors_by_budget, metric_errors_by_budget):
        random_mean = random_errors.mean()
        metric_mean = metric_errors.mean()
        if random_mean > 0:
            rel_improvement = (random_mean - metric_mean) / random_mean
            relative_improvements.append(rel_improvement)

    if len(relative_improvements) == 0:
        return None

    observed_mean = np.mean(relative_improvements)

    # Bootstrap confidence intervals
    rng = np.random.default_rng(seed)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        boot_relative_improvements = []
        for random_errors, metric_errors in zip(random_errors_by_budget, metric_errors_by_budget):
            # Resample with replacement for each budget
            boot_random = rng.choice(random_errors, size=len(random_errors), replace=True)
            boot_metric = rng.choice(metric_errors, size=len(metric_errors), replace=True)

            boot_random_mean = boot_random.mean()
            boot_metric_mean = boot_metric.mean()

            if boot_random_mean > 0:
                boot_rel_improvement = (boot_random_mean - boot_metric_mean) / boot_random_mean
                boot_relative_improvements.append(boot_rel_improvement)

        if len(boot_relative_improvements) > 0:
            bootstrap_means.append(np.mean(boot_relative_improvements))

    if len(bootstrap_means) == 0:
        return {"mean": observed_mean, "ci_lower": observed_mean, "ci_upper": observed_mean}

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return {"mean": observed_mean, "ci_lower": ci_lower, "ci_upper": ci_upper}


def plot_metric(all_df, metric, methods, output_dir, datasets_used, dataset_filter=None):
    df = all_df[all_df["method"].isin(methods)].copy()
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
        # color = METHOD_COLORS.get(display)
        ax.errorbar(
            mdata["budget"], mdata["mean"], yerr=mdata["ci_hw"],
            marker="o", linewidth=2.5, capsize=5, capthick=2,
            label=display, alpha=0.85,
        )

    ax.set_xlabel("Human Annotation Budget", fontsize=16)
    ax.set_ylabel(METRIC_YLABELS.get(metric, f"Absolute {metric.upper()} Error"), fontsize=16)
    if dataset_filter:
        datasets_str = dataset_filter
        title_suffix = f"Dataset: {datasets_str} | Averaged over all models & axes with 95% CI"
    else:
        # datasets_used[-1] = "hanna" if "hannaaa" in datasets_used[-1] else datasets_used[-1]
        datasets_str = ", ".join(datasets_used)
        title_suffix = f"Datasets: {datasets_str} | Averaged over all models & axes with 95% CI"
    ax.set_title(
        f"{TITLE.get(metric)} Estimation Error",
        fontsize=18,
    )
    ax.legend(title="Method", fontsize=16, title_fontsize=16, loc="upper right")
    ax.tick_params(axis='both', which='major', labelsize=14)
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
    parser = argparse.ArgumentParser(description="Plot estimation errors from precomputed results.")
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Root results directory (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: results_dir/estimation_error_plots)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = OUTPUT_PATH
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    datasets = find_datasets(results_dir)
    if not datasets:
        print(f"No datasets with dataframes found in {results_dir}")
        sys.exit(1)
    print(f"Found datasets: {[d[0] for d in datasets]}")

    # Store relative improvements for summary
    relative_improvements = {}

    for metric in ["icc", "alpha", "rho", "tau", "mse"]:
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

        # Compute relative improvement for metric_matched vs random
        if mm:
            rel_improvement_result = compute_relative_improvement(all_df, mm)
            if rel_improvement_result is not None:
                relative_improvements[metric] = rel_improvement_result
                mean = rel_improvement_result["mean"]
                ci_lower = rel_improvement_result["ci_lower"]
                ci_upper = rel_improvement_result["ci_upper"]
                print(f"  Average relative improvement over random: {mean:.4f} ({mean*100:.2f}%) "
                      f"[95% CI: {ci_lower:.4f} to {ci_upper:.4f}]")
            else:
                print(f"  Could not compute relative improvement (missing data)")

        # Plot averaged over all datasets
        plot_metric(all_df, metric, methods, output_dir, [d[0] for d in datasets])

        # For alpha metric, also create a plot for just the hanna dataset
        # if metric == "alpha" and "hanna" in all_df["dataset"].values:
        #     print(f"  Creating additional hanna-only plot for alpha")
        #     plot_metric(all_df, metric, methods, output_dir, [d[0] for d in datasets], dataset_filter="hanna")

    print(f"\nAll plots saved to: {output_dir}")

    # Print summary of relative improvements
    if relative_improvements:
        print("\n" + "="*80)
        print("SUMMARY: Average Relative Improvement over Random with 95% CI")
        print("="*80)
        for metric, result in relative_improvements.items():
            mean = result["mean"]
            ci_lower = result["ci_lower"]
            ci_upper = result["ci_upper"]
            print(f"{metric.upper():8s}: {mean:7.4f} ({mean*100:6.2f}%) "
                  f"[95% CI: {ci_lower:.4f} to {ci_upper:.4f}]")
        print("="*80)

    return relative_improvements


if __name__ == "__main__":
    main()
