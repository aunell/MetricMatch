import argparse
import os
import pandas as pd
import numpy as np

DEFAULT_RESULTS_DIR = "/Users/alyssaunell/code/SmartSample_local/results/05_02_VM_alyssa_full"
OUTPUT_PATH = "/Users/alyssaunell/code/SmartSample_local/results/05_02_VM_alyssa_full/win_rates_vm"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "rho", "tau", "mse"]
BASELINE = "random"
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
THRESHOLDS = [.7, .8, .9]

# Per-metric variant of the metric_matched method
METRIC_MATCHED = {
    "icc": "metric_matched_icc",
    "alpha": "metric_matched_alpha",
    "rho": "metric_matched_rho",
    "tau": "metric_matched_tau",
    "mse": "metric_matched_mse",
}

# Target method strategies
# "metric_matched": Use per-metric methods (metric_matched_icc, metric_matched_alpha, etc.)
# "variance_matched_weighted_.9": Use fixed method variance_matched_weighted_.9 for all metrics
# Any other string: Use that specific method name for all metrics
TARGET_METHOD_STRATEGY =  "variance_matched_weighted_.9" #metric_matched"  # Default to metric_matched


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_target_method(metric, strategy=TARGET_METHOD_STRATEGY):
    """
    Get the target method name based on the strategy.

    Args:
        metric: The metric name (e.g., "icc", "alpha")
        strategy: The target method strategy
            - "metric_matched": Use per-metric methods from METRIC_MATCHED dict
            - Any other string: Use that method name for all metrics

    Returns:
        Method name string, or None if not found
    """
    if strategy == "metric_matched":
        return METRIC_MATCHED.get(metric)
    else:
        # Use the strategy string as the method name directly
        return strategy


# ---------------------------------------------------------------------------
# Dataset discovery helpers
# ---------------------------------------------------------------------------

def find_datasets(results_dir):
    """Return list of (dataset_name, dataframes_path) using template path."""
    datasets = []
    dataset_names = ["medval", "mslr", "summeval", "hanna"]

    for name in dataset_names:
        # Inject dataset name into the path
        dataset_root = results_dir.replace("_alyssa", f"_{name}_alyssa")

        candidate = os.path.join(dataset_root, name, "dataframes")

        if os.path.isdir(candidate):
            if any(os.path.exists(os.path.join(candidate, f"{m}_results.csv"))
                   for m in METRICS):
                datasets.append((name, candidate))
            else:
                print(f"  Found dir but no metric CSVs: {candidate}")
        else:
            print(f"  Missing dataset dir: {candidate}")

    return datasets


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def load_estimation_data(results_dir=DEFAULT_RESULTS_DIR, target_method_strategy=TARGET_METHOD_STRATEGY):
    """Load {metric}_results.csv for each dataset and metric, return combined df."""
    datasets = find_datasets(results_dir)
    if not datasets:
        print(f"No datasets with dataframes found in {results_dir}")
        return pd.DataFrame()

    print(f"Found datasets: {[d[0] for d in datasets]}")
    print(f"Using target method strategy: {target_method_strategy}")

    frames = []
    for dataset_name, df_dir in datasets:
        for metric in METRICS:
            csv_path = os.path.join(df_dir, f"{metric}_results.csv")
            if not os.path.exists(csv_path):
                print(f"  Skipping {metric} | {dataset_name}: file not found")
                continue
            df = pd.read_csv(csv_path)
            # Get method name based on strategy
            our_method = get_target_method(metric, target_method_strategy)
            if our_method is None:
                print(f"  Skipping {metric} | {dataset_name}: no target method defined")
                continue
            # Filter for our method and baseline
            df = df[df["method"].isin([our_method, BASELINE])][
                ["model", "budget", "method", "estimation_error", "axis"]
            ]
            if len(df) == 0:
                print(f"  Skipping {metric} | {dataset_name}: no data for method {our_method}")
                continue
            df["dataset"] = dataset_name
            df["metric"] = metric
            frames.append(df)
            print(f"  Loaded {len(df)} rows from {dataset_name} - {metric} (using {our_method})")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def assign_run_index(df):
    """Assign positional run index within each (dataset, metric, axis, model, budget, method) group."""
    df = df.sort_values(
        ["dataset", "metric", "axis", "model", "budget", "method"]
    ).copy()
    df["run"] = df.groupby(
        ["dataset", "metric", "axis", "model", "budget", "method"]
    ).cumcount()
    return df


def _estimation_pivot(merged):
    win_rates = (
        merged.groupby(["budget", "metric"])["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "win_rate"})
    )
    result = win_rates.pivot(index="budget", columns="metric", values="win_rate")
    available = [m for m in METRICS if m in result.columns]
    result = result[available]
    result.index.name = "budget"
    result.columns.name = None
    return result.sort_index()


def _macro_merge(df, target_method_strategy=TARGET_METHOD_STRATEGY):
    """Average over runs then compare our method vs random."""
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "method", "metric"])[
            "estimation_error"
        ]
        .mean()
        .reset_index()
    )
    # Filter for our target method (varies by strategy)
    if target_method_strategy == "metric_matched":
        # Filter for any method starting with "metric_matched_"
        ours = avg[avg["method"].str.startswith("metric_matched_")].rename(columns={"estimation_error": "our_err"})
    else:
        # Filter for the specific method name
        ours = avg[avg["method"] == target_method_strategy].rename(columns={"estimation_error": "our_err"})

    base = avg[avg["method"] == BASELINE].rename(columns={"estimation_error": "random_err"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


def _micro_merge(df, target_method_strategy=TARGET_METHOD_STRATEGY):
    """Pair runs by position then compare our method vs random."""
    df = assign_run_index(df)
    # Filter for our target method (varies by strategy)
    if target_method_strategy == "metric_matched":
        # Filter for any method starting with "metric_matched_"
        ours = df[df["method"].str.startswith("metric_matched_")].rename(columns={"estimation_error": "our_err"})
    else:
        # Filter for the specific method name
        ours = df[df["method"] == target_method_strategy].rename(columns={"estimation_error": "our_err"})

    base = df[df["method"] == BASELINE].rename(columns={"estimation_error": "random_err"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def load_threshold_raw_data(results_dir=DEFAULT_RESULTS_DIR, target_method_strategy=TARGET_METHOD_STRATEGY):
    """Load raw metric results with predicted and true values for threshold analysis."""
    datasets = find_datasets(results_dir)
    if not datasets:
        print(f"No datasets with dataframes found in {results_dir}")
        return pd.DataFrame()

    print(f"Found datasets: {[d[0] for d in datasets]}")
    print(f"Using target method strategy: {target_method_strategy}")

    frames = []
    for dataset_name, df_dir in datasets:
        for metric in METRICS:
            csv_path = os.path.join(df_dir, f"{metric}_results.csv")
            if not os.path.exists(csv_path):
                print(f"  Skipping {metric} | {dataset_name}: file not found")
                continue
            df = pd.read_csv(csv_path)
            pred_col, true_col = f"predicted_{metric}", f"true_{metric}"
            if pred_col not in df.columns or true_col not in df.columns:
                if metric=="mse":
                    pred_col, true_col = f"predicted_msre", f"true_msre"
            if pred_col not in df.columns or true_col not in df.columns:
                print(f"  Skipping {metric} | {dataset_name}: missing columns {pred_col}/{true_col}")
                continue
            # Get method name based on strategy
            our_method = get_target_method(metric, target_method_strategy)
            if our_method is None:
                print(f"  Skipping {metric} | {dataset_name}: no target method defined")
                continue
            # Filter for our method and baseline
            df = df[df["method"].isin([our_method, BASELINE])][
                ["model", "budget", "method", pred_col, true_col, "axis"]
            ].rename(columns={pred_col: "predicted", true_col: "true_val"})
            if len(df) == 0:
                print(f"  Skipping {metric} | {dataset_name}: no data for method {our_method}")
                continue
            df["dataset"] = dataset_name
            df["metric"] = metric
            frames.append(df)
            print(f"  Loaded {len(df)} rows from {dataset_name} - {metric} (using {our_method})")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _classify(series, threshold):
    """Return boolean Series: True if value >= threshold."""
    return series >= threshold


def _threshold_micro_merge(df, threshold, target_method_strategy=TARGET_METHOD_STRATEGY):
    """
    Per-run classification correctness, paired by position.
    correct = (predicted_class == true_class) for each individual run.
    Ties (both methods same) are discarded.
    """
    df = df.copy()
    df["correct"] = (_classify(df["predicted"], threshold) == _classify(df["true_val"], threshold)).astype(float)

    df = df.sort_values(["dataset", "metric", "axis", "model", "budget", "method"])
    df["run"] = df.groupby(["dataset", "metric", "axis", "model", "budget", "method"]).cumcount()

    # Filter for our target method (varies by strategy)
    if target_method_strategy == "metric_matched":
        # Filter for any method starting with "metric_matched_"
        ours = df[df["method"].str.startswith("metric_matched_")].rename(columns={"correct": "our_correct"})
    else:
        # Filter for the specific method name
        ours = df[df["method"] == target_method_strategy].rename(columns={"correct": "our_correct"})

    base = df[df["method"] == BASELINE].rename(columns={"correct": "random_correct"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])

    merged = merged[merged["our_correct"] != merged["random_correct"]].copy()
    merged["win"] = merged["our_correct"] > merged["random_correct"]
    return merged


def _threshold_macro_merge(df, threshold, target_method_strategy=TARGET_METHOD_STRATEGY):
    """
    For each run: classify predicted and true as above/below threshold,
    correct = (predicted_class == true_class).
    Macro: average correct over 100 runs per (dataset, axis, model, budget, method),
    then compare our method vs random. Ties discarded.
    """
    df = df.copy()
    df["correct"] = (_classify(df["predicted"], threshold) == _classify(df["true_val"], threshold)).astype(float)

    # Assign run index within each group
    df = df.sort_values(["dataset", "metric", "axis", "model", "budget", "method"])
    df["run"] = df.groupby(["dataset", "metric", "axis", "model", "budget", "method"]).cumcount()

    # Average over runs per (dataset, axis, model, budget, method, metric)
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "method", "metric"])["correct"]
        .mean()
        .reset_index()
    )

    # Filter for our target method (varies by strategy)
    if target_method_strategy == "metric_matched":
        # Filter for any method starting with "metric_matched_"
        ours = avg[avg["method"].str.startswith("metric_matched_")].rename(columns={"correct": "our_acc"})
    else:
        # Filter for the specific method name
        ours = avg[avg["method"] == target_method_strategy].rename(columns={"correct": "our_acc"})

    base = avg[avg["method"] == BASELINE].rename(columns={"correct": "random_acc"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])

    # Discard ties
    merged = merged[merged["our_acc"] != merged["random_acc"]].copy()
    merged["win"] = merged["our_acc"] > merged["random_acc"]
    merged["n"] = len(merged)
    # breakpoint()
    return merged


def _threshold_pivot(merged):
    win_rates = (
        merged.groupby(["budget", "metric"])["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "win_rate"})
    )
    result = win_rates.pivot(index="budget", columns="metric", values="win_rate")
    available = [m for m in METRICS if m in result.columns]
    result = result[available]
    result.index.name = "budget"
    result.columns.name = None
    return result.sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save(df, path):
    df.to_csv(path)
    print(f"Saved: {path}")
    print(df.round(3).to_string())
    print()


def main(results_dir=DEFAULT_RESULTS_DIR, target_method_strategy=TARGET_METHOD_STRATEGY):
    est_dir = os.path.join(OUTPUT_PATH, "estimation")
    thr_dir = os.path.join(OUTPUT_PATH, "threshold")
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(thr_dir, exist_ok=True)

    # -- Estimation --
    print("Loading estimation data...")
    est_df = load_estimation_data(results_dir, target_method_strategy)

    if est_df.empty:
        print("No estimation data found. Exiting.")
        return

    print("\n--- Macro estimation win rates ---")
    macro_merged = _macro_merge(est_df, target_method_strategy)
    save(_estimation_pivot(macro_merged), os.path.join(est_dir, "macro_win_rates_estimation.csv"))
    for dataset in DATASETS:
        save(
            _estimation_pivot(macro_merged[macro_merged["dataset"] == dataset]),
            os.path.join(est_dir, f"macro_win_rates_estimation_{dataset}.csv"),
        )

    print("\n--- Micro estimation win rates ---")
    micro_merged = _micro_merge(est_df, target_method_strategy)
    save(_estimation_pivot(micro_merged), os.path.join(est_dir, "micro_win_rates_estimation.csv"))
    for dataset in DATASETS:
        save(
            _estimation_pivot(micro_merged[micro_merged["dataset"] == dataset]),
            os.path.join(est_dir, f"micro_win_rates_estimation_{dataset}.csv"),
        )

    # -- Summary: single win rate collapsed over all budgets, metrics, datasets, models --
    print("\n--- Estimation summary win rates ---")
    micro_total = len(micro_merged)
    micro_wins  = micro_merged["win"].sum()
    macro_total = len(macro_merged)
    macro_wins  = macro_merged["win"].sum()
    summary = pd.DataFrame([
        {"avg":  "micro (all runs)",   "total_comparisons": micro_total, "wins": micro_wins, "win_rate": micro_wins / micro_total},
        {"avg":  "macro (avg runs)",   "total_comparisons": macro_total, "wins": macro_wins, "win_rate": macro_wins / macro_total},
    ]).set_index("avg")
    save(summary, os.path.join(est_dir, "summary_win_rates_estimation.csv"))

    # -- Threshold --
    print("Loading raw data for threshold analysis...")
    thr_raw = load_threshold_raw_data(results_dir, target_method_strategy)

    if thr_raw.empty:
        print("No threshold data found. Skipping threshold analysis.")
        return

    for threshold in THRESHOLDS:
        t_str = f"T{threshold}"
        print(f"\nComputing threshold win rates ({t_str})...")
        macro_merged = _threshold_macro_merge(thr_raw, threshold, target_method_strategy)
        breakpoint()
        micro_merged = _threshold_micro_merge(thr_raw, threshold, target_method_strategy)

        print(f"\n--- Macro threshold win rates ({t_str}) ---")
        save(_threshold_pivot(macro_merged), os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}.csv"))
        for dataset in DATASETS:
            save(
                _threshold_pivot(macro_merged[macro_merged["dataset"] == dataset]),
                os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}_{dataset}.csv"),
            )

        print(f"\n--- Micro threshold win rates ({t_str}) ---")
        save(_threshold_pivot(micro_merged), os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}.csv"))
        for dataset in DATASETS:
            save(
                _threshold_pivot(micro_merged[micro_merged["dataset"] == dataset]),
                os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}_{dataset}.csv"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Win rates and threshold analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target method strategies:
  metric_matched              Use per-metric methods (metric_matched_icc, metric_matched_alpha, etc.)
  variance_matched_weighted_.9  Use this fixed method for all metrics
  <any_other_string>          Use that specific method name for all metrics
        """
    )
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Root results directory (default: %(default)s)")
    parser.add_argument("--target-method", default=TARGET_METHOD_STRATEGY,
                        help="Target method strategy (default: %(default)s)")
    args = parser.parse_args()
    main(results_dir=args.results_dir, target_method_strategy=args.target_method)
