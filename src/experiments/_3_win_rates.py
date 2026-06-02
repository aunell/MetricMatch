import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_RESULTS_DIR = "results/current_results"
OUTPUT_PATH = "results/current_results/win_rates"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "rho", "tau", "mse"]
BASELINE = "random"
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
THRESHOLDS = [.6, .7, .8]
SPLIT_ON = #TODO  # Part of the path to split on when injecting dataset names

# Per-metric variant of the metric_matched method
METRIC_MATCHED = {
    "icc": "metric_matched_icc",
    "alpha": "metric_matched_alpha",
    "rho": "metric_matched_rho",
    "tau": "metric_matched_tau",
    "mse": "metric_matched_mse",
}

TARGET_METHOD_STRATEGY = "metric_matched"  # Use per-metric methods

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
    dataset_names = DATASETS
    for name in dataset_names:
        # Inject dataset name into the path
        dataset_root = results_dir.replace(f"_{SPLIT_ON}", f"_{name}_{SPLIT_ON}")

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
# True metric values extraction
# ---------------------------------------------------------------------------

def extract_true_metric_values(results_dir=DEFAULT_RESULTS_DIR):
    """
    Extract true metric values for each (dataset, axis) pair.
    Returns DataFrame with dataset,axis as rows and metrics as columns.
    """
    datasets = find_datasets(results_dir)
    if not datasets:
        print(f"No datasets with dataframes found in {results_dir}")
        return pd.DataFrame()

    print("Extracting true metric values...")
    
    # Dictionary to store true values: {(dataset, axis): {metric: value}}
    true_values = {}
    
    for dataset_name, df_dir in datasets:
        for metric in METRICS:
            csv_path = os.path.join(df_dir, f"{metric}_results.csv")
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            true_col = f"true_{metric}"
            if true_col not in df.columns:
                if metric == "mse":
                    true_col = "true_msre"
                if true_col not in df.columns:
                    continue
            
            # Get unique (axis, true_value) pairs
            # True values should be the same for all methods/budgets/models for a given axis
            axis_true_values = df[["axis", true_col]].drop_duplicates()
            
            for _, row in axis_true_values.iterrows():
                axis = row["axis"]
                true_val = row[true_col]
                
                key = (dataset_name, axis)
                if key not in true_values:
                    true_values[key] = {}
                true_values[key][metric] = true_val
    
    # Convert to DataFrame
    rows = []
    for (dataset, axis), metrics_dict in true_values.items():
        row_data = {"dataset": dataset, "axis": axis}
        row_data.update(metrics_dict)
        rows.append(row_data)
    
    result_df = pd.DataFrame(rows)
    
    # Set multi-index and sort
    if not result_df.empty:
        result_df = result_df.set_index(["dataset", "axis"])
        
        # Reorder columns to match METRICS order
        available_metrics = [m for m in METRICS if m in result_df.columns]
        result_df = result_df[available_metrics]
        
        result_df = result_df.sort_index()
    
    return result_df


# ---------------------------------------------------------------------------
# Dataset-axis level win rates
# ---------------------------------------------------------------------------

def compute_dataset_axis_win_rates(merged_df, analysis_type="estimation"):
    """
    Compute win rates at the dataset,axis level for each metric.
    
    Args:
        merged_df: DataFrame with 'dataset', 'axis', 'metric', and 'win' columns
        analysis_type: "estimation" or "threshold"
    
    Returns:
        DataFrame with (dataset, axis) as rows and metrics as columns
    """
    print(f"\nComputing dataset-axis level win rates for {analysis_type}...")
    
    # Group by dataset, axis, metric and compute win rate
    win_rates = (
        merged_df.groupby(["dataset", "axis", "metric"])["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "win_rate"})
    )
    
    # Pivot to get metrics as columns
    result = win_rates.pivot_table(
        index=["dataset", "axis"], 
        columns="metric", 
        values="win_rate"
    )
    
    # Reorder columns to match METRICS order
    available = [m for m in METRICS if m in result.columns]
    result = result[available]
    
    # Sort by dataset, then axis
    result = result.sort_index()
    
    result.columns.name = None
    
    return result


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
    if exclusions:
        print(f"Applying exclusions: {exclusions}")

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
    
    combined_df = pd.concat(frames, ignore_index=True)

    
    return combined_df


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
    
    # Sort by budget first (before adding average row)
    result = result.sort_index()
    
    # Add average row (average across budgets for each metric)
    average_row = result[available].mean(axis=0)
    result.loc["average"] = average_row
    
    result.index.name = "budget"
    result.columns.name = None
    return result


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

def load_threshold_raw_data(results_dir=DEFAULT_RESULTS_DIR, target_method_strategy=TARGET_METHOD_STRATEGY, exclusions=EXCLUSIONS):
    """Load raw metric results with predicted and true values for threshold analysis."""
    datasets = find_datasets(results_dir)
    if not datasets:
        print(f"No datasets with dataframes found in {results_dir}")
        return pd.DataFrame()

    print(f"Found datasets: {[d[0] for d in datasets]}")
    print(f"Using target method strategy: {target_method_strategy}")
    if exclusions:
        print(f"Applying exclusions: {exclusions}")

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
    
    combined_df = pd.concat(frames, ignore_index=True)
    
    return combined_df


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
    
    # Sort by budget first (before adding average row)
    result = result.sort_index()
    
    # Add average row (average across budgets for each metric)
    average_row = result[available].mean(axis=0)
    result.loc["average"] = average_row
    
    result.index.name = "budget"
    result.columns.name = None
    return result


def compute_threshold_summary(thr_dir, avg_type="macro"):
    """
    Compute average win rates across all available thresholds.
    
    Args:
        thr_dir: Directory containing threshold CSVs
        avg_type: "macro" or "micro"
    
    Returns:
        DataFrame with averaged win rates across thresholds
    """
    # Find all available threshold files
    pattern = f"{avg_type}_win_rates_threshold_T"
    available_files = []
    available_thresholds = []
    
    for filename in os.listdir(thr_dir):
        if filename.startswith(pattern) and filename.endswith(".csv") and "_hanna" not in filename and "_medval" not in filename and "_mslr" not in filename and "_summeval" not in filename:
            available_files.append(os.path.join(thr_dir, filename))
            # Extract threshold value from filename
            threshold_str = filename.replace(pattern, "").replace(".csv", "")
            available_thresholds.append(threshold_str)
    
    if not available_files:
        print(f"  No {avg_type} threshold files found")
        return None
    
    print(f"  Found {avg_type} threshold files for: {available_thresholds}")
    
    # Load all threshold files
    dfs = []
    for filepath in available_files:
        df = pd.read_csv(filepath, index_col=0)
        # Remove the "average" row if it exists
        if "average" in df.index:
            df = df.drop("average")
        dfs.append(df)
    
    # Average across all thresholds
    # Stack all dataframes and compute mean
    combined = pd.concat(dfs)
    summary = combined.groupby(combined.index).mean()
    
    # Now add the average row for the summary (average across budgets)
    available = [m for m in METRICS if m in summary.columns]
    average_row = summary[available].mean(axis=0)
    summary.loc["average"] = average_row
    
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_win_rates(df, title, output_path, metrics_to_plot=None):
    """
    Plot win rates across budgets for each metric.

    Args:
        df: DataFrame with budget as index and metrics as columns
        title: Title for the plot
        output_path: Path to save the plot
        metrics_to_plot: List of metrics to plot (default: alpha, icc, rho, tau)
    """
    # Default to plotting alpha, icc, rho, tau (excluding mse)
    if metrics_to_plot is None:
        metrics_to_plot = ["alpha", "icc", "rho", "tau"]

    # Filter to only available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    if not metrics_to_plot:
        print(f"  No metrics available for plotting in {title}")
        return

    # Remove 'average' row if present for plotting
    plot_df = df.copy()
    if "average" in plot_df.index:
        plot_df = plot_df.drop("average")

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot each metric as a line
    for metric in metrics_to_plot:
        plt.plot(plot_df.index, plot_df[metric], marker='o', label=metric, linewidth=2)

    # Add horizontal line at 0.5 (tie with random)
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Tie with random')

    # Formatting
    plt.xlabel('Budget', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


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

    # -- Extract and save true metric values --
    print("\n=== EXTRACTING TRUE METRIC VALUES ===")
    true_metrics_df = extract_true_metric_values(results_dir, exclusions)
    if not true_metrics_df.empty:
        save(true_metrics_df, os.path.join(OUTPUT_PATH, "true_im_metric_values.csv"))
    else:
        print("No true metric values found.")

    # -- Estimation --
    print("\n=== ESTIMATION ANALYSIS ===")
    print("Loading estimation data...")
    est_df = load_estimation_data(results_dir, target_method_strategy, exclusions)

    if est_df.empty:
        print("No estimation data found. Skipping estimation analysis.")
    else:
        print("\n--- Macro estimation win rates ---")
        macro_merged = _macro_merge(est_df, target_method_strategy)
        macro_pivot = _estimation_pivot(macro_merged)
        save(macro_pivot, os.path.join(est_dir, "macro_win_rates_estimation.csv"))

        # Plot macro estimation win rates
        plot_win_rates(
            macro_pivot,
            "Macro Estimation Win Rates (All Datasets)",
            os.path.join(est_dir, "macro_win_rates_estimation.png")
        )

        for dataset in DATASETS:
            dataset_pivot = _estimation_pivot(macro_merged[macro_merged["dataset"] == dataset])
            save(
                dataset_pivot,
                os.path.join(est_dir, f"macro_win_rates_estimation_{dataset}.csv"),
            )
            # Plot per-dataset macro estimation win rates
            plot_win_rates(
                dataset_pivot,
                f"Macro Estimation Win Rates ({dataset.capitalize()})",
                os.path.join(est_dir, f"macro_win_rates_estimation_{dataset}.png")
            )

        # Save dataset-axis level win rates for macro estimation
        dataset_axis_macro = compute_dataset_axis_win_rates(macro_merged, "estimation")
        save(dataset_axis_macro, os.path.join(est_dir, "dataset_results_estimation_error_win_rate_macro.csv"))

        print("\n--- Micro estimation win rates ---")
        micro_merged = _micro_merge(est_df, target_method_strategy)
        micro_pivot = _estimation_pivot(micro_merged)
        save(micro_pivot, os.path.join(est_dir, "micro_win_rates_estimation.csv"))

        # Plot micro estimation win rates
        plot_win_rates(
            micro_pivot,
            "Micro Estimation Win Rates (All Datasets)",
            os.path.join(est_dir, "micro_win_rates_estimation.png")
        )

        for dataset in DATASETS:
            dataset_pivot = _estimation_pivot(micro_merged[micro_merged["dataset"] == dataset])
            save(
                dataset_pivot,
                os.path.join(est_dir, f"micro_win_rates_estimation_{dataset}.csv"),
            )
            # Plot per-dataset micro estimation win rates
            plot_win_rates(
                dataset_pivot,
                f"Micro Estimation Win Rates ({dataset.capitalize()})",
                os.path.join(est_dir, f"micro_win_rates_estimation_{dataset}.png")
            )

        # Save dataset-axis level win rates for micro estimation
        dataset_axis_micro = compute_dataset_axis_win_rates(micro_merged, "estimation")
        save(dataset_axis_micro, os.path.join(est_dir, "dataset_results_estimation_error_win_rate_micro.csv"))

        # -- Summary: single win rate collapsed over all budgets, metrics, datasets, models --
        print("\n--- Estimation summary win rates ---")
        micro_total = len(micro_merged)
        micro_wins  = micro_merged["win"].sum()
        macro_total = len(macro_merged)
        macro_wins  = macro_merged["win"].sum()
        summary = pd.DataFrame([
            {"avg":  "micro (all runs)",   "total_comparisons": micro_total, "wins": micro_wins, "win_rate": micro_wins / micro_total if micro_total > 0 else 0},
            {"avg":  "macro (avg runs)",   "total_comparisons": macro_total, "wins": macro_wins, "win_rate": macro_wins / macro_total if macro_total > 0 else 0},
        ]).set_index("avg")
        save(summary, os.path.join(est_dir, "summary_win_rates_estimation.csv"))
# -- Threshold --
    print("\n=== THRESHOLD ANALYSIS ===")
    print("Loading raw data for threshold analysis...")
    thr_raw = load_threshold_raw_data(results_dir, target_method_strategy, exclusions)

    if thr_raw.empty:
        print("No threshold data found. Skipping threshold analysis.")
    else:
        for threshold in THRESHOLDS:
            t_str = f"T{threshold}"
            print(f"\nComputing threshold win rates ({t_str})...")
            macro_merged = _threshold_macro_merge(thr_raw, threshold, target_method_strategy)
            micro_merged = _threshold_micro_merge(thr_raw, threshold, target_method_strategy)

            print(f"\n--- Macro threshold win rates ({t_str}) ---")
            macro_pivot = _threshold_pivot(macro_merged)
            save(macro_pivot, os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}.csv"))

            # Plot macro threshold win rates
            plot_win_rates(
                macro_pivot,
                f"Macro Threshold Win Rates {t_str} (All Datasets)",
                os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}.png")
            )

            for dataset in DATASETS:
                dataset_pivot = _threshold_pivot(macro_merged[macro_merged["dataset"] == dataset])
                save(
                    dataset_pivot,
                    os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}_{dataset}.csv"),
                )
                # Plot per-dataset macro threshold win rates
                plot_win_rates(
                    dataset_pivot,
                    f"Macro Threshold Win Rates {t_str} ({dataset.capitalize()})",
                    os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}_{dataset}.png")
                )

            # Save dataset-axis level win rates for macro threshold
            dataset_axis_macro = compute_dataset_axis_win_rates(macro_merged, f"threshold_{t_str}")
            save(dataset_axis_macro, os.path.join(thr_dir, f"dataset_results_threshold_win_rate_{t_str}_macro.csv"))

            print(f"\n--- Micro threshold win rates ({t_str}) ---")
            micro_pivot = _threshold_pivot(micro_merged)
            save(micro_pivot, os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}.csv"))

            # Plot micro threshold win rates
            plot_win_rates(
                micro_pivot,
                f"Micro Threshold Win Rates {t_str} (All Datasets)",
                os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}.png")
            )

            for dataset in DATASETS:
                dataset_pivot = _threshold_pivot(micro_merged[micro_merged["dataset"] == dataset])
                save(
                    dataset_pivot,
                    os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}_{dataset}.csv"),
                )
                # Plot per-dataset micro threshold win rates
                plot_win_rates(
                    dataset_pivot,
                    f"Micro Threshold Win Rates {t_str} ({dataset.capitalize()})",
                    os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}_{dataset}.png")
                )

            # Save dataset-axis level win rates for micro threshold
            dataset_axis_micro = compute_dataset_axis_win_rates(micro_merged, f"threshold_{t_str}")
            save(dataset_axis_micro, os.path.join(thr_dir, f"dataset_results_threshold_win_rate_{t_str}_micro.csv"))

        # -- Threshold Summary: Average across all thresholds --
        print("\n--- Computing threshold summary (average across thresholds) ---")
        
        print("\n--- Macro threshold summary ---")
        macro_summary = compute_threshold_summary(thr_dir, avg_type="macro")
        if macro_summary is not None:
            save(macro_summary, os.path.join(thr_dir, "macro_win_rates_threshold_summary.csv"))
            # Plot macro threshold summary
            plot_win_rates(
                macro_summary,
                "Macro Threshold Win Rates Summary (Avg Across Thresholds)",
                os.path.join(thr_dir, "macro_win_rates_threshold_summary.png")
            )

        print("\n--- Micro threshold summary ---")
        micro_summary = compute_threshold_summary(thr_dir, avg_type="micro")
        if micro_summary is not None:
            save(micro_summary, os.path.join(thr_dir, "micro_win_rates_threshold_summary.csv"))
            # Plot micro threshold summary
            plot_win_rates(
                micro_summary,
                "Micro Threshold Win Rates Summary (Avg Across Thresholds)",
                os.path.join(thr_dir, "micro_win_rates_threshold_summary.png")
            )
        
        # -- Compute dataset-axis level win rates averaged across thresholds --
        print("\n--- Computing dataset-axis threshold summary (average across thresholds) ---")
        
        # Collect all dataset-axis threshold results and average them
        macro_dataset_axis_dfs = []
        micro_dataset_axis_dfs = []
        
        for threshold in THRESHOLDS:
            t_str = f"T{threshold}"
            macro_path = os.path.join(thr_dir, f"dataset_results_threshold_win_rate_{t_str}_macro.csv")
            micro_path = os.path.join(thr_dir, f"dataset_results_threshold_win_rate_{t_str}_micro.csv")
            
            if os.path.exists(macro_path):
                df = pd.read_csv(macro_path, index_col=[0, 1])
                macro_dataset_axis_dfs.append(df)
            
            if os.path.exists(micro_path):
                df = pd.read_csv(micro_path, index_col=[0, 1])
                micro_dataset_axis_dfs.append(df)
        
        if macro_dataset_axis_dfs:
            print("\n--- Macro dataset-axis threshold summary ---")
            combined_macro = pd.concat(macro_dataset_axis_dfs)
            summary_macro = combined_macro.groupby(combined_macro.index).mean()
            save(summary_macro, os.path.join(thr_dir, "dataset_results_threshold_win_rate_summary_macro.csv"))
        
        if micro_dataset_axis_dfs:
            print("\n--- Micro dataset-axis threshold summary ---")
            combined_micro = pd.concat(micro_dataset_axis_dfs)
            summary_micro = combined_micro.groupby(combined_micro.index).mean()
            save(summary_micro, os.path.join(thr_dir, "dataset_results_threshold_win_rate_summary_micro.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Win rates and threshold analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Target method strategies:
  metric_matched              Use per-metric methods (metric_matched_icc, metric_matched_alpha, etc.)
  variance_matched_weighted_.9  Use this fixed method for all metrics
  <any_other_string>          Use that specific method name for all metrics

Exclusions:
  Edit the EXCLUSIONS list at the top of the script to exclude specific (dataset, axis) pairs.
  Example: EXCLUSIONS = [("hannaaa", "model"), ("medval", "example")]
        """
    )
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR,
                        help="Root results directory (default: %(default)s)")
    parser.add_argument("--target-method", default=TARGET_METHOD_STRATEGY,
                        help="Target method strategy (default: %(default)s)")
    parser.add_argument("--exclude", nargs=2, action="append", metavar=("DATASET", "AXIS"),
                        help="Exclude a specific (dataset, axis) pair. Can be used multiple times.")
    args = parser.parse_args()
    
    main(results_dir=args.results_dir, target_method_strategy=args.target_method)