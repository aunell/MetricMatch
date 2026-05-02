"""
Comprehensive Win Rate Analysis for Sampling Strategies.

This script calculates macro and micro win rates comparing various sampling strategies
against random baseline across multiple metrics (ICC, Alpha, MSE, Rho, Tau).

Win Rate Types:
---------------
1. ESTIMATION WIN RATES: Compare estimation error (lower is better)
2. THRESHOLD WIN RATES: Compare classification accuracy at a threshold

Win Rate Definitions:
---------------------
Macro win rate: Average estimation errors/correctness over all trials within each condition
                group (dataset, axis, model, budget), then compare method vs baseline.
                This gives the percentage of condition groups where the method beats random.

Micro win rate: Pair individual trials by position and compare directly.
                This gives the percentage of individual trials where the method beats random.

Threshold Analysis:
-------------------
For a given threshold (e.g., 0.7), classify predicted and true metric values as above/below.
Correctness = (predicted_class == true_class).
Win rate = percentage where method is more correct than baseline.

Usage:
------
# Run with default settings (uses 04_30_*_small_ens directories)
python src/experiments/_8_win_rates_comprehensive.py

# Run quietly (suppress detailed output)
python src/experiments/_8_win_rates_comprehensive.py --quiet

# Specify custom base directory
python src/experiments/_8_win_rates_comprehensive.py --base-dir /path/to/results

# Specify custom output directory
python src/experiments/_8_win_rates_comprehensive.py --output-dir /path/to/output

Output Files:
-------------
Estimation Win Rates:
  - estimation_win_rates_overall.csv: Win rates aggregated across all datasets
  - estimation_win_rates_by_dataset.csv: Win rates broken down by dataset
  - estimation_win_rates_summary.csv: Formatted summary table with percentages

Threshold Win Rates:
  - threshold_win_rates_overall.csv: Win rates aggregated across all datasets
  - threshold_win_rates_by_dataset.csv: Win rates broken down by dataset
  - threshold_win_rates_summary.csv: Formatted summary table with percentages

Example Results:
----------------
Estimation Win Rates:
Method                       Metric  Micro Win Rate  Macro Win Rate
variance_matched_weighted_.9 ALPHA   51.96%          82.46%
variance_matched_msb         ALPHA   51.58%          80.53%
metric_matched_alpha         ALPHA   51.52%          79.47%

Threshold Win Rates (T=0.7):
Method                       Metric  Micro Win Rate  Macro Win Rate
variance_matched_weighted_.9 ALPHA   52.34%          75.21%
variance_matched_msb         ALPHA   51.89%          73.45%
"""

import argparse
import os
import pandas as pd
import numpy as np


# Default configuration
DEFAULT_RESULTS_DIRS = {
    'hanna': 'results/04_30_hanna_small_all/hanna/dataframes',
    'mslr': 'results/04_30_mslr_small_all/mslr/dataframes',
    'summeval': 'results/04_30_summeval_small_all/summeval/dataframes',
    'medval': 'results/04_30_medval_small_all/medval/dataframes',
}

DEFAULT_RESULTS_DIRS = {
    'hanna': 'results/04_33_hanna/hanna/dataframes',
    'mslr': 'results/04_33_mslr/mslr/dataframes',
    'summeval': 'results/04_33_summeval/summeval/dataframes',
    'medval': 'results/04_33_medval/medval/dataframes',
}

METRICS = ['icc', 'alpha', 'mse', 'rho', 'tau']

METHODS_TO_COMPARE = [
    'metric_matched_icc',
    'metric_matched_alpha',
    'metric_matched_rho',
    'metric_matched_tau',
    'metric_matched_mse',
    'variance_matched_msb',
    'variance_matched_weighted_.9'
]

BASELINE = 'random'
THRESHOLDS = [0.7]  # Thresholds for classification analysis


def load_all_results(results_dirs, base_dir=None):
    """
    Load all metric results from specified directories.

    Args:
        results_dirs: Dict mapping dataset name to dataframes directory path
        base_dir: Optional base directory to prepend to paths

    Returns:
        Dict mapping metric name to combined DataFrame
    """
    all_results = {}

    for metric in METRICS:
        all_metric_results = []

        for dataset_name, path in results_dirs.items():
            if base_dir:
                path = os.path.join(base_dir, path)

            try:
                if metric == 'mse':
                    df = pd.read_csv(os.path.join(path, 'mse_results.csv'))
                else:
                    df = pd.read_csv(os.path.join(path, f'{metric}_results.csv'))
                df['dataset'] = dataset_name
                all_metric_results.append(df)
                print(f"  Loaded {dataset_name} {metric}: {len(df)} rows")
            except Exception as e:
                print(f"  Warning: Could not load {dataset_name} {metric}: {e}")

        if all_metric_results:
            all_results[metric] = pd.concat(all_metric_results, ignore_index=True)
            print(f"Combined {metric}: {len(all_results[metric])} total rows")
        else:
            print(f"No data loaded for {metric}")

    return all_results


def calculate_macro_win_rate(df, method, baseline):
    """
    Macro win rate calculation:
    1. Average estimation_error over all trials for each (dataset, axis, model, budget, method)
    2. Compare averaged errors between method and baseline
    3. Calculate win rate from comparisons

    Args:
        df: DataFrame with columns [dataset, axis, model, budget, method, estimation_error]
        method: Name of method to compare
        baseline: Name of baseline method

    Returns:
        DataFrame with win/loss results for each condition group
    """
    # Average over all trials (runs) within each group
    avg_errors = (
        df.groupby(['dataset', 'axis', 'model', 'budget', 'method'])['estimation_error']
        .mean()
        .reset_index()
    )

    # Separate method and baseline
    method_avg = avg_errors[avg_errors['method'] == method].rename(
        columns={'estimation_error': 'method_err'}
    )
    baseline_avg = avg_errors[avg_errors['method'] == baseline].rename(
        columns={'estimation_error': 'baseline_err'}
    )

    # Merge on grouping variables
    merged = method_avg.merge(
        baseline_avg,
        on=['dataset', 'axis', 'model', 'budget'],
        how='inner'
    )

    # Calculate wins (method has lower error than baseline)
    merged['win'] = merged['method_err'] < merged['baseline_err']

    return merged


def calculate_micro_win_rate(df, method, baseline):
    """
    Micro win rate calculation:
    Pair individual trials and compare directly.

    Args:
        df: DataFrame with columns [dataset, axis, model, budget, method, estimation_error]
        method: Name of method to compare
        baseline: Name of baseline method

    Returns:
        DataFrame with win/loss results for each paired trial
    """
    method_df = df[df['method'] == method].copy().reset_index(drop=True)
    baseline_df = df[df['method'] == baseline].copy().reset_index(drop=True)

    # Assign run index within each group
    for temp_df in [method_df, baseline_df]:
        temp_df.sort_values(['dataset', 'axis', 'model', 'budget'], inplace=True)
        temp_df['run'] = temp_df.groupby(['dataset', 'axis', 'model', 'budget']).cumcount()

    # Merge on grouping variables + run
    method_df = method_df.rename(columns={'estimation_error': 'method_err'})
    baseline_df = baseline_df.rename(columns={'estimation_error': 'baseline_err'})

    merged = method_df.merge(
        baseline_df,
        on=['dataset', 'axis', 'model', 'budget', 'run'],
        how='inner'
    )

    # Calculate wins (method has lower error than baseline)
    merged['win'] = merged['method_err'] < merged['baseline_err']

    return merged


def compute_win_rates(all_results, methods_to_compare, baseline, verbose=True):
    """
    Compute macro and micro win rates for all methods and metrics.

    Args:
        all_results: Dict mapping metric name to DataFrame
        methods_to_compare: List of method names to compare
        baseline: Name of baseline method
        verbose: Whether to print detailed results

    Returns:
        DataFrame with all win rate results
    """
    results_table = []

    for metric_name, df in all_results.items():
        if verbose:
            print(f"\n{'='*90}")
            print(f"METRIC: {metric_name.upper()}")
            print(f"{'='*90}")

        # Check what methods are available in this metric
        available_methods = df['method'].unique()

        for method in methods_to_compare:
            if method not in available_methods or baseline not in available_methods:
                continue

            # Calculate macro win rate
            macro_merged = calculate_macro_win_rate(df, method, baseline)
            if len(macro_merged) == 0:
                continue

            macro_win_rate = macro_merged['win'].mean() * 100
            macro_wins = macro_merged['win'].sum()
            macro_total = len(macro_merged)

            # Calculate micro win rate
            micro_merged = calculate_micro_win_rate(df, method, baseline)
            if len(micro_merged) == 0:
                continue

            micro_win_rate = micro_merged['win'].mean() * 100
            micro_wins = micro_merged['win'].sum()
            micro_total = len(micro_merged)

            if verbose:
                print(f"\n{method}:")
                print(f"  Macro win rate: {macro_win_rate:.2f}% ({macro_wins}/{macro_total} condition groups)")
                print(f"  Micro win rate: {micro_win_rate:.2f}% ({micro_wins}/{micro_total} individual trials)")

                # Show by dataset
                print(f"  By dataset (macro):")
                for dataset_name in sorted(macro_merged['dataset'].unique()):
                    dataset_macro = macro_merged[macro_merged['dataset'] == dataset_name]
                    dataset_macro_wr = dataset_macro['win'].mean() * 100
                    print(f"    {dataset_name}: {dataset_macro_wr:.2f}%")

            # Store for summary
            results_table.append({
                'Method': method,
                'Metric': metric_name.upper(),
                'Micro_Win_Rate': micro_win_rate,
                'Macro_Win_Rate': macro_win_rate,
                'Micro_Wins': int(micro_wins),
                'Micro_Total': int(micro_total),
                'Macro_Wins': int(macro_wins),
                'Macro_Total': int(macro_total),
            })

            # Add per-dataset results
            for dataset_name in sorted(macro_merged['dataset'].unique()):
                dataset_macro = macro_merged[macro_merged['dataset'] == dataset_name]
                dataset_micro = micro_merged[micro_merged['dataset'] == dataset_name]

                results_table.append({
                    'Method': method,
                    'Metric': metric_name.upper(),
                    'Dataset': dataset_name,
                    'Micro_Win_Rate': dataset_micro['win'].mean() * 100,
                    'Macro_Win_Rate': dataset_macro['win'].mean() * 100,
                    'Micro_Wins': int(dataset_micro['win'].sum()),
                    'Micro_Total': len(dataset_micro),
                    'Macro_Wins': int(dataset_macro['win'].sum()),
                    'Macro_Total': len(dataset_macro),
                })

    return pd.DataFrame(results_table)


# ---------------------------------------------------------------------------
# Threshold Win Rate Functions
# ---------------------------------------------------------------------------

def _classify(series, threshold):
    """Return boolean Series: True if value >= threshold."""
    return series >= threshold


def calculate_threshold_micro_win_rate(df, method, baseline, threshold, metric_name):
    """
    Per-run classification correctness, paired by position.
    correct = (predicted_class == true_class) for each individual run.
    Ties (both methods same) are discarded.

    Args:
        df: DataFrame with columns [dataset, axis, model, budget, method, predicted_{metric}, true_{metric}]
        method: Name of method to compare
        baseline: Name of baseline method
        threshold: Threshold value for classification
        metric_name: Name of metric (for column names)

    Returns:
        DataFrame with win/loss results for each paired trial
    """
    pred_col = f'predicted_{metric_name}'
    true_col = f'true_{metric_name}'

    # Check if required columns exist
    if pred_col not in df.columns or true_col not in df.columns:
        return pd.DataFrame()

    # Calculate correctness for each row
    df = df.copy()
    df['correct'] = (_classify(df[pred_col], threshold) == _classify(df[true_col], threshold)).astype(float)

    # Filter to method and baseline
    method_df = df[df['method'] == method].copy().reset_index(drop=True)
    baseline_df = df[df['method'] == baseline].copy().reset_index(drop=True)

    # Assign run index within each group
    for temp_df in [method_df, baseline_df]:
        temp_df.sort_values(['dataset', 'axis', 'model', 'budget'], inplace=True)
        temp_df['run'] = temp_df.groupby(['dataset', 'axis', 'model', 'budget']).cumcount()

    # Merge on grouping variables + run
    method_df = method_df.rename(columns={'correct': 'method_correct'})
    baseline_df = baseline_df.rename(columns={'correct': 'baseline_correct'})

    merged = method_df.merge(
        baseline_df,
        on=['dataset', 'axis', 'model', 'budget', 'run'],
        how='inner'
    )

    # Discard ties
    merged = merged[merged['method_correct'] != merged['baseline_correct']].copy()

    # Calculate wins (method is more correct than baseline)
    merged['win'] = merged['method_correct'] > merged['baseline_correct']

    return merged


def calculate_threshold_macro_win_rate(df, method, baseline, threshold, metric_name):
    """
    For each run: classify predicted and true as above/below threshold,
    correct = (predicted_class == true_class).
    Macro: average correct over runs per (dataset, axis, model, budget, method),
    then compare method vs baseline. Ties discarded.

    Args:
        df: DataFrame with columns [dataset, axis, model, budget, method, predicted_{metric}, true_{metric}]
        method: Name of method to compare
        baseline: Name of baseline method
        threshold: Threshold value for classification
        metric_name: Name of metric (for column names)

    Returns:
        DataFrame with win/loss results for each condition group
    """
    pred_col = f'predicted_{metric_name}'
    true_col = f'true_{metric_name}'

    # Check if required columns exist
    if pred_col not in df.columns or true_col not in df.columns:
        return pd.DataFrame()

    # Calculate correctness for each row
    df = df.copy()
    df['correct'] = (_classify(df[pred_col], threshold) == _classify(df[true_col], threshold)).astype(float)

    # Average over all trials (runs) within each group
    avg_correct = (
        df.groupby(['dataset', 'axis', 'model', 'budget', 'method'])['correct']
        .mean()
        .reset_index()
    )

    # Separate method and baseline
    method_avg = avg_correct[avg_correct['method'] == method].rename(
        columns={'correct': 'method_acc'}
    )
    baseline_avg = avg_correct[avg_correct['method'] == baseline].rename(
        columns={'correct': 'baseline_acc'}
    )

    # Merge on grouping variables
    merged = method_avg.merge(
        baseline_avg,
        on=['dataset', 'axis', 'model', 'budget'],
        how='inner'
    )

    # Discard ties
    merged = merged[merged['method_acc'] != merged['baseline_acc']].copy()

    # Calculate wins (method has higher accuracy than baseline)
    merged['win'] = merged['method_acc'] > merged['baseline_acc']

    return merged


def compute_threshold_win_rates(all_results, methods_to_compare, baseline, thresholds, verbose=True):
    """
    Compute macro and micro threshold classification win rates for all methods and metrics.

    Args:
        all_results: Dict mapping metric name to DataFrame
        methods_to_compare: List of method names to compare
        baseline: Name of baseline method
        thresholds: List of threshold values to test
        verbose: Whether to print detailed results

    Returns:
        DataFrame with all threshold win rate results
    """
    results_table = []

    for threshold in thresholds:
        if verbose:
            print(f"\n{'='*90}")
            print(f"THRESHOLD: {threshold}")
            print(f"{'='*90}")

        for metric_name, df in all_results.items():
            pred_col = f'predicted_{metric_name}'
            true_col = f'true_{metric_name}'

            # Check if required columns exist
            if pred_col not in df.columns or true_col not in df.columns:
                if verbose:
                    print(f"  Skipping {metric_name}: missing {pred_col} or {true_col}")
                continue

            if verbose:
                print(f"\nMetric: {metric_name.upper()}")

            # Check what methods are available in this metric
            available_methods = df['method'].unique()

            for method in methods_to_compare:
                if method not in available_methods or baseline not in available_methods:
                    continue

                # Calculate macro threshold win rate
                macro_merged = calculate_threshold_macro_win_rate(df, method, baseline, threshold, metric_name)
                if len(macro_merged) == 0:
                    continue

                macro_win_rate = macro_merged['win'].mean() * 100
                macro_wins = macro_merged['win'].sum()
                macro_total = len(macro_merged)

                # Calculate micro threshold win rate
                micro_merged = calculate_threshold_micro_win_rate(df, method, baseline, threshold, metric_name)
                if len(micro_merged) == 0:
                    continue

                micro_win_rate = micro_merged['win'].mean() * 100
                micro_wins = micro_merged['win'].sum()
                micro_total = len(micro_merged)

                if verbose:
                    print(f"  {method}:")
                    print(f"    Macro: {macro_win_rate:.2f}% ({macro_wins}/{macro_total})")
                    print(f"    Micro: {micro_win_rate:.2f}% ({micro_wins}/{micro_total})")

                # Store for summary
                results_table.append({
                    'Threshold': threshold,
                    'Method': method,
                    'Metric': metric_name.upper(),
                    'Micro_Win_Rate': micro_win_rate,
                    'Macro_Win_Rate': macro_win_rate,
                    'Micro_Wins': int(micro_wins),
                    'Micro_Total': int(micro_total),
                    'Macro_Wins': int(macro_wins),
                    'Macro_Total': int(macro_total),
                })

                # Add per-dataset results
                for dataset_name in sorted(macro_merged['dataset'].unique()):
                    dataset_macro = macro_merged[macro_merged['dataset'] == dataset_name]
                    dataset_micro = micro_merged[micro_merged['dataset'] == dataset_name]

                    results_table.append({
                        'Threshold': threshold,
                        'Method': method,
                        'Metric': metric_name.upper(),
                        'Dataset': dataset_name,
                        'Micro_Win_Rate': dataset_micro['win'].mean() * 100,
                        'Macro_Win_Rate': dataset_macro['win'].mean() * 100,
                        'Micro_Wins': int(dataset_micro['win'].sum()),
                        'Micro_Total': len(dataset_micro),
                        'Macro_Wins': int(dataset_macro['win'].sum()),
                        'Macro_Total': len(dataset_macro),
                    })

    return pd.DataFrame(results_table)


def save_results(results_df, output_dir, prefix='win_rates'):
    """Save win rate results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Overall results
    overall = results_df[~results_df['Dataset'].notna()].copy()
    overall_path = os.path.join(output_dir, f'{prefix}_overall.csv')
    overall.to_csv(overall_path, index=False)
    print(f"\nSaved overall results to: {overall_path}")

    # Per-dataset results
    per_dataset = results_df[results_df['Dataset'].notna()].copy()
    per_dataset_path = os.path.join(output_dir, f'{prefix}_by_dataset.csv')
    per_dataset.to_csv(per_dataset_path, index=False)
    print(f"Saved per-dataset results to: {per_dataset_path}")

    # Summary table (formatted)
    summary_cols = ['Method', 'Metric', 'Micro_Win_Rate', 'Macro_Win_Rate']
    # Add Threshold column if it exists
    if 'Threshold' in overall.columns:
        summary_cols = ['Threshold'] + summary_cols
    summary = overall[summary_cols].copy()
    summary['Micro_Win_Rate'] = summary['Micro_Win_Rate'].apply(lambda x: f"{x:.2f}%")
    summary['Macro_Win_Rate'] = summary['Macro_Win_Rate'].apply(lambda x: f"{x:.2f}%")
    summary_path = os.path.join(output_dir, f'{prefix}_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")

    return overall_path, per_dataset_path, summary_path


def main(results_dirs=None, base_dir=None, output_dir=None, verbose=True):
    """
    Main execution function.

    Args:
        results_dirs: Dict mapping dataset name to dataframes directory path
        base_dir: Optional base directory to prepend to paths
        output_dir: Directory to save results (default: <base_dir>/win_rates)
        verbose: Whether to print detailed results
    """
    if results_dirs is None:
        results_dirs = DEFAULT_RESULTS_DIRS

    if output_dir is None:
        if base_dir:
            output_dir = os.path.join(base_dir, 'win_rates_comprehensive')
        else:
            output_dir = 'results/win_rates_comprehensive'

    print("=" * 90)
    print("COMPREHENSIVE WIN RATE ANALYSIS")
    print("=" * 90)
    print(f"\nDatasets: {list(results_dirs.keys())}")
    print(f"Metrics: {METRICS}")
    print(f"Methods: {METHODS_TO_COMPARE}")
    print(f"Baseline: {BASELINE}")

    print("\n" + "=" * 90)
    print("LOADING DATA")
    print("=" * 90)
    all_results = load_all_results(results_dirs, base_dir)

    if not all_results:
        print("\nError: No data loaded. Please check paths.")
        return

    # ================================================================================
    # ESTIMATION WIN RATES
    # ================================================================================
    print("\n" + "=" * 90)
    print("COMPUTING ESTIMATION WIN RATES")
    print("=" * 90)
    estimation_results = compute_win_rates(all_results, METHODS_TO_COMPARE, BASELINE, verbose=verbose)

    # Save estimation results
    print("\n" + "=" * 90)
    print("SAVING ESTIMATION RESULTS")
    print("=" * 90)
    save_results(estimation_results, output_dir, prefix='estimation_win_rates')

    # Print estimation summary
    if verbose:
        print("\n" + "=" * 90)
        print("ESTIMATION SUMMARY TABLE")
        print("=" * 90)
        overall = estimation_results[~estimation_results['Dataset'].notna()].copy()
        summary = overall[['Method', 'Metric', 'Micro_Win_Rate', 'Macro_Win_Rate']].copy()
        summary['Micro_Win_Rate'] = summary['Micro_Win_Rate'].apply(lambda x: f"{x:.2f}%")
        summary['Macro_Win_Rate'] = summary['Macro_Win_Rate'].apply(lambda x: f"{x:.2f}%")
        print(summary.to_string(index=False))

    # ================================================================================
    # THRESHOLD WIN RATES
    # ================================================================================
    print("\n" + "=" * 90)
    print("COMPUTING THRESHOLD WIN RATES")
    print("=" * 90)
    threshold_results = compute_threshold_win_rates(
        all_results, METHODS_TO_COMPARE, BASELINE, THRESHOLDS, verbose=verbose
    )

    if len(threshold_results) > 0:
        # Save threshold results
        print("\n" + "=" * 90)
        print("SAVING THRESHOLD RESULTS")
        print("=" * 90)
        save_results(threshold_results, output_dir, prefix='threshold_win_rates')

        # Print threshold summary
        if verbose:
            print("\n" + "=" * 90)
            print("THRESHOLD SUMMARY TABLE")
            print("=" * 90)
            overall_thr = threshold_results[~threshold_results['Dataset'].notna()].copy()
            summary_thr = overall_thr[['Threshold', 'Method', 'Metric', 'Micro_Win_Rate', 'Macro_Win_Rate']].copy()
            summary_thr['Micro_Win_Rate'] = summary_thr['Micro_Win_Rate'].apply(lambda x: f"{x:.2f}%")
            summary_thr['Macro_Win_Rate'] = summary_thr['Macro_Win_Rate'].apply(lambda x: f"{x:.2f}%")
            print(summary_thr.to_string(index=False))
    else:
        print("\nNo threshold results (missing predicted/true columns)")

    return estimation_results, threshold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive win rate analysis for sampling strategies"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing results (default: current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for win rates (default: <base-dir>/win_rates_comprehensive)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    main(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
