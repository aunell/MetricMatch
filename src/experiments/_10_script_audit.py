"""
Audit script to compare cross_metric results with metric-specific results files
to ensure they contain the same data with different column names.

Cross metric file columns:
- model, budget, method, match_metric, est_metric, est_error, model_metric_value,
  true_metric_value, best_ids, axis

Metric results file columns (e.g., alpha_results.csv, icc_results.csv):
- model, budget, method, estimation_error, predicted_{metric}, true_{metric}, best_ids, axis

Expected mappings:
- est_error (cross_metric) == estimation_error (metric_results)
- model_metric_value (cross_metric) == predicted_{metric} (metric_results)
- true_metric_value (cross_metric) == true_{metric} (metric_results)
- best_ids should be the same
- method mapping: "metric_match" (cross_metric) == "metric_matched_{metric}" (metric_results)
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path


def normalize_best_ids(best_ids_str):
    """
    Normalize best_ids string to a sorted tuple for comparison.
    Handles both string representations and array-like strings.
    """
    if pd.isna(best_ids_str):
        return tuple()

    # Try to parse as a Python literal (list)
    try:
        parsed = ast.literal_eval(best_ids_str)
        if isinstance(parsed, list):
            return tuple(sorted(parsed))
    except (ValueError, SyntaxError):
        pass

    # If that fails, try to clean up and parse
    # Remove brackets and quotes, split by whitespace/commas
    cleaned = str(best_ids_str).strip("[]'\"").replace("'", "").replace('"', '')
    items = [item.strip() for item in cleaned.split() if item.strip()]
    return tuple(sorted(items))


def compare_dataframes(cross_metric_path, metric_results_path, metric_name, cross_metric_name=None):
    """
    Compare the two CSV files row-by-row in order.
    For each (model, budget, method) combination, the rows should match in sequence.

    Args:
        cross_metric_path: Path to cross_metric_{metric}_match_results.csv
        metric_results_path: Path to {metric}_results.csv
        metric_name: Name of the metric as used in metric_results files (e.g., 'alpha', 'icc', 'rho')
        cross_metric_name: Name of the metric as used in cross_metric files (e.g., 'spearman' for 'rho')

    Returns:
        dict: Dictionary with comparison results
    """
    # Use cross_metric_name if provided, otherwise same as metric_name
    if cross_metric_name is None:
        cross_metric_name = metric_name

    # Read the CSV files
    print("Reading CSV files...")
    cross_df = pd.read_csv(cross_metric_path)
    metric_df = pd.read_csv(metric_results_path)

    print(f"Cross metric file: {len(cross_df)} rows")
    print(f"Metric results file: {len(metric_df)} rows")

    # Filter cross_metric to only include est_metric == cross_metric_name
    cross_df_filtered = cross_df[cross_df['est_metric'] == cross_metric_name].copy()
    print(f"Cross metric file (est_metric == '{cross_metric_name}'): {len(cross_df_filtered)} rows")

    # Filter to only include relevant methods
    cross_methods = ['random', 'metric_match']
    metric_methods = ['random', f'metric_matched_{metric_name}']

    cross_df_filtered = cross_df_filtered[cross_df_filtered['method'].isin(cross_methods)].copy()
    metric_df_filtered = metric_df[metric_df['method'].isin(metric_methods)].copy()

    print(f"Cross metric file (filtered methods): {len(cross_df_filtered)} rows")
    print(f"Metric results file (filtered methods): {len(metric_df_filtered)} rows")

    # Normalize method names for comparison
    cross_df_filtered['method_normalized'] = cross_df_filtered['method'].replace({
        'metric_match': f'metric_matched_{metric_name}',
        'random': 'random'
    })
    metric_df_filtered['method_normalized'] = metric_df_filtered['method']

    # Rename columns in cross_df to match metric_df for easier comparison
    cross_df_renamed = cross_df_filtered.rename(columns={
        'est_error': 'estimation_error',
        'model_metric_value': f'predicted_{metric_name}',
        'true_metric_value': f'true_{metric_name}'
    })

    # Normalize best_ids in both dataframes (if column exists)
    has_best_ids = 'best_ids' in metric_df_filtered.columns
    if has_best_ids:
        print("\nNormalizing best_ids...")
        cross_df_renamed['best_ids_normalized'] = cross_df_renamed['best_ids'].apply(normalize_best_ids)
        metric_df_filtered['best_ids_normalized'] = metric_df_filtered['best_ids'].apply(normalize_best_ids)
    else:
        print("\nNote: metric_results file does not have best_ids column - skipping best_ids comparison")

    # Compare row-by-row within each (model, budget, method) group
    print("\nComparing rows in order for each (model, budget, method) combination...")

    results = {
        'total_rows_compared': 0,
        'matching_rows': 0,
        'mismatches': [],
        'all_match': True
    }

    tolerance = 1e-10

    # Get all unique combinations (including axis)
    cross_groups = cross_df_renamed.groupby(['model', 'budget', 'method_normalized', 'axis'])
    metric_groups = metric_df_filtered.groupby(['model', 'budget', 'method_normalized', 'axis'])

    cross_group_keys = set(cross_groups.groups.keys())
    metric_group_keys = set(metric_groups.groups.keys())

    common_groups = cross_group_keys & metric_group_keys
    cross_only_groups = cross_group_keys - metric_group_keys
    metric_only_groups = metric_group_keys - cross_group_keys

    print(f"\nCommon (model, budget, method, axis) combinations: {len(common_groups)}")
    print(f"Only in cross_metric: {len(cross_only_groups)}")
    print(f"Only in metric_results: {len(metric_only_groups)}")

    if cross_only_groups:
        print(f"\nGroups only in cross_metric: {cross_only_groups}")
    if metric_only_groups:
        print(f"Groups only in metric_results: {metric_only_groups}")

    # Compare each common group
    for group_key in sorted(common_groups):
        model, budget, method, axis = group_key

        cross_group = cross_df_renamed[
            (cross_df_renamed['model'] == model) &
            (cross_df_renamed['budget'] == budget) &
            (cross_df_renamed['method_normalized'] == method) &
            (cross_df_renamed['axis'] == axis)
        ].reset_index(drop=True)

        metric_group = metric_df_filtered[
            (metric_df_filtered['model'] == model) &
            (metric_df_filtered['budget'] == budget) &
            (metric_df_filtered['method_normalized'] == method) &
            (metric_df_filtered['axis'] == axis)
        ].reset_index(drop=True)

        # print(f"\n{model}, budget={budget}, method={method}, axis={axis}:")
        # print(f"  Cross metric: {len(cross_group)} rows")
        # print(f"  Metric results: {len(metric_group)} rows")

        if len(cross_group) != len(metric_group):
            print(f"  ⚠ Row count mismatch!")
            results['all_match'] = False
            continue

        # Compare row by row in order
        for i in range(len(cross_group)):
            results['total_rows_compared'] += 1
            cross_row = cross_group.iloc[i]
            metric_row = metric_group.iloc[i]

            row_matches = True
            row_mismatches = []

            # Compare best_ids (if available)
            # if has_best_ids and cross_row['best_ids_normalized'] != metric_row['best_ids_normalized']:
            #     row_matches = False
            #     row_mismatches.append({
            #         'field': 'best_ids',
            #         'cross_value': str(cross_row['best_ids_normalized'])[:100],
            #         'metric_value': str(metric_row['best_ids_normalized'])[:100]
            #     })

            # Compare estimation_error
            if not np.isclose(cross_row['estimation_error'], metric_row['estimation_error'], atol=tolerance):
                row_matches = False
                row_mismatches.append({
                    'field': 'estimation_error',
                    'cross_value': cross_row['estimation_error'],
                    'metric_value': metric_row['estimation_error'],
                    'difference': abs(cross_row['estimation_error'] - metric_row['estimation_error'])
                })

            # Compare predicted metric value (if available)
            predicted_col = f'predicted_{metric_name}'
            if predicted_col in metric_row and predicted_col in cross_row:
                if not np.isclose(cross_row[predicted_col], metric_row[predicted_col], atol=tolerance):
                    row_matches = False
                    row_mismatches.append({
                        'field': predicted_col,
                        'cross_value': cross_row[predicted_col],
                        'metric_value': metric_row[predicted_col],
                        'difference': abs(cross_row[predicted_col] - metric_row[predicted_col])
                    })

            # Compare true metric value (if available)
            true_col = f'true_{metric_name}'
            if true_col in metric_row and true_col in cross_row:
                if not np.isclose(cross_row[true_col], metric_row[true_col], atol=tolerance):
                    row_matches = False
                    row_mismatches.append({
                        'field': true_col,
                        'cross_value': cross_row[true_col],
                        'metric_value': metric_row[true_col],
                        'difference': abs(cross_row[true_col] - metric_row[true_col])
                    })

            if row_matches:
                results['matching_rows'] += 1
            else:
                results['all_match'] = False
                results['mismatches'].append({
                    'model': model,
                    'budget': budget,
                    'method': method,
                    'row_index': i,
                    'issues': row_mismatches,
                    'axis': axis
                })

    print(f"\n{'='*80}")
    print(f"Total rows compared: {results['total_rows_compared']}")
    print(f"Matching rows: {results['matching_rows']}")
    print(f"Mismatching rows: {len(results['mismatches'])}")

    # Breakdown by method
    # random_mismatches = [m for m in results['mismatches'] if m['method'] == 'random']
    # metric_matched_mismatches = [m for m in results['mismatches'] if m['method'] == f'metric_matched_{metric_name}']

    # print(f"\nMismatches by method:")
    # print(f"  random: {len(random_mismatches)}")
    # print(f"  metric_matched_{metric_name}: {len(metric_matched_mismatches)}")

    if results['all_match']:
        print("\n✓ All rows match!")
    else:
        print(f"\n✗ Found {len(results['mismatches'])} mismatching rows")
        print("\nFirst 5 mismatches:")
        for mismatch in results['mismatches'][:5]:
            print(f"\n  {mismatch['model']}, axis = {mismatch['axis']}, budget={mismatch['budget']}, method={mismatch['method']}, row {mismatch['row_index']}:")
            for issue in mismatch['issues']:
                print(f"    {issue['field']}:")
                print(f"      Cross: {issue['cross_value']}")
                print(f"      Metric: {issue['metric_value']}")
                if 'difference' in issue:
                    print(f"      Diff: {issue['difference']}")

    return results



def compare_without_best_ids(cross_metric_path, metric_results_path, metric_name, cross_metric_name=None):
    """
    Compare ignoring best_ids to see if the issue is just different random seeds.
    """
    # Use cross_metric_name if provided, otherwise same as metric_name
    if cross_metric_name is None:
        cross_metric_name = metric_name

    print("\n" + "=" * 80)
    print("ALTERNATIVE COMPARISON: Ignoring best_ids (checking if distributions match)")
    print("=" * 80)

    cross_df = pd.read_csv(cross_metric_path)
    metric_df = pd.read_csv(metric_results_path)

    # Filter
    cross_df_filtered = cross_df[cross_df['est_metric'] == cross_metric_name].copy()
    cross_df_filtered = cross_df_filtered[cross_df_filtered['method'].isin(['random', 'metric_match'])]

    metric_df_filtered = metric_df[metric_df['method'].isin(['random', f'metric_matched_{metric_name}'])].copy()

    # Normalize method names
    cross_df_filtered['method_normalized'] = cross_df_filtered['method'].replace({
        'metric_match': f'metric_matched_{metric_name}',
        'random': 'random'
    })

    # Rename columns
    cross_df_filtered = cross_df_filtered.rename(columns={
        'est_error': 'estimation_error',
        'model_metric_value': f'predicted_{metric_name}',
        'true_metric_value': f'true_{metric_name}'
    })

    # Group by model, budget, method and compare statistics
    print("\nComparing distributions by (model, budget, method):")
    print()

    for model in cross_df_filtered['model'].unique():
        for budget in sorted(cross_df_filtered['budget'].unique()):
            for method in ['random', f'metric_matched_{metric_name}']:
                cross_subset = cross_df_filtered[
                    (cross_df_filtered['model'] == model) &
                    (cross_df_filtered['budget'] == budget) &
                    (cross_df_filtered['method_normalized'] == method)
                ]

                metric_subset = metric_df_filtered[
                    (metric_df_filtered['model'] == model) &
                    (metric_df_filtered['budget'] == budget) &
                    (metric_df_filtered['method'] == method)
                ]

                # if len(cross_subset) > 0 or len(metric_subset) > 0:
                #     print(f"{model}, budget={budget}, method={method}:")
                #     print(f"  Cross metric: {len(cross_subset)} rows, est_error mean={cross_subset['estimation_error'].mean():.4f}, std={cross_subset['estimation_error'].std():.4f}")
                #     print(f"  Metric results: {len(metric_subset)} rows, est_error mean={metric_subset['estimation_error'].mean():.4f}, std={metric_subset['estimation_error'].std():.4f}")


def main(metric_name='alpha', dataset='medval'):
    """Main function to run the audit.

    Args:
        metric_name: Name of the metric (e.g., 'alpha', 'icc', 'spearman')
        dataset: Dataset name (e.g., 'medval', 'hanna', 'mslr')
        cross_base_dir: Base directory for cross_metric files
        metric_base_dir: Base directory for metric_results files
    """
    cross_base_dir=f'results/05_01_natalie_audit/metric_matched_subsets'
    metric_base_dir=f'results/05_01_VM_{dataset}_audit'
    # Map abbreviated metric names to their full names used in cross_metric files
    # The cross_metric files use full names (spearman, kendalltau, mean_sq_error)
    # But the metric_results files use abbreviated names (rho, tau, msre)
    metric_name_to_cross_file = {
        'rho': 'spearman',
        'tau': 'kendalltau',
        'mse': 'mean_sq_error'
    }

    # Get the cross-metric file name (full name if abbreviated, or same as metric_name)
    cross_metric_file_name = metric_name_to_cross_file.get(metric_name, metric_name)

    cross_metric_path = Path(f"/Users/alyssaunell/code/SmartSample_local/{cross_base_dir}/{dataset}/cross_metric_{cross_metric_file_name}_match_results.csv")
    metric_results_path = Path(f"/Users/alyssaunell/code/SmartSample_local/{metric_base_dir}/{dataset}/dataframes/{metric_name}_results.csv")

    if not cross_metric_path.exists():
        print(f"Error: {cross_metric_path} does not exist")
        return

    if not metric_results_path.exists():
        print(f"Error: {metric_results_path} does not exist")
        return

    print("=" * 80)
    print(f"AUDIT: Comparing cross_metric_{cross_metric_file_name}_match_results.csv and {metric_name}_results.csv")
    print("=" * 80)

    results = compare_dataframes(cross_metric_path, metric_results_path, metric_name, cross_metric_name=cross_metric_file_name)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rows compared: {results['total_rows_compared']}")
    print(f"Matching rows: {results['matching_rows']}")
    print(f"Mismatching rows: {len(results['mismatches'])}")

    if results['all_match']:
        print("\n✓ SUCCESS: All rows match in order!")
    else:
        print(f"\n✗ FAILED: Found {len(results['mismatches'])} mismatching rows")
        # breakpoint()

    # Try comparing without best_ids
    compare_without_best_ids(cross_metric_path, metric_results_path, metric_name, cross_metric_name=cross_metric_file_name)
    return results


if __name__ == "__main__":
    import sys

    # Allow command line arguments: python script.py [metric_name] [dataset]
    metric_name = sys.argv[1] if len(sys.argv) > 1 else 'alpha'
    dataset = sys.argv[2] if len(sys.argv) > 2 else 'medval'

    main(metric_name=metric_name, dataset=dataset)
