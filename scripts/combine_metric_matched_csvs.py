#!/usr/bin/env python3
"""
Script to combine metric-matched subset results:
- ALL rows from msb+msre CSV
- Matching metric rows (match_metric==est_metric) from other cross_metric files
- Random baseline rows from each cross_metric file (for that file's target metric)
- Baseline methods (random_imc, stratified, variance_matched_msb) from 04_30 results
- Updated random_imc and stratified for MSE from 04_31 results
"""

import pandas as pd
from pathlib import Path

def process_dataset(dataset_name):
    """Process a single dataset and create combined CSV"""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}\n")

    # Define the data directory
    data_dir = Path("/Users/alyssaunell/code/SmartSample_local/data/metric_matched_subsets") / dataset_name
    output_dir = Path("/Users/alyssaunell/code/SmartSample_local/data/metric_matched_subsets")

    # Read the msb+msre file - we want ALL rows from this file
    msb_msre_file = data_dir / "cross_metric_msb+msre_match_results.csv"
    print(f"Reading {msb_msre_file}...")
    msb_msre_df = pd.read_csv(msb_msre_file)

    # Get only the random rows from msb+msre
    random_rows = msb_msre_df[msb_msre_df['method'] == 'random'].copy()
    print(f"Found {len(random_rows)} random rows from msb+msre")

    # Get ALL other rows from msb+msre (the metric_match rows)
    all_msb_msre_rows = msb_msre_df.copy()
    print(f"Total rows from msb+msre: {len(all_msb_msre_rows)}")

    # Initialize list to collect all dataframes
    all_dfs = [all_msb_msre_rows]

    # Define the OTHER cross-metric files (not msb+msre)
    cross_metric_files = {
        'alpha': data_dir / "cross_metric_alpha_match_results.csv",
        'icc': data_dir / "cross_metric_icc_match_results.csv",
        'kendalltau': data_dir / "cross_metric_kendalltau_match_results.csv",
        'mean_sq_error': data_dir / "cross_metric_mean_sq_error_match_results.csv",
        'spearman': data_dir / "cross_metric_spearman_match_results.csv",
    }

    # Process each cross-metric file
    for metric_name, file_path in cross_metric_files.items():
        print(f"\nProcessing {file_path.name}...")
        df = pd.read_csv(file_path)

        # Filter rows where match_metric == est_metric (metric-matched results)
        # Note: match_metric might have NaN values, so we need to handle that
        matched_rows = df[df['match_metric'] == df['est_metric']].copy()
        print(f"Found {len(matched_rows)} rows where match_metric == est_metric")

        all_dfs.append(matched_rows)

        # Also extract random method baseline from this file
        # The random rows should be recorded for this file's target metric
        random_rows = df[df['method'] == 'random'].copy()
        if len(random_rows) > 0:
            # Ensure est_metric is set correctly for random rows
            random_rows['est_metric'] = metric_name
            random_rows['match_metric'] = ''  # Random has no match metric
            print(f"Found {len(random_rows)} random baseline rows for {metric_name}")
            all_dfs.append(random_rows)

    # Add baseline methods from 04_30 results
    print("\n" + "="*60)
    print("Adding baseline methods from 04_30 results")
    print("="*60)

    # Determine the correct path for 04_30 results
    baseline_dir = Path(f"/Users/alyssaunell/code/SmartSample_local/results/04_30_{dataset_name}/{dataset_name}/dataframes")

    # Define the metric result files
    metric_files = {
        'alpha': 'alpha_results.csv',
        'icc': 'icc_results.csv',
        'spearman': 'rho_results.csv',
        'kendalltau': 'tau_results.csv',
        'mean_sq_error': 'mse_results.csv'
    }

    # Methods to extract
    baseline_methods = ['random_imc', 'stratified', 'variance_matched_msb']

    for est_metric, filename in metric_files.items():
        file_path = baseline_dir / filename
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping...")
            continue

        print(f"\nReading {file_path.name}...")
        df = pd.read_csv(file_path)

        # Filter for baseline methods
        baseline_df = df[df['method'].isin(baseline_methods)].copy()

        if len(baseline_df) == 0:
            print(f"  No baseline methods found in {filename}")
            continue

        # Rename columns to match the combined format
        # The 04_29 files have: model, budget, method, estimation_error, predicted_X, true_X, axis
        # We need: model, budget, method, match_metric, est_metric, est_error, axis

        baseline_df = baseline_df.rename(columns={'estimation_error': 'est_error'})
        baseline_df['est_metric'] = est_metric

        # Set match_metric based on method
        # variance_matched_weighted_.9 uses msb+msre as the matching metric
        # Other baseline methods don't have a match_metric
        baseline_df['match_metric'] = baseline_df['method'].apply(
            lambda x: 'msb+msre' if x == 'variance_matched_weighted_.9' else ''
        )

        # Select only the columns we need
        baseline_df = baseline_df[['model', 'budget', 'method', 'match_metric', 'est_metric', 'est_error', 'axis']]

        print(f"  Found {len(baseline_df)} baseline rows for {est_metric}")
        print(f"    Method breakdown: {baseline_df['method'].value_counts().to_dict()}")

        all_dfs.append(baseline_df)

    # Add baseline methods from 04_31 results (MSE only - updated random_imc and stratified)
    print("\n" + "="*60)
    print("Adding updated random_imc and stratified from 04_31 results (MSE only)")
    print("="*60)

    baseline_04_31_dir = Path(f"/Users/alyssaunell/code/SmartSample_local/results/04_31_{dataset_name}/{dataset_name}/dataframes")
    mse_file_path = baseline_04_31_dir / "mse_results.csv"

    if mse_file_path.exists():
        print(f"\nReading {mse_file_path.name}...")
        df = pd.read_csv(mse_file_path)

        # Filter for random_imc and stratified only
        baseline_04_31_df = df[df['method'].isin(['random_imc', 'stratified'])].copy()

        if len(baseline_04_31_df) > 0:
            # Rename and format columns
            baseline_04_31_df = baseline_04_31_df.rename(columns={'estimation_error': 'est_error'})
            baseline_04_31_df['est_metric'] = 'mean_sq_error'
            baseline_04_31_df['match_metric'] = ''
            baseline_04_31_df = baseline_04_31_df[['model', 'budget', 'method', 'match_metric', 'est_metric', 'est_error', 'axis']]

            print(f"  Found {len(baseline_04_31_df)} rows for MSE from 04_31")
            print(f"    Method breakdown: {baseline_04_31_df['method'].value_counts().to_dict()}")

            all_dfs.append(baseline_04_31_df)
        else:
            print(f"  No random_imc or stratified methods found in {mse_file_path.name}")
    else:
        print(f"Warning: {mse_file_path} does not exist, skipping 04_31 MSE update...")

    # Combine all dataframes
    print("\n" + "="*60)
    print("Combining all dataframes...")
    print("="*60)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total combined rows: {len(combined_df)}")

    # Save the combined file
    output_file = output_dir / f"{dataset_name}_combined_results.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined results to: {output_file}")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Method distribution:")
    print(combined_df['method'].value_counts())
    print(f"\nMatch metric distribution:")
    print(combined_df['match_metric'].value_counts(dropna=False))
    print(f"\nEstimation metric distribution:")
    print(combined_df['est_metric'].value_counts())

# Process all datasets
datasets = ['hanna', 'medval', 'mslr', 'summeval']

for dataset in datasets:
    process_dataset(dataset)
