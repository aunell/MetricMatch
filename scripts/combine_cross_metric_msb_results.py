#!/usr/bin/env python3
"""
Script to combine cross_metric_msb_match_results.csv files from all datasets.
Similar to combine_metric_matched_csvs.py but specifically for MSB matching results.
"""

import pandas as pd
from pathlib import Path

def combine_msb_results():
    """Combine cross_metric_msb_match_results.csv from all datasets"""
    print("=" * 80)
    print("COMBINING CROSS-METRIC MSB MATCH RESULTS")
    print("=" * 80)

    # Define datasets and paths
    datasets = ['hanna', 'medval', 'mslr', 'summeval']
    base_dir = Path("/Users/alyssaunell/code/SmartSample_local/results/04_31_late/metric_matched_subsets")
    output_dir = Path("/Users/alyssaunell/code/SmartSample_local/results/05_01")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    # Process each dataset
    for dataset in datasets:
        dataset_file = base_dir / dataset / "cross_metric_msb_match_results.csv"

        if not dataset_file.exists():
            print(f"Warning: {dataset_file} does not exist, skipping {dataset}...")
            continue

        print(f"\nReading {dataset}...")
        df = pd.read_csv(dataset_file)

        # Add dataset column
        df['dataset'] = dataset

        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Methods: {df['method'].unique()}")
        if 'match_metric' in df.columns:
            print(f"  Match metrics: {df['match_metric'].unique()}")
        if 'est_metric' in df.columns:
            print(f"  Estimation metrics: {df['est_metric'].unique()}")

        all_dfs.append(df)

    if not all_dfs:
        print("\nError: No data files found!")
        return

    # Combine all dataframes
    print("\n" + "=" * 60)
    print("COMBINING ALL DATAFRAMES")
    print("=" * 60)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total combined rows: {len(combined_df)}")

    # Save the combined file
    output_file = output_dir / "cross_metric_msb_match_results_combined.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined results to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nDataset distribution:")
    print(combined_df['dataset'].value_counts())

    print(f"\nMethod distribution:")
    print(combined_df['method'].value_counts())

    if 'match_metric' in combined_df.columns:
        print(f"\nMatch metric distribution:")
        print(combined_df['match_metric'].value_counts(dropna=False))

    if 'est_metric' in combined_df.columns:
        print(f"\nEstimation metric distribution:")
        print(combined_df['est_metric'].value_counts())

    if 'axis' in combined_df.columns:
        print(f"\nAxis distribution:")
        print(combined_df['axis'].value_counts())

    if 'model' in combined_df.columns:
        print(f"\nModel distribution:")
        print(combined_df['model'].value_counts())

    print(f"\n{output_file} created successfully!")

if __name__ == "__main__":
    combine_msb_results()
