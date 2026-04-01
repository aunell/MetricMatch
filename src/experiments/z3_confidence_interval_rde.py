import argparse
import datetime
import os
import pandas as pd
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
# Default plots output directory - change this to specify where plots should be saved
DEFAULT_PLOTS_DIR = "results/plots"  # Default: results/plots
# Alternative examples:
# DEFAULT_PLOTS_DIR = "01_11_plots"
# DEFAULT_PLOTS_DIR = "/path/to/custom/plots/directory"

# Dataset name mapping (used across plotting functions)
DATASET_NAMES = {
    "hanna": "HANNA",
    "mslr": "MSLR",
    "medval": "MedVal",
    "summeval": "SummEval"
}

# ============================================================================
# Confidence Interval Calculation Functions
# ============================================================================

def fisher_icc_ci(icc_value, n_subjects, n_raters, confidence_level=0.95):
    """
    Calculate confidence interval using Fisher's Z-transformation.

    NOTE: Assumes random sampling and may be inappropriate for
    non-random sampling strategies like clustering.
    """
    if icc_value <= 0 or icc_value >= 1:
        return np.nan, np.nan

    alpha = 1 - confidence_level
    df1 = n_subjects - 1
    df2 = n_subjects * (n_raters - 1)

    f_lower = f.ppf(alpha/2, df1, df2)
    f_upper = f.ppf(1 - alpha/2, df1, df2)

    F_obs = (1 + (n_raters - 1) * icc_value) / (1 - icc_value)
    F_lower = F_obs / f_upper
    F_upper = F_obs / f_lower

    icc_lower = (F_lower - 1) / (F_lower + n_raters - 1)
    icc_upper = (F_upper - 1) / (F_upper + n_raters - 1)

    return max(0, icc_lower), min(1, icc_upper)


def empirical_icc_ci(icc_values, confidence_level=0.95):
    """
    Calculate empirical confidence interval from multiple rollouts.

    This method is distribution-free and works for any sampling strategy,
    including non-random strategies like clustering.
    """
    if len(icc_values) < 2:
        return np.nan, np.nan

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(icc_values, lower_percentile)
    ci_upper = np.percentile(icc_values, upper_percentile)

    return ci_lower, ci_upper


def analyze_icc_confidence_intervals(df, n_subjects, n_raters):
    """
    Extract ICC values from rollouts and calculate Fisher CIs.
    Returns DataFrame with per-rollout ICC and Fisher CI statistics.
    """
    results = []
    for _, row in df.iterrows():
        icc_val = row['subset_icc']
        n_subjects_actual = row['n_expensive']

        ci_lower, ci_upper = fisher_icc_ci(icc_val, n_subjects_actual, n_raters)
        ci_width = ci_upper - ci_lower

        results.append({
            'strategy': row['strategy'],
            'n_expensive': n_subjects_actual,
            'icc': icc_val,
            'fisher_ci_lower': ci_lower,
            'fisher_ci_upper': ci_upper,
            'fisher_ci_width': ci_width,
            'dataset': row['dataset'],
            'rollout': row.get('rollout', None)
        })
    return pd.DataFrame(results)


def calculate_empirical_cis(df, confidence_level=0.95):
    """
    Calculate empirical confidence intervals from rollout data.

    This is the PRIMARY CI method - it captures real variability including
    the effect of sampling strategy (clustering vs random).
    """
    results = []

    for (strategy, n_expensive), group in df.groupby(['strategy', 'n_expensive']):
        icc_values = group['icc'].values

        # Calculate empirical CI
        emp_ci_lower, emp_ci_upper = empirical_icc_ci(icc_values, confidence_level)
        emp_ci_width = emp_ci_upper - emp_ci_lower

        # Include Fisher CI stats for comparison
        fisher_ci_width_mean = group['fisher_ci_width'].mean()
        fisher_ci_width_std = group['fisher_ci_width'].std()

        results.append({
            'strategy': strategy,
            'n_expensive': n_expensive,
            'mean_icc': icc_values.mean(),
            'std_icc': icc_values.std(),
            'ci_lower': emp_ci_lower,
            'ci_upper': emp_ci_upper,
            'ci_width': emp_ci_width,
            'fisher_ci_width_mean': fisher_ci_width_mean,
            'fisher_ci_width_std': fisher_ci_width_std,
            'n_rollouts': len(icc_values)
        })

    return pd.DataFrame(results)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_ci_width_with_variability(empirical_df, dataset_name, out_path_empirical=None, out_path_fisher=None):
    """
    Generate two separate plots: one for empirical CI and one for Fisher's CI.

    Returns:
        fig_empirical, fig_fisher: Tuple of matplotlib figure objects
    """
    display_name = DATASET_NAMES.get(dataset_name, dataset_name)

    # Plot 1: Empirical CI Width
    fig_empirical = plt.figure(figsize=(12, 8))
    for strategy in ['Random', 'Cluster']:
        if strategy not in empirical_df['strategy'].values:
            continue
        sub = empirical_df[empirical_df['strategy'] == strategy].sort_values('n_expensive')
        plt.errorbar(sub['n_expensive'],
                     sub['ci_width'],
                     yerr=sub['std_icc'],
                     label=strategy,
                     marker='o', capsize=5, linewidth=2, markersize=8)

    plt.xlabel('Number of Expensive Ratings (n_expensive)', fontsize=14)
    plt.ylabel('Empirical CI Width', fontsize=14)
    plt.title(f"Empirical Confidence Interval Width\n({display_name} dataset)",
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path_empirical:
        plt.savefig(out_path_empirical, dpi=300, bbox_inches='tight')

    # Plot 2: Fisher's CI Width
    fig_fisher = plt.figure(figsize=(12, 8))
    for strategy in ['Random', 'Cluster']:
        if strategy not in empirical_df['strategy'].values:
            continue
        sub = empirical_df[empirical_df['strategy'] == strategy].sort_values('n_expensive')
        plt.errorbar(sub['n_expensive'],
                     sub['fisher_ci_width_mean'],
                     yerr=sub['fisher_ci_width_std'],
                     label=strategy,
                     marker='o', capsize=5, linewidth=2, markersize=8)

    plt.xlabel('Number of Expensive Ratings (n_expensive)', fontsize=14)
    plt.ylabel('Mean Fisher CI Width', fontsize=14)
    plt.title(f"Fisher's Confidence Interval Width\n({display_name} dataset)",
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path_fisher:
        plt.savefig(out_path_fisher, dpi=300, bbox_inches='tight')

    return fig_empirical, fig_fisher


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_dir_for_dataset(base_dir, n_subjects=300, n_raters=2, dataset_name="hanna", dimension=None):
    """
    Analyze all rollout files for a given dataset and compute empirical CIs.

    Args:
        base_dir: Base directory containing CSV files
        n_subjects: Number of subjects
        n_raters: Number of raters
        dataset_name: Dataset name to filter
        dimension: Specific dimension to analyze (e.g., 'fluency', 'coherence')
                  If None, analyze all dimensions
    """
    all_icc_data = []

    for fname in os.listdir(base_dir):
        if dataset_name in fname.lower() and fname.endswith(".csv"):
            # Skip output files from previous runs (empirical_ci_results, cluster_improvement)
            if fname.startswith(("empirical_ci_results_", "cluster_improvement_")):
                continue

            # If dimension is specified, filter by dimension
            if dimension is not None:
                if dimension not in fname.lower():
                    continue

            fpath = os.path.join(base_dir, fname)
            print(f"Processing {fpath}")
            df = pd.read_csv(fpath)

            # Extract ICC values from rollouts
            icc_data = analyze_icc_confidence_intervals(df, n_subjects, n_raters)
            icc_data['file'] = fname
            all_icc_data.append(icc_data)

    if not all_icc_data:
        dim_msg = f" (dimension: {dimension})" if dimension else ""
        print(f"No {dataset_name} CSVs found in {base_dir}{dim_msg}")
        return None

    # Combine all ICC data
    combined_icc_data = pd.concat(all_icc_data, ignore_index=True)

    # Calculate empirical CIs from the combined rollout data
    empirical_ci_df = calculate_empirical_cis(combined_icc_data)

    return empirical_ci_df

def compute_cluster_vs_random_improvement(df, value_col="ci_width"):
    """
    Compute improvement of Cluster over Random for a given metric.

    Args:
        df: DataFrame with columns ["strategy", "n_expensive", value_col]
        value_col: Column name containing the metric to compare

    Returns:
        DataFrame with cluster vs random comparison and improvement metrics
    """
    cluster_df = df[df['strategy'] == 'Cluster']
    random_df = df[df['strategy'] == 'Random']

    print(f"Cluster n_expensive values: {sorted(cluster_df['n_expensive'].unique())}")
    print(f"Random n_expensive values: {sorted(random_df['n_expensive'].unique())}")

    # Merge on n_expensive
    merged = pd.merge(
        cluster_df, random_df,
        on=['n_expensive'],
        suffixes=('_cluster', '_random')
    )

    # Compute improvements (positive = Cluster is better)
    merged['improvement'] = merged[f'{value_col}_random'] - merged[f'{value_col}_cluster']
    merged['improvement_%'] = (merged['improvement'] / merged[f'{value_col}_random'] * 100)

    return merged[['n_expensive',
                   f'{value_col}_cluster', f'{value_col}_random',
                   'improvement', 'improvement_%']]


# ============================================================================
# Main Execution
# ============================================================================

def process_single_analysis(base_dir, args, dimension=None):
    """Process and save results for a single analysis (one dimension or aggregated)."""
    empirical_ci_df = analyze_dir_for_dataset(
        base_dir, args.n_subjects, args.n_raters,
        dataset_name=args.dataset, dimension=dimension
    )

    if empirical_ci_df is None:
        print(f"No data found for dataset: {args.dataset}" + (f", dimension: {dimension}" if dimension else ""))
        return

    # Compute improvement
    improvement = compute_cluster_vs_random_improvement(empirical_ci_df, value_col=args.value_col)

    # Determine file suffix
    suffix = f"_{dimension}" if dimension else "_aggregated"

    # Generate plots
    plot_path_emp = f"{base_dir}/empirical_ci_width_{args.dataset}{suffix}.png"
    plot_path_fish = f"{base_dir}/fisher_ci_width_{args.dataset}{suffix}.png"
    plot_ci_width_with_variability(
        empirical_ci_df,
        dataset_name=args.dataset,
        out_path_empirical=plot_path_emp,
        out_path_fisher=plot_path_fish
    )

    # Print results
    dim_label = f" - {dimension.upper()}" if dimension else " (Aggregated)"
    print("\n" + "="*80)
    print(f"EMPIRICAL CI RESULTS{dim_label}")
    print("="*80)
    print(empirical_ci_df.to_string())

    print("\n" + "="*80)
    print("CLUSTER VS RANDOM IMPROVEMENT")
    print("="*80)
    print(improvement.to_string())

    print(f"\nPlots saved to:\n  - {plot_path_emp}\n  - {plot_path_fish}")

    # Save results to CSV
    csv_path = f"{base_dir}/empirical_ci_results_{args.dataset}{suffix}.csv"
    empirical_ci_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    improvement_csv_path = f"{base_dir}/cluster_improvement_{args.dataset}{suffix}.csv"
    improvement.to_csv(improvement_csv_path, index=False)
    print(f"Improvement results saved to: {improvement_csv_path}")


def find_dimensions_in_dir(base_dir, dataset_name):
    """Find all dimensions in the dataset files."""
    dimensions = set()
    for fname in os.listdir(base_dir):
        if dataset_name in fname.lower() and fname.endswith(".csv"):
            if not fname.startswith(("empirical_ci_results_", "cluster_improvement_")):
                # Expected format: results_dataset_model_dimension_simulation_results.csv
                parts = fname.replace(".csv", "").split("_")
                if len(parts) >= 4:
                    dimensions.add(parts[3])
    return sorted(dimensions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ICC results with empirical confidence intervals.")
    parser.add_argument("--dataset", type=str, default="medval", help="Dataset name")
    parser.add_argument("--dimension", type=str, default=None,
                        help="Specific dimension (if None, processes all found in files)")
    parser.add_argument("--n_subjects", type=int, default=300, help="Number of subjects")
    parser.add_argument("--n_raters", type=int, default=2, help="Number of raters")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base results directory. If not given, auto-generate with date.")
    parser.add_argument("--value_col", type=str, default="ci_width",
                        help="Metric column to compute improvement on")
    parser.add_argument("--aggregate", action="store_true", default=True,
                        help="Aggregate dimensions (default: True)")
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                        help="Generate separate plots for each dimension")
    args = parser.parse_args()

    # Construct base directory if not provided
    if args.base_dir is None:
        base_dir = DEFAULT_PLOTS_DIR
    else:
        base_dir = args.base_dir

    print(f"Using base_dir: {base_dir}")
    print(f"Analyzing dataset: {args.dataset}")
    if args.dimension:
        print(f"Dimension: {args.dimension}")
    print(f"Aggregate mode: {args.aggregate}")
    print("="*80)

    if args.aggregate or args.dimension:
        # Single analysis (aggregated or specific dimension)
        dimension_filter = args.dimension if not args.aggregate else None
        process_single_analysis(base_dir, args, dimension=dimension_filter)
    else:
        # Per-dimension analysis
        dimensions = find_dimensions_in_dir(base_dir, args.dataset)
        print(f"\nFound dimensions: {dimensions}")

        for dimension in dimensions:
            print(f"\n{'='*80}")
            print(f"Analyzing dimension: {dimension.upper()}")
            print("="*80)
            process_single_analysis(base_dir, args, dimension=dimension)