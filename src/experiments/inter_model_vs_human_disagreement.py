#!/usr/bin/env python3
"""
Experiment: Inter-Model Disagreement vs Model-Human Disagreement

Research Question:
When different LLM judges disagree with each other (high inter-model variance),
are they more likely to disagree with human ratings?

Analysis:
1. Load judge scores from different models for the same dimension
2. Calculate inter-model disagreement (std of model scores for each item)
3. Calculate model-human disagreement (avg difference between models and humans)
4. Analyze correlation between inter-model disagreement and model-human disagreement
5. Compare across datasets and dimensions

Metrics:
- inter_model_std: Standard deviation of LLM scores across different models for same item
- avg_model_human_diff: Average absolute difference between each model's score and human score
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# CONFIG
# -------------------------
# Default plots output directory - change this to specify where plots should be saved
DEFAULT_PLOTS_DIR = "results/plots"  # Default: results/plots
# Alternative examples:
# DEFAULT_PLOTS_DIR = "01_11_plots"
# DEFAULT_PLOTS_DIR = "/path/to/custom/plots/directory"


def get_available_models(base_dir: Path, dataset_name: str, dimension: str) -> List[str]:
    """
    Find all available model files for a given dataset and dimension.

    Args:
        base_dir: Base directory containing judge scores
        dataset_name: Name of dataset (e.g., 'hanna', 'summeval')
        dimension: Dimension name (e.g., 'Coherence', 'coherence')

    Returns:
        List of model names
    """
    dataset_dir = base_dir / dataset_name

    if not dataset_dir.exists():
        return []

    # Pattern: results_{dataset}_{model}_{dimension}.json
    models = []

    for file in dataset_dir.glob(f"results_{dataset_name}_*_{dimension}.json"):
        # Extract model name from filename
        # Format: results_{dataset}_{model}_{dimension}.json
        parts = file.stem.split('_')

        # Find where the model name starts (after dataset name)
        dataset_parts = dataset_name.split('_')
        start_idx = len(dataset_parts) + 1  # +1 for 'results'

        # Find where dimension starts (from the end)
        # The dimension is the last part
        end_idx = len(parts) - 1

        # Model name is everything in between
        model_name = '_'.join(parts[start_idx:end_idx])

        # Skip aggregated judge files
        if 'agg-judge' not in model_name and 'combined' not in model_name:
            models.append(model_name)

    return sorted(list(set(models)))


def load_model_scores(base_dir: Path, dataset_name: str, dimension: str,
                     models: List[str]) -> pd.DataFrame:
    """
    Load scores from multiple models for a given dataset and dimension.

    Args:
        base_dir: Base directory containing judge scores
        dataset_name: Name of dataset
        dimension: Dimension name
        models: List of model names to load

    Returns:
        DataFrame with columns: text_id, original_score, {model}_score for each model
    """
    dataset_dir = base_dir / dataset_name

    # Dictionary to store data by text_id
    data_by_id = {}

    for model in models:
        file_path = dataset_dir / f"results_{dataset_name}_{model}_{dimension}.json"

        if not file_path.exists():
            print(f"  Warning: File not found: {file_path.name}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data['detailed_results']:
            text_id = item['text_id']

            if text_id not in data_by_id:
                data_by_id[text_id] = {
                    'text_id': text_id,
                    'original_score': item.get('original_score')
                }

            # Store this model's score (handle different formats)
            if 'evaluation' in item['evaluation']:
                llm_score = item['evaluation']['evaluation']['score']
            else:
                llm_score = item['evaluation']['score']
            data_by_id[text_id][f'{model}_score'] = llm_score

    # Convert to DataFrame
    df = pd.DataFrame(list(data_by_id.values()))

    return df


def calculate_inter_model_metrics(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """
    Calculate inter-model disagreement and model-human disagreement metrics.

    Args:
        df: DataFrame with text_id, original_score, and model scores
        models: List of model names

    Returns:
        DataFrame with added columns: inter_model_std, avg_model_human_diff, etc.
    """
    results = []

    for idx, row in df.iterrows():
        text_id = row['text_id']
        human_score = row['original_score']

        # Skip if no human score
        if pd.isna(human_score):
            continue

        # Collect model scores
        model_scores = []
        model_human_diffs = []

        for model in models:
            score_col = f'{model}_score'
            if score_col in row and not pd.isna(row[score_col]):
                model_score = row[score_col]
                model_scores.append(model_score)
                model_human_diffs.append(abs(model_score - human_score))

        # Need at least 2 models to calculate inter-model std
        if len(model_scores) < 2:
            continue

        # Calculate metrics
        inter_model_std = np.std(model_scores, ddof=1)  # Sample std
        inter_model_range = max(model_scores) - min(model_scores)
        avg_model_score = np.mean(model_scores)
        avg_model_human_diff = np.mean(model_human_diffs)

        results.append({
            'text_id': text_id,
            'human_score': human_score,
            'inter_model_std': inter_model_std,
            'inter_model_range': inter_model_range,
            'avg_model_score': avg_model_score,
            'avg_model_human_diff': avg_model_human_diff,
            'n_models': len(model_scores)
        })

    return pd.DataFrame(results)


def analyze_dimension(base_dir: Path, dataset_name: str, dimension: str) -> pd.DataFrame:
    """
    Analyze inter-model vs model-human disagreement for a dimension.

    Args:
        base_dir: Base directory
        dataset_name: Dataset name
        dimension: Dimension name

    Returns:
        DataFrame with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, Dimension: {dimension}")
    print(f"{'='*60}")

    # Get available models
    models = get_available_models(base_dir, dataset_name, dimension)

    if len(models) < 2:
        print(f"  Insufficient models found ({len(models)}). Need at least 2.")
        return pd.DataFrame()

    print(f"  Found {len(models)} models: {', '.join(models)}")

    # Load scores
    df = load_model_scores(base_dir, dataset_name, dimension, models)

    if df.empty:
        print(f"  No data loaded")
        return pd.DataFrame()

    print(f"  Loaded {len(df)} items")

    # Calculate metrics
    metrics_df = calculate_inter_model_metrics(df, models)

    if metrics_df.empty:
        print(f"  No valid items after metric calculation")
        return pd.DataFrame()

    print(f"  Calculated metrics for {len(metrics_df)} items")
    print(f"  Inter-model std range: [{metrics_df['inter_model_std'].min():.4f}, {metrics_df['inter_model_std'].max():.4f}]")
    print(f"  Avg model-human diff range: [{metrics_df['avg_model_human_diff'].min():.4f}, {metrics_df['avg_model_human_diff'].max():.4f}]")

    # Calculate correlation
    if len(metrics_df) >= 3:
        pearson_r, pearson_p = stats.pearsonr(metrics_df['inter_model_std'],
                                               metrics_df['avg_model_human_diff'])
        spearman_r, spearman_p = stats.spearmanr(metrics_df['inter_model_std'],
                                                  metrics_df['avg_model_human_diff'])

        print(f"\n  Correlation Results:")
        print(f"    Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
        print(f"    Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.6f})")

        # Add metadata
        metrics_df['dataset'] = dataset_name
        metrics_df['dimension'] = dimension
        metrics_df['pearson_r'] = pearson_r
        metrics_df['pearson_p'] = pearson_p
        metrics_df['spearman_r'] = spearman_r
        metrics_df['spearman_p'] = spearman_p

    return metrics_df


def plot_inter_model_vs_human(df: pd.DataFrame, output_dir: Path,
                              title_suffix: str = ""):
    """
    Create scatter plot of inter-model disagreement vs model-human disagreement.

    Args:
        df: DataFrame with metrics
        output_dir: Output directory
        title_suffix: Optional suffix for title and filename
    """
    if df.empty or len(df) < 3:
        print(f"  Skipping plot: insufficient data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Scatter with regression
    ax1.scatter(df['inter_model_std'], df['avg_model_human_diff'],
               alpha=0.3, s=20, color='steelblue')

    # Regression line
    z = np.polyfit(df['inter_model_std'], df['avg_model_human_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['inter_model_std'].min(), df['inter_model_std'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2,
            label=f'y = {z[0]:.3f}x + {z[1]:.3f}')

    # Calculate correlation
    pearson_r, pearson_p = stats.pearsonr(df['inter_model_std'], df['avg_model_human_diff'])
    spearman_r, spearman_p = stats.spearmanr(df['inter_model_std'], df['avg_model_human_diff'])

    ax1.set_xlabel('Inter-Model Disagreement (Std Dev)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg Model-Human Disagreement', fontsize=12, fontweight='bold')
    ax1.set_title(f'Inter-Model vs Model-Human Disagreement\n{title_suffix}',
                 fontsize=14, fontweight='bold')

    # Add correlation info
    text = f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})\n"
    text += f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})\n"
    text += f"n = {len(df)}"
    ax1.text(0.05, 0.95, text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Binned analysis
    bins = pd.qcut(df['inter_model_std'], q=10, duplicates='drop')
    binned_data = df.groupby(bins).agg({
        'avg_model_human_diff': ['mean', 'std', 'count'],
        'inter_model_std': 'mean'
    }).reset_index(drop=True)

    binned_data.columns = ['_'.join(col).strip('_') for col in binned_data.columns]

    ax2.errorbar(binned_data['inter_model_std_mean'],
                binned_data['avg_model_human_diff_mean'],
                yerr=binned_data['avg_model_human_diff_std'],
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                color='darkgreen', label='Mean ± SD')

    ax2.set_xlabel('Inter-Model Std (binned)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Model-Human Diff', fontsize=12, fontweight='bold')
    ax2.set_title('Binned Analysis\n(10 quantiles)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save
    filename = f"inter_model_vs_human_{title_suffix.replace(' ', '_').replace(',', '')}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  Saved plot: {filename}")

    plt.close()


def main():
    """Main analysis function."""

    base_dir = Path("/share/pi/nigam/users/aunell/SmartSample_local/data/judge_scores")
    output_dir = Path(DEFAULT_PLOTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define datasets and dimensions
    datasets_config = {
        'summeval': ['coherence', 'consistency', 'fluency', 'relevance'],
        'hanna': ['Coherence', 'Complexity', 'Empathy', 'Engagement', 'Relevance', 'Surprise'],
        'mslr': ['fluency', 'intervention', 'outcome', 'population'],
    }

    print("="*80)
    print("INTER-MODEL DISAGREEMENT vs MODEL-HUMAN DISAGREEMENT ANALYSIS")
    print("="*80)
    print("\nResearch Question:")
    print("When LLM judges disagree with each other, are they more likely to")
    print("disagree with human ratings?")

    all_results = []
    correlation_summary = []

    # Analyze each dataset and dimension
    for dataset_name, dimensions in datasets_config.items():
        for dimension in dimensions:
            metrics_df = analyze_dimension(base_dir, dataset_name, dimension)

            if not metrics_df.empty:
                all_results.append(metrics_df)

                # Plot for this dimension
                plot_inter_model_vs_human(metrics_df, output_dir,
                                         f"{dataset_name}_{dimension}")

                # Store correlation for summary
                if 'pearson_r' in metrics_df.columns:
                    correlation_summary.append({
                        'dataset': dataset_name,
                        'dimension': dimension,
                        'n': len(metrics_df),
                        'pearson_r': metrics_df['pearson_r'].iloc[0],
                        'pearson_p': metrics_df['pearson_p'].iloc[0],
                        'spearman_r': metrics_df['spearman_r'].iloc[0],
                        'spearman_p': metrics_df['spearman_p'].iloc[0]
                    })

    # Combine all results
    if all_results:
        print(f"\n{'='*80}")
        print("COMBINED ANALYSIS (ALL DATASETS & DIMENSIONS)")
        print(f"{'='*80}")

        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal items: {len(combined_df)}")
        print(f"Datasets: {combined_df['dataset'].nunique()}")
        print(f"Dimensions: {combined_df['dimension'].nunique()}")

        # Overall correlation
        pearson_r, pearson_p = stats.pearsonr(combined_df['inter_model_std'],
                                              combined_df['avg_model_human_diff'])
        spearman_r, spearman_p = stats.spearmanr(combined_df['inter_model_std'],
                                                  combined_df['avg_model_human_diff'])

        print(f"\nOverall Correlation:")
        print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
        print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.6f})")

        # Overall plot
        plot_inter_model_vs_human(combined_df, output_dir, "All_Combined")

        # Save combined data
        combined_df.to_csv(output_dir / "combined_data.csv", index=False)

        # Save correlation summary
        if correlation_summary:
            corr_df = pd.DataFrame(correlation_summary)
            corr_df.to_csv(output_dir / "correlation_summary.csv", index=False)

            print(f"\n{'='*80}")
            print("CORRELATION SUMMARY BY DIMENSION")
            print(f"{'='*80}")
            print(corr_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
