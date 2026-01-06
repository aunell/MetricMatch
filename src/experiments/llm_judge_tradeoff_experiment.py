"""
Experiment: LLM Judge vs Human Raters ICC Estimation Error with Fixed Budget

This experiment simulates a scenario where:
- We have a fixed budget (300 annotations)
- We can choose between fewer raters with more samples each, or more raters with fewer samples
- Human raters are drawn from one distribution
- LLM judge is drawn from a shifted distribution
- We measure the ICC estimation error for different numbers of raters
- We vary the disagreement level between raters

Key parameters:
- Budget: 300 annotations total
- True number of items: 300
- For n raters: each rater annotates 300/n items
- True ICC: Measured between all 10 human annotators vs the LLM judge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Tuple, Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.calculate_icc import calculate_icc

def generate_human_ratings(n_items: int, n_raters: int, true_icc: float,
                          random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ratings from human raters all drawn from the same distribution.

    Parameters:
    -----------
    n_items : int
        Number of items to rate
    n_raters : int
        Number of human raters
    true_icc : float
        Target ICC among human raters (controls disagreement)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    ratings : np.ndarray
        Array of shape (n_items, n_raters) with ratings from 1-5
    true_item_quality : np.ndarray
        Array of shape (n_items,) with true item quality scores
    """
    if random_state is not None:
        np.random.seed(random_state)

    # ICC = between_var / (between_var + within_var)
    # Set total variance to 1
    total_var = 1.0
    between_var = true_icc * total_var  # Item variance (true differences)
    within_var = (1 - true_icc) * total_var  # Rater variance (disagreement)

    # Generate true item qualities (what all raters are trying to measure)
    true_item_quality = np.random.normal(0, np.sqrt(between_var), n_items)

    # Generate ratings for each rater
    ratings = np.zeros((n_items, n_raters))
    for r in range(n_raters):
        # Each rater has some error in perceiving the true quality
        rater_error = np.random.normal(0, np.sqrt(within_var), n_items)
        additional_error = np.random.random()
        ratings[:, r] = 5.0 + true_item_quality + rater_error+additional_error

    # Discretize to 1-5 scale
    ratings = np.clip(np.round(ratings), 1, 5)

    return ratings, true_item_quality


def generate_llm_ratings(n_items: int, true_item_quality: np.ndarray,
                        shift: float = 0.5, target_llm_human_icc: float = 0.7,
                        between_var: float = None,
                        random_state: int = None) -> np.ndarray:
    """
    Generate ratings from LLM judge drawn from a shifted distribution.

    Parameters:
    -----------
    n_items : int
        Number of items to rate
    true_item_quality : np.ndarray
        True quality of each item (shared with human raters)
    shift : float
        Mean shift of LLM distribution (e.g., 0.5 means LLM rates 0.5 points higher on average)
    target_llm_human_icc : float
        Target ICC between LLM and average human ratings (0-1)
        Higher values mean LLM is more consistent with human consensus
    between_var : float
        Between-item variance (from human rating generation)
        Used to calibrate LLM noise to achieve target ICC
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    ratings : np.ndarray
        Array of shape (n_items,) with LLM ratings from 1-5
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate LLM noise to achieve target ICC
    # ICC = between_var / (between_var + within_var)
    # within_var = between_var * (1/ICC - 1)
    if between_var is not None and target_llm_human_icc > 0:
        llm_within_var = between_var * (1.0 / target_llm_human_icc - 1)
        noise_scale = np.sqrt(llm_within_var)
    else:
        # Fallback to default noise
        noise_scale = 0.3

    # LLM perceives the same true quality but with a systematic shift and noise
    llm_error = np.random.normal(0, noise_scale, n_items)
    ratings = 5.0 + true_item_quality + shift + llm_error

    # Discretize to 1-5 scale
    ratings = np.clip(np.round(ratings), 1, 5)

    return ratings


def calculate_true_icc_human_vs_llm(human_ratings: np.ndarray,
                                    llm_ratings: np.ndarray) -> float:
    """
    Calculate the true ICC between all human raters and the LLM judge.

    This is done by averaging all human ratings to get a consensus,
    then calculating ICC between the consensus and the LLM.

    Parameters:
    -----------
    human_ratings : np.ndarray
        Array of shape (n_items, n_raters)
    llm_ratings : np.ndarray
        Array of shape (n_items,)

    Returns:
    --------
    icc : float
        ICC between averaged human ratings and LLM ratings
    """
    # Average across all human raters to get consensus rating for each item
    avg_human_ratings = np.mean(human_ratings, axis=1)

    # Calculate ICC between average human rating and LLM rating
    icc = calculate_icc(avg_human_ratings, llm_ratings)

    return icc


def estimate_icc_with_sample(human_ratings: np.ndarray,
                             llm_ratings: np.ndarray,
                             n_raters_to_use: int,
                             budget: int = 300,
                             random_state: int = None) -> float:
    """
    Estimate ICC using a subset of raters and items based on budget.

    All raters rate the SAME set of items (determined by budget/n_raters).

    Parameters:
    -----------
    human_ratings : np.ndarray
        Full array of shape (n_items, n_total_raters)
    llm_ratings : np.ndarray
        Full array of shape (n_items,)
    n_raters_to_use : int
        Number of human raters to use for estimation
    budget : int
        Total number of annotations available
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    estimated_icc : float
        ICC estimated from the sampled data
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Calculate number of items each rater can annotate
    n_items_per_rater = budget // n_raters_to_use

    # Randomly select which raters to use
    n_total_raters = human_ratings.shape[1]
    selected_raters = np.random.choice(n_total_raters, size=n_raters_to_use, replace=False)

    # Sample a SINGLE set of items that ALL raters will rate
    total_items = human_ratings.shape[0]
    sampled_items = np.random.choice(total_items, size=n_items_per_rater, replace=False)

    # Get the sampled human ratings for selected raters
    sampled_human_ratings = human_ratings[sampled_items][:, selected_raters]

    # Average across the selected raters to get consensus
    avg_human_sample = np.mean(sampled_human_ratings, axis=1)

    # Get LLM ratings for the same items
    llm_sample = llm_ratings[sampled_items]

    # Calculate ICC between averaged human ratings and LLM
    estimated_icc = calculate_icc(avg_human_sample, llm_sample)

    return estimated_icc


def run_single_trial(n_items: int = 300,
                     n_total_raters: int = 10,
                     true_human_icc: float = 0.7,
                     llm_shift: float = 0.5,
                     target_llm_human_icc: float = 0.7,
                     budget: int = 300,
                     rater_counts: List[int] = None,
                     random_state: int = None) -> Dict:
    """
    Run a single trial of the experiment.

    Returns a dictionary with true ICC and estimated ICCs for different rater counts.
    """
    if rater_counts is None:
        rater_counts = [1, 2, 3, 5, 6, 10, 15, 30, 50, 75, 100, 150, 300]

    if random_state is not None:
        np.random.seed(random_state)

    # Generate full dataset
    # First generate human ratings and get the true item quality
    human_ratings, true_item_quality = generate_human_ratings(n_items, n_total_raters, true_human_icc, random_state)

    # Calculate between-item variance for LLM generation
    total_var = 1.0
    between_var = true_human_icc * total_var

    # Generate LLM ratings based on the SAME true item quality that humans are rating
    llm_ratings = generate_llm_ratings(n_items, true_item_quality, llm_shift, target_llm_human_icc, between_var, random_state)

    # Calculate ICC between human raters (inter-rater reliability)
    human_rater_iccs = []
    for i in range(n_total_raters):
        for j in range(i + 1, n_total_raters):
            icc = calculate_icc(human_ratings[:, i], human_ratings[:, j])
            human_rater_iccs.append(icc)
    icc_between_raters = np.mean(human_rater_iccs)

    # Calculate true ICC between all human raters and LLM
    true_icc = calculate_true_icc_human_vs_llm(human_ratings, llm_ratings)

    # Print ICCs
    print(f"\nICC between human raters: {icc_between_raters:.4f}")
    print(f"ICC between human raters and LLM model: {true_icc:.4f}")

    # Estimate ICC for different numbers of raters
    results = {
        'true_icc': true_icc,
        'icc_between_raters': icc_between_raters,
        'true_human_icc': true_human_icc,
        'target_llm_human_icc': target_llm_human_icc,
        'llm_shift': llm_shift,
        'estimates': {}
    }

    for n_raters in rater_counts:
        if n_raters <= n_total_raters and budget // n_raters >= 10:  # Need at least 10 items per rater
            estimated_icc = estimate_icc_with_sample(
                human_ratings, llm_ratings, n_raters, budget, random_state
            )
            results['estimates'][n_raters] = estimated_icc

    return results


def run_experiment(n_trials: int = 50,
                   n_items: int = 300,
                   n_total_raters: int = 10,
                   disagreement_levels: List[float] = None,
                   llm_shift: float = 0.5,
                   target_llm_human_icc: float = 0.7,
                   budget: int = 300,
                   rater_counts: List[int] = None,
                   output_dir: str = 'results') -> pd.DataFrame:
    """
    Run the full experiment across multiple trials and disagreement levels.

    Parameters:
    -----------
    n_trials : int
        Number of trials to run for each configuration
    n_items : int
        Total number of items (true dataset size)
    n_total_raters : int
        Total number of human raters available
    disagreement_levels : List[float]
        Different ICC levels among human raters (lower ICC = more disagreement)
    llm_shift : float
        Systematic shift in LLM ratings
    target_llm_human_icc : float
        Target ICC between LLM and average human ratings
    budget : int
        Total annotation budget
    rater_counts : List[int]
        Different numbers of raters to test
    output_dir : str
        Directory to save results

    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with all results
    """
    if disagreement_levels is None:
        # Lower ICC = more disagreement
        disagreement_levels = [0.3, 0.5, 0.7, 0.9]

    if rater_counts is None:
        rater_counts = [1, 2, 3, 5, 6, 10, 15, 30, 50, 75, 100, 150, 300]

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    for disagreement_icc in disagreement_levels:
        print(f"\nRunning trials for human ICC = {disagreement_icc:.2f} (disagreement level)")

        for trial in range(n_trials):
            if trial % 10 == 0:
                print(f"  Trial {trial}/{n_trials}")

            trial_results = run_single_trial(
                n_items=n_items,
                n_total_raters=n_total_raters,
                true_human_icc=disagreement_icc,
                llm_shift=llm_shift,
                target_llm_human_icc=target_llm_human_icc,
                budget=budget,
                rater_counts=rater_counts,
                random_state=trial
            )

            # Store results for each rater count
            for n_raters, estimated_icc in trial_results['estimates'].items():
                all_results.append({
                    'trial': trial,
                    'human_icc': disagreement_icc,
                    'true_icc_human_llm': trial_results['true_icc'],
                    'target_llm_human_icc': target_llm_human_icc,
                    'n_raters': n_raters,
                    'n_items_per_rater': budget // n_raters,
                    'estimated_icc': estimated_icc,
                    'estimation_error': abs(estimated_icc - trial_results['true_icc']),
                    'llm_shift': llm_shift
                })

    results_df = pd.DataFrame(all_results)

    # Save raw results
    results_df.to_csv(f"{output_dir}/llm_judge_tradeoff_raw_results.csv", index=False)

    return results_df


def plot_results(results_df: pd.DataFrame, output_dir: str = 'results'):
    """
    Create visualization showing ICC estimation error vs number of raters.

    Creates a line plot with:
    - X-axis: Number of raters
    - Y-axis: Mean estimation error
    - Different lines for different disagreement levels (human ICC)
    """
    plt.figure(figsize=(12, 8))

    # Calculate mean and std error for each configuration
    summary = results_df.groupby(['human_icc', 'n_raters'])['estimation_error'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary = summary.reset_index()

    # Create line plot
    disagreement_levels = sorted(results_df['human_icc'].unique())
    colors = sns.color_palette("viridis", len(disagreement_levels))

    for i, human_icc in enumerate(disagreement_levels):
        data = summary[summary['human_icc'] == human_icc]
        plt.plot(data['n_raters'], data['mean'],
                marker='o', linewidth=2, markersize=8,
                color=colors[i], label=f'Human ICC = {human_icc:.1f}')

        # Add confidence bands (±1 SE)
        plt.fill_between(data['n_raters'],
                        data['mean'] - data['se'],
                        data['mean'] + data['se'],
                        alpha=0.2, color=colors[i])

    plt.xlabel('Number of Raters', fontsize=14, fontweight='bold')
    plt.ylabel('ICC Estimation Error (MAE)', fontsize=14, fontweight='bold')
    plt.title('Tradeoff: Number of Raters vs ICC Estimation Error\n(Fixed Budget = 300 annotations)',
             fontsize=16, fontweight='bold')
    plt.legend(title='Disagreement Level\n(among humans)', fontsize=11, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()

    # Save figure
    plt.savefig(f"{output_dir}/llm_judge_tradeoff_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/llm_judge_tradeoff_plot.pdf", bbox_inches='tight')
    print(f"\nPlot saved to {output_dir}/llm_judge_tradeoff_plot.png")

    plt.show()

    # Also create a plot showing the budget tradeoff more explicitly
    plt.figure(figsize=(12, 8))

    for i, human_icc in enumerate(disagreement_levels):
        data = summary[summary['human_icc'] == human_icc]
        plt.plot(data['n_raters'], data['mean'],
                marker='o', linewidth=2, markersize=8,
                color=colors[i], label=f'Human ICC = {human_icc:.1f}')

        # Add confidence bands
        plt.fill_between(data['n_raters'],
                        data['mean'] - data['se'],
                        data['mean'] + data['se'],
                        alpha=0.2, color=colors[i])

        # Add text annotations showing items per rater for key points
        for idx in [0, len(data)//2, len(data)-1]:
            if idx < len(data):
                row = data.iloc[idx]
                items_per_rater = 300 // row['n_raters']
                plt.annotate(f'{items_per_rater} items/rater',
                           xy=(row['n_raters'], row['mean']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    plt.xlabel('Number of Raters', fontsize=14, fontweight='bold')
    plt.ylabel('ICC Estimation Error (MAE)', fontsize=14, fontweight='bold')
    plt.title('Budget Allocation: More Raters vs More Items per Rater\n(Total Budget = 300 annotations)',
             fontsize=16, fontweight='bold')
    plt.legend(title='Disagreement Level\n(among humans)', fontsize=11, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()

    plt.savefig(f"{output_dir}/llm_judge_budget_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/llm_judge_budget_tradeoff.pdf", bbox_inches='tight')
    print(f"Budget tradeoff plot saved to {output_dir}/llm_judge_budget_tradeoff.png")

    plt.show()


def print_summary_statistics(results_df: pd.DataFrame):
    """Print summary statistics from the experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("="*80)

    for human_icc in sorted(results_df['human_icc'].unique()):
        data = results_df[results_df['human_icc'] == human_icc]
        print(f"\nHuman ICC = {human_icc:.2f} (Disagreement Level):")
        print("-" * 60)

        summary = data.groupby('n_raters')['estimation_error'].agg(['mean', 'std', 'min', 'max'])
        summary['n_items_per_rater'] = 300 // summary.index

        print(summary.to_string())

        # Find optimal number of raters (minimum error)
        optimal_idx = summary['mean'].idxmin()
        optimal_n_raters = optimal_idx
        optimal_items_per_rater = 300 // optimal_n_raters
        optimal_error = summary.loc[optimal_idx, 'mean']

        print(f"\nOptimal configuration:")
        print(f"  - Number of raters: {optimal_n_raters}")
        print(f"  - Items per rater: {optimal_items_per_rater}")
        print(f"  - Mean estimation error: {optimal_error:.4f}")


if __name__ == "__main__":
    # Set parameters
    N_TRIALS = 5
    N_ITEMS = 300
    N_TOTAL_RATERS = 20
    BUDGET = 300
    DISAGREEMENT_LEVELS = [0.3, 0.5, 0.7, 0.9]  # Lower = more disagreement
    LLM_SHIFT = 0.5  # LLM systematically rates 0.5 points higher
    TARGET_LLM_HUMAN_ICC = 0.7  # Target ICC between LLM and average human rating
    # Rater counts must be <= N_TOTAL_RATERS
    RATER_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 20]

    OUTPUT_DIR = 'results/llm_judge_tradeoff'

    print("="*80)
    print("LLM JUDGE VS HUMAN RATERS: ICC ESTIMATION ERROR EXPERIMENT")
    print("="*80)
    print(f"\nExperiment Configuration:")
    print(f"  - Number of trials: {N_TRIALS}")
    print(f"  - True number of items: {N_ITEMS}")
    print(f"  - Total annotation budget: {BUDGET}")
    print(f"  - Total human raters available: {N_TOTAL_RATERS}")
    print(f"  - LLM systematic shift: {LLM_SHIFT}")
    print(f"  - Target LLM-Human ICC: {TARGET_LLM_HUMAN_ICC}")
    print(f"  - Human disagreement levels (ICC): {DISAGREEMENT_LEVELS}")
    print(f"  - Rater counts to test: {RATER_COUNTS}")
    print(f"  - Output directory: {OUTPUT_DIR}")

    # Run experiment
    print("\nRunning experiment...")
    results_df = run_experiment(
        n_trials=N_TRIALS,
        n_items=N_ITEMS,
        n_total_raters=N_TOTAL_RATERS,
        disagreement_levels=DISAGREEMENT_LEVELS,
        llm_shift=LLM_SHIFT,
        target_llm_human_icc=TARGET_LLM_HUMAN_ICC,
        budget=BUDGET,
        rater_counts=RATER_COUNTS,
        output_dir=OUTPUT_DIR
    )

    # Print summary statistics
    print_summary_statistics(results_df)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(results_df, OUTPUT_DIR)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - Raw data: llm_judge_tradeoff_raw_results.csv")
    print(f"  - Main plot: llm_judge_tradeoff_plot.png")
    print(f"  - Budget plot: llm_judge_budget_tradeoff.png")
