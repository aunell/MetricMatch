import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.calculate_icc import calculate_icc
from utils.selection_strategies import *

# -------------------------
# CONFIG
# -------------------------
# Default plots output directory - change this to specify where plots should be saved
# This is used as the default when --images_dir is not provided
DEFAULT_PLOTS_DIR = "results/plots"  # Default: results/plots
# Alternative examples:
# DEFAULT_PLOTS_DIR = "01_11_plots"
# DEFAULT_PLOTS_DIR = "/path/to/custom/plots/directory"

# ------------------------
# Data Loading
# ------------------------
def load_data(data_path, categorical=False, model_name="gpt-4o", model_name_2=None):
    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    scores = {
        'expensive_score': [int(r['original_score']) if categorical else r['original_score']
                            for r in data['detailed_results']],
        'cheap_score': []
    }

    for result in data['detailed_results']:
        scores['cheap_score'].append(
            result.get('evaluation', {}).get('evaluation', {}).get('score',
            result.get('evaluation', {}).get('score', None))
        )

    size = 300
    cheap_ratings_2 = None

    # Attempt to load second set of cheap ratings if model_name_2 is provided
    if model_name_2:
        # Replace the model name in the path with model_name_2
        data_path_2 = data_path.replace(f"_{model_name}_", f"_{model_name_2}_")
    else:
        data_path_2 = None

    if data_path_2 and os.path.exists(data_path_2):
        print(f"Loading second set of ratings from {data_path_2}")
        with open(data_path_2) as f2:
            data_2 = json.load(f2)

        scores_2 = [
            r.get('evaluation', {}).get('evaluation', {}).get('score',
            r.get('evaluation', {}).get('score', None))
            for r in data_2['detailed_results']
        ]
        cheap_ratings_2 = np.array(scores_2[:300])
        size = min(len(scores['cheap_score']), len(scores['expensive_score']), len(cheap_ratings_2))
        cheap_ratings_2 = cheap_ratings_2[:size]

    return (
        np.array(scores['cheap_score'][:size]),
        np.array(scores['expensive_score'][:size]).astype(int),
        cheap_ratings_2
    )


# ------------------------
# Evaluation
# ------------------------
def evaluate_selection(cheap_ratings, expensive_ratings, selected_indices, strategy_name, oracle_icc=None):
    try:
        if not oracle_icc:
            oracle_icc = calculate_icc_fn(cheap_ratings, expensive_ratings)
        subset_icc = calculate_icc_fn(
            [cheap_ratings[i] for i in selected_indices],
            [expensive_ratings[i] for i in selected_indices]
        )
        preservation_error = abs(subset_icc - oracle_icc)

        return {
            'strategy': strategy_name,
            'selected_indices': selected_indices,
            'subset_icc': subset_icc,
            'oracle_icc': oracle_icc,
            'preservation_error': preservation_error,
            'n_selected': len(selected_indices)
        }, oracle_icc
    except Exception as e:
        print(f"Error in {strategy_name}: {e}")


# ------------------------
# Simulation
# ------------------------
def run_simulation(cheap_ratings, expensive_ratings, n_expensive_range,
                   n_rollouts=100, cheap_ratings_2=None):
    """
    Run simulation for multiple selection strategies.
    `seeds`: list of integer seeds for reproducibility. If None, use np.random default.
    """
    print("\nRunning simulation...")
    all_results = []

    strategies = {
        "Stratified QBC": hybrid_selection,
        "QBC": QBC_selection,
        'Random': random_selection,
        'Stratified': stratified_selection,
        'Cluster': cluster_selection,
        'Maximum-Variation': maximum_variation_selection,
        'Density-Based': density_based_selection,
        'Variance-Matching': variance_matching
    }

    if cheap_ratings_2 is None:
        for key in ["Stratified QBC", "QBC"]:
            strategies.pop(key, None)

    oracle_icc=None
    for n_expensive in tqdm(n_expensive_range, desc="Testing different n_expensive"):
        for seed in range(n_rollouts):
            for strategy_name, strategy_func in strategies.items():
                if strategy_name in {"Stratified QBC", "QBC"}:
                    selected = strategy_func(cheap_ratings, cheap_ratings_2, n_expensive, seed=seed)
                else:
                    selected = strategy_func(cheap_ratings, n_expensive, seed=seed)

                result, oracle_icc = evaluate_selection(cheap_ratings, expensive_ratings, selected, strategy_name, oracle_icc)
                if result:
                    # Record rollout, n_expensive, and seed
                    result.update({'rollout': seed, 'n_expensive': n_expensive, 'seed': seed})
                    all_results.append(result)

    return pd.DataFrame(all_results)

def save_strategy_csvs(df, out_dir="csv_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    for dataset in df["dataset_name"].unique():
        subset = df[df["dataset_name"] == dataset]

        # pivot mean
        mean_pivot = subset.pivot(
            index="n_expensive",
            columns="strategy",
            values="mean"
        )

        # pivot std and rename columns
        std_pivot = subset.pivot(
            index="n_expensive",
            columns="strategy",
            values="std"
        )
        std_pivot.columns = [f"{c}_std" for c in std_pivot.columns]

        # combine mean + std into one table
        final_df = mean_pivot.join(std_pivot)

        # save CSV
        out_path = os.path.join(out_dir, f"{dataset}.csv")
        final_df.to_csv(out_path, float_format="%.6f")
        print(f"Saved: {out_path}")


# ------------------------
# Plotting
# ------------------------
def plot_progression_results(all_dataset_results, show_seed_points=False, aggregate=True):
    print("\nPlotting progression results...")
    all_dataset_results['dataset_name'] = all_dataset_results['dataset'].str.extract(r'results_([^_]+)_')

    # Extract dimension from filename
    # Format: results_{dataset}_{model}_{dimension}
    # We need to get everything after the last underscore
    # Split by underscore and take the last part
    def extract_dimension(dataset_str):
        parts = dataset_str.split('_')
        # The dimension is the last part after splitting
        return parts[-1] if len(parts) > 3 else 'unknown'

    all_dataset_results['dimension'] = all_dataset_results['dataset'].apply(extract_dimension)

    dataset_names={"hanna": "HANNA",
                   "mslr": "MSLR",
                   "medval": "MedVAL",
                   "summeval": "SummEval"}

    if aggregate:
        # Original aggregated behavior - combine all dimensions
        print("Generating aggregated plots (all dimensions combined)")
        mean_results = (
            all_dataset_results
            .groupby(['dataset_name', 'strategy', 'n_expensive'])['preservation_error']
            .agg(['mean', 'std'])
            .reset_index()
        )

        for dataset in mean_results['dataset_name'].unique():
            if "oracle_icc" in all_dataset_results.columns:
                oracle_val = all_dataset_results.loc[
                    all_dataset_results['dataset_name'] == dataset, "oracle_icc"
                ].mean()  # Average across dimensions
                print(f"Dataset: {dataset} | Average Oracle ICC: {oracle_val:.3f}")

            plt.figure(figsize=(10, 6))
            dataset_data = mean_results[mean_results['dataset_name'] == dataset]

            for strategy in dataset_data['strategy'].unique():
                strategy_data = dataset_data[dataset_data['strategy'] == strategy]
                plt.plot(strategy_data['n_expensive'], strategy_data['mean'], label=strategy, marker='o')
                print(dataset, strategy_data['mean'])
                if show_seed_points:
                    seed_points = all_dataset_results[
                        (all_dataset_results['dataset_name'] == dataset) &
                        (all_dataset_results['strategy'] == strategy)
                    ]
                    plt.scatter(seed_points['n_expensive'], seed_points['preservation_error'],
                                alpha=0.3, s=20, color='gray')

            plt.title(
                f'Estimation Error vs Number of Expensive Ratings\nDataset: {dataset_names.get(dataset, dataset)} (Aggregated)',
                fontsize=20
            )
            plt.xlabel('Number of Expensive Ratings', fontsize=18)
            plt.ylabel('Estimation Error', fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            filename = f"progression_analysis_{dataset}_aggregated.png"
            plt.savefig(os.path.join(IMAGES_DIR, filename))
            print("SAVED FIG TO:", os.path.join(IMAGES_DIR, filename))
            plt.close()

    else:
        # Per-dimension behavior - separate plot for each dimension
        print("Generating per-dimension plots")
        mean_results = (
            all_dataset_results
            .groupby(['dataset_name', 'dimension', 'strategy', 'n_expensive'])['preservation_error']
            .agg(['mean', 'std'])
            .reset_index()
        )

        for dataset in mean_results['dataset_name'].unique():
            for dimension in mean_results[mean_results['dataset_name'] == dataset]['dimension'].unique():
                dimension_data = mean_results[
                    (mean_results['dataset_name'] == dataset) &
                    (mean_results['dimension'] == dimension)
                ]

                if "oracle_icc" in all_dataset_results.columns:
                    oracle_val = all_dataset_results.loc[
                        (all_dataset_results['dataset_name'] == dataset) &
                        (all_dataset_results['dimension'] == dimension), "oracle_icc"
                    ].iloc[0]
                    print(f"Dataset: {dataset} | Dimension: {dimension} | Oracle ICC: {oracle_val:.3f}")

                plt.figure(figsize=(10, 6))

                for strategy in dimension_data['strategy'].unique():
                    strategy_data = dimension_data[dimension_data['strategy'] == strategy]
                    plt.plot(strategy_data['n_expensive'], strategy_data['mean'], label=strategy, marker='o')
                    print(f"{dataset}/{dimension}:", strategy_data['mean'].values)
                    if show_seed_points:
                        seed_points = all_dataset_results[
                            (all_dataset_results['dataset_name'] == dataset) &
                            (all_dataset_results['dimension'] == dimension) &
                            (all_dataset_results['strategy'] == strategy)
                        ]
                        plt.scatter(seed_points['n_expensive'], seed_points['preservation_error'],
                                    alpha=0.3, s=20, color='gray')

                plt.title(
                    f'Estimation Error vs Number of Expensive Ratings\nDataset: {dataset_names.get(dataset, dataset)} - {dimension.capitalize()}',
                    fontsize=20
                )
                plt.xlabel('Number of Expensive Ratings', fontsize=18)
                plt.ylabel('Estimation Error', fontsize=18)
                plt.legend(fontsize=14)
                plt.grid(True)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)

                filename = f"progression_analysis_{dataset}_{dimension}.png"
                plt.savefig(os.path.join(IMAGES_DIR, filename))
                print("SAVED FIG TO:", os.path.join(IMAGES_DIR, filename))
                plt.close()

    mean_results.to_json(os.path.join(IMAGES_DIR, "progression_mean_results.json"),
                         orient="records", indent=2)
    return mean_results



# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICC calculation with configurable parameters.")
    parser.add_argument("--judge_model_type", type=str, default="openai", help="Judge model type (openai or anthropic)")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name (e.g., gpt-4o, gpt-4o-mini, claude-3.5-sonnet)")
    parser.add_argument("--model_name_2", type=str, default=None, help="Second model name for QBC strategies (optional)")
    parser.add_argument("--dataset_size", type=int, default=300, help="Number of samples in dataset")
    parser.add_argument("--n_rollouts", type=int, default=20, help="Number of rollouts")
    parser.add_argument("--judge_scores_dir", type=str, default="data/judge_scores", help="Directory for judge scores")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory for saving results")
    parser.add_argument("--aggregate", action="store_true", default=True, help="Aggregate dimensions (default: True)")
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false", help="Generate separate plots for each dimension")

    args = parser.parse_args()

    # ------------------------
    # Centralized File Locations
    # ------------------------
    DATE = datetime.now().strftime("%Y-%m-%d")
    judge_model_type = args.judge_model_type
    model_name = args.model_name
    model_name_2 = args.model_name_2
    dataset_size = args.dataset_size
    n_rollouts = args.n_rollouts
    JUDGE_SCORES_DIR = args.judge_scores_dir
    IMAGES_DIR = args.images_dir or DEFAULT_PLOTS_DIR

    calculate_icc_fn = calculate_icc

    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("Starting simulation experiment...")


    all_dataset_results = []

    print(f"Searching for files in: {JUDGE_SCORES_DIR}")
    print(f"Looking for files matching pattern: *_{model_name}_*.json")

    for root, dirs, files in os.walk(JUDGE_SCORES_DIR):
        for file in files:
            # Filter files that match the specified model_name
            if file.endswith(".json"):
                print(f"Found JSON file: {file}")
                if f"_{model_name}_" in file:
                    print(f"  -> Matches model name filter!")
                else:
                    print(f"  -> Does not match model name '{model_name}'")

            if file.endswith(".json") and f"_{model_name}_" in file:
                file_path = os.path.join(root, file)
                filename = file.split(".json")[0]  # Remove .json extension
                # The filename includes the full path with dimension, keep it all
                results_file = os.path.join(IMAGES_DIR, f"{os.path.basename(filename)}_simulation_results.csv")

                if not os.path.exists(results_file):
                    print(f"\nProcessing {file}")
                    cheap_ratings, expensive_ratings, cheap_ratings_2 = load_data(
                        file_path, model_name=model_name, model_name_2=model_name_2
                    )

                    total_annotations = 100 
                    print("len(expensive_ratings)", len(expensive_ratings) )
                    step = total_annotations//10
                    n_expensive_range = range(step, total_annotations, step)

                    simulation_results = run_simulation(
                        cheap_ratings, expensive_ratings,
                        n_expensive_range, n_rollouts,
                        cheap_ratings_2=cheap_ratings_2)
                    simulation_results['dataset'] = filename
                    simulation_results.to_csv(results_file, index=False)
                    all_dataset_results.append(simulation_results)
                else:
                    print(f"\nSkipping {file} - results already exist")
                    simulation_results = pd.read_csv(results_file)
                    all_dataset_results.append(simulation_results)

    if len(all_dataset_results) == 0:
        print(f"\nERROR: No files found matching pattern '*_{model_name}_*.json' in {JUDGE_SCORES_DIR}")
        print(f"Please check:")
        print(f"  1. The judge_scores_dir path is correct")
        print(f"  2. The model_name '{model_name}' matches the file naming convention")
        print(f"  3. JSON files exist in the directory")
        exit(1)

    all_dataset_results = pd.concat(all_dataset_results, ignore_index=True)
    plot_progression_results(all_dataset_results, aggregate=args.aggregate)
    print("Simulation experiment complete!")