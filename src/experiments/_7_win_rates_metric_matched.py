import argparse
import os
import pandas as pd
import numpy as np

DATA_DIR = "/Users/alyssaunell/code/SmartSample_local/data/metric_matched_subsets"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "kendalltau", "mean_sq_error", "spearman"]
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metric_matched_data(data_dir=DATA_DIR):
    """Load dataset_combined_results2.csv for each dataset and return combined df."""
    frames = []
    for dataset in DATASETS:
        csv_path = os.path.join(data_dir, f"{dataset}_combined_results2.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {dataset}: file not found")
            continue
        df = pd.read_csv(csv_path)
        # Add dataset column
        df["dataset"] = dataset
        frames.append(df)

    if not frames:
        raise ValueError("No data files found")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def prepare_data(df):
    """
    Filter data for our method vs random comparison.
    Our method: method='metric_match' AND match_metric==est_metric
    Baseline: method='random'
    """
    # Filter for our method: metric_match where match_metric == est_metric
    our_method = df[
        (df["method"] == "metric_match") &
        (df["match_metric"] == df["est_metric"])
    ].copy()
    our_method["comparison_method"] = "metric_match_aligned"

    # Filter for baseline: random
    baseline = df[df["method"] == "random"].copy()
    baseline["comparison_method"] = "random"

    # Combine both
    combined = pd.concat([our_method, baseline], ignore_index=True)

    # Rename columns to match original script naming
    combined = combined.rename(columns={
        "est_error": "estimation_error",
        "est_metric": "metric"
    })

    # Select relevant columns
    combined = combined[["dataset", "model", "budget", "comparison_method",
                        "estimation_error", "axis", "metric"]]

    return combined


def assign_run_index(df):
    """Assign positional run index within each (dataset, metric, axis, model, budget, comparison_method) group."""
    df = df.sort_values(
        ["dataset", "metric", "axis", "model", "budget", "comparison_method"]
    ).copy()
    df["run"] = df.groupby(
        ["dataset", "metric", "axis", "model", "budget", "comparison_method"]
    ).cumcount()
    return df


# ---------------------------------------------------------------------------
# Win rate calculation - Macro
# ---------------------------------------------------------------------------

def _macro_merge(df):
    """Average over runs then compare our method vs random."""
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "comparison_method", "metric"])[
            "estimation_error"
        ]
        .mean()
        .reset_index()
    )
    ours = avg[avg["comparison_method"] == "metric_match_aligned"].rename(
        columns={"estimation_error": "our_err"}
    )
    base = avg[avg["comparison_method"] == "random"].rename(
        columns={"estimation_error": "random_err"}
    )
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


# ---------------------------------------------------------------------------
# Win rate calculation - Micro
# ---------------------------------------------------------------------------

def _micro_merge(df):
    """Pair runs by position then compare our method vs random."""
    df = assign_run_index(df)
    ours = df[df["comparison_method"] == "metric_match_aligned"].rename(
        columns={"estimation_error": "our_err"}
    )
    base = df[df["comparison_method"] == "random"].rename(
        columns={"estimation_error": "random_err"}
    )
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


# ---------------------------------------------------------------------------
# Pivoting and formatting
# ---------------------------------------------------------------------------

def _estimation_pivot(merged):
    """Create pivot table of win rates by budget and metric."""
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
    """Save dataframe and print results."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved: {path}")
    print(df.round(3).to_string())
    print()


def main(data_dir=DATA_DIR, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(data_dir, "win_rates")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading metric-matched data...")
    raw_df = load_metric_matched_data(data_dir)

    print("Preparing data for comparison (metric_match_aligned vs random)...")
    df = prepare_data(raw_df)

    print(f"\nData summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Datasets: {df['dataset'].unique()}")
    print(f"  Methods: {df['comparison_method'].unique()}")
    print(f"  Metrics: {df['metric'].unique()}")
    print(f"  Budgets: {sorted(df['budget'].unique())}")

    # -- Macro win rates --
    print("\n--- Macro estimation win rates ---")
    macro_merged = _macro_merge(df)
    save(_estimation_pivot(macro_merged), os.path.join(output_dir, "macro_win_rates.csv"))

    for dataset in DATASETS:
        dataset_data = macro_merged[macro_merged["dataset"] == dataset]
        if len(dataset_data) > 0:
            save(
                _estimation_pivot(dataset_data),
                os.path.join(output_dir, f"macro_win_rates_{dataset}.csv"),
            )

    # -- Micro win rates --
    print("\n--- Micro estimation win rates ---")
    micro_merged = _micro_merge(df)
    save(_estimation_pivot(micro_merged), os.path.join(output_dir, "micro_win_rates.csv"))

    for dataset in DATASETS:
        dataset_data = micro_merged[micro_merged["dataset"] == dataset]
        if len(dataset_data) > 0:
            save(
                _estimation_pivot(dataset_data),
                os.path.join(output_dir, f"micro_win_rates_{dataset}.csv"),
            )

    # -- Summary: single win rate collapsed over all budgets, metrics, datasets, models --
    print("\n--- Summary win rates ---")
    micro_total = len(micro_merged)
    micro_wins  = micro_merged["win"].sum()
    macro_total = len(macro_merged)
    macro_wins  = macro_merged["win"].sum()

    summary = pd.DataFrame([
        {"avg":  "micro (all runs)",   "total_comparisons": micro_total, "wins": micro_wins, "win_rate": micro_wins / micro_total},
        {"avg":  "macro (avg runs)",   "total_comparisons": macro_total, "wins": macro_wins, "win_rate": macro_wins / macro_total},
    ]).set_index("avg")
    save(summary, os.path.join(output_dir, "summary_win_rates.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Win rates for metric-matched subsets (metric_match_aligned vs random)"
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Data directory with combined_results2.csv files")
    parser.add_argument("--output-dir", default=None, help="Output directory for win rates")
    args = parser.parse_args()
    main(data_dir=args.data_dir, output_dir=args.output_dir)
