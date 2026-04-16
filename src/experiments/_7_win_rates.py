import os
import pandas as pd
import numpy as np

RESULTS_DIR = "/share/pi/nigam/users/aunell/SmartSample_local/results/04_07_kendall_spearman"
THRESHOLD_DIR = "/share/pi/nigam/users/aunell/SmartSample_local/results/04_08_kendall_spearman_jqa_threshold/dataframes"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "rho", "tau"]
OUR_METHOD = "variance_matched_weighted_.9"
BASELINE = "random"
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def load_estimation_data():
    """Load {metric}_results.csv for each dataset and metric, return combined df."""
    frames = []
    for dataset in DATASETS:
        for metric in METRICS:
            csv_path = os.path.join(
                RESULTS_DIR, dataset, dataset, "dataframes", f"{metric}_results.csv"
            )
            df = pd.read_csv(csv_path)
            df = df[df["method"].isin([OUR_METHOD, BASELINE])][
                ["model", "budget", "method", "estimation_error", "axis"]
            ]
            df["dataset"] = dataset
            df["metric"] = metric
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


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
    result = result[METRICS]
    result.index.name = "budget"
    result.columns.name = None
    return result.sort_index()


def _macro_merge(df):
    """Average over runs then compare our method vs random."""
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "method", "metric"])[
            "estimation_error"
        ]
        .mean()
        .reset_index()
    )
    ours = avg[avg["method"] == OUR_METHOD].rename(columns={"estimation_error": "our_err"})
    base = avg[avg["method"] == BASELINE].rename(columns={"estimation_error": "random_err"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


def _micro_merge(df):
    """Pair runs by position then compare our method vs random."""
    df = assign_run_index(df)
    ours = df[df["method"] == OUR_METHOD].rename(columns={"estimation_error": "our_err"})
    base = df[df["method"] == BASELINE].rename(columns={"estimation_error": "random_err"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def load_threshold_raw_data():
    """Load raw metric results with predicted and true values for threshold analysis."""
    frames = []
    for dataset in DATASETS:
        for metric in METRICS:
            csv_path = os.path.join(
                RESULTS_DIR, dataset, dataset, "dataframes", f"{metric}_results.csv"
            )
            df = pd.read_csv(csv_path)
            df = df[df["method"].isin([OUR_METHOD, BASELINE])][
                ["model", "budget", "method", f"predicted_{metric}", f"true_{metric}", "axis"]
            ].rename(columns={f"predicted_{metric}": "predicted", f"true_{metric}": "true_val"})
            df["dataset"] = dataset
            df["metric"] = metric
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _classify(series, threshold):
    """Return boolean Series: True if value >= threshold."""
    return series >= threshold


def _threshold_micro_merge(df, threshold):
    """
    Per-run classification correctness, paired by position.
    correct = (predicted_class == true_class) for each individual run.
    Ties (both methods same) are discarded.
    """
    df = df.copy()
    df["correct"] = (_classify(df["predicted"], threshold) == _classify(df["true_val"], threshold)).astype(float)

    df = df.sort_values(["dataset", "metric", "axis", "model", "budget", "method"])
    df["run"] = df.groupby(["dataset", "metric", "axis", "model", "budget", "method"]).cumcount()

    ours = df[df["method"] == OUR_METHOD].rename(columns={"correct": "our_correct"})
    base = df[df["method"] == BASELINE].rename(columns={"correct": "random_correct"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])

    merged = merged[merged["our_correct"] != merged["random_correct"]].copy()
    merged["win"] = merged["our_correct"] > merged["random_correct"]
    return merged


def _threshold_macro_merge(df, threshold):
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

    ours = avg[avg["method"] == OUR_METHOD].rename(columns={"correct": "our_acc"})
    base = avg[avg["method"] == BASELINE].rename(columns={"correct": "random_acc"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])

    # Discard ties
    merged = merged[merged["our_acc"] != merged["random_acc"]].copy()
    merged["win"] = merged["our_acc"] > merged["random_acc"]
    return merged


def _threshold_pivot(merged):
    win_rates = (
        merged.groupby(["budget", "metric"])["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "win_rate"})
    )
    result = win_rates.pivot(index="budget", columns="metric", values="win_rate")
    result = result[METRICS]
    result.index.name = "budget"
    result.columns.name = None
    return result.sort_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save(df, path):
    df.to_csv(path)
    print(f"Saved: {path}")
    print(df.round(3).to_string())
    print()


def main(results_dir=RESULTS_DIR):
    est_dir = os.path.join(results_dir, "win_rates", "estimation")
    thr_dir = os.path.join(results_dir, "win_rates", "threshold")
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(thr_dir, exist_ok=True)

    # -- Estimation --
    print("Loading estimation data...")
    est_df = load_estimation_data()

    print("\n--- Macro estimation win rates ---")
    macro_merged = _macro_merge(est_df)
    save(_estimation_pivot(macro_merged), os.path.join(est_dir, "macro_win_rates_estimation.csv"))
    for dataset in DATASETS:
        save(
            _estimation_pivot(macro_merged[macro_merged["dataset"] == dataset]),
            os.path.join(est_dir, f"macro_win_rates_estimation_{dataset}.csv"),
        )

    print("\n--- Micro estimation win rates ---")
    micro_merged = _micro_merge(est_df)
    save(_estimation_pivot(micro_merged), os.path.join(est_dir, "micro_win_rates_estimation.csv"))
    for dataset in DATASETS:
        save(
            _estimation_pivot(micro_merged[micro_merged["dataset"] == dataset]),
            os.path.join(est_dir, f"micro_win_rates_estimation_{dataset}.csv"),
        )

    # -- Summary: single win rate collapsed over all budgets, metrics, datasets, models --
    print("\n--- Estimation summary win rates ---")
    micro_total = len(micro_merged)
    micro_wins  = micro_merged["win"].sum()
    macro_total = len(macro_merged)
    macro_wins  = macro_merged["win"].sum()
    summary = pd.DataFrame([
        {"avg":  "micro (all runs)",   "total_comparisons": micro_total, "wins": micro_wins, "win_rate": micro_wins / micro_total},
        {"avg":  "macro (avg runs)",   "total_comparisons": macro_total, "wins": macro_wins, "win_rate": macro_wins / macro_total},
    ]).set_index("avg")
    save(summary, os.path.join(est_dir, "summary_win_rates_estimation.csv"))

    # -- Threshold --
    print("Loading raw data for threshold analysis...")
    thr_raw = load_threshold_raw_data()

    for threshold in THRESHOLDS:
        t_str = f"T{threshold}"
        print(f"\nComputing threshold win rates ({t_str})...")
        macro_merged = _threshold_macro_merge(thr_raw, threshold)
        micro_merged = _threshold_micro_merge(thr_raw, threshold)

        print(f"\n--- Macro threshold win rates ({t_str}) ---")
        save(_threshold_pivot(macro_merged), os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}.csv"))
        for dataset in DATASETS:
            save(
                _threshold_pivot(macro_merged[macro_merged["dataset"] == dataset]),
                os.path.join(thr_dir, f"macro_win_rates_threshold_{t_str}_{dataset}.csv"),
            )

        print(f"\n--- Micro threshold win rates ({t_str}) ---")
        save(_threshold_pivot(micro_merged), os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}.csv"))
        for dataset in DATASETS:
            save(
                _threshold_pivot(micro_merged[micro_merged["dataset"] == dataset]),
                os.path.join(thr_dir, f"micro_win_rates_threshold_{t_str}_{dataset}.csv"),
            )


if __name__ == "__main__":
    main()
