import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "/Users/alyssaunell/code/SmartSample_local/results/04_01_downstream_task_small_ensemble_and_target"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "rho", "tau"]
OUR_METHOD = "variance_matched_weighted_.9"
BASELINE = "random"
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
BASELINE_BUDGET = 50


def load_data(results_dir=RESULTS_DIR):
    frames = []
    for dataset in DATASETS:
        for metric in METRICS:
            csv_path = os.path.join(
                results_dir, dataset, dataset, "dataframes", f"{metric}_results.csv"
            )
            if not os.path.exists(csv_path):
                print(f"  Skipping {metric} | {dataset}: file not found")
                continue
            df = pd.read_csv(csv_path)
            df = df[df["method"].isin([OUR_METHOD, BASELINE])][
                ["model", "budget", "method", "estimation_error", "axis"]
            ]
            df["dataset"] = dataset
            df["metric"] = metric
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_annotations_saved(df):
    """
    For each (dataset, axis, model, metric):
      1. Compute random's mean error at BASELINE_BUDGET (averaged over 100 runs).
      2. For each budget of our method, compute mean error over 100 runs.
      3. Find the smallest budget where our mean error <= random's baseline error.
      4. annotations_saved = BASELINE_BUDGET - that budget.
         If our method never achieves the target, annotations_saved = NaN.
    Returns a detail DataFrame with one row per (dataset, axis, model, metric).
    """
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "method", "metric"])["estimation_error"]
        .mean()
        .reset_index()
        .rename(columns={"estimation_error": "mean_error"})
    )

    random_all = avg[avg["method"] == BASELINE][
        ["dataset", "axis", "model", "budget", "metric", "mean_error"]
    ].rename(columns={"mean_error": "random_error"})

    random_baseline = random_all[random_all["budget"] == BASELINE_BUDGET][
        ["dataset", "axis", "model", "metric", "random_error"]
    ].rename(columns={"random_error": "random_error_50"})

    ours = avg[avg["method"] == OUR_METHOD][
        ["dataset", "axis", "model", "budget", "metric", "mean_error"]
    ].rename(columns={"mean_error": "our_error"})

    combined = ours.merge(random_baseline, on=["dataset", "axis", "model", "metric"])
    combined["beats_baseline"] = combined["our_error"] <= combined["random_error_50"]

    # Also prepare random at all budgets for the reverse calculation
    our_baseline = avg[
        (avg["method"] == OUR_METHOD) & (avg["budget"] == BASELINE_BUDGET)
    ][["dataset", "axis", "model", "metric", "mean_error"]].rename(
        columns={"mean_error": "our_error_50"}
    )
    random_combined = random_all.merge(our_baseline, on=["dataset", "axis", "model", "metric"])
    random_combined["random_beats_ours"] = random_combined["random_error"] <= random_combined["our_error_50"]

    records = []
    group_cols = ["dataset", "axis", "model", "metric"]
    for key, grp in combined.groupby(group_cols):
        grp_sorted = grp.sort_values("budget")
        random_err_50 = grp_sorted["random_error_50"].iloc[0]
        our_err_50 = grp_sorted.loc[grp_sorted["budget"] == BASELINE_BUDGET, "our_error"].iloc[0]

        # Our method: minimum budget to match random@50
        beats = grp_sorted[grp_sorted["beats_baseline"]]
        if not beats.empty:
            min_budget_needed = beats["budget"].min()
            saved = BASELINE_BUDGET - min_budget_needed
        else:
            min_budget_needed = np.nan
            saved = np.nan

        # Random method: minimum budget to match our_method@50 (for losing combos)
        rgrp = random_combined[
            (random_combined["dataset"] == key[0]) &
            (random_combined["axis"] == key[1]) &
            (random_combined["model"] == key[2]) &
            (random_combined["metric"] == key[3])
        ].sort_values("budget")
        r_beats = rgrp[rgrp["random_beats_ours"]]
        if not r_beats.empty:
            random_min_budget = r_beats["budget"].min()
            random_saved = BASELINE_BUDGET - random_min_budget
        else:
            random_min_budget = np.nan
            random_saved = np.nan

        records.append({
            "dataset": key[0],
            "axis": key[1],
            "model": key[2],
            "metric": key[3],
            "random_error_50": random_err_50,
            "our_error_50": our_err_50,
            "budget_needed": min_budget_needed,
            "annotations_saved": saved,
            "random_budget_needed": random_min_budget,
            "random_annotations_saved": random_saved,
        })

    return pd.DataFrame(records)


def summary_table(detail, index_col, col_order=None):
    """Mean annotations saved (excluding NaN) pivoted by metric."""
    tbl = (
        detail.groupby([index_col, "metric"])["annotations_saved"]
        .mean()
        .reset_index()
        .pivot(index=index_col, columns="metric", values="annotations_saved")
    )
    if col_order:
        tbl = tbl[col_order]
    tbl.columns.name = None
    tbl["mean"] = tbl.mean(axis=1)
    return tbl


def plot_distribution(detail, out_path, col="annotations_saved", title_suffix="", subtitle=""):
    """Bar plot of annotation savings distribution in bins of 10."""
    """Bar plot of annotation savings distribution in bins of 10."""
    bins = [0, 10, 20, 30, 40, 50]
    labels = ["0–10", "10–20", "20–30", "30–40", "40–50"]

    saved = detail[col].dropna()
    never_achieved = detail[col].isna().sum()

    counts = pd.cut(saved, bins=bins, right=False, labels=labels).value_counts().reindex(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, counts.values, color="steelblue", edgecolor="white", linewidth=0.5)

    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Annotations Saved", fontsize=12)
    ax.set_ylabel("Number of Combos", fontsize=12)
    ax.set_title(
        f"Distribution of Annotations Saved{title_suffix}\n"
        f"{subtitle}\n"
        f"Never achieved target: {never_achieved} combos",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main(results_dir=RESULTS_DIR):
    out_dir = os.path.join(results_dir, "annotations_saved")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(results_dir)

    print("Computing annotations saved...")
    detail = compute_annotations_saved(df)

    never_achieved = detail["annotations_saved"].isna().sum()
    print(f"Combos where our method never reached random@50 error: {never_achieved} / {len(detail)}")

    # Save full detail
    detail_path = os.path.join(out_dir, "annotations_saved_detail.csv")
    detail.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    # Summary by dataset x metric
    available_metrics = [m for m in METRICS if m in detail["metric"].unique()]
    by_dataset = summary_table(detail, "dataset", col_order=available_metrics)
    ds_path = os.path.join(out_dir, "annotations_saved_by_dataset.csv")
    by_dataset.to_csv(ds_path)
    print(f"\nSaved: {ds_path}")
    print(by_dataset.round(2).to_string())

    # Summary by model x metric
    by_model = summary_table(detail, "model", col_order=available_metrics)
    model_path = os.path.join(out_dir, "annotations_saved_by_model.csv")
    by_model.to_csv(model_path)
    print(f"\nSaved: {model_path}")
    print(by_model.round(2).to_string())

    # Overall summary
    overall = pd.DataFrame([{
        "total_combos": len(detail),
        "combos_with_savings": detail["annotations_saved"].notna().sum(),
        "never_achieved_target": never_achieved,
        "mean_annotations_saved": detail["annotations_saved"].mean(),
        "median_annotations_saved": detail["annotations_saved"].median(),
    }])
    overall_path = os.path.join(out_dir, "annotations_saved_overall.csv")
    overall.to_csv(overall_path, index=False)
    print(f"\nSaved: {overall_path}")
    print(overall.round(2).to_string())

    # Per-metric summary
    per_metric = (
        detail.groupby("metric")["annotations_saved"]
        .agg(
            mean="mean",
            median="median",
            combos_with_savings=lambda x: x.notna().sum(),
            never_achieved=lambda x: x.isna().sum(),
        )
        .round(2)
    )
    metric_path = os.path.join(out_dir, "annotations_saved_by_metric.csv")
    per_metric.to_csv(metric_path)
    print(f"\nSaved: {metric_path}")
    print(per_metric.to_string())

    # Bar plot — our method wins
    plot_distribution(
        detail,
        os.path.join(out_dir, "annotations_saved_distribution.jpg"),
        col="annotations_saved",
        title_suffix=" — Our Method vs Random@50",
        subtitle=f"variance_matched_weighted_.9 matches random at budget 50",
    )

    # Bar plot — random wins (63 losing combos): how many annotations random saves vs our_method@50
    losing = detail[detail["annotations_saved"].isna()].copy()
    plot_distribution(
        losing,
        os.path.join(out_dir, "annotations_saved_distribution_losing.jpg"),
        col="random_annotations_saved",
        title_suffix=" — Random vs Our Method@50 (losing combos)",
        subtitle=f"Random matches variance_matched_weighted_.9 at budget 50 ({len(losing)} combos)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotations saved analysis")
    parser.add_argument("--folder", default=RESULTS_DIR, help="Results directory")
    args = parser.parse_args()
    main(results_dir=args.folder)
