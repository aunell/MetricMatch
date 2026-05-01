import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/Users/alyssaunell/code/SmartSample_local/data/metric_matched_subsets"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "kendalltau", "mean_sq_error", "spearman"]
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
PLOT_BUDGETS = [b for b in BUDGETS if b >= 10]  # rb=5 excluded: no vm budgets below 5 to observe
BASELINE_BUDGET = 50
OUR_METHOD = "metric_match_aligned"


def load_metric_matched_data(data_dir=DATA_DIR):
    """Load dataset_combined_results2.csv for each dataset and return combined df."""
    frames = []
    for dataset in DATASETS:
        csv_path = os.path.join(data_dir, f"{dataset}_combined_results2.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {dataset}: file not found")
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        df["dataset"] = dataset
        frames.append(df)

    if not frames:
        raise ValueError("No data files found")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def prepare_data(df, add_is_ours=False):
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

    # Add is_ours flag for budget equivalence computation
    if add_is_ours:
        combined["is_ours"] = combined["comparison_method"] == "metric_match_aligned"

    return combined


def compute_annotations_saved(df):
    """
    For each (dataset, axis, model, metric):
      1. Compute random's mean error at BASELINE_BUDGET (averaged over runs).
      2. For each budget of our method, compute mean error over runs.
      3. Find the smallest budget where our mean error <= random's baseline error.
      4. annotations_saved = BASELINE_BUDGET - that budget.
         If our method never achieves the target, annotations_saved = NaN.
    Returns a detail DataFrame with one row per (dataset, axis, model, metric).
    """
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "comparison_method", "metric"])["estimation_error"]
        .mean()
        .reset_index()
        .rename(columns={"estimation_error": "mean_error"})
    )

    random_all = avg[avg["comparison_method"] == "random"][
        ["dataset", "axis", "model", "budget", "metric", "mean_error"]
    ].rename(columns={"mean_error": "random_error"})

    random_baseline = random_all[random_all["budget"] == BASELINE_BUDGET][
        ["dataset", "axis", "model", "metric", "random_error"]
    ].rename(columns={"random_error": "random_error_50"})

    ours = avg[avg["comparison_method"] == "metric_match_aligned"][
        ["dataset", "axis", "model", "budget", "metric", "mean_error"]
    ].rename(columns={"mean_error": "our_error"})

    combined = ours.merge(random_baseline, on=["dataset", "axis", "model", "metric"])
    combined["beats_baseline"] = combined["our_error"] <= combined["random_error_50"]

    # Also prepare random at all budgets for the reverse calculation
    our_baseline = avg[
        (avg["comparison_method"] == "metric_match_aligned") & (avg["budget"] == BASELINE_BUDGET)
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
        our_err_50 = grp_sorted.loc[grp_sorted["budget"] == BASELINE_BUDGET, "our_error"]

        if our_err_50.empty:
            # Skip if we don't have data at budget 50
            continue
        our_err_50 = our_err_50.iloc[0]

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
        available_cols = [c for c in col_order if c in tbl.columns]
        tbl = tbl[available_cols]
    tbl.columns.name = None
    tbl["mean"] = tbl.mean(axis=1)
    return tbl


def _interpolate_crossing(target_error, vm_budgets, vm_errors):
    """
    Find the smallest vm budget (via linear interpolation between adjacent points) where
    vm_error first drops to target_error.
    Returns NaN if the crossover is outside the observable range [vm_budgets[0], vm_budgets[-1]].
    """
    # vm already at or below target at minimum budget → floor at minimum budget
    if vm_errors[0] <= target_error:
        return float(vm_budgets[0])
    # vm never reaches the target even at max budget → cap at maximum budget
    if np.min(vm_errors) > target_error:
        return float(vm_budgets[-1])
    # Find first downward crossing: vm_errors[i] > target >= vm_errors[i+1]
    for i in range(len(vm_budgets) - 1):
        e1, e2 = vm_errors[i], vm_errors[i + 1]
        b1, b2 = float(vm_budgets[i]), float(vm_budgets[i + 1])
        if e1 > target_error and e2 <= target_error:
            if e1 == e2:
                return b1
            return b1 + (target_error - e1) / (e2 - e1) * (b2 - b1)
    return np.nan


def compute_budget_equivalence(df):
    """
    For each (dataset, axis, model, metric, random_budget):
      Linearly interpolate within the vm error curve to find the exact vm budget where
      vm_error == random_error at random_budget.
      Returns NaN where the crossover falls outside [5, 50].
    """
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "comparison_method", "metric", "is_ours"])["estimation_error"]
        .mean()
        .reset_index()
        .rename(columns={"estimation_error": "mean_error"})
    )

    random_avg = (
        avg[~avg["is_ours"]]
        [["dataset", "axis", "model", "metric", "budget", "mean_error"]]
        .rename(columns={"budget": "random_budget", "mean_error": "random_error"})
    )
    vm_avg = (
        avg[avg["is_ours"]]
        [["dataset", "axis", "model", "metric", "budget", "mean_error"]]
        .sort_values(["dataset", "axis", "model", "metric", "budget"])
    )

    records = []
    group_cols = ["dataset", "axis", "model", "metric"]
    for key, vm_grp in vm_avg.groupby(group_cols):
        vm_budgets = vm_grp["budget"].values
        vm_errors = vm_grp["mean_error"].values
        rand_grp = random_avg[
            (random_avg["dataset"] == key[0]) & (random_avg["axis"] == key[1]) &
            (random_avg["model"] == key[2]) & (random_avg["metric"] == key[3])
        ]
        for _, row in rand_grp.iterrows():
            equiv = _interpolate_crossing(row["random_error"], vm_budgets, vm_errors)
            records.append({
                "dataset": key[0], "axis": key[1], "model": key[2],
                "metric": key[3], "random_budget": row["random_budget"],
                "equivalent_budget": equiv,
            })
    return pd.DataFrame(records)


def plot_distribution(detail, out_path, col="annotations_saved", title_suffix="", subtitle=""):
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


def _savings_label(name, rb, eb):
    """Legend label with average absolute and relative savings."""
    rb = np.asarray(rb, dtype=float)
    eb = np.asarray(eb, dtype=float)
    savings = np.nanmean(rb - eb)
    rel_savings = np.nanmean((rb - eb) / rb)
    if np.isnan(savings):
        return f"{name} (avg savings: N/A)"
    return f"{name} (avg savings: {savings:.1f}, avg rel: {rel_savings:.1%})"


def _style_equiv_ax(ax, plot_budgets=None):
    x = np.array(plot_budgets if plot_budgets is not None else BUDGETS)
    ax.plot(x, x, "k--", lw=1, alpha=0.35, zorder=0, label="y = x (no savings)")
    ax.set_xticks(x)
    ax.set_yticks(np.array(BUDGETS))
    ax.set_xlabel("Random Sampling Budget", fontsize=11)
    ax.set_ylabel(f"Equivalent {OUR_METHOD} Budget", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_budget_equiv_summary(equivs, out_dir):
    """One plot: lines = metrics, averaged over all datasets/axes/models."""
    equivs = equivs[equivs["random_budget"].isin(PLOT_BUDGETS)]
    avg = equivs.groupby(["metric", "random_budget"])["equivalent_budget"].mean().reset_index()
    active_metrics = sorted(equivs["metric"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    _style_equiv_ax(ax, PLOT_BUDGETS)
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(active_metrics)))
    for color, metric in zip(colors, active_metrics):
        mdata = avg[avg["metric"] == metric].sort_values("random_budget")
        label = _savings_label(metric, mdata["random_budget"].values, mdata["equivalent_budget"].values)
        ax.plot(mdata["random_budget"], mdata["equivalent_budget"], marker="o", color=color, label=label)
    ax.set_title("Budget Equivalence — All Datasets (averaged)", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "budget_equiv_summary.jpg")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_budget_equiv_by_dataset(equivs, out_dir):
    """4 plots (one per dataset): lines = metrics, averaged over axes/models."""
    equivs = equivs[equivs["random_budget"].isin(PLOT_BUDGETS)]
    active_metrics = sorted(equivs["metric"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(active_metrics)))
    for dataset in DATASETS:
        ddata = equivs[equivs["dataset"] == dataset]
        avg = ddata.groupby(["metric", "random_budget"])["equivalent_budget"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 6))
        _style_equiv_ax(ax, PLOT_BUDGETS)
        for color, metric in zip(colors, active_metrics):
            mdata = avg[avg["metric"] == metric].sort_values("random_budget")
            label = _savings_label(metric, mdata["random_budget"].values, mdata["equivalent_budget"].values)
            ax.plot(mdata["random_budget"], mdata["equivalent_budget"], marker="o", color=color, label=label)
        ax.set_title(f"Budget Equivalence — {dataset}", fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"budget_equiv_dataset_{dataset}.jpg")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def plot_budget_equiv_by_metric(equivs, out_dir):
    """One plot per metric: lines = dataset-axis combos, averaged over models."""
    equivs = equivs[equivs["random_budget"].isin(PLOT_BUDGETS)]
    da_combos = sorted(
        equivs[["dataset", "axis"]].drop_duplicates().itertuples(index=False, name=None)
    )
    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(da_combos) - 1, 1)) for i in range(len(da_combos))]
    for metric in sorted(equivs["metric"].unique()):
        mdata = equivs[equivs["metric"] == metric]
        avg = mdata.groupby(["dataset", "axis", "random_budget"])["equivalent_budget"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(11, 7))
        _style_equiv_ax(ax, PLOT_BUDGETS)
        for color, (dataset, axis) in zip(colors, da_combos):
            daData = avg[(avg["dataset"] == dataset) & (avg["axis"] == axis)].sort_values("random_budget")
            if daData.empty:
                continue
            label = _savings_label(
                f"{dataset}/{axis}", daData["random_budget"].values, daData["equivalent_budget"].values
            )
            ax.plot(daData["random_budget"], daData["equivalent_budget"],
                    marker="o", color=color, label=label, lw=1.5)
        ax.set_title(f"Budget Equivalence — {metric}", fontsize=13)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, borderaxespad=0)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"budget_equiv_metric_{metric}.jpg")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def main(data_dir=DATA_DIR, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(data_dir, "annotations_saved")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading metric-matched data...")
    raw_df = load_metric_matched_data(data_dir)

    print("Preparing data for comparison (metric_match_aligned vs random)...")
    df = prepare_data(raw_df, add_is_ours=True)

    print(f"\nData summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Datasets: {df['dataset'].unique()}")
    print(f"  Methods: {df['comparison_method'].unique()}")
    print(f"  Metrics: {df['metric'].unique()}")
    print(f"  Budgets: {sorted(df['budget'].unique())}")

    print("\nComputing annotations saved...")
    detail = compute_annotations_saved(df)

    never_achieved = detail["annotations_saved"].isna().sum()
    print(f"Combos where our method never reached random@50 error: {never_achieved} / {len(detail)}")

    # Save full detail
    detail_path = os.path.join(output_dir, "annotations_saved_detail.csv")
    detail.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    # Summary by dataset x metric
    available_metrics = [m for m in METRICS if m in detail["metric"].unique()]
    by_dataset = summary_table(detail, "dataset", col_order=available_metrics)
    ds_path = os.path.join(output_dir, "annotations_saved_by_dataset.csv")
    by_dataset.to_csv(ds_path)
    print(f"\nSaved: {ds_path}")
    print(by_dataset.round(2).to_string())

    # Summary by model x metric
    by_model = summary_table(detail, "model", col_order=available_metrics)
    model_path = os.path.join(output_dir, "annotations_saved_by_model.csv")
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
    overall_path = os.path.join(output_dir, "annotations_saved_overall.csv")
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
    metric_path = os.path.join(output_dir, "annotations_saved_by_metric.csv")
    per_metric.to_csv(metric_path)
    print(f"\nSaved: {metric_path}")
    print(per_metric.to_string())

    # Bar plot — our method wins
    plot_distribution(
        detail,
        os.path.join(output_dir, "annotations_saved_distribution.jpg"),
        col="annotations_saved",
        title_suffix=" — Metric-Matched vs Random@50",
        subtitle=f"metric_match_aligned matches random at budget 50",
    )

    # Bar plot — random wins (losing combos): how many annotations random saves vs our_method@50
    losing = detail[detail["annotations_saved"].isna()].copy()
    if len(losing) > 0:
        plot_distribution(
            losing,
            os.path.join(output_dir, "annotations_saved_distribution_losing.jpg"),
            col="random_annotations_saved",
            title_suffix=" — Random vs Metric-Matched@50 (losing combos)",
            subtitle=f"Random matches metric_match_aligned at budget 50 ({len(losing)} combos)",
        )

    # Budget equivalence plots
    print("\nComputing budget equivalence...")
    equivs = compute_budget_equivalence(df)
    equiv_dir = os.path.join(output_dir, "budget_equivalence")
    os.makedirs(equiv_dir, exist_ok=True)
    equiv_csv = os.path.join(equiv_dir, "budget_equivalence_detail.csv")
    equivs.to_csv(equiv_csv, index=False)
    print(f"Saved: {equiv_csv}")
    plot_budget_equiv_summary(equivs, equiv_dir)
    plot_budget_equiv_by_dataset(equivs, equiv_dir)
    plot_budget_equiv_by_metric(equivs, equiv_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotations saved analysis for metric-matched subsets"
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Data directory with combined_results2.csv files")
    parser.add_argument("--output-dir", default=None, help="Output directory for annotations saved")
    args = parser.parse_args()
    main(data_dir=args.data_dir, output_dir=args.output_dir)
