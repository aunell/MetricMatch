import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_DIR = "/Users/alyssaunell/code/SmartSample_local/results/04_01_downstream_task_small_ensemble_and_target"
DATASETS = ["hanna", "medval", "mslr", "summeval"]
METRICS = ["alpha", "icc", "rho", "tau"]
OUR_METHOD = "variance_matched_weighted_.9"
MEAN_SQUARED_ERROR_METHOD = "metric_matched_mean_squared_error"
BASELINE = "random"
BUDGETS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
THRESHOLDS = [0.7]


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def load_estimation_data(results_dir=RESULTS_DIR, track_mse_match=False):
    """Load {metric}_results.csv for each dataset and metric, return combined df."""
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
            df["is_ours"] = df["method"] == OUR_METHOD
            frames.append(df)
        if track_mse_match:
            csv_path = os.path.join(
                results_dir, dataset, dataset, "dataframes", "mse_results.csv"
            )
            if not os.path.exists(csv_path):
                print(f"  Skipping mean_squared_error | {dataset}: file not found")
            else:
                df = pd.read_csv(csv_path)
                df = df[df["method"].isin([MEAN_SQUARED_ERROR_METHOD, BASELINE])][
                    ["model", "budget", "method", "estimation_error", "axis"]
                ]
                df["dataset"] = dataset
                df["metric"] = "mean_squared_error"
                df["is_ours"] = df["method"] == MEAN_SQUARED_ERROR_METHOD
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
    available = [m for m in METRICS + ["mean_squared_error"] if m in result.columns]
    result = result[available]
    result.index.name = "budget"
    result.columns.name = None
    return result.sort_index()


def _macro_merge(df):
    """Average over runs then compare our method vs random."""
    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "method", "metric", "is_ours"])[
            "estimation_error"
        ]
        .mean()
        .reset_index()
    )
    ours = avg[avg["is_ours"]].rename(columns={"estimation_error": "our_err"})
    base = avg[~avg["is_ours"]].rename(columns={"estimation_error": "random_err"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


def _micro_merge(df):
    """Pair runs by position then compare our method vs random."""
    df = assign_run_index(df)
    ours = df[df["is_ours"]].rename(columns={"estimation_error": "our_err"})
    base = df[~df["is_ours"]].rename(columns={"estimation_error": "random_err"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])
    merged["win"] = merged["our_err"] < merged["random_err"]
    return merged


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def load_threshold_raw_data(results_dir=RESULTS_DIR):
    """Load raw metric results with predicted and true values for threshold analysis."""
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
            pred_col, true_col = f"predicted_{metric}", f"true_{metric}"
            if pred_col not in df.columns or true_col not in df.columns:
                print(f"  Skipping {metric} | {dataset}: missing columns {pred_col}/{true_col}")
                continue
            df = df[df["method"].isin([OUR_METHOD, BASELINE])][
                ["model", "budget", "method", pred_col, true_col, "axis"]
            ].rename(columns={pred_col: "predicted", true_col: "true_val"})
            df["dataset"] = dataset
            df["metric"] = metric
            df["is_ours"] = df["method"] == OUR_METHOD
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _classify(series, threshold):
    """Return boolean Series: True if value >= threshold."""
    return series >= threshold


def _threshold_micro_merge(df, threshold):
    """
    Per-run classification correctness, paired by position.
    correct = (predicted_class == true_class) for each individual run. Ties discarded.
    """
    df = df.copy()
    df["correct"] = (_classify(df["predicted"], threshold) == _classify(df["true_val"], threshold)).astype(float)

    df = df.sort_values(["dataset", "metric", "axis", "model", "budget", "method"])
    df["run"] = df.groupby(["dataset", "metric", "axis", "model", "budget", "method"]).cumcount()

    ours = df[df["is_ours"]].rename(columns={"correct": "our_correct"})
    base = df[~df["is_ours"]].rename(columns={"correct": "random_correct"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric", "run"])

    n_total = len(merged)
    n_ties = (merged["our_correct"] == merged["random_correct"]).sum()
    merged = merged[merged["our_correct"] != merged["random_correct"]].copy()
    print(f"  [threshold micro T{threshold}] {n_total} comparisons, {n_ties} ties removed, {len(merged)} remaining")
    merged["win"] = merged["our_correct"] > merged["random_correct"]
    return merged


def _threshold_macro_merge(df, threshold):
    """
    Macro: average correct over runs per (dataset, axis, model, budget, method, metric),
    then compare our method vs random. Ties discarded.
    """
    df = df.copy()
    df["correct"] = (_classify(df["predicted"], threshold) == _classify(df["true_val"], threshold)).astype(float)

    df = df.sort_values(["dataset", "metric", "axis", "model", "budget", "method"])
    df["run"] = df.groupby(["dataset", "metric", "axis", "model", "budget", "method"]).cumcount()

    avg = (
        df.groupby(["dataset", "axis", "model", "budget", "method", "metric", "is_ours"])["correct"]
        .mean()
        .reset_index()
    )

    ours = avg[avg["is_ours"]].rename(columns={"correct": "our_acc"})
    base = avg[~avg["is_ours"]].rename(columns={"correct": "random_acc"})
    merged = ours.merge(base, on=["dataset", "axis", "model", "budget", "metric"])

    n_total = len(merged)
    n_ties = (merged["our_acc"] == merged["random_acc"]).sum()
    merged = merged[merged["our_acc"] != merged["random_acc"]].copy()
    print(f"  [threshold macro T{threshold}] {n_total} comparisons, {n_ties} ties removed, {len(merged)} remaining")
    merged["win"] = merged["our_acc"] > merged["random_acc"]
    return merged


def _ds_axis_key(df):
    return df["dataset"].str.lower() + "-" + df["axis"].str.lower()


def _plot_win_rates_by_dataset(macro_est, micro_est, thr_macro, thr_micro, out_path, suptitle="Win Rates by Dataset-Axis"):
    """2×2 figure: win rate vs budget with one line per dataset-axis."""
    def _wr_by_budget(merged):
        df = merged.copy()
        df["key"] = _ds_axis_key(df)
        return df.groupby(["key", "budget"])["win"].mean().reset_index()

    panel_dfs = [
        _wr_by_budget(macro_est),
        _wr_by_budget(thr_macro),
        _wr_by_budget(micro_est),
        _wr_by_budget(thr_micro),
    ]
    panel_titles = ["Macro — Estimation", "Macro — Threshold (T0.7)", "Micro — Estimation", "Micro — Threshold (T0.7)"]
    panels = list(zip(panel_dfs, panel_titles))

    all_keys = sorted(set().union(*[set(df["key"]) for df in panel_dfs if not df.empty]))
    print(f"  [{suptitle}] {len(all_keys)} dataset-axis keys: {all_keys}")

    cmap = cm.get_cmap("tab20")
    colors = {k: cmap(i / max(len(all_keys) - 1, 1)) for i, k in enumerate(all_keys)}

    fig, axes_grid = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]

    for ax, (df, title) in zip(axes_flat, panels):
        for key in all_keys:
            sub = df[df["key"] == key].sort_values("budget")
            if sub.empty:
                continue
            ax.plot(sub["budget"], sub["win"], label=key, color=colors[key],
                    marker="o", markersize=3, linewidth=1.5)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Budget")
        ax.set_ylabel("Win Rate")
        ax.set_ylim(0, 1)
        ax.set_xticks(BUDGETS)

    handles = [plt.Line2D([0], [0], color=colors[k], label=k, linewidth=1.5) for k in all_keys]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.18, 0.5), fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _threshold_pivot(merged):
    win_rates = (
        merged.groupby(["budget", "metric"])["win"]
        .mean()
        .reset_index()
        .rename(columns={"win": "win_rate"})
    )
    result = win_rates.pivot(index="budget", columns="metric", values="win_rate")
    available = [m for m in METRICS + ["mean_squared_error"] if m in result.columns]
    result = result[available]
    result.index.name = "budget"
    result.columns.name = None
    return result.sort_index()


# ---------------------------------------------------------------------------
# Sanity helpers
# ---------------------------------------------------------------------------

def _print_wr_by_ds_axis(merged, label):
    """Print win rate per (dataset, axis), averaged over budgets/models/metrics."""
    key = _ds_axis_key(merged)
    wr = merged.groupby(key)["win"].mean().sort_index()
    print(f"  Win rates by dataset-axis [{label}]:")
    for k, v in wr.items():
        print(f"    {k}: {v:.3f}")


def _pivot_by_ds_axis(merged, value_col, agg="mean"):
    """Pivot (budget × dataset-axis) table for a given value column."""
    df = merged.copy()
    df["key"] = _ds_axis_key(df)
    grouped = df.groupby(["budget", "key"])[value_col]
    tbl = (grouped.mean() if agg == "mean" else grouped.count()).reset_index()
    return tbl.pivot(index="budget", columns="key", values=value_col)


def _build_estimation_table(macro_est, micro_est):
    """Budget × (dataset-axis, {macro_estimation, micro_estimation}) win rate table.
    Includes an 'average' row at the end averaging over all budgets for each column.
    """
    frames = []
    for atype, merged in [("macro_estimation", macro_est), ("micro_estimation", micro_est)]:
        if merged is None or merged.empty:
            continue
        tbl = _pivot_by_ds_axis(merged, "win")
        tbl.columns = pd.MultiIndex.from_tuples([(col, atype) for col in tbl.columns])
        frames.append(tbl)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, axis=1).sort_index(axis=1).sort_index()
    result.index.name = "budget"
    avg_row = result.mean(axis=0).rename("average")
    result = pd.concat([result, avg_row.to_frame().T])
    return result


def _build_threshold_table(thr_macro, thr_micro):
    """Budget × (dataset-axis, {macro_threshold, macro_threshold_n, micro_threshold, micro_threshold_n}) table.

    *_n columns show the number of non-tie comparisons the win rate is based on.
    """
    frames = []
    for atype, merged in [("macro_threshold", thr_macro), ("micro_threshold", thr_micro)]:
        if merged is None or merged.empty:
            continue
        wr_tbl = _pivot_by_ds_axis(merged, "win", agg="mean")
        wr_tbl.columns = pd.MultiIndex.from_tuples([(col, atype) for col in wr_tbl.columns])
        n_tbl = _pivot_by_ds_axis(merged, "win", agg="count")
        n_tbl.columns = pd.MultiIndex.from_tuples([(col, f"{atype}_n") for col in n_tbl.columns])
        frames.extend([wr_tbl, n_tbl])
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, axis=1).sort_index(axis=1).sort_index()
    result.index.name = "budget"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save(df, path):
    df.to_csv(path)
    print(f"Saved: {path}")
    print(df.round(3).to_string())
    print()


def main(results_dir=RESULTS_DIR, track_mse_match=False):
    est_dir = os.path.join(results_dir, "win_rates", "estimation")
    thr_dir = os.path.join(results_dir, "win_rates", "threshold")
    disagg_ds_dir = os.path.join(results_dir, "win_rates", "disaggregated_by_dataset")
    disagg_metric_dir = os.path.join(results_dir, "win_rates", "disagg_metric")
    os.makedirs(est_dir, exist_ok=True)
    os.makedirs(thr_dir, exist_ok=True)
    os.makedirs(disagg_ds_dir, exist_ok=True)
    os.makedirs(disagg_metric_dir, exist_ok=True)

    active_metrics = METRICS + (["mean_squared_error"] if track_mse_match else [])

    # -- Estimation --
    print("Loading estimation data...")
    est_df = load_estimation_data(results_dir, track_mse_match=track_mse_match)
    print(f"  Loaded {len(est_df)} rows | datasets: {sorted(est_df['dataset'].unique())} | "
          f"metrics: {sorted(est_df['metric'].unique())} | models: {sorted(est_df['model'].unique())} | "
          f"budgets: {sorted(est_df['budget'].unique())}")

    print("\n--- Macro estimation win rates ---")
    est_macro_merged = _macro_merge(est_df)
    print(f"  {len(est_macro_merged)} macro comparisons")
    _print_wr_by_ds_axis(est_macro_merged, "macro estimation")
    save(_estimation_pivot(est_macro_merged), os.path.join(est_dir, "macro_win_rates_estimation.csv"))
    for dataset in DATASETS:
        save(
            _estimation_pivot(est_macro_merged[est_macro_merged["dataset"] == dataset]),
            os.path.join(est_dir, f"macro_win_rates_estimation_{dataset}.csv"),
        )

    print("\n--- Micro estimation win rates ---")
    est_micro_merged = _micro_merge(est_df)
    print(f"  {len(est_micro_merged)} micro comparisons")
    _print_wr_by_ds_axis(est_micro_merged, "micro estimation")
    save(_estimation_pivot(est_micro_merged), os.path.join(est_dir, "micro_win_rates_estimation.csv"))
    for dataset in DATASETS:
        save(
            _estimation_pivot(est_micro_merged[est_micro_merged["dataset"] == dataset]),
            os.path.join(est_dir, f"micro_win_rates_estimation_{dataset}.csv"),
        )

    print("\n--- Estimation summary win rates ---")
    micro_total = len(est_micro_merged)
    micro_wins  = est_micro_merged["win"].sum()
    macro_total = len(est_macro_merged)
    macro_wins  = est_macro_merged["win"].sum()
    summary = pd.DataFrame([
        {"avg": "micro (all runs)",  "total_comparisons": micro_total, "wins": micro_wins, "win_rate": micro_wins / micro_total},
        {"avg": "macro (avg runs)", "total_comparisons": macro_total, "wins": macro_wins, "win_rate": macro_wins / macro_total},
    ]).set_index("avg")
    save(summary, os.path.join(est_dir, "summary_win_rates_estimation.csv"))

    # -- Threshold --
    print("\nLoading raw data for threshold analysis...")
    thr_raw = load_threshold_raw_data(results_dir)
    print(f"  Loaded {len(thr_raw)} rows for threshold analysis")

    thr_macro_07 = thr_micro_07 = thr_macro_07_plot = thr_micro_07_plot = None
    for threshold in THRESHOLDS:
        t_str = f"T{threshold}"
        print(f"\nComputing threshold win rates ({t_str})...")
        macro_merged = _threshold_macro_merge(thr_raw, threshold)
        micro_merged = _threshold_micro_merge(thr_raw, threshold)

        _print_wr_by_ds_axis(macro_merged, f"threshold macro {t_str}")
        _print_wr_by_ds_axis(micro_merged, f"threshold micro {t_str}")

        if threshold == 0.7:
            thr_macro_07 = macro_merged
            thr_micro_07 = micro_merged

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

    thr_macro_plot = thr_macro_07 if thr_macro_07 is not None else pd.DataFrame()
    thr_micro_plot = thr_micro_07 if thr_micro_07 is not None else pd.DataFrame()

    # -- Win rates by dataset-axis, averaged over all metrics --
    print("\n--- Win rates by dataset (all metrics) ---")
    _plot_win_rates_by_dataset(
        est_macro_merged, est_micro_merged,
        thr_macro_plot, thr_micro_plot,
        os.path.join(disagg_ds_dir, "win_rates_by_dataset.png"),
    )
    def _save_tables(macro_est, micro_est, thr_macro, thr_micro, out_dir, suffix=""):
        est_tbl = _build_estimation_table(macro_est, micro_est)
        thr_tbl = _build_threshold_table(thr_macro, thr_micro)
        est_path = os.path.join(out_dir, f"win_rate_table_estimation{suffix}.csv")
        thr_path = os.path.join(out_dir, f"win_rate_table_threshold{suffix}.csv")
        est_tbl.to_csv(est_path)
        thr_tbl.to_csv(thr_path)
        print(f"  Saved: {est_path} ({len(est_tbl)} budgets × {len(est_tbl.columns)} cols)")
        print(f"  Saved: {thr_path} ({len(thr_tbl)} budgets × {len(thr_tbl.columns)} cols)")

    _save_tables(est_macro_merged, est_micro_merged, thr_macro_plot, thr_micro_plot, disagg_ds_dir)

    # -- Win rates by dataset-axis, one plot per metric --
    print("\n--- Win rates by metric ---")
    for metric in active_metrics:
        print(f"  Plotting metric: {metric}")
        macro_m = est_macro_merged[est_macro_merged["metric"] == metric]
        micro_m = est_micro_merged[est_micro_merged["metric"] == metric]
        thr_macro_m = thr_macro_plot[thr_macro_plot["metric"] == metric] if not thr_macro_plot.empty else pd.DataFrame()
        thr_micro_m = thr_micro_plot[thr_micro_plot["metric"] == metric] if not thr_micro_plot.empty else pd.DataFrame()

        _plot_win_rates_by_dataset(
            macro_m, micro_m, thr_macro_m, thr_micro_m,
            os.path.join(disagg_metric_dir, f"win_rates_{metric}.png"),
            suptitle=f"Win Rates by Dataset-Axis — {metric}",
        )
        _save_tables(macro_m, micro_m, thr_macro_m, thr_micro_m, disagg_metric_dir, suffix=f"_{metric}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Win rates and threshold analysis")
    parser.add_argument("--folder", default=RESULTS_DIR, help="Results directory")
    parser.add_argument(
        "--track_mse_match", action="store_true",
        help="Also track metric_matched_mean_squared_error for the mean_squared_error metric",
    )
    args = parser.parse_args()
    main(results_dir=args.folder, track_mse_match=args.track_mse_match)
