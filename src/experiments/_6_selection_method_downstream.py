"""
Selection Method Downstream Task: Model Ranking Recovery.

Accepts either:
  --folder <root>   auto-discovers all <dataset>/<dataset>/dataframes/icc_results.csv
                    and alpha_results.csv beneath <root>
  --input  <file>   single icc_results.csv or alpha_results.csv (legacy)

For each (dataset, metric, axis):
  1. Average predicted_value / estimation_error / true_value over runs.
  2. Rank models by predicted vs. true value at each (budget, method).
  3. Compute Spearman ρ between predicted and true rankings.
  4. Plot per-dataset and aggregated correlation curves.

Usage:
    python -m src.experiments._6_selection_method_downstream --folder results/04_01_downstream_task  --output results/04_01_downstream_task

    python -m src.experiments._6_selection_method_downstream \\
        --input  results/04_01_downstream_task/medval/medval/dataframes/icc_results.csv \\
        --output results/06_selection_method_downstream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[2]))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = "results/04_01_downstream_task_small_ensemble_and_target"

ORACLE_METHODS = {"oracle_msb_mse", "oracle_msb", "oracle_mse", "oracle_icc", "oracle_alpha"}

METHOD_LABELS = {
    "random":                       "Random",
    "variance_matched_combined":    "VM Combined",
    "variance_matched_msb":         "VM MSB",
    "variance_matched_weighted_.2": "VM Weighted 0.2",
    "variance_matched_weighted_.5": "VM Weighted 0.5",
    "variance_matched_weighted_.7": "VM Weighted 0.7",
    "variance_matched_weighted_.9": "VM Weighted 0.9",
    "oracle_msb_mse":               "Oracle MSB+MSE",
    "oracle_msb":                   "Oracle MSB",
    "oracle_mse":                   "Oracle MSE",
    "oracle_icc":                   "Oracle ICC",
    "metric_matched_icc":           "Metric Matched ICC",
}

METRIC_LABEL = {"icc": "ICC", "alpha": "Krippendorff α"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_one(path: Path, dataset: str, metric: str) -> pd.DataFrame:
    """
    Load a single icc_results.csv or alpha_results.csv and normalise to:
      dataset, metric, model, budget, method, axis,
      predicted_value, true_value, estimation_error
    """
    df = pd.read_csv(path)
    pred_col = f"predicted_{metric}"
    true_col = f"true_{metric}"
    if pred_col not in df.columns or true_col not in df.columns:
        raise ValueError(
            f"{path}: expected columns '{pred_col}' and '{true_col}', "
            f"got {df.columns.tolist()}"
        )
    df = df.rename(columns={pred_col: "predicted_value", true_col: "true_value"})
    df["dataset"] = dataset
    df["metric"] = metric
    return df[["dataset", "metric", "model", "budget", "method", "axis",
               "predicted_value", "true_value", "estimation_error"]]


def load_folder(folder: Path) -> pd.DataFrame:
    """
    Discover all {dataset}/{dataset}/dataframes/icc_results.csv and
    alpha_results.csv under `folder` and concatenate into one DataFrame.
    """
    frames = []
    for metric in ("icc", "alpha"):
        for csv_path in sorted(folder.rglob(f"{metric}_results.csv")):
            # Expect path like <root>/<dataset>/<dataset>/dataframes/<metric>_results.csv
            parts = csv_path.relative_to(folder).parts
            dataset = parts[0]
            print(f"  Found {metric} | dataset={dataset}: {csv_path}")
            try:
                frames.append(_load_one(csv_path, dataset, metric))
            except ValueError as e:
                print(f"  WARNING: skipping {csv_path}: {e}")
    if not frames:
        raise FileNotFoundError(f"No icc_results.csv or alpha_results.csv found under {folder}")
    return pd.concat(frames, ignore_index=True)


def load_single(path: Path) -> pd.DataFrame:
    """Load a single results CSV, inferring metric from filename."""
    if "alpha" in path.name:
        metric = "alpha"
    else:
        metric = "icc"
    # Infer dataset from grandparent folder name
    dataset = path.parents[1].name
    return _load_one(path, dataset, metric)


# ---------------------------------------------------------------------------
# Step 1: Average over runs
# ---------------------------------------------------------------------------

def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Average predicted_value, true_value, estimation_error over runs."""
    return (
        df.groupby(["dataset", "metric", "model", "budget", "method", "axis"], sort=True)
        .agg(
            predicted_value=("predicted_value", "mean"),
            true_value=("true_value", "mean"),
            estimation_error=("estimation_error", "mean"),
            n_runs=("predicted_value", "count"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Step 2-3: Rankings + Spearman correlation
# ---------------------------------------------------------------------------

def compute_ranking_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman ρ per run, then average over runs.

    Operating on per-run predicted values (rather than run-averaged values) ensures
    that random sampling is evaluated fairly: averaging many random runs before ranking
    would let the law of large numbers inflate its ρ artificially.
    """
    df = df.copy()
    # Assign a run index within each (dataset, metric, budget, method, axis, model).
    # All models within the same (budget, method, axis) share the same run index.
    df["run"] = df.groupby(
        ["dataset", "metric", "budget", "method", "axis", "model"]
    ).cumcount()

    rows = []
    for (dataset, metric, budget, method, axis, run), grp in df.groupby(
        ["dataset", "metric", "budget", "method", "axis", "run"]
    ):
        grp = grp.sort_values("model").reset_index(drop=True)
        if len(grp) < 2:
            continue
        pred_rank = grp["predicted_value"].rank(ascending=False, method="average")
        true_rank = grp["true_value"].rank(ascending=False, method="average")
        rho, p = stats.spearmanr(pred_rank, true_rank)
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "budget": budget,
            "method": method,
            "axis": axis,
            "run": run,
            "spearman_rho": rho,
            "spearman_p": p,
            "n_models": len(grp),
        })

    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run

    return (
        per_run.groupby(["dataset", "metric", "budget", "method", "axis"])
        .agg(
            spearman_rho=("spearman_rho", "mean"),
            spearman_p=("spearman_p", "mean"),
            n_models=("n_models", "first"),
            n_runs=("spearman_rho", "count"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(agg: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    for metric in rank_df["metric"].unique():
        m_rank = rank_df[rank_df["metric"] == metric]
        m_agg  = agg[agg["metric"] == metric]

        print(f"\n{'='*70}")
        print(f"METRIC: {METRIC_LABEL.get(metric, metric.upper())}")
        print(f"{'='*70}")

        # True rankings per dataset/axis
        print("\n--- True model rankings (mean true_value) ---")
        true_tbl = (
            m_agg.groupby(["dataset", "axis", "model"])["true_value"]
            .mean().reset_index()
            .sort_values(["dataset", "axis", "true_value"], ascending=[True, True, False])
        )
        print(true_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Rho pivot per dataset: method × budget, averaged over axes
        for dataset in sorted(m_rank["dataset"].unique()):
            d_rank = m_rank[m_rank["dataset"] == dataset]
            print(f"\n--- Spearman ρ (method × budget) | dataset={dataset} ---")
            pivot = d_rank.pivot_table(
                index="method", columns="budget", values="spearman_rho", aggfunc="mean"
            )
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
            pivot.index = [METHOD_LABELS.get(m, m) for m in pivot.index]
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

        # Aggregated rho pivot
        print(f"\n--- Spearman ρ (method × budget) | AGGREGATED (mean over datasets × axes) ---")
        agg_pivot = m_rank.pivot_table(
            index="method", columns="budget", values="spearman_rho", aggfunc="mean"
        )
        agg_pivot = agg_pivot.loc[agg_pivot.mean(axis=1).sort_values(ascending=False).index]
        agg_pivot.index = [METHOD_LABELS.get(m, m) for m in agg_pivot.index]
        print(agg_pivot.to_string(float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _method_color_marker(methods: list[str]) -> tuple[dict, dict]:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+", "x", "d"]
    color_map  = {m: colors[i % len(colors)]  for i, m in enumerate(methods)}
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods)}
    return color_map, marker_map


def _rho_line_axes(ax, sub_rank: pd.DataFrame, methods: list[str],
                   color_map: dict, marker_map: dict, title: str) -> None:
    """Draw rho-vs-budget lines onto `ax`, averaging sub_rank over axes."""
    budgets = sorted(sub_rank["budget"].unique())
    for method in methods:
        rhos = (
            sub_rank[sub_rank["method"] == method]
            .groupby("budget")["spearman_rho"].mean()
            .reindex(budgets)
        )
        ax.plot(
            budgets, rhos.values,
            label=METHOD_LABELS.get(method, method),
            color=color_map[method],
            marker=marker_map[method],
            linewidth=1.6,
            markersize=4,
        )
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Budget", fontsize=8)
    ax.set_ylabel("Spearman ρ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())


def plot_rho_vs_budget(rank_df: pd.DataFrame, out_dir: Path) -> None:
    """
    For each metric: one figure with one panel per dataset + one aggregated panel.
    Lines = non-oracle methods. Rho averaged over axes within each panel.
    """
    rank_df = rank_df[~rank_df["method"].isin(ORACLE_METHODS)]
    # Consistent method order (by overall mean rho descending)
    methods = list(
        rank_df.groupby("method")["spearman_rho"].mean()
        .sort_values(ascending=False).index
    )
    color_map, marker_map = _method_color_marker(methods)

    for metric in rank_df["metric"].unique():
        m_rank = rank_df[rank_df["metric"] == metric]
        datasets = sorted(m_rank["dataset"].unique())
        ncols = len(datasets) + 1          # +1 for aggregated
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), sharey=False)
        if ncols == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            _rho_line_axes(
                ax, m_rank[m_rank["dataset"] == dataset],
                methods, color_map, marker_map,
                title=dataset,
            )

        # Aggregated panel
        _rho_line_axes(
            axes[-1], m_rank,
            methods, color_map, marker_map,
            title="Aggregated",
        )

        # Shared legend outside
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, fontsize=7,
                   bbox_to_anchor=(1.01, 0.5), loc="center left")
        fig.suptitle(
            f"Ranking Recovery — {METRIC_LABEL.get(metric, metric)} (ρ vs. budget)",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        path = out_dir / f"rho_vs_budget_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}")


def plot_rho_per_budget_grouped(rank_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Grouped bar chart for each metric: panels = (datasets + aggregated),
    groups = budgets, bars = methods. Rho averaged over axes.
    """
    rank_df = rank_df[~rank_df["method"].isin(ORACLE_METHODS)]
    methods = list(
        rank_df.groupby("method")["spearman_rho"].mean()
        .sort_values(ascending=False).index
    )
    budgets = sorted(rank_df["budget"].unique())
    color_map, _ = _method_color_marker(methods)
    n_methods = len(methods)
    width = 0.8 / n_methods
    x = np.arange(len(budgets))

    for metric in rank_df["metric"].unique():
        m_rank = rank_df[rank_df["metric"] == metric]
        datasets = sorted(m_rank["dataset"].unique())
        panels = datasets + ["Aggregated"]
        ncols = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), sharey=False)
        if ncols == 1:
            axes = [axes]

        for ax, panel in zip(axes, panels):
            sub = m_rank if panel == "Aggregated" else m_rank[m_rank["dataset"] == panel]
            for i, method in enumerate(methods):
                rhos = (
                    sub[sub["method"] == method]
                    .groupby("budget")["spearman_rho"].mean()
                    .reindex(budgets).values
                )
                offset = (i - n_methods / 2 + 0.5) * width
                ax.bar(
                    x + offset, rhos, width,
                    label=METHOD_LABELS.get(method, method),
                    color=color_map[method],
                    alpha=0.85,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(budgets, fontsize=7)
            ax.set_title(panel, fontsize=9)
            ax.set_xlabel("Budget", fontsize=8)
            ax.set_ylabel("Spearman ρ", fontsize=8)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(labelsize=7)

        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, fontsize=7,
                   bbox_to_anchor=(1.01, 0.5), loc="center left")
        fig.suptitle(
            f"Ranking Recovery per Budget — {METRIC_LABEL.get(metric, metric)}",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        path = out_dir / f"rho_per_budget_grouped_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}")


def plot_estimation_error_heatmap(agg: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: mean estimation error by method × budget, one figure per (dataset, metric)."""
    agg = agg[~agg["method"].isin(ORACLE_METHODS)]
    for (dataset, metric), sub in agg.groupby(["dataset", "metric"]):
        pivot = sub.pivot_table(
            index="method", columns="budget", values="estimation_error", aggfunc="mean"
        )
        pivot.index = [METHOD_LABELS.get(m, m) for m in pivot.index]
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        fig, ax = plt.subplots(figsize=(11, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Budget", fontsize=10)
        ax.set_title(
            f"Mean Estimation Error — {dataset} / {METRIC_LABEL.get(metric, metric)}",
            fontsize=11,
        )
        plt.colorbar(im, ax=ax, label="Mean estimation error")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=6, color="black")
        fig.tight_layout()
        path = out_dir / f"estimation_error_heatmap_{dataset}_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved → {path}")


THRESHOLD = 0.7
CM_METHODS = ["random", "variance_matched_weighted_.9"]


def _build_cm(sub: pd.DataFrame, method: str, budget: int) -> np.ndarray:
    """
    2×2 confusion matrix for one (method, budget).
    Rows = True label (Accept / Reject), Cols = Predicted label (Accept / Reject).
    Accept = value >= THRESHOLD.
    Returns [[TP, FN], [FP, TN]] as int array.
    """
    ms = sub[(sub["method"] == method) & (sub["budget"] == budget)]
    if ms.empty:
        return np.zeros((2, 2), dtype=int)
    true_acc = ms["true_value"] >= THRESHOLD
    pred_acc = ms["predicted_value"] >= THRESHOLD
    TP = int((true_acc  & pred_acc).sum())
    FN = int((true_acc  & ~pred_acc).sum())
    FP = int((~true_acc & pred_acc).sum())
    TN = int((~true_acc & ~pred_acc).sum())
    return np.array([[TP, FN], [FP, TN]], dtype=int)


def _draw_cm(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    """Draw a 2×2 confusion matrix heatmap onto ax."""
    labels = ["Accept", "Reject"]
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("True", fontsize=6)
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.tick_params(length=2, pad=1)
    cell_labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cell_labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")


def plot_predicted_vs_true(agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter + confusion matrices, one figure per (dataset, metric).
    Row 0: scatter panels (one per sampled budget), all methods.
    Rows 1+: one CM row per CM_METHODS (random, VM Weighted 0.9).
    """
    agg = agg[~agg["method"].isin(ORACLE_METHODS)]
    n_cm = len(CM_METHODS)

    for (dataset, metric), sub in agg.groupby(["dataset", "metric"]):
        budgets = sorted(sub["budget"].unique())
        show_budgets = budgets[::max(1, len(budgets) // 5)][:5]
        methods = sorted(sub["method"].unique())
        color_map, marker_map = _method_color_marker(methods)
        ncols = len(show_budgets)
        mlabel = METRIC_LABEL.get(metric, metric)

        # gridspec: scatter row (height 3) + one CM row per CM_METHOD (height 1 each)
        fig = plt.figure(figsize=(4 * ncols, 4 + 2.2 * n_cm))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(
            1 + n_cm, ncols,
            height_ratios=[3] + [1] * n_cm,
            hspace=0.55, wspace=0.3,
        )

        # ---- Row 0: scatter -------------------------------------------------
        scatter_axes = [fig.add_subplot(gs[0, c]) for c in range(ncols)]
        for ax, budget in zip(scatter_axes, show_budgets):
            bsub = sub[sub["budget"] == budget]
            for method in methods:
                ms = bsub[bsub["method"] == method]
                ax.scatter(
                    ms["true_value"], ms["predicted_value"],
                    label=METHOD_LABELS.get(method, method),
                    color=color_map[method], marker=marker_map[method],
                    s=30, alpha=0.7,
                )
            mn = min(bsub["true_value"].min(), bsub["predicted_value"].min()) - 0.05
            mx = max(bsub["true_value"].max(), bsub["predicted_value"].max()) + 0.05
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8, alpha=0.5)
            ax.axvline(THRESHOLD, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
            ax.axhline(THRESHOLD, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
            ax.set_title(f"Budget={budget}", fontsize=9)
            ax.set_xlabel(f"True {mlabel}", fontsize=8)
            if ax is scatter_axes[0]:
                ax.set_ylabel(f"Predicted {mlabel}", fontsize=8)
            ax.tick_params(labelsize=7)

        handles, labels_ = scatter_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, fontsize=7,
                   bbox_to_anchor=(1.01, 0.75), loc="upper left")

        # ---- Rows 1+: confusion matrices ------------------------------------
        for r, cm_method in enumerate(CM_METHODS):
            row_label = METHOD_LABELS.get(cm_method, cm_method)
            for c, budget in enumerate(show_budgets):
                ax_cm = fig.add_subplot(gs[1 + r, c])
                cm = _build_cm(sub, cm_method, budget)
                _draw_cm(ax_cm, cm, title=row_label)

        fig.suptitle(
            f"Predicted vs. True {mlabel} — {dataset}  (threshold={THRESHOLD})",
            fontsize=11,
        )
        path = out_dir / f"predicted_vs_true_{dataset}_{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Save dataframes
# ---------------------------------------------------------------------------

def save_results(agg: pd.DataFrame, rank_df: pd.DataFrame, out_dir: Path) -> None:
    df_dir = out_dir / "dataframes"
    df_dir.mkdir(parents=True, exist_ok=True)

    agg.to_csv(df_dir / "averaged_values.csv", index=False)
    print(f"Saved → {df_dir / 'averaged_values.csv'}")

    rank_flat = rank_df
    rank_flat.to_csv(df_dir / "ranking_correlations.csv", index=False)
    print(f"Saved → {df_dir / 'ranking_correlations.csv'}")

    # Aggregated method summary (mean ρ over all datasets × axes × budgets)
    summary = (
        rank_df.groupby(["metric", "method"])["spearman_rho"]
        .agg(mean_rho="mean", std_rho="std", n="count")
        .reset_index()
        .sort_values(["metric", "mean_rho"], ascending=[True, False])
    )
    summary["method_label"] = summary["method"].map(lambda m: METHOD_LABELS.get(m, m))
    summary.to_csv(df_dir / "method_ranking_summary.csv", index=False)
    print(f"Saved → {df_dir / 'method_ranking_summary.csv'}")

    # Pivot: method × budget (aggregated mean over all datasets × axes)
    for metric in rank_df["metric"].unique():
        pivot = rank_df[rank_df["metric"] == metric].pivot_table(
            index="method", columns="budget", values="spearman_rho", aggfunc="mean"
        )
        pivot.to_csv(df_dir / f"rho_method_budget_pivot_{metric}.csv")
        print(f"Saved → {df_dir / f'rho_method_budget_pivot_{metric}.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selection method downstream: model ranking recovery"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--folder",
        help="Root results folder; auto-discovers all icc_results.csv and alpha_results.csv"
    )
    src.add_argument(
        "--input",
        help="Path to a single icc_results.csv or alpha_results.csv"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    plot_dir = out_dir / "downstream_plots"

    # ---- Load ---------------------------------------------------------------
    if args.folder:
        folder = Path(args.folder)
        print(f"Scanning {folder} for results …")
        df = load_folder(folder)
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        df = load_single(path)

    print(f"\n  {len(df):,} rows | datasets: {sorted(df['dataset'].unique())} "
          f"| metrics: {sorted(df['metric'].unique())}")

    # ---- Aggregate runs -----------------------------------------------------
    print("Averaging over runs …")
    agg = aggregate_runs(df)
    print(f"  → {len(agg):,} rows after averaging")

    # ---- Ranking correlations -----------------------------------------------
    print("Computing ranking correlations …")
    rank_df = compute_ranking_correlations(df)
    print(f"  → {len(rank_df):,} (dataset, metric, budget, method, axis) combinations")

    # ---- Print + save -------------------------------------------------------
    print_summary(agg, rank_df)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_results(agg, rank_df, out_dir)

    if not args.no_plots:
        print("\nGenerating plots …")
        plot_rho_vs_budget(rank_df, plot_dir)
        plot_rho_per_budget_grouped(rank_df, plot_dir)
        plot_estimation_error_heatmap(agg, plot_dir)
        plot_predicted_vs_true(agg, plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
