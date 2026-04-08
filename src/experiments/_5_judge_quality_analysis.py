"""
Judge Quality Analysis: evaluate LLM judges (J1–J5) against human ground truth (H).

For each (dataset, axis, judge Ji):
  Our metrics (quality of Ji as a proxy for H):
    ICC(H, Ji)   – intraclass correlation coefficient (ICC3k), higher = better
    Alpha(H, Ji) – Krippendorff's alpha (ordinal), higher = better

  Downstream metrics (ground-truth quality of Ji relative to H):
    Spearman ρ   – rank correlation of per-item scores, higher = better
    Pearson r    – linear correlation of per-item scores, higher = better
    MSE          – mean squared error per item, lower = better
    MeanAcc      – |mean(Ji) - mean(H)| across all items, lower = better

Evaluation (4 tables × 8 metric-downstream pairs):
  Table 1 – Spearman ρ:     rank the judges by our metric vs. downstream metric;
             Spearman ρ over the judges per (dataset, axis), averaged.
  Table 2 – Pearson r:      same, Pearson r over the judges, averaged.
  Table 3 – Best judge:     fraction where argbest(our metric) == argbest(downstream metric).
  Table 4 – Classification: fraction of (dataset, axis, judge) triples where
             classifying as above/below T=0.7 agrees between our metric and downstream metric.
             (higher-is-better metrics: "good" if > T; lower-is-better: "good" if < T)

Usage:
    python -m src.experiments._5_judge_quality_analysis [--output-dir results/judge_quality_analysis]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.data_loading import load_judge_scores
from src.utils.reliability_metrics import compute_icc_pingouin, compute_krippendorff_alpha

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = ["hanna", "medval", "mslr", "summeval"]

EVALUATION_AXES = {
    "hanna":    ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval":   ["Risk"],
    "mslr":     ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"],
}

ALL_MODELS = [
    "claude-3.5-sonnet",
    "gpt-4.1",
    "gpt-5",
    "deepseek-r1",
    "gemini-2.5-pro",
    # "gpt-4o-mini",
    # "meta-llama-Llama-3.1-8B-Instruct",
    # "google-gemma-3-1b-it",
    # "Qwen-Qwen2.5-7B-Instruct",
]

DEFAULT_DATA_DIR = "data/judge_scores"
DEFAULT_OUTPUT_DIR = "results/judge_quality_analysis"

OUR_METRICS = ["icc", "krippendorff_alpha", "spearman_rho", "kendall_tau"]
DOWNSTREAM_METRICS = ["spearman_rho", "pearson_r", "mse", "mean_acc", "icc", "krippendorff_alpha", "kendall_tau"]

# True = higher is better, False = lower is better
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "icc":               True,
    "krippendorff_alpha": True,
    "spearman_rho":      True,
    "pearson_r":         True,
    "mse":               False,
    "mean_acc":          False,
    "kendall_tau":       True,
}

CLASSIFICATION_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Per-judge metric computation
# ---------------------------------------------------------------------------

def _align_model_human(df: pd.DataFrame, model: str, axis: str) -> pd.DataFrame | None:
    """
    Return a DataFrame with model and human scores for text_ids that have both,
    filtered to the given axis.
    """
    ax_df = df[df["evaluation_axis"] == axis]
    model_df = ax_df[ax_df["model_name"] == model][["text_id", "evaluation_score"]].rename(
        columns={"evaluation_score": "model_score"}
    )
    human_df = ax_df[ax_df["model_name"] == "original"][["text_id", "evaluation_score"]].rename(
        columns={"evaluation_score": "human_score"}
    )
    merged = model_df.merge(human_df, on="text_id")
    if len(merged) < 4:
        return None
    return merged


def compute_icc_alpha(
    df: pd.DataFrame, model: str, axis: str, valid_ids=None
) -> tuple[float, float]:
    """Compute ICC(3,k) and Krippendorff's alpha for one judge vs. human."""
    ax_df = df[df["evaluation_axis"] == axis]
    pair_df = ax_df[ax_df["model_name"].isin([model, "original"])]

    counts = pair_df.groupby("text_id")["model_name"].nunique()
    all_valid = counts[counts == 2].index
    if valid_ids is not None:
        all_valid = all_valid.intersection(valid_ids)
    pair_df = pair_df[pair_df["text_id"].isin(all_valid)]

    if len(pair_df) < 4:
        return np.nan, np.nan

    icc = compute_icc_pingouin(pair_df, models=[model, "original"])
    alpha = compute_krippendorff_alpha(pair_df, models=[model, "original"])
    if isinstance(alpha, dict):
        alpha = alpha.get(axis, np.nan)
    return icc, alpha


def compute_downstream_metrics(merged: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """
    Compute all downstream metrics from an aligned (model_score, human_score) DataFrame.

    Returns: (spearman_rho, pearson_r, mse, mean_acc, kendall_tau)
      spearman_rho – rank correlation, higher = better
      pearson_r    – linear correlation, higher = better
      mse          – mean squared error per item, lower = better
      mean_acc     – |mean(model) - mean(human)|, lower = better
      kendall_tau  – Kendall's tau-b rank correlation, higher = better
    """
    if len(merged) < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    spearman_rho, _ = stats.spearmanr(merged["model_score"], merged["human_score"])
    pearson_r, _ = stats.pearsonr(merged["model_score"], merged["human_score"])
    mse = float(np.mean((merged["model_score"] - merged["human_score"]) ** 2))
    mean_acc = float(abs(merged["model_score"].mean() - merged["human_score"].mean()))
    kendall_tau, _ = stats.kendalltau(merged["model_score"], merged["human_score"])

    return float(spearman_rho), float(pearson_r), mse, mean_acc, float(kendall_tau)


# ---------------------------------------------------------------------------
# Build per-judge metric table
# ---------------------------------------------------------------------------

def run_analysis(data_dir: str, models: list[str]) -> pd.DataFrame:
    """
    Compute ICC, Alpha, and all downstream metrics for each (dataset, axis, judge).

    Returns a long-form DataFrame with columns:
        dataset, axis, model, icc, krippendorff_alpha,
        spearman_rho, pearson_r, mse, mean_acc, n_items
    """
    rows = []

    for dataset in DATASETS:
        print(f"\n--- Dataset: {dataset} ---")
        axes = EVALUATION_AXES[dataset]
        ds_dir = os.path.join(data_dir, dataset)

        available = [
            m for m in models
            if any(
                os.path.exists(os.path.join(ds_dir, f"results_{dataset}_{m}_{ax}.json"))
                for ax in axes
            )
        ]

        if not available:
            print(f"  No judge score files found, skipping.")
            continue

        df = load_judge_scores(dataset, available, data_dir, EVALUATION_AXES)
        if df.empty:
            print(f"  Empty DataFrame after loading, skipping.")
            continue

        for model in available:
            for axis in axes:
                merged = _align_model_human(df, model, axis)
                if merged is None:
                    continue
                merged = merged.head(300)

                capped_ids = merged["text_id"].values
                icc, alpha = compute_icc_alpha(df, model, axis, valid_ids=capped_ids)
                spearman, pearson, mse, mean_acc, kendall_tau = compute_downstream_metrics(merged)

                rows.append({
                    "dataset": dataset,
                    "axis": axis,
                    "model": model,
                    "icc": icc,
                    "krippendorff_alpha": alpha,
                    "spearman_rho": spearman,
                    "pearson_r": pearson,
                    "mse": mse,
                    "mean_acc": mean_acc,
                    "kendall_tau": kendall_tau,
                    "n_items": len(merged),
                })

        print(f"  Processed {len(available)} judges × {len(axes)} axes")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _argbest(values: np.ndarray, higher_is_better: bool) -> int | None:
    """Return index of the best value; None if all NaN."""
    valid = ~np.isnan(values)
    if not valid.any():
        return None
    return int(np.nanargmax(values)) if higher_is_better else int(np.nanargmin(values))


def _is_good(value: float, higher_is_better: bool, threshold: float) -> bool | None:
    """Classify value relative to threshold; None if NaN."""
    if np.isnan(value):
        return None
    return (value > threshold) if higher_is_better else (value < threshold)


# ---------------------------------------------------------------------------
# Four evaluation tables
# ---------------------------------------------------------------------------

def compute_eval_tables(
    metrics_df: pd.DataFrame,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce all 4 evaluation tables from the per-judge metric DataFrame.

    Each table has 8 rows — one per (our_metric × downstream_metric) pair:
      Table 1: avg Spearman ρ across judges, per dataset-axis, then averaged
      Table 2: avg Pearson r across judges, per dataset-axis, then averaged
      Table 3: fraction of dataset-axes where best judge by our metric == best by downstream
      Table 4: fraction of (dataset-axis, judge) pairs where classification at T=0.7 agrees
    """
    pairs = [(om, dm) for om in OUR_METRICS for dm in DOWNSTREAM_METRICS]
    dataset_axes = metrics_df[["dataset", "axis"]].drop_duplicates().values.tolist()

    spearman_acc:   dict[tuple, list[float]] = {p: [] for p in pairs}
    pearson_acc:    dict[tuple, list[float]] = {p: [] for p in pairs}
    best_judge_acc: dict[tuple, list[float]] = {p: [] for p in pairs}
    class_acc:      dict[tuple, list[float]] = {p: [] for p in pairs}

    for dataset, axis in dataset_axes:
        sub = metrics_df[
            (metrics_df["dataset"] == dataset) & (metrics_df["axis"] == axis)
        ].reset_index(drop=True)
        if sub.empty:
            continue

        for om, dm in pairs:
            om_vals = sub[om].values.astype(float)
            dm_vals = sub[dm].values.astype(float)
            valid = ~np.isnan(om_vals) & ~np.isnan(dm_vals)

            # Tables 1 & 2: Spearman / Pearson correlation across judges
            if valid.sum() >= 3:
                rho, _ = stats.spearmanr(om_vals[valid], dm_vals[valid])
                spearman_acc[(om, dm)].append(float(rho))

                r, _ = stats.pearsonr(om_vals[valid], dm_vals[valid])
                pearson_acc[(om, dm)].append(float(r))

            # Table 3: best judge identification
            best_by_om = _argbest(om_vals, METRIC_HIGHER_IS_BETTER[om])
            best_by_dm = _argbest(dm_vals, METRIC_HIGHER_IS_BETTER[dm])
            if best_by_om is not None and best_by_dm is not None:
                best_judge_acc[(om, dm)].append(float(best_by_om == best_by_dm))

            # Table 4: classification at threshold per (dataset-axis, judge) pair
            for i in range(len(sub)):
                om_good = _is_good(om_vals[i], METRIC_HIGHER_IS_BETTER[om],
                                   threshold)
                dm_good = _is_good(dm_vals[i], METRIC_HIGHER_IS_BETTER[dm],
                                   threshold)
                if om_good is not None and dm_good is not None:
                    class_acc[(om, dm)].append(float(om_good == dm_good))

    def _to_df(acc: dict, value_col: str, count_col: str) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "our_metric": om,
                "downstream_metric": dm,
                value_col: float(np.mean(v)) if v else np.nan,
                count_col: len(v),
            }
            for (om, dm), v in acc.items()
        ])

    return (
        _to_df(spearman_acc,   "avg_spearman_rho",          "n_dataset_axes"),
        _to_df(pearson_acc,    "avg_pearson_r",              "n_dataset_axes"),
        _to_df(best_judge_acc, "frac_best_judge_match",      "n_dataset_axes"),
        _to_df(class_acc,      "frac_classification_agree",  "n_judge_pairs"),
    )


# ---------------------------------------------------------------------------
# Raw values + corpus-level correlations
# ---------------------------------------------------------------------------

def print_raw_upstream_values(metrics_df: pd.DataFrame) -> None:
    """
    Print all (model, dataset, axis) ICC and alpha values (135 rows for 9 models × 15 axes),
    then print per-model averages across all dataset-axes.
    """
    print(f"\n{'='*90}")
    print("Raw upstream metric values — all (model, dataset, axis) observations")
    print(f"  Total rows: {len(metrics_df)}")
    print("=" * 90)
    display = (
        metrics_df[["model", "dataset", "axis", "icc", "krippendorff_alpha", "spearman_rho", "kendall_tau"]]
        .sort_values(["model", "dataset", "axis"])
        .reset_index(drop=True)
    )
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*70}")
    print("Per-model average upstream metrics (across all dataset-axes)")
    print("=" * 70)
    avg = metrics_df.groupby("model")[["icc", "krippendorff_alpha", "spearman_rho", "kendall_tau"]].mean()
    avg.insert(0, "n_dataset_axes", metrics_df.groupby("model")["icc"].count())
    print(avg.to_string(float_format=lambda x: f"{x:.4f}"))


def compute_corpus_level_correlations(
    metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman and Pearson correlation between each upstream (our) metric and
    each downstream metric.

    Two granularities:
      - Non-aggregated: across all (model, dataset, axis) rows (~135 observations)
      - Aggregated:     per-model averages (one row per model → ~9 observations)

    When our_metric == downstream_metric (e.g. ICC vs ICC), the correlation should
    be 1.0 (sanity check).

    Returns (nonagg_df, agg_df) each with columns:
        our_metric, downstream_metric, spearman_rho, pearson_r, n_obs
    """
    def _corr_row(om, dm, vals):
        if om == dm:
            # Same metric: correlation is exactly 1 (sanity check)
            n = int(vals[om].notna().sum())
            return 1.0, 1.0, n
        x = vals[om].values.astype(float)
        y = vals[dm].values.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 3:
            return np.nan, np.nan, n
        xv, yv = x[mask], y[mask]
        if len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
            return np.nan, np.nan, n
        rho, _ = stats.spearmanr(xv, yv)
        r, _ = stats.pearsonr(xv, yv)
        return float(rho), float(r), n

    rows_nonagg, rows_agg = [], []
    all_metric_cols = list(dict.fromkeys(OUR_METRICS + DOWNSTREAM_METRICS))  # deduplicated
    agg_df = metrics_df.groupby("model")[all_metric_cols].mean().reset_index()

    for om in OUR_METRICS:
        for dm in DOWNSTREAM_METRICS:
            rho, r, n = _corr_row(om, dm, metrics_df)
            sanity = " ← sanity check (should be 1.0)" if om == dm else ""
            rows_nonagg.append({
                "our_metric": om, "downstream_metric": dm,
                "spearman_rho": rho, "pearson_r": r,
                "n_obs": n, "note": sanity.strip(),
            })

            rho_a, r_a, n_a = _corr_row(om, dm, agg_df)
            rows_agg.append({
                "our_metric": om, "downstream_metric": dm,
                "spearman_rho": rho_a, "pearson_r": r_a,
                "n_obs": n_a, "note": sanity.strip(),
            })

    return pd.DataFrame(rows_nonagg), pd.DataFrame(rows_agg)


def print_corpus_correlations(
    nonagg_df: pd.DataFrame,
    agg_df: pd.DataFrame,
) -> None:
    print(f"\n{'='*90}")
    print("Corpus-level correlations — non-aggregated (all model×dataset×axis observations)")
    print("  Spearman ρ and Pearson r between upstream (our) metric and downstream metric")
    print("  Sanity check: when our_metric == downstream_metric, both correlations should be 1.0")
    print("=" * 90)
    print(nonagg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'='*90}")
    print("Corpus-level correlations — aggregated (per-model averages across dataset-axes)")
    print("=" * 90)
    print(agg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Selection-method analysis: use existing ICC/alpha sampling results
# ---------------------------------------------------------------------------

BUDGETS_SUBSET = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
VM_METHOD = "variance_matched_weighted_.9"   # representative variance-matching method
DEFAULT_SELECTION_RESULTS_DIR = "results/04_01_downstream_task"

# For METRIC_HIGHER_IS_BETTER lookups in the selection analysis
_OM_HIGHER_IS_BETTER = {"icc": True, "alpha": True, "rho": True, "tau": True}


def load_selection_results(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discover and load all icc/alpha/rho/tau_results.csv under `results_dir`.

    Expects files at <results_dir>/<dataset>/<dataset>/dataframes/{metric}_results.csv
    but also falls back to <results_dir>/<dataset>/dataframes/{metric}_results.csv.
    Automatically discovers all datasets from the folder structure.

    Returns (icc_df, alpha_df, rho_df, tau_df) each with an added `dataset` column.
    """
    base = Path(results_dir)
    icc_frames:   list[pd.DataFrame] = []
    alpha_frames: list[pd.DataFrame] = []
    rho_frames:   list[pd.DataFrame] = []
    tau_frames:   list[pd.DataFrame] = []

    for metric, frames in [("icc", icc_frames), ("alpha", alpha_frames),
                            ("rho", rho_frames), ("tau", tau_frames)]:
        for csv_path in sorted(base.rglob(f"{metric}_results.csv")):
            # Determine dataset from the first path component under base
            rel_parts = csv_path.relative_to(base).parts
            dataset = rel_parts[0]
            print(f"  Loading {metric} | dataset={dataset}: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                df["dataset"] = dataset
                frames.append(df)
            except Exception as e:
                print(f"  WARNING: could not load {csv_path}: {e}")

    icc_df   = pd.concat(icc_frames,   ignore_index=True) if icc_frames   else pd.DataFrame()
    alpha_df = pd.concat(alpha_frames, ignore_index=True) if alpha_frames else pd.DataFrame()
    rho_df   = pd.concat(rho_frames,   ignore_index=True) if rho_frames   else pd.DataFrame()
    tau_df   = pd.concat(tau_frames,   ignore_index=True) if tau_frames   else pd.DataFrame()

    if not icc_df.empty:
        print(f"  Loaded ICC results:   {len(icc_df):,} rows | "
              f"datasets={sorted(icc_df['dataset'].unique())}")
    if not alpha_df.empty:
        print(f"  Loaded Alpha results: {len(alpha_df):,} rows | "
              f"datasets={sorted(alpha_df['dataset'].unique())}")
    if not rho_df.empty:
        print(f"  Loaded Rho results:   {len(rho_df):,} rows | "
              f"datasets={sorted(rho_df['dataset'].unique())}")
    if not tau_df.empty:
        print(f"  Loaded Tau results:   {len(tau_df):,} rows | "
              f"datasets={sorted(tau_df['dataset'].unique())}")

    return icc_df, alpha_df, rho_df, tau_df


def _eval_one_cell(
    est_vals: np.ndarray,
    dm_vals: np.ndarray,
    om_name: str,
    dm_name: str,
    table_type: str,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> float | None:
    """
    Compute one evaluation value given arrays of estimated and downstream metric
    values across judges.  Returns None if inputs are insufficient.
    """
    om_hib = _OM_HIGHER_IS_BETTER.get(om_name, True)
    dm_hib = METRIC_HIGHER_IS_BETTER.get(dm_name, True)
    valid  = ~np.isnan(est_vals) & ~np.isnan(dm_vals)

    if table_type == "spearman":
        if valid.sum() < 3:
            return None
        ev, dv = est_vals[valid], dm_vals[valid]
        if np.all(ev == ev[0]) or np.all(dv == dv[0]):
            return None
        rho, _ = stats.spearmanr(ev, dv)
        return float(rho) if not np.isnan(rho) else None

    elif table_type == "pearson":
        if valid.sum() < 3:
            return None
        ev, dv = est_vals[valid], dm_vals[valid]
        if np.all(ev == ev[0]) or np.all(dv == dv[0]):
            return None
        r, _ = stats.pearsonr(ev, dv)
        return float(r) if not np.isnan(r) else None

    elif table_type == "best_judge":
        best_est = _argbest(est_vals, om_hib)
        best_dm  = _argbest(dm_vals,  dm_hib)
        if best_est is None or best_dm is None:
            return None
        return float(best_est == best_dm)

    elif table_type == "classification":
        vals = []
        for i in range(len(est_vals)):
            eg = _is_good(est_vals[i], om_hib, threshold)
            dg = _is_good(dm_vals[i],  dm_hib, threshold)
            if eg is not None and dg is not None:
                vals.append(float(eg == dg))
        return float(np.mean(vals)) if vals else None

    return None



def _mean_over_runs(
    df: pd.DataFrame,
    om_name: str,
    dm_name: str,
    table_type: str,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> float:
    """
    For a pre-filtered (dataset, axis, budget, method) slice: compute _eval_one_cell
    per run, then return the mean over runs.

    Per-run evaluation avoids LLN inflation that pre-averaging predictions would give.
    """
    run_vals: list[float] = []
    for _, run_df in df.groupby("run"):
        v = _eval_one_cell(
            run_df["predicted"].values.astype(float),
            run_df[dm_name].values.astype(float),
            om_name, dm_name, table_type,
            threshold=threshold,
        )
        if v is not None:
            run_vals.append(v)
    return float(np.mean(run_vals)) if run_vals else np.nan


def _est_err_over_runs(df: pd.DataFrame, true_col: str) -> float:
    """
    Mean absolute error between predicted and true metric values across judges,
    averaged over runs.
    """
    if true_col not in df.columns:
        return np.nan
    run_vals: list[float] = []
    for _, run_df in df.groupby("run"):
        mae = float(np.nanmean(np.abs(run_df["predicted"] - run_df[true_col])))
        if not np.isnan(mae):
            run_vals.append(mae)
    return float(np.mean(run_vals)) if run_vals else np.nan


def _std_over_runs(
    df: pd.DataFrame,
    om_name: str,
    dm_name: str,
    table_type: str,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> float:
    """
    Std of _eval_one_cell across runs for a pre-filtered (dataset, axis, budget, method) slice.
    Captures run-to-run variability — high std means the method is unstable across draws.
    """
    run_vals: list[float] = []
    for _, run_df in df.groupby("run"):
        v = _eval_one_cell(
            run_df["predicted"].values.astype(float),
            run_df[dm_name].values.astype(float),
            om_name, dm_name, table_type,
            threshold=threshold,
        )
        if v is not None:
            run_vals.append(v)
    return float(np.std(run_vals)) if len(run_vals) >= 2 else np.nan


def compute_selection_eval_tables(
    icc_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    downstream_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    budgets: list[int] = BUDGETS_SUBSET,
    threshold: float = CLASSIFICATION_THRESHOLD,
    rho_df: pd.DataFrame | None = None,
    tau_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Produce 4 evaluation tables (one per table type).

    Each row is one (our_metric, downstream_metric, dataset, axis) combination —
    no aggregation over axes.

    Columns: our_metric | downstream_metric | dataset | axis | true_downstream
             | random_B | vm_B | delta_B | random_est_err_B | vm_est_err_B
             (for each B in budgets)

    true_downstream   = same computation as eval_* tables (full-data ICC/alpha, all judges)
    random_B / vm_B   = avg per-run value using predicted ICC/alpha at budget B
    delta_B           = vm_B - random_B  (positive → variance matching is better)
    *_est_err_B       = mean |predicted - true| across judges, averaged over runs
    """
    table_types = ["spearman", "pearson", "best_judge", "classification"]
    dm_names    = DOWNSTREAM_METRICS

    # Map selection om_name -> column name in metrics_df and true column in raw_df
    _OM_TO_METRICS_COL = {
        "icc":   "icc",
        "alpha": "krippendorff_alpha",
        "rho":   "spearman_rho",
        "tau":   "kendall_tau",
    }
    _OM_TO_TRUE_COL = {
        "icc":   "true_icc",
        "alpha": "true_alpha",
        "rho":   "true_rho",
        "tau":   "true_tau",
    }

    configs = [
        # (om_name, pred_col,         raw_df)
        ("icc",   "predicted_icc",   icc_df),
        ("alpha", "predicted_alpha",  alpha_df),
        ("rho",   "predicted_rho",    rho_df if rho_df is not None else pd.DataFrame()),
        ("tau",   "predicted_tau",    tau_df if tau_df is not None else pd.DataFrame()),
    ]

    # Prepare per-metric merged dataframes (with run index)
    prepared: dict[str, pd.DataFrame] = {}  # om_name -> df_with_run
    for om_name, pred_col, raw_df in configs:
        if raw_df.empty:
            continue

        df = raw_df.copy()
        df["run"] = df.groupby(["dataset", "axis", "model", "budget", "method"]).cumcount()
        df = df.merge(downstream_df, on=["dataset", "axis", "model"], how="inner")
        df = df.rename(columns={pred_col: "predicted"})

        if df.empty:
            print(f"  WARNING: no rows after merging {om_name} with downstream metrics")
            continue

        prepared[om_name] = df

    results: dict[str, pd.DataFrame] = {}

    for table_type in table_types:
        rows: list[dict] = []

        for om_name in prepared:
            df = prepared[om_name]
            metrics_col = _OM_TO_METRICS_COL[om_name]
            true_col    = _OM_TO_TRUE_COL[om_name]

            dataset_axes = df[["dataset", "axis"]].drop_duplicates().values.tolist()

            for dm_name in dm_names:
                om_hib = _OM_HIGHER_IS_BETTER.get(om_name, True)
                dm_hib = METRIC_HIGHER_IS_BETTER.get(dm_name, True)

                for dataset, axis in dataset_axes:
                    row: dict = {
                        "our_metric": om_name,
                        "downstream_metric": dm_name,
                        "dataset": dataset,
                        "axis": axis,
                    }

                    # True baseline for this (dataset, axis)
                    da_metrics = metrics_df[
                        (metrics_df["dataset"] == dataset) & (metrics_df["axis"] == axis)
                    ]
                    om_vals = da_metrics[metrics_col].values.astype(float)
                    dm_vals = da_metrics[dm_name].values.astype(float)

                    if table_type == "classification":
                        true_vals_flat: list[float] = []
                        for om_v, dm_v in zip(om_vals, dm_vals):
                            og = _is_good(om_v, om_hib, threshold)
                            dg = _is_good(dm_v, dm_hib, threshold)
                            if og is not None and dg is not None:
                                true_vals_flat.append(float(og == dg))
                        row["true_downstream"] = float(np.mean(true_vals_flat)) if true_vals_flat else np.nan
                    else:
                        v = _eval_one_cell(om_vals, dm_vals, om_name, dm_name, table_type)
                        row["true_downstream"] = v if v is not None else np.nan

                    da_df = df[(df["dataset"] == dataset) & (df["axis"] == axis)]

                    for budget in budgets:
                        r_data = da_df[(da_df["budget"] == budget) & (da_df["method"] == "random")]
                        v_data = da_df[(da_df["budget"] == budget) & (da_df["method"] == VM_METHOD)]

                        r_val = _mean_over_runs(r_data, om_name, dm_name, table_type, threshold=threshold)
                        v_val = _mean_over_runs(v_data, om_name, dm_name, table_type, threshold=threshold)
                        delta = (v_val - r_val) if not (np.isnan(r_val) or np.isnan(v_val)) else np.nan

                        row[f"random_{budget}"]        = r_val
                        row[f"vm_{budget}"]            = v_val
                        row[f"delta_{budget}"]         = delta
                        row[f"random_est_err_{budget}"] = _est_err_over_runs(r_data, true_col)
                        row[f"vm_est_err_{budget}"]    = _est_err_over_runs(v_data, true_col)
                        row[f"random_std_{budget}"]    = _std_over_runs(r_data, om_name, dm_name, table_type, threshold=threshold)
                        row[f"vm_std_{budget}"]        = _std_over_runs(v_data, om_name, dm_name, table_type, threshold=threshold)

                    rows.append(row)

        results[table_type] = pd.DataFrame(rows)

    return results


_TRUE_COL_TO_METRICS_COL = {
    "true_icc":   "icc",
    "true_alpha": "krippendorff_alpha",
    "true_rho":   "spearman_rho",
    "true_tau":   "kendall_tau",
}


def compute_prediction_spread(
    icc_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    budgets: list[int] = BUDGETS_SUBSET,
    rho_df: pd.DataFrame | None = None,
    tau_df: pd.DataFrame | None = None,
    metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For each (metric, budget, method), compute the mean std of predicted values
    across judges within each (dataset, axis, run) group, then average over groups.

    A compressed std relative to true_std indicates variance-matching is shrinking
    inter-judge discriminability, which would explain lower rank-correlation scores
    even when absolute prediction error is lower.

    When true_col is absent from a results CSV (e.g. old rho/tau files), the true
    values are pulled from metrics_df using _TRUE_COL_TO_METRICS_COL.
    """
    rows = []
    configs = [
        ("icc",   "predicted_icc",   "true_icc",   icc_df),
        ("alpha", "predicted_alpha", "true_alpha",  alpha_df),
        ("rho",   "predicted_rho",   "true_rho",    rho_df if rho_df is not None else pd.DataFrame()),
        ("tau",   "predicted_tau",   "true_tau",    tau_df if tau_df is not None else pd.DataFrame()),
    ]
    for om_name, pred_col, true_col, raw_df in configs:
        if raw_df.empty:
            continue
        df = raw_df.copy()
        df["run"] = df.groupby(["dataset", "axis", "model", "budget", "method"]).cumcount()

        # If true_col is missing (old results file), backfill from metrics_df.
        if true_col not in df.columns and metrics_df is not None:
            src_col = _TRUE_COL_TO_METRICS_COL.get(true_col)
            if src_col is not None and src_col in metrics_df.columns:
                lookup = metrics_df[["dataset", "axis", "model", src_col]].rename(
                    columns={src_col: true_col}
                )
                df = df.merge(lookup, on=["dataset", "axis", "model"], how="left")

        # True std: std of true metric across judges per (dataset, axis)
        if true_col not in df.columns:
            true_std = np.nan
        else:
            true_std_vals = (
                df.groupby(["dataset", "axis", "model"])[[true_col]]
                .first()
                .reset_index()
                .groupby(["dataset", "axis"])[true_col]
                .std()
            )
            true_std = float(np.nanmean(true_std_vals.values))

        for budget in budgets:
            for method in df["method"].unique():
                sub = df[(df["budget"] == budget) & (df["method"] == method)]
                # std of predicted values across judges per (dataset, axis, run)
                grp_std = (
                    sub.groupby(["dataset", "axis", "run"])[pred_col]
                    .std()
                    .dropna()
                )
                # std of predicted values across runs per (dataset, axis, model)
                judge_std = (
                    sub.groupby(["dataset", "axis", "model"])[pred_col]
                    .std()
                    .dropna()
                )
                mean_pred_std_val = float(np.nanmean(grp_std.values)) if len(grp_std) else np.nan
                rows.append({
                    "metric":           om_name,
                    "budget":           budget,
                    "method":           method,
                    "mean_pred_std":    mean_pred_std_val,
                    "mean_judge_std":   float(np.nanmean(judge_std.values)) if len(judge_std) else np.nan,
                    "true_std":         true_std,
                    "compression":      float(mean_pred_std_val / true_std)
                                    if (true_std > 0 and not np.isnan(mean_pred_std_val)) else np.nan,
                })
    return pd.DataFrame(rows)


def print_prediction_spread(spread_df: pd.DataFrame) -> None:
    print(f"\n{'='*90}")
    print("Prediction spread diagnostic: mean std of predicted metric across judges")
    print("compression = mean_pred_std / true_std  (< 1 → VM compresses inter-judge variance)")
    print("=" * 90)
    print(spread_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_selection_tables(tables: dict[str, pd.DataFrame], budgets: list[int] = BUDGETS_SUBSET) -> None:
    _TABLE_TITLES = {
        "spearman":       "Spearman ρ across judges (estimated metric → downstream)",
        "pearson":        "Pearson r across judges (estimated metric → downstream)",
        "best_judge":     "Best judge ID: frac(argbest(estimated) == argbest(downstream))",
        "classification": f"Classification (T={CLASSIFICATION_THRESHOLD}): frac agree above/below",
    }
    col_order = ["our_metric", "downstream_metric", "dataset", "axis", "true_downstream"]
    for b in budgets:
        col_order += [f"random_{b}", f"vm_{b}", f"delta_{b}",
                      f"random_est_err_{b}", f"vm_est_err_{b}"]

    for ttype, title in _TABLE_TITLES.items():
        if ttype not in tables:
            continue
        df = tables[ttype]
        print(f"\n{'='*110}")
        print(f"Table: {title}")
        print(f"vm = {VM_METHOD}   |   delta = vm - random  (positive → vm is better)")
        print(f"true_downstream = full-data ICC/alpha as estimator (baseline)")
        print("=" * 110)
        display = df[[c for c in col_order if c in df.columns]]
        print(display.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if not np.isnan(x) else "    nan"))


def save_selection_results(tables: dict[str, pd.DataFrame], output_dir: str) -> None:
    out = Path(output_dir) / "dataframes"
    out.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        fname = f"selection_{key}.csv"
        df.to_csv(out / fname, index=False)
        print(f"Saved {fname} → {out / fname}")


def _plot_convergence_table(
    df: pd.DataFrame,
    title: str,
    valid_budgets: list[int],
    out: Path,
    fname: str,
) -> None:
    """Render one convergence figure (mean + std rows) for a single table DataFrame."""
    our_metrics = sorted(df["our_metric"].unique())
    ncols = len(DOWNSTREAM_METRICS)
    nrows = 2 * len(our_metrics)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), squeeze=False)

    for om_idx, om in enumerate(our_metrics):
        mean_row = 2 * om_idx
        std_row  = 2 * om_idx + 1

        for col_idx, dm in enumerate(DOWNSTREAM_METRICS):
            ax_mean = axes[mean_row][col_idx]
            ax_std  = axes[std_row][col_idx]

            sub = df[(df["our_metric"] == om) & (df["downstream_metric"] == dm)]
            if sub.empty:
                ax_mean.set_visible(False)
                ax_std.set_visible(False)
                continue

            plot_budgets = [b for b in valid_budgets if f"random_{b}" in sub.columns]

            # ---- top: mean eval metric ----
            true_val  = sub["true_downstream"].mean()
            rand_vals = [sub[f"random_{b}"].mean() for b in plot_budgets]
            vm_vals   = [sub[f"vm_{b}"].mean()     for b in plot_budgets]

            ax_mean.axhline(true_val, color="gray", linestyle="--", linewidth=1.2,
                            label="true (full data)")
            ax_mean.plot(plot_budgets, rand_vals, marker="o", color="steelblue",
                         linewidth=1.5, label="random")
            ax_mean.plot(plot_budgets, vm_vals, marker="s", color="darkorange",
                         linewidth=1.5, label="vm")
            ax_mean.set_title(f"{om} → {dm}", fontsize=8)
            ax_mean.set_xticks(plot_budgets)
            ax_mean.tick_params(labelsize=7)
            ax_mean.set_xticklabels([])
            if col_idx == 0 and om_idx == 0:
                ax_mean.legend(fontsize=7)

            # ---- bottom: std across runs ----
            rand_stds = [sub[f"random_std_{b}"].mean() for b in plot_budgets
                         if f"random_std_{b}" in sub.columns]
            vm_stds   = [sub[f"vm_std_{b}"].mean()     for b in plot_budgets
                         if f"vm_std_{b}"     in sub.columns]

            ax_std.plot(plot_budgets, rand_stds, marker="o", color="steelblue",
                        linewidth=1.5, linestyle="--")
            ax_std.plot(plot_budgets, vm_stds, marker="s", color="darkorange",
                        linewidth=1.5, linestyle="--")
            ax_std.set_title(f"std ({om} → {dm})", fontsize=7)
            ax_std.set_xticks(plot_budgets)
            ax_std.tick_params(labelsize=7)
            if std_row == nrows - 1:
                ax_std.set_xlabel("budget", fontsize=8)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    path = out / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {path}")


def plot_selection_convergence(
    tables: dict[str, pd.DataFrame],
    budgets: list[int],
    output_dir: str,
    classification_by_threshold: dict[float, pd.DataFrame] | None = None,
) -> None:
    """
    For each non-classification table type, plot random_B and vm_B vs budget alongside
    the constant true_downstream line, averaged over (dataset, axis).

    For classification, generate one separate plot per threshold in
    classification_by_threshold (keys are threshold values).

    One figure per table type / threshold; one subplot per (our_metric × downstream_metric) pair.
    """
    _TABLE_TITLES = {
        "spearman":   "Spearman ρ across judges",
        "pearson":    "Pearson r across judges",
        "best_judge": "Best judge identification (frac correct)",
    }

    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    valid_budgets = [b for b in budgets if b > 0]

    # --- Non-classification tables ---
    for ttype, title in _TABLE_TITLES.items():
        if ttype not in tables:
            continue
        _plot_convergence_table(
            tables[ttype], title, valid_budgets, out,
            fname=f"convergence_{ttype}.png",
        )

    # --- Classification: one plot per threshold ---
    if classification_by_threshold:
        for t, df in sorted(classification_by_threshold.items()):
            t_str = f"{t:.2f}".rstrip("0").rstrip(".")
            _plot_convergence_table(
                df,
                title=f"Classification agreement (T={t})",
                valid_budgets=valid_budgets,
                out=out,
                fname=f"convergence_classification_T{t_str}.png",
            )
    elif "classification" in tables:
        # Fallback: single plot using whatever threshold was baked into the table
        _plot_convergence_table(
            tables["classification"],
            title=f"Classification agreement (T={CLASSIFICATION_THRESHOLD})",
            valid_budgets=valid_budgets,
            out=out,
            fname="convergence_classification.png",
        )

    # --- Per-dataset disaggregated plots ---
    all_datasets = sorted(
        set().union(*[
            set(df["dataset"].unique())
            for df in list(tables.values()) + list((classification_by_threshold or {}).values())
            if not df.empty and "dataset" in df.columns
        ])
    )
    for dataset in all_datasets:
        ds_out = out / dataset
        ds_out.mkdir(parents=True, exist_ok=True)

        for ttype, title in _TABLE_TITLES.items():
            if ttype not in tables:
                continue
            ds_df = tables[ttype][tables[ttype]["dataset"] == dataset]
            if ds_df.empty:
                continue
            _plot_convergence_table(
                ds_df, f"{title} — {dataset}", valid_budgets, ds_out,
                fname=f"convergence_{ttype}.png",
            )

        if classification_by_threshold:
            for t, df in sorted(classification_by_threshold.items()):
                t_str = f"{t:.2f}".rstrip("0").rstrip(".")
                ds_df = df[df["dataset"] == dataset]
                if ds_df.empty:
                    continue
                _plot_convergence_table(
                    ds_df,
                    title=f"Classification agreement (T={t}) — {dataset}",
                    valid_budgets=valid_budgets,
                    out=ds_out,
                    fname=f"convergence_classification_T{t_str}.png",
                )
        elif "classification" in tables:
            ds_df = tables["classification"][tables["classification"]["dataset"] == dataset]
            if not ds_df.empty:
                _plot_convergence_table(
                    ds_df,
                    title=f"Classification agreement (T={CLASSIFICATION_THRESHOLD}) — {dataset}",
                    valid_budgets=valid_budgets,
                    out=ds_out,
                    fname="convergence_classification.png",
                )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_pivot(df: pd.DataFrame, value_col: str, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    pivot = df.pivot(index="our_metric", columns="downstream_metric", values=value_col)
    pivot = pivot.reindex(index=OUR_METRICS, columns=DOWNSTREAM_METRICS)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


def save_results(
    metrics_df: pd.DataFrame,
    spearman_table: pd.DataFrame,
    pearson_table: pd.DataFrame,
    best_judge_table: pd.DataFrame,
    classification_table: pd.DataFrame,
    output_dir: str,
    nonagg_corr: pd.DataFrame | None = None,
    agg_corr: pd.DataFrame | None = None,
) -> None:
    out = Path(output_dir) / "dataframes"
    out.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out / "per_judge_metrics.csv", index=False)
    print(f"\nSaved per-judge metrics → {out / 'per_judge_metrics.csv'}")

    for fname, df in [
        ("eval_spearman.csv",       spearman_table),
        ("eval_pearson.csv",        pearson_table),
        ("eval_best_judge.csv",     best_judge_table),
        ("eval_classification.csv", classification_table),
    ]:
        df.to_csv(out / fname, index=False)
        print(f"Saved {fname} → {out / fname}")

    if nonagg_corr is not None:
        nonagg_corr.to_csv(out / "corpus_corr_nonagg.csv", index=False)
        print(f"Saved corpus_corr_nonagg.csv → {out / 'corpus_corr_nonagg.csv'}")
    if agg_corr is not None:
        agg_corr.to_csv(out / "corpus_corr_agg.csv", index=False)
        print(f"Saved corpus_corr_agg.csv → {out / 'corpus_corr_agg.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge quality analysis: ICC/alpha vs downstream metrics"
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument(
        "--selection-results-dir",
        default=None,
        help=(
            "Root folder containing ICC/alpha sampling results "
            "(e.g. results/04_01_downstream_task). "
            "When provided, runs the selection-method analysis (8 additional tables). "
            f"Defaults to '{DEFAULT_SELECTION_RESULTS_DIR}' when flag is given without a value."
        ),
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=BUDGETS_SUBSET,
        help="Budgets for selection analysis (default: 5 10 15)",
    )
    parser.add_argument(
        "--vm-method", default=VM_METHOD,
        help=f"Variance-matching method name to compare against random (default: {VM_METHOD})",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.6, 0.7, 0.8],
        help="Classification thresholds; one convergence_classification plot is produced per "
             "threshold (default: 0.6 0.7 0.8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global DATASETS, VM_METHOD
    DATASETS = args.datasets
    VM_METHOD = args.vm_method

    print(f"Datasets : {DATASETS}")
    print(f"Models   : {args.models}")
    print(f"Classification thresholds: {args.thresholds}")

    metrics_df = run_analysis(data_dir=args.data_dir, models=args.models)

    if metrics_df.empty:
        print("No results produced. Check --data-dir and model names.")
        return

    # --- Raw upstream values + per-model averages ----------------------------
    print_raw_upstream_values(metrics_df)

    # --- Corpus-level Spearman/Pearson correlations --------------------------
    nonagg_corr, agg_corr = compute_corpus_level_correlations(metrics_df)
    print_corpus_correlations(nonagg_corr, agg_corr)

    print(f"\nPer-judge metrics table: {len(metrics_df)} rows")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nComputing evaluation tables...")
    # Non-classification tables don't depend on threshold — compute once
    spearman_table, pearson_table, best_judge_table, _ = \
        compute_eval_tables(metrics_df, threshold=args.thresholds[0])

    _print_pivot(
        spearman_table, "avg_spearman_rho",
        "Table 1 — Spearman ρ: how well does our metric rank judges vs. downstream? "
        "(Spearman ρ across judges per dataset-axis, averaged)",
    )
    _print_pivot(
        pearson_table, "avg_pearson_r",
        "Table 2 — Pearson r: linear correlation between our metric and downstream metric "
        "across judges (per dataset-axis, averaged)",
    )
    _print_pivot(
        best_judge_table, "frac_best_judge_match",
        "Table 3 — Best judge identification: fraction of dataset-axes where "
        "argbest(our metric) == argbest(downstream metric)",
    )

    # Classification table: one per threshold
    eval_class_by_threshold: dict[float, pd.DataFrame] = {}
    for t in args.thresholds:
        _, _, _, class_table_t = compute_eval_tables(metrics_df, threshold=t)
        eval_class_by_threshold[t] = class_table_t
        _print_pivot(
            class_table_t, "frac_classification_agree",
            f"Table 4 — Classification (T={t}): fraction of "
            "(dataset-axis, judge) pairs where our metric and downstream metric agree "
            "on above/below threshold",
        )

    # Save non-classification tables + per-threshold classification CSVs
    out_df = Path(args.output_dir) / "dataframes"
    out_df.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_df / "per_judge_metrics.csv", index=False)
    print(f"\nSaved per-judge metrics → {out_df / 'per_judge_metrics.csv'}")

    for fname, df in [
        ("eval_spearman.csv",   spearman_table),
        ("eval_pearson.csv",    pearson_table),
        ("eval_best_judge.csv", best_judge_table),
    ]:
        df.to_csv(out_df / fname, index=False)
        print(f"Saved {fname} → {out_df / fname}")

    for t, df_t in eval_class_by_threshold.items():
        t_str = f"{t:.2f}".rstrip("0").rstrip(".")
        fname = f"eval_classification_T{t_str}.csv"
        df_t.to_csv(out_df / fname, index=False)
        print(f"Saved {fname} → {out_df / fname}")

    if nonagg_corr is not None:
        nonagg_corr.to_csv(out_df / "corpus_corr_nonagg.csv", index=False)
        print(f"Saved corpus_corr_nonagg.csv → {out_df / 'corpus_corr_nonagg.csv'}")
    if agg_corr is not None:
        agg_corr.to_csv(out_df / "corpus_corr_agg.csv", index=False)
        print(f"Saved corpus_corr_agg.csv → {out_df / 'corpus_corr_agg.csv'}")

    # ---- Selection-method analysis (8 additional tables) --------------------
    sel_dir = args.selection_results_dir
    if sel_dir is None:
        # Check if default location exists; skip silently if not
        sel_dir = DEFAULT_SELECTION_RESULTS_DIR
        if not Path(sel_dir).exists():
            return

    print(f"\n{'='*70}")
    print("Selection-method analysis (ICC/alpha estimated from subsets)")
    print(f"Results dir : {sel_dir}")
    print(f"Budgets     : {args.budgets}")
    print(f"VM method   : {VM_METHOD}")
    print("=" * 70)

    icc_sel_df, alpha_sel_df, rho_sel_df, tau_sel_df = load_selection_results(sel_dir)

    if icc_sel_df.empty and alpha_sel_df.empty and rho_sel_df.empty and tau_sel_df.empty:
        print("No selection results found. Skipping selection analysis.")
        return

    # Downstream metrics from the main analysis (full-data, true human labels)
    # Includes icc and krippendorff_alpha so selection eval tables can reference them.
    downstream_df = metrics_df[
        ["dataset", "axis", "model",
         "spearman_rho", "pearson_r", "mse", "mean_acc",
         "icc", "krippendorff_alpha", "kendall_tau"]
    ].copy()

    print("\nComputing prediction spread diagnostic...")
    spread_df = compute_prediction_spread(
        icc_sel_df, alpha_sel_df, budgets=args.budgets,
        rho_df=rho_sel_df if not rho_sel_df.empty else None,
        tau_df=tau_sel_df if not tau_sel_df.empty else None,
        metrics_df=metrics_df,
    )
    print_prediction_spread(spread_df)
    spread_out = Path(args.output_dir) / "dataframes" / "selection_prediction_spread.csv"
    spread_out.parent.mkdir(parents=True, exist_ok=True)
    spread_df.to_csv(spread_out, index=False)
    print(f"Saved prediction spread → {spread_out}")

    thresholds = args.thresholds
    print(f"Classification thresholds: {thresholds}")

    _rho_arg = rho_sel_df if not rho_sel_df.empty else None
    _tau_arg = tau_sel_df if not tau_sel_df.empty else None

    print("\nComputing selection evaluation tables (non-classification)...")
    selection_tables = compute_selection_eval_tables(
        icc_sel_df, alpha_sel_df, downstream_df, metrics_df, budgets=args.budgets,
        threshold=thresholds[0],  # threshold only affects classification table
        rho_df=_rho_arg, tau_df=_tau_arg,
    )
    non_class_tables = {k: v for k, v in selection_tables.items() if k != "classification"}

    print(f"\nComputing classification tables for {len(thresholds)} threshold(s): {thresholds}")
    classification_by_threshold: dict[float, pd.DataFrame] = {}
    for t in thresholds:
        tables_t = compute_selection_eval_tables(
            icc_sel_df, alpha_sel_df, downstream_df, metrics_df, budgets=args.budgets,
            threshold=t, rho_df=_rho_arg, tau_df=_tau_arg,
        )
        classification_by_threshold[t] = tables_t["classification"]
        print(f"  T={t}: {len(tables_t['classification'])} rows")

    print_selection_tables(selection_tables, budgets=args.budgets)
    save_selection_results(selection_tables, args.output_dir)
    # Also save per-threshold classification CSVs
    class_out = Path(args.output_dir) / "dataframes"
    class_out.mkdir(parents=True, exist_ok=True)
    for t, df_t in classification_by_threshold.items():
        t_str = f"{t:.2f}".rstrip("0").rstrip(".")
        fname = class_out / f"selection_classification_T{t_str}.csv"
        df_t.to_csv(fname, index=False)
        print(f"Saved classification T={t} → {fname}")

    print("\nGenerating convergence plots...")
    plot_selection_convergence(
        non_class_tables,
        budgets=args.budgets,
        output_dir=args.output_dir,
        classification_by_threshold=classification_by_threshold,
    )
    print("\nSelection analysis complete.")


if __name__ == "__main__":
    main()
