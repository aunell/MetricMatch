"""
Meta-correlation analysis: predictor summary CSV generation.

Reads a predictor_records.csv produced by run_predictor_scatter.py and computes
three sets of summary files:

    predictor_summary_{metric}.csv          — per (dataset, axis, model) row with
                                              best method at budget=30 and predictor
                                              correlations.

    predictor_summary_{metric}_overall.csv  — same rows but using the single globally
                                              best method (selected by lowest average
                                              gap over all triples and budgets).

    predictor_meta_summary.csv              — one row per metric with Pearson
                                              correlations between predictors, human
                                              agreement, and the performance gap.

Usage:
    python src/experiments/meta_correlation_analysis.py \\
        --input-csv results/03_25/predictor_scatter/predictor_records.csv \\
        --output-dir results/03_25/predictor_scatter \\
        --human-agreement results/03_25_human_aggreement/human_agreement.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def is_oracle(method):
    """Return True if the method name contains 'oracle'."""
    return "oracle" in method.lower()


def load_human_agreement(human_agreement_path):
    """Load and normalise the human_agreement CSV into a lookup dict keyed by (dataset, axis)."""
    if human_agreement_path is None or not os.path.exists(human_agreement_path):
        return {}
    ha = pd.read_csv(human_agreement_path)
    lookup = {}
    for _, row in ha.iterrows():
        key = (str(row["dataset"]).strip(), str(row["axis"]).strip().lower())
        lookup[key] = {
            "human_icc":   row.get("icc", np.nan),
            "human_alpha": row.get("krippendorff_alpha", np.nan),
            "human_mse":   row.get("mse", np.nan),
        }
    return lookup


def pearsonr_safe(x, y):
    """Return (r, p) or (nan, nan) if fewer than 3 valid pairs."""
    from scipy import stats
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(valid["x"], valid["y"])
    return round(float(r), 4), round(float(p), 4)


# ---------------------------------------------------------------------------
# Per-metric summary
# ---------------------------------------------------------------------------

def save_predictor_summary(df, output_dir, metric="icc", human_agreement_path=None):
    """
    Build and save predictor_summary_{metric}.csv.

    One row per (dataset, axis, model) with columns:
        Dataset, Axis, Model,
        Budget=30 ({Metric}),
        Best method (budget=30),
        MSB corr (budget=30, n=10),
        MSE corr (budget=30, n=10),
        ICC corr (budget=30, n=10),
        human_icc, human_alpha, human_mse

    Args:
        df: predictor_records DataFrame.
        output_dir: Directory to write the CSV into.
        metric: One of "icc", "alpha", or "mse".
        human_agreement_path: Optional path to human_agreement.csv.
    """
    gap_prefix = f"{metric}_gap_"
    metric_label = {"icc": "ICC", "alpha": "Alpha", "mse": "MSE"}.get(metric, metric.upper())

    if not any(c.startswith(gap_prefix) for c in df.columns):
        print(f"No {gap_prefix}* columns found; skipping predictor_summary_{metric}.csv")
        return

    human_lookup = load_human_agreement(human_agreement_path)

    # Methods available at budget=30 (excluding oracle variants)
    b30_methods = sorted(
        m for m in (
            c[len(gap_prefix):c.rfind("_b30")]
            for c in df.columns
            if c.startswith(gap_prefix) and c.endswith("_b30")
        )
        if not is_oracle(m)
    )

    summary_rows = []
    for _, row in df.iterrows():
        dataset = row.get("dataset", "")
        axis = row["axis"]
        model = row["model"]

        # Best method at budget=30 (lowest gap = most improvement over random)
        best_method_b30, best_val_b30 = None, np.nan
        for method in b30_methods:
            col = f"{gap_prefix}{method}_b30"
            val = row.get(col, np.nan)
            if np.isfinite(val) and (best_method_b30 is None or val < best_val_b30):
                best_method_b30, best_val_b30 = method, val

        ha = human_lookup.get((dataset, axis), {})
        rec = {
            "Dataset": dataset,
            "Axis": axis,
            "Model": model,
            f"Budget=30 ({metric_label})": best_val_b30,
            "Best method (budget=30)": best_method_b30 if best_method_b30 else "",
            "MSB corr (budget=30, n=10)": row.get("correlation_msb_b30", np.nan),
            "MSE corr (budget=30, n=10)": row.get("correlation_mse_b30", np.nan),
            "ICC corr (budget=30, n=10)": row.get("correlation_icc_b30", np.nan),
            "human_icc":   ha.get("human_icc",   np.nan),
            "human_alpha": ha.get("human_alpha", np.nan),
            "human_mse":   ha.get("human_mse",   np.nan),
        }
        summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)
    filename = f"predictor_summary_{metric}.csv"
    summary_path = os.path.join(output_dir, filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {filename} ({len(summary_df)} rows) → {summary_path}")


# ---------------------------------------------------------------------------
# Overall-best-method summary
# ---------------------------------------------------------------------------

def save_predictor_summary_metric_overall(df, output_dir, metric="icc",
                                          human_agreement_path=None):
    """
    Build and save predictor_summary_{metric}_overall.csv.

    Selects the single globally best method by averaging its gap over random
    across ALL (dataset, axis, model) triples and ALL budgets, then reports
    how that method performs in each triple at every budget.

    One row per (dataset, axis, model) with columns:
        Dataset, Axis, Model,
        Overall best method,
        Gap avg all budgets ({Metric}),
        Gap budget=B ({Metric})  for each discovered budget B,
        MSB corr (budget=30, n=10),
        MSE corr (budget=30, n=10),
        ICC corr (budget=30, n=10),
        human_icc, human_alpha, human_mse

    Args:
        df: predictor_records DataFrame.
        output_dir: Directory to write the CSV into.
        metric: One of "icc", "alpha", or "mse".
        human_agreement_path: Optional path to human_agreement.csv.
    """
    import re
    gap_prefix = f"{metric}_gap_"
    metric_label = {"icc": "ICC", "alpha": "Alpha", "mse": "MSE"}.get(metric, metric.upper())

    if not any(c.startswith(gap_prefix) for c in df.columns):
        print(f"No {gap_prefix}* columns found; skipping predictor_summary_{metric}_overall.csv")
        return

    human_lookup = load_human_agreement(human_agreement_path)

    # Discover all budgets from per-budget gap columns
    budget_pattern = re.compile(r"_b(\d+)$")
    budgets = set()
    for col in df.columns:
        m = budget_pattern.search(col)
        if m and col.startswith(gap_prefix):
            budgets.add(int(m.group(1)))
    budgets = sorted(budgets)

    # Discover all non-oracle methods that have at least one per-budget column
    methods = set()
    for col in df.columns:
        if col.startswith(gap_prefix):
            m = budget_pattern.search(col)
            if m:
                method_name = col[len(gap_prefix):col.rfind(f"_b{m.group(1)}")]
                if not is_oracle(method_name):
                    methods.add(method_name)
    methods = sorted(methods)

    if not methods:
        print(f"No non-oracle methods found; skipping predictor_summary_{metric}_overall.csv")
        return

    # For each method, compute its average gap across ALL rows and ALL budgets.
    method_avg_gaps = {}
    for method in methods:
        budget_cols = [f"{gap_prefix}{method}_b{b}" for b in budgets
                       if f"{gap_prefix}{method}_b{b}" in df.columns]
        if not budget_cols:
            continue
        all_vals = df[budget_cols].values.flatten()
        valid_vals = all_vals[np.isfinite(all_vals)]
        method_avg_gaps[method] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan

    # Select the globally best method (most negative = most improvement over random)
    best_method = min(
        (m for m, v in method_avg_gaps.items() if np.isfinite(v)),
        key=lambda m: method_avg_gaps[m],
        default=None,
    )
    if best_method is None:
        print(f"Could not determine globally best method for {metric}; skipping.")
        return

    print(f"[{metric}_overall] Globally best method: {best_method} "
          f"(mean gap across all triples & budgets = {method_avg_gaps[best_method]:.4f})")

    # Build one summary row per (dataset, axis, model) triple
    summary_rows = []
    for _, row in df.iterrows():
        dataset = row.get("dataset", "")
        axis = row["axis"]
        model = row["model"]

        budget_cols = [f"{gap_prefix}{best_method}_b{b}" for b in budgets
                       if f"{gap_prefix}{best_method}_b{b}" in df.columns]
        per_budget_vals = [row.get(col, np.nan) for col in budget_cols]
        finite_vals = [v for v in per_budget_vals if np.isfinite(v)]
        overall_gap = float(np.mean(finite_vals)) if finite_vals else np.nan

        ha = human_lookup.get((dataset, axis), {})
        rec = {
            "Dataset": dataset,
            "Axis": axis,
            "Model": model,
            "Overall best method": best_method,
            f"Gap avg all budgets ({metric_label})": overall_gap,
        }
        for b in budgets:
            col = f"{gap_prefix}{best_method}_b{b}"
            rec[f"Gap budget={b} ({metric_label})"] = row.get(col, np.nan)
        rec.update({
            "MSB corr (budget=30, n=10)": row.get("correlation_msb_b30", np.nan),
            "MSE corr (budget=30, n=10)": row.get("correlation_mse_b30", np.nan),
            "ICC corr (budget=30, n=10)": row.get("correlation_icc_b30", np.nan),
            "human_icc":   ha.get("human_icc",   np.nan),
            "human_alpha": ha.get("human_alpha", np.nan),
            "human_mse":   ha.get("human_mse",   np.nan),
        })
        summary_rows.append(rec)

    summary_df = pd.DataFrame(summary_rows)
    filename = f"predictor_summary_{metric}_overall.csv"
    summary_path = os.path.join(output_dir, filename)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {filename} ({len(summary_df)} rows) → {summary_path}")


# ---------------------------------------------------------------------------
# Meta-summary
# ---------------------------------------------------------------------------

def save_predictor_meta_summary(output_dir):
    """
    Read all predictor_summary_{metric}.csv files and compute, for each metric,
    Pearson correlations between:
      - each predictor (MSB corr, MSE corr, ICC corr) and the budget=30 gap
      - each human agreement column (human_icc, human_alpha, human_mse) and the gap
      - each predictor and each human agreement column

    Saves predictor_meta_summary.csv with one row per metric.
    """
    predictor_cols = [
        "MSB corr (budget=30, n=10)",
        "MSE corr (budget=30, n=10)",
        "ICC corr (budget=30, n=10)",
    ]
    human_cols = ["human_icc", "human_alpha", "human_mse"]

    meta_rows = []
    for metric in ("icc", "alpha", "mse"):
        metric_label = {"icc": "ICC", "alpha": "Alpha", "mse": "MSE"}[metric]
        summary_path = os.path.join(output_dir, f"predictor_summary_{metric}.csv")
        if not os.path.exists(summary_path):
            print(f"  Skipping meta-summary for {metric}: {summary_path} not found")
            continue

        sdf = pd.read_csv(summary_path)
        gap_col = f"Budget=30 ({metric_label})"
        if gap_col not in sdf.columns:
            print(f"  Skipping meta-summary for {metric}: column '{gap_col}' not found")
            continue

        rec = {"Metric": metric_label, "N": int(sdf[gap_col].notna().sum())}

        # Predictor vs gap
        for pred_col in predictor_cols:
            short = pred_col.split(" ")[0]  # "MSB", "MSE", "ICC"
            r, p = pearsonr_safe(sdf.get(pred_col), sdf[gap_col])
            rec[f"r({short} corr~gap)"] = r
            rec[f"p({short} corr~gap)"] = p

        # Human agreement vs gap
        for hcol in human_cols:
            if hcol not in sdf.columns:
                rec[f"r({hcol}~gap)"] = np.nan
                rec[f"p({hcol}~gap)"] = np.nan
                continue
            r, p = pearsonr_safe(sdf[hcol], sdf[gap_col])
            rec[f"r({hcol}~gap)"] = r
            rec[f"p({hcol}~gap)"] = p

        # Predictor vs human agreement
        for pred_col in predictor_cols:
            short = pred_col.split(" ")[0]
            for hcol in human_cols:
                if hcol not in sdf.columns:
                    rec[f"r({short} corr~{hcol})"] = np.nan
                    rec[f"p({short} corr~{hcol})"] = np.nan
                    continue
                r, p = pearsonr_safe(sdf.get(pred_col), sdf[hcol])
                rec[f"r({short} corr~{hcol})"] = r
                rec[f"p({short} corr~{hcol})"] = p

        meta_rows.append(rec)

    if not meta_rows:
        print("No meta-summary rows produced.")
        return

    meta_df = pd.DataFrame(meta_rows)
    meta_path = os.path.join(output_dir, "predictor_meta_summary.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved predictor_meta_summary.csv ({len(meta_df)} rows) → {meta_path}")
    print(meta_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Per-triple-budget CSVs
# ---------------------------------------------------------------------------

_OUTCOME_COL = "Best method improvement at target budget vs random"

_PREDICTOR_COL_MAP = {
    "_RAW Model-Model MSB":              "im_msb",
    "_RAW Model-Model MSE":              "im_mse",
    "_RAW Model-Model ICC":              "im_icc",
    "_inter-Human ICC":                  None,   # from human_lookup
    "_inter-Human Krippendorff Alpha":   None,   # from human_lookup
    "_inter-Human Mean squared error":   None,   # from human_lookup
    "_Human,model MSB corr (b=30,n=10)": "correlation_msb_b30",
    "_Human,model MSE corr (b=30,n=10)": "correlation_mse_b30",
    "_Human,model ICC corr (b=30,n=10)": "correlation_icc_b30",
}


def save_per_triple_budget_csvs(df, triple_csv_dir, human_agreement_path=None):
    """
    For each (dataset, axis, model, budget) combination save one CSV with three
    rows — one per metric (ICC, Alpha, MSE).

    Columns
    -------
    Metric, Dataset, Axis, Model, Target Budget,
    Best method name at budget,
    Best method improvement at target budget vs random,   ← outcome
    _RAW Model-Model MSB,
    _RAW Model-Model MSE,
    _RAW Model-Model ICC,
    _inter-Human ICC,
    _inter-Human Krippendorff Alpha,
    _inter-Human Mean squared error,
    _Human,model MSB corr (b=30,n=10),
    _Human,model MSE corr (b=30,n=10),
    _Human,model ICC corr (b=30,n=10)

    Files are saved as  {triple_csv_dir}/{dataset}_{axis}_{model}_b{budget}.csv
    """
    import re
    budget_pattern = re.compile(r"_b(\d+)$")

    human_lookup = load_human_agreement(human_agreement_path)

    # Discover all budgets present in per-budget gap columns
    budgets = set()
    for col in df.columns:
        m = budget_pattern.search(col)
        if m and any(col.startswith(f"{met}_gap_") for met in ("icc", "alpha", "mse")):
            budgets.add(int(m.group(1)))
    budgets = sorted(budgets)

    if not budgets:
        print("No per-budget gap columns found; skipping per-triple-budget CSVs.")
        return

    # Pre-compute available non-oracle methods per metric per budget
    avail_methods: dict = {}
    for metric in ("icc", "alpha", "mse"):
        gap_prefix = f"{metric}_gap_"
        avail_methods[metric] = {}
        for b in budgets:
            suffix = f"_b{b}"
            avail_methods[metric][b] = [
                col[len(gap_prefix):col.rfind(suffix)]
                for col in df.columns
                if col.startswith(gap_prefix) and col.endswith(suffix)
                and not is_oracle(col[len(gap_prefix):col.rfind(suffix)])
            ]

    os.makedirs(triple_csv_dir, exist_ok=True)
    n_saved = 0

    for _, row in df.iterrows():
        dataset = str(row.get("dataset", ""))
        axis    = str(row["axis"]).lower()
        model   = str(row["model"])

        ha = human_lookup.get((dataset, axis), {})

        # Predictor values that don't vary with budget or metric for this triple
        raw_shared = {
            "_RAW Model-Model MSB": row.get("im_msb", np.nan),
            "_RAW Model-Model MSE": row.get("im_mse", np.nan),
            "_RAW Model-Model ICC": row.get("im_icc", np.nan),
            "_inter-Human ICC":                ha.get("human_icc",   np.nan),
            "_inter-Human Krippendorff Alpha": ha.get("human_alpha", np.nan),
            "_inter-Human Mean squared error": ha.get("human_mse",   np.nan),
        }

        for budget in budgets:
            rows = []
            for metric in ("icc", "alpha", "mse"):
                gap_prefix   = f"{metric}_gap_"
                metric_label = {"icc": "ICC", "alpha": "Alpha", "mse": "MSE"}[metric]

                best_method, best_val = None, np.nan
                for method in avail_methods[metric][budget]:
                    val = row.get(f"{gap_prefix}{method}_b{budget}", np.nan)
                    if pd.notna(val) and (best_method is None or val < best_val):
                        best_method, best_val = method, val

                rows.append({
                    "Metric":       metric_label,
                    "Dataset":      dataset,
                    "Axis":         axis,
                    "Model":        model,
                    "Target Budget": budget,
                    "Best method name at budget": best_method or "",
                    _OUTCOME_COL:   best_val,
                    **raw_shared,
                    "_Human,model MSB corr (b={},n=10)".format(budget):
                        row.get(f"correlation_msb_b{budget}", np.nan),
                    "_Human,model MSE corr (b={},n=10)".format(budget):
                        row.get(f"correlation_mse_b{budget}", np.nan),
                    "_Human,model ICC corr (b={},n=10)".format(budget):
                        row.get(f"correlation_icc_b{budget}", np.nan),
                })

            safe_axis  = axis.replace("/", "_").replace(" ", "_")
            safe_model = model.replace("/", "_").replace(" ", "_")
            fname = f"{dataset}_{safe_axis}_{safe_model}_b{budget}.csv"
            pd.DataFrame(rows).to_csv(os.path.join(triple_csv_dir, fname), index=False)
            n_saved += 1

    print(f"Saved {n_saved} per-triple-budget CSVs → {triple_csv_dir}")


# ---------------------------------------------------------------------------
# Overview correlation CSVs
# ---------------------------------------------------------------------------

def _corr_row(pred, sub, outcome_col=_OUTCOME_COL):
    r, p = pearsonr_safe(sub[pred], sub[outcome_col])
    n = int(sub[[pred, outcome_col]].dropna().__len__())
    return {"Predictor": pred, "r": r, "p": p, "N": n}


def save_overview_correlations(triple_csv_dir, output_dir):
    """
    Load all per-triple-budget CSVs, combine them, and write six overview tables:

    overview_corr_raw.csv               – correlations across all rows/metrics
    overview_corr_by_metric.csv         – one block per metric (ICC/Alpha/MSE)
    overview_corr_by_budget.csv         – one block per target budget
    overview_corr_by_budget_wide.csv    – predictors as rows, budgets as columns
    overview_corr_by_triple.csv         – one block per (Dataset, Axis, Model)
    overview_corr_by_triple_wide.csv    – triples as rows, predictors as columns

    Each "block" reports the Pearson r between the predictor and
    "Best method improvement at target budget vs random".
    """
    import glob as _glob

    csv_files = _glob.glob(os.path.join(triple_csv_dir, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {triple_csv_dir}; skipping overview correlations.")
        return

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not frames:
        print("No valid per-triple-budget CSVs loaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    predictor_cols = [c for c in combined.columns if c.startswith("_")]

    def _save(rows, fname):
        path = os.path.join(output_dir, fname)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Saved {fname}")

    # 1. Raw (all rows, all metrics)
    _save([_corr_row(p, combined) for p in predictor_cols],
          "overview_corr_raw.csv")

    # 2. By metric
    metric_rows = []
    for metric, sub in combined.groupby("Metric"):
        for p in predictor_cols:
            r = _corr_row(p, sub)
            r["Metric"] = metric
            metric_rows.append(r)
    _save(metric_rows, "overview_corr_by_metric.csv")

    # 3. By budget  (long)
    budget_rows = []
    for budget in sorted(combined["Target Budget"].unique()):
        sub = combined[combined["Target Budget"] == budget]
        for p in predictor_cols:
            r = _corr_row(p, sub)
            r["Budget"] = budget
            budget_rows.append(r)
    _save(budget_rows, "overview_corr_by_budget.csv")

    # 4. By budget  (wide: predictors × budgets)
    if budget_rows:
        bdf   = pd.DataFrame(budget_rows)
        wide  = bdf.pivot_table(index="Predictor", columns="Budget", values="r")
        wide.columns = [f"r_b{b}" for b in wide.columns]
        _save(wide.reset_index().to_dict("records"),
              "overview_corr_by_budget_wide.csv")

    # 5. By (dataset, axis, model) triple  (long)
    triple_rows = []
    for (dataset, axis, model), sub in combined.groupby(["Dataset", "Axis", "Model"]):
        for p in predictor_cols:
            r = _corr_row(p, sub)
            r.update({"Dataset": dataset, "Axis": axis, "Model": model})
            triple_rows.append(r)
    _save(triple_rows, "overview_corr_by_triple.csv")

    # 6. By triple  (wide: triples × predictors)
    if triple_rows:
        tdf  = pd.DataFrame(triple_rows)
        wide = tdf.pivot_table(
            index=["Dataset", "Axis", "Model"], columns="Predictor", values="r"
        )
        wide.columns = [f"r_{p}" for p in wide.columns]
        _save(wide.reset_index().to_dict("records"),
              "overview_corr_by_triple_wide.csv")

    print(f"\nOverview correlation CSVs saved → {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(df, output_dir, human_agreement_path=None):
    """
    Run the full meta-correlation analysis pipeline.

    Stage 1 — per (dataset, axis, model, budget) CSVs:
        One CSV per combination (3 rows × n_budgets), saved under
        output_dir/per_triple_budget/.

    Stage 2 — overview correlation CSVs:
        Six tables correlating each predictor with the outcome, sliced by
        metric / budget / triple.

    Stage 3 — legacy per-metric summary CSVs (predictor_summary_*.csv,
        predictor_summary_*_overall.csv, predictor_meta_summary.csv).

    Args:
        df: predictor_records DataFrame produced by compute_predictor_records.
        output_dir: Root directory to write all outputs into.
        human_agreement_path: Optional path to human_agreement.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: per-triple-budget CSVs
    triple_csv_dir = os.path.join(output_dir, "per_triple_budget")
    print("\n[Stage 1] Generating per-triple-budget CSVs ...")
    save_per_triple_budget_csvs(df, triple_csv_dir,
                                human_agreement_path=human_agreement_path)

    # Stage 2: overview correlation tables
    print("\n[Stage 2] Generating overview correlation tables ...")
    save_overview_correlations(triple_csv_dir, output_dir)

    # Stage 3: legacy summary CSVs
    print("\n[Stage 3] Generating legacy summary CSVs ...")
    for metric in ("icc", "alpha", "mse"):
        save_predictor_summary(df, output_dir, metric=metric,
                               human_agreement_path=human_agreement_path)

    for metric in ("icc", "alpha", "mse"):
        save_predictor_summary_metric_overall(df, output_dir, metric=metric,
                                              human_agreement_path=human_agreement_path)

    save_predictor_meta_summary(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True,
                        help="Path to predictor_records.csv produced by run_predictor_scatter.py")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save summary CSVs "
                             "(default: same directory as --input-csv)")
    parser.add_argument("--human-agreement", default=None,
                        help="Path to human_agreement.csv with columns: "
                             "dataset, axis, icc, krippendorff_alpha, mse")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"ERROR: input CSV not found: {args.input_csv}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_csv))

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    run(df, output_dir, human_agreement_path=args.human_agreement)

    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
