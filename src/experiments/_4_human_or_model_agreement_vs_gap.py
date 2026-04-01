"""
Calculate inter-human and human-model agreement metrics (ICC, Krippendorff's alpha, MSE)
for each axis and dataset, then plot predictor vs gap scatter figures.

Datasets:
- hanna (hanna_stories_annotations.csv): 1056 stories, 3 crowd workers, 6 axes
- mslr (data_with_overlap_scores.json): 39 items with 2 annotators, 4 axes
- summeval (HuggingFace mteb/summeval): scores are averaged over 3 expert raters
    — individual rater scores are not available in the HF dataset, so ICC/alpha/MSE
      cannot be computed. Rows are included with NaN values and a note.
- inter_physician (inter_physician.csv): 89 items, 2-5 physicians per task

Output: results/03_25_human_aggreement/human_agreement.csv
        results/03_25_human_aggreement/plots/
"""

from __future__ import annotations

import json
import sys
from functools import partial
from itertools import combinations

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils.reliability_metrics import compute_ms_components


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parents[2] / "data" / "raw_data"
OUTPUT_DIR = Path(__file__).parents[2] / "results" / "03_30_deepseek_gemini"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_SCORES_DIR = Path(__file__).parents[2] / "data" / "judge_scores"
RESULTS_DIR = Path(__file__).parents[2] / "results" / "03_30_deepseek_gemini"

# Models used for IM and HM correlation computations
MODELS_TO_PLOT = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5", "deepseek-r1", "gemini-2.5"]
MODEL_COLORS = {
    "claude-3.5-sonnet": "#E91E63",
    "gpt-4.1":           "#FF9800",
    "gpt-5":             "#4CAF50",
    "deepseek-r1":       "#9C27B0",
    "gemini-2.5":        "#009688",
}
HUMAN_COLOR = "#2196F3"

# Axes that have judge score files (inter_physician has no LLM judge scores)
DATASET_AXES = {
    "hanna":    ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"],
    "mslr":     ["fluency", "population", "intervention", "outcome"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"],
    "medval":   ["Risk"],
}

# Bootstrap parameters for HM MSB/MSE correlation computation
N_CORR_SAMPLES = 30
CORR_SAMPLE_SIZE = 30

# Plotting constants
MODEL_MARKERS = {
    "claude-3.5-sonnet": "o",
    "gpt-4.1":           "s",
    "gpt-5":             "^",
    "deepseek-r1":       "D",
    "gemini-2.5":        "P",
}

DATASET_COLORS = {
    "hanna":    "#E91E63",
    "mslr":     "#FF9800",
    "summeval": "#4CAF50",
    "medval":   "#2196F3",
}

# Scatter plot axes: (column_name, display_label)
X_VARS = [
    ("icc",                 "Human ICC"),
    ("krippendorff_alpha",  "Human Alpha"),
    ("mse",                 "Human MSE"),
    ("human_msb_mse_ratio", "Human MSB/MSE"),
    ("im_msb",              "IM MSB"),
    ("im_mse",              "IM MSE"),
    ("im_icc",              "IM ICC"),
    ("hm_msb_mse_ratio",    "HM MSB/MSE"),
    ("hm_msb_corr",         "HM MSB Corr"),
    ("hm_mse_corr",         "HM MSE Corr"),
    ("hm_icc_corr",         "HM ICC Corr"),
]

# (gap_col, best_method_col, display_label)
Y_VARS = [
    ("avg_gap_icc",   "best_method_icc",   "Avg Gap (ICC)"),
    ("avg_gap_alpha", "best_method_alpha",  "Avg Gap (Alpha)"),
    ("avg_gap_mse",   "best_method_mse",    "Avg Gap (MSE)"),
]


# ---------------------------------------------------------------------------
# Inter-human agreement metrics
# ---------------------------------------------------------------------------

def compute_icc(ratings_matrix: np.ndarray, icc_type: str = "ICC3k",
                nan_policy: str = "listwise") -> float:
    """
    Compute ICC from a (n_items, n_raters) matrix.

    icc_type options (pingouin labels):
      "ICC1"  - one-way random, absolute, single measures
      "ICC2"  - two-way random, absolute, single measures
      "ICC3"  - two-way mixed, consistency, single measures
      "ICC1k" - one-way random, absolute, average measures
      "ICC2k" - two-way random, absolute, average measures
      "ICC3k" - two-way mixed, consistency, average measures  ← default

    nan_policy:
      "listwise" (default) - drop rows where any rater has NaN (balanced design)
      "omit"               - include all items; NaN entries omitted from long-format
                             data, allowing unbalanced designs (pingouin nan_policy)
    """
    try:
        if nan_policy == "listwise":
            matrix = ratings_matrix[~np.isnan(ratings_matrix).any(axis=1)]
        else:
            matrix = ratings_matrix

        n_items, n_raters = matrix.shape
        if n_items < 2 or n_raters < 2:
            return np.nan

        rows = []
        for item_idx in range(n_items):
            for rater_idx in range(n_raters):
                val = matrix[item_idx, rater_idx]
                if not np.isnan(val):
                    rows.append({"item": item_idx, "rater": rater_idx, "rating": val})
        df = pd.DataFrame(rows)

        kwargs = {"data": df, "targets": "item", "raters": "rater", "ratings": "rating"}
        if nan_policy != "listwise":
            kwargs["nan_policy"] = nan_policy

        icc_result = pg.intraclass_corr(**kwargs)
        icc_row = icc_result[icc_result["Type"] == icc_type]
        if icc_row.empty:
            return np.nan
        return float(icc_row["ICC"].values[0])
    except Exception as e:
        print(f"  ICC computation failed: {e}")
        return np.nan


def compute_krippendorff_alpha(ratings_matrix: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha (ordinal scale).
    ratings_matrix: shape (n_raters, n_items), may contain NaN.
    """
    try:
        alpha = krippendorff.alpha(
            reliability_data=ratings_matrix,
            level_of_measurement="ordinal",
        )
        return float(alpha)
    except Exception as e:
        print(f"  Krippendorff alpha computation failed: {e}")
        return np.nan


def compute_pairwise_mse(ratings_matrix: np.ndarray) -> float:
    """
    Compute average pairwise MSE between raters.
    ratings_matrix: shape (n_items, n_raters), may contain NaN.
    """
    n_items, n_raters = ratings_matrix.shape
    pair_mses = []
    for r1, r2 in combinations(range(n_raters), 2):
        col1 = ratings_matrix[:, r1]
        col2 = ratings_matrix[:, r2]
        mask = ~np.isnan(col1) & ~np.isnan(col2)
        if mask.sum() == 0:
            continue
        mse = np.mean((col1[mask] - col2[mask]) ** 2)
        pair_mses.append(mse)
    return float(np.mean(pair_mses)) if pair_mses else np.nan


def _ratings_matrix_to_ms_df(ratings_matrix: np.ndarray) -> pd.DataFrame:
    """Convert (n_items, n_raters) numpy matrix to DataFrame for compute_ms_components."""
    n_items, n_raters = ratings_matrix.shape
    rows = []
    for i in range(n_items):
        for j in range(n_raters):
            if not np.isnan(ratings_matrix[i, j]):
                rows.append({"text_id": i, "model_name": j,
                              "evaluation_score": ratings_matrix[i, j]})
    return pd.DataFrame(rows)


def _safe_msb_mse_ratio(ms) -> float:
    """Return MSB/MSE ratio; NaN if undefined."""
    if ms is None:
        return np.nan
    msb = float(ms.msb) if np.isfinite(ms.msb) else np.nan
    mse = float(ms.mse) if np.isfinite(ms.mse) and ms.mse != 0 else np.nan
    if np.isnan(msb) or np.isnan(mse):
        return np.nan
    return msb / mse


def compute_metrics(ratings_matrix: np.ndarray, icc_type: str = "ICC3k") -> dict:
    """Compute ICC, Krippendorff's alpha, MSE, and MSB/MSE ratio from an (n_items, n_raters) matrix."""
    icc = compute_icc(ratings_matrix, icc_type=icc_type)
    alpha = compute_krippendorff_alpha(ratings_matrix.T)
    mse = compute_pairwise_mse(ratings_matrix)
    ms_df = _ratings_matrix_to_ms_df(ratings_matrix)
    ms = compute_ms_components(ms_df) if not ms_df.empty else None
    return {"icc": icc, "krippendorff_alpha": alpha, "mse": mse,
            "human_msb_mse_ratio": _safe_msb_mse_ratio(ms)}


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def process_hanna() -> list[dict]:
    """
    hanna_stories_annotations.csv: 1056 stories each rated by exactly 3 crowd
    workers on 6 axes (Relevance, Coherence, Empathy, Surprise, Engagement,
    Complexity, scale 1-5).

    Workers are interchangeable, so we assign anonymous rater ranks (0, 1, 2)
    per story to produce a balanced (1056 x 3) matrix.
    """
    print("Processing hanna...")
    df = pd.read_csv(DATA_DIR / "hanna_stories_annotations.csv")

    axes = ["Relevance", "Coherence", "Empathy", "Surprise", "Engagement", "Complexity"]
    results = []

    df = df.sort_values(["Story ID", "Worker ID"]).copy()
    df["rater_rank"] = df.groupby("Story ID").cumcount()
    n_raters = int(df["rater_rank"].max() + 1)

    for axis in axes:
        print(f"  {axis}")
        pivot = df.pivot_table(
            index="Story ID", columns="rater_rank", values=axis, aggfunc="first"
        )
        ratings = pivot.values.astype(float)
        metrics = compute_metrics(ratings)
        results.append({
            "dataset": "hanna",
            "axis": axis.lower(),
            **metrics,
            "n_items": ratings.shape[0],
            "n_raters": n_raters,
            "note": "",
        })

    return results


def process_mslr() -> list[dict]:
    """
    data_with_overlap_scores.json (MSLR): Cochrane systematic review summaries.
    Each prediction can have 1 or 2 human annotations. We use only the 39
    (review_id, exp_short) pairs that have exactly 2 annotations.
    Axes: fluency, population, intervention, outcome (scale 0-2).
    """
    print("Processing mslr...")
    data = []
    with open(DATA_DIR / "data_with_overlap_scores.json") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    rows = []
    for item in data:
        for pred in item.get("predictions", []):
            anns = pred.get("annotations", [])
            if len(anns) >= 2:
                item_id = f"{item['review_id']}_{pred['exp_short']}"
                for ann in anns:
                    rows.append({
                        "item_id": item_id,
                        "annot_id": ann["annot_id"],
                        "fluency": ann.get("fluency"),
                        "population": ann.get("population"),
                        "intervention": ann.get("intervention"),
                        "outcome": ann.get("outcome"),
                    })

    df = pd.DataFrame(rows)
    axes = ["fluency", "population", "intervention", "outcome"]
    results = []

    for axis in axes:
        print(f"  {axis}")
        pivot = df.pivot_table(
            index="item_id", columns="annot_id", values=axis, aggfunc="first"
        )
        ratings = pivot.values.astype(float)
        metrics = compute_metrics(ratings)
        results.append({
            "dataset": "mslr",
            "axis": axis,
            **metrics,
            "n_items": pivot.shape[0],
            "n_raters": pivot.shape[1],
            "note": "",
        })

    return results


def process_summeval() -> list[dict]:
    """
    SummEval (HuggingFace mteb/summeval): 100 source docs, 16 machine summaries
    each, rated on coherence, consistency, fluency, relevance (scale 1-5).
    NOTE: The mteb/summeval dataset stores scores averaged over 3 expert
    annotators. Individual rater scores are not available in this HF dataset,
    so ICC, Krippendorff's alpha, and MSE between raters cannot be computed.
    """
    print("Processing summeval (HuggingFace mteb/summeval)...")
    print("  NOTE: scores are pre-averaged; individual rater scores not available.")
    axes = ["coherence", "consistency", "fluency", "relevance"]
    results = []
    for axis in axes:
        results.append({
            "dataset": "summeval",
            "axis": axis,
            "icc": np.nan,
            "krippendorff_alpha": np.nan,
            "mse": np.nan,
            "n_items": 1600,  # 100 docs x 16 machine summaries
            "n_raters": 3,
            "note": "mteb/summeval stores averages of 3 expert raters; individual scores unavailable",
        })
    return results


def process_medval() -> list[dict]:
    """
    inter_physician.csv: 90 medical items across 6 tasks, rated by 2-5 physicians
    per task on a 1-4 scale (risk grade). Treated as a single medval/Risk dataset.

    Because each task uses a completely different set of physician columns, raters
    are assigned anonymous ranks (0, 1, ..., max_raters-1) per item so that all
    90 items can be pooled into one (90 x max_raters) matrix with NaN padding.
    ICC is computed on the subset of items where all rater slots are filled;
    alpha and MSE handle NaN natively.
    """
    print("Processing medval (inter_physician pooled as Risk)...")
    df = pd.read_csv(DATA_DIR / "inter_physician.csv")

    physician_cols = [c for c in df.columns if c.startswith("physician_")]

    # Build anonymous-rank scores: for each item collect non-NaN scores in order
    rows = []
    for _, item_row in df.iterrows():
        scores = [item_row[c] for c in physician_cols if pd.notna(item_row[c])]
        rows.append(scores)

    max_raters = max(len(r) for r in rows)
    ratings = np.full((len(rows), max_raters), np.nan)
    for i, scores in enumerate(rows):
        ratings[i, :len(scores)] = scores

    n_raters_per_item = (~np.isnan(ratings)).sum(axis=1)
    print(f"  {len(rows)} items, raters per item: {dict(zip(*np.unique(n_raters_per_item, return_counts=True)))}")

    icc = compute_icc(ratings, nan_policy="omit")
    alpha = compute_krippendorff_alpha(ratings.T)
    mse = compute_pairwise_mse(ratings)
    ms_df = _ratings_matrix_to_ms_df(ratings)
    ms = compute_ms_components(ms_df) if not ms_df.empty else None
    return [{
        "dataset": "medval",
        "axis": "risk",
        "icc": icc,
        "krippendorff_alpha": alpha,
        "mse": mse,
        "human_msb_mse_ratio": _safe_msb_mse_ratio(ms),
        "n_items": len(rows),
        "n_raters": max_raters,
        "note": "pooled across 6 tasks; anonymous rater ranks; all 90 items used",
    }]


def process_inter_physician() -> list[dict]:
    """
    inter_physician.csv: 89 medical items (6 tasks), rated by 2-5 physicians
    per task on a 1-4 scale (risk grade).
    """
    print("Processing inter_physician...")
    df = pd.read_csv(DATA_DIR / "inter_physician.csv")

    physician_cols = [c for c in df.columns if c.startswith("physician_")]
    results = []

    for task in df["task"].unique():
        task_df = df[df["task"] == task].copy()
        active_cols = [c for c in physician_cols if task_df[c].notna().any()]
        if len(active_cols) < 2:
            print(f"  {task}: fewer than 2 raters, skipping")
            continue
        print(f"  {task} ({len(active_cols)} raters, {len(task_df)} items)")
        ratings = task_df[active_cols].values.astype(float)
        metrics = compute_metrics(ratings)
        results.append({
            "dataset": "inter_physician",
            "axis": task,
            **metrics,
            "n_items": ratings.shape[0],
            "n_raters": len(active_cols),
            "note": "",
        })

    return results


# ---------------------------------------------------------------------------
# Model score loading
# ---------------------------------------------------------------------------

def _extract_model_score(result: dict):
    """Extract model score from a result dict (handles two JSON shapes)."""
    ev = result.get("evaluation", {})
    if isinstance(ev, str):
        try:
            ev = json.loads(ev)
        except Exception:
            return None
    try:
        return float(ev["evaluation"]["score"])
    except (KeyError, TypeError):
        pass
    try:
        return float(ev["score"])
    except (KeyError, TypeError):
        return None


def load_judge_scores_for_model(dataset: str, axis: str, model: str):
    """
    Load judge scores for a specific dataset/axis/model.
    Returns a dict mapping text_id -> (human_score, model_score).
    """
    score_dir = JUDGE_SCORES_DIR / dataset
    if not score_dir.exists():
        return {}

    matches = list(score_dir.glob(f"results_{dataset}_{model}_{axis}.json"))
    if not matches:
        for fname in score_dir.iterdir():
            if fname.stem.lower() == f"results_{dataset}_{model}_{axis}".lower():
                matches = [fname]
                break
    if not matches:
        return {}

    with open(matches[0]) as f:
        data = json.load(f)

    scores = {}
    for r in data.get("detailed_results", []):
        model_score = _extract_model_score(r)
        human_score = r.get("original_score")
        if model_score is not None and human_score is not None:
            scores[r.get("text_id")] = (float(human_score), float(model_score))
    return scores


def load_model_scores_df(dataset: str, axis: str, models: list[str]) -> pd.DataFrame:
    """
    Load judge scores for a specific dataset/axis for the given models.

    Returns a DataFrame with columns: text_id, model_name, evaluation_score, human_score.
    Only rows where both human and model scores are available are included.
    """
    score_dir = JUDGE_SCORES_DIR / dataset
    if not score_dir.exists():
        return pd.DataFrame()

    rows = []
    for model in models:
        matches = list(score_dir.glob(f"results_{dataset}_{model}_{axis}.json"))
        if not matches:
            for fname in score_dir.iterdir():
                if fname.stem.lower() == f"results_{dataset}_{model}_{axis}".lower():
                    matches = [fname]
                    break
        if not matches:
            continue

        with open(matches[0]) as f:
            data = json.load(f)

        for r in data.get("detailed_results", []):
            model_score = _extract_model_score(r)
            human_score = r.get("original_score")
            text_id = r.get("text_id")
            if model_score is not None and human_score is not None and text_id is not None:
                rows.append({
                    "text_id": text_id,
                    "model_name": model,
                    "evaluation_score": float(model_score),
                    "human_score": float(human_score),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inter-model (IM) and human-model (HM) statistics
# ---------------------------------------------------------------------------

def _build_im_pairwise_df_subset(sub: pd.DataFrame, model: str, all_models: list) -> pd.DataFrame:
    """
    Build a 2-rater IM DataFrame for one sample subset using average_pairwise mode:
      rater 1: target model scores
      rater 2: average score of all other models per text_id
    """
    other_models = [m for m in all_models if m != model]
    model_rows = sub[sub["model_name"] == model][["text_id", "evaluation_score"]].copy()
    model_rows["model_name"] = model

    avg_rows = (
        sub[sub["model_name"].isin(other_models)]
        .groupby("text_id")["evaluation_score"]
        .mean()
        .reset_index()
    )
    avg_rows["model_name"] = "avg_other"

    return pd.concat(
        [model_rows[["text_id", "model_name", "evaluation_score"]],
         avg_rows[["text_id", "model_name", "evaluation_score"]]],
        ignore_index=True,
    )


def compute_im_stats(scores_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute full-dataset inter-model MSB, MSE, and ICC using average_pairwise mode,
    returning values per target model.

    For each target model m:
      IM DataFrame = (m, avg_of_other_models) as 2 raters on shared text_ids.

    Returns:
        dict mapping model_name -> {"im_msb": float, "im_mse": float, "im_icc": float}
    """
    if scores_df.empty:
        return {}

    models = list(scores_df["model_name"].unique())
    if len(models) < 2:
        return {}

    result = {}
    for model in models:
        im_df = _build_im_pairwise_df_subset(scores_df, model, models)
        ms = compute_ms_components(im_df)
        result[model] = {
            "im_msb": float(ms.msb) if ms is not None and np.isfinite(ms.msb) else np.nan,
            "im_mse": float(ms.mse) if ms is not None and np.isfinite(ms.mse) else np.nan,
            "im_icc": (float(ms.icc) if ms is not None and hasattr(ms, "icc")
                       and np.isfinite(ms.icc) else np.nan),
        }
    return result


def compute_hm_stats(scores_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute full-dataset HM (target model + human) MSB/MSE ratio per model.

    For each target model m:
      HM DataFrame = (m, human) as 2 raters on shared text_ids.

    Returns:
        dict mapping model_name -> {"hm_msb_mse_ratio": float}
    """
    if scores_df.empty:
        return {}

    models = list(scores_df["model_name"].unique())
    result = {}
    for model in models:
        model_sub = scores_df[scores_df["model_name"] == model].copy()
        human_rows = (model_sub[["text_id", "human_score"]]
                      .rename(columns={"human_score": "evaluation_score"})
                      .assign(model_name="original"))
        model_rows = model_sub[["text_id", "evaluation_score"]].assign(model_name=model)
        hm_df = pd.concat([human_rows, model_rows], ignore_index=True)
        ms = compute_ms_components(hm_df)
        result[model] = {"hm_msb_mse_ratio": _safe_msb_mse_ratio(ms)}
    return result


def compute_hm_ms_corr(scores_df: pd.DataFrame,
                        n_samples: int = N_CORR_SAMPLES,
                        sample_size: int = CORR_SAMPLE_SIZE,
                        seed: int = 123) -> dict[str, tuple[float, float, float]]:
    """
    Compute per-model Pearson correlations between IM and HM MSB/MSE/ICC.

    For each target model m:
      - Draw n_samples sets of sample_size text_ids.
      - Per sample:
          IM: (m, avg_of_other_models) → MSB/MSE/ICC
          HM: (m, human) → MSB/MSE/ICC
      - Pearson-correlate the n_samples pairs → (r_msb, r_mse, r_icc).

    Returns:
        dict mapping model_name -> (msb_corr, mse_corr, icc_corr)
    """
    if scores_df.empty:
        return {}

    models = list(scores_df["model_name"].unique())
    if len(models) < 2:
        return {}

    id_counts = scores_df.groupby("text_id")["model_name"].nunique()
    shared_ids = np.array(id_counts[id_counts == len(models)].index)

    if len(shared_ids) < sample_size:
        return {}

    rng = np.random.RandomState(seed)
    fast_ms = partial(compute_ms_components, validate=False)

    def _pearson(x, y):
        if len(x) < 5:
            return np.nan
        xa, ya = np.array(x), np.array(y)
        if np.std(xa) == 0 or np.std(ya) == 0:
            return np.nan
        return float(np.corrcoef(xa, ya)[0, 1])

    per_model: dict[str, tuple[float, float, float]] = {}

    for model in models:
        im_msb_vals, hm_msb_vals = [], []
        im_mse_vals, hm_mse_vals = [], []
        im_icc_vals, hm_icc_vals = [], []

        for _ in range(n_samples):
            sample_ids = rng.choice(shared_ids, size=sample_size, replace=False)
            sub = scores_df[scores_df["text_id"].isin(sample_ids)]

            im_df = _build_im_pairwise_df_subset(sub, model, models)
            im_ms = fast_ms(im_df)

            model_sub = sub[sub["model_name"] == model].copy()
            human_rows = (model_sub[["text_id", "human_score"]]
                          .rename(columns={"human_score": "evaluation_score"})
                          .assign(model_name="original"))
            model_rows = model_sub[["text_id", "evaluation_score"]].assign(model_name=model)
            hm_df = pd.concat([human_rows, model_rows], ignore_index=True)
            hm_ms = fast_ms(hm_df)

            if im_ms is None or hm_ms is None:
                continue

            if np.isfinite(im_ms.msb) and np.isfinite(hm_ms.msb):
                im_msb_vals.append(im_ms.msb)
                hm_msb_vals.append(hm_ms.msb)
            if np.isfinite(im_ms.mse) and np.isfinite(hm_ms.mse):
                im_mse_vals.append(im_ms.mse)
                hm_mse_vals.append(hm_ms.mse)
            if (hasattr(im_ms, "icc") and hasattr(hm_ms, "icc")
                    and np.isfinite(im_ms.icc) and np.isfinite(hm_ms.icc)):
                im_icc_vals.append(im_ms.icc)
                hm_icc_vals.append(hm_ms.icc)

        per_model[model] = (
            _pearson(im_msb_vals, hm_msb_vals),
            _pearson(im_mse_vals, hm_mse_vals),
            _pearson(im_icc_vals, hm_icc_vals),
        )

    return per_model


# ---------------------------------------------------------------------------
# Best method gap computation
# ---------------------------------------------------------------------------

def compute_best_method_gap(dataset: str, axis: str, metric: str,
                             budget: int | None = None,
                             model: str | None = None) -> tuple[str, float]:
    """
    Find the variance_matched method with the most negative avg gap vs random
    for a single metric (icc, alpha, or mse).

    Args:
        budget: if None, average across all budgets; if int, restrict to that budget.
        model: if provided, restrict to that specific model only.

    Excludes oracle_* and metric_matched_* methods.

    Returns:
        (best_method, avg_gap) or ("", np.nan) if data unavailable.
    """
    dataset_dir = RESULTS_DIR / dataset / dataset / "dataframes"
    if not dataset_dir.exists():
        return "", np.nan

    fpath = dataset_dir / f"{metric}_results.csv"
    if not fpath.exists():
        return "", np.nan

    df = pd.read_csv(fpath)

    axis_data = df[df["axis"].str.lower() == axis.lower()]
    if axis_data.empty:
        return "", np.nan

    if budget is not None:
        axis_data = axis_data[axis_data["budget"] == budget]
    if model is not None:
        axis_data = axis_data[axis_data["model"] == model]
    if axis_data.empty:
        return "", np.nan

    exclude_mask = (
        axis_data["method"].str.startswith("oracle") |
        axis_data["method"].str.startswith("metric_matched")
    )
    axis_data = axis_data[~exclude_mask]

    random_data = axis_data[axis_data["method"] == "random"]
    non_random_data = axis_data[axis_data["method"] != "random"]

    if random_data.empty or non_random_data.empty:
        return "", np.nan

    random_baseline = random_data.groupby(["model", "budget"])["estimation_error"].mean()

    method_gaps: dict[str, list[float]] = {}
    for method in non_random_data["method"].unique():
        method_rows = non_random_data[non_random_data["method"] == method]
        gaps = []
        for (model, bdg), grp in method_rows.groupby(["model", "budget"]):
            try:
                rand_err = random_baseline[(model, bdg)]
                gaps.append(float(grp["estimation_error"].mean() - rand_err))
            except KeyError:
                pass
        if gaps:
            method_gaps[method] = gaps

    if not method_gaps:
        return "", np.nan

    avg_gaps = {m: float(np.mean(g)) for m, g in method_gaps.items()}
    best_method = min(avg_gaps, key=avg_gaps.get)
    return best_method, avg_gaps[best_method]


def compute_gap_for_method(dataset: str, axis: str, metric: str,
                            method_name: str, model: str | None = None,
                            budget: int | None = None) -> float:
    """Return the avg gap vs random for a specific named method."""
    dataset_dir = RESULTS_DIR / dataset / dataset / "dataframes"
    fpath = dataset_dir / f"{metric}_results.csv"
    if not fpath.exists():
        return np.nan
    df = pd.read_csv(fpath)
    axis_data = df[df["axis"].str.lower() == axis.lower()]
    if model:
        axis_data = axis_data[axis_data["model"] == model]
    if budget is not None:
        axis_data = axis_data[axis_data["budget"] == budget]
    random_data = axis_data[axis_data["method"] == "random"]
    method_data = axis_data[axis_data["method"] == method_name]
    if random_data.empty or method_data.empty:
        return np.nan
    random_baseline = random_data.groupby(["model", "budget"])["estimation_error"].mean()
    gaps = []
    for (mdl, bdg), grp in method_data.groupby(["model", "budget"]):
        try:
            gaps.append(float(grp["estimation_error"].mean() - random_baseline[(mdl, bdg)]))
        except KeyError:
            pass
    return float(np.mean(gaps)) if gaps else np.nan


def find_global_best_method(output_df: pd.DataFrame,
                             budget: int | None = None) -> tuple[str, float]:
    """
    Find the single variance_matched method with the most negative average gap
    across all (dataset, axis, model, metric) combinations.

    Args:
        budget: if None, aggregate across all budgets; if int, restrict to that budget.

    Returns:
        (best_method, avg_gap_across_all)
    """
    method_gaps: dict[str, list[float]] = {}

    for _, row in output_df.iterrows():
        dataset = row["dataset"]
        axis = row["axis"]
        model = row.get("model") or None
        if not model:
            continue
        for metric in ("icc", "alpha"):
            dataset_dir = RESULTS_DIR / dataset / dataset / "dataframes"
            fpath = dataset_dir / f"{metric}_results.csv"
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            axis_data = df[(df["axis"].str.lower() == axis.lower()) &
                           (df["model"] == model)]
            if budget is not None:
                axis_data = axis_data[axis_data["budget"] == budget]
            exclude = (axis_data["method"].str.startswith("oracle") |
                       axis_data["method"].str.startswith("metric_matched"))
            axis_data = axis_data[~exclude]
            random_data = axis_data[axis_data["method"] == "random"]
            non_random = axis_data[axis_data["method"] != "random"]
            if random_data.empty or non_random.empty:
                continue
            random_baseline = random_data.groupby(["model", "budget"])["estimation_error"].mean()
            for method in non_random["method"].unique():
                for (mdl, bdg), grp in non_random[non_random["method"] == method].groupby(["model", "budget"]):
                    try:
                        gap = float(grp["estimation_error"].mean() - random_baseline[(mdl, bdg)])
                        method_gaps.setdefault(method, []).append(gap)
                    except KeyError:
                        pass

    if not method_gaps:
        return "", np.nan
    avg_gaps = {m: float(np.mean(g)) for m, g in method_gaps.items()}
    best = min(avg_gaps, key=avg_gaps.get)
    return best, avg_gaps[best]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_score_distributions():
    """
    For each dataset and axis, plot a histogram comparing human score distribution
    vs. each of the model score distributions.
    Saves one figure per dataset to OUTPUT_DIR/plots/.
    """
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    for dataset, axes in DATASET_AXES.items():
        n_axes = len(axes)
        fig, axes_grid = plt.subplots(
            len(MODELS_TO_PLOT), n_axes,
            figsize=(4 * n_axes, 3.5 * len(MODELS_TO_PLOT)),
            squeeze=False,
        )
        fig.suptitle(f"{dataset} — human vs. model score distributions", fontsize=13)

        for col, axis_name in enumerate(axes):
            for row, model in enumerate(MODELS_TO_PLOT):
                ax = axes_grid[row][col]
                scores = load_judge_scores_for_model(dataset, axis_name, model)

                if not scores:
                    ax.set_visible(False)
                    continue

                human_vals = np.array([v[0] for v in scores.values()])
                model_vals = np.array([v[1] for v in scores.values()])

                all_vals = np.concatenate([human_vals, model_vals])
                bins = np.linspace(all_vals.min() - 0.5, all_vals.max() + 0.5, 20)

                ax.hist(human_vals, bins=bins, alpha=0.55, label="human",
                        color=HUMAN_COLOR, weights=np.ones(len(human_vals)) / len(human_vals))
                ax.hist(model_vals, bins=bins, alpha=0.55, label=model,
                        color=MODEL_COLORS[model], weights=np.ones(len(model_vals)) / len(model_vals))

                if row == 0:
                    ax.set_title(axis_name, fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{model}\nproportion", fontsize=8)
                ax.set_xlabel("score", fontsize=8)
                ax.set_ylim(0, 1)
                ax.legend(fontsize=7)
                ax.tick_params(labelsize=7)

        plt.tight_layout()
        out_path = plots_dir / f"{dataset}_score_distributions.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {out_path}")


def plot_predictor_gap_scatter(df: pd.DataFrame,
                               budget: int | None = None,
                               method_color: dict | None = None) -> None:
    """
    Grid of scatter plots: rows = avg_gap metric, cols = x-axis predictor.
    Points are colored by best_method for that row's metric.

    Args:
        df: DataFrame with avg_gap_* and best_method_* columns for this budget.
        budget: None → overall (all budgets); int → specific budget.
        method_color: shared color map dict {method: color} for consistency across plots.
    """
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    if method_color is None:
        all_methods = set()
        for _, method_col, _ in Y_VARS:
            if method_col in df.columns:
                all_methods.update(df[method_col].dropna().unique())
        all_methods = sorted(all_methods)
        palette = plt.colormaps.get_cmap("tab10")
        method_color = {m: palette(i / max(len(all_methods) - 1, 1))
                        for i, m in enumerate(all_methods)}

    title_suffix = "Overall (all budgets)" if budget is None else f"Budget {budget}"
    fname_suffix = "overall" if budget is None else f"b{budget}"

    n_rows = len(Y_VARS)
    n_cols = len(X_VARS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    fig.suptitle(f"Predictor vs avg gap vs random — {title_suffix}", fontsize=13)

    for row_idx, (y_col, method_col, y_label) in enumerate(Y_VARS):
        for col_idx, (x_col, x_label) in enumerate(X_VARS):
            ax = axes[row_idx][col_idx]

            cols = [x_col, y_col, method_col]
            if "model" in df.columns:
                cols.append("model")
            plot_df = df[cols].dropna(subset=[x_col, y_col, method_col])
            if plot_df.empty:
                ax.set_visible(False)
                continue

            for method, method_grp in plot_df.groupby(method_col):
                color = method_color.get(method, "gray")
                if "model" in plot_df.columns:
                    for mdl, mdl_grp in method_grp.groupby("model"):
                        marker = MODEL_MARKERS.get(mdl, "o")
                        ax.scatter(mdl_grp[x_col], mdl_grp[y_col],
                                   color=color, marker=marker,
                                   s=45, alpha=0.85, zorder=3)
                else:
                    ax.scatter(method_grp[x_col], method_grp[y_col],
                               color=color, s=45, alpha=0.85, zorder=3)

            if len(plot_df) >= 3:
                r = float(np.corrcoef(plot_df[x_col], plot_df[y_col])[0, 1])
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, va="top")

            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=9)
            if row_idx == 0:
                ax.set_title(x_label, fontsize=9)
            ax.set_xlabel(x_label, fontsize=8)
            ax.tick_params(labelsize=7)

    all_methods_used = sorted(method_color.keys())
    method_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=method_color[m], markersize=7, label=m)
                      for m in all_methods_used]
    model_handles = [plt.Line2D([0], [0], marker=mk, color="gray",
                                 markersize=7, linestyle="None", label=mdl)
                     for mdl, mk in MODEL_MARKERS.items()]

    leg1 = fig.legend(handles=method_handles,
                      loc="lower center", ncol=min(len(all_methods_used), 4),
                      fontsize=7, bbox_to_anchor=(0.5, -0.02), title="Best method")
    fig.add_artist(leg1)
    fig.legend(handles=model_handles,
               loc="lower center", ncol=len(MODEL_MARKERS),
               fontsize=7, bbox_to_anchor=(0.5, -0.07), title="Model")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = plots_dir / f"predictor_gap_scatter_{fname_suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")


def plot_global_best_method(output_df: pd.DataFrame,
                             budget: int | None = None) -> None:
    """
    Find the single globally best method (across all models, metrics, axes),
    then plot predictor vs its gap, coloring by dataset and shaping by model.

    Args:
        budget: if None, aggregate across all budgets; if int, restrict to that budget.
    """
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    budget_label = "overall" if budget is None else f"b{budget}"
    budget_title = "Overall (all budgets)" if budget is None else f"Budget {budget}"

    print(f"  Finding globally best method ({budget_title})...")
    best_method, best_avg_gap = find_global_best_method(output_df, budget=budget)
    if not best_method:
        print(f"  No data — skipping global best method plot ({budget_title}).")
        return
    print(f"  Globally best method ({budget_title}): {best_method} (avg gap = {best_avg_gap:.4f})")

    # Build plot DataFrame: one row per (dataset, axis, model)
    plot_rows = []
    pred_cols = [x for x, _ in X_VARS]
    for _, row in output_df.iterrows():
        model = row.get("model") or None
        if not model:
            continue
        prow = {c: row[c] for c in pred_cols + ["dataset", "axis", "model"] if c in row}
        for metric in ("icc", "alpha", "mse"):
            prow[f"avg_gap_{metric}"] = compute_gap_for_method(
                row["dataset"], row["axis"], metric, best_method,
                model=model, budget=budget)
        plot_rows.append(prow)

    plot_df = pd.DataFrame(plot_rows)

    n_rows, n_cols = len(Y_VARS), len(X_VARS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Global best method: {best_method}  (avg gap = {best_avg_gap:.4f})  [{budget_title}]",
        fontsize=13)

    for row_idx, (y_col, _, y_label) in enumerate(Y_VARS):
        for col_idx, (x_col, x_label) in enumerate(X_VARS):
            ax = axes[row_idx][col_idx]
            sub = plot_df[[x_col, y_col, "dataset", "model"]].dropna(subset=[x_col, y_col])
            if sub.empty:
                ax.set_visible(False)
                continue
            for dataset, ds_grp in sub.groupby("dataset"):
                color = DATASET_COLORS.get(dataset, "gray")
                for mdl, mdl_grp in ds_grp.groupby("model"):
                    ax.scatter(mdl_grp[x_col], mdl_grp[y_col],
                               color=color, marker=MODEL_MARKERS.get(mdl, "o"),
                               s=45, alpha=0.85, zorder=3)
            if len(sub) >= 3:
                r = float(np.corrcoef(sub[x_col], sub[y_col])[0, 1])
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, va="top")
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
            if col_idx == 0:
                ax.set_ylabel(y_label, fontsize=9)
            if row_idx == 0:
                ax.set_title(x_label, fontsize=9)
            ax.set_xlabel(x_label, fontsize=8)
            ax.tick_params(labelsize=7)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=DATASET_COLORS.get(d, "gray"), markersize=7, label=d)
        for d in DATASET_COLORS
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=mk, color="gray", markersize=7,
                   linestyle="None", label=mdl)
        for mdl, mk in MODEL_MARKERS.items()
    ]
    leg1 = fig.legend(handles=dataset_handles, loc="lower center",
                      ncol=len(DATASET_COLORS), fontsize=7,
                      bbox_to_anchor=(0.5, -0.02), title="Dataset")
    fig.add_artist(leg1)
    fig.legend(handles=model_handles, loc="lower center",
               ncol=len(MODEL_MARKERS), fontsize=7,
               bbox_to_anchor=(0.5, -0.07), title="Model")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = plots_dir / f"predictor_gap_scatter_global_best_{budget_label}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    all_results = []
    all_results.extend(process_hanna())
    all_results.extend(process_mslr())
    all_results.extend(process_summeval())
    all_results.extend(process_medval())
    all_results.extend(process_inter_physician())

    print("\nComputing per-model IM stats, HM correlations, and best method gaps...")
    expanded_results = []
    nan3 = {"im_msb": np.nan, "im_mse": np.nan, "im_icc": np.nan}
    nan_hm = {"hm_msb_mse_ratio": np.nan}

    for row in all_results:
        dataset = row["dataset"]
        axis = row["axis"]
        # ensure human_msb_mse_ratio exists (summeval rows set it via NaN fallback)
        row.setdefault("human_msb_mse_ratio", np.nan)

        if dataset in DATASET_AXES:
            print(f"  {dataset}/{axis}: loading scores...")
            scores_df = load_model_scores_df(dataset, axis, MODELS_TO_PLOT)
            im_stats_per_model = compute_im_stats(scores_df)   # model -> {im_msb,...}
            hm_stats_per_model = compute_hm_stats(scores_df)   # model -> {hm_msb_mse_ratio}
            corr_per_model = compute_hm_ms_corr(scores_df)     # model -> (r_msb, r_mse, r_icc)

            for model in MODELS_TO_PLOT:
                mrow = dict(row)
                mrow["model"] = model
                im = im_stats_per_model.get(model, nan3)
                mrow["im_msb"] = im["im_msb"]
                mrow["im_mse"] = im["im_mse"]
                mrow["im_icc"] = im["im_icc"]
                hm = hm_stats_per_model.get(model, nan_hm)
                mrow["hm_msb_mse_ratio"] = hm["hm_msb_mse_ratio"]
                r_msb, r_mse, r_icc = corr_per_model.get(model, (np.nan, np.nan, np.nan))
                mrow["hm_msb_corr"] = r_msb
                mrow["hm_mse_corr"] = r_mse
                mrow["hm_icc_corr"] = r_icc
                for metric in ("icc", "alpha", "mse"):
                    bm, ag = compute_best_method_gap(dataset, axis, metric, model=model)
                    mrow[f"best_method_{metric}"] = bm
                    mrow[f"avg_gap_{metric}"] = ag
                expanded_results.append(mrow)
        else:
            # No model scores (e.g. inter_physician) — keep one row with NaN model cols
            row["model"] = ""
            row.update(nan3)
            row["hm_msb_mse_ratio"] = np.nan
            row["hm_msb_corr"] = np.nan
            row["hm_mse_corr"] = np.nan
            row["hm_icc_corr"] = np.nan
            for metric in ("icc", "alpha", "mse"):
                row[f"best_method_{metric}"] = ""
                row[f"avg_gap_{metric}"] = np.nan
            expanded_results.append(row)

    output_df = pd.DataFrame(expanded_results)
    output_df = output_df[
        ["dataset", "axis", "model",
         "icc", "krippendorff_alpha", "mse",
         "human_msb_mse_ratio",
         "n_items", "n_raters",
         "im_msb", "im_mse", "im_icc",
         "hm_msb_mse_ratio",
         "hm_msb_corr", "hm_mse_corr", "hm_icc_corr",
         "best_method_icc", "avg_gap_icc",
         "best_method_alpha", "avg_gap_alpha",
         "best_method_mse", "avg_gap_mse",
         "note"]
    ]

    output_path = OUTPUT_DIR / "human_agreement.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(output_df.to_string(index=False))

    print("\nGenerating score distribution plots...")
    plot_score_distributions()

    print("\nGenerating predictor vs gap scatter plots...")
    PLOT_BUDGETS = [None, 10, 20, 30, 40, 50]

    # Build per-budget DataFrames (recompute best_method/avg_gap at each budget)
    plot_dfs: dict = {None: output_df}
    for bdg in PLOT_BUDGETS[1:]:
        print(f"  Computing gaps for budget {bdg}...")
        budget_rows = []
        for _, base_row in output_df.iterrows():
            row = dict(base_row)
            mdl = row.get("model") or None
            for metric in ("icc", "alpha", "mse"):
                bm, ag = compute_best_method_gap(
                    row["dataset"], row["axis"], metric,
                    budget=bdg, model=mdl if mdl else None)
                row[f"best_method_{metric}"] = bm
                row[f"avg_gap_{metric}"] = ag
            budget_rows.append(row)
        plot_dfs[bdg] = pd.DataFrame(budget_rows)

    # Build a shared color map across all budgets for consistency
    all_methods: set = set()
    for df_b in plot_dfs.values():
        for _, method_col, _ in Y_VARS:
            if method_col in df_b.columns:
                all_methods.update(df_b[method_col].dropna().unique())
    all_methods = sorted(all_methods)
    palette = plt.colormaps.get_cmap("tab10")
    shared_method_color = {m: palette(i / max(len(all_methods) - 1, 1))
                           for i, m in enumerate(all_methods)}

    for bdg in PLOT_BUDGETS:
        plot_predictor_gap_scatter(plot_dfs[bdg], budget=bdg,
                                   method_color=shared_method_color)

    print("\nGenerating global best method plots (overall + per budget)...")
    for bdg in PLOT_BUDGETS:
        plot_global_best_method(output_df, budget=bdg)


if __name__ == "__main__":
    main()
