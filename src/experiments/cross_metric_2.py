"""
Variance Selection Analysis for Reliability Estimation.

This experiment evaluates different sampling strategies for estimating with limited human annotation budgets. It compares:
- Random sampling (baseline)
- Variance-matched sampling (matches inter-model variance structure)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

import warnings
from functools import partial
from collections import defaultdict
from tqdm import tqdm

from src.utils.data_loading import load_judge_scores
from src.utils.reliability_metrics import (
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_spearman_rho,
    compute_kendall_tau,
    compute_ms_components
)
# from src.utils.match_metrics import compute_weighted_msb_msre
from src.utils.selection_strategies import metric_matched_selection

# Set random seed for reproducibility
np.random.seed(42)

# -------------------------
# CONFIGURATION
# -------------------------
N_BOOTSTRAP_SAMPLES = 100
N_CANDIDATE_SUBSETS = 20

EVALUATION_AXES = {
    "hanna": ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval": ["Risk"],
    "mslr": ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"]
}

## METRIC CONFIGURATION
METRIC_NAMES = [
    "msb+msre",
    "icc",
    "alpha",
    "spearman",
    "kendalltau"
]
METRIC_FNS = [
    compute_weighted_msb_msre,
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_spearman_rho,
    compute_kendall_tau
]
EXCLUDE_EST = [
    "msb+msre",
    # "icc",
    # "alpha",
    # "spearman",
    # "kendalltau"
]
EXCLUDE_MATCH = [
    # "msb+msre",
    # "icc",
    # "alpha",
    # "spearman",
    # "kendalltau"
]

# Runtime configuration
# dataset = "summeval"
datasets = ["medval", "mslr", "summeval", "hanna"]
# model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "gpt-5", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
# model_names = ["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5", "gemini-2.5-pro", "deepseek-r1"]
DATA_DIR = "data/judge_scores"
PLOTS_DIR = "results/04_28/metric_matched_subsets"
COMPARISON_MODE = "pairwise"  # "pairwise" or "aggregate"

# os.makedirs(os.path.join(PLOTS_DIR, dataset), exist_ok=True)


# -------------------------
# VARIANCE ALIGNMENT
# -------------------------
def compute_metric_alignment(df, model_names, compute_metric_fns, mode="aggregate"):
    """
    Compute inter-model and human-model variance components.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score
        model_names: List of model names (excluding "original")
        mode: "aggregate" or "pairwise"

    Returns:
        per_model_variance: dict mapping model_name to {"im": [im_metrics], "hm": [hm_metrics]} (ordered as in compute_metric_fns)
        aggregate_stats: dict with overall statistics
    """
    per_model_metric = defaultdict(lambda: defaultdict(list))
    aggregate_stats = []

    if mode == "aggregate":
        im_df = df.loc[df["model_name"] != "original"]
        for compute_metric_fn in compute_metric_fns:
            im_obj = compute_metric_fn(im_df)
            for m in model_names:
                hm_df = df.loc[df["model_name"].isin([m, "original"])]
                hm_obj = compute_metric_fn(hm_df)
                per_model_metric[m]["im"].append(im_obj)
                per_model_metric[m]["hm"].append(hm_obj)

                print(f"\nMode: AGGREGATE (k={len(model_names)} for IM, k=2 for HM)")
                print(f"Inter-model (all models): metric={im_obj:.4f}")

                aggregate_stats.append({
                    "im": im_obj,
                    "hm_list": [per_model_metric[m]["hm"] for m in model_names]
                })

    elif mode == "pairwise":
        im_dict_list = defaultdict(list)
        hm_dict_list = defaultdict(list)

        im_subset = df[df["model_name"].isin(model_names)].copy()
        im_grouped = im_subset.groupby("text_id")

        for m in model_names:
            other_models = [x for x in model_names if x != m]

            # Inter-model: this model vs avg of other models
            im_avg_df = []
            for text_id, text_data in im_grouped:
                model_score = text_data.loc[text_data["model_name"] == m, "evaluation_score"]
                if len(model_score) > 0:
                    im_avg_df.append({
                        "text_id": text_id,
                        "model_name": m,
                        "evaluation_score": model_score.iloc[0]
                    })
                other_scores = text_data.loc[
                    text_data["model_name"].isin(other_models), "evaluation_score"
                ]
                if len(other_scores) > 0:
                    im_avg_df.append({
                        "text_id": text_id,
                        "model_name": "avg_other",
                        "evaluation_score": other_scores.mean()
                    })

            im_pair_df = pd.DataFrame(im_avg_df)
            for compute_metric_fn in compute_metric_fns:
                if len(im_pair_df) > 0:
                    im_obj = compute_metric_fn(im_pair_df)
                else:
                    im_obj = np.nan
                im_dict_list[m].append(im_obj)

                # Human-model: this model vs human
                hm_df = df.loc[df["model_name"].isin([m, "original"])]
                hm_obj = compute_metric_fn(hm_df)
                hm_dict_list[m].append(hm_obj)

                per_model_metric[m]["im"].append(im_obj)
                per_model_metric[m]["hm"].append(hm_obj)

        print(f"\nMode: PAIRWISE (k=2 for both IM and HM)")
        print(f"Inter-model metrics (each model vs avg of others): {[[f'{v:.4f}' for v in x] for x in im_dict_list.values()]}")
        im_metric_mean = np.mean(np.array([v for _, v in im_dict_list.items()]), axis=0)
        print(f"Inter-model mean: metrics={[f'{x:.4f}' for x in im_metric_mean]}")

        aggregate_stats = {
            "im": im_metric_mean,
            "hm_list": hm_dict_list,
        }

    print(f"Human-model metrics: {[[f'{v:.4f}' for v in x] for x in aggregate_stats['hm_list'].values()]}")

    if len(aggregate_stats['hm_list']) > 1:
        hm_mean = np.mean(np.array([v for _, v in aggregate_stats["hm_list"].items()]), axis=0)
        print(f"\nMetrics: IM={[f'{x:.4f}' for x in aggregate_stats['im']]}, HM mean={[f'{x:.4f}' for x in hm_mean]}")

    return per_model_metric, aggregate_stats


# -------------------------
# RELIABILITY ESTIMATION EXPERIMENT
# -------------------------
def _build_im_pairwise_df(df, model, model_names):
    """Build inter-model DataFrame for pairwise mode (model vs avg of others)."""
    other_models = [x for x in model_names if x != model]
    im_subset = df[df["model_name"].isin(model_names)].copy()
    im_grouped = im_subset.groupby("text_id")

    im_full_df = []
    for text_id, text_data in im_grouped:
        model_score = text_data.loc[text_data["model_name"] == model, "evaluation_score"]
        if len(model_score) > 0:
            im_full_df.append({
                "text_id": text_id,
                "model_name": model,
                "evaluation_score": model_score.iloc[0]
            })
        other_scores = text_data.loc[text_data["model_name"].isin(other_models), "evaluation_score"]
        if len(other_scores) > 0:
            im_full_df.append({
                "text_id": text_id,
                "model_name": "avg_other",
                "evaluation_score": other_scores.mean()
            })

    return pd.DataFrame(im_full_df)


def _run_random_trials(text_ids, k, n_trials, hm_full_df, model, metric_names, hm_targets, compute_metric_fns, exclude_est=[]):
    """Run random sampling trials and collect estimation errors."""
    errors_dict_list = []

    for trial_idx in range(n_trials): #tqdm(range(n_trials), f"Running random trials..."):
        np.random.seed(42 + trial_idx)
        sampled_ids = np.random.choice(text_ids, size=min(k, len(text_ids)), replace=False)
        hm_sample = hm_full_df[hm_full_df["text_id"].isin(sampled_ids)]

        for metric_name, hm_target, compute_metric_fn in zip(metric_names, hm_targets, compute_metric_fns):
            if metric_name in exclude_est:
                continue
            try:
                est_metric = compute_metric_fn(hm_sample)
                if np.isfinite(est_metric):
                    errors_dict_list.append({
                        "model": model, "budget": k, "method": "random",
                        "match_metric": None,
                        "est_metric": metric_name,
                        "est_error": abs(est_metric - hm_target),
                        # "sp_corr": np.nan,
                        # "sp_corr_diff": np.nan
                    })
            except Exception:
                pass

    return errors_dict_list


def _run_metric_matched_trials(text_ids, k, n_trials, hm_full_df, im_full_df,
                                  model, metric_names, hm_targets, im_targets, compute_metric_fns, exclude_match=[], exclude_est=[]):
    """Run variance-matched sampling trials and collect estimation errors for given model."""

    # est_compute_metric_fns = {est_metric_name: fn for est_metric_name, fn in zip(metric_names, compute_metric_fns) \
    #                           if est_metric_name not in exclude_est}
    # hm_targets_dict = {est_metric_name: hm_target for est_metric_name, hm_target in zip(metric_names, hm_targets) \
    #                           if est_metric_name not in exclude_est}
    errors_dict_list = []
    for match_metric_name, im_target, match_compute_metric_fn in zip(metric_names, im_targets, compute_metric_fns):
        if match_metric_name in exclude_match:
            continue
        for trial_idx in range(n_trials): # tqdm(range(n_trials), f"Running metric matched trials on {match_metric_name}..."):
            # best_ids, metadata = metric_matched_selection(
            #     text_ids, k, im_full_df, hm_full_df, im_target, hm_targets_dict,
            #     match_compute_metric_fn, est_compute_metric_fns, alpha_weight=1., seed=42 + trial_idx, n_candidates=N_CANDIDATE_SUBSETS
            # )
            best_ids = metric_matched_selection(text_ids, k, im_full_df, 
                                                im_target, 
                                                match_metric_name, 
                                                compute_ms_fn=compute_ms_components,
                                                compute_icc_fn=compute_icc_pingouin,
                                                compute_alpha_fn=compute_krippendorff_alpha,
                                                compute_rho_fn=compute_spearman_rho,
                                                compute_tau_fn=compute_kendall_tau,
                                                seed=42 + trial_idx, 
                                                n_candidates=N_CANDIDATE_SUBSETS)

            if best_ids is not None:
                hm_sample = hm_full_df[hm_full_df["text_id"].isin(best_ids)]
                for est_metric_name, hm_target, est_compute_metric_fn in zip(metric_names, hm_targets, compute_metric_fns):
                    if est_metric_name in exclude_est:
                        continue
                    try:
                        #with warnings.filterwarnings("ignore", category=stats.ConstantInputWarning):
                        est_metric = est_compute_metric_fn(hm_sample)
                        if np.isfinite(est_metric):
                            errors_dict_list.append({
                                "model": model, "budget": k, "method": "metric_match",
                                "match_metric": match_metric_name,
                                "est_metric": est_metric_name,
                                "est_error": abs(est_metric - hm_target),
                                # "sp_corr": metadata[est_metric_name]["sp_corr"],
                                # "sp_corr_diff": metadata[est_metric_name]["sp_corr_diff"],
                            })
                    except Exception:
                        pass

    return errors_dict_list


def evaluate_cross_metric_matching(df, model_names, per_model_metric, metric_names, compute_metric_fns,
                                     budgets=range(5, 55, 5), n_trials=None, exclude_match=[], exclude_est=[]):
    """
    Evaluate metric estimators with different sampling strategies.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score, evaluation_axis
        model_names: List of model names
        per_model_metric: Dict from compute_metric_alignment
        budgets: Range of annotation budgets to test
        n_trials: Number of bootstrap trials (defaults to N_BOOTSTRAP_SAMPLES)

    Returns:
        results_dict_list: DataFrame with estimation errors
    """
    results_dict_list = []

    if n_trials is None:
        n_trials = N_BOOTSTRAP_SAMPLES

    for model in model_names:
        im_targets = per_model_metric[model]["im"]
        hm_targets = per_model_metric[model]["hm"]

        hm_full_df = df.loc[df["model_name"].isin([model, "original"])]

        text_ids = hm_full_df["text_id"].unique()

        if COMPARISON_MODE == "pairwise":
            im_full_df = _build_im_pairwise_df(df, model, model_names)
        else:
            im_full_df = df[df["model_name"] != "original"]

        for k in budgets:
            # Random baseline
            # random_error_dict_list = []
            random_error_dict_list = _run_random_trials(
                text_ids, k, n_trials, hm_full_df, model, metric_names, hm_targets, compute_metric_fns, exclude_est=exclude_est
            )

            # Variance-matched
            matched_error_dict_list = _run_metric_matched_trials(
                text_ids, k, n_trials, hm_full_df, im_full_df,
                                  model, metric_names, hm_targets, im_targets, compute_metric_fns, exclude_match=exclude_match, exclude_est=exclude_est
            )
            
            results_dict_list.extend(random_error_dict_list)
            results_dict_list.extend(matched_error_dict_list)

    return pd.DataFrame(results_dict_list)


# -------------------------
# MAIN
# -------------------------
def main(dataset):
    """Main execution function."""
    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    df = load_judge_scores(dataset, model_names, DATA_DIR, EVALUATION_AXES)

    print("\n" + "=" * 50)
    print("Running metric estimation experiment...")
    print("=" * 50)

    # Run experiment for each axis separately
    results_by_axis = {}

    for axis in EVALUATION_AXES[dataset]:
        print(f"\n{'=' * 50}")
        print(f"Running for axis: {axis}")
        print(f"{'=' * 50}")

        axis_df = df.loc[df["evaluation_axis"] == axis]

        # Recompute metric components for this axis
        axis_per_model_metric, _ = compute_metric_alignment(
            axis_df, model_names, compute_metric_fns=METRIC_FNS, mode=COMPARISON_MODE
        )

        # Run evaluation
        axis_results = evaluate_cross_metric_matching(
            axis_df, model_names, axis_per_model_metric, metric_names=METRIC_NAMES, compute_metric_fns=METRIC_FNS, exclude_est=EXCLUDE_EST, exclude_match=EXCLUDE_MATCH
        )
        results_by_axis[axis] = axis_results
        print(f"Collected {len(axis_results)} results for {axis}")

    # Combine per-axis results
    all_results = []
    for axis in EVALUATION_AXES[dataset]:
        if axis in results_by_axis and len(results_by_axis[axis]) > 0:
            axis_results = results_by_axis[axis].copy()
            axis_results["axis"] = axis
            all_results.append(axis_results)

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    return results


if __name__ == "__main__":

    for dataset in datasets:
        print("\n" + "=" * 50)
        print(f"RUNNING DATASET {dataset}")
        print("=" * 50)
        results = main(dataset)

        os.makedirs(os.path.join(PLOTS_DIR, dataset), exist_ok=True)
        metrics_for_match = [mn for mn in METRIC_NAMES if mn not in EXCLUDE_MATCH]
        for metric_matched in metrics_for_match:
            output_path = os.path.join(PLOTS_DIR, dataset, f"cross_metric_{metric_matched}_match_results.csv")
            results.loc[np.logical_or(results["method"] == "random", results["match_metric"] == metric_matched)].to_csv(output_path, index=False) # save ones that are random AND metric matched for each metric matched