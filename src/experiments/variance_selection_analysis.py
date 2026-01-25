"""
Variance Selection Analysis for Reliability Estimation.

This experiment evaluates different sampling strategies for estimating ICC and
Krippendorff's alpha with limited human annotation budgets. It compares:
- Random sampling (baseline)
- Variance-matched sampling (matches inter-model variance structure)
- Max-expand sampling (selects most informative points)
"""

import os
import numpy as np
import pandas as pd

from src.utils.data_loading import load_judge_scores
from src.utils.reliability_metrics import (
    compute_ms_components,
    compute_icc_pingouin,
    compute_krippendorff_alpha,
)
from src.utils.selection_strategies import (
    variance_matched_selection_ms,
    max_expand_selection,
)
from src.utils.plotting import plot_all_results
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

# Runtime configuration
dataset = "hanna" #"medval" 
# model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "gpt-5", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
# model_names = ["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5"]
DATA_DIR = "data/judge_scores"
PLOTS_DIR = "results/01_25_big"
COMPARISON_MODE = "pairwise"  # "pairwise" or "aggregate"

os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------
# VARIANCE ALIGNMENT
# -------------------------
def compute_variance_alignment(df, model_names, mode="aggregate"):
    """
    Compute inter-model and human-model variance components.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score
        model_names: List of model names (excluding "original")
        mode: "aggregate" or "pairwise"

    Returns:
        per_model_variance: dict mapping model_name to {im_msb, im_mse, hm_msb, hm_mse}
        aggregate_stats: dict with overall statistics
    """
    per_model_variance = {}

    if mode == "aggregate":
        im_df = df[df["model_name"] != "original"]
        im_icc_obj = compute_ms_components(im_df)
        msb_expand_global, im_msb_global, mse_expand_global, im_mse_global, icc_expand_global, im_icc_global = \
            im_icc_obj.msb_expand, im_icc_obj.msb, im_icc_obj.mse_expand, im_icc_obj.mse, im_icc_obj.icc_expand, im_icc_obj.icc

        for m in model_names:
            hm_icc_obj = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            hm_msb_expand, hm_msb, hm_mse_expand, hm_mse, hm_icc_expand, hm_icc = \
            hm_icc_obj.msb_expand, hm_icc_obj.msb, hm_icc_obj.mse_expand, hm_icc_obj.mse, hm_icc_obj.icc_expand, hm_icc_obj.icc
            per_model_variance[m] = {
                "im_msb": im_msb_global,
                "im_mse": im_mse_global,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        print(f"\nMode: AGGREGATE (k=4 for IM, k=2 for HM)")
        print(f"Inter-model (all 4 models): MSB={im_msb_global:.4f}, MSE={im_mse_global:.4f}")

        aggregate_stats = {
            "im_msb": im_msb_global,
            "im_mse": im_mse_global,
            "hm_msb_list": [per_model_variance[m]["hm_msb"] for m in model_names],
            "hm_mse_list": [per_model_variance[m]["hm_mse"] for m in model_names]
        }

    elif mode == "pairwise":
        im_msb_list, im_mse_list = [], []
        hm_msb_list, hm_mse_list = [], []

        im_subset = df[df["model_name"].isin(model_names)].copy()
        im_grouped = im_subset.groupby("text_id")

        for m in model_names:
            # other_models = [x for x in model_names if x != m]

            # # Inter-model: this model vs avg of other models
            # im_avg_df = []
            # for text_id, text_data in im_grouped:
            #     model_score = text_data.loc[text_data["model_name"] == m, "evaluation_score"]
            #     if len(model_score) > 0:
            #         im_avg_df.append({
            #             "text_id": text_id,
            #             "model_name": m,
            #             "evaluation_score": model_score.iloc[0]
            #         })
            #     other_scores = text_data.loc[
            #         text_data["model_name"].isin(other_models), "evaluation_score"
            #     ]
            #     if len(other_scores) > 0:
            #         im_avg_df.append({
            #             "text_id": text_id,
            #             "model_name": "avg_other",
            #             "evaluation_score": other_scores.mean()
            #         })
            im_pair_df = _build_im_pairwise_df(df, m, model_names)
            # im_pair_df = pd.DataFrame(im_avg_df)
            if len(im_pair_df) > 0:
                im_icc_obj = compute_ms_components(im_pair_df)
                im_msb_expand, im_msb, im_mse_expand, im_mse, im_icc_expand, im_icc = im_icc_obj.msb_expand, im_icc_obj.msb, im_icc_obj.mse_expand, im_icc_obj.mse, im_icc_obj.icc_expand, im_icc_obj.icc
                im_msb_list.append(im_msb)
                im_mse_list.append(im_mse)
            else:
                im_msb, im_mse = np.nan, np.nan
                im_msb_list.append(im_msb)
                im_mse_list.append(im_mse)

            # Human-model: this model vs human
            hm_icc_obj = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            hm_msb_expand, hm_msb, hm_mse_expand, hm_mse, hm_icc_expand, hm_icc = hm_icc_obj.msb_expand, hm_icc_obj.msb, hm_icc_obj.mse_expand, hm_icc_obj.mse, hm_icc_obj.icc_expand, hm_icc_obj.icc
            hm_msb_list.append(hm_msb)
            hm_mse_list.append(hm_mse)

            per_model_variance[m] = {
                "im_msb": im_msb,
                "im_mse": im_mse,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        print(f"\nMode: PAIRWISE (k=2 for both IM and HM)")
        print(f"Inter-model MSBs (each model vs avg of others): {[f'{x:.4f}' for x in im_msb_list]}")
        print(f"Inter-model MSEs: {[f'{x:.4f}' for x in im_mse_list]}")
        im_msb_mean = np.mean(im_msb_list)
        im_mse_mean = np.mean(im_mse_list)
        print(f"Inter-model mean: MSB={im_msb_mean:.4f}, MSE={im_mse_mean:.4f}")

        aggregate_stats = {
            "im_msb": im_msb_mean,
            "im_mse": im_mse_mean,
            "hm_msb_list": hm_msb_list,
            "hm_mse_list": hm_mse_list
        }

    print(f"Human-model MSBs: {[f'{x:.4f}' for x in aggregate_stats['hm_msb_list']]}")
    print(f"Human-model MSEs: {[f'{x:.4f}' for x in aggregate_stats['hm_mse_list']]}")

    if len(aggregate_stats['hm_msb_list']) > 1:
        print(f"\nMSB: IM={aggregate_stats['im_msb']:.4f}, HM mean={np.mean(aggregate_stats['hm_msb_list']):.4f}")
        print(f"MSE: IM={aggregate_stats['im_mse']:.4f}, HM mean={np.mean(aggregate_stats['hm_mse_list']):.4f}")

    return per_model_variance, aggregate_stats


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


def _run_random_trials(text_ids, k, n_trials, hm_full_df, model, true_icc, true_alpha):
    """Run random sampling trials and collect estimation errors."""
    icc_errors = []
    alpha_errors = []

    for trial_idx in range(n_trials):
        np.random.seed(42 + trial_idx)
        sampled_ids = np.random.choice(text_ids, size=min(k, len(text_ids)), replace=False)
        hm_sample = hm_full_df[hm_full_df["text_id"].isin(sampled_ids)]

        try:
            est_icc = compute_icc_pingouin(hm_sample, models=[model, "original"])
            if np.isfinite(est_icc):
                icc_errors.append(min(2, abs(est_icc - true_icc)))

            est_alpha = compute_krippendorff_alpha(hm_sample, models=[model, "original"])
            if np.isfinite(est_alpha):
                alpha_errors.append(min(2, abs(est_alpha - true_alpha)))
        except Exception:
            pass

    return icc_errors, alpha_errors


def _run_variance_matched_trials(text_ids, k, n_trials, hm_full_df, im_full_df,
                                  model, true_icc, true_alpha, im_msb_target, im_mse_target):
    """Run variance-matched sampling trials and collect estimation errors."""
    icc_errors = []
    alpha_errors = []

    for trial_idx in range(n_trials):
        best_ids = variance_matched_selection_ms(
            text_ids, k, im_full_df, im_msb_target, im_mse_target,
            compute_ms_components, seed=42 + trial_idx, n_candidates=N_CANDIDATE_SUBSETS
        )

        if best_ids is not None:
            hm_sample = hm_full_df[hm_full_df["text_id"].isin(best_ids)]
            try:
                est_icc = compute_icc_pingouin(hm_sample, models=[model, "original"])
                if np.isfinite(est_icc):
                    icc_errors.append(min(2, abs(est_icc - true_icc)))

                est_alpha = compute_krippendorff_alpha(hm_sample, models=[model, "original"])
                if np.isfinite(est_alpha):
                    alpha_errors.append(min(2, abs(est_alpha - true_alpha)))
            except Exception:
                pass

    return icc_errors, alpha_errors


def _run_max_expand_trial(k, hm_full_df, im_full_df, model, true_icc, true_alpha):
    """Run max-expand sampling (deterministic, single trial)."""
    icc_errors = []
    alpha_errors = []

    best_ids = max_expand_selection(im_full_df, k, compute_ms_components)

    if best_ids is not None:
        hm_sample = hm_full_df[hm_full_df["text_id"].isin(best_ids)]
        try:
            est_icc = compute_icc_pingouin(hm_sample, models=[model, "original"])
            if np.isfinite(est_icc):
                icc_errors.append(min(2, abs(est_icc - true_icc)))

            est_alpha = compute_krippendorff_alpha(hm_sample, models=[model, "original"])
            if np.isfinite(est_alpha):
                alpha_errors.append(min(2, abs(est_alpha - true_alpha)))
        except Exception:
            pass

    return icc_errors, alpha_errors


def evaluate_reliability_estimators(df, model_names, per_model_variance,
                                     budgets=range(5, 55, 5), n_trials=None):
    """
    Evaluate ICC and Krippendorff's alpha estimators with different sampling strategies.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score, evaluation_axis
        model_names: List of model names
        per_model_variance: Dict from compute_variance_alignment
        budgets: Range of annotation budgets to test
        n_trials: Number of bootstrap trials (defaults to N_BOOTSTRAP_SAMPLES)

    Returns:
        icc_results: DataFrame with ICC estimation errors
        alpha_results: DataFrame with Krippendorff's alpha estimation errors
        reliability_metadata: dict with true ICC and alpha values for each model
    """
    icc_results = []
    alpha_results = []
    reliability_metadata = {}

    if n_trials is None:
        n_trials = N_BOOTSTRAP_SAMPLES

    for model in model_names:
        im_msb_target = per_model_variance[model]["im_msb"]
        im_mse_target = per_model_variance[model]["im_mse"]

        hm_full_df = df[df["model_name"].isin([model, "original"])]

        # Compute true metrics
        true_icc = compute_icc_pingouin(hm_full_df, models=[model, "original"])
        print(f"\nComputing true Krippendorff's alpha for model: {model}")
        true_alpha = compute_krippendorff_alpha(hm_full_df, models=[model, "original"])
        print(f"{model}: ICC={true_icc:.4f}, Alpha={true_alpha:.4f}")

        # Build inter-model DataFrame
        if COMPARISON_MODE == "pairwise":
            im_full_df = _build_im_pairwise_df(df, model, model_names)
            im_icc = compute_icc_pingouin(im_full_df, models=[model, "avg_other"])
            im_alpha = compute_krippendorff_alpha(im_full_df, models=[model, "avg_other"])
        else:
            im_full_df = df[df["model_name"] != "original"]
            im_icc = compute_icc_pingouin(im_full_df, models=model_names)
            im_alpha = compute_krippendorff_alpha(im_full_df, models=model_names)

        reliability_metadata[model] = {
            "true_hm_icc": true_icc,
            "true_hm_alpha": true_alpha,
            "im_icc": im_icc,
            "im_alpha": im_alpha
        }

        text_ids = hm_full_df["text_id"].unique()

        for k in budgets:
            # Random baseline
            random_icc, random_alpha = _run_random_trials(
                text_ids, k, n_trials, hm_full_df, model, true_icc, true_alpha
            )
            for error in random_icc:
                icc_results.append({"model": model, "budget": k, "method": "random", "estimation_error": error})
            for error in random_alpha:
                alpha_results.append({"model": model, "budget": k, "method": "random", "estimation_error": error})

            # Variance-matched
            matched_icc, matched_alpha = _run_variance_matched_trials(
                text_ids, k, n_trials, hm_full_df, im_full_df,
                model, true_icc, true_alpha, im_msb_target, im_mse_target
            )
            for error in matched_icc:
                icc_results.append({"model": model, "budget": k, "method": "variance_matched", "estimation_error": error})
            for error in matched_alpha:
                alpha_results.append({"model": model, "budget": k, "method": "variance_matched", "estimation_error": error})

            # Max-expand
            max_icc, max_alpha = _run_max_expand_trial(
                k, hm_full_df, im_full_df, model, true_icc, true_alpha
            )
            for error in max_icc:
                icc_results.append({"model": model, "budget": k, "method": "max_expand", "estimation_error": error})
            for error in max_alpha:
                alpha_results.append({"model": model, "budget": k, "method": "max_expand", "estimation_error": error})

    return pd.DataFrame(icc_results), pd.DataFrame(alpha_results), reliability_metadata


# -------------------------
# MAIN
# -------------------------
def main():
    """Main execution function."""
    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    df = load_judge_scores(dataset, model_names, DATA_DIR, EVALUATION_AXES)

    print("\n" + "=" * 50)
    print("Running ICC and Krippendorff's Alpha estimation experiment...")
    print("=" * 50)

    # Run experiment for each axis separately
    icc_results_by_axis = {}
    alpha_results_by_axis = {}
    reliability_metadata_by_axis = {}

    for axis in EVALUATION_AXES[dataset]:
        print(f"\n{'=' * 50}")
        print(f"Running for axis: {axis}")
        print(f"{'=' * 50}")

        axis_df = df[df["evaluation_axis"] == axis]
        # how many unique models
        # number of models
        num_models = axis_df["model_name"].nunique()

        # count models per text_id
        texts_per_model = (
            axis_df.groupby("text_id")["model_name"]
            .nunique()
        )

        # text_ids shared across all models
        shared_text_ids = texts_per_model[texts_per_model == num_models].index

        # keep only shared text_ids
        axis_df = axis_df[axis_df["text_id"].isin(shared_text_ids)]

        print(f"Remaining rows: {len(axis_df)}")
        

        # Recompute variance components for this axis
        axis_per_model_variance, _ = compute_variance_alignment(
            axis_df, model_names, mode=COMPARISON_MODE
        )

        # Run evaluation
        axis_icc, axis_alpha, axis_metadata = evaluate_reliability_estimators(
            axis_df, model_names, axis_per_model_variance
        )
        icc_results_by_axis[axis] = axis_icc
        alpha_results_by_axis[axis] = axis_alpha
        reliability_metadata_by_axis[axis] = axis_metadata
        print(f"Collected {len(axis_icc)} ICC results and {len(axis_alpha)} Alpha results for {axis}")

    # Combine per-axis results
    all_icc_results = []
    all_alpha_results = []
    for axis in EVALUATION_AXES[dataset]:
        if axis in icc_results_by_axis and len(icc_results_by_axis[axis]) > 0:
            axis_icc = icc_results_by_axis[axis].copy()
            axis_icc["axis"] = axis
            all_icc_results.append(axis_icc)
        if axis in alpha_results_by_axis and len(alpha_results_by_axis[axis]) > 0:
            axis_alpha = alpha_results_by_axis[axis].copy()
            axis_alpha["axis"] = axis
            all_alpha_results.append(axis_alpha)

    icc_results = pd.concat(all_icc_results, ignore_index=True) if all_icc_results else pd.DataFrame()
    alpha_results = pd.concat(all_alpha_results, ignore_index=True) if all_alpha_results else pd.DataFrame()

    # Compute aggregate reliability metadata
    reliability_metadata_all = {}
    for model in model_names:
        hm_icc_vals = [
            reliability_metadata_by_axis[ax][model]["true_hm_icc"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["true_hm_icc"])
        ]
        hm_alpha_vals = [
            reliability_metadata_by_axis[ax][model]["true_hm_alpha"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["true_hm_alpha"])
        ]
        im_icc_vals = [
            reliability_metadata_by_axis[ax][model]["im_icc"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["im_icc"])
        ]
        im_alpha_vals = [
            reliability_metadata_by_axis[ax][model]["im_alpha"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["im_alpha"])
        ]

        reliability_metadata_all[model] = {
            "true_hm_icc": np.mean(hm_icc_vals) if hm_icc_vals else np.nan,
            "true_hm_alpha": np.mean(hm_alpha_vals) if hm_alpha_vals else np.nan,
            "im_icc": np.mean(im_icc_vals) if im_icc_vals else np.nan,
            "im_alpha": np.mean(im_alpha_vals) if im_alpha_vals else np.nan,
        }

    print(f"\n{'=' * 50}")
    print("AGGREGATE RESULTS (combined from per-axis)")
    print(f"{'=' * 50}")
    print(f"Total ICC results collected: {len(icc_results)}")
    print(f"Total Alpha results collected: {len(alpha_results)}")

    if len(icc_results) > 0:
        print(f"ICC Results by method:")
        for method in icc_results["method"].unique():
            count = len(icc_results[icc_results["method"] == method])
            print(f"  {method}: {count}")

    return (icc_results, alpha_results, icc_results_by_axis, alpha_results_by_axis,
            reliability_metadata_all, reliability_metadata_by_axis)


if __name__ == "__main__":
    (icc_results, alpha_results, icc_results_by_axis, alpha_results_by_axis,
     reliability_metadata_all, reliability_metadata_by_axis) = main()

    plot_all_results(
        icc_results, alpha_results,
        icc_results_by_axis, alpha_results_by_axis,
        reliability_metadata_all, reliability_metadata_by_axis,
        dataset, PLOTS_DIR, COMPARISON_MODE
    )
