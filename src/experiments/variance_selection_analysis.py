"""
Variance Selection Analysis for Reliability Estimation.

This experiment evaluates different sampling strategies for estimating ICC and
Krippendorff's alpha with limited human annotation budgets. It compares:
- Random sampling (baseline)
- Variance-matched sampling (matches inter-model variance structure)
- Max-expand sampling (selects most informative points)
"""

import os
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd

from src.utils.data_loading import load_judge_scores
from src.utils.reliability_metrics import (
    compute_ms_components,
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_reliability_ppi_corrected,
)
from src.utils.selection_strategies import (
    variance_matched_selection_ms,
    metric_matched_selection,
    max_expand_selection,
)
from src.utils.plotting import plot_all_results, load_results_dataframes
# Set random seed for reproducibility
np.random.seed(42)

# -------------------------
# CONFIGURATION (defaults)
# -------------------------
DEFAULT_N_BOOTSTRAP_SAMPLES = 10
DEFAULT_N_CANDIDATE_SUBSETS = 20
DEFAULT_TOTAL_ANNOTATIONS = 300
DEFAULT_DATASET = "hanna"
DEFAULT_MODEL_NAMES = ["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
# ["claude-3.5-sonnet", "gpt-4.1", "gpt-5"]
DEFAULT_TARGET_MODELS = None   # None → same as model_names
DEFAULT_ENSEMBLE_MODELS = None  # None → same as model_names
DEFAULT_DATA_DIR = "data/judge_scores"
DEFAULT_PLOTS_DIR = "results/02_18_large"
DEFAULT_COMPARISON_MODE = "average_pairwise"
# ONLINE_ACQUISITION=True  → cumulative/incremental selection: IDs chosen at budget k are
#                            locked in and carried forward to budget k+n (simulates a real
#                            annotation session where labels already collected are reused).
# ONLINE_ACQUISITION=False → batch selection: each budget level independently samples k
#                            items from scratch without any carryover.
DEFAULT_ONLINE_ACQUISITION = True

# Sampling strategies to compare
# Variance matching methods: "variance_matched_msb", "variance_matched_mse", "variance_matched_combined"
# Append "_imc" to any strategy name to apply inter-model control variate correction.
# Append "_tc"  to any variance-matched strategy name to apply adaptive bias correction to
#               the MSB/MSE selection targets (target = orig + mean(IM_obs) - mean(HM_obs)).
#               "_tc" and "_imc" can be combined (e.g. "variance_matched_combined_tc_imc").
SAMPLING_STRATEGIES = [
    "random",
    "random_imc",
    "variance_matched_combined",
    # "variance_matched_combined_imc",
    "variance_matched_combined_tc",
    # "variance_matched_combined_tc_imc",
    "metric_matched_icc",
    "metric_matched_alpha",
    "metric_matched_mse",
]

EVALUATION_AXES = {
    "hanna": ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval": ["Risk"],
    "mslr": ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"]
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Variance Selection Analysis for Reliability Estimation"
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        choices=list(EVALUATION_AXES.keys()),
        help=f"Dataset to use (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--model-names", type=str, nargs="+", default=DEFAULT_MODEL_NAMES,
        help="List of model names to load (default target + ensemble if not set)"
    )
    parser.add_argument(
        "--target-models", type=str, nargs="+", default=DEFAULT_TARGET_MODELS,
        help="Models to evaluate independently (default: same as --model-names)"
    )
    parser.add_argument(
        "--ensemble-models", type=str, nargs="+", default=DEFAULT_ENSEMBLE_MODELS,
        help="Models used for inter-model variance matching and IMC correction "
             "(default: same as --model-names). Each target is excluded from its own ensemble."
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Directory containing judge scores (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--plots-dir", type=str, default=DEFAULT_PLOTS_DIR,
        help=f"Directory to save plots (default: {DEFAULT_PLOTS_DIR})"
    )
    parser.add_argument(
        "--comparison-mode", type=str, default=DEFAULT_COMPARISON_MODE,
        choices=["average_pairwise", "pairwise_average", "aggregate"],
        help=f"Comparison mode (default: {DEFAULT_COMPARISON_MODE})"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP_SAMPLES,
        help=f"Number of bootstrap samples (default: {DEFAULT_N_BOOTSTRAP_SAMPLES})"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=DEFAULT_N_CANDIDATE_SUBSETS,
        help=f"Number of candidate subsets for variance matching (default: {DEFAULT_N_CANDIDATE_SUBSETS})"
    )
    parser.add_argument(
        "--total-annotations", type=int, default=DEFAULT_TOTAL_ANNOTATIONS,
        help=f"Total annotations budget (default: {DEFAULT_TOTAL_ANNOTATIONS})"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to a previously saved results directory (plots_dir from a prior run). "
             "If provided, skips computation and loads saved DataFrames to regenerate plots."
    )
    parser.add_argument(
        "--online-acquisition", action=argparse.BooleanOptionalAction,
        default=DEFAULT_ONLINE_ACQUISITION,
        help="If True (default), IDs selected at budget k are locked in and carried forward "
             "to larger budgets (online/incremental). If False, each budget level independently "
             "samples k items from scratch (batch selection)."
    )
    return parser.parse_args()


# Parse arguments (will use defaults if run without args)
args = parse_args()

# Set config from args
N_BOOTSTRAP_SAMPLES = args.n_bootstrap
N_CANDIDATE_SUBSETS = args.n_candidates
TOTAL_ANNOTATIONS = args.total_annotations
dataset = args.dataset
model_names = args.model_names
target_models = args.target_models if args.target_models is not None else model_names
ensemble_models = args.ensemble_models if args.ensemble_models is not None else model_names
# All models that need data loaded (union of target + ensemble, preserving order)
_seen = set()
models_to_load = [m for m in (target_models + ensemble_models) if not (m in _seen or _seen.add(m))]
DATA_DIR = args.data_dir
PLOTS_DIR = args.plots_dir
COMPARISON_MODE = args.comparison_mode
ONLINE_ACQUISITION = args.online_acquisition

os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------
# VARIANCE ALIGNMENT
# -------------------------
def compute_variance_alignment(df, target_models, ensemble_models, mode="aggregate"):
    """
    Compute inter-model and human-model variance components.

    For each target model t, the inter-model (IM) comparison uses t together
    with the ensemble models (excluding t itself so a model never compares
    against itself).  Only text_ids shared by all models in the IM set are used
    for IM computation.  The human-model (HM) comparison is always t vs "original".

    Args:
        df: DataFrame with text_id, model_name, evaluation_score
        target_models: List of target model names to evaluate
        ensemble_models: List of ensemble model names for inter-model comparison
        mode: "aggregate", "average_pairwise", or "pairwise_average"

    Returns:
        per_model_variance: dict mapping target model name to {im_msb, im_mse, hm_msb, hm_mse}
        aggregate_stats: dict with overall statistics
    """
    per_model_variance = {}
    im_msb_list, im_mse_list = [], []
    hm_msb_list, hm_mse_list = [], []

    if mode == "aggregate":
        for m in target_models:
            # Exclude the target itself from its own ensemble
            eff_ensemble = [e for e in ensemble_models if e != m]
            im_model_set = [m] + eff_ensemble

            # Filter to text_ids shared by all models in the IM set
            im_subset = df[df["model_name"].isin(im_model_set)]
            im_grouped = im_subset.groupby("text_id")["model_name"].nunique()
            shared_im_ids = im_grouped[im_grouped == len(im_model_set)].index
            im_df = im_subset[im_subset["text_id"].isin(shared_im_ids)]

            im_icc_obj = compute_ms_components(im_df)
            im_msb = im_icc_obj.msb
            im_mse = im_icc_obj.mse
            im_msb_list.append(im_msb)
            im_mse_list.append(im_mse)

            hm_icc_obj = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            hm_msb = hm_icc_obj.msb
            hm_mse = hm_icc_obj.mse
            hm_msb_list.append(hm_msb)
            hm_mse_list.append(hm_mse)

            per_model_variance[m] = {
                "im_msb": im_msb,
                "im_mse": im_mse,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        k_im = 1 + len([e for e in ensemble_models if e != target_models[0]]) if target_models else 0
        print(f"\nMode: AGGREGATE (k={k_im} for IM, k=2 for HM)")
        im_msb_mean = np.nanmean(im_msb_list) if im_msb_list else np.nan
        im_mse_mean = np.nanmean(im_mse_list) if im_mse_list else np.nan
        print(f"Inter-model (target + ensemble): MSB mean={im_msb_mean:.4f}, MSE mean={im_mse_mean:.4f}")

        aggregate_stats = {
            "im_msb": im_msb_mean,
            "im_mse": im_mse_mean,
            "hm_msb_list": hm_msb_list,
            "hm_mse_list": hm_mse_list
        }

    elif mode == "average_pairwise":
        for m in target_models:
            # Exclude the target itself from its own ensemble
            eff_ensemble = [e for e in ensemble_models if e != m]
            im_model_set = [m] + eff_ensemble
            im_pair_df = _build_im_pairwise_df(df, m, im_model_set)

            if len(im_pair_df) > 0:
                im_icc_obj = compute_ms_components(im_pair_df)
                im_msb = im_icc_obj.msb
                im_mse = im_icc_obj.mse
            else:
                im_msb, im_mse = np.nan, np.nan
            im_msb_list.append(im_msb)
            im_mse_list.append(im_mse)

            # Human-model: this model vs human
            hm_icc_obj = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            hm_msb = hm_icc_obj.msb
            hm_mse = hm_icc_obj.mse
            hm_msb_list.append(hm_msb)
            hm_mse_list.append(hm_mse)

            per_model_variance[m] = {
                "im_msb": im_msb,
                "im_mse": im_mse,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        print(f"\nMode: AVERAGE_PAIRWISE (k=2 for both IM and HM; ensemble averaged before computing metrics)")
        print(f"Inter-model MSBs (each target vs avg of ensemble): {[f'{x:.4f}' for x in im_msb_list]}")
        print(f"Inter-model MSEs: {[f'{x:.4f}' for x in im_mse_list]}")
        im_msb_mean = np.nanmean(im_msb_list) if im_msb_list else np.nan
        im_mse_mean = np.nanmean(im_mse_list) if im_mse_list else np.nan
        print(f"Inter-model mean: MSB={im_msb_mean:.4f}, MSE={im_mse_mean:.4f}")

        aggregate_stats = {
            "im_msb": im_msb_mean,
            "im_mse": im_mse_mean,
            "hm_msb_list": hm_msb_list,
            "hm_mse_list": hm_mse_list
        }

    elif mode == "pairwise_average":
        for m in target_models:
            # Exclude the target itself from its own ensemble
            eff_ensemble = [e for e in ensemble_models if e != m]

            # Compute pairwise MSB/MSE for each (target, ensemble_model) pair, then average
            pair_msb_list, pair_mse_list = [], []
            for e in eff_ensemble:
                pair_df = df[df["model_name"].isin([m, e])]
                if len(pair_df) == 0:
                    continue
                pair_icc_obj = compute_ms_components(pair_df)
                if pair_icc_obj is not None:
                    pair_msb_list.append(pair_icc_obj.msb)
                    pair_mse_list.append(pair_icc_obj.mse)

            im_msb = np.nanmean(pair_msb_list) if pair_msb_list else np.nan
            im_mse = np.nanmean(pair_mse_list) if pair_mse_list else np.nan
            im_msb_list.append(im_msb)
            im_mse_list.append(im_mse)

            # Human-model: this model vs human
            hm_icc_obj = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            hm_msb = hm_icc_obj.msb
            hm_mse = hm_icc_obj.mse
            hm_msb_list.append(hm_msb)
            hm_mse_list.append(hm_mse)

            per_model_variance[m] = {
                "im_msb": im_msb,
                "im_mse": im_mse,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        print(f"\nMode: PAIRWISE_AVERAGE (k=2 per pair; MSB/MSE averaged across pairs)")
        print(f"Inter-model MSBs (averaged across target-ensemble pairs): {[f'{x:.4f}' for x in im_msb_list]}")
        print(f"Inter-model MSEs: {[f'{x:.4f}' for x in im_mse_list]}")
        im_msb_mean = np.nanmean(im_msb_list) if im_msb_list else np.nan
        im_mse_mean = np.nanmean(im_mse_list) if im_mse_list else np.nan
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
        print(f"\nMSB: IM={aggregate_stats['im_msb']:.4f}, HM mean={np.nanmean(aggregate_stats['hm_msb_list']):.4f}")
        print(f"MSE: IM={aggregate_stats['im_mse']:.4f}, HM mean={np.nanmean(aggregate_stats['hm_mse_list']):.4f}")

    return per_model_variance, aggregate_stats


# -------------------------
# RELIABILITY ESTIMATION EXPERIMENT
# -------------------------
def _build_im_pairwise_df(df, model, model_names):
    """Build inter-model DataFrame for average_pairwise mode (model vs avg of others)."""
    other_models = [x for x in model_names if x != model]
    im_subset = df[df["model_name"].isin(model_names)]

    model_rows = (
        im_subset[im_subset["model_name"] == model][["text_id", "evaluation_score"]]
        .copy()
    )
    model_rows["model_name"] = model

    avg_rows = (
        im_subset[im_subset["model_name"].isin(other_models)]
        .groupby("text_id")["evaluation_score"]
        .mean()
        .reset_index()
    )
    avg_rows["model_name"] = "avg_other"

    return pd.concat(
        [model_rows[["text_id", "model_name", "evaluation_score"]],
         avg_rows[["text_id", "model_name", "evaluation_score"]]],
        ignore_index=True
    )


def _parse_strategy(strategy_name):
    """Parse strategy name into (base_name, im_correct).

    Suffixes:
        _imc – inter-model control variate correction (uses IM ICC as reference)
    """
    base = strategy_name
    im_correct = "_imc" in base
    base = base.replace("_imc", "")
    return base, im_correct


_SCORE_METHOD_MAP = {
    "variance_matched_msb": "msb_only",
    "variance_matched_mse": "mse_only",
    "variance_matched_combined": "combined",
    "variance_matched_combined_tc": "combined",   # same method; targets are bias-corrected
}

# Base strategy names that apply adaptive bias correction to the MSB/MSE selection
# targets.  "_tc" is intentionally NOT stripped by _parse_strategy so it stays in the
# base name and forms its own sampling group, separate from the non-corrected variants.
_TARGET_BC_BASES = {"variance_matched_combined_tc"}

# Maps metric-matched base strategy names to the single metric they should report errors for.
# Strategies not in this map report errors for all metrics.
_METRIC_MATCH_TARGET = {
    "metric_matched_icc": "icc",
    "metric_matched_alpha": "alpha",
    "metric_matched_mse": "mse",
}


def _run_trials_for_base(base_strategy, strategy_variants, text_ids, k, n_trials,
                          hm_full_df, im_full_df, model, true_icc, true_alpha, true_mse,
                          im_msb_target=None, im_mse_target=None,
                          im_models=None, true_im_icc=None, true_im_alpha=None,
                          past_im_msb_obs=None, past_im_mse_obs=None,
                          past_hm_msb_obs=None, past_hm_mse_obs=None,
                          prev_selected_per_trial=None,
                          online_acquisition=True,
                          fast_ms_fn=None):
    """Run trials for all strategy variants that share the same base sampling method.

    Samples text_ids once per trial, then computes every required correction
    on that single subsample. The _imc suffix uses the PPI correction:

        corrected = true_im + (hm_subset - im_subset)

    max_expand is deterministic, so it is evaluated only once regardless of
    n_trials.

    Cumulative vs batch selection (controlled by online_acquisition):
        online_acquisition=True  → IDs selected at a prior budget level are passed in via
            prev_selected_per_trial (a dict mapping trial_idx -> previously selected ID
            array). Each trial only samples the incremental IDs needed to reach k from
            the remaining pool, so the final selected set always includes every ID chosen
            at earlier budget levels.
        online_acquisition=False → each budget level independently samples k items from
            the full pool; prev_selected_per_trial is ignored and never updated.

    Bias-corrected selection targets:
        Before each trial the variance-matching targets are adjusted using the
        average discrepancy between IM and HM MS components observed on past
        subsets (from earlier trials in this call and from prior budget levels):

            effective_msb_target = im_msb_target + mean(past_im_msb) - mean(past_hm_msb)
            effective_mse_target = im_mse_target + mean(past_im_mse) - mean(past_hm_mse)

        Intuitively: if past subsets show IM MSB > HM MSB on average, we raise
        the selection target so chosen subsets better reflect the human-model
        variance structure.  With no past observations the original targets are
        used unchanged.

    After every trial the observed IM/HM MS components are accumulated and
    returned so the caller can thread them across budget levels.

    Returns:
        tuple: (
            results                   – dict mapping strategy name -> {"icc_errors", "alpha_errors"},
            new_im_msb_obs            – IM MSB values observed in this call's trials,
            new_im_mse_obs            – IM MSE values observed in this call's trials,
            new_hm_msb_obs            – HM MSB values observed in this call's trials,
            new_hm_mse_obs            – HM MSE values observed in this call's trials,
            updated_selected_per_trial – updated dict mapping trial_idx -> selected IDs,
        )
    """
    if past_im_msb_obs is None:
        past_im_msb_obs = []
    if past_im_mse_obs is None:
        past_im_mse_obs = []
    if past_hm_msb_obs is None:
        past_hm_msb_obs = []
    if past_hm_mse_obs is None:
        past_hm_mse_obs = []
    if prev_selected_per_trial is None:
        prev_selected_per_trial = {}
    # fast_ms_fn skips expensive pivot_table validation on clean candidate subsets.
    if fast_ms_fn is None:
        fast_ms_fn = compute_ms_components

    needs_ppi = any(imc for _, imc in [_parse_strategy(s) for s in strategy_variants])
    needs_plain = any(not imc for _, imc in [_parse_strategy(s) for s in strategy_variants])

    results = {s: {"icc_errors": [], "alpha_errors": [], "mse_errors": []} for s in strategy_variants}

    actual_trials = 1 if base_strategy == "max_expand" else n_trials

    # Observations accumulated within this call; returned to caller so they can
    # be threaded across successive budget levels.
    new_im_msb_obs = []
    new_im_mse_obs = []
    new_hm_msb_obs = []
    new_hm_mse_obs = []

    # Copy so we can update and return without mutating the caller's dict.
    updated_selected_per_trial = dict(prev_selected_per_trial)

    for trial_idx in range(actual_trials):
        seed = 42 + trial_idx

        # IDs locked in from prior budget levels for this trial (online mode only).
        if online_acquisition:
            forced_ids = updated_selected_per_trial.get(trial_idx, np.array([], dtype=text_ids.dtype))
        else:
            forced_ids = np.array([], dtype=text_ids.dtype)

        # ── Compute effective selection targets ────────────────────────────────
        # For _tc (target-corrected) strategies: adjust targets using the mean
        # IM-minus-HM MSB/MSE discrepancy observed on past subsets.
        # For all other strategies: use the original targets unchanged.
        if base_strategy in _TARGET_BC_BASES:
            all_im_msb_obs = past_im_msb_obs + new_im_msb_obs
            all_im_mse_obs = past_im_mse_obs + new_im_mse_obs
            all_hm_msb_obs = past_hm_msb_obs + new_hm_msb_obs
            all_hm_mse_obs = past_hm_mse_obs + new_hm_mse_obs

            if (im_msb_target is not None
                    and all_im_msb_obs and all_hm_msb_obs
                    and len(all_im_msb_obs) == len(all_hm_msb_obs)):
                effective_msb_target = (
                    im_msb_target + np.mean(all_im_msb_obs) - np.mean(all_hm_msb_obs)
                )
            else:
                effective_msb_target = im_msb_target

            if (im_mse_target is not None
                    and all_im_mse_obs and all_hm_mse_obs
                    and len(all_im_mse_obs) == len(all_hm_mse_obs)):
                effective_mse_target = (
                    im_mse_target + np.mean(all_im_mse_obs) - np.mean(all_hm_mse_obs)
                )
            else:
                effective_mse_target = im_mse_target
        else:
            effective_msb_target = im_msb_target
            effective_mse_target = im_mse_target

        # ── Sample once (only incremental IDs beyond forced_ids) ───────────────
        if base_strategy == "random":
            np.random.seed(seed)
            available = np.setdiff1d(text_ids, forced_ids)
            n_new = min(k - len(forced_ids), len(available))
            if n_new > 0:
                new_ids = np.random.choice(available, size=n_new, replace=False)
                sampled_ids = np.concatenate([forced_ids, new_ids]) if len(forced_ids) > 0 else new_ids
            else:
                sampled_ids = forced_ids[:k]
        elif base_strategy in _SCORE_METHOD_MAP:
            sampled_ids = variance_matched_selection_ms(
                text_ids, k, im_full_df, effective_msb_target, effective_mse_target,
                fast_ms_fn, seed=seed, n_candidates=N_CANDIDATE_SUBSETS,
                score_method=_SCORE_METHOD_MAP[base_strategy], forced_ids=forced_ids
            )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_icc":
            sampled_ids = metric_matched_selection(
                text_ids, k, im_full_df, true_im_icc, "icc",
                fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                forced_ids=forced_ids
            )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_alpha":
            sampled_ids = metric_matched_selection(
                text_ids, k, im_full_df, true_im_alpha, "alpha",
                fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                forced_ids=forced_ids
            )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_mse":
            sampled_ids = metric_matched_selection(
                text_ids, k, im_full_df, im_mse_target, "mse",
                fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                forced_ids=forced_ids
            )
            if sampled_ids is None:
                continue
        elif base_strategy == "max_expand":
            sampled_ids = max_expand_selection(im_full_df, k, compute_ms_components,
                                               forced_ids=forced_ids)
            if sampled_ids is None:
                continue
        else:
            continue

        # Record this trial's selected IDs for the next budget level (online mode only).
        if online_acquisition:
            updated_selected_per_trial[trial_idx] = np.asarray(sampled_ids)

        hm_sample = hm_full_df[hm_full_df["text_id"].isin(sampled_ids)]

        # ── Observe IM and HM MS components on this subset ───────────────────
        im_sample = im_full_df[im_full_df["text_id"].isin(sampled_ids)]
        im_ms = fast_ms_fn(im_sample)
        if im_ms is not None:
            new_im_msb_obs.append(im_ms.msb)
            new_im_mse_obs.append(im_ms.mse)

        hm_ms = fast_ms_fn(hm_sample)
        if hm_ms is not None:
            new_hm_msb_obs.append(hm_ms.msb)
            new_hm_mse_obs.append(hm_ms.mse)

        # ── Compute each correction type exactly once ───────────────────────
        plain_icc = plain_alpha = None
        ppi_icc = ppi_alpha = None

        try:
            if needs_plain:
                plain_icc = compute_icc_pingouin(hm_sample, models=[model, "original"])
                plain_alpha = compute_krippendorff_alpha(hm_sample, models=[model, "original"])

            if needs_ppi:
                ppi_icc, ppi_alpha = compute_reliability_ppi_corrected(
                    hm_sample, im_full_df, true_im_icc, true_im_alpha,
                    hm_models=[model, "original"], im_models=im_models
                )
        except Exception:
            continue

        # ── MSE (plain and PPI-corrected) ────────────────────────────────────
        plain_mse = hm_ms.mse if hm_ms is not None else None
        ppi_mse = None
        if (im_mse_target is not None and hm_ms is not None and im_ms is not None
                and np.isfinite(im_mse_target) and np.isfinite(hm_ms.mse) and np.isfinite(im_ms.mse)):
            ppi_mse = im_mse_target + (hm_ms.mse - im_ms.mse)

        # ── Record errors for each strategy variant ─────────────────────────
        for strategy in strategy_variants:
            base, imc = _parse_strategy(strategy)
            matched_metric = _METRIC_MATCH_TARGET.get(base)  # None means report all metrics

            if imc:
                est_icc, est_alpha = ppi_icc, ppi_alpha
                est_mse = ppi_mse
            else:
                est_icc, est_alpha = plain_icc, plain_alpha
                est_mse = plain_mse

            if matched_metric in (None, "icc"):
                if est_icc is not None and np.isfinite(est_icc):
                    results[strategy]["icc_errors"].append(min(2, abs(est_icc - true_icc)))
            if matched_metric in (None, "alpha"):
                if est_alpha is not None and np.isfinite(est_alpha):
                    results[strategy]["alpha_errors"].append(min(2, abs(est_alpha - true_alpha)))
            if matched_metric in (None, "mse"):
                if est_mse is not None and np.isfinite(est_mse) and true_mse is not None and np.isfinite(true_mse):
                    results[strategy]["mse_errors"].append(abs(est_mse - true_mse))

    return results, new_im_msb_obs, new_im_mse_obs, new_hm_msb_obs, new_hm_mse_obs, updated_selected_per_trial


def evaluate_reliability_estimators(df, target_models, ensemble_models, per_model_variance,
                                     budgets=range(5, 55, 5), n_trials=None,
                                     online_acquisition=True):
    """
    Evaluate ICC and Krippendorff's alpha estimators with different sampling strategies.

    For each target model, the inter-model DataFrame is built from that target
    plus the ensemble models (excluding the target from its own ensemble).

    Args:
        df: DataFrame with text_id, model_name, evaluation_score, evaluation_axis
        target_models: List of target model names to evaluate independently
        ensemble_models: List of ensemble model names for variance matching / IMC
        per_model_variance: Dict from compute_variance_alignment
        budgets: Range of annotation budgets to test
        n_trials: Number of bootstrap trials (defaults to N_BOOTSTRAP_SAMPLES)
        online_acquisition: If True (default), selected IDs carry forward across budget
            levels. If False, each budget level samples k items independently (batch).

    Returns:
        icc_results: DataFrame with ICC estimation errors
        alpha_results: DataFrame with Krippendorff's alpha estimation errors
        reliability_metadata: dict with true ICC and alpha values for each model
    """
    icc_results = []
    alpha_results = []
    mse_results = []
    reliability_metadata = {}

    if n_trials is None:
        n_trials = N_BOOTSTRAP_SAMPLES

    # Group strategies by base sampling method once — shared across all models/budgets
    strategies_by_base = {}
    for strategy in SAMPLING_STRATEGIES:
        base, _ = _parse_strategy(strategy)
        strategies_by_base.setdefault(base, []).append(strategy)

    # Reusable fast MS function that skips expensive pivot_table validation.
    # Safe because all candidate subsets are drawn from pre-filtered shared text_ids.
    _fast_ms = partial(compute_ms_components, validate=False)

    for model in target_models:
        im_msb_target = per_model_variance[model]["im_msb"]
        im_mse_target = per_model_variance[model]["im_mse"]

        hm_full_df = df[df["model_name"].isin([model, "original"])]

        # Compute true metrics
        true_icc = compute_icc_pingouin(hm_full_df, models=[model, "original"])
        print(f"\nComputing true Krippendorff's alpha for model: {model}")
        true_alpha = compute_krippendorff_alpha(hm_full_df, models=[model, "original"])
        hm_ms_full = compute_ms_components(hm_full_df[hm_full_df["model_name"].isin([model, "original"])])
        true_mse = hm_ms_full.mse if hm_ms_full is not None else np.nan
        print(f"{model}: ICC={true_icc:.4f}, Alpha={true_alpha:.4f}, MSE={true_mse:.4f}")

        # Build inter-model DataFrame using the target + ensemble (excluding target from its own ensemble)
        eff_ensemble = [e for e in ensemble_models if e != model]
        im_model_set = [model] + eff_ensemble
        if COMPARISON_MODE == "average_pairwise":
            im_full_df = _build_im_pairwise_df(df, model, im_model_set)
            im_models = [model, "avg_other"]
            im_icc = compute_icc_pingouin(im_full_df, models=im_models)
            im_alpha = compute_krippendorff_alpha(im_full_df, models=im_models)
        elif COMPARISON_MODE == "pairwise_average":
            # Compute ICC and alpha for each (target, ensemble_model) pair, then average
            pair_icc_list, pair_alpha_list = [], []
            for e in eff_ensemble:
                pair_df = df[df["model_name"].isin([model, e])]
                pair_icc = compute_icc_pingouin(pair_df, models=[model, e])
                pair_alpha = compute_krippendorff_alpha(pair_df, models=[model, e])
                if np.isfinite(pair_icc):
                    pair_icc_list.append(pair_icc)
                if np.isfinite(pair_alpha):
                    pair_alpha_list.append(pair_alpha)
            im_icc = np.nanmean(pair_icc_list) if pair_icc_list else np.nan
            im_alpha = np.nanmean(pair_alpha_list) if pair_alpha_list else np.nan
            # Use average_pairwise df for selection strategies (candidate subset evaluation)
            im_full_df = _build_im_pairwise_df(df, model, im_model_set)
            im_models = [model, "avg_other"]
        else:
            im_subset = df[df["model_name"].isin(im_model_set)]
            im_grouped = im_subset.groupby("text_id")["model_name"].nunique()
            shared_im_ids = im_grouped[im_grouped == len(im_model_set)].index
            im_full_df = im_subset[im_subset["text_id"].isin(shared_im_ids)]
            im_models = im_model_set
            im_icc = compute_icc_pingouin(im_full_df, models=im_models)
            im_alpha = compute_krippendorff_alpha(im_full_df, models=im_models)

        reliability_metadata[model] = {
            "true_hm_icc": true_icc,
            "true_hm_alpha": true_alpha,
            "true_hm_mse": true_mse,
            "im_icc": im_icc,
            "im_alpha": im_alpha,
            "im_mse": im_mse_target,
        }

        text_ids = hm_full_df["text_id"].unique()

        # Outer loop over strategies so each base strategy accumulates its own
        # observation history across budget levels for bias-corrected targeting,
        # and so that each trial's selected IDs are carried forward to the next
        # budget level (cumulative selection).
        for base_strategy, strategy_variants in strategies_by_base.items():
            past_im_msb_obs = []
            past_im_mse_obs = []
            past_hm_msb_obs = []
            past_hm_mse_obs = []
            prev_selected_per_trial = {}  # trial_idx -> array of IDs selected so far

            for k in budgets:
                trial_results, new_im_msb, new_im_mse, new_hm_msb, new_hm_mse, prev_selected_per_trial = (
                    _run_trials_for_base(
                        base_strategy, strategy_variants, text_ids, k, n_trials,
                        hm_full_df, im_full_df, model, true_icc, true_alpha, true_mse,
                        im_msb_target=im_msb_target, im_mse_target=im_mse_target,
                        im_models=im_models, true_im_icc=im_icc, true_im_alpha=im_alpha,
                        past_im_msb_obs=past_im_msb_obs, past_im_mse_obs=past_im_mse_obs,
                        past_hm_msb_obs=past_hm_msb_obs, past_hm_mse_obs=past_hm_mse_obs,
                        prev_selected_per_trial=prev_selected_per_trial,
                        online_acquisition=online_acquisition,
                        fast_ms_fn=_fast_ms,
                    )
                )
                # Extend history with this budget level's observations so the
                # next budget level benefits from all prior data.
                past_im_msb_obs.extend(new_im_msb)
                past_im_mse_obs.extend(new_im_mse)
                past_hm_msb_obs.extend(new_hm_msb)
                past_hm_mse_obs.extend(new_hm_mse)

                for strategy, errors in trial_results.items():
                    for error in errors["icc_errors"]:
                        icc_results.append({"model": model, "budget": k, "method": strategy, "estimation_error": error})
                    for error in errors["alpha_errors"]:
                        alpha_results.append({"model": model, "budget": k, "method": strategy, "estimation_error": error})
                    for error in errors["mse_errors"]:
                        mse_results.append({"model": model, "budget": k, "method": strategy, "estimation_error": error})

    return pd.DataFrame(icc_results), pd.DataFrame(alpha_results), pd.DataFrame(mse_results), reliability_metadata


# -------------------------
# PER-AXIS WORKER  (module-level so multiprocessing can pickle it)
# -------------------------
def _run_axis_worker(args):
    """Process a single evaluation axis. Runs in a worker process via multiprocessing."""
    axis, axis_df = args
    axis_per_model_variance, _ = compute_variance_alignment(
        axis_df, target_models, ensemble_models, mode=COMPARISON_MODE
    )
    axis_icc, axis_alpha, axis_mse, axis_metadata = evaluate_reliability_estimators(
        axis_df, target_models, ensemble_models, axis_per_model_variance,
        online_acquisition=ONLINE_ACQUISITION
    )
    return axis, axis_icc, axis_alpha, axis_mse, axis_metadata


# -------------------------
# MAIN
# -------------------------
def main():
    """Main execution function."""
    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    df = load_judge_scores(dataset, models_to_load, DATA_DIR, EVALUATION_AXES)

    print("\n" + "=" * 50)
    print("Running ICC and Krippendorff's Alpha estimation experiment...")
    print("=" * 50)

    # Pre-filter each axis DataFrame to shared text_ids, then run axes in parallel.
    # Each axis is fully independent, so we can parallelize freely.
    axes = EVALUATION_AXES[dataset]
    axis_jobs = []
    for axis in axes:
        axis_df = df[df["evaluation_axis"] == axis]
        num_models = axis_df["model_name"].nunique()
        texts_per_model = axis_df.groupby("text_id")["model_name"].nunique()
        shared_text_ids = texts_per_model[texts_per_model == num_models].index[:TOTAL_ANNOTATIONS]
        axis_df = axis_df[axis_df["text_id"].isin(shared_text_ids)]
        print(f"Axis '{axis}': {len(axis_df)} rows after filtering to {len(shared_text_ids)} shared text_ids")
        axis_jobs.append((axis, axis_df))

    n_workers = min(len(axis_jobs), os.cpu_count() or 1)
    print(f"\nRunning {len(axis_jobs)} axes across {n_workers} parallel workers...")

    if n_workers > 1:
        # Use fork-based pool so worker processes inherit all module-level globals
        # (COMPARISON_MODE, ONLINE_ACQUISITION, N_BOOTSTRAP_SAMPLES, etc.).
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            axis_results = pool.map(_run_axis_worker, axis_jobs)
    else:
        axis_results = [_run_axis_worker(job) for job in axis_jobs]

    icc_results_by_axis = {}
    alpha_results_by_axis = {}
    mse_results_by_axis = {}
    reliability_metadata_by_axis = {}
    for axis, axis_icc, axis_alpha, axis_mse, axis_metadata in axis_results:
        icc_results_by_axis[axis] = axis_icc
        alpha_results_by_axis[axis] = axis_alpha
        mse_results_by_axis[axis] = axis_mse
        reliability_metadata_by_axis[axis] = axis_metadata
        print(f"Collected {len(axis_icc)} ICC, {len(axis_alpha)} Alpha, {len(axis_mse)} MSE results for {axis}")

    # Combine per-axis results
    all_icc_results = []
    all_alpha_results = []
    all_mse_results = []
    for axis in EVALUATION_AXES[dataset]:
        if axis in icc_results_by_axis and len(icc_results_by_axis[axis]) > 0:
            axis_icc = icc_results_by_axis[axis].copy()
            axis_icc["axis"] = axis
            all_icc_results.append(axis_icc)
        if axis in alpha_results_by_axis and len(alpha_results_by_axis[axis]) > 0:
            axis_alpha = alpha_results_by_axis[axis].copy()
            axis_alpha["axis"] = axis
            all_alpha_results.append(axis_alpha)
        if axis in mse_results_by_axis and len(mse_results_by_axis[axis]) > 0:
            axis_mse = mse_results_by_axis[axis].copy()
            axis_mse["axis"] = axis
            all_mse_results.append(axis_mse)

    icc_results = pd.concat(all_icc_results, ignore_index=True) if all_icc_results else pd.DataFrame()
    alpha_results = pd.concat(all_alpha_results, ignore_index=True) if all_alpha_results else pd.DataFrame()
    mse_results = pd.concat(all_mse_results, ignore_index=True) if all_mse_results else pd.DataFrame()

    # Compute aggregate reliability metadata
    reliability_metadata_all = {}
    for model in target_models:
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
        hm_mse_vals = [
            reliability_metadata_by_axis[ax][model]["true_hm_mse"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["true_hm_mse"])
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
        im_mse_vals = [
            reliability_metadata_by_axis[ax][model]["im_mse"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["im_mse"])
        ]

        reliability_metadata_all[model] = {
            "true_hm_icc": np.mean(hm_icc_vals) if hm_icc_vals else np.nan,
            "true_hm_alpha": np.mean(hm_alpha_vals) if hm_alpha_vals else np.nan,
            "true_hm_mse": np.mean(hm_mse_vals) if hm_mse_vals else np.nan,
            "im_icc": np.mean(im_icc_vals) if im_icc_vals else np.nan,
            "im_alpha": np.mean(im_alpha_vals) if im_alpha_vals else np.nan,
            "im_mse": np.mean(im_mse_vals) if im_mse_vals else np.nan,
        }

    print(f"\n{'=' * 50}")
    print("AGGREGATE RESULTS (combined from per-axis)")
    print(f"{'=' * 50}")
    print(f"Total ICC results collected: {len(icc_results)}")
    print(f"Total Alpha results collected: {len(alpha_results)}")
    print(f"Total MSE results collected: {len(mse_results)}")

    if len(icc_results) > 0:
        print(f"ICC Results by method:")
        for method in icc_results["method"].unique():
            count = len(icc_results[icc_results["method"] == method])
            print(f"  {method}: {count}")

    return (icc_results, alpha_results, mse_results,
            icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
            reliability_metadata_all, reliability_metadata_by_axis)


if __name__ == "__main__":
    if args.results_dir is not None:
        print(f"\nLoading saved results from: {args.results_dir} (dataset={dataset})")
        (icc_results, alpha_results, mse_results,
         icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
         reliability_metadata_all, reliability_metadata_by_axis) = load_results_dataframes(args.results_dir, dataset)
    else:
        (icc_results, alpha_results, mse_results,
         icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
         reliability_metadata_all, reliability_metadata_by_axis) = main()

    plot_all_results(
        icc_results, alpha_results, mse_results,
        icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
        reliability_metadata_all, reliability_metadata_by_axis,
        dataset, PLOTS_DIR, COMPARISON_MODE
    )
