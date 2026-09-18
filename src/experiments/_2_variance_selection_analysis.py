import os
import argparse
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.utils.data_loading import load_judge_scores, retrieve_embeddings

from src.utils.match_metrics import (
    compute_ms_components,
    compute_msb,
    compute_msre,
    compute_weighted_msb_msre,
    compute_mean_sq_err,
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_spearman_rho,
    compute_reliability_ppi_corrected,
    compute_reliability_active_corrected,
    compute_kendall_tau,
    active_uncertainty_fn
)
from src.utils.selection_strategies import (
    variance_matched_selection_ms,
    metric_matched_selection,
    max_expand_selection,
    stratified_target_selection,
    batch_active_statistical_inf_selection,
    cluster_text_based
    # pairwise_metric_matched_selection,
    # pairwise_variance_matched_selection_ms
)

from src.utils.plotting import plot_all_results, load_results_dataframes, save_predictor_inputs

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# -------------------------
# CONFIGURATION (defaults)
# -------------------------

DEFAULT_N_BOOTSTRAP_SAMPLES = 40
DEFAULT_N_CANDIDATE_SUBSETS = 20
DEFAULT_TOTAL_ANNOTATIONS = 300
DEFAULT_DATASET = "medval"
DEFAULT_MODEL_NAMES = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5", "deepseek-r1", "gemini-2.5-pro"] # ["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
DEFAULT_TARGET_MODELS = None   # None → same as model_names
DEFAULT_ENSEMBLE_MODELS = None #["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"] #("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct") #("claude-3.5-sonnet" "gpt-4.1" "gpt-5" "deepseek-r1" "gemini-2.5-pro") #("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")   # None → same as model_names
DEFAULT_DATA_DIR = "data/judge_scores"
DEFAULT_PLOTS_DIR = f"results/09_18_cluster_text_only/{DEFAULT_DATASET}"
DEFAULT_COMPARISON_MODE = "pairwise_average"
DEFAULT_ONLINE_ACQUISITION = False
DEFAULT_STEP_SIZE = 5
DEFAULT_MAX_BUDGET = 50
DEFAULT_UNCERTAINTY_FN = active_uncertainty_fn

SAMPLING_STRATEGIES = [
    # "random",
    # "random_imc",
    # "stratified",
    # "variance_matched_combined",
    # "variance_matched_combined_imc",
    # "variance_matched_combined_tc",
    # "variance_matched_combined_tc_imc",
    # "variance_matched_msb",
    # "variance_matched_msb_imc",
    # "variance_matched_msb_tc",
    # "variance_matched_msb_tc_imc",
    # "variance_matched_weighted_.2",
    # "variance_matched_weighted_.2_imc",
    # "variance_matched_weighted_.5",
    # "variance_matched_weighted_.5_imc",
    # "variance_matched_weighted_.7",
    # "variance_matched_weighted_.7_imc",
    # "variance_matched_weighted_.9",
    # "variance_matched_weighted_.9_imc",
    # "proxy_oracle",
    # "proxy_oracle_imc",
    # "oracle_msb_mse",
    # "oracle_msb",
    # "oracle_mse",
    # "oracle_icc",
    # "oracle_alpha",
    # "oracle_mean_squared_error",
    # "oracle_rho",
    # "oracle_tau",
    # "metric_matched_icc", 
    # "metric_matched_alpha",
    # "metric_matched_rho",
    # "metric_matched_tau",
    # "metric_matched_mse",
    # "batch_active_statistical_inf"
    "cluster_text_based"
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
    parser.add_argument(
        "--step-size", type=int, default=DEFAULT_STEP_SIZE,
        help=f"Step size for annotation budget levels (default: {DEFAULT_STEP_SIZE}). "
             "E.g. 5 → budgets [5, 10, 15, ...], 1 → budgets [5, 6, 7, ...]."
    )
    parser.add_argument(
        "--max-budget", type=int, default=DEFAULT_MAX_BUDGET,
        help=f"Maximum annotation budget to evaluate (default: {DEFAULT_MAX_BUDGET}). "
             "Budgets are tested from 5 to this value in steps of --step-size."
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
STEP_SIZE = args.step_size
MAX_BUDGET = args.max_budget

os.makedirs(PLOTS_DIR, exist_ok=True)

def compute_variance_alignment(df, target_models, ensemble_models, mode="aggregate"):
    """
    Compute inter-model and human-model ICC variance components.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score
        target_models: List of target model names to evaluate
        ensemble_models: List of ensemble model names for inter-model comparison
        mode: "aggregate", "average_pairwise", or "pairwise_average"

    Returns:
        per_model_variance: dict mapping target model name to {im_msb, im_mse, hm_msb, hm_mse}
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
            im_mse = im_icc_obj.mse  # ANOVA MSE
            im_msb_list.append(im_msb)
            im_mse_list.append(im_mse)

            hm_df = df[df["model_name"].isin([m, "original"])]
            hm_icc_obj = compute_ms_components(hm_df)
            hm_msb = hm_icc_obj.msb
            hm_mse = hm_icc_obj.mse  # ANOVA MSE
            hm_msb_list.append(hm_msb)
            hm_mse_list.append(hm_mse)

            per_model_variance[m] = {
                "im_msb": im_msb,
                "im_mse": im_mse,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        k_im = 1 + len([e for e in ensemble_models if e != target_models[0]]) if target_models else 0
        im_msb_mean = np.nanmean(im_msb_list) if im_msb_list else np.nan
        im_mse_mean = np.nanmean(im_mse_list) if im_mse_list else np.nan

    elif mode == "average_pairwise":
        for m in target_models:
            # Exclude the target itself from its own ensemble
            eff_ensemble = [e for e in ensemble_models if e != m]
            im_model_set = [m] + eff_ensemble
            im_pair_df = _build_im_pairwise_df(df, m, im_model_set)

            if len(im_pair_df) > 0:
                im_icc_obj = compute_ms_components(im_pair_df)
                im_msb = im_icc_obj.msb
                im_mse = im_icc_obj.mse  # ANOVA MSE
            else:
                im_msb, im_mse = np.nan, np.nan
            im_msb_list.append(im_msb)
            im_mse_list.append(im_mse)

            # Human-model: this model vs human
            hm_df = df[df["model_name"].isin([m, "original"])]
            hm_icc_obj = compute_ms_components(hm_df)
            try:
                hm_msb = hm_icc_obj.msb
                hm_mse = hm_icc_obj.mse  # ANOVA MSE
            except Exception as e:
                print(f"Error computing HM MS components for model {m}: {e}")
                hm_msb, hm_mse = np.nan, np.nan
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
                    pair_mse_list.append(pair_icc_obj.mse)  # ANOVA MSE

            im_msb = np.nanmean(pair_msb_list) if pair_msb_list else np.nan
            im_mse = np.nanmean(pair_mse_list) if pair_mse_list else np.nan
            im_msb_list.append(im_msb)
            im_mse_list.append(im_mse)

            # Human-model: this model vs human
            hm_df = df[df["model_name"].isin([m, "original"])]
            hm_icc_obj = compute_ms_components(hm_df)
            hm_msb = hm_icc_obj.msb
            hm_mse = hm_icc_obj.mse  # ANOVA MSE
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

    return per_model_variance


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
    ret = pd.concat(
        [model_rows[["text_id", "model_name", "evaluation_score"]],
         avg_rows[["text_id", "model_name", "evaluation_score"]]],
        ignore_index=True
    )
    # Filter to only the models we care about
    im_subset = df[df["model_name"].isin(model_names)]

    # Count how many models each text_id has
    text_id_model_counts = im_subset.groupby("text_id")["model_name"].nunique()

    # Find text_ids that have ALL models
    shared_text_ids = text_id_model_counts[text_id_model_counts == len(model_names)].index

    # Count of shared text_ids
    num_shared = len(shared_text_ids)

    print(f"Number of models required: {len(model_names)}")
    print(f"Model names: {model_names}")
    print(f"Number of text_ids with ALL models: {num_shared}")
    print(f"Total text_ids in df: {df['text_id'].nunique()}")
    return ret


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
    "variance_matched_msb_tc": "msb_only",        # same method; targets are bias-corrected
    "variance_matched_mse": "mse_only",
    "variance_matched_combined": "combined",
    "variance_matched_combined_tc": "combined",   # same method; targets are bias-corrected
    "variance_matched_weighted_.2": "weighted",
    "variance_matched_weighted_.5": "weighted",
    "variance_matched_weighted_.7": "weighted",
    "variance_matched_weighted_.9": "weighted",
}

# MSB weight for each "weighted" strategy (MSE weight = 1 - msb_weight).
_WEIGHTED_MSB_WEIGHTS = {
    "variance_matched_weighted_.2": 0.2,
    "variance_matched_weighted_.5": 0.5,
    "variance_matched_weighted_.7": 0.7,
    "variance_matched_weighted_.9": 0.9,
}

# Proxy oracle: uses IM scores for selection but targets true HM MSB/MSE.
# Isolates whether target misspecification is the bottleneck.
_PROXY_ORACLE_BASES = {"proxy_oracle"}

# True oracle: uses HM scores directly for both scoring and targeting.
# Upper bound — requires all human annotations at selection time.
# oracle_msb_mse      – matches combined MSB+MSE
# oracle_msb          – matches MSB only
# oracle_mse          – matches MSE only
# oracle_weighted_.X  – matches using weighted score (msb_weight=X) with HM targets
_ORACLE_BASES = {"oracle_msb_mse", "oracle_msb", "oracle_mse",
                 "oracle_weighted_.2", "oracle_weighted_.5",
                 "oracle_weighted_.7", "oracle_weighted_.9"}

_ORACLE_SCORE_METHOD = {
    "oracle_msb_mse":     "combined",
    "oracle_msb":         "msb_only",
    "oracle_mse":         "mse_only",
    "oracle_weighted_.2": "weighted",
    "oracle_weighted_.5": "weighted",
    "oracle_weighted_.7": "weighted",
    "oracle_weighted_.9": "weighted",
}

# MSB weight for oracle weighted strategies.
_ORACLE_WEIGHTED_MSB_WEIGHTS = {
    "oracle_weighted_.2": 0.2,
    "oracle_weighted_.5": 0.5,
    "oracle_weighted_.7": 0.7,
    "oracle_weighted_.9": 0.9,
}


_ORACLE_TARGET_BASES = {
    "oracle_icc":                ("icc",   "true_icc"),
    "oracle_alpha":              ("alpha", "true_alpha"),
    "oracle_mean_squared_error": ("msre",  "true_msre"),
    "oracle_rho":                ("rho",   "true_rho"),
    "oracle_tau":                ("tau",   "true_tau"),
}

# Base strategy names that apply adaptive bias correction to the MSB/MSE selection
# targets.  "_tc" is intentionally NOT stripped by _parse_strategy so it stays in the
# base name and forms its own sampling group, separate from the non-corrected variants.
_TARGET_BC_BASES = {"variance_matched_combined_tc", "variance_matched_msb_tc"}

# Maps metric-matched base strategy names to the single metric they should report errors for.
# Strategies not in this map report errors for all metrics.
_METRIC_MATCH_TARGET = {
    "metric_matched_icc":        "icc",
    "metric_matched_alpha":      "alpha",
    "metric_matched_rho":        "rho",
    "metric_matched_tau":        "tau",
    "metric_matched_mse":        "msre",
    "oracle_icc":                "icc",
    "oracle_alpha":              "alpha",
    "oracle_mean_squared_error": "msre",
    "oracle_rho":                "rho",
    "oracle_tau":                "tau",
}


def _run_trials_for_base(base_strategy, strategy_variants, text_ids, k, n_trials, text_info,
                          hm_full_df, im_full_df, im_pair_df, model, true_icc, true_alpha, true_msre,
                          true_rho=None, true_tau=None,
                          im_msb_target=None, im_mse_target=None,
                          hm_msb_target=None, hm_mse_target=None,
                          im_models=None, true_im_icc=None, true_im_alpha=None, true_im_rho=None, true_im_tau=None, true_im_msre=None,
                          past_im_msb_obs=None, past_im_mse_obs=None,
                          past_hm_msb_obs=None, past_hm_mse_obs=None,
                          prev_selected_per_trial=None,
                          online_acquisition=True,
                          fast_ms_fn=None):
    """Run trials for all strategy variants that share the same base sampling method. Accepts strategy name as well as target values to match on
    and past observations to use for bias correction of those targets (for "_tc" strategies).

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


    target_scores_for_stratified = (
        hm_full_df[hm_full_df["model_name"] == model]
        .groupby("text_id")["evaluation_score"].mean()
        if base_strategy == "stratified" else None
    )

    needs_ppi = any(imc for _, imc in [_parse_strategy(s) for s in strategy_variants])
    needs_plain = any(not imc for _, imc in [_parse_strategy(s) for s in strategy_variants]) or \
                (base_strategy == "batch_active_statistical_inf")
    needs_inf = (base_strategy == "batch_active_statistical_inf")

    results = {s: {"icc_errors": [], "alpha_errors": [], "msre_errors": [],
                   "rho_errors": [], "tau_errors": [],
                   "icc_preds": [], "alpha_preds": [], "msre_preds": [],
                   "rho_preds": [], "tau_preds": []} for s in strategy_variants}

    actual_trials = 1 if base_strategy == "max_expand" else n_trials

    # Observations accumulated within this call; returned to caller so they can
    # be threaded across successive budget levels.
    new_im_msb_obs = []
    new_im_mse_obs = []
    new_hm_msb_obs = []
    new_hm_mse_obs = []

    # Copy so we can update and return without mutating the caller's dict.
    updated_selected_per_trial = dict(prev_selected_per_trial)
    sampled_ids_list=[]
    for trial_idx in range(actual_trials):
        seed = SEED + trial_idx

        # IDs locked in from prior budget levels for this trial (online mode only).
        if online_acquisition:
            breakpoint()
            forced_ids = updated_selected_per_trial.get(trial_idx, np.array([], dtype=text_ids.dtype))
        else:
            try:
                forced_ids = np.array([], dtype=text_ids.dtype)
            except:
                forced_ids = []
                

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
                    im_msb_target + np.mean(all_hm_msb_obs) - np.mean(all_im_msb_obs)
                )
            else:
                effective_msb_target = im_msb_target

            if (im_mse_target is not None
                    and all_im_mse_obs and all_hm_mse_obs
                    and len(all_im_mse_obs) == len(all_hm_mse_obs)):
                effective_mse_target = (
                    im_mse_target + np.mean(all_hm_mse_obs) - np.mean(all_im_mse_obs)
                )
            else:
                effective_mse_target = im_mse_target
        else:
            effective_msb_target = im_msb_target
            effective_mse_target = im_mse_target

        # ── Sample once (only incremental IDs beyond forced_ids) ───────────────
        # breakpoint()
        if base_strategy == "random":
            np.random.seed(seed)
            # available = np.setdiff1d(text_ids, forced_ids)
            # n_new = min(k - len(forced_ids), len(available))
            # if n_new > 0:
            sampled_ids = np.random.choice(text_ids, size=min(k, len(text_ids)), replace=False)
            # breakpoint()
                # sampled_ids = np.concatenate([forced_ids, new_ids]) if len(forced_ids) > 0 else new_ids
            # else:
                # sampled_ids = forced_ids[:k]

        elif base_strategy == "stratified":
            sampled_ids = stratified_target_selection(
                text_ids, k, target_scores_for_stratified, seed=seed, forced_ids=forced_ids
            )
        elif base_strategy in _SCORE_METHOD_MAP:
            if COMPARISON_MODE == "pairwise_average":
                sampled_ids = pairwise_variance_matched_selection_ms(
                    text_ids, k, im_pair_df, effective_msb_target, effective_mse_target,
                    fast_ms_fn, seed=seed, n_candidates=N_CANDIDATE_SUBSETS,
                    score_method=_SCORE_METHOD_MAP[base_strategy],
                    msb_weight=_WEIGHTED_MSB_WEIGHTS.get(base_strategy, 0.5),
                    forced_ids=forced_ids, target_model=model
                )
            else:
                sampled_ids = variance_matched_selection_ms(
                    text_ids, k, im_full_df, effective_msb_target, effective_mse_target,
                    fast_ms_fn, seed=seed, n_candidates=N_CANDIDATE_SUBSETS,
                    score_method=_SCORE_METHOD_MAP[base_strategy],
                    msb_weight=_WEIGHTED_MSB_WEIGHTS.get(base_strategy, 0.5),
                    forced_ids=forced_ids
                )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_icc":
            if COMPARISON_MODE == "pairwise_average":
                sampled_ids = pairwise_metric_matched_selection(
                    text_ids, k, im_pair_df, true_im_icc, "icc",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, target_model=model,
                    compute_rho_fn=compute_spearman_rho, compute_tau_fn=compute_kendall_tau
                )
            else:
                sampled_ids = metric_matched_selection(
                    text_ids, k, im_full_df, true_im_icc, "icc",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids
                )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_alpha":
            if COMPARISON_MODE == "pairwise_average":
                sampled_ids = pairwise_metric_matched_selection(
                    text_ids, k, im_pair_df, true_im_alpha, "alpha",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, target_model=model,
                    compute_rho_fn=compute_spearman_rho, compute_tau_fn=compute_kendall_tau
                )
            else:
                sampled_ids = metric_matched_selection(
                    text_ids, k, im_full_df, true_im_alpha, "alpha",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids
                )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_rho":
            if COMPARISON_MODE == "pairwise_average":
                sampled_ids = pairwise_metric_matched_selection(
                    text_ids, k, im_pair_df, true_im_rho, "rho",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, target_model=model,
                    compute_rho_fn=compute_spearman_rho, compute_tau_fn=compute_kendall_tau
                )
            else:
                sampled_ids = metric_matched_selection(
                    text_ids, k, im_full_df, true_im_rho, "rho",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, compute_rho_fn=compute_spearman_rho
                )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_tau":
            if COMPARISON_MODE == "pairwise_average":
                sampled_ids = pairwise_metric_matched_selection(
                    text_ids, k, im_pair_df, true_im_tau, "tau",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, target_model=model,
                    compute_rho_fn=compute_spearman_rho, compute_tau_fn=compute_kendall_tau
                )
            else:
                sampled_ids = metric_matched_selection(
                    text_ids, k, im_full_df, true_im_tau, "tau",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, compute_tau_fn=compute_kendall_tau
                )
            if sampled_ids is None:
                continue
        elif base_strategy == "metric_matched_mse":
            if COMPARISON_MODE == "pairwise_average":
                sampled_ids = pairwise_metric_matched_selection(
                    text_ids, k, im_pair_df, true_im_msre, "mse",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids, target_model=model,
                    compute_rho_fn=compute_spearman_rho, compute_tau_fn=compute_kendall_tau
                )
            else:
                sampled_ids = metric_matched_selection(
                    text_ids, k, im_full_df, true_im_msre, "mse",
                    fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                    seed=seed, n_candidates=N_CANDIDATE_SUBSETS, im_models=im_models,
                    forced_ids=forced_ids
                )
            if sampled_ids is None:
                continue
        elif base_strategy == "batch_active_statistical_inf":
            # sample for either comparison mode
            sampled_ids = batch_active_statistical_inf_selection(
                    text_ids, k, im_full_df, u_fn=DEFAULT_UNCERTAINTY_FN, seed=seed, target_model=model)
        elif base_strategy == "cluster_text_based":
            sampled_ids = cluster_text_based(
                text_ids, k, im_full_df, text_info, seed=seed, target_model=model)
        elif base_strategy in _PROXY_ORACLE_BASES:
            # Proxy oracle: IM scores for selection, HM MSB/MSE as targets.
            # Isolates whether target misspecification is the bottleneck —
            # "how much better would variance matching be if you knew the
            # true HM variance targets?" while still constrained to IM scores.
            sampled_ids = variance_matched_selection_ms(
                text_ids, k, im_full_df, hm_msb_target, hm_mse_target,
                fast_ms_fn, seed=seed, n_candidates=N_CANDIDATE_SUBSETS,
                score_method="combined", forced_ids=forced_ids
            )
            if sampled_ids is None:
                continue
        elif base_strategy in _ORACLE_BASES:
            # True oracle (MSB/MSE): uses HM scores for both scoring and targeting.
            # Upper bound — requires all human annotations at selection time.
            sampled_ids = variance_matched_selection_ms(
                text_ids, k, hm_full_df, hm_msb_target, hm_mse_target,
                fast_ms_fn, seed=seed, n_candidates=N_CANDIDATE_SUBSETS,
                score_method=_ORACLE_SCORE_METHOD[base_strategy],
                msb_weight=_ORACLE_WEIGHTED_MSB_WEIGHTS.get(base_strategy, 0.5),
                forced_ids=forced_ids
            )
            if sampled_ids is None:
                continue
        elif base_strategy in _ORACLE_TARGET_BASES:
            # True oracle for target metric: uses HM scores for both scoring and targeting.
            target_metric, target_attr = _ORACLE_TARGET_BASES[base_strategy]
            target_val = {"true_icc": true_icc, "true_alpha": true_alpha,
                          "true_msre": true_msre, "true_rho": true_rho,
                          "true_tau": true_tau}[target_attr]
            sampled_ids = metric_matched_selection(
                text_ids, k, hm_full_df, target_val, target_metric,
                fast_ms_fn, compute_icc_pingouin, compute_krippendorff_alpha,
                seed=seed, n_candidates=N_CANDIDATE_SUBSETS,
                im_models=[model, "original"], forced_ids=forced_ids,
                compute_rho_fn=compute_spearman_rho,
                compute_tau_fn=compute_kendall_tau,
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
        sampled_ids_list.append(sampled_ids)
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
        plain_icc = plain_alpha = plain_rho = plain_tau = plain_msre = None
        ppi_icc = ppi_alpha = ppi_rho = ppi_tau = ppi_msre = None
        active_icc = active_alpha = active_rho = active_tau = active_msre = None

        try:
            if needs_plain:
                plain_icc = compute_icc_pingouin(hm_sample, models=[model, "original"])
                plain_alpha = compute_krippendorff_alpha(hm_sample, models=[model, "original"])
                plain_rho = compute_spearman_rho(hm_sample, models=[model, "original"])
                plain_tau = compute_kendall_tau(hm_sample, models=[model, "original"])
                plain_msre = compute_mean_sq_err(hm_sample) if len(hm_sample) > 0 else None

            if needs_ppi:
                ppi_results = compute_reliability_ppi_corrected(
                    hm_sample, im_full_df, true_im_icc, true_im_alpha, true_im_rho, true_im_tau, true_im_msre,
                    hm_models=[model, "original"], im_models=im_models
                )

                ppi_icc = ppi_results.get("icc")
                ppi_alpha = ppi_results.get("alpha")
                ppi_rho = ppi_results.get("rho")
                ppi_tau = ppi_results.get("tau")
                ppi_msre = ppi_results.get("msre")

            if needs_inf: # active statistical inference correction
                active_results = compute_reliability_active_corrected(
                    hm_sample, im_full_df, # true_im_icc, true_im_alpha, true_im_rho, true_im_tau, true_im_msre,
                    u_fn=DEFAULT_UNCERTAINTY_FN,
                    hm_models=[model, "original"], im_models=im_models
                ) 

                active_icc = active_results.get("icc")
                active_alpha = active_results.get("alpha")
                active_rho = active_results.get("rho")
                active_tau = active_results.get("tau")
                active_msre = active_results.get("msre")
        except Exception as e:
            print(e)
            continue

        # ── Record errors for each strategy variant ─────────────────────────
        for strategy in strategy_variants:
            base, imc = _parse_strategy(strategy)
            matched_metric = _METRIC_MATCH_TARGET.get(base)  # None means report all metrics

            if base == "batch_active_statistical_inf":
                # Use plain estimates for non-icc metrics
                # est_icc, est_alpha = plain_icc, plain_alpha
                est_alpha = plain_alpha
                est_msre = plain_msre
                # Rho and tau use plain estimates only (no PPI correction defined)
                est_rho = plain_rho
                est_tau = plain_tau

                # Only use active icc right now
                est_icc = active_icc
                # est_icc, est_alpha = active_icc, active_alpha
                # est_msre = active_msre
                # est_rho = active_rho
                # est_tau = active_tau
            elif imc:
                est_icc, est_alpha = ppi_icc, ppi_alpha
                est_msre = ppi_msre
                est_rho = ppi_rho
                est_tau = ppi_tau
            else:
                est_icc, est_alpha = plain_icc, plain_alpha
                est_msre = plain_msre
            # Rho and tau use plain estimates only (no PPI correction defined)
                est_rho = plain_rho
                est_tau = plain_tau

            if matched_metric in (None, "icc"):
                if est_icc is not None and np.isfinite(est_icc):
                    results[strategy]["icc_errors"].append(min(2, abs(est_icc - true_icc)))
                    results[strategy]["icc_preds"].append(est_icc)
            if matched_metric in (None, "alpha"):
                if est_alpha is not None and np.isfinite(est_alpha):
                    results[strategy]["alpha_errors"].append(min(2, abs(est_alpha - true_alpha)))
                    results[strategy]["alpha_preds"].append(est_alpha)
            if matched_metric in (None, "msre"):
                if est_msre is not None and np.isfinite(est_msre) and true_msre is not None and np.isfinite(true_msre):
                    results[strategy]["msre_errors"].append(abs(est_msre - true_msre))
                    results[strategy]["msre_preds"].append(est_msre)
            if matched_metric in (None, "rho"):
                if est_rho is not None and np.isfinite(est_rho) and true_rho is not None and np.isfinite(true_rho):
                    results[strategy]["rho_errors"].append(min(2, abs(est_rho - true_rho)))
                    results[strategy]["rho_preds"].append(est_rho)
            if matched_metric in (None, "tau"):
                if est_tau is not None and np.isfinite(est_tau) and true_tau is not None and np.isfinite(true_tau):
                    results[strategy]["tau_errors"].append(min(2, abs(est_tau - true_tau)))
                    results[strategy]["tau_preds"].append(est_tau)
    return results, new_im_msb_obs, new_im_mse_obs, new_hm_msb_obs, new_hm_mse_obs, updated_selected_per_trial, sampled_ids_list


def evaluate_reliability_estimators(df, text_info, target_models, ensemble_models, per_model_variance,
                                     budgets=range(5, 55, 5), n_trials=None,
                                     online_acquisition=False):
    """
    Evaluate reliability estimators with different sampling strategies.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score, evaluation_axis
        text_info: DataFrame with text_id, input_text, source_text
        target_models: List of target model names to evaluate independently
        ensemble_models: List of ensemble model names for variance matching / IMC
        per_model_variance: Dict from compute_variance_alignment
        budgets: Range of annotation budgets to test
        n_trials: Number of bootstrap trials (defaults to N_BOOTSTRAP_SAMPLES)
        online_acquisition: If True (default), selected IDs carry forward across budget
            levels. If False, each budget level samples k items independently (batch).

    Returns:
        results for each target reliability metric
        reliability_metadata: dict with true ICC and alpha values for each model
    """
    icc_results = []
    alpha_results = []
    msre_results = []
    rho_results = []
    tau_results = []
    reliability_metadata = {}

    if n_trials is None:
        n_trials = N_BOOTSTRAP_SAMPLES

    # Group strategies by base sampling method once — shared across all models/budgets
    strategies_by_base = {}
    for strategy in SAMPLING_STRATEGIES:
        base, _ = _parse_strategy(strategy)
        strategies_by_base.setdefault(base, []).append(strategy)
    print("STRATEGIES BY BASE", strategies_by_base)
    # Reusable fast MS function that skips expensive pivot_table validation.
    # Safe because all candidate subsets are drawn from pre-filtered shared text_ids.
    # _fast_ms = partial(compute_ms_components, validate=False)
    _fast_ms = compute_ms_components

    for model in target_models:
        im_msb_target = per_model_variance[model]["im_msb"]
        im_mse_target = per_model_variance[model]["im_mse"]
        hm_msb_target = per_model_variance[model]["hm_msb"]
        hm_mse_target = per_model_variance[model]["hm_mse"]

        hm_full_df = df[df["model_name"].isin([model, "original"])]

        # Compute true metrics
        true_icc = compute_icc_pingouin(hm_full_df, models=[model, "original"])
        print(f"\nComputing true Krippendorff's alpha for model: {model}")
        true_alpha = compute_krippendorff_alpha(hm_full_df, models=[model, "original"])
        # hm_ms_full = compute_ms_components(hm_full_df[hm_full_df["model_name"].isin([model, "original"])])
        true_msre = compute_mean_sq_err(hm_full_df[hm_full_df["model_name"].isin([model, "original"])])
        true_rho = compute_spearman_rho(hm_full_df, models=[model, "original"])
        true_tau = compute_kendall_tau(hm_full_df, models=[model, "original"])
        print(f"{model}: ICC={true_icc:.4f}, Alpha={true_alpha:.4f}, MSRE={true_msre:.4f}, Rho={true_rho:.4f}, Tau={true_tau:.4f}")

        # Build inter-model DataFrame using the target + ensemble (excluding target from its own ensemble)
        eff_ensemble = [e for e in ensemble_models if e != model]
        im_model_set = [model] + eff_ensemble
        # breakpoint()
        im_pair_df = []
        if COMPARISON_MODE == "average_pairwise":
            im_full_df = _build_im_pairwise_df(df, model, im_model_set)
            im_models = [model, "avg_other"]
            im_icc = compute_icc_pingouin(im_full_df, models=im_models)
            im_alpha = compute_krippendorff_alpha(im_full_df, models=im_models)
            im_rho = compute_spearman_rho(im_full_df, models=im_models)
            im_tau = compute_kendall_tau(im_full_df, models=im_models)
            im_msre = compute_mean_sq_err(im_full_df)
            # breakpoint()
        elif COMPARISON_MODE == "pairwise_average":
            # Compute ICC and alpha for each (target, ensemble_model) pair, then average
            pair_icc_list, pair_alpha_list, pair_rho_list, pair_tau_list, pair_msre_list = [], [], [], [], []
            for e in eff_ensemble:
                pair_df = df[df["model_name"].isin([model, e])]
                pair_icc = compute_icc_pingouin(pair_df, models=[model, e])
                pair_alpha = compute_krippendorff_alpha(pair_df, models=[model, e])
                pair_rho = compute_spearman_rho(pair_df, models=[model, e])
                pair_tau = compute_kendall_tau(pair_df, models=[model, e])
                pair_msre = compute_mean_sq_err(pair_df) if len(pair_df) > 0 else None

                if np.isfinite(pair_icc):
                    pair_icc_list.append(pair_icc)
                if np.isfinite(pair_alpha):
                    pair_alpha_list.append(pair_alpha)
                if np.isfinite(pair_rho):
                    pair_rho_list.append(pair_rho)
                if np.isfinite(pair_tau):
                    pair_tau_list.append(pair_tau)
                if pair_msre is not None and np.isfinite(pair_msre):
                    pair_msre_list.append(pair_msre)
            im_icc = np.nanmean(pair_icc_list) if pair_icc_list else np.nan
            im_alpha = np.nanmean(pair_alpha_list) if pair_alpha_list else np.nan
            im_rho = np.nanmean(pair_rho_list) if pair_rho_list else np.nan
            im_tau = np.nanmean(pair_tau_list) if pair_tau_list else np.nan
            im_msre = np.nanmean(pair_msre_list) if pair_msre_list else np.nan
            # Use average_pairwise df for selection strategies (candidate subset evaluation)
            # breakpoint()
            im_full_df = _build_im_pairwise_df(df, model, im_model_set)
            im_pair_df = df.copy()  # Keep full pairwise df for selection strategies that need it
            im_models = [model, "avg_other"]
        else:
            im_subset = df[df["model_name"].isin(im_model_set)]
            im_grouped = im_subset.groupby("text_id")["model_name"].nunique()
            shared_im_ids = im_grouped[im_grouped == len(im_model_set)].index
            im_full_df = im_subset[im_subset["text_id"].isin(shared_im_ids)]
            im_models = im_model_set
            im_icc = compute_icc_pingouin(im_full_df, models=im_models)
            im_alpha = compute_krippendorff_alpha(im_full_df, models=im_models)
            im_rho = compute_spearman_rho(im_full_df, models=im_models)
            im_tau = compute_kendall_tau(im_full_df, models=im_models)
            im_msre = compute_mean_sq_err(im_full_df) #, models=im_models)

        reliability_metadata[model] = {
            "true_hm_icc": true_icc,
            "true_hm_alpha": true_alpha,
            "true_hm_msre": true_msre,
            "true_hm_rho": true_rho,
            "true_hm_tau": true_tau,
            "im_icc": im_icc,
            "im_alpha": im_alpha,
            "im_rho": im_rho,
            "im_tau": im_tau,
            "im_msre": im_msre,
            "im_mse": im_mse_target,
        }

        text_ids = hm_full_df["text_id"].unique()
        try:
            text_ids.sort()
        except:
            text_ids = text_ids.tolist()
            text_ids.sort()

        # Outer loop over strategies so each base strategy accumulates its own
        # observation history across budget levels for bias-corrected targeting,
        # and so that each trial's selected IDs are carried forward to the next
        # budget level (cumulative selection).
        for base_strategy, strategy_variants in strategies_by_base.items():
            print(f"\nEvaluating strategies with base sampling method: {base_strategy}")
            past_im_msb_obs = []
            past_im_mse_obs = []
            past_hm_msb_obs = []
            past_hm_mse_obs = []
            prev_selected_per_trial = {}  # trial_idx -> array of IDs selected so far

            for k in budgets:
                trial_results, new_im_msb, new_im_mse, new_hm_msb, new_hm_mse, prev_selected_per_trial, sampled_ids_list = (
                    _run_trials_for_base(
                        base_strategy, strategy_variants, text_ids, k, n_trials, text_info,
                        hm_full_df, im_full_df, im_pair_df, model, true_icc, true_alpha, true_msre,
                        true_rho=true_rho, true_tau=true_tau,
                        im_msb_target=im_msb_target, im_mse_target=im_mse_target,
                        hm_msb_target=hm_msb_target, hm_mse_target=hm_mse_target,
                        im_models=im_models, true_im_icc=im_icc, true_im_alpha=im_alpha, true_im_rho=im_rho, true_im_tau=im_tau, true_im_msre=im_msre,
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
                    for i, (error, pred) in enumerate(zip(errors["icc_errors"], errors["icc_preds"])):
                        icc_results.append({"model": model, "budget": k, "method": strategy,
                                            "estimation_error": error, "predicted_icc": pred,
                                            "true_icc": true_icc, "best_ids": sampled_ids_list[i]})
                    for i, (error, pred) in enumerate(zip(errors["alpha_errors"], errors["alpha_preds"])):
                        alpha_results.append({"model": model, "budget": k, "method": strategy,
                                              "estimation_error": error, "predicted_alpha": pred,
                                              "true_alpha": true_alpha, "best_ids": sampled_ids_list[i]})
                    for i, (error, pred) in enumerate(zip(errors["msre_errors"], errors["msre_preds"])):
                        msre_results.append({"model": model, "budget": k, "method": strategy,
                                            "estimation_error": error, "predicted_msre": pred,
                                            "true_msre": true_msre, "best_ids": sampled_ids_list[i]})
                    for i, (error, pred) in enumerate(zip(errors["rho_errors"], errors["rho_preds"])):
                        rho_results.append({"model": model, "budget": k, "method": strategy,
                                            "estimation_error": error, "predicted_rho": pred,
                                            "true_rho": true_rho, "best_ids": sampled_ids_list[i]})
                    for i, (error, pred) in enumerate(zip(errors["tau_errors"], errors["tau_preds"])):
                        tau_results.append({"model": model, "budget": k, "method": strategy,
                                            "estimation_error": error, "predicted_tau": pred,
                                            "true_tau": true_tau, "best_ids": sampled_ids_list[i]})
    print("returning base strategy", base_strategy)
    return (pd.DataFrame(icc_results), pd.DataFrame(alpha_results), pd.DataFrame(msre_results),
            pd.DataFrame(rho_results), pd.DataFrame(tau_results), reliability_metadata)


# -------------------------
# PER-AXIS WORKER  (module-level so multiprocessing can pickle it)
# -------------------------
def _run_axis_worker(args):
    """Process a single evaluation axis. Runs in a worker process via multiprocessing."""
    axis, axis_df, axis_text_info = args
    axis_per_model_variance = compute_variance_alignment(
        axis_df, target_models, ensemble_models, mode=COMPARISON_MODE
    )
    axis_icc, axis_alpha, axis_msre, axis_rho, axis_tau, axis_metadata = evaluate_reliability_estimators(
        axis_df, axis_text_info, target_models, ensemble_models, axis_per_model_variance,
        budgets=range(5, MAX_BUDGET + 1, STEP_SIZE),
        online_acquisition=ONLINE_ACQUISITION
    )
    return axis, axis_icc, axis_alpha, axis_msre, axis_rho, axis_tau, axis_metadata, axis_per_model_variance


# -------------------------
# MAIN
# -------------------------
def main():
    """Main execution function."""
    print("\n" + "=" * 50)
    print("Loading data...")
    print("=" * 50)

    df, text_info = load_judge_scores(dataset, models_to_load, DATA_DIR, EVALUATION_AXES)
    text_info = retrieve_embeddings(text_info)

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
        axis_text_info = text_info[text_info["text_id"].isin(shared_text_ids)]
        print(f"Axis '{axis}': {len(axis_df)} rows after filtering to {len(shared_text_ids)} shared text_ids")
        axis_jobs.append((axis, axis_df, axis_text_info))

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
    msre_results_by_axis = {}
    rho_results_by_axis = {}
    tau_results_by_axis = {}
    reliability_metadata_by_axis = {}
    per_model_variance_by_axis = {}
    for axis, axis_icc, axis_alpha, axis_msre, axis_rho, axis_tau, axis_metadata, axis_per_model_var in axis_results:
        icc_results_by_axis[axis] = axis_icc
        alpha_results_by_axis[axis] = axis_alpha
        msre_results_by_axis[axis] = axis_msre
        rho_results_by_axis[axis] = axis_rho
        tau_results_by_axis[axis] = axis_tau
        reliability_metadata_by_axis[axis] = axis_metadata
        per_model_variance_by_axis[axis] = axis_per_model_var
        print(f"Collected {len(axis_icc)} ICC, {len(axis_alpha)} Alpha, {len(axis_msre)} MSRE, "
              f"{len(axis_rho)} Rho, {len(axis_tau)} Tau results for {axis}")

    # Combine per-axis results
    all_icc_results = []
    all_alpha_results = []
    all_msre_results = []
    all_rho_results = []
    all_tau_results = []
    for axis in EVALUATION_AXES[dataset]:
        if axis in icc_results_by_axis and len(icc_results_by_axis[axis]) > 0:
            axis_icc = icc_results_by_axis[axis].copy()
            axis_icc["axis"] = axis
            all_icc_results.append(axis_icc)
        if axis in alpha_results_by_axis and len(alpha_results_by_axis[axis]) > 0:
            axis_alpha = alpha_results_by_axis[axis].copy()
            axis_alpha["axis"] = axis
            all_alpha_results.append(axis_alpha)
        if axis in msre_results_by_axis and len(msre_results_by_axis[axis]) > 0:
            axis_msre = msre_results_by_axis[axis].copy()
            axis_msre["axis"] = axis
            all_msre_results.append(axis_msre)
        if axis in rho_results_by_axis and len(rho_results_by_axis[axis]) > 0:
            axis_rho = rho_results_by_axis[axis].copy()
            axis_rho["axis"] = axis
            all_rho_results.append(axis_rho)
        if axis in tau_results_by_axis and len(tau_results_by_axis[axis]) > 0:
            axis_tau = tau_results_by_axis[axis].copy()
            axis_tau["axis"] = axis
            all_tau_results.append(axis_tau)

    icc_results = pd.concat(all_icc_results, ignore_index=True) if all_icc_results else pd.DataFrame()
    alpha_results = pd.concat(all_alpha_results, ignore_index=True) if all_alpha_results else pd.DataFrame()
    msre_results = pd.concat(all_msre_results, ignore_index=True) if all_msre_results else pd.DataFrame()
    rho_results = pd.concat(all_rho_results, ignore_index=True) if all_rho_results else pd.DataFrame()
    tau_results = pd.concat(all_tau_results, ignore_index=True) if all_tau_results else pd.DataFrame()

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
        hm_msre_vals = [
            reliability_metadata_by_axis[ax][model]["true_hm_msre"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["true_hm_msre"])
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
        im_rho_vals = [
            reliability_metadata_by_axis[ax][model]["im_rho"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["im_rho"])
        ]
        im_tau_vals = [
            reliability_metadata_by_axis[ax][model]["im_tau"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["im_tau"])
        ]
        im_mse_vals = [
            reliability_metadata_by_axis[ax][model]["im_mse"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["im_mse"])
        ]
        hm_rho_vals = [
            reliability_metadata_by_axis[ax][model]["true_hm_rho"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["true_hm_rho"])
        ]
        hm_tau_vals = [
            reliability_metadata_by_axis[ax][model]["true_hm_tau"]
            for ax in reliability_metadata_by_axis
            if model in reliability_metadata_by_axis[ax]
            and np.isfinite(reliability_metadata_by_axis[ax][model]["true_hm_tau"])
        ]

        reliability_metadata_all[model] = {
            "true_hm_icc": np.mean(hm_icc_vals) if hm_icc_vals else np.nan,
            "true_hm_alpha": np.mean(hm_alpha_vals) if hm_alpha_vals else np.nan,
            "true_hm_msre": np.mean(hm_msre_vals) if hm_msre_vals else np.nan,
            "true_hm_rho": np.mean(hm_rho_vals) if hm_rho_vals else np.nan,
            "true_hm_tau": np.mean(hm_tau_vals) if hm_tau_vals else np.nan,
            "im_icc": np.mean(im_icc_vals) if im_icc_vals else np.nan,
            "im_alpha": np.mean(im_alpha_vals) if im_alpha_vals else np.nan,
            "im_rho": np.mean(im_rho_vals) if im_rho_vals else np.nan,
            "im_tau": np.mean(im_tau_vals) if im_tau_vals else np.nan,
            "im_mse": np.mean(im_mse_vals) if im_mse_vals else np.nan,
        }

    print(f"\n{'=' * 50}")
    print("AGGREGATE RESULTS (combined from per-axis)")
    print(f"{'=' * 50}")
    print(f"Total ICC results collected: {len(icc_results)}")
    print(f"Total Alpha results collected: {len(alpha_results)}")
    print(f"Total MSRE results collected: {len(msre_results)}")
    print(f"Total Rho results collected: {len(rho_results)}")
    print(f"Total Tau results collected: {len(tau_results)}")

    if len(icc_results) > 0:
        print(f"ICC Results by method:")
        for method in icc_results["method"].unique():
            count = len(icc_results[icc_results["method"] == method])
            print(f"  {method}: {count}")

    return (icc_results, alpha_results, msre_results, rho_results, tau_results,
            icc_results_by_axis, alpha_results_by_axis, msre_results_by_axis,
            rho_results_by_axis, tau_results_by_axis,
            reliability_metadata_all, reliability_metadata_by_axis,
            axis_jobs, per_model_variance_by_axis)


if __name__ == "__main__":
    axis_jobs_for_predictors = None
    per_model_variance_by_axis_for_predictors = None

    if args.results_dir is not None:
        print(f"\nLoading saved results from: {args.results_dir} (dataset={dataset})")
        (icc_results, alpha_results, msre_results,
         icc_results_by_axis, alpha_results_by_axis, msre_results_by_axis,
         reliability_metadata_all, reliability_metadata_by_axis,
         rho_results_by_axis, tau_results_by_axis) = load_results_dataframes(args.results_dir, dataset)
        rho_results = pd.DataFrame()
        tau_results = pd.DataFrame()
    else:
        (icc_results, alpha_results, msre_results, rho_results, tau_results,
         icc_results_by_axis, alpha_results_by_axis, msre_results_by_axis,
         rho_results_by_axis, tau_results_by_axis,
         reliability_metadata_all, reliability_metadata_by_axis,
         axis_jobs_for_predictors, per_model_variance_by_axis_for_predictors) = main()

    plot_all_results(
        icc_results, alpha_results, msre_results, rho_results, tau_results,
        icc_results_by_axis, alpha_results_by_axis, msre_results_by_axis,
        rho_results_by_axis, tau_results_by_axis,
        reliability_metadata_all, reliability_metadata_by_axis,
        dataset, PLOTS_DIR, COMPARISON_MODE
    )

    # Save predictor inputs for post-hoc scatter analysis (only available when running from scratch)
    if axis_jobs_for_predictors is not None and per_model_variance_by_axis_for_predictors is not None:
        print("\n" + "=" * 60)
        print("SAVING PREDICTOR INPUTS")
        print("=" * 60)
        save_predictor_inputs(
            PLOTS_DIR, dataset,
            axis_jobs_for_predictors,
            per_model_variance_by_axis_for_predictors,
            COMPARISON_MODE, ensemble_models,
        )