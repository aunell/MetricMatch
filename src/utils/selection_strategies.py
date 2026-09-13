import numpy as np
import pandas as pd
from src.utils.match_metrics import compute_mean_sq_err

def random_selection(cheap_ratings, n_expensive, seed):
    """Random selection strategy."""
    np.random.seed(seed)
    return np.random.choice(len(cheap_ratings), n_expensive, replace=False)

def stratified_target_selection(text_ids, k, target_scores, seed=42, n_strata=4, forced_ids=None):
    """Select k items stratified by target model score quantiles.

    Splits the target model's scores into n_strata equal-probability quantile bins
    and selects k // n_strata items from each bin (remainder distributed to first bins).
    Uses only the target model's own scores — no ensemble involved.

    Args:
        text_ids: Array of text IDs to sample from
        k: Number of items to select
        target_scores: pandas Series indexed by text_id with the target model's scores
        seed: Random seed
        n_strata: Number of quantile bins (default 4)
        forced_ids: IDs that must be included (online acquisition mode)

    Returns:
        Array of selected text_ids
    """
    rng = np.random.RandomState(seed)

    forced_ids = np.asarray(forced_ids) if forced_ids is not None and len(forced_ids) > 0 else np.array([], dtype=text_ids.dtype)
    available_ids = np.setdiff1d(text_ids, forced_ids)
    n_new = min(k - len(forced_ids), len(available_ids))

    if n_new <= 0:
        return forced_ids[:k]

    scores = target_scores.reindex(available_ids)
    valid_mask = scores.notna().values
    valid_ids = available_ids[valid_mask]
    valid_scores = scores.values[valid_mask]

    if len(valid_ids) == 0:
        chosen = rng.choice(available_ids, size=n_new, replace=False)
        return np.concatenate([forced_ids, chosen]) if len(forced_ids) > 0 else chosen

    quantile_edges = np.quantile(valid_scores, np.linspace(0, 1, n_strata + 1))
    bin_indices = np.digitize(valid_scores, quantile_edges[1:-1])  # 0-indexed 0..n_strata-1

    per_stratum = n_new // n_strata
    remainder = n_new % n_strata

    selected = []
    for stratum in range(n_strata):
        stratum_ids = valid_ids[bin_indices == stratum]
        n_select = min(per_stratum + (1 if stratum < remainder else 0), len(stratum_ids))
        if n_select > 0:
            selected.extend(rng.choice(stratum_ids, size=n_select, replace=False).tolist())

    selected = np.array(selected)

    if len(selected) < n_new:
        used = set(selected.tolist())
        remaining = np.array([tid for tid in available_ids if tid not in used])
        shortfall = n_new - len(selected)
        extra = rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False)
        selected = np.concatenate([selected, extra])

    return np.concatenate([forced_ids, selected]) if len(forced_ids) > 0 else selected

def variance_matched_selection_ms(text_ids, k, im_full_df, im_msb_target, im_mse_target,
                                   compute_ms_fn, seed=42, n_candidates=20, score_method="combined",
                                   msb_weight=0.5, forced_ids=None):
    """
    Select subset that best matches target inter-model variance using MS components.

    Tries multiple random subsets and picks the one with MSB/MSE closest to target.
    Used for reliability estimation experiments where we want to match the variance
    structure of inter-model comparisons.

    Args:
        text_ids: Array of text IDs to sample from
        k: Number of items to select
        im_full_df: DataFrame with inter-model data (text_id, model_name, evaluation_score)
        im_msb_target: Target mean square between (MSB) value
        im_mse_target: Target mean square error (MSE) value
        compute_ms_fn: Function to compute MS components (returns PointwiseICC object)
        seed: Random seed for reproducibility
        n_candidates: Number of candidate subsets to try (default: 20)
        score_method: Method for computing score. Options:
            - "msb_only": score = abs(cand_msb - im_msb_target)
            - "mse_only": score = abs(cand_mse - im_mse_target)
            - "combined": score = abs(cand_msb - im_msb_target) + abs(cand_mse - im_mse_target)
            - "weighted": score = msb_weight * |Δmsb|/msb_target + (1-msb_weight) * |Δmse|/mse_target
                          Normalizes each component by its target to account for scale differences
                          between MSB (larger) and MSE (smaller). Use msb_weight to control emphasis.
        msb_weight: Weight on the MSB term for score_method="weighted" (default: 0.5 = equal).
                    The MSE term receives weight (1 - msb_weight). Ignored for other score methods.
        forced_ids: IDs that must be included in the selection (ONLINE_ACQUISITION mode only —
                    these are IDs already annotated at a prior budget level). Only the
                    incremental IDs needed to reach k are sampled from the remaining pool.
                    The score is computed on the full combined set (forced + new).
                    Pass None or an empty array for batch/independent selection.

    Returns:
        Array of selected text_ids, or None if no valid subset found
    """
    rng = np.random.RandomState(seed)
    best_ids = None
    best_score = float('inf')

    forced_ids = np.asarray(forced_ids) if forced_ids is not None and len(forced_ids) > 0 else np.array([], dtype=text_ids.dtype)
    available_ids = np.setdiff1d(text_ids, forced_ids)
    n_new = min(k - len(forced_ids), len(available_ids))

    if n_new <= 0:
        return forced_ids[:k]

    for _ in range(n_candidates):
        new_ids = rng.choice(available_ids, size=n_new, replace=False)
        candidate_ids = np.concatenate([forced_ids, new_ids]) if len(forced_ids) > 0 else new_ids
        im_candidate = im_full_df[im_full_df["text_id"].isin(candidate_ids)]

        if len(im_candidate) == 0:
            continue

        cand_obj = compute_ms_fn(im_candidate)
        if cand_obj is None or cand_obj.msb is None or cand_obj.mse is None:
            continue
        cand_msb, cand_mse = cand_obj.msb, cand_obj.mse

        if not (np.isfinite(cand_msb) and np.isfinite(cand_mse)):
            continue

        # Compute score based on selected method
        if score_method == "msb_only":
            score = abs(cand_msb - im_msb_target)
        elif score_method == "mse_only":
            score = abs(cand_mse - im_mse_target)
        elif score_method == "weighted":
            dmsb = (abs(cand_msb - im_msb_target) / im_msb_target
                    if im_msb_target != 0 else abs(cand_msb - im_msb_target))
            dmse = (abs(cand_mse - im_mse_target) / im_mse_target
                    if im_mse_target != 0 else abs(cand_mse - im_mse_target))
            score = msb_weight * dmsb + (1.0 - msb_weight) * dmse
        else:  # "combined" (default)
            score = abs(cand_msb - im_msb_target) + abs(cand_mse - im_mse_target)

        if score < best_score:
            best_score = score
            best_ids = candidate_ids

    return best_ids


def metric_matched_selection(text_ids, k, im_full_df, target_value, target_metric,
                              compute_ms_fn, compute_icc_fn, compute_alpha_fn,
                              seed=42, n_candidates=20, im_models=None,
                              forced_ids=None,
                              compute_rho_fn=None, compute_tau_fn=None,
                              target_model=None):
    """
    Select subset whose inter-model metric best matches a target value.

    Analogous to variance_matched_selection_ms but matches on a scalar reliability
    metric (ICC, Krippendorff's alpha, mean_squared_error, Spearman rho, or Kendall tau)
    rather than MSB/MSE components.

    Args:
        text_ids: Array of text IDs to sample from
        k: Number of items to select
        im_full_df: DataFrame with inter-model data (text_id, model_name, evaluation_score)
        target_value: Target metric value to match (e.g. full-dataset IM ICC)
        target_metric: Which metric to match — "icc", "alpha", "mean_squared_error",
                       "rho", or "tau". Use "mean_squared_error" for average pairwise
                       sklearn mean squared error between raters.
        compute_ms_fn: Function that returns a PointwiseICC object (for ANOVA mse/icc)
        compute_icc_fn: Function to compute ICC given a DataFrame and models kwarg
        compute_alpha_fn: Function to compute Krippendorff's alpha given a DataFrame and models kwarg
        seed: Base random seed
        n_candidates: Number of candidate subsets to evaluate
        im_models: Model names to pass to the metric functions
        forced_ids: IDs that must be included in the selection (ONLINE_ACQUISITION mode only —
                    these are IDs already annotated at a prior budget level). Only the
                    incremental IDs needed to reach k are sampled from the remaining pool.
                    The score is computed on the full combined set (forced + new).
                    Pass None or an empty array for batch/independent selection.
        target_model: Required when target_metric == "mean_squared_error". The model whose
                      scores are compared pairwise against all other models in the candidate subset.

    Returns:
        Array of selected text_ids, or None if no valid subset found
    """
    if not np.isfinite(target_value):
        return None

    rng = np.random.RandomState(seed)
    best_ids = None
    best_score = float('inf')

    text_ids.sort()
    for _ in range(n_candidates):
        candidate_ids = rng.choice(text_ids, size=min(k, len(text_ids)), replace=False)
        im_candidate = im_full_df[im_full_df["text_id"].isin(candidate_ids)]
        if len(im_candidate) == 0:
            continue

        if target_metric == "icc":
            cand_value = compute_icc_fn(im_candidate, models=im_models) 
        elif target_metric == "alpha":
            cand_value = compute_alpha_fn(im_candidate, models=im_models)
        elif target_metric == "mse":
            cand_value = compute_mean_sq_err(im_candidate) #, models=im_models)
        elif target_metric == "rho":
            if compute_rho_fn is None:
                continue
            cand_value = compute_rho_fn(im_candidate, models=im_models)
        elif target_metric == "tau":
            if compute_tau_fn is None:
                continue
            cand_value = compute_tau_fn(im_candidate, models=im_models)
        else:
            continue

        if not np.isfinite(cand_value):
            continue

        score = abs(cand_value - target_value)
        if score < best_score:
            best_score = score
            best_ids = candidate_ids
    return best_ids

def max_expand_selection(im_full_df, k, compute_ms_fn, alpha_weight=0.5, forced_ids=None):
    """
    Select most informative points based on MS component contributions.

    Ranks text_ids by their combined contribution to MSB (between-subject variance)
    and MSE (within-subject variance), then selects the top k items. This method
    prioritizes items that contribute most to the variance structure.

    Args:
        im_full_df: DataFrame with inter-model data (text_id, model_name, evaluation_score)
        k: Number of items to select
        compute_ms_fn: Function to compute MS components (returns Pointwise ICC object)
        alpha_weight: Weight for MSB contribution vs MSE (default: 0.5 for equal weighting)
                      Higher values prioritize between-subject variance (more informative for ICC)
        forced_ids: IDs that must be included in the selection (ONLINE_ACQUISITION mode only —
                    these are IDs already annotated at a prior budget level). The remaining
                    slots are filled from the top of the score-ranked list, excluding forced_ids.
                    Pass None or an empty array for batch/independent selection.

    Returns:
        List of selected text_ids, or None if computation fails
    """
    if len(im_full_df) == 0:
        return None

    try:
        im_obj = compute_ms_fn(im_full_df)
        im_msb_expand, im_mse_expand, icc_expand = im_obj.msb_expand, im_obj.mse_expand, im_obj.icc_expand

        if not isinstance(im_msb_expand, pd.Series) or len(im_msb_expand) == 0:
            return None

        # Linear combination of MSB and MSE contributions
        combined_score = alpha_weight * im_msb_expand + (1 - alpha_weight) * im_mse_expand
        sorted_text_ids = combined_score.sort_values(ascending=False).index.tolist()

        if forced_ids is not None and len(forced_ids) > 0:
            forced_set = set(forced_ids)
            n_new = k - len(forced_ids)
            additional = [tid for tid in sorted_text_ids if tid not in forced_set][:n_new]
            return list(forced_ids) + additional

        return sorted_text_ids[:min(k, len(sorted_text_ids))]
    except Exception:
        return None


def batch_active_statistical_inf_selection(text_ids, k, im_full_df, u_fn, seed, target_model, tau=0.5):
    """
    Select subset via batch active statistical inference sampling method as in Zrnic & Candes (2024).
    https://arxiv.org/pdf/2403.03208

    Args:
        text_ids: Array of text IDs to sample from
        k: Number of items to select
        im_full_df: DataFrame with inter-model data (text_id, model_name, evaluation_score)
        u_fn: Uncertainty function that computes the uncertainty on each text id ==> returns 
        seed: Base random seed
        target_model: The model whose scores are used from im_full_df.
        tau: the scaling parameter for the uniform function vs. the determined sampling distribution

    Returns:
        Array of selected text_ids, or None if no valid subset found
    """
    rng = np.random.RandomState(seed)
    selected_ids = []

    # Sort text ids and get uncertainty measure values
    try:
        text_ids.sort()
    except:
        text_ids = text_ids.tolist()
        text_ids.sort()
    n = len(text_ids)
    u_values = u_fn(text_ids, im_full_df, target_model) # numpy array of size len(im_full_df)

    # mixture components for sampling strategy
    sample = (k / (n * np.mean(u_values))) * u_values
    unif = k / n
    mixture_probs = (1 - tau) * sample + (tau * unif)

    # sample from each bernoulli by determined mixture probability
    for i in range(len(text_ids)):
        try:
            draw = rng.binomial(1, mixture_probs[i])
        except:
            draw = 0
        if draw:
            selected_ids.append(text_ids[i])
    
    if not len(selected_ids):
        selected_ids = rng.choice(text_ids, k, replace=False)
        # selected_ids = None
    elif len(selected_ids) < k:
        unselected_ids = [id for id in text_ids if id not in selected_ids]
        new_selected_ids = rng.choice(unselected_ids, k-len(selected_ids), replace=False)
        selected_ids.extend(new_selected_ids)
    elif len(selected_ids) > k:
        selected_ids = rng.choice(selected_ids, k, replace=False)

    return selected_ids