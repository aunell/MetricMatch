import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.utils.match_metrics import compute_mean_sq_err_multi

def random_selection(cheap_ratings, n_expensive, seed):
    """Random selection strategy."""
    np.random.seed(seed)
    return np.random.choice(len(cheap_ratings), n_expensive, replace=False)

def stratified_selection(cheap_ratings, n_expensive, seed):
    """Stratified selection by cheap rating quantiles."""
    np.random.seed(seed)
    n_items = len(cheap_ratings)
    
    # Create quantile-based strata
    quantiles = np.linspace(0, 1, n_expensive + 1)
    thresholds = np.quantile(cheap_ratings, quantiles)
    
    selected = []
    for i in range(n_expensive):
        # Find items in this stratum
        if i == 0:
            mask = cheap_ratings <= thresholds[i + 1]
        elif i == n_expensive - 1:
            mask = cheap_ratings >= thresholds[i]
        else:
            mask = (cheap_ratings >= thresholds[i]) & (cheap_ratings <= thresholds[i + 1])
        
        candidates = np.where(mask)[0]
        if len(candidates) > 0:
            selected.append(np.random.choice(candidates))
        else:
            # Fallback to random selection if stratum is empty
            remaining = [j for j in range(n_items) if j not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))
    
    return np.array(selected[:n_expensive])

def QBC_selection(cheap_ratings, cheap_ratings_2, n_expensive, seed):
    """Select items where cheap raters disagree most."""
    np.random.seed(seed)
    disagreement = np.abs(cheap_ratings - cheap_ratings_2)
    return np.argsort(disagreement)[-n_expensive:]

def hybrid_selection(cheap_ratings, cheap_ratings_2, n_expensive, seed):
    """Hybrid: stratified + disagreement."""
    # Select half using stratification
    np.random.seed(seed)
    n_strat = n_expensive // 2
    strat_indices = stratified_selection(cheap_ratings, n_strat, seed)
    
    # Select remaining using disagreement (excluding already selected)
    disagreement = np.abs(cheap_ratings - cheap_ratings_2)
    disagreement = disagreement.copy()  # Make a copy to avoid modifying original
    disagreement[strat_indices] = -1  # Use -1 instead of -inf to avoid overflow
    disagree_indices = np.argsort(disagreement)[-(n_expensive - n_strat):]
    return np.concatenate([strat_indices, disagree_indices])

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


def cluster_selection(cheap_ratings, n_expensive, seed, random_state=None, epsilon=0.1, k=10):
    """
    Cluster-based selection: Use K-means clustering on cheap ratings and
    select representative items from each cluster to ensure diverse coverage.

    If clusters don't provide full coverage, remaining items are selected
    using variance-matching random selection.

    Args:
        cheap_ratings: Array of cheap ratings
        n_expensive: Number of items to select
        seed: Random seed for reproducibility
        random_state: Random state for KMeans
        epsilon: Maximum allowed relative difference in variance (default: 0.1)
        k: Maximum number of redraws for variance matching (default: 10)
    """
    np.random.seed(seed)
    n_items = len(cheap_ratings)

    # Reshape for sklearn (needs 2D array)
    ratings_2d = cheap_ratings.reshape(-1, 1)

    # Use K-means with k=5 clusters
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(ratings_2d)

    selected = []

    # Step 1: Select closest-to-center item from each cluster
    cluster_sizes = []
    for cluster_id in range(n_clusters):
        cluster_items = np.where(cluster_labels == cluster_id)[0]
        cluster_sizes.append(len(cluster_items))

        if len(cluster_items) > 0:
            # Select item closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id][0]
            distances = np.abs(cheap_ratings[cluster_items] - cluster_center)
            closest_idx = cluster_items[np.argmin(distances)]
            selected.append(closest_idx)

    # Step 2: Sample remaining items proportionally to cluster size
    remaining_needed = n_expensive - len(selected)
    if remaining_needed > 0:
        # Calculate proportions based on cluster sizes
        cluster_sizes = np.array(cluster_sizes)
        cluster_proportions = cluster_sizes / cluster_sizes.sum()

        # Allocate remaining items to clusters proportionally
        samples_per_cluster = np.round(cluster_proportions * remaining_needed).astype(int)

        # Adjust to ensure we get exactly remaining_needed items
        while samples_per_cluster.sum() < remaining_needed:
            # Add to largest cluster that still has available items
            for idx in np.argsort(cluster_proportions)[::-1]:
                cluster_items = np.where(cluster_labels == idx)[0]
                available = len([i for i in cluster_items if i not in selected])
                if samples_per_cluster[idx] < available:
                    samples_per_cluster[idx] += 1
                    break

        while samples_per_cluster.sum() > remaining_needed:
            # Remove from cluster with most samples
            idx = np.argmax(samples_per_cluster)
            samples_per_cluster[idx] -= 1

        # Sample from each cluster
        for cluster_id in range(n_clusters):
            n_samples = samples_per_cluster[cluster_id]
            if n_samples > 0:
                cluster_items = np.array([i for i in np.where(cluster_labels == cluster_id)[0]
                                        if i not in selected])
                if len(cluster_items) >= n_samples:
                    sampled = np.random.choice(cluster_items, n_samples, replace=False)
                    selected.extend(sampled)
                elif len(cluster_items) > 0:
                    # Take all available if we need more than available
                    selected.extend(cluster_items)

    # If we still need more items (clusters didn't provide full coverage)
    if len(selected) < n_expensive:
        print(f"Cluster selection incomplete: {len(selected)}/{n_expensive} items selected from clusters")
        remaining = [j for j in range(n_items) if j not in selected]
        n_needed = n_expensive - len(selected)
        print(f"Need {n_needed} more items, {len(remaining)} remaining candidates")

        if len(remaining) >= n_needed:
            # Use variance matching for remaining selections
            target_variance = np.var(cheap_ratings)
            print(f"Target variance (full cheap_ratings): {target_variance:.6f}")
            best_additional = None
            best_variance_diff = float('inf')

            # Try k rollouts of random selection for the additional items
            for attempt in range(k):
                additional = list(np.random.choice(remaining, n_needed, replace=False))
                full_selection = selected + additional
                subset_variance = np.var(cheap_ratings[full_selection])

                # Check variance difference
                if target_variance == 0:
                    variance_diff = abs(subset_variance)
                else:
                    variance_diff = abs(subset_variance - target_variance) / target_variance

                if variance_diff < best_variance_diff:
                    best_variance_diff = variance_diff
                    best_additional = additional
                    print(f"  Attempt {attempt + 1}/{k}: variance={subset_variance:.6f}, rel_diff={variance_diff:.6f} (best so far)")
                else:
                    print(f"  Attempt {attempt + 1}/{k}: variance={subset_variance:.6f}, rel_diff={variance_diff:.6f}")

                # If we found a good match, stop early
                if variance_diff <= epsilon:
                    print(f"Found match within epsilon ({epsilon}), stopping early")
                    break

            # Add the best additional items found
            if best_additional is not None:
                final_variance = np.var(cheap_ratings[selected + best_additional])
                print(f"Selected best rollout: final_variance={final_variance:.6f}, rel_diff={best_variance_diff:.6f}")
                selected.extend(best_additional)
            else:
                # Ultimate fallback
                print("WARNING: No valid selection found, using ultimate fallback")
                selected.extend(np.random.choice(remaining, n_needed, replace=False))

            # # ALTERNATIVE: Simple random selection (no variance matching)
            # additional = list(np.random.choice(remaining, n_needed, replace=False))
            # selected.extend(additional)
            # print(f"Randomly selected {n_needed} additional items")
        else:
            # Not enough remaining items, just add what's left
            print(f"WARNING: Not enough remaining items ({len(remaining)} < {n_needed}), using all remaining")
            selected.extend(remaining[:n_needed])

    return np.array(selected[:n_expensive])

def maximum_variation_selection(cheap_ratings, n_expensive, seed):
    """
    Variance-weighted selection: Select items that maximize the variance
    of the selected subset while maintaining representativeness.
    This helps preserve the between-item variance component crucial for ICC.
    """
    np.random.seed(seed)
    selected = []
    
    # Start with the item having median rating to anchor the selection
    median_idx = np.argmin(np.abs(cheap_ratings - np.median(cheap_ratings)))
    selected.append(median_idx)
    
    # Iteratively add items that maximize subset variance
    for step in range(1, n_expensive):
        best_item = None
        best_variance = -1
        
        candidates = [i for i in range(len(cheap_ratings)) if i not in selected]
        
        for candidate in candidates:
            temp_selected = selected + [candidate]
            temp_ratings = cheap_ratings[temp_selected]
            temp_variance = np.var(temp_ratings)
            
            # Bonus for maintaining good distribution coverage
            temp_range = np.max(temp_ratings) - np.min(temp_ratings)
            full_range = np.max(cheap_ratings) - np.min(cheap_ratings)
            coverage_bonus = (temp_range / full_range) * 0.1 if full_range > 0 else 0
            
            score = temp_variance # + coverage_bonus
            
            if score > best_variance:
                best_variance = score
                best_item = candidate
        
        if best_item is not None:
            selected.append(best_item)
    
    return np.array(selected)

def density_based_selection(cheap_ratings, n_expensive, seed):
    """
    Density-based selection: Select items from both high-density regions
    (where many items cluster) and low-density regions (outliers/extremes).
    This balances representation of typical cases with edge cases.
    """
    from scipy.stats import gaussian_kde
    np.random.seed(seed)

    # Create KDE of cheap ratings
    kde = gaussian_kde(cheap_ratings)
    densities = kde(cheap_ratings)

    # Normalize densities
    densities = (densities - densities.min()) / (densities.max() - densities.min())

    # Select mix of high-density and low-density items
    n_high_density = n_expensive // 2
    n_low_density = n_expensive - n_high_density

    # Select high-density items (representative of typical cases)
    high_density_indices = np.argsort(densities)[-n_high_density*2:]  # Get top candidates
    high_density_selected = np.random.choice(high_density_indices, n_high_density, replace=False)

    # Select low-density items (outliers/edge cases), excluding already selected
    remaining_indices = [i for i in range(len(cheap_ratings)) if i not in high_density_selected]
    remaining_densities = densities[remaining_indices]

    if len(remaining_indices) >= n_low_density:
        low_density_candidates = np.argsort(remaining_densities)[:n_low_density*2]  # Bottom candidates
        if len(low_density_candidates) >= n_low_density:
            low_density_selected_rel = np.random.choice(low_density_candidates, n_low_density, replace=False)
            low_density_selected = [remaining_indices[i] for i in low_density_selected_rel]
        else:
            low_density_selected = remaining_indices[:n_low_density]
    else:
        low_density_selected = remaining_indices

    # Combine selections
    selected = np.concatenate([high_density_selected, low_density_selected])
    return selected[:n_expensive]

def variance_matching(cheap_ratings, n_expensive, seed, epsilon=0.1, k=10):
    """
    Variance-matching selection: Random selection with variance constraint.

    Randomly selects items, but ensures the variance of the subset falls within
    epsilon of the variance of the full cheap_ratings set. If the variance
    constraint is not met, redraws the sample up to k times.

    Args:
        cheap_ratings: Array of cheap ratings
        n_expensive: Number of items to select
        seed: Random seed for reproducibility
        epsilon: Maximum allowed relative difference in variance (default: 0.1)
        k: Maximum number of redraws (default: 100)

    Returns:
        Array of selected indices
    """
    np.random.seed(seed)

    # Calculate target variance
    target_variance = np.var(cheap_ratings)

    # Try up to k times to find a sample with matching variance
    curr_selected=[]
    curr_best = 1
    for attempt in range(k):
        # Random selection
        selected = np.random.choice(len(cheap_ratings), n_expensive, replace=False)
        subset_variance = np.var(cheap_ratings[selected])

        # Check if variance falls within epsilon of target
        if target_variance == 0:
            # If target variance is 0, accept only if subset variance is also 0
            if subset_variance == 0:
                return selected
        else:
            # Check relative difference
            relative_diff = abs(subset_variance - target_variance) / target_variance
            print("relative diff", relative_diff)
            if relative_diff <= epsilon:
                return selected
            elif relative_diff<curr_best:
                curr_selected=selected

    # If no valid sample found after k attempts, return the last attempt
    print("VARIANCE MATCHING DIDNT WORK")
    return curr_selected


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

def dev_metric_matched_selection(text_ids, k, im_full_df, target_metric_values, compute_metric_fns, alpha_weight, seed=42, n_candidates=20):
    """
    Select subset that best matches target inter-model metric.

    Tries multiple random subsets and picks the one with metric value closest to target.
    Used for reliability estimation experiments where we want to match the metric
    structure of inter-model comparisons.

    Args:
        text_ids: Array of text IDs to sample from
        k: Number of items to select
        im_full_df: DataFrame with inter-model data (text_id, model_name, evaluation_score)
        target_metric_values: Array of target metric values
        compute_metric_fns: Function to compute metric value given a subset of data (must be same size as target metric values)
        alpha_weight: weighting of metrics to match to target (must be same size as target metric values and sum to 1, will be normalized to sum to 1 otherwise)
        seed: Random seed for reproducibility
        n_candidates: Number of candidate subsets to try (default: 20)

    Returns:
        Array of selected text_ids, or None if no valid subset found
    """

    if isinstance(target_metric_values, list):
        assert isinstance(compute_metric_fns, list)
        assert isinstance(alpha_weight, list)
        assert len(target_metric_values) == len(compute_metric_fns) and len(target_metric_values) == len(alpha_weight), \
        "Each metric to match must have an associated function and relative weight."
    else:
        target_metric_values = [target_metric_values]
        compute_metric_fns = [compute_metric_fns]
        alpha_weight = [alpha_weight]
    
    if np.abs(1 - np.sum(alpha_weight)) > 1e-3:
        print("WARNING: Given alpha weight does not sum to 1, will be re-normalized.")
        alpha_weight = alpha_weight / np.sum(alpha_weight)
    
    rng = np.random.RandomState(seed)
    best_ids = None
    best_score = float('inf')

    for _ in range(n_candidates):
        candidate_ids = rng.choice(text_ids, size=min(k, len(text_ids)), replace=False)
        im_candidate = im_full_df[im_full_df["text_id"].isin(candidate_ids)]

        if len(im_candidate) == 0:
            continue
        
        cand_metrics = []
        for compute_metric_fn in compute_metric_fns:
            cand_metric = compute_metric_fn(im_candidate)
            cand_metrics.append(cand_metric)

        if not (np.isfinite(np.all(cand_metrics)) and np.isfinite(np.all(cand_metrics))):
            continue

        score = np.array(alpha_weight).T @ np.abs(np.array(target_metric_values) - np.array(cand_metrics))

        if score < best_score:
            best_score = score
            best_ids = candidate_ids

    return best_ids


def metric_matched_selection(text_ids, k, im_full_df, target_value, target_metric,
                              compute_ms_fn, compute_icc_fn, compute_alpha_fn,
                              seed=42, n_candidates=20, im_models=None,
                              forced_ids=None,
                              compute_rho_fn=None, compute_tau_fn=None):
    """
    Select subset whose inter-model metric best matches a target value.

    Analogous to variance_matched_selection_ms but matches on a scalar reliability
    metric (ICC, Krippendorff's alpha, MSE, Spearman rho, or Kendall tau) rather than
    MSB/MSE components.

    Args:
        text_ids: Array of text IDs to sample from
        k: Number of items to select
        im_full_df: DataFrame with inter-model data (text_id, model_name, evaluation_score)
        target_value: Target metric value to match (e.g. full-dataset IM ICC)
        target_metric: Which metric to match — "icc", "alpha", "mse", "rho", or "tau"
        compute_ms_fn: Function that returns a PointwiseICC object (for MSE)
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

    Returns:
        Array of selected text_ids, or None if no valid subset found
    """
    if not np.isfinite(target_value):
        return None

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

        if target_metric == "icc":
            # Use MS components instead of pingouin to avoid ANOVA overhead.
            # ICC3k = (MSB - MSE) / MSB, identical to compute_icc_pingouin's ICC3k formula.
            ms_obj = compute_ms_fn(im_candidate)
            cand_value = ms_obj.icc if (ms_obj is not None and ms_obj.icc is not None) else np.nan
        elif target_metric == "alpha":
            cand_value = compute_alpha_fn(im_candidate, models=im_models)
        elif target_metric == "mse":
            cand_value = compute_mean_sq_err_multi(im_candidate, models=im_models)
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