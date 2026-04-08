"""
Reliability metrics for inter-rater agreement analysis.

Contains functions for computing:
- Mean Square components (MSB, MSE) for ICC calculation
- ICC(3,k) using pingouin
- Krippendorff's alpha for inter-rater reliability
"""

import numpy as np
import pandas as pd
import pingouin as pg
import krippendorff
from scipy import stats as scipy_stats

# from intraclass_corr import PointwiseICC
from src.utils.intraclass_corr import PointwiseICC


def compute_ms_components(data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score", validate: bool = True):
    """
    Compute MSB and MSE components for ICC calculation.

    MSB (Mean Square Between) captures variance between subjects/texts.
    MSE (Mean Square Error) captures variance within subjects across raters.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score
        validate: If True (default), validate and clean the data before computation
                  (drops text_ids with missing raters and deduplicates). Set to False
                  for candidate subsets pre-filtered to shared text_ids for a speedup.

    Returns:
        PointwiseICC object with the following attributes:
            - data: formatted DataFrame with columns: text_id, model_name, evaluation_score
            - n: number of unique targets (text_id) used in the calculations
            - k: number of unique raters (model_name) used in the calculations
            - msb_expand: Per-text contributions to MSB (Series indexed by text_id)
            - msb: Scalar MSB value
            - mse_expand: Per-text contributions to MSE (Series indexed by text_id)
            - mse: Scalar MSE value
            - icc_expand: Per-text contributions to ICC (Series indexed by text_id)
            - icc: Scalar ICC value
        Returns None if computation fails
    """
    k = data["model_name"].nunique()
    n = data["text_id"].nunique()

    if n <= 1 or k <= 1:
        return None

    icc_obj = PointwiseICC(n=n, k=k, data=data, normalize=True, validate=validate, targets=targets, raters=raters, ratings=ratings)

    return icc_obj


def compute_icc_pingouin(data, models=None):
    """
    Compute ICC(3,k) using pingouin library.

    Filters to only include text_ids that have all required raters.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score
        models: Optional list of model names to include. If None, uses all models in data.
                Can include special names like "original" or "avg_other".

    Returns:
        ICC(3,k) value or np.nan if computation fails
    """
    if len(data) == 0:
        return np.nan

    if models is not None:
        data = data[data["model_name"].isin(models)].copy()

    if len(data) == 0:
        return np.nan

    required_raters = data["model_name"].unique()
    n_raters = len(required_raters)

    if n_raters < 2:
        return np.nan

    # Filter to only include text_ids that have all required raters
    counts = data.groupby('text_id')['model_name'].nunique()
    valid_ids = counts[counts == n_raters].index
    data_filtered = data[data['text_id'].isin(valid_ids)]

    if len(data_filtered) == 0:
        return np.nan

    try:
        icc_result = pg.intraclass_corr(
            data=data_filtered,
            targets='text_id',
            raters='model_name',
            ratings='evaluation_score'
        )

        icc_3k_row = icc_result[icc_result['Type'] == 'ICC3k']
        if len(icc_3k_row) > 0:
            return icc_3k_row['ICC'].values[0]
        else:
            return np.nan
    except Exception:
        return np.nan


def _compute_single_krippendorff_alpha(data):
    """
    Compute Krippendorff's alpha for a single evaluation axis.

    Args:
        data: DataFrame with text_id, model_name, evaluation_score (single axis)

    Returns:
        Krippendorff's alpha value or np.nan if computation fails
    """
    if len(data) == 0:
        return np.nan

    required_raters = data["model_name"].unique()
    n_raters = len(required_raters)

    if n_raters < 2:
        return np.nan

    # Filter to only include text_ids that have all required raters
    counts = data.groupby('text_id')['model_name'].nunique()
    valid_ids = counts[counts == n_raters].index
    data_filtered = data[data['text_id'].isin(valid_ids)]

    if len(data_filtered) == 0:
        return np.nan

    try:
        # Drop duplicates before pivoting
        data_filtered = data_filtered.drop_duplicates(
            subset=['text_id', 'model_name'], keep='first'
        )

        # Pivot to create reliability data matrix (raters x units)
        pivot_table = data_filtered.pivot(
            index='model_name', columns='text_id', values='evaluation_score'
        )

        # Convert to numpy array for krippendorff library
        reliability_data = pivot_table.values

        # Compute Krippendorff's alpha (interval level for continuous scores)
        alpha = krippendorff.alpha(reliability_data, level_of_measurement='interval')
        return alpha
    except Exception as e:
        print(f"Krippendorff alpha computation failed: {e}")
        return np.nan


def compute_icc_bias_corrected(data, models=None, n_bootstrap=30, seed=None):
    """
    Compute ICC with bootstrap bias correction.

    Uses the standard BC estimator: corrected = 2*raw - mean(bootstrap estimates).
    Bootstraps over subjects (text_ids) with replacement.

    Args:
        data: DataFrame with text_id, model_name, evaluation_score
        models: Optional list of model names to include
        n_bootstrap: Number of bootstrap resamples (default: 30)
        seed: Random seed for reproducibility

    Returns:
        Bias-corrected ICC estimate, or raw ICC if bootstrap fails
    """
    raw_icc = compute_icc_pingouin(data, models=models)
    if not np.isfinite(raw_icc):
        return raw_icc

    if models is not None:
        data = data[data["model_name"].isin(models)].copy()

    rng = np.random.RandomState(seed)
    text_ids = data["text_id"].unique()
    n = len(text_ids)

    boot_iccs = []
    for b in range(n_bootstrap):
        boot_ids = rng.choice(text_ids, size=n, replace=True)
        boot_rows = []
        for new_id, orig_id in enumerate(boot_ids):
            rows = data[data["text_id"] == orig_id].copy()
            rows["text_id"] = f"__boot_{b}_{new_id}__"
            boot_rows.append(rows)
        boot_data = pd.concat(boot_rows, ignore_index=True)
        boot_icc = compute_icc_pingouin(boot_data, models=models)
        if np.isfinite(boot_icc):
            boot_iccs.append(boot_icc)

    if not boot_iccs:
        return raw_icc

    return 2 * raw_icc - np.mean(boot_iccs)


def compute_krippendorff_alpha_bias_corrected(data, models=None, n_bootstrap=30, seed=None):
    """
    Compute Krippendorff's alpha with bootstrap bias correction.

    Uses the standard BC estimator: corrected = 2*raw - mean(bootstrap estimates).
    
    Bootstraps over subjects (text_ids) with replacement.

    Args:
        data: DataFrame with text_id, model_name, evaluation_score
        models: Optional list of model names to include
        n_bootstrap: Number of bootstrap resamples (default: 30)
        seed: Random seed for reproducibility

    Returns:
        Bias-corrected alpha estimate, or raw alpha if bootstrap fails
    """
    raw_alpha = compute_krippendorff_alpha(data, models=models)
    if not np.isfinite(raw_alpha):
        return raw_alpha

    if models is not None:
        data = data[data["model_name"].isin(models)].copy()

    rng = np.random.RandomState(seed)
    text_ids = data["text_id"].unique()
    n = len(text_ids)

    boot_alphas = []
    for b in range(n_bootstrap):
        boot_ids = rng.choice(text_ids, size=n, replace=True)
        boot_rows = []
        for new_id, orig_id in enumerate(boot_ids):
            rows = data[data["text_id"] == orig_id].copy()
            rows["text_id"] = f"__boot_{b}_{new_id}__"
            boot_rows.append(rows)
        boot_data = pd.concat(boot_rows, ignore_index=True)
        boot_alpha = compute_krippendorff_alpha(boot_data, models=models)
        if np.isfinite(boot_alpha):
            boot_alphas.append(boot_alpha)

    if not boot_alphas:
        return raw_alpha

    return 2 * raw_alpha - np.mean(boot_alphas)


# def compute_icc_im_corrected(hm_data, im_data, true_im_icc,
#                               hm_models=None, im_models=None,
#                               n_bootstrap=30, seed=None):
#     """
#     Compute ICC with inter-model control variate correction.

#     Combines standard bootstrap bias correction with a control variate term
#     derived from the inter-model ICC:

#         corrected = 2*raw_hm - mean(boot_hm) - beta*(sample_im_icc - true_im_icc)

#     where:
#         - raw_hm_icc    = ICC on the current human-model subsample
#         - mean(boot_hm) = mean of bootstrap HM-ICC values (captures sampling bias)
#         - sample_im_icc = ICC computed on the *same* subsample items using inter-model scores
#         - true_im_icc   = ICC on the full inter-model dataset (the "past" / reference ICC)
#         - beta           = Cov(boot_hm_icc, boot_im_icc) / Var(boot_im_icc), estimated
#                           from paired bootstrap resamples

#     The intuition: if the subsample's inter-model ICC is higher (lower) than the
#     known true inter-model ICC, the same items are likely to inflate (deflate) the
#     human-model ICC estimate as well. beta quantifies that coupling, allowing us to
#     remove the correlated component of the error.

#     Args:
#         hm_data: DataFrame with human-model data (text_id, model_name, evaluation_score)
#         im_data: DataFrame with inter-model data for the *same* text_ids
#                  (text_id, model_name, evaluation_score)
#         true_im_icc: Pre-computed ICC on the full inter-model dataset (reference value)
#         hm_models: Optional list of model names to include in HM ICC computation
#         im_models: Optional list of model names to include in IM ICC computation
#         n_bootstrap: Number of bootstrap resamples (default: 30)
#         seed: Random seed for reproducibility

#     Returns:
#         Corrected ICC estimate, or raw ICC if correction cannot be applied
#     """
#     raw_hm_icc = compute_icc_pingouin(hm_data, models=hm_models)
#     if not np.isfinite(raw_hm_icc):
#         return raw_hm_icc

#     # If the reference IM ICC is unavailable, fall back to standard BC
#     if not np.isfinite(true_im_icc):
#         return compute_icc_bias_corrected(hm_data, models=hm_models,
#                                           n_bootstrap=n_bootstrap, seed=seed)

#     if hm_models is not None:
#         hm_data = hm_data[hm_data["model_name"].isin(hm_models)].copy()

#     # Restrict IM data to the same text_ids that appear in the HM sample
#     shared_ids = hm_data["text_id"].unique()
#     im_sample = im_data[im_data["text_id"].isin(shared_ids)]
#     sample_im_icc = compute_icc_pingouin(im_sample, models=im_models)

#     rng = np.random.RandomState(seed)
#     n = len(shared_ids)

#     boot_hm_iccs = []
#     boot_im_iccs = []

#     for b in range(n_bootstrap):
#         boot_ids = rng.choice(shared_ids, size=n, replace=True)
#         hm_boot_rows = []
#         im_boot_rows = []

#         for new_idx, orig_id in enumerate(boot_ids):
#             new_text_id = f"__boot_{b}_{new_idx}__"

#             hm_rows = hm_data[hm_data["text_id"] == orig_id].copy()
#             hm_rows["text_id"] = new_text_id
#             hm_boot_rows.append(hm_rows)

#             im_rows = im_sample[im_sample["text_id"] == orig_id].copy()
#             im_rows["text_id"] = new_text_id
#             im_boot_rows.append(im_rows)

#         hm_boot_data = pd.concat(hm_boot_rows, ignore_index=True)
#         im_boot_data = pd.concat(im_boot_rows, ignore_index=True)

#         boot_hm_icc = compute_icc_pingouin(hm_boot_data, models=hm_models)
#         boot_im_icc = compute_icc_pingouin(im_boot_data, models=im_models)

#         if np.isfinite(boot_hm_icc) and np.isfinite(boot_im_icc):
#             boot_hm_iccs.append(boot_hm_icc)
#             boot_im_iccs.append(boot_im_icc)

#     if not boot_hm_iccs:
#         return raw_hm_icc

#     mean_boot_hm = np.mean(boot_hm_iccs)

#     # Estimate control variate coefficient beta via bootstrap regression
#     im_correction = 0.0
#     if np.isfinite(sample_im_icc) and len(boot_im_iccs) > 1:
#         boot_hm_arr = np.array(boot_hm_iccs)
#         boot_im_arr = np.array(boot_im_iccs)
#         im_var = np.var(boot_im_arr, ddof=1)
#         if im_var > 1e-10:
#             im_hm_cov = np.cov(boot_hm_arr, boot_im_arr, ddof=1)[0, 1]
#             beta = im_hm_cov / im_var
#             im_correction = beta * (sample_im_icc - true_im_icc)

#     # Combined correction: standard BC + inter-model control variate
#     return 2 * raw_hm_icc - mean_boot_hm - im_correction


def compute_reliability_ppi_corrected(hm_data, im_data, true_im_icc, true_im_alpha,
                                       hm_models=None, im_models=None):
    """
    Compute PPI-corrected ICC and Krippendorff's alpha.

    Following the Post-Prediction Inference framework, the correction is:

        corrected = true_im + (hm_subset - im_subset)

    where:
        - true_im     = metric on the full inter-model dataset (reference)
        - hm_subset   = metric on the human-model subsample
        - im_subset   = metric on the inter-model data for the same subset items

    The intuition: the inter-model ICC on the subset tells us how biased the
    subset is relative to the full population, and we use that same bias to
    correct the human-model estimate. No bootstrapping is needed.

    Args:
        hm_data: DataFrame with human-model subsample (text_id, model_name, evaluation_score)
        im_data: Full inter-model DataFrame (all text_ids)
        true_im_icc: ICC computed on the full inter-model dataset
        true_im_alpha: Krippendorff's alpha on the full inter-model dataset
        hm_models: Optional model names to include for HM computation
        im_models: Optional model names to include for IM computation

    Returns:
        (corrected_icc, corrected_alpha): PPI-corrected estimates. Falls back
        to the raw HM estimate for each metric if the IM reference is not finite.
    """
    hm_icc = compute_icc_pingouin(hm_data, models=hm_models)
    hm_alpha = compute_krippendorff_alpha(hm_data, models=hm_models)

    # Restrict IM data to the same text_ids as the HM subsample
    subset_ids = hm_data["text_id"].unique()
    im_subset = im_data[im_data["text_id"].isin(subset_ids)]

    im_subset_icc = compute_icc_pingouin(im_subset, models=im_models)
    im_subset_alpha = compute_krippendorff_alpha(im_subset, models=im_models)

    corrected_icc = (
        true_im_icc + (hm_icc - im_subset_icc)
        if np.isfinite(hm_icc) and np.isfinite(im_subset_icc) and np.isfinite(true_im_icc)
        else hm_icc
    )
    corrected_alpha = (
        true_im_alpha + (hm_alpha - im_subset_alpha)
        if np.isfinite(hm_alpha) and np.isfinite(im_subset_alpha) and np.isfinite(true_im_alpha)
        else hm_alpha
    )

    return corrected_icc, corrected_alpha


def compute_spearman_rho(data, models=None):
    """
    Compute Spearman's rank correlation coefficient for inter-rater reliability.

    For two raters, computes the Spearman correlation between their score vectors.
    For k > 2 raters, computes the average pairwise Spearman correlation.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score
        models: Optional list of model names to include. If None, uses all models in data.

    Returns:
        Spearman's rho value or np.nan if computation fails
    """
    if len(data) == 0:
        return np.nan

    if models is not None:
        data = data[data["model_name"].isin(models)].copy()

    if len(data) == 0:
        return np.nan

    rater_names = data["model_name"].unique()
    n_raters = len(rater_names)
    if n_raters < 2:
        return np.nan

    counts = data.groupby("text_id")["model_name"].nunique()
    valid_ids = counts[counts == n_raters].index
    data_filtered = data[data["text_id"].isin(valid_ids)]
    if len(data_filtered) == 0:
        return np.nan

    try:
        data_filtered = data_filtered.drop_duplicates(subset=["text_id", "model_name"], keep="first")
        pivot = data_filtered.pivot(index="text_id", columns="model_name", values="evaluation_score")

        if n_raters == 2:
            rho, _ = scipy_stats.spearmanr(pivot.iloc[:, 0], pivot.iloc[:, 1])
            return float(rho) if np.isfinite(rho) else np.nan
        else:
            rater_list = list(pivot.columns)
            rho_vals = []
            for i in range(len(rater_list)):
                for j in range(i + 1, len(rater_list)):
                    rho, _ = scipy_stats.spearmanr(pivot[rater_list[i]], pivot[rater_list[j]])
                    if np.isfinite(rho):
                        rho_vals.append(float(rho))
            return float(np.nanmean(rho_vals)) if rho_vals else np.nan
    except Exception:
        return np.nan


def compute_kendall_tau(data, models=None):
    """
    Compute Kendall's tau-b for inter-rater reliability.

    For two raters, computes Kendall's tau between their score vectors.
    For k > 2 raters, computes the average pairwise Kendall's tau.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score
        models: Optional list of model names to include. If None, uses all models in data.

    Returns:
        Kendall's tau value or np.nan if computation fails
    """
    if len(data) == 0:
        return np.nan

    if models is not None:
        data = data[data["model_name"].isin(models)].copy()

    if len(data) == 0:
        return np.nan

    rater_names = data["model_name"].unique()
    n_raters = len(rater_names)
    if n_raters < 2:
        return np.nan

    counts = data.groupby("text_id")["model_name"].nunique()
    valid_ids = counts[counts == n_raters].index
    data_filtered = data[data["text_id"].isin(valid_ids)]
    if len(data_filtered) == 0:
        return np.nan

    try:
        data_filtered = data_filtered.drop_duplicates(subset=["text_id", "model_name"], keep="first")
        pivot = data_filtered.pivot(index="text_id", columns="model_name", values="evaluation_score")

        if n_raters == 2:
            tau, _ = scipy_stats.kendalltau(pivot.iloc[:, 0], pivot.iloc[:, 1])
            return float(tau) if np.isfinite(tau) else np.nan
        else:
            rater_list = list(pivot.columns)
            tau_vals = []
            for i in range(len(rater_list)):
                for j in range(i + 1, len(rater_list)):
                    tau, _ = scipy_stats.kendalltau(pivot[rater_list[i]], pivot[rater_list[j]])
                    if np.isfinite(tau):
                        tau_vals.append(float(tau))
            return float(np.nanmean(tau_vals)) if tau_vals else np.nan
    except Exception:
        return np.nan


def compute_krippendorff_alpha(data, models=None):
    """
    Compute Krippendorff's alpha for inter-rater reliability.

    If data contains multiple evaluation axes, computes alpha for each axis separately.

    Args:
        data: DataFrame with text_id, model_name, evaluation_score, and optionally evaluation_axis
        models: Optional list of model names to include. If None, uses all models in data.

    Returns:
        If single axis (or no axis column): float alpha value or np.nan
        If multiple axes: dict mapping axis -> alpha value
    """
    if len(data) == 0:
        return np.nan

    if models is not None:
        data = data[data["model_name"].isin(models)].copy()

    if len(data) == 0:
        return np.nan

    # Check if there are multiple evaluation axes
    if "evaluation_axis" in data.columns and data["evaluation_axis"].nunique() > 1:
        alphas = {}
        for axis in data["evaluation_axis"].unique():
            axis_data = data[data["evaluation_axis"] == axis]
            alphas[axis] = _compute_single_krippendorff_alpha(axis_data)
        return alphas
    else:
        return _compute_single_krippendorff_alpha(data)
