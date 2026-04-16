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

from scipy import stats
from sklearn.metrics import mean_squared_error

from src.utils.intraclass_corr import PointwiseICC

def compute_ms_components(data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
    """
    Compute MSB and MSE components for ICC calculation.

    MSB (Mean Square Between) captures variance between subjects/texts.
    MSE (Mean Square Error) captures variance within subjects across raters.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score

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
    
    icc_obj = PointwiseICC(n=n, k=k, data=data, normalize=True, targets=targets, raters=raters, ratings=ratings)

    return icc_obj

def compute_msb(data: pd.DataFrame):
    icc_obj = compute_ms_components(data)
    if icc_obj is None:
        return np.nan
    msb = icc_obj.msb
    if msb is None:
        return np.nan
    return msb

def compute_msre(data: pd.DataFrame):
    icc_obj = compute_ms_components(data)
    if icc_obj is None:
        return np.nan
    mse = icc_obj.mse
    if mse is None:
        return np.nan
    return mse

# targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"

def compute_pearson(data: pd.DataFrame):
    model_names = data["model_name"].unique()
    n_raters = len(model_names)
    if n_raters != 2:
        print("Pearson correlation can only be computed on two raters, returning nan")
        return np.nan
    
    data_filtered = (
        data.groupby('text_id')
            .filter(lambda x: x['model_name'].nunique() == n_raters)
    )

    if len(data_filtered) == 0:
        return np.nan
    
    data_filtered = data_filtered.groupby(
            by=['text_id', 'model_name']
        )["evaluation_score"].mean().reset_index()
    
    pearson_r, pearson_p = stats.pearsonr(data_filtered.loc[data_filtered["model_name"] == model_names[0]]["evaluation_score"].values,
                                               data_filtered.loc[data_filtered["model_name"] == model_names[1]]["evaluation_score"].values)

    # print(f"\n  Correlation Results:")
    # print(f"    Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
    
    return pearson_r

def compute_spearman(data: pd.DataFrame):
    model_names = data["model_name"].unique()
    n_raters = len(model_names)
    if n_raters != 2:
        print("Spearman rank correlation can only be computed on two raters, returning nan")
        return np.nan
    
    data_filtered = (
        data.groupby('text_id')
            .filter(lambda x: x['model_name'].nunique() == n_raters)
    )

    if len(data_filtered) == 0:
        return np.nan
    
    data_filtered = data_filtered.groupby(
            by=['text_id', 'model_name']
        )["evaluation_score"].mean().reset_index()
    
    spearman_r, spearman_p = stats.spearmanr(data_filtered.loc[data_filtered["model_name"] == model_names[0]]["evaluation_score"].values,
                                               data_filtered.loc[data_filtered["model_name"] == model_names[1]]["evaluation_score"].values)
    # print(f"\n  Correlation Results:")
    # print(f"    Spearman rank r = {spearman_r:.4f} (p = {spearman_p:.6f})")

    return spearman_r

def compute_mean_sq_err(data: pd.DataFrame):
    model_names = data["model_name"].unique()
    n_raters = len(model_names)
    if n_raters != 2:
        print("Mean squared error can only be computed on two raters, returning nan")
        return np.nan

    data_filtered = (
        data.groupby('text_id')
            .filter(lambda x: x['model_name'].nunique() == n_raters)
    )

    if len(data_filtered) == 0:
        return np.nan

    data_filtered = data_filtered.groupby(
            by=['text_id', 'model_name']
        )["evaluation_score"].mean().reset_index()

    mse = mean_squared_error(y_true=data_filtered.loc[data_filtered["model_name"] == model_names[0]]["evaluation_score"].values,
                                               y_pred=data_filtered.loc[data_filtered["model_name"] == model_names[1]]["evaluation_score"].values)

    return mse


def compute_mean_sq_err_multi(data: pd.DataFrame, models=None):
    """Compute MSE for inter-rater reliability.

    For two raters, computes MSE between their score vectors.
    For k > 2 raters, computes average pairwise MSE across all model pairs.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score
        models: Optional list of model names to include. If None, uses all models in data.

    Returns:
        MSE value or np.nan if computation fails
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

    data_filtered = (
        data.groupby('text_id')
            .filter(lambda x: x['model_name'].nunique() == n_raters)
    )

    if len(data_filtered) == 0:
        return np.nan

    try:
        data_filtered = data_filtered.drop_duplicates(subset=['text_id', 'model_name'], keep='first')
        pivot = data_filtered.pivot(index='text_id', columns='model_name', values='evaluation_score')

        if n_raters == 2:
            return float(mean_squared_error(pivot.iloc[:, 0], pivot.iloc[:, 1]))

        rater_list = list(pivot.columns)
        mse_vals = []
        for i in range(len(rater_list)):
            for j in range(i + 1, len(rater_list)):
                mse_val = mean_squared_error(pivot[rater_list[i]], pivot[rater_list[j]])
                if np.isfinite(mse_val):
                    mse_vals.append(float(mse_val))
        return float(np.nanmean(mse_vals)) if mse_vals else np.nan
    except Exception:
        return np.nan

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
    data_filtered = (
        data.groupby('text_id')
            .filter(lambda x: x['model_name'].nunique() == n_raters)
    )

    if len(data_filtered) == 0:
        return np.nan

    try:
        icc_result = pg.intraclass_corr(
            data=data_filtered,
            targets='text_id',
            raters='model_name',
            ratings='evaluation_score'
        )

        icc_3k_row = icc_result.iloc[5]
        if len(icc_3k_row) > 0:
            return icc_3k_row['ICC']
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
    data_filtered = (
        data.groupby('text_id')
            .filter(lambda x: x['model_name'].nunique() == n_raters)
    )

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
        breakpoint()
        alphas = {}
        for axis in data["evaluation_axis"].unique():
            axis_data = data[data["evaluation_axis"] == axis]
            alphas[axis] = _compute_single_krippendorff_alpha(axis_data)
        return alphas
    else:
        return _compute_single_krippendorff_alpha(data)