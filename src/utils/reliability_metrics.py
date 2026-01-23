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

from intraclass_corr import PointwiseICC


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
