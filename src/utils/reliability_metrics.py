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


def compute_ms_components(data):
    """
    Compute MSB and MSE components for ICC calculation.

    MSB (Mean Square Between) captures variance between subjects/texts.
    MSE (Mean Square Error) captures variance within subjects across raters.

    Args:
        data: DataFrame with columns: text_id, model_name, evaluation_score

    Returns:
        Tuple of (msb_expand, msb, mse_expand, mse) where:
            - msb_expand: Per-text contributions to MSB (Series indexed by text_id)
            - msb: Scalar MSB value
            - mse_expand: Per-text contributions to MSE (Series indexed by text_id)
            - mse: Scalar MSE value
        Returns (np.nan, np.nan, np.nan, np.nan) if computation fails
    """
    k = data["model_name"].nunique()
    n = data["text_id"].nunique()

    if n <= 1 or k <= 1:
        return np.nan, np.nan, np.nan, np.nan

    grouped_by_text = data.groupby("text_id")["evaluation_score"]
    grouped_by_model = data.groupby("model_name")["evaluation_score"]

    s = grouped_by_text.mean()
    m = grouped_by_model.mean()
    xbar = data["evaluation_score"].mean()

    # For each text i, (S_i - x_tot)^2
    msb_expand = (s - xbar) ** 2
    msb = (k / (n - 1)) * msb_expand.sum()

    # For each text i, (1 / k) sum_j=1^k (x_ij - M_j)^2
    mse_partial_expand = data.groupby("text_id")[["model_name", "evaluation_score"]].apply(
        lambda x: np.mean((x.set_index("model_name").squeeze() - m) ** 2)
    )

    mse = (1 / ((n - 1) * (k - 1))) * (mse_partial_expand.sum() - (k * msb_expand.sum()))

    return msb_expand, msb, mse_partial_expand, mse


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
