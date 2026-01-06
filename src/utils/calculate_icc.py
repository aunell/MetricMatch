import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score


def calculate_krip(rater1, rater2, metric='interval'):
    """
    Compute Krippendorff's alpha reliability coefficient for two raters.

    Parameters:
    rater1, rater2 : array-like
        Ratings from two raters. Must be same length.
        Use np.nan for missing values.
    metric : str
        'nominal', 'ordinal', or 'interval'.

    Returns:
    float: Krippendorff's alpha
    """
    # Stack raters into matrix (n_raters x n_items)
    data = np.vstack([rater1, rater2]).astype(float)

    # Handle missing data
    mask = ~np.isnan(data)
    valid_counts = mask.sum(axis=0)
    valid_items = valid_counts > 1
    data = data[:, valid_items]
    mask = mask[:, valid_items]

    # Define distance metric
    def delta(x, y):
        if metric == 'nominal':
            return 0 if x == y else 1
        elif metric == 'ordinal':
            return ((x - y) / (np.nanmax(data) - np.nanmin(data))) ** 2
        elif metric == 'interval':
            return (x - y) ** 2
        else:
            raise ValueError("metric must be 'nominal', 'ordinal', or 'interval'")

    # Observed disagreement
    Do = 0.0
    total_pairs = 0
    for j in range(data.shape[1]):
        ratings = data[:, j][mask[:, j]]
        for i in range(len(ratings)):
            for k in range(i + 1, len(ratings)):
                Do += delta(ratings[i], ratings[k])
                total_pairs += 1
    Do /= total_pairs

    # Expected disagreement
    all_ratings = data[mask]
    De = 0.0
    total_pairs = 0
    for i in range(len(all_ratings)):
        for k in range(i + 1, len(all_ratings)):
            De += delta(all_ratings[i], all_ratings[k])
            total_pairs += 1
    De /= total_pairs

    return 1 - Do / De if De != 0 else 1.0

def calculate_cohen(rater1, rater2, weights='linear'):
    """
    Calculate weighted Cohen's kappa for ordinal data.
    
    Parameters:
    rater1, rater2: array-like, ratings from two raters
    weights: str, 'linear' or 'quadratic'
    
    Returns:
    float: Weighted Cohen's kappa coefficient
    """
    return cohen_kappa_score(rater1, rater2, weights=weights)

def calculate_icc(ratings1, ratings2):
    """
    Main ICC calculation method - backwards compatible with original code.
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("Rating arrays must have same length")
    
    n = len(ratings1)
    if n < 2:
        return np.nan
    
    # Create dataframe in long format for pingouin
    data = []
    for i in range(n):
        data.append({'Subject': i, 'Rater': 'Rater1', 'Rating': ratings1[i]})
        data.append({'Subject': i, 'Rater': 'Rater2', 'Rating': ratings2[i]})
    
    df = pd.DataFrame(data)
    
    # Calculate ICC(3,k) - this is the most appropriate for two-rater consistency
    icc_result = pg.intraclass_corr(data=df, targets='Subject', raters='Rater', ratings='Rating')
    
    # Get ICC(3,k) which is "ICC3k" in pingouin
    icc3_row = icc_result[icc_result['Type'] == 'ICC3k']
    
    if len(icc3_row) > 0:
        return max(0,float(icc3_row['ICC'].iloc[0]))
    else:
        raise ValueError("ICC(3,k) calculation failed")