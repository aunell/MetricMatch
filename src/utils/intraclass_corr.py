from __future__ import annotations
import os
import json
from functools import reduce

import numpy as np
import pandas as pd

import warnings
from collections.abc import Iterable


## Util function for obtaining evaluation score with differing data organization 
def get_deepest_key(d):
    if not isinstance(d, dict) or not d:
        return None
    
    # Get the first key-value pair at the current level
    _, value = next(iter(d.items()))
    
    # If the value is another dictionary, recurse
    if isinstance(value, dict):
        return get_deepest_key(value)
    else:
        # Otherwise, this is the lowest level
        return value
    
class PointwiseICC:
    def __init__(self, n: int, k: int, data: pd.DataFrame = None, normalize: bool = True, validate: bool = True, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
        self.n = n
        self.k = k

        self.data = data = self._format_data(data, targets, raters, ratings, validate=validate)
        self.normalize = normalize
        self.targets = targets
        self.raters = raters
        self.ratings = ratings

        if normalize and self.data is not None:
            self._compute_pointwise_normalized_anova(data, targets, raters, ratings)
        elif not normalize and self.data is not None:
            self._compute_pointwise_unnormalized_anova(data, targets, raters, ratings)

    def _format_data(self, data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score", validate: bool = True):
        if data is None:
            return data

        # Fast path for clean candidate subsets: skip expensive pivot_table validation.
        # Safe when data is pre-filtered to shared text_ids (no missing raters, no duplicates).
        if not validate:
            return data

        ## If there are targets that do not include ratings from all raters, drop them
        pivoted_table = data.pivot_table(values = ratings, index = targets, columns = raters)
        nan_targets = pivoted_table[pivoted_table.isna().any(axis=1)].index.tolist()
        if len(nan_targets):
            print("Dropped {} rows due to missing ratings for some targets.".format(len(nan_targets)))
            fdata = data.loc[~data[targets].isin(nan_targets)]
            self.n = fdata[targets].unique().size
        else:
            fdata = data

        ## If there are duplicates, average them (not sure if this is really correct to do because introduces \\
        ## bias via the ICC, better to do N-way ANOVA, TODO later)
        grp_both = fdata.groupby([targets, raters], observed=True, group_keys=False)[ratings]

        if grp_both.count().nunique() != 1:
            fdata = grp_both.mean().reset_index()
            print("Dropped {} rows due to duplicate ratings by same rater on target.".format(self.n - fdata[targets].unique().size))
            self.n = fdata[targets].unique().size
        return fdata

    def _compute_pointwise_unnormalized_anova(self, data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
        if self.normalize:
            self.normalize = not self.normalize
            warnings.warn("Called un-normalized ANOVA on normalized object, will change normalization parameter.")
        k = data[raters].unique().size
        n = data[targets].unique().size

        assert self.k == k, "Number of unique raters in provided data does not match initialized k."
        assert self.n == n, "Number of unique targets in provided data does not match initialized n."
        
        s = data.groupby(targets)[ratings].mean()
        m = data.groupby(raters)[ratings].mean()

        x_tot = data[ratings].mean()

        ## for each text i, (S_i - x_tot)^2
        self.unnormalized_msb_expand = msb_expand = (s - x_tot) ** 2 # n x 1 each element in the array is contribution of text i to msb
        self.msb = msb = (k / (n - 1)) * msb_expand.sum()

        self.msb_expand = None

        ## for each text i, (1 / k) sum_j=1^k (x_ij - M_j)^2
        self.unnormalized_mse_partial_expand = mse_partial_expand = data.groupby(targets)[[raters, ratings]].apply(lambda x: np.mean((x.set_index(raters).squeeze() - m) ** 2))
        # n x 1 each element in the array is contribution of text i to mse

        self.mse = mse = (1 / ((n - 1)*(k - 1))) * (mse_partial_expand.sum() - (k * msb_expand.sum()))

        self.mse_expand = None

        self.icc_expand = None
        self.icc = (msb - mse) / msb

        return

    def _compute_pointwise_normalized_anova(self, data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
        if not self.normalize:
            self.normalize = not self.normalize
            warnings.warn("Called normalized ANOVA on un-normalized object, will change normalization parameter.")
        
        k = data[raters].unique().size
        n = data[targets].unique().size

        assert self.k == k, "Number of unique raters in provided data does not match initialized k."
        assert self.n == n, "Number of unique targets in provided data does not match initialized n."
        
        s = data.groupby(targets)[ratings].mean()
        m = data.groupby(raters)[ratings].mean()

        x_tot = data[ratings].mean()

        self.unnormalized_msb_expand = None

        ## for each text i, (k / (n-1)) (S_i - x_tot)^2
        try:
            self.msb_expand = msb_expand = (k / (n - 1)) * (s - x_tot) ** 2 # n x 1 each element in the array is contribution of text i to msb
            self.msb = msb = msb_expand.sum()
        except:
            self.msb_expand = msb_expand = pd.Series(np.zeros(n), index=data[targets].unique())
            self.msb=None
            self.mse=None
            self.icc=None
            return
        
        # Vectorized MSE: map each row's rater to its mean, compute (score - rater_mean)^2,
        # then sum per target — equivalent to the original apply+lambda but much faster.
        rater_means = data[raters].map(m)
        sq_diff = (data[ratings] - rater_means) ** 2
        unnormalized_mse_partial_expand = sq_diff.groupby(data[targets]).sum()
        self.unnormalized_mse_partial_expand = unnormalized_mse_partial_expand

        ## for each text i, sum_j=1^k (x_ij - M_j)^2
        # n x 1 each element in the array is contribution of text i to mse
        self.mse_expand = mse_expand = ((1 / ((n - 1) * (k - 1))) * unnormalized_mse_partial_expand) - ((1 / (k - 1)) * msb_expand)
        
        self.mse = mse = mse_expand.sum()

        self.icc_expand = (msb_expand - mse_expand) / msb ## TODO: Need to fix this part, because this introduces some biases
        self.icc = (msb - mse) / msb
        return
    
    @staticmethod
    def check_dims(p1: PointwiseICC, p2: PointwiseICC):
        if p1.normalize != p2.normalize:
            return False
        if p1.normalize:
            return p1.n == p2.n and p1.k == p2.k
        else:
            return p1.n == p2.n
        
class MacroAveragePointwiseICC:
    def __init__(self, data_list: Iterable[PointwiseICC], targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
        ok_to_add = np.all([PointwiseICC.check_dims(data_list[i], data_list[i+1]) for i in range(len(data_list)-1)]) # np.logical_and(reduce(lambda left, right: PointwiseICC.check_dims(left, right), data_list)) ## need to fix
        if not ok_to_add:
            self.data = None
            error_msg = "Dimensions and normalizations of ICC computations do not match. Dimensions/normalizations below:\n"
            error_msg += "\n".join(["k: {}, n: {}, normalize: {}".format(pwICC.k, pwICC.n, pwICC.normalize) for pwICC in data_list])
            raise ValueError(error_msg)
        self.normalize = np.all([pwICC.normalize for pwICC in data_list])
        
        self.data = reduce(lambda left, right: pd.merge(left, right, how="inner"), [pwICC.data for pwICC in data_list])
        self.data_list = data_list
        self.targets = targets
        self.raters = raters
        self.ratings = ratings

        self._initialize_self(self.data, data_list, targets, raters, ratings)
        
    def _initialize_self(self, data: pd.DataFrame, data_list: Iterable[PointwiseICC], targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
        self.k = k = data[raters].unique().size
        self.n = n = data[targets].unique().size

        avg_msb_expand = np.zeros(n)
        avg_msb = 0
        avg_mse_expand = np.zeros(n)
        avg_mse = 0
        avg_icc_expand = np.zeros(n)
        avg_icc = 0
        for pwICC in data_list:
            if self.normalize:
                avg_msb_expand += pwICC.msb_expand
                avg_msb += pwICC.msb
                avg_mse_expand += pwICC.mse_expand
                avg_mse += pwICC.mse
                avg_icc_expand += pwICC.icc_expand
                avg_icc += pwICC.icc

            else:
                avg_msb_expand += pwICC.unnormalized_msb_expand
                avg_msb += pwICC.msb
                avg_mse_expand += pwICC.unnormalized_mse_partial_expand
                avg_mse += pwICC.mse
                avg_icc += pwICC.icc

        avg_msb_expand /= len(self.data_list)
        avg_msb /= len(self.data_list)
        avg_mse_expand /= len(self.data_list)
        avg_mse /= len(self.data_list)
        avg_icc_expand /= len(self.data_list)
        avg_icc /= len(self.data_list)

        self.msb_expand = avg_msb_expand
        self.msb = avg_msb
        self.mse_expand = avg_mse_expand
        self.mse = avg_mse
        self.icc_expand = avg_icc_expand
        self.icc = avg_icc