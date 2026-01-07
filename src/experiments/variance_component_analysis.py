import os
import json
from functools import reduce

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
# import pingouin as pg

import matplotlib.pyplot as plt
import seaborn as sb

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

## Variance-based
def compute_pointwise_ms(data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
    k = data[raters].unique().size
    n = data[targets].unique().size
    
    s = data.groupby(targets)[ratings].mean()
    m = data.groupby(raters)[ratings].mean()

    x_tot = data[ratings].mean()

    ## for each text i, (S_i - x_tot)^2
    msb_expand = (s - x_tot) ** 2 # n x 1 each element in the array is contribution of text i to msb
    msb = (k / (n - 1)) * msb_expand.sum()

    ## for each text i, (1 / k) sum_j=1^k (x_ij - M_j)^2
    mse_partial_expand = data.groupby(targets)[[raters, ratings]].apply(lambda x: np.mean((x.set_index(raters).squeeze() - m) ** 2))
    # n x 1 each element in the array is contribution of text i to mse

    mse = (1 / ((n - 1)*(k - 1))) * (mse_partial_expand.sum() - (k * msb_expand.sum()))

    return msb_expand, msb, mse_partial_expand, mse

datasets = ["hanna", "medval", "mslr", "summeval"]
model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct"]
evaluation_axes = {
    "hanna": ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval": ["Risk"],
    "mslr": ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"]
}

dataset = "summeval"
mode = "pairwise" #"pairwise"  # "aggregate" or "pairwise"

plotpath = os.path.join(os.getcwd(), "plots")
plotpath = os.path.join(plotpath, dataset)
if not os.path.exists(plotpath):
    os.makedirs(plotpath, exist_ok = True)

## Data cleaning
# for dataset in datasets:
dfs = []
for model_name in model_names:
    for ev_ax in evaluation_axes[dataset]:
        filepath = os.path.join(os.getcwd(), "data/judge_scores", dataset, "results_{}_{}_{}.json".format(dataset, model_name, ev_ax))
        if os.path.exists(filepath):
            with open(filepath, "r") as fp:
                results = json.load(fp)
        else: 
            print(filepath, "does not exist")
            continue

        results = results["detailed_results"]
        if model_name == "gpt-4o-mini" and dataset == "summeval": # special case because sometimes seems to be formatted differently
            results = [dict(x, evaluation_score=get_deepest_key(x.pop("evaluation"))) for x in results]
        else:
            results = [dict(x, evaluation_score=x.pop("evaluation")["evaluation"]["score"]) for x in results]

        results = pd.DataFrame.from_records(results)
        results.drop("score_difference", axis=1, inplace=True)
        results["model_name"] = model_name
        results["evaluation_axis"] = ev_ax
        dfs.append(results)

df = pd.concat(dfs, axis=0, ignore_index=True)
text_info = df[["text_id", "evaluation_axis", "input_text", "source_text", "ground_truth", "original_score"]].drop_duplicates()
df.drop(labels=text_info.columns[2:], axis=1, inplace=True)

if np.all(text_info["ground_truth"].isnull()):
    text_info.drop("ground_truth", axis=1, inplace=True)

for ev_ax in evaluation_axes[dataset]:
    df = pd.concat([df,
                    text_info.assign(model_name="original").rename({"original_score": "evaluation_score"}, axis=1)[df.columns]
                    ],
                    axis=0)

    if mode == "aggregate":
        # Original aggregate mode: inter-model variance for all models together
        im_msb_expand, im_msb, im_mse_partial_expand, im_mse = compute_pointwise_ms(df.loc[df["model_name"] != "original"])

        ## Human-model ICC components
        hm = {}
        for model_name in df["model_name"].unique():
            if "original" in model_name:
                continue
            hm_msb_expand, hm_msb, hm_mse_partial_expand, hm_mse = compute_pointwise_ms(df.loc[df["model_name"].isin([model_name, "original"])])

            hm[model_name] = (hm_msb_expand, hm_msb, hm_mse_partial_expand, hm_mse)

        # MSB comparison
        hm_msb_expand = pd.concat([hm[model_name][0] for model_name in model_names], axis=1, keys=model_names)
        hm_msb_expand_mean = hm_msb_expand.mean(axis=1)
        hm_msb = np.mean([hm[model_name][1] for model_name in model_names])

        comparison_msb_expand = pd.concat([im_msb_expand, hm_msb_expand_mean], axis=1, keys=["inter_model", "human_model"])

        # MSE comparison
        hm_mse_partial_expand = pd.concat([hm[model_name][-2] for model_name in model_names], axis=1, keys=model_names)
        hm_mse_partial_expand_mean = hm_mse_partial_expand.mean(axis=1)
        hm_mse = np.mean([hm[model_name][-1] for model_name in model_names])

        comparison_mse_partial_expand = pd.concat([im_mse_partial_expand, hm_mse_partial_expand_mean], axis=1, keys=["inter_model", "human_model"])

    elif mode == "pairwise":
        # Pairwise mode: for each model, compare to average of other models
        ## Human-model ICC components
        hm = {}
        for model_name in df["model_name"].unique():
            if "original" in model_name:
                continue
            hm_msb_expand, hm_msb, hm_mse_partial_expand, hm_mse = compute_pointwise_ms(df.loc[df["model_name"].isin([model_name, "original"])])

            hm[model_name] = (hm_msb_expand, hm_msb, hm_mse_partial_expand, hm_mse)

        # For each model, compute inter-model variance with the other models
        im = {}
        for target_model in model_names:
            # Get all models except the target model
            other_models = [m for m in model_names if m != target_model]
            im_msb_expand, im_msb, im_mse_partial_expand, im_mse = compute_pointwise_ms(df.loc[df["model_name"].isin(other_models)])

            im[target_model] = (im_msb_expand, im_msb, im_mse_partial_expand, im_mse)

        # MSB comparison - aggregate across all pairwise comparisons
        hm_msb_expand = pd.concat([hm[model_name][0] for model_name in model_names], axis=1, keys=model_names)
        hm_msb_expand_mean = hm_msb_expand.mean(axis=1)
        hm_msb = np.mean([hm[model_name][1] for model_name in model_names])

        im_msb_expand = pd.concat([im[model_name][0] for model_name in model_names], axis=1, keys=model_names)
        im_msb_expand_mean = im_msb_expand.mean(axis=1)
        im_msb = np.mean([im[model_name][1] for model_name in model_names])

        comparison_msb_expand = pd.concat([im_msb_expand_mean, hm_msb_expand_mean], axis=1, keys=["inter_model", "human_model"])

        # MSE comparison - aggregate across all pairwise comparisons
        hm_mse_partial_expand = pd.concat([hm[model_name][-2] for model_name in model_names], axis=1, keys=model_names)
        hm_mse_partial_expand_mean = hm_mse_partial_expand.mean(axis=1)
        hm_mse = np.mean([hm[model_name][-1] for model_name in model_names])

        im_mse_partial_expand = pd.concat([im[model_name][-2] for model_name in model_names], axis=1, keys=model_names)
        im_mse_partial_expand_mean = im_mse_partial_expand.mean(axis=1)
        im_mse = np.mean([im[model_name][-1] for model_name in model_names])

        comparison_mse_partial_expand = pd.concat([im_mse_partial_expand_mean, hm_mse_partial_expand_mean], axis=1, keys=["inter_model", "human_model"])

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'aggregate' or 'pairwise'.")

    ## Plot comparison of human-model and inter-model
    r, p = pearsonr(comparison_msb_expand["inter_model"].values, comparison_msb_expand["human_model"].values)

    g = sb.regplot(data=comparison_msb_expand, x="inter_model", y="human_model")
    xlim = g.get_xlim()
    ylim = g.get_ylim()
    g.text(x=xlim[0] + 0.05*(xlim[1]-xlim[0]),
        y=ylim[1] - 0.05*(ylim[1]-ylim[0]),
        s="r={:.2f}, p={:.2g}".format(r, p))
    g.set_title("{} {} {} un-normalized MSB contribution:\n".format(dataset, ev_ax, mode)
                + r"$(S_i - \overline{X}_{\text{tot}})^2$, "
                + "human-model MSB: {:.2f}, inter-model MSB: {:.2f}".format(hm_msb, im_msb))
    plt.savefig(os.path.join(
        plotpath,
        "{}_{}_{}_msb_comparison.jpg".format(dataset, ev_ax, mode)
    ), dpi=300)
    plt.close()

    r, p = pearsonr(comparison_mse_partial_expand["inter_model"].values, comparison_mse_partial_expand["human_model"].values)

    g = sb.regplot(data=comparison_mse_partial_expand, x="inter_model", y="human_model")
    xlim = g.get_xlim()
    ylim = g.get_ylim()
    g.text(x=xlim[0] + 0.05*(xlim[1]-xlim[0]),
        y=ylim[1] - 0.05*(ylim[1]-ylim[0]),
        s="r={:.2f}, p={:.2g}".format(r, p))
    g.set_title("{} {} {} un-normalized MSE contribution:\n".format(dataset, ev_ax, mode)
                + r"$\frac{1}{k}\sum_{j=1}^k (x_{ij} - M_j)^2$, "
                + "human-model MSE: {:.2f}, inter-model MSE: {:.2f}".format(hm_mse, im_mse))
    plt.tight_layout()
    plt.savefig(os.path.join(
        plotpath,
        "{}_{}_{}_mse_partial_comparison_.jpg".format(dataset, ev_ax, mode)
    ), dpi=300)
    print(plotpath)
    plt.close()

##

# ## Components of ICC analysis
# def compute_icc_anova(data: pd.DataFrame, targets: str = "text_id", raters: str = "model_name", ratings: str = "evaluation_score"):
#     data = data.pivot_table(index=targets, columns=raters, values=ratings, observed=True)
#     data = data.reset_index().melt(id_vars=targets, value_name=ratings)
#     aov = pg.anova(data=data, dv=ratings, between=[targets, raters], ss_type=2)

#     ## Extract mean squares
#     msb = aov.at[0, "MS"]
#     mse = aov.at[2, "MS"]

#     ## ICC(3, k)
#     icc3k = (msb - mse) / msb

#     return msb, mse, icc3k

# ## Inter-model ICC components
# im_msb, im_mse, im_icc3k = compute_icc_anova(df.loc[df["model_name"] != "original"])

# ## Human-model ICC components
# hm = {}
# for model_name in df["model_name"].unique():
#     if "original" in model_name:
#         continue
#     hm_msb, hm_mse, hm_icc3k = compute_icc_anova(df.loc[df["model_name"].isin([model_name, "original"])])

#     hm[model_name] = (hm_msb, hm_mse, hm_icc3k)

# hm_msb, hm_mse, hm_icc3k = np.mean(list(hm.values()), axis=0)

# print(im_msb, im_mse, im_icc3k)
# print(hm_msb, hm_mse, hm_icc3k)