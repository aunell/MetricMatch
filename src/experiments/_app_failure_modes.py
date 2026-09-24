import os
import itertools

import numpy as np
import pandas as pd

from src.utils.data_loading import load_judge_scores

DEFAULT_TOTAL_ANNOTATIONS = 300
DEFAULT_MODEL_NAMES = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5", "deepseek-r1", "gemini-2.5-pro"] # ["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
DEFAULT_DATA_DIR = "data/judge_scores"

TOTAL_ANNOTATIONS = DEFAULT_TOTAL_ANNOTATIONS
datasets = [
    "hanna",
    "medval",
    "mslr",
    "summeval"
]
model_names = DEFAULT_MODEL_NAMES
# All models that need data loaded (union of target + ensemble, preserving order)
_seen = set()
models_to_load = [m for m in model_names if not (m in _seen or _seen.add(m))]
DATA_DIR = DEFAULT_DATA_DIR

EVALUATION_AXES = {
    "hanna": ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval": ["Risk"],
    "mslr": ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"]
}

axis_wise = []
for dataset in datasets:
    df, _ = load_judge_scores(dataset, models_to_load, DATA_DIR, EVALUATION_AXES)

    axes = EVALUATION_AXES[dataset]
    for axis in axes:
        axis_df = df.loc[df["evaluation_axis"] == axis]
        num_models = axis_df["model_name"].nunique()
        texts_per_model = axis_df.groupby("text_id")["model_name"].nunique()
        shared_text_ids = texts_per_model[texts_per_model == num_models].index[:TOTAL_ANNOTATIONS]
        axis_df = axis_df.loc[axis_df["text_id"].isin(shared_text_ids)]
        means = axis_df.loc[axis_df["model_name"].isin([*model_names, "original"])].groupby("model_name")["evaluation_score"].mean()
        axis_wise.append({
            "dataset": dataset,
            "axis": axis,
            "comparison": "model",
            "mean_diff": np.mean([
                (means.loc[m1] - means.loc[m2]) ** 2 for m1, m2 in itertools.combinations(list(means.index), 2) if "original" not in [m1, m2]
            ]),
            "var": axis_df.loc[axis_df["model_name"].isin(model_names)]["evaluation_score"].var()
        })
        axis_wise.append({
                "dataset": dataset,
                "axis": axis,
                "comparison": "human",
                "mean_diff": np.mean([
                    (means.loc[m1] - means.loc[m2]) ** 2 for m1, m2 in itertools.combinations(list(means.index), 2) if "original" in [m1, m2]
                ]),
                "var": axis_df.loc[axis_df["model_name"] == "original"]["evaluation_score"].var()
            })

summary_stats = pd.DataFrame.from_records(axis_wise)
print(summary_stats)
print(summary_stats.replace({
    "comparison": {
        "model": r"$X,Y = $ model",
        "human": r"$X,Y = $ human, model"
    }
}).set_index(keys=["dataset", "axis", "comparison"]).to_latex(
    header=True,
    float_format="%.2f",
    multirow=True
))