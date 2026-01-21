"""
Data loading utilities for judge score experiments.

Contains functions for loading and preprocessing judge scores from JSON files.
"""

import os
import json
import pandas as pd


def load_judge_scores(dataset, model_names, data_dir, evaluation_axes):
    """
    Load judge scores from JSON files and combine with human scores.

    Args:
        dataset: Dataset name (e.g., "medval", "hanna", "mslr", "summeval")
        model_names: List of model names to load
        data_dir: Base directory for data files
        evaluation_axes: Dict mapping dataset -> list of evaluation axes

    Returns:
        DataFrame with columns: text_id, model_name, evaluation_score, evaluation_axis
        Includes both model scores and human ("original") scores.
    """
    dfs = []

    for model_name in model_names:
        for ev_ax in evaluation_axes[dataset]:
            path = os.path.join(
                data_dir,
                dataset,
                f"results_{dataset}_{model_name}_{ev_ax}.json"
            )
            if not os.path.exists(path):
                continue

            with open(path) as f:
                results = json.load(f)["detailed_results"]

            rows = []
            for r in results:
                try:
                    score = r["evaluation"]["evaluation"]["score"]
                except (KeyError, TypeError):
                    score = r["evaluation"]["score"]
                row = {k: v for k, v in r.items() if k != "evaluation"}
                row["evaluation_score"] = score
                row["model_name"] = model_name
                row["evaluation_axis"] = ev_ax
                rows.append(row)

            dfs.append(pd.DataFrame(rows))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Add human ("original") scores
    text_info = (
        df[["text_id", "input_text", "source_text", "original_score", "evaluation_axis"]]
        .drop_duplicates()
    )

    human_df = (
        text_info[["text_id", "original_score", "evaluation_axis"]]
        .rename(columns={"original_score": "evaluation_score"})
    )
    human_df["model_name"] = "original"

    df = pd.concat(
        [df[["text_id", "model_name", "evaluation_score", "evaluation_axis"]], human_df],
        ignore_index=True
    )

    return df
