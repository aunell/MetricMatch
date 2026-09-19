"""
Data loading utilities for judge score experiments.

Contains functions for loading and preprocessing judge scores from JSON files.
"""

import os
import json
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer

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

    return df, text_info[["text_id", "input_text", "source_text"]].drop_duplicates()

def mean_pooling(model_output, attention_mask):
    """
    Averages the token embeddings while ignoring padding tokens
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0] 
    
    # Expand attention mask to match token embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum embeddings along the sequence length (ignoring padding)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    
    # Divide by total number of non-padding tokens
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def retrieve_embeddings(text_info, 
                        text_to_include=["source_text", "input_text"], 
                        lightweight=True, 
                        normalize=False):
    # Define text sequences to embed
    sentences = text_info[text_to_include].replace({None: ""}).agg(' '.join, axis=1).values.tolist()

    if lightweight:
        model = SentenceTransformer('all-MiniLM-L6-v2')

        sentence_embeddings = model.encode(sentences)

        if normalize:
            norm = np.linalg.norm(sentence_embeddings, ord=2, axis=1, keepdims=True)
            sentence_embeddings = sentence_embeddings / np.maximum(norm, 1e-12)
    else:
        # Load the ModernBERT model and tokenizer
        model_id = "nomic-ai/modernbert-embed-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa").to("cpu")

        print("Got model ... encoding input")
        # Tokenize input sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        print("Got input ... running through model")
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        print("Got output .. running mean pooling")
        # Apply mean pooling to get sentence-level embeddings
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        if normalize:
            # Normalize embeddings (Optional, recommended for Cosine Similarity calculations)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sentence_embeddings = sentence_embeddings.numpy()

    copy_info = text_info.copy()
    copy_info["embedding"] = list(sentence_embeddings)
    return copy_info