import pandas as pd
import json

def generate_latex_table(data_json_path):
    # Load JSON
    with open(data_json_path, "r") as f:
        data_json = json.load(f)

    df = pd.DataFrame(data_json)

    # Keep only Cluster and Random
    df = df[df["strategy"].isin(["Cluster", "Random"])]

    # Pivot so Random and Cluster are side by side
    pivot_df = df.pivot_table(
        index=["dataset_name", "n_expensive"],
        columns="strategy",
        values="mean"  # use mean for relative improvement
    ).reset_index()

    # Compute relative improvement
    pivot_df["rel_improvement"] = round((pivot_df["Random"] - pivot_df["Cluster"]) , 3)

    # Format for LaTeX table
    latex_table = pivot_df.pivot_table(
        index="n_expensive",
        columns="dataset_name",
        values="rel_improvement"
    ).applymap(lambda x: f"{x:.2}")  # format as percentage
    return latex_table
