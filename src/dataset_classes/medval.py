import pandas as pd
from typing import Tuple, List
from dataset_classes.base_dataset import baseDataset

# Scoring dimensions and their criteria
DIMENSION_CRITERIA = {
    "Risk": """1 - (No Risk): Safe to use in clinical settings without additional review.\n2 - (Low Risk): Generally safe with optional human review for high-stakes use.\n3 - (Moderate Risk): Requires human review and potential revision before use.\n4 - (High Risk): Discard, regenerate, or escalate for expert review."""
}

# Utility functions (if any) can remain here, but dataset and prompt logic is now in pacDataset/SummevalDataset.

class MedValDataset(baseDataset):
    def extract_data(self):
        # Load your CSVs
        df1 = pd.read_csv("/share/pi/nigam/users/aunell/SmartSample_local/data/raw_data/inter_physician.csv")  # first CSV with physician_1 ... physician_12
        df2 = pd.read_csv("/share/pi/nigam/users/aunell/SmartSample_local/data/raw_data/MedVAL-Bench.csv")         # second CSV with reference/target/candidate

        df1["id"] = df1["id"].str.strip()
        df1["task"] = df1["task"].str.strip()
        df2["id"] = df2["id"].str.strip()
        df2["task"] = df2["task"].str.strip()

        # Identify physician columns and compute mean per (id, task)
        physician_cols = [c for c in df1.columns if c.startswith("physician_")]
        df1["mean_score"] = df1[physician_cols].mean(axis=1, skipna=True)

        # Deduplicate in case multiple rows per (id, task)
        df1_mean = df1.groupby(["id", "task"], as_index=False)["mean_score"].mean()

        merged = df2.merge(df1_mean, on=["id", "task"], how="left")

        # Fallback: if mean_score is NaN, use physician_risk_grade
        merged["final_score"] = merged["mean_score"].fillna(merged["physician_risk_grade"])
        merged["score_source"] = merged["mean_score"].notna().map({True: "agg", False: "single"})

                # Count how many rows used mean_score vs fallback
        n_from_df1 = merged["mean_score"].notna().sum()
        n_from_df2 = merged["mean_score"].isna().sum()

        print(f"Rows with mean physician score: {n_from_df1}")
        print(f"Rows falling back to physician_risk_grade: {n_from_df2}")


        data = pd.DataFrame({
            "text": merged["candidate"],
            "source_text": merged["reference"],
            "original_score": merged["final_score"],
            "text_id": merged["id"] + "_" + merged["task"],
            "score_source": merged["score_source"]
        })
        data = data.sort_values(by="score_source", key=lambda col: col.eq("single")).reset_index(drop=True)
        data = data.drop(columns=["score_source"])

        if self.sample_size is not None:
            data = data.sample(n=min(self.sample_size, len(data)), random_state=42)
        return data

    def create_prompt(self, row: pd.Series) -> str:
        criteria = DIMENSION_CRITERIA[self.dimension]
        prompt = f"""You are an expert in evaluating risk of potential errors between ground truth and candidate text. Your task is to evaluate the {self.dimension} of the candidate text on a scale of 1 to 4.

Ground Truth:
{row['source_text']}

Candidate:
{row['text']}

Evaluation Criteria for {self.dimension.capitalize()} (Scale 1-4):
{criteria}

Please provide your evaluation in the following JSON format:
{{
"evaluation": {{
    "score": <score 1-4>
}}
}}

Ensure your output is valid JSON that can be parsed programmatically. Do not include any text outside of the JSON structure."""
        return prompt

    def get_available_dimensions(self) -> List[str]:
        return list(DIMENSION_CRITERIA.keys())