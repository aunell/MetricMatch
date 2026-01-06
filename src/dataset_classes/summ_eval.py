import pandas as pd
from datasets import load_dataset
from typing import List
from dataset_classes.base_dataset import baseDataset
# Scoring dimensions and their criteria
DIMENSION_CRITERIA = {
    'coherence': """1 - Incoherent: Text is disjointed, lacks logical flow, and ideas are scattered.\n2 - Poor coherence: Text has major gaps in logic and organization.\n3 - Moderate coherence: Text is somewhat organized but has noticeable gaps or inconsistencies.\n4 - Good coherence: Text flows well with minor issues in organization or clarity.\n5 - Excellent coherence: Text is exceptionally well-organized with clear logical flow throughout.""",
    'consistency': """1 - Highly inconsistent: Contains multiple contradictions and conflicting statements.\n2 - Poor consistency: Has noticeable contradictions in ideas or facts.\n3 - Moderately consistent: Generally consistent with some minor contradictions.\n4 - Good consistency: Maintains consistent ideas with very few minor inconsistencies.\n5 - Excellent consistency: Perfectly consistent throughout, with no contradictions.""",
    'fluency': """1 - Not fluent: Text is difficult to read with major grammatical/structural issues.\n2 - Poor fluency: Contains frequent grammatical errors or awkward phrasing.\n3 - Moderate fluency: Generally readable but with some unnatural language.\n4 - Good fluency: Reads naturally with minor language issues.\n5 - Excellent fluency: Perfectly natural language throughout.""",
    'relevance': """1 - Irrelevant: Content does not address the topic or purpose.\n2 - Poor relevance: Most content is off-topic or tangential.\n3 - Moderately relevant: Some content is relevant but includes off-topic material.\n4 - Good relevance: Most content is relevant with minor deviations.\n5 - Excellent relevance: All content is highly relevant to the topic."""
}

# Utility functions (if any) can remain here, but dataset and prompt logic is now in pacDataset/SummevalDataset.

class SummevalDataset(baseDataset):
    def extract_data(self):
        dataset = load_dataset("mteb/summeval")
        df = pd.DataFrame(dataset['test'])
        all_texts = []
        all_scores = []
        all_ground_truths = []
        all_source_texts = []
        for _, row in df.iterrows():
            summaries = row['machine_summaries']
            dimension_scores = row[self.dimension]
            ground_truth = row['human_summaries'][0]
            source_text = row['text']
            for summary, score in zip(summaries, dimension_scores):
                all_texts.append(summary)
                all_scores.append(score)
                all_ground_truths.append(ground_truth)
                all_source_texts.append(source_text)
        text_ids = list(range(len(all_texts)))
        data = pd.DataFrame({
            'text': all_texts,
            'original_score': all_scores,
            'ground_truth': all_ground_truths,
            'source_text': all_source_texts,
            'text_id': text_ids
        })
        if self.sample_size is not None:
            data = data.sample(n=min(self.sample_size, len(data)), random_state=42)
        return data

    def create_prompt(self, row: pd.Series) -> str:
        criteria = DIMENSION_CRITERIA[self.dimension]
        prompt = f"""You are an expert in evaluating text summaries. Your task is to evaluate the {self.dimension} of the given summary on a scale of 1 to 5.\n\nOriginal Text to be Summarized:\n{row['source_text']}\n\nSummary to Evaluate:\n{row['text']}\n\nReference (Ground Truth) Text:\n{row['ground_truth']}\n\nEvaluation Criteria for {self.dimension.capitalize()} (Scale 1-5):\n{criteria}\n\nPlease provide your evaluation in the following JSON format:\n{{\n\"evaluation\": {{\n    \"score\": <score 1-5>\n}}\n}}\n\nEnsure your output is valid JSON that can be parsed programmatically. Do not include any text outside of the JSON structure.\n"""
        return prompt
    
    def get_available_dimensions(self) -> List[str]:
        return list(DIMENSION_CRITERIA.keys())