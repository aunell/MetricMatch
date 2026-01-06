import pandas as pd
from datasets import load_dataset
from typing import Tuple, List
import json
from dataset_classes.base_dataset import baseDataset
# Scoring dimensions and their criteria
DIMENSION_CRITERIA = {
    "Relevance": """1 - Irrelevant: Content does not address the topic or purpose.\n2 - Poor relevance: Most content is off-topic or tangential.\n3 - Moderately relevant: Some content is relevant but includes off-topic material.\n4 - Good relevance: Most content is relevant with minor deviations.\n5 - Excellent relevance: All content is highly relevant to the topic.""",
    "Coherence": """1 - Incoherent: Text is disjointed, lacks logical flow, and ideas are scattered.\n2 - Poor coherence: Text has major gaps in logic and organization.\n3 - Moderate coherence: Text is somewhat organized but has noticeable gaps or inconsistencies.\n4 - Good coherence: Text flows well with minor issues in organization or clarity.\n5 - Excellent coherence: Text is exceptionally well-organized with clear logical flow throughout.""",
    "Empathy": """1 - Lack of empathy: Story lacks any evidence of understanding or consideration for the character's feelings.\n2 - Poor empathy: Story shows minimal understanding of the character's emotions.\n3 - Moderate empathy: Story hints at some understanding of the character's feelings.\n4 - Good empathy: Story demonstrates a good understanding of the character's emotions.\n5 - Excellent empathy: Story fully captures the character's emotions and demonstrates deep understanding.""",
    "Surprise": """1 - Lack of surprise: Story is predictable and lacks unexpected elements.\n2 - Poor surprise: Story has some predictable elements.\n3 - Moderate surprise: Story has some unexpected elements.\n4 - Good surprise: Story has several unexpected elements.\n5 - Excellent surprise: Story has many unexpected elements.""",
    "Engagement": """1 - Lack of engagement: Story is boring and lacks interesting elements.\n2 - Poor engagement: Story has some interesting elements.\n3 - Moderate engagement: Story has some interesting elements.\n4 - Good engagement: Story has several interesting elements.\n5 - Excellent engagement: Story has many interesting elements.""",
    "Complexity": """1 - Lack of complexity: Story is simple and lacks interesting elements.\n2 - Poor complexity: Story has some interesting elements.\n3 - Moderate complexity: Story has some interesting elements.\n4 - Good complexity: Story has several interesting elements.\n5 - Excellent complexity: Story has many interesting elements."""
}

# Utility functions (if any) can remain here, but dataset and prompt logic is now in pacDataset/SummevalDataset.

class HannaDataset(baseDataset):
    def extract_data(self):
        # Read the CSV file
        df = pd.read_csv('/share/pi/nigam/users/aunell/SmartSample_local/data/raw_data/hanna_stories_annotations.csv')
        
        # Assuming the CSV has columns for text, scores, etc.
        # Adjust column names based on actual CSV structure
        data = pd.DataFrame({
            'text': df.groupby('Story ID')['Story'].first(),  # Take first text for each ID
            'source_text': df.groupby('Story ID')['Prompt'].first(),
            'original_score': df.groupby('Story ID')[self.dimension].mean(), # Take mean score for each ID
            'text_id': df['Story ID'].unique()
        })

        if self.sample_size is not None:
            data = data.sample(n=min(self.sample_size, len(data)), random_state=42)
        return data

    def create_prompt(self, row: pd.Series) -> str:
        criteria = DIMENSION_CRITERIA[self.dimension]
        prompt = f"""You are an expert in evaluating a story in response to a prompt. Your task is to evaluate the {self.dimension} of the story on a scale of 1 to 5.

Prompt:
{row['source_text']}

Story:
{row['text']}

Evaluation Criteria for {self.dimension.capitalize()} (Scale 1-5):
{criteria}

Please provide your evaluation in the following JSON format:
{{
"evaluation": {{
    "score": <score 1-5>
}}
}}

Ensure your output is valid JSON that can be parsed programmatically. Do not include any text outside of the JSON structure."""
        return prompt

    def get_available_dimensions(self) -> List[str]:
        return list(DIMENSION_CRITERIA.keys())