import pandas as pd
from typing import Tuple, List
import json
from dataset_classes.base_dataset import baseDataset

# Scoring dimensions and their criteria
DIMENSION_CRITERIA = {
        'fluency': 'Is the generated summary fluent?',
        'population': 'Is the *population* in the generated summary the same as the population in the target summary?',
        'intervention': 'Is the *intervention* in the generated summary the same as the intervention in the target summary?',
        'outcome': 'Is the *outcome* in the generated summary the same as the outcome in the target summary?',
    }

def process_single_item(data):
    """
    Process a single dictionary from the list and extract relevant data.
    
    Parameters:
    data (dict): Dictionary containing review data
    
    Returns:
    list: List of row dictionaries with extracted data
    """
    # Extract the target text
    target = data.get('target', '')
    review_id = data.get('review_id', '')
    subtask = data.get('subtask', '')
    
    # Create a list to hold the rows from this item
    rows = []
    
    # Process each prediction
    for prediction in data.get('predictions', []):
        exp_short = prediction.get('exp_short', '')
        prediction_text = prediction.get('prediction', '')
        
        # Check if there are annotations
        annotations = prediction.get('annotations', [])
        
        if annotations:
            # If there are annotations, extract the data from each one
            for annotation in annotations:
                row = {
                    'review_id': review_id,
                    'subtask': subtask,
                    'exp_short': exp_short,
                    'target': target,
                    'prediction': prediction_text,
                    'fluency': annotation.get('fluency'),
                    'population': annotation.get('population'),
                    'intervention': annotation.get('intervention'),
                    'outcome': annotation.get('outcome')
                }
                rows.append(row)
        else:
            # If there are no annotations, add a row with None values for annotation metrics
            row = {
                'review_id': review_id,
                'subtask': subtask,
                'exp_short': exp_short,
                'target': target,
                'prediction': prediction_text,
                'fluency': None,
                'population': None,
                'intervention': None,
                'outcome': None,
            }
            rows.append(row)
    
    return rows

def json_to_dataframe(json_data):
    """
    Convert JSON data with predictions and annotations to a pandas DataFrame.
    Handles both single dictionary and list of dictionaries.
    
    Parameters:
    json_data (str, dict, or list): The JSON data
    
    Returns:
    pd.DataFrame: DataFrame with target, prediction, fluency, population, intervention, outcome
    """
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    else:
        data = json_data
    
    # Create a list to hold all rows for our DataFrame
    all_rows = []
    
    # Check if we have a list of dictionaries or a single dictionary
    if isinstance(data, list):
        # Process each dictionary in the list
        for item in data:
            rows = process_single_item(item)
            all_rows.extend(rows)
    else:
        # Process the single dictionary
        rows = process_single_item(data)
        all_rows.extend(rows)
    
    # Create DataFrame from all collected rows
    df = pd.DataFrame(all_rows)
    
    # Rename prediction column to text column

    return df


class MSLRDataset(baseDataset):
    def extract_data(self):
        # Read the JSON file
        # Initialize an empty list to store the data
        data = []
        with open('/share/pi/nigam/users/aunell/SmartSample_local/data/raw_data/data_with_overlap_scores.json', 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    json_obj = json.loads(line)
                    data.append(json_obj)
        df = json_to_dataframe(data)
        df_clean = df.dropna().drop(columns=['subtask'])
        df_clean['text_id'] = df_clean['exp_short'].astype(str) + '_' + df_clean['review_id'].astype(str)
        if self.sample_size is not None:
            df_clean = df_clean.sample(n=min(self.sample_size, len(df_clean)), random_state=42)
        df_clean = df_clean.rename(columns={'prediction': 'text'})
        df_clean = df_clean.rename(columns={self.dimension: 'original_score'})
        df_clean = df_clean.rename(columns={'target': 'ground_truth'})
        return df_clean

    def create_prompt(self, row: pd.Series) -> str:
        criteria = DIMENSION_CRITERIA[self.dimension]
        ANSWER_KEYS = {
        'Is the generated summary fluent?': {
            2: ['2: Yes--there are no errors that impact comprehension of the summary'],
            1: ['1: Somewhat--there are some grammatical or lexical errors but I can understand the meaning'],
            0: ['0: No--there are major grammatical or lexical errors that impact comprehension']
        },
        'Is the *population* in the generated summary the same as the population in the target summary?': {
            0: ['0: No'],
            1: ['1: Partially'],
            2: ['2: Yes']
        },
        'Is the *intervention* in the generated summary the same as the intervention in the target summary?': {
            0: ['0: No'],
            1: ['1: Partially'],
            2: ['2: Yes']
        },
        'Is the *outcome* in the generated summary the same as the outcome in the target summary?': {
            0: ['0: No'],
            1: ['1: Partially'],
            2: ['2: Yes']
        }
    }
        criteria = '\n'.join([item[0] for item in ANSWER_KEYS[DIMENSION_CRITERIA[self.dimension]].values()])

        prompt = f"""You are an expert in evaluating the quality of a summary of a review. Your task is to evaluate the {self.dimension} of the summary on a scale of 0 to 3.

Question:
{DIMENSION_CRITERIA[self.dimension]}

Evaluation Criteria:
{criteria}

Generated Summary:
{row['text']}

Target Summary:
{row['ground_truth']}

Please provide your evaluation in the following JSON format:
{{
"evaluation": {{
    "score": <score 0-3>
}}
}}

Ensure your output is valid JSON that can be parsed programmatically. Do not include any text outside of the JSON structure."""
        return prompt

    def get_available_dimensions(self) -> List[str]:
        return list(DIMENSION_CRITERIA.keys())