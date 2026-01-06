import pandas as pd
from typing import List, Dict, Any, Optional

class baseDataset:
    def __init__(self, dimension: str, sample_size: Optional[int] = None):
        self.dimension = dimension if dimension else self.get_available_dimensions()[0]
        if dimension and dimension not in self.get_available_dimensions():
            raise ValueError(f"Dimension {dimension} not found in available dimensions: {self.get_available_dimensions()}")
        self.sample_size = sample_size
        self.data = None

    def get_available_dimensions(self) -> List[str]:
        """
        Returns a list of available dimensions for the dataset.
        """
        raise NotImplementedError

    def extract_data(self):
        """
        Extract and flatten data from the dataset. Should be implemented by subclasses.
        """
        raise NotImplementedError

    def create_prompt(self, row: pd.Series) -> str:
        """
        Create an evaluation prompt for a given row. Should be implemented by subclasses.
        """
        raise NotImplementedError

    def format_result(self, evaluation: Dict[str, Any], data_row: pd.Series) -> Dict[str, Any]:
        """
        Format the result for output. Can be overridden by subclasses for custom formatting.
        """
        return {
            "text_id": data_row['text_id'],
            "input_text": data_row['text'],
            "original_score": float(data_row['original_score']),
            "evaluation": evaluation,
            "source_text": data_row.get('source_text', None),
            "ground_truth": data_row.get('ground_truth', None)
        }

    def get_data_and_prompts(self) -> pd.DataFrame:
        """
        Returns a DataFrame with all data and a 'prompt_text' column.
        """
        if self.data is None:
            self.data = self.extract_data()
        data = self.data.copy()
        data['prompt_text'] = data.apply(self.create_prompt, axis=1)
        return data