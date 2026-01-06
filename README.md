# Smarter_Sampling

## Overview
We aim to provide theoretical and experimental guarantees regarding improved sampling method to accurately predict ICC of human and LLM judge rankings of non-verifiable text, improving robustness of LLM-judge applications in data poor regimes.

## Installation

```bash
# Clone the repository
git clone 
cd SmartSample_local

# Install dependencies
pip install -r requirements.txt
```

## Original Sampling Code
In the scripts folder, we find the following:
```bash
_1_get_judge_scores.sh
_2_real_data_experiment.sh
_3_confidence_interval_rde.sh
```

The first script generates the LLM judge scores for given datasets. The second script employs the different selection mechanisms to select the most informative subset of annotations to obtain. The third script compares the confidence interval widths of our best performing method to random selection of annotation subset.

## Additional Experiments
In the src/experiments folder, we find the additional:
```bash
inter_model_vs_human_disagreement.py
llm_judge_tradeoff_experiment.py
variance_selection_analysis.py
```

The first experiment copmares the standard deviation between model scores to the differene between mean model score and human score, indicating that there is a positive correlation as models disagree with each other that they also disagree with humans as well.\\

The second script uses simulated data to show the tradeoff of getting more human annotations on a single data point with respect to inter-human agreement. \\

The third script further formalizes this correlation by showing the relationship between inter-model ICC and model-human ICC variance components, and employs a variance based selection algorithm to highlight that this selection algorithm leads to consistent gains beyond random. 


## Dataset Class

The codebase uses a flexible dataset abstraction:
- `baseDataset` (in `src/datasets/base_dataset.py`): Base class for datasets. Handles loading, prompt creation, and result formatting.
- `SummevalDataset` (in `src/datasets/summ_eval.py`): Loads the Summeval dataset, flattens it, and generates prompts for LLM evaluation. Prompts are tailored to each evaluation dimension.
- To add a new dataset, subclass `baseDataset` and implement `extract_data` and `create_prompt`.

## Dataset Information
MSLR: https://github.com/allenai/mslr-annotated-dataset/blob/main/data/data_with_overlap_scores.json \\

HANNA: https://github.com/dig-team/hanna-benchmark-asg/blob/main/hanna_stories_annotations.csv \\

SummEval: https://huggingface.co/datasets/mteb/summeval \\

MedVal: https://arxiv.org/pdf/2507.03152 \\