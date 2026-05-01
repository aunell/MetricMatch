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

## Experiments
In the `src/experiments` folder, experiments are numbered in order of execution:

```bash
_1_get_judge_scores.py
_2_variance_selection_analysis.py
_3_model_model_vs_model_human_disagreement.py
_4_human_or_model_agreement_vs_gap.py
_5_judge_quality_analysis.py
_6_selection_method_downstream.py
_7_win_rates.py
_8_annotations_saved.py
```

**_1_get_judge_scores.py** — Queries LLM judges (OpenAI, Anthropic, Llama, Qwen, Gemma, Gemini) to score text on a given dataset and dimension. Outputs judge scores as JSON and CSV.

**_2_variance_selection_analysis.py** — Evaluates different sampling strategies (variance-matched, random, oracle) for estimating ICC and Krippendorff's alpha under limited annotation budgets across datasets.

**_3_model_model_vs_model_human_disagreement.py** — Compares inter-model variance to model-human disagreement, showing that when LLM judges disagree with each other they also tend to disagree with humans.

**_4_human_or_model_agreement_vs_gap.py** — Computes inter-human and human-model agreement metrics (ICC, Krippendorff's alpha, MSE) across datasets and axes.

**_5_judge_quality_analysis.py** — Evaluates LLM judge quality against human ground truth across reliability metrics and downstream tasks.

**_6_selection_method_downstream.py** — Measures model ranking recovery (Spearman ρ) for each selection method as a downstream task; accepts `--folder` or `--input`.

**_7_win_rates.py** — Computes estimation and threshold win rates of `variance_matched_weighted_.9` vs random across datasets, metrics, and budgets; accepts `--folder`.

**_8_annotations_saved.py** — Computes how many annotations our method saves relative to random at budget 50 (i.e., the smallest budget at which our method matches random's error at full budget); accepts `--folder`.


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