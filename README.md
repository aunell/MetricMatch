# Metric Match

## Overview
We aim to provide theoretical and experimental guarantees regarding improved sampling method to accurately predict ICC of human and LLM judge rankings of non-verifiable text, improving robustness of LLM-judge applications in data poor regimes.

## Installation

```bash
# Clone the repository
git clone 
cd Metric Match

# Create and activate the conda environment
conda env create -f environment.yml
conda activate metric_match
```

## Scripts
The `scripts/` folder contains SLURM-ready bash scripts that wrap the Python experiments:

```bash
get_judge_scores.sh          # Submits a job to run _1_get_judge_scores.py
run_variance_analysis.sh     # Submits per-dataset SLURM jobs to run _2_variance_selection_analysis.py
run_win_rates.sh             # Submits a job to run _3_win_rates.py
run_annotations_saved.sh     # Submits a job to run _4_annotations_saved.py
```

All scripts accept a `--folder` (or `--plots-dir`) argument to specify the results directory. Run any script with `-h` to see full usage.

## Experiments
In the `src/experiments` folder, experiments are numbered in order of execution:

```bash
_1_get_judge_scores.py
_2_variance_selection_analysis.py
_3_win_rates.py
_4_annotations_saved.py
_5_estimation_error.py
```

**_1_get_judge_scores.py** — Queries LLM judges (OpenAI, Anthropic, Llama, Qwen, Gemma, Gemini, DeepSeek) to score text on a given dataset and dimension. Outputs judge scores as JSON and CSV to `data/judge_scores/`.

**_2_variance_selection_analysis.py** — Core experiment. Evaluates sampling strategies (variance-matched, metric-matched, random, stratified, random-imc) for estimating reliability metrics (ICC, Krippendorff's alpha, Spearman ρ, Kendall τ, MSE) under limited annotation budgets. Supports a `--target-models` / `--ensemble-models` split so small cheap models act as the variance signal and large models are the evaluation targets. Saves per-dataset result dataframes to `results/<run>/`.

**_3_win_rates.py** — Computes estimation and threshold win rates of the target method vs. random across datasets, metrics, and budgets. Reads from a `--folder` of result dataframes produced by experiment 2.

**_4_annotations_saved.py** — Computes how many annotations the target method saves relative to random (i.e., the smallest budget at which the method matches random's error at full budget). Reads from a `--folder` of result dataframes produced by experiment 2.

**_5_estimation_error.py** — Plots estimation error curves per metric (ICC, Krippendorff's alpha, Spearman ρ, Kendall τ), with one line per method averaged across datasets/models/axes with 95% CI. Reads from a `--folder` of result dataframes produced by experiment 2.


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
