#!/bin/bash
#
# Run judge quality analysis (experiment 5) as a single sbatch job.
#
# Usage:
#   ./scripts/run_judge_quality_analysis.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --output-dir DIR            Directory to save results (default: results/judge_quality_analysis)
#   --data-dir DIR              Directory containing judge scores (default: data/judge_scores)
#   --selection-results-dir D   Root folder with ICC/alpha sampling results for selection analysis
#                               (default: results/04_01_downstream_task)
#   --models M1 M2 ...          Judge models to evaluate (default: claude-3.5-sonnet gpt-4.1 ...)
#   --datasets D1 D2 ...        Datasets to evaluate (default: hanna medval mslr summeval)
#   --budgets B1 B2 ...         Annotation budgets for selection analysis (default: 5 10 15 ... 50)
#   --vm-method METHOD          Variance-matching method name (default: variance_matched_weighted_.9)
#   --thresholds T1 T2 ...      Classification thresholds; one plot per threshold (default: 0.6 0.7 0.8)
#
# Examples:
#
#   Run with all defaults
#   ./scripts/run_judge_quality_analysis.sh
#
#   Custom output dir and thresholds
#   ./scripts/run_judge_quality_analysis.sh --output-dir results/my_run --thresholds 0.5 0.7 0.9
#
#   Skip selection analysis (no --selection-results-dir and default doesn't exist)
#   ./scripts/run_judge_quality_analysis.sh --output-dir results/quick

set -e  # Exit on error

# Default values
OUTPUT_DIR="results/04_08_kendall_spearman_jqa_threshold"
DATA_DIR="data/judge_scores"
SELECTION_RESULTS_DIR="results/04_07_kendall_spearman"
MODELS=("claude-3.5-sonnet" "gpt-4.1" "gpt-5" "deepseek-r1" "gemini-2.5-pro")
DATASETS=("hanna" "medval" "mslr" "summeval")
BUDGETS=(5 10 15 20 25 30 35 40 45 50)
VM_METHOD="variance_matched_weighted_.9"
THRESHOLDS=(0.6 0.7 0.8 0.9)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --selection-results-dir)
      SELECTION_RESULTS_DIR="$2"
      shift 2
      ;;
    --vm-method)
      VM_METHOD="$2"
      shift 2
      ;;
    --models)
      MODELS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --datasets)
      DATASETS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --budgets)
      BUDGETS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        BUDGETS+=("$1")
        shift
      done
      ;;
    --thresholds)
      THRESHOLDS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        THRESHOLDS+=("$1")
        shift
      done
      ;;
    -h|--help)
      head -35 "$0" | tail -33
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=============================================="
echo "Judge Quality Analysis - Configuration"
echo "=============================================="
echo "OUTPUT_DIR:            $OUTPUT_DIR"
echo "DATA_DIR:              $DATA_DIR"
echo "SELECTION_RESULTS_DIR: $SELECTION_RESULTS_DIR"
echo "MODELS:                ${MODELS[*]}"
echo "DATASETS:              ${DATASETS[*]}"
echo "BUDGETS:               ${BUDGETS[*]}"
echo "VM_METHOD:             $VM_METHOD"
echo "THRESHOLDS:            ${THRESHOLDS[*]}"
echo "=============================================="
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Submitting sbatch job for judge quality analysis..."

sbatch \
  --job-name="judge_quality" \
  --partition=nigam-h100 \
  --nodelist=secure-gpu-14 \
  --gres=gpu:1 \
  --mem=100G \
  --time=12:00:00 \
  --ntasks=1 \
  --output="${OUTPUT_DIR}/slurm_%j.out" \
  --error="${OUTPUT_DIR}/slurm_%j.err" \
  --wrap="
    source \$CONDA_DIR/etc/profile.d/conda.sh
    conda activate pac_judge
    cd /share/pi/nigam/users/aunell/SmartSample_local
    python -m src.experiments._5_judge_quality_analysis \
      --data-dir '$DATA_DIR' \
      --output-dir '$OUTPUT_DIR' \
      --selection-results-dir '$SELECTION_RESULTS_DIR' \
      --models ${MODELS[*]} \
      --datasets ${DATASETS[*]} \
      --budgets ${BUDGETS[*]} \
      --vm-method '$VM_METHOD' \
      --thresholds ${THRESHOLDS[*]}
  "

echo "  → Job submitted"
echo ""
echo "=============================================="
echo "Logs saved to: $OUTPUT_DIR/slurm_<jobid>.out/.err"
echo "=============================================="
