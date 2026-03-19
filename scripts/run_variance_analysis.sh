#!/bin/bash
#
# Run variance selection analysis across all datasets
# Each dataset is launched as a separate sbatch job.
#
# Usage:
#   ./scripts/run_variance_analysis.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --n-bootstrap N         Number of bootstrap samples (default: 100)
#   --n-candidates N        Number of candidate subsets (default: 20)
#   --total-annotations N   Total annotations budget (default: 300)
#   --plots-dir DIR         Directory to save plots (default: results/01_26_small)
#   --data-dir DIR          Directory containing judge scores (default: data/judge_scores)
#   --comparison-mode M     Comparison mode: average_pairwise, pairwise_average, or aggregate (default: average_pairwise)
#   --datasets D1 D2 ...    Datasets to run (default: all - hanna medval mslr summeval)
#   --model-names M1 M2     All model names to load (default: gpt-4o-mini ...)
#   --target-models M1 M2   Models to evaluate independently (default: same as --model-names)
#   --ensemble-models E1    Models used for variance matching / IMC (default: same as --model-names;
#                           each target is automatically excluded from its own ensemble)
#
# Examples:
#
#   Run with all defaults on all datasets
#   ./scripts/run_variance_analysis.sh
#
#
#   Run only on summeval and hanna with custom plots dir
#   ./scripts/run_variance_analysis.sh --datasets summeval hanna --plots-dir results/custom
#
#
#   Run with fewer bootstrap samples for faster testing
#   ./scripts/run_variance_analysis.sh --n-bootstrap 10

set -e  # Exit on error

# Default values
N_BOOTSTRAP=100
N_CANDIDATES=20
TOTAL_ANNOTATIONS=300
PLOTS_DIR="results/03_17_small"
DATA_DIR="data/judge_scores"
COMPARISON_MODE="pairwise_average"
ONLINE_ACQUISITION=true  # true → cumulative/incremental selection; false → batch selection
DATASETS=("medval" "summeval" "mslr" "hanna") #("hanna" "medval" "mslr" "summeval")
MODEL_NAMES=("claude-3.5-sonnet" "gpt-4.1" "gpt-5" )
# MODEL_NAMES=("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")
TARGET_MODELS=()    #("claude-3.5-sonnet" "gpt-4.1" "gpt-5")  # empty = use MODEL_NAMES
ENSEMBLE_MODELS=("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")  # empty = use MODEL_NAMES

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --n-bootstrap)
      N_BOOTSTRAP="$2"
      shift 2
      ;;
    --n-candidates)
      N_CANDIDATES="$2"
      shift 2
      ;;
    --total-annotations)
      TOTAL_ANNOTATIONS="$2"
      shift 2
      ;;
    --plots-dir)
      PLOTS_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --comparison-mode)
      COMPARISON_MODE="$2"
      shift 2
      ;;
    --online-acquisition)
      ONLINE_ACQUISITION=true
      shift
      ;;
    --no-online-acquisition)
      ONLINE_ACQUISITION=false
      shift
      ;;
    --datasets)
      DATASETS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --model-names)
      MODEL_NAMES=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODEL_NAMES+=("$1")
        shift
      done
      ;;
    --target-models)
      TARGET_MODELS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        TARGET_MODELS+=("$1")
        shift
      done
      ;;
    --ensemble-models)
      ENSEMBLE_MODELS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        ENSEMBLE_MODELS+=("$1")
        shift
      done
      ;;
    -h|--help)
      head -30 "$0" | tail -28
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
echo "Variance Selection Analysis - Configuration"
echo "=============================================="
echo "N_BOOTSTRAP:        $N_BOOTSTRAP"
echo "N_CANDIDATES:       $N_CANDIDATES"
echo "TOTAL_ANNOTATIONS:  $TOTAL_ANNOTATIONS"
echo "PLOTS_DIR:          $PLOTS_DIR"
echo "DATA_DIR:           $DATA_DIR"
echo "COMPARISON_MODE:    $COMPARISON_MODE"
echo "ONLINE_ACQUISITION: $ONLINE_ACQUISITION"
echo "DATASETS:           ${DATASETS[*]}"
echo "MODEL_NAMES:        ${MODEL_NAMES[*]}"
echo "TARGET_MODELS:      ${TARGET_MODELS[*]:-<same as MODEL_NAMES>}"
echo "ENSEMBLE_MODELS:    ${ENSEMBLE_MODELS[*]:-<same as MODEL_NAMES>}"
echo "=============================================="
echo

# Create plots directory if it doesn't exist
mkdir -p "$PLOTS_DIR"

# Build the extra args string (shared across all jobs)
EXTRA_ARGS=()
if [ ${#TARGET_MODELS[@]} -gt 0 ]; then
  EXTRA_ARGS+=(--target-models "${TARGET_MODELS[@]}")
fi
if [ ${#ENSEMBLE_MODELS[@]} -gt 0 ]; then
  EXTRA_ARGS+=(--ensemble-models "${ENSEMBLE_MODELS[@]}")
fi
if [ "$ONLINE_ACQUISITION" = "true" ]; then
  EXTRA_ARGS+=(--online-acquisition)
else
  EXTRA_ARGS+=(--no-online-acquisition)
fi

# Loop through datasets — submit each as its own sbatch job
for dataset in "${DATASETS[@]}"; do
  echo "Submitting sbatch job for dataset: $dataset"

  # Create per-dataset output directory
  mkdir -p "$PLOTS_DIR/$dataset"

  sbatch \
    --job-name="${dataset}" \
    --partition=nigam-h100 \
    --nodelist=secure-gpu-14 \
    --gres=gpu:1 \
    --mem=100G \
    --time=20:00:00 \
    --ntasks=1 \
    --output="${PLOTS_DIR}/${dataset}/slurm_%j.out" \
    --error="${PLOTS_DIR}/${dataset}/slurm_%j.err" \
    --wrap="
      source \$CONDA_DIR/etc/profile.d/conda.sh
      conda activate pac_judge
      cd SmartSample_local
      python -m src.experiments.variance_selection_analysis \
        --dataset '$dataset' \
        --model-names ${MODEL_NAMES[*]} \
        --data-dir '$DATA_DIR' \
        --plots-dir '$PLOTS_DIR/$dataset' \
        --comparison-mode '$COMPARISON_MODE' \
        --n-bootstrap '$N_BOOTSTRAP' \
        --n-candidates '$N_CANDIDATES' \
        --total-annotations '$TOTAL_ANNOTATIONS' \
        ${EXTRA_ARGS[*]}
    "

  echo "  → Job submitted for: $dataset"
  echo ""
done

echo "=============================================="
echo "All jobs submitted!"
echo "Logs saved to: $PLOTS_DIR/<dataset>/slurm_<jobid>.out/.err"
echo "=============================================="