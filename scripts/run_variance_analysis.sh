#!/bin/bash
#
# Run variance selection analysis across all datasets
# Each dataset is launched as a separate background process for local parallelism.
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
#   --step-size N           Step size for annotation budget levels (default: 5).
#                           E.g. --step-size 1 tests every budget [5,6,7,...,max-budget].
#   --max-budget N          Maximum annotation budget to evaluate (default: 50).
#   --conda-env ENV         Conda environment name (default: pac_judge)
#   --max-parallel N        Max concurrent dataset processes (default: 4).
#                           Use --max-parallel 2 when running two scripts simultaneously.
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

# set -e  # Exit on error

# Default values
N_BOOTSTRAP=100
N_CANDIDATES=20
TOTAL_ANNOTATIONS=300
PLOTS_DIR="results/04_16"
DATA_DIR="data/judge_scores"
COMPARISON_MODE="pairwise_average"
ONLINE_ACQUISITION=true  # true → cumulative/incremental selection; false → batch selection
STEP_SIZE=5              # step size for annotation budget levels (e.g. 1, 5, 10)
MAX_BUDGET=50            # maximum annotation budget to evaluate
DATASETS=("medval" "summeval" "mslr" "hanna") #("hanna" "medval" "mslr" "summeval")
MODEL_NAMES=("claude-3.5-sonnet" "gpt-4.1" "gpt-5" "deepseek-r1" "gemini-2.5-pro") # "gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")
# MODEL_NAMES=("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")
TARGET_MODELS=() #("claude-3.5-sonnet" "gpt-4.1" "gpt-5" "deepseek-r1" "gemini-2.5-pro")  # empty = use MODEL_NAMES
ENSEMBLE_MODELS=() #("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")  # empty = use MODEL_NAMES
CONDA_ENV="pac_judge"
MAX_PARALLEL=4  # 12 cores / ~3 cores per dataset; safe for one script invocation

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
    --step-size)
      STEP_SIZE="$2"
      shift 2
      ;;
    --max-budget)
      MAX_BUDGET="$2"
      shift 2
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
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    -h|--help)
      head -30 "$0" | tail -28
      # exit 0
      ;;
    *)
      echo "Unknown option: $1"
      # exit 1
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
echo "STEP_SIZE:          $STEP_SIZE"
echo "MAX_BUDGET:         $MAX_BUDGET"
echo "DATASETS:           ${DATASETS[*]}"
echo "MODEL_NAMES:        ${MODEL_NAMES[*]}"
echo "TARGET_MODELS:      ${TARGET_MODELS[*]:-<same as MODEL_NAMES>}"
echo "ENSEMBLE_MODELS:    ${ENSEMBLE_MODELS[*]:-<same as MODEL_NAMES>}"
echo "CONDA_ENV:          $CONDA_ENV"
echo "MAX_PARALLEL:       $MAX_PARALLEL"
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
EXTRA_ARGS+=(--step-size "$STEP_SIZE")
EXTRA_ARGS+=(--max-budget "$MAX_BUDGET")

# Resolve the project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PIDS=()
DATASET_NAMES=()

# Wait until fewer than MAX_PARALLEL background jobs are running.
# Uses polling because macOS bash 3.2 lacks `wait -n`.
wait_for_slot() {
  while true; do
    local running=0
    for p in "${PIDS[@]}"; do
      kill -0 "$p" 2>/dev/null && running=$((running + 1))
    done
    [ "$running" -lt "$MAX_PARALLEL" ] && return
    sleep 2
  done
}

# Loop through datasets — launch each as an independent background process
for dataset in "${DATASETS[@]}"; do
  wait_for_slot

  echo "Launching background process for dataset: $dataset"

  mkdir -p "$PLOTS_DIR/$dataset"
  LOG_OUT="${PLOTS_DIR}/${dataset}/local_run.out"
  LOG_ERR="${PLOTS_DIR}/${dataset}/local_run.err"

  conda run -n "$CONDA_ENV" --no-capture-output \
    python -m src.experiments._2_variance_selection_analysis \
      --dataset "$dataset" \
      --model-names "${MODEL_NAMES[@]}" \
      --data-dir "$DATA_DIR" \
      --plots-dir "$PLOTS_DIR/$dataset" \
      --comparison-mode "$COMPARISON_MODE" \
      --n-bootstrap "$N_BOOTSTRAP" \
      --n-candidates "$N_CANDIDATES" \
      --total-annotations "$TOTAL_ANNOTATIONS" \
      "${EXTRA_ARGS[@]}" \
    >"$LOG_OUT" 2>"$LOG_ERR" &

  PIDS+=($!)
  DATASET_NAMES+=("$dataset")
  echo "  → PID $! started for: $dataset  (logs: $LOG_OUT)"
  echo ""
done

echo "=============================================="
echo "All processes launched. Waiting for completion..."
echo "=============================================="
echo ""

# Wait for each process and report exit status
FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  ds="${DATASET_NAMES[$i]}"
  if wait "$pid"; then
    echo "  ✓ $ds completed (PID $pid)"
  else
    echo "  ✗ $ds FAILED (PID $pid) — see ${PLOTS_DIR}/${ds}/local_run.err"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
  echo "All datasets completed successfully!"
else
  echo "$FAILED dataset(s) failed. Check logs in $PLOTS_DIR/<dataset>/local_run.err"
fi
echo "Logs saved to: $PLOTS_DIR/<dataset>/local_run.out/.err"
echo "=============================================="
# exit "$FAILED"