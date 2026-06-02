#!/bin/bash
#
# Run variance selection analysis across all datasets
# Each dataset is launched as a separate SLURM job.
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
#   --partition PARTITION   SLURM partition to use (default: nigam-h100)
#   --nodelist NODES        SLURM nodelist (optional)
#   --mem MEM               Memory per job (default: 32G)
#   --time TIME             Time limit per job (default: 48:00:00)
#   --cpus-per-task N       CPUs per task (default: 4)
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
N_BOOTSTRAP=40
N_CANDIDATES=20
TOTAL_ANNOTATIONS=300
PLOTS_DIR="results/06_01_small_ens_logging"
DATA_DIR="data/judge_scores"
COMPARISON_MODE="pairwise_average"
ONLINE_ACQUISITION=false  # true → cumulative/incremental selection; false → batch selection
STEP_SIZE=5              # step size for annotation budget levels (e.g. 1, 5, 10)
MAX_BUDGET=50            # maximum annotation budget to evaluate
DATASETS=("medval" "summeval" "mslr" "hanna") #("hanna" "medval" "mslr" "summeval")
MODEL_NAMES=("claude-3.5-sonnet" "gpt-4.1" "gpt-5" "deepseek-r1" "gemini-2.5-pro") # "gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")
# MODEL_NAMES=("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")
TARGET_MODELS=("claude-3.5-sonnet" "gpt-4.1" "deepseek-r1" "gemini-2.5-pro" "gpt-5")  # empty = use MODEL_NAMES
ENSEMBLE_MODELS=("gpt-4o-mini" "meta-llama-Llama-3.1-8B-Instruct" "google-gemma-3-1b-it" "Qwen-Qwen2.5-7B-Instruct")  # empty = use MODEL_NAMES
CONDA_ENV="pac_judge"
SLURM_PARTITION="nigam-h100"
SLURM_NODELIST=""
SLURM_MEM="32G"
SLURM_TIME="48:00:00"
SLURM_CPUS="4"

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
    --partition)
      SLURM_PARTITION="$2"
      shift 2
      ;;
    --nodelist)
      SLURM_NODELIST="$2"
      shift 2
      ;;
    --mem)
      SLURM_MEM="$2"
      shift 2
      ;;
    --time)
      SLURM_TIME="$2"
      shift 2
      ;;
    --cpus-per-task)
      SLURM_CPUS="$2"
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
echo "SLURM_PARTITION:    $SLURM_PARTITION"
echo "SLURM_NODELIST:     ${SLURM_NODELIST:-<none>}"
echo "SLURM_MEM:          $SLURM_MEM"
echo "SLURM_TIME:         $SLURM_TIME"
echo "SLURM_CPUS:         $SLURM_CPUS"
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

JOB_IDS=()
DATASET_NAMES=()

# Loop through datasets — launch each as an independent SLURM job
for dataset in "${DATASETS[@]}"; do
  echo "Submitting SLURM job for dataset: $dataset"

  mkdir -p "$PLOTS_DIR/$dataset"
  LOG_OUT="${PLOTS_DIR}/${dataset}/slurm_run.out"
  LOG_ERR="${PLOTS_DIR}/${dataset}/slurm_run.err"

  # Build SBATCH command
  SBATCH_CMD="sbatch"
  SBATCH_CMD+=" --job-name=gpt5_${dataset}"
  SBATCH_CMD+=" --partition=${SLURM_PARTITION}"
  [ -n "$SLURM_NODELIST" ] && SBATCH_CMD+=" --nodelist=${SLURM_NODELIST}"
  SBATCH_CMD+=" --gres=gpu:0"
  SBATCH_CMD+=" --mem=${SLURM_MEM}"
  SBATCH_CMD+=" --time=${SLURM_TIME}"
  SBATCH_CMD+=" --ntasks=1"
  SBATCH_CMD+=" --cpus-per-task=${SLURM_CPUS}"
  SBATCH_CMD+=" --output=${LOG_OUT}"
  SBATCH_CMD+=" --error=${LOG_ERR}"

  # Create a temporary job script for this dataset
  TEMP_SCRIPT=$(mktemp)
  cat > "$TEMP_SCRIPT" <<EOF
#!/bin/bash

# Activate conda environment
source \$CONDA_DIR/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Change to project directory
cd ${PROJECT_DIR}

# Verify conda environment
echo "Using Conda environment: ${CONDA_ENV}"
echo "Using dataset: ${dataset}"

# Run the variance analysis
python -m src.experiments._2_variance_selection_analysis \\
  --dataset "${dataset}" \\
  --model-names ${MODEL_NAMES[@]} \\
  --data-dir "${DATA_DIR}" \\
  --plots-dir "${PLOTS_DIR}/${dataset}" \\
  --comparison-mode "${COMPARISON_MODE}" \\
  --n-bootstrap ${N_BOOTSTRAP} \\
  --n-candidates ${N_CANDIDATES} \\
  --total-annotations ${TOTAL_ANNOTATIONS} \\
  ${EXTRA_ARGS[@]}
EOF

  # Submit the job and capture job ID
  JOB_OUTPUT=$(eval "$SBATCH_CMD $TEMP_SCRIPT")
  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')

  if [ -n "$JOB_ID" ]; then
    JOB_IDS+=("$JOB_ID")
    DATASET_NAMES+=("$dataset")
    echo "  → Job $JOB_ID submitted for: $dataset  (logs: $LOG_OUT)"
  else
    echo "  ✗ Failed to submit job for: $dataset"
  fi

  # Clean up temp script
  rm -f "$TEMP_SCRIPT"
  echo ""
done

echo "=============================================="
echo "All SLURM jobs submitted!"
echo "=============================================="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor jobs with: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "Cancel all jobs with: scancel $(IFS=' '; echo "${JOB_IDS[*]}")"
echo ""
echo "Logs will be saved to: $PLOTS_DIR/<dataset>/slurm_run.out/.err"
echo "=============================================="