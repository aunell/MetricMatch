#!/bin/bash
#
#SBATCH --job-name=smartsample_pipeline
#SBATCH --partition=nigam-h100
#SBATCH --nodelist=secure-gpu-14
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --ntasks=1

# ==============================================================================
# SMARTSAMPLE PIPELINE - Steps 1, 2, and 3 Combined
# ==============================================================================
# This script runs all three steps of the SmartSample pipeline:
# 1. Get judge scores
# 2. Run real data experiment
# 3. Generate confidence interval analysis
#
# Simply uncomment the model and datasets you want to use below.
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Simply comment/uncomment the options you want to use
# ==============================================================================

# Conda environment
COND_ENV="pac_judge"

# ---------- DATASET OPTIONS ----------
# Uncomment the datasets you want to run (can select multiple):
DATASETS=(
    "summeval"
    # "hanna"
    # "mslr"
    # "medval"
)

# Or comment out the above and uncomment individual datasets:
# DATASETS=("summeval")
# DATASETS=("hanna")
# DATASETS=("mslr")
# DATASETS=("medval")

# ---------- PRIMARY MODEL OPTIONS ----------
# Uncomment ONE model for judge scoring:

# OpenAI models:
MODEL_NAME="gpt-4o-mini-agg-judge"
# MODEL_NAME="gpt-4o"
# MODEL_NAME="gpt-4o-mini"

# Anthropic models:
# MODEL_NAME="claude-3.5-sonnet-agg-judge"

# Llama models (use full HuggingFace path):
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

# ---------- SECONDARY MODEL OPTIONS (for QBC strategies in step 2) ----------
# Uncomment ONE model for query-by-committee (optional):

# OpenAI models:
# MODEL_NAME_2="gpt-4.1"
# MODEL_NAME_2="gpt-4o"
# MODEL_NAME_2="gpt-4o-mini"

# Anthropic models:
MODEL_NAME_2="claude-3.5-sonnet"

# Llama models (use safe naming - slashes already replaced with dashes):
# MODEL_NAME_2="meta-llama-Llama-3.1-8B-Instruct"
# MODEL_NAME_2="meta-llama-Llama-3.1-70B-Instruct"

# ---------- EXPERIMENT PARAMETERS ----------
SAMPLE_SIZE=350          # For step 1
DATASET_SIZE=300         # For step 2
N_ROLLOUTS=20           # For step 2
N_SUBJECTS=300          # For step 3
N_RATERS=2              # For step 3

# ==============================================================================
# AUTO-CONFIGURATION (Do not edit below this line)
# ==============================================================================

# Auto-infer judge model type from MODEL_NAME
if [[ "$MODEL_NAME" == *"gpt"* ]]; then
    JUDGE_MODEL="openai"
    JUDGE_MODEL_TYPE="openai"
elif [[ "$MODEL_NAME" == *"claude"* ]]; then
    JUDGE_MODEL="anthropic"
    JUDGE_MODEL_TYPE="anthropic"
elif [[ "$MODEL_NAME" == *"llama"* ]] || [[ "$MODEL_NAME" == *"meta-llama"* ]]; then
    JUDGE_MODEL="llama"
    JUDGE_MODEL_TYPE="llama"
else
    echo "ERROR: Could not infer judge model type from MODEL_NAME: $MODEL_NAME"
    exit 1
fi

# Replace / with - for file paths (for HuggingFace model names)
MODEL_NAME_SAFE="${MODEL_NAME//\//-}"

# Activate conda environment
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate "${COND_ENV}"
cd SmartSample_local

# Get date for results directory
DATE=$(date +%Y-%m-%d)

# Verify Conda environment
echo "=============================================================================="
echo "SMARTSAMPLE PIPELINE CONFIGURATION"
echo "=============================================================================="
echo "Conda environment: ${COND_ENV}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Model name: ${MODEL_NAME}"
echo "Safe model name: ${MODEL_NAME_SAFE}"
echo "Secondary model: ${MODEL_NAME_2}"
echo "Datasets to process: ${DATASETS[@]}"
echo "Date: ${DATE}"
echo "=============================================================================="
echo ""

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "=============================================================================="
    echo "PROCESSING DATASET: ${DATASET}"
    echo "=============================================================================="
    echo ""

    # --------------------------------------------------------------------------
    # STEP 1: Get Judge Scores
    # --------------------------------------------------------------------------
    echo "*** STEP 1/3: Getting judge scores for ${DATASET} ***"
    echo ""

    JUDGE_SCORES_FILE="data/judge_scores/${DATASET}/results_${DATASET}_${MODEL_NAME_SAFE}.json"

    python src/experiments/_1_get_judge_scores.py \
        --output_file "$JUDGE_SCORES_FILE" \
        --sample_size "$SAMPLE_SIZE" \
        --dataset "$DATASET" \
        --judge_model "$JUDGE_MODEL" \
        --model_name "$MODEL_NAME"

    if [ $? -ne 0 ]; then
        echo "ERROR: Step 1 failed for dataset ${DATASET}. Skipping to next dataset."
        continue
    fi

    echo ""
    echo "Step 1 completed successfully for ${DATASET}"
    echo ""

    # --------------------------------------------------------------------------
    # STEP 2: Real Data Experiment
    # --------------------------------------------------------------------------
    echo "*** STEP 2/3: Running real data experiment for ${DATASET} ***"
    echo ""

    JUDGE_SCORES_DIR="/share/pi/nigam/users/aunell/SmartSample_local/data/judge_scores/${DATASET}"
    IMAGES_DIR="results/${DATE}-${MODEL_NAME_SAFE}"

    python src/experiments/_2_real_data_experiment.py \
        --judge_model_type "$JUDGE_MODEL_TYPE" \
        --model_name "$MODEL_NAME_SAFE" \
        --model_name_2 "$MODEL_NAME_2" \
        --dataset_size "$DATASET_SIZE" \
        --n_rollouts "$N_ROLLOUTS" \
        --judge_scores_dir "$JUDGE_SCORES_DIR" \
        --images_dir "$IMAGES_DIR" \
        # --no-aggregate

    if [ $? -ne 0 ]; then
        echo "ERROR: Step 2 failed for dataset ${DATASET}. Skipping to next dataset."
        continue
    fi

    echo ""
    echo "Step 2 completed successfully for ${DATASET}"
    echo ""

    # --------------------------------------------------------------------------
    # STEP 3: Confidence Interval Analysis
    # --------------------------------------------------------------------------
    echo "*** STEP 3/3: Generating confidence interval analysis for ${DATASET} ***"
    echo ""

    BASE_DIR="results/${DATE}-${MODEL_NAME_SAFE}-rwe-icc-results-${N_SUBJECTS}"

    python src/experiments/_3_confidence_interval_rde.py \
        --dataset "$DATASET" \
        --n_subjects "$N_SUBJECTS" \
        --n_raters "$N_RATERS" \
        --base_dir "$IMAGES_DIR" \
        --no-aggregate

    if [ $? -ne 0 ]; then
        echo "ERROR: Step 3 failed for dataset ${DATASET}. Continuing to next dataset."
        continue
    fi

    echo ""
    echo "Step 3 completed successfully for ${DATASET}"
    echo ""
    echo "=============================================================================="
    echo "COMPLETED ALL STEPS FOR DATASET: ${DATASET}"
    echo "=============================================================================="
    echo ""
done

echo ""
echo "=============================================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================================="
echo "All datasets processed: ${DATASETS[@]}"
echo "Results saved to: results/${DATE}-${MODEL_NAME_SAFE}-*"
echo "=============================================================================="
