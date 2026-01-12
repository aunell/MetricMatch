#!/bin/bash
#
#SBATCH --job-name=medval_gemma
#SBATCH --partition=nigam-h100
#SBATCH --nodelist=secure-gpu-14
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

# ==============================================================================
# CONFIGURATION - Simply comment/uncomment the options you want to use
# ==============================================================================

# Conda environment
COND_ENV="pac_judge"

# ---------- DATASET OPTIONS ----------
# Uncomment ONE dataset:
DATASET="summeval"
# DATASET="hanna"
# DATASET="mslr"
# DATASET="medval"

# ---------- MODEL OPTIONS ----------
# Uncomment ONE model:

# OpenAI models:
# MODEL_NAME="gpt-4.1"
# MODEL_NAME="gpt-4o"
# MODEL_NAME="gpt-4o-mini"
# MODEL_NAME="gpt-5"

# Anthropic models:
# MODEL_NAME="claude-3.5-sonnet"

# Llama models (use full HuggingFace path):
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"

# Qwen models (use full HuggingFace path):
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Gemma models (use full HuggingFace path):
# MODEL_NAME="google/gemma-3-1b-it"

# Gemini models:
MODEL_NAME="gemini-2.5-pro"

# ==============================================================================
# AUTO-CONFIGURATION (Do not edit below this line)
# ==============================================================================

# Auto-infer judge model type from MODEL_NAME
if [[ "$MODEL_NAME" == *"gpt"* ]]; then
    JUDGE_MODEL="openai"
elif [[ "$MODEL_NAME" == *"claude"* ]]; then
    JUDGE_MODEL="anthropic"
elif [[ "$MODEL_NAME" == *"llama"* ]] || [[ "$MODEL_NAME" == *"meta-llama"* ]]; then
    JUDGE_MODEL="llama"
elif [[ "$MODEL_NAME" == *"Qwen"* ]] || [[ "$MODEL_NAME" == *"qwen"* ]]; then
    JUDGE_MODEL="qwen"
elif [[ "$MODEL_NAME" == *"gemma"* ]] || [[ "$MODEL_NAME" == *"Gemma"* ]]; then
    JUDGE_MODEL="gemma"
elif [[ "$MODEL_NAME" == *"gemini"* ]] || [[ "$MODEL_NAME" == *"Gemini"* ]]; then
    JUDGE_MODEL="gemini"
else
    echo "ERROR: Could not infer judge model type from MODEL_NAME: $MODEL_NAME"
    exit 1
fi

# Replace / with - for file paths (for HuggingFace model names)
MODEL_NAME_SAFE="${MODEL_NAME//\//-}"

source $CONDA_DIR/etc/profile.d/conda.sh
conda activate "${COND_ENV}"
cd SmartSample_local

# Verify Conda environment
echo "Using Conda environment: ${COND_ENV}"
echo "Using judge model: ${JUDGE_MODEL}"
echo "Using model name: ${MODEL_NAME}"
echo "Using safe model name for files: ${MODEL_NAME_SAFE}"
python src/experiments/_1_get_judge_scores.py \
    --output_file "data/judge_scores/${DATASET}/results_${DATASET}_${MODEL_NAME_SAFE}.json" \
    --sample_size 350 \
    --dataset "$DATASET" \
    --judge_model "$JUDGE_MODEL" \
    --model_name "$MODEL_NAME"
