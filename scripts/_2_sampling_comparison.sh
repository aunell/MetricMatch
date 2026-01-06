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

# ---------- PRIMARY MODEL OPTIONS ----------
# Uncomment ONE model (this will be the primary model for analysis):

# OpenAI models (use safe naming with dashes):
# MODEL_NAME="gpt-4.1"
# MODEL_NAME="gpt-4o"
# MODEL_NAME="gpt-4o-mini"

# Anthropic models:
# MODEL_NAME="claude-3.5-sonnet"

# Llama models (use safe naming - slashes already replaced with dashes):
MODEL_NAME="meta-llama-Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-llama-Llama-3.1-70B-Instruct"

# ---------- SECONDARY MODEL OPTIONS (for QBC strategies) ----------
# Uncomment ONE model for query-by-committee:

# OpenAI models:
# MODEL_NAME_2="gpt-4.1"
# MODEL_NAME_2="gpt-4o"
# MODEL_NAME_2="gpt-4o-mini"

# Anthropic models:
# MODEL_NAME_2="claude-3.5-sonnet"

# Llama models:
# MODEL_NAME_2="meta-llama-Llama-3.1-8B-Instruct"

# ==============================================================================
# AUTO-CONFIGURATION (Do not edit below this line)
# ==============================================================================

# Auto-infer judge model type from MODEL_NAME
if [[ "$MODEL_NAME" == *"gpt"* ]]; then
    JUDGE_MODEL_TYPE="openai"
elif [[ "$MODEL_NAME" == *"claude"* ]]; then
    JUDGE_MODEL_TYPE="anthropic"
elif [[ "$MODEL_NAME" == *"llama"* ]] || [[ "$MODEL_NAME" == *"meta-llama"* ]]; then
    JUDGE_MODEL_TYPE="llama"
else
    echo "ERROR: Could not infer judge model type from MODEL_NAME: $MODEL_NAME"
    exit 1
fi

# Set paths
JUDGE_SCORES_DIR="/share/pi/nigam/users/aunell/SmartSample_local/data/judge_scores/${DATASET}"
DATE=$(date +%Y-%m-%d)
IMAGES_DIR="results/${DATE}-${MODEL_NAME}-rwe-icc-results-300"

source $CONDA_DIR/etc/profile.d/conda.sh
conda activate "${COND_ENV}"
cd SmartSample_local

echo "Using judge model type: ${JUDGE_MODEL_TYPE}"
echo "Using model name: ${MODEL_NAME}"
echo "Using second model name: ${MODEL_NAME_2}"
echo "Using judge scores directory: ${JUDGE_SCORES_DIR}"
echo "Using images directory: ${IMAGES_DIR}"

python src/experiments/_2_sampling_comparison.py \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --model_name_2 "$MODEL_NAME_2" \
    --dataset_size 300 \
    --n_rollouts 20 \
    --judge_scores_dir "$JUDGE_SCORES_DIR" \
    --images_dir "$IMAGES_DIR"