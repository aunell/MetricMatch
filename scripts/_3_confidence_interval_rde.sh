# ==============================================================================
# CONFIGURATION - Simply comment/uncomment the options you want to use
# ==============================================================================

# Conda environment
COND_ENV="pac_judge"

# ---------- DATASET OPTIONS ----------
# Uncomment ONE dataset:
# DATASET="summeval"
# DATASET="hanna"
# DATASET="mslr"
DATASET="medval"

# ---------- MODEL OPTIONS ----------
# Uncomment ONE model (must match the model used in _2_real_data_experiment.sh):

# OpenAI models (use safe naming with dashes):
MODEL_NAME="gpt-4.1"
# MODEL_NAME="gpt-4o"
# MODEL_NAME="gpt-4o-mini"

# Anthropic models:
# MODEL_NAME="claude-3.5-sonnet"

# Llama models (use safe naming - slashes already replaced with dashes):
# MODEL_NAME="meta-llama-Llama-3.1-8B-Instruct"
# MODEL_NAME="meta-llama-Llama-3.1-70B-Instruct"

# ---------- EXPERIMENT PARAMETERS ----------
N_SUBJECTS=300
N_RATERS=2

# ==============================================================================
# AUTO-CONFIGURATION (Do not edit below this line)
# ==============================================================================

# Set paths (must match output from _2_real_data_experiment.sh)
DATE=$(date +%Y-%m-%d)
BASE_DIR="results/${DATE}-${MODEL_NAME}-rwe-icc-results-${N_SUBJECTS}"

source $CONDA_DIR/etc/profile.d/conda.sh
conda activate "${COND_ENV}"
cd SmartSample_local

echo "Using dataset: ${DATASET}"
echo "Using model name: ${MODEL_NAME}"
echo "Number of subjects: ${N_SUBJECTS}"
echo "Number of raters: ${N_RATERS}"
echo "Base directory: ${BASE_DIR}"

python src/experiments/_3_confidence_interval_rde.py \
    --dataset "$DATASET" \
    --n_subjects "$N_SUBJECTS" \
    --n_raters "$N_RATERS" \
    --base_dir "$BASE_DIR"