#!/bin/bash
#
# Run predictor scatter analysis across datasets as a single sbatch job.
#
# Usage:
#   ./scripts/run_predictor_scatter.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --results-dir DIR       Base results directory from rollouts (default: results/03_13_old_version_corr_analysis)
#   --datasets D1 D2 ...    Datasets to include (default: medval summeval hanna mslr)
#   --output-dir DIR        Where to save scatter plots (default: results-dir/predictor_scatter)
#   --n-samples N           Bootstrap samples for correlation predictor (default: 500)
#   --sample-size N         Items per bootstrap sample (default: 10)
#   --human-agreement FILE  Path to human_agreement.csv (default: results/03_25_human_aggreement/human_agreement.csv)
#
# Examples:
#
#   Run with all defaults
#   ./scripts/run_predictor_scatter.sh
#
#   Run on a specific results dir with two datasets
#   ./scripts/run_predictor_scatter.sh --results-dir results/03_13 --datasets medval hanna

set -e

# Defaults
RESULTS_DIR="/share/pi/nigam/users/aunell/SmartSample_local/results/04_07_kendall_spearman"
DATASETS=("medval" "summeval" "hanna" "mslr")
OUTPUT_DIR=""
N_SAMPLES=500
SAMPLE_SIZE=10
HUMAN_AGREEMENT="/share/pi/nigam/users/aunell/SmartSample_local/results/03_25_human_aggreement/human_agreement.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --results-dir)
      RESULTS_DIR="$2"
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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --n-samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --human-agreement)
      HUMAN_AGREEMENT="$2"
      shift 2
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

# Default output dir
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${RESULTS_DIR}/predictor_scatter"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Predictor Scatter Analysis - Configuration"
echo "=============================================="
echo "RESULTS_DIR:  $RESULTS_DIR"
echo "DATASETS:     ${DATASETS[*]}"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "N_SAMPLES:    $N_SAMPLES"
echo "SAMPLE_SIZE:  $SAMPLE_SIZE"
echo "HUMAN_AGREEMENT: $HUMAN_AGREEMENT"
echo "=============================================="
echo

sbatch \
  --job-name="predictor_scatter" \
  --partition=nigam-h100 \
  --nodelist=secure-gpu-14 \
  --gres=gpu:1 \
  --mem=50G \
  --time=4:00:00 \
  --ntasks=1 \
  --output="${OUTPUT_DIR}/slurm_%j.out" \
  --error="${OUTPUT_DIR}/slurm_%j.err" \
  --wrap="
    source \$CONDA_DIR/etc/profile.d/conda.sh
    conda activate pac_judge
    cd SmartSample_local
    python src/experiments/predictor_scatter.py \
      --results-dir '${RESULTS_DIR}' \
      --datasets ${DATASETS[*]} \
      --output-dir '${OUTPUT_DIR}' \
      --n-samples '${N_SAMPLES}' \
      --sample-size '${SAMPLE_SIZE}' \
      --human-agreement '${HUMAN_AGREEMENT}'
  "

echo "Job submitted. Logs → ${OUTPUT_DIR}/slurm_<jobid>.out/.err"
