#!/bin/bash
#
# Run meta_correlation_analysis.py as a single sbatch job.
#
# Usage:
#   ./scripts/run_meta_correlation_analysis.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --input-csv FILE        Path to predictor_records.csv (default: results-dir/predictor_scatter/predictor_records.csv)
#   --results-dir DIR       Base results directory, used to derive --input-csv default
#                           (default: /share/pi/nigam/users/aunell/SmartSample_local/results/03_25_new_var_weight)
#   --output-dir DIR        Where to save output CSVs (default: same directory as --input-csv)
#   --human-agreement FILE  Path to human_agreement.csv
#                           (default: results/03_25_human_aggreement/human_agreement.csv)
#
# Examples:
#
#   Run with all defaults
#   ./scripts/run_meta_correlation_analysis.sh
#
#   Point at a specific predictor_records.csv
#   ./scripts/run_meta_correlation_analysis.sh \
#     --input-csv results/03_25/predictor_scatter/predictor_records.csv \
#     --output-dir results/03_25/predictor_scatter

set -e

# Defaults
RESULTS_DIR="/share/pi/nigam/users/aunell/SmartSample_local/results/03_27_deepseek_gemini"
INPUT_CSV=""
OUTPUT_DIR="/share/pi/nigam/users/aunell/SmartSample_local/results/03_27_deepseek_gemini_meta"
HUMAN_AGREEMENT="/share/pi/nigam/users/aunell/SmartSample_local/results/03_25_human_aggreement/human_agreement.csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --input-csv)
      INPUT_CSV="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --human-agreement)
      HUMAN_AGREEMENT="$2"
      shift 2
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

# Derive input CSV from results dir if not specified
if [ -z "$INPUT_CSV" ]; then
  INPUT_CSV="${RESULTS_DIR}/predictor_scatter/predictor_records.csv"
fi

# Derive output dir from input CSV if not specified
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$(dirname "${INPUT_CSV}")"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Meta Correlation Analysis - Configuration"
echo "=============================================="
echo "INPUT_CSV:       $INPUT_CSV"
echo "OUTPUT_DIR:      $OUTPUT_DIR"
echo "HUMAN_AGREEMENT: $HUMAN_AGREEMENT"
echo "=============================================="
echo

sbatch \
  --job-name="meta_corr_analysis" \
  --partition=nigam-h100 \
  --nodelist=secure-gpu-14 \
  --gres=gpu:1 \
  --mem=20G \
  --time=1:00:00 \
  --ntasks=1 \
  --output="${OUTPUT_DIR}/slurm_%j.out" \
  --error="${OUTPUT_DIR}/slurm_%j.err" \
  --wrap="
    source \$CONDA_DIR/etc/profile.d/conda.sh
    conda activate pac_judge
    cd SmartSample_local
    python src/experiments/meta_correlation_analysis.py \
      --input-csv '${INPUT_CSV}' \
      --output-dir '${OUTPUT_DIR}' \
      --human-agreement '${HUMAN_AGREEMENT}'
  "

echo "Job submitted. Logs → ${OUTPUT_DIR}/slurm_<jobid>.out/.err"
