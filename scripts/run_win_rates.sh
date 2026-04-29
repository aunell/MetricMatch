#!/bin/bash
#
# Run win rates analysis (experiment 7) as a single sbatch job.
#
# Usage:
#   ./scripts/run_win_rates.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --folder DIR         Results directory containing per-dataset dataframes (default: results/04_22_add_metric_match)
#   --track_mse_match    Also track metric_matched_mean_squared_error for the mean_squared_error metric
#
# Examples:
#
#   Run with all defaults
#   ./scripts/run_win_rates.sh
#
#   Custom results folder
#   ./scripts/run_win_rates.sh --folder results/my_run
#
#   Include mean_squared_error metric tracking
#   ./scripts/run_win_rates.sh --track_mse_match


# Default values
FOLDER="results/04_29_batch_large"
TRACK_MSE_MATCH=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --folder)
      FOLDER="$2"
      shift 2
      ;;
    --track_mse_match)
      TRACK_MSE_MATCH=true
      shift
      ;;
    -h|--help)
      head -26 "$0" | tail -24
      ;;
    *)
      echo "Unknown option: $1"
      ;;
  esac
done

# Build optional flags
EXTRA_FLAGS=""
if [ "$TRACK_MSE_MATCH" = true ]; then
  EXTRA_FLAGS="--track_mse_match"
fi

# Print configuration
echo "=============================================="
echo "Win Rates Analysis - Configuration"
echo "=============================================="
echo "FOLDER:           $FOLDER"
echo "TRACK_MSE_MATCH:  $TRACK_MSE_MATCH"
echo "=============================================="
echo

mkdir -p "$FOLDER"

echo "Submitting sbatch job for win rates analysis..."

sbatch \
  --job-name="win_rates" \
  --partition=nigam-h100 \
  --nodelist=secure-gpu-14 \
  --gres=gpu:1 \
  --mem=20G \
  --time=1:00:00 \
  --ntasks=1 \
  --output="${FOLDER}/slurm_%j.out" \
  --error="${FOLDER}/slurm_%j.err" \
  --wrap="
    source \$CONDA_DIR/etc/profile.d/conda.sh
    conda activate pac_judge
    cd /share/pi/nigam/users/aunell/SmartSample_local
    python -m src.experiments._7_win_rates \
      --folder '$FOLDER' \
      $EXTRA_FLAGS
  "

echo "  → Job submitted"
echo ""
echo "=============================================="
echo "Logs saved to: $FOLDER/slurm_<jobid>.out/.err"
echo "Results saved to: $FOLDER/win_rates/"
echo "=============================================="
