#!/bin/bash
#
# Run annotations saved analysis (experiment 8) as a single sbatch job.
#
# Usage:
#   ./scripts/run_annotations_saved.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --folder DIR    Results directory containing per-dataset dataframes (default: results/04_16_new_baselines_small_only)
#
# Examples:
#
#   Run with all defaults
#   ./scripts/run_annotations_saved.sh
#
#   Custom results folder
#   ./scripts/run_annotations_saved.sh --folder results/my_run


# Default values
FOLDER="results/04_22_add_metric_match"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --folder)
      FOLDER="$2"
      shift 2
      ;;
    -h|--help)
      head -20 "$0" | tail -18
      ;;
    *)
      echo "Unknown option: $1"
      ;;
  esac
done

# Print configuration
echo "=============================================="
echo "Annotations Saved Analysis - Configuration"
echo "=============================================="
echo "FOLDER: $FOLDER"
echo "=============================================="
echo

mkdir -p "$FOLDER"

echo "Submitting sbatch job for annotations saved analysis..."

sbatch \
  --job-name="annotations_saved" \
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
    python -m src.experiments._4_annotations_saved \
      --folder '$FOLDER'
  "

echo "  → Job submitted"
echo ""
echo "=============================================="
echo "Logs saved to: $FOLDER/slurm_<jobid>.out/.err"
echo "Results saved to: $FOLDER/annotations_saved/"
echo "=============================================="
