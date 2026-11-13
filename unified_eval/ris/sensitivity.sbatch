#!/bin/bash
#SBATCH --job-name=zoo-sens
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sens_%j.out
#SBATCH --error=logs/sens_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=8
cd "$BASE/model_zoo/unified_eval"
python -u sensitivity.py
python -u val_vs_test.py
chmod -R go-rwx "$ZOO_RUN_DIR"
