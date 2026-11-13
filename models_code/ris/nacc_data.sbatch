#!/bin/bash
# NACC 预训练数据：sbatch ris/nacc_data.sbatch [--check_rosmap]   -> $ZOO_RUN_DIR/nacc/nacc_data.npz（人级，只留在 RIS）
#SBATCH --job-name=nacc-data
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nacc_data_%j.out
#SBATCH --error=logs/nacc_data_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs PYTHONWARNINGS=ignore
cd $BASE/model_zoo/models_code
python -u nacc_data.py "$@"
