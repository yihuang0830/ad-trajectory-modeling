#!/bin/bash
#SBATCH --job-name=zoo-timing
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/timing_%j.out
#SBATCH --error=logs/timing_%j.err
# 时间预测（timing.py）：全部打包盒 -> packages/<名字>/timing.json；给名字就只算这几个
#   sbatch ris/timing.sbatch [名字 ...]
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
umask 077
cd "$BASE/model_zoo/unified_eval"
if [ $# -gt 0 ]; then python -u timing.py --name "$@"; else python -u timing.py; fi
