#!/bin/bash
# 统一评估：注册表里全部 Delphi ckpt -> 打分 -> 指标 + 图。
#   sbatch ris/run_delphi_all.sbatch [--only name1,name2]
#SBATCH --job-name=zoo-eval
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/zoo_%j.out
#SBATCH --error=logs/zoo_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  args: $*"
python -u run_delphi_all.py --device cpu "$@"
