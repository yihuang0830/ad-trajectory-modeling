#!/bin/bash
# 第二轮审查后的补跑（一个能生成轨迹的 Delphi 模型）：第二部分模拟逼真度的逐年读出 + 观测一侧 + 卒中反号的机制检验
#   sbatch ris/rollout_extra.sbatch --name delphi-nodedup
#SBATCH --job-name=zoo-extra
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/extra_%j.out
#SBATCH --error=logs/extra_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=4
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  $*"
python -u rollout_extra.py --device cpu "$@"
chmod -R go-rwx "$ZOO_RUN_DIR"
