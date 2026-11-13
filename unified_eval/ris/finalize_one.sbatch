#!/bin/bash
#SBATCH --job-name=zoo-final1
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/final1_%j.out
#SBATCH --error=logs/final1_%j.err
# 给新加的模型补收尾（finalize.sbatch 的计算部分，不重跑别的模型的第二部分汇总，也不改目录权限）：
#   第二部分的汇总（只算给出的模型）-> 敏感性 -> val 对 test -> MC 颗粒度（后三个按注册表全部重算，结果对已有模型逐位不变）
#   sbatch ris/finalize_one.sbatch delphi-dedup-nojitter [别的名字 ...]
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
umask 077
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  args: $*"
for n in "$@"; do
  python -u rollout_metrics.py --name "$n" --scores "$ZOO_RUN_DIR/scores/${n}__s42.csv"
done
python -u sensitivity.py
python -u val_vs_test.py
python -u mc_granularity.py
