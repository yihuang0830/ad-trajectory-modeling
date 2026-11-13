#!/bin/bash
# 第二部分：一个 Delphi 模型的轨迹生成 + 扰动（CPU），再汇总（在同一个作业里；汇总时画的旧图只留在 RIS）。
#   sbatch ris/rollout_cpu.sbatch --name delphi-nodedup [run_rollout_delphi.py 的其余参数]
#SBATCH --job-name=zoo-rcpu
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rcpu_%j.out
#SBATCH --error=logs/rcpu_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=4
cd "$BASE/model_zoo/unified_eval"
# 从参数里取模型名（--name X 或 --name=X，放在哪个位置都行）
NAME="" prev=""
for a in "$@"; do
  if [ "$prev" = "--name" ]; then NAME="$a"; fi
  case "$a" in --name=*) NAME="${a#--name=}" ;; esac
  prev="$a"
done
[ -n "$NAME" ] || { echo "用法：sbatch ris/rollout_cpu.sbatch --name <模型名>" >&2; exit 2; }
echo "host $(hostname)  name $NAME  (cpu)"
python -u run_rollout_delphi.py --device cpu "$@"
python -u rollout_metrics.py --name "$NAME" --scores "$ZOO_RUN_DIR/scores/${NAME}__s42.csv"
