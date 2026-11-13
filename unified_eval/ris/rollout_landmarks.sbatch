#!/bin/bash
# rollout 版 Delphi 打分表（rollout_landmarks.py）：每个站点生成 100 条 10 年轨迹，数首次越线当风险，再评估。
#   sbatch ris/rollout_landmarks.sbatch --name delphi-dedup
#   sbatch --cpus-per-task=8 ris/rollout_landmarks.sbatch --name delphi-dedup --limit 40 --n_mc 20   # 冒烟
# 30 分钟跑不完就续跑：每 100 个站点存一次，同一条命令再交一次（或 --dependency=afterany:<上一个>）。
#SBATCH --job-name=zoo-rlm
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rlm_%j.out
#SBATCH --error=logs/rlm_%j.err
set -euo pipefail
umask 077                       # 人级产物只给自己看（_runs 本身是 0700）
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONWARNINGS=ignore
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
cd "$BASE/model_zoo/unified_eval"
# 锁：Slurm 控制器不稳时 sbatch 超时其实已经交上，重试会交重复；锁里记作业号，那个作业还在跑就直接退出（sacct 走数据库，控制器挂了也能查）
L="$ZOO_RUN_DIR/rollout_landmarks/.lock"
mkdir -p "$(dirname "$L")"
if [ -e "$L" ] && [ "$(sacct -X -n -j "$(cat "$L")" -o State 2>/dev/null | tr -d ' ')" = RUNNING ]; then
  echo "job $(cat "$L") is already running; exit"; exit 0
fi
echo "$SLURM_JOB_ID" > "$L"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-1}  $*"
python -u rollout_landmarks.py --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
chmod -R go-rwx "$ZOO_RUN_DIR/rollout_landmarks" 2>/dev/null || true
