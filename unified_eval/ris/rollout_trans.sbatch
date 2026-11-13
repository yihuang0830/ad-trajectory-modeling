#!/bin/bash
# 旧 figure 2 的轨迹评估（从第一次访视起 10 年、查询 token 起步）：一个能生成轨迹的 Delphi 模型，生成 + 汇总在同一个作业里。
#   sbatch ris/rollout_trans.sbatch --name delphi-dedup                          # 完整：val 全部人 x 100 条
#   sbatch --cpus-per-task=8 ris/rollout_trans.sbatch --name delphi-dedup --limit 20 --n_mc 20   # 冒烟
#   sbatch ris/rollout_trans.sbatch --write_na                                   # 7 个不能生成的打包盒写 applicable=false
# 并行：按人分给 SLURM_CPUS_PER_TASK 个进程（spawn，每个进程只加载一次模型、单线程）。
#SBATCH --job-name=zoo-rtrans
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rtrans_%j.out
#SBATCH --error=logs/rtrans_%j.err
set -euo pipefail
umask 077                       # 人级产物只给自己看（_runs 本身是 0700）
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
# 每个进程单线程：并行在人这一层；numpy / pandas 先于 torch 导入（rollout_trans.py 顶部），配合下面的 libstdc++
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-1}  $*"
python -u rollout_trans.py --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
chmod -R go-rwx "$ZOO_RUN_DIR/rollout_trans" 2>/dev/null || true
