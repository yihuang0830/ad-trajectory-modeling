#!/bin/bash
# 一个 CPU 工人（nacc_scaling_ar.py work）：领还没做的单元做到 27 分钟；网络每个 epoch、模拟每 1000 个站点存一次，下一个工人接着做。
#   cd $AD_BASE/model_zoo/experiments/nacc_scaling_ar && sbatch ris/worker.sbatch [prep | status | collect]
# 多个工人并行：ris/keeper.sh（登录节点上按队列补交）
#SBATCH --job-name=ns3-work
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/work_%j.out
#SBATCH --error=logs/work_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export EVAL_PY=$CONDA_PREFIX/bin/python MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
cd $BASE/model_zoo/experiments/nacc_scaling_ar
echo "host $(hostname)  $(date)  args: ${*:-work}"
if [ "${1:-work}" = work ]; then
  $E/venvs/ad-baselines/bin/python -u nacc_scaling_ar.py work --budget 27
else
  $E/venvs/ad-baselines/bin/python -u nacc_scaling_ar.py "$@"
fi
