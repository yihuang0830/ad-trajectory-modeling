#!/bin/bash
# 一个 CPU 工人（nacc_scale.py work）：领还没做的单元做到 27 分钟；网络每个 epoch 存一次，下一个工人接着训。
#   cd $AD_BASE/model_zoo/experiments/nacc_scale && sbatch ris/worker.sbatch
# 多个工人并行：ris/keeper.sh（登录节点上按队列补交）
#SBATCH --job-name=ns-work
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
cd $BASE/model_zoo/experiments/nacc_scale
echo "host $(hostname)  $(date)  args: ${*:-work}"
if [ "${1:-work}" = work ]; then
  $E/venvs/ad-baselines/bin/python -u nacc_scale.py work --budget 27
elif [[ $1 == *.py ]]; then
  $E/venvs/ad-baselines/bin/python -u "$@"
else
  $E/venvs/ad-baselines/bin/python -u nacc_scale.py "$@"
fi
