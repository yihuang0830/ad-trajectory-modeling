#!/bin/bash
# 一个 GPU 工人（nacc_scaling_ar.py work --gpu）：先做大的 transformer（M / L），再帮着做别的网络（训练 + 模拟）；27 分钟停，下一个接着做。
#   cd $AD_BASE/model_zoo/experiments/nacc_scaling_ar && sbatch ris/gpu.sbatch
#SBATCH --job-name=ns3-gpu
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export EVAL_PY=$CONDA_PREFIX/bin/python MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
cd $BASE/model_zoo/experiments/nacc_scaling_ar
echo "host $(hostname)  $(date)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$E/venvs/ad-baselines/bin/python -u nacc_scaling_ar.py work --budget 27 --gpu
