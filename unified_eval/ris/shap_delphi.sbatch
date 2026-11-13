#!/bin/bash
# SHAP（上游 Delphi 的 PartitionExplainer，完整 Owen 展开）：人级 SHAP / 冒烟测试 / 截断误差 / 汇总出图。
#   sbatch --cpus-per-task=6  ris/shap_delphi.sbatch smoke --model delphi-dedup --seed 42 --n 6
#   sbatch --cpus-per-task=10 ris/shap_delphi.sbatch sens  --model delphi-dedup --seed 42 --n 20
#   sbatch --cpus-per-task=16 ris/shap_delphi.sbatch run   --model delphi-dedup --seed 42,43,44
#   sbatch --cpus-per-task=2  ris/shap_delphi.sbatch aggregate
# 在 model_zoo/unified_eval 下提交（日志写 logs/）。进程数默认 = SLURM_CPUS_PER_TASK，每个进程单线程。
#SBATCH --job-name=zoo-shap
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/shap_%j.out
#SBATCH --error=logs/shap_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
ENVD=/path/to/env
source "$ENVD/miniforge3/etc/profile.d/conda.sh"
conda activate ad-projection
# shap 0.49.1（+ slicer、cloudpickle）装在环境外的独立目录里：只加进 PYTHONPATH，不动 conda 环境
export PYTHONPATH="$ENVD/pylib_shap${PYTHONPATH:+:$PYTHONPATH}"
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
umask 077                                   # 人级产物（phi_*.npz）只给自己读
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  args: $*"
python -u shap_delphi.py "$@"
