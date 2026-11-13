#!/bin/bash
# 训练 + 评估模型（Delphi 以外）：sbatch ris/run.sbatch <名字> [<名字> ...] [-- run.py 的其余参数]
#   GPU：sbatch --partition=general-gpu --gres=gpu:1 ris/run.sbatch gru grud ...
#SBATCH --job-name=zoo-bl
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bl_%j.out
#SBATCH --error=logs/bl_%j.err
set -euo pipefail
umask 077                                   # 人级产物（_runs 下）只给自己读写
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore MPLBACKEND=Agg
PY=$E/venvs/ad-baselines/bin/python
export EVAL_PY=$CONDA_PREFIX/bin/python
cd $BASE/model_zoo/models_code
echo "host $(hostname)  args: $*"
models=(); rest=()
seen=0
for a in "$@"; do
  if [ "$a" = "--" ]; then seen=1; continue; fi
  if [ $seen = 0 ]; then models+=("$a"); else rest+=("$a"); fi
done
$PY -u run.py --models "${models[@]}" "${rest[@]}"
