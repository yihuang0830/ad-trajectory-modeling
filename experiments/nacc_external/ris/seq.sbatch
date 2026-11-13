#!/bin/bash
# 串行跑本目录的几条命令（每个参数是一条，引号括起来）：sbatch ris/seq.sbatch "ext_score.py --name x --set rosmap" "..."
#SBATCH --job-name=nacc-ext
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ext_%j.out
#SBATCH --error=logs/ext_%j.err
set -euo pipefail
umask 077                                   # 人级产物（_runs/nacc_ext 下）只给自己读写
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export EVAL_PY=$CONDA_PREFIX/bin/python
cd $BASE/model_zoo/experiments/nacc_external
echo "host $(hostname)  job $SLURM_JOB_ID  args: $*  git: $(cat GIT_REV 2>/dev/null || echo none)"

for cmd in "$@"; do
  echo "== $cmd"
  ${PY:-$E/venvs/ad-baselines/bin/python} -u $cmd
done
