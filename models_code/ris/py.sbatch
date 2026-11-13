#!/bin/bash
# models 下跑一个 python 脚本（基线环境）：sbatch -p general-short --time=00:30:00 ris/py.sbatch <脚本> [参数...]
#SBATCH --job-name=bl-py
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/py_%j.out
#SBATCH --error=logs/py_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore MPLBACKEND=Agg
export EVAL_PY=$CONDA_PREFIX/bin/python
cd $BASE/model_zoo/models_code
echo "host $(hostname)  args: $*"
$E/venvs/ad-baselines/bin/python -u "$@"
