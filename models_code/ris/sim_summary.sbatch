#!/bin/bash
# 所有 rollout 版跑完以后（afterany）：原模型 vs rollout 版的对比表 -> packages/_rollout_versions/summary.{json,md}
#SBATCH --job-name=bl-sim-summary
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/simsum_%j.out
#SBATCH --error=logs/simsum_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs
cd $BASE/model_zoo/models_code
python -u sim_summary.py
