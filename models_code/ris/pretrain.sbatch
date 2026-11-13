#!/bin/bash
# NACC 预训练（检查点 -> $ZOO_RUN_DIR/nacc/ckpt/）：sbatch ris/pretrain.sbatch --what nm0 [nm2 nm4 seq] [--seeds 5]
#SBATCH --job-name=nacc-pre
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pre_%j.out
#SBATCH --error=logs/pre_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore
cd $BASE/model_zoo/models_code
echo "host $(hostname)  args: $*"
$E/venvs/ad-baselines/bin/python -u pretrain.py "$@"
chmod -R go-rwx $ZOO_RUN_DIR/nacc
