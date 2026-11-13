#!/bin/bash
# 基因第二版的作业（geno 环境：PRS-CS 要的 numpy 1.26 / scipy 1.11 / h5py 3.10，外加 htslib / bcftools）：
#   sbatch ris/geno2.sbatch geno2_hm3.py --chr 22
#   sbatch ris/geno2.sbatch ../_runs/geno/ref/PRScs/PRScs.py --ref_dir=... --chrom=22 ...
#SBATCH --job-name=geno2
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/geno2_%j.out
#SBATCH --error=logs/geno2_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate geno
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
cd $BASE/model_zoo/models_code
echo "host $(hostname)  args: $*"
python -u "$@"
