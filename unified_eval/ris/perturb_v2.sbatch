#!/bin/bash
# 第三部分扰动 v2（perturb_v2.py）。CPU 数用 sbatch -c 覆盖，脚本把 SLURM_CPUS_PER_TASK 传给 --cpus
# （direct = torch 线程数；rollout = 按人并行的 worker 数，每个 worker 单线程）。在 unified_eval/ 下提交：
#   sbatch -c 4  ris/perturb_v2.sbatch direct --all
#   sbatch -c 16 ris/perturb_v2.sbatch rollout --name delphi-fullvisit [--arms orig,mmse_b,...] [--n_mc 64]
#   sbatch -c 2 --time 00:30:00 ris/perturb_v2.sbatch aggregate
# 人级产物只在 $ZOO_RUN_DIR/perturb_v2/（0700）；聚合 JSON 在 $ZOO_RUN_DIR/packages/（图在本地由 figures.py 画）。
#SBATCH --job-name=zoo-pv2
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pv2_%j.out
#SBATCH --error=logs/pv2_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
# OMP 单线程：rollout 的并行在进程层（每个 worker torch.set_num_threads(1)）；direct 在脚本里按 --cpus 设 torch 线程
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  $*"
python -u perturb_v2.py "$@" --cpus "${SLURM_CPUS_PER_TASK:-8}"
chmod -R go-rwx "$ZOO_RUN_DIR/perturb_v2" 2>/dev/null || true
