#!/bin/bash
# 第三部分（疾病分期转换）：真值 -> 全部 Delphi ckpt 打分 -> 指标（transitions.json）。
#   sbatch ris/run_trans_all.sbatch [--only name1,name2]
# 并行跑多个 --only 子集：先投一个 `--truth_only`（重建真值并打指纹），其余子集带 `--skip_truth` 依赖它
# （--dependency=afterok:<那个作业>）。直接跑 transition_truth.py 不打指纹，后面的作业会以为真值过期、同时改写同一个 CSV。
#SBATCH --job-name=zoo-trans
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/trans_%j.out
#SBATCH --error=logs/trans_%j.err
set -euo pipefail
umask 077                                   # 人级产物（_runs 下）只给自己读写
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
cd "$BASE/model_zoo/unified_eval"
echo "host $(hostname)  args: $*"
python -u run_trans_all.py --device cpu "$@"
