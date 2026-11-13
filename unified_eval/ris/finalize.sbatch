#!/bin/bash
# 收尾（README "复现"第 7 步）：第二部分的汇总（fidelity.json、perturb.json）-> 敏感性 -> val 对 test -> MC 颗粒度
# 前提：run_delphi_all、traj_truth + perturb_reference、每个 Delphi 模型的 rollout_cpu / rollout_gpu、
#       11 个能生成轨迹的模型的 rollout_extra（不带参数和带 --query_start 各一次）都跑完了
#   sbatch ris/finalize.sbatch
#SBATCH --job-name=zoo-final
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
cd "$BASE/model_zoo/unified_eval"
# 模型清单直接取注册表，加模型时不用改这里
for n in $(python -c "from registry import DELPHI; print(' '.join(m[0] for m in DELPHI))"); do
  python -u rollout_metrics.py --name "$n" --scores "$ZOO_RUN_DIR/scores/${n}__s42.csv"
done
python -u sensitivity.py
python -u val_vs_test.py
python -u mc_granularity.py
chmod -R go-rwx "$ZOO_RUN_DIR"
