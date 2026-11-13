#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）提交 G×E 第二轮（2026-11-12）：5 个 seed × 5 个合并症，每个一个 30 分钟的 CPU 作业
# （24 个基因臂 × 有无合并症 = 48 次模拟，约 5 分钟）；最后 geno2_gxe_compare.py 等它们全部结束（afterany）。
# 前提：run.py --variant geno2s --save_model 已经把 5 个模型存到 $ZOO_RUN_DIR/models/。
#   bash ris/submit_geno2_gxe.sh       # 作业号记到 logs/submit_geno2_gxe_<时间>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/submit_geno2_gxe_$(date +%Y%m%d_%H%M).txt
IDS=()
for s in 0 1 2 3 4; do
  for at in htn dm heart stroke depression; do
    a=$(sbatch --parsable --job-name=gxe-$s-$at --partition=general-short --cpus-per-task=8 --mem=32G --time=00:30:00 \
          ris/py.sbatch geno2_gxe.py --seed $s --comorb $at)
    IDS+=("$a"); echo "seed $s $at $a" | tee -a $LOG
  done
done
dep=$(IFS=:; echo "${IDS[*]}")
c=$(sbatch --parsable --job-name=gxe-compare --partition=general-short --cpus-per-task=4 --mem=32G --time=00:30:00 \
      --dependency=afterany:$dep ris/py.sbatch geno2_gxe_compare.py)
echo "compare $c" | tee -a $LOG
