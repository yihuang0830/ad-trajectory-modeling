#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）提交基因第二版的扩大扰动（2026-11-11）：5 个 seed × main（val + test，2 块）+ train（3 块），
# 每块一个 30 分钟的 CPU 作业；最后 geno2_pert_compare.py 等它们全部结束（afterany）。
# 前提：run.py --variant geno2s --save_model 已经把 5 个模型存到 $ZOO_RUN_DIR/models/。
#   bash ris/submit_geno2_pert.sh       # 作业号记到 logs/submit_geno2_pert_<时间>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/submit_geno2_pert_$(date +%Y%m%d_%H%M).txt
IDS=()
for s in 0 1 2 3 4; do
  for spec in "main 2" "train 3"; do
    set -- $spec
    for ((p = 0; p < $2; p++)); do
      a=$(sbatch --parsable --job-name=g2p-$1-$s-$p --partition=general-short --cpus-per-task=8 --mem=32G --time=00:30:00 \
            ris/py.sbatch geno2_perturb.py --seed $s --cohort $1 --part $p --nparts $2)
      IDS+=("$a"); echo "$1 seed $s part $p/$2 $a" | tee -a $LOG
    done
  done
done
dep=$(IFS=:; echo "${IDS[*]}")
c=$(sbatch --parsable --job-name=geno2-pert-compare --partition=general-short --cpus-per-task=4 --mem=32G --time=00:30:00 \
      --dependency=afterany:$dep ris/py.sbatch geno2_pert_compare.py)
echo "compare $c" | tee -a $LOG
