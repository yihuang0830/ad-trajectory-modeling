#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）提交基因第二版：VISTA（transformer）× geno2 / geno2nx × seed 0..4，每次运行一个 30 分钟 CPU 作业；
# 最后 geno2_compare.py 等它们全部结束（afterany）。对照 gflag / geno 用 2026-11-05 已有的打包盒。
# 前提：geno2_tier1.py、geno2_score.py、input_data_geno2.py 都跑完。
#   bash ris/submit_geno2.sh            # 作业号记到 logs/submit_geno2_<时间>.txt
#   bash ris/submit_geno2.sh geno2i     # 只跑给定的变体（2026-11-11：geno2i = 交互扰动，同 seed 重训）
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/submit_geno2_$(date +%Y%m%d_%H%M).txt
IDS=()
for v in ${*:-geno2 geno2nx}; do
  for s in 0 1 2 3 4; do
    a=$(sbatch --parsable --job-name=g2-$v-$s --partition=general-short --cpus-per-task=16 --mem=64G \
          --time=00:30:00 ris/run.sbatch transformer -- --variant $v --seed $s)
    IDS+=("$a"); echo "transformer $v $s $a" | tee -a $LOG
  done
done
dep=$(IFS=:; echo "${IDS[*]}")
c=$(sbatch --parsable --job-name=geno2-compare --partition=general-short --cpus-per-task=4 --mem=16G --time=00:30:00 \
      --dependency=afterany:$dep ris/py.sbatch geno2_compare.py)
echo "compare $c" | tee -a $LOG
