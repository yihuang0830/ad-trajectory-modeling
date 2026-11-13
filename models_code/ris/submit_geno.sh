#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）提交基因实验：3 个结构 × base / gflag / geno × seed 0..4，每次运行一个 30 分钟的 CPU 作业
# （GPU 队列太长；CPU 上 transformer 约 13 分钟、minimalrnn 约 5、xgb-l2c-sim 约 9）。打包盒已经有 baseline.json 的自动跳过。
# 最后一个 geno_compare.py 等它们全部结束（afterany）。
#   bash ris/submit_geno.sh            # 作业号记到 logs/submit_geno_<时间>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/submit_geno_$(date +%Y%m%d_%H%M).txt
IDS=()
for m in transformer minimalrnn xgb-l2c-sim; do
  for v in base gflag geno; do
    for s in 0 1 2 3 4; do
      a=$(sbatch --parsable --job-name=geno-$m-$v-$s --partition=general-short --cpus-per-task=16 --mem=64G \
            --time=00:30:00 ris/run.sbatch $m -- --variant $v --seed $s)
      IDS+=("$a"); echo "$m $v $s $a" | tee -a $LOG
    done
  done
done
dep=$(IFS=:; echo "${IDS[*]}")
c=$(sbatch --parsable --job-name=geno-compare --partition=general-short --cpus-per-task=4 --mem=16G --time=00:30:00 \
      --dependency=afterany:$dep ris/py.sbatch geno_compare.py)
echo "compare $c" | tee -a $LOG
