#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）提交"VISTA 先在 NACC 上预训练，再到 ROSMAP 上微调"（2026-11-10）：
#   1. NACC 预训练 5 个 seed（pretrain.py --what seqar；一个 30 分钟的 CPU 作业，5 个 seed 并行；超时了第二个作业接着训）
#   2. 每个 seed 一次微调 + 打分 + 扰动 + 轨迹（run.py --variant naccft --seed k -> 打包盒 transformer-naccft-s<k>）
#   3. vista_nacc_compare.py 等它们全部结束（afterany），和基因实验已有的 transformer-base-s0..s4 比
# 打包盒已经有 baseline.json 的自动跳过；检查点已经有的直接读。
#   bash ris/submit_vista_nacc.sh            # 作业号记到 logs/submit_vista_nacc_<时间>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/submit_vista_nacc_$(date +%Y%m%d_%H%M).txt
SHORT=(--partition=general-short --time=00:30:00 --cpus-per-task=16 --mem=64G)
p1=$(sbatch --parsable --job-name=vn-pre1 "${SHORT[@]}" ris/py.sbatch pretrain.py --what seqar --seeds 0 1 2 3 4)
p2=$(sbatch --parsable --job-name=vn-pre2 "${SHORT[@]}" --dependency=afterany:$p1 \
       ris/py.sbatch pretrain.py --what seqar --seeds 0 1 2 3 4)
echo "pretrain $p1 $p2" | tee -a $LOG
IDS=()
for s in 0 1 2 3 4; do
  a=$(sbatch --parsable --job-name=vn-ft-$s "${SHORT[@]}" --dependency=afterok:$p2 \
        ris/run.sbatch transformer -- --variant naccft --seed $s)
  IDS+=("$a"); echo "naccft $s $a" | tee -a $LOG
done
dep=$(IFS=:; echo "${IDS[*]}")
c=$(sbatch --parsable --job-name=vn-compare --partition=general-short --cpus-per-task=4 --mem=16G --time=00:30:00 \
      --dependency=afterany:$dep ris/py.sbatch vista_nacc_compare.py)
echo "compare $c" | tee -a $LOG
