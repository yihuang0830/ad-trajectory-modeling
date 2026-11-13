#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）跑 PRS-CS-auto（全局收缩参数 phi 从数据学，不用我们的结局调参），一条染色体一个 30 分钟作业。
# 前提：geno2_hm3.py（22 条）和 geno2_prep.py 跑完。GWAS 样本量取 prep.json 里的有效样本量中位数。
#   bash ris/submit_geno2_prscs.sh [染色体...]      # 不给就 1..22；作业号记到 logs/submit_geno2_prscs_<时间>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
G=../_runs/geno
N=$(python3 -c "import json; print(json.load(open('$G/prscs/prep.json'))['sumstats']['n_gwas_median_neff'])")
LOG=logs/submit_geno2_prscs_$(date +%Y%m%d_%H%M).txt
CHRS=${*:-$(seq 1 22)}
for c in $CHRS; do
  a=$(sbatch --parsable --job-name=prscs-$c --cpus-per-task=4 --mem=24G ris/geno2.sbatch \
        $G/ref/PRScs/PRScs.py --ref_dir=$G/ref/ldblk_1kg_eur --bim_prefix=$G/hm3/target --sst_file=$G/prscs/sst.txt \
        --n_gwas=$N --out_dir=$G/prscs/bell --chrom=$c --seed=1)
  echo "chr$c $a" | tee -a $LOG
done
