#!/bin/bash
# Delphi 的 NACC 预训练 / ROSMAP 微调（nacc_delphi.py）。5 个 seed 在同一块 GPU 上并行（模型只有 0.69M 参数，一个进程用不满卡），
# 这样 general-short 的一个 30 分钟 GPU 作业就够了。
#   sbatch -p general-short --time=00:30:00 ris/delphi_nacc.sbatch pre   # NACC 上从零训练 -> delphi/Delphi-NACC-nojitter[-sNN]/
#   sbatch -p general-short --time=00:30:00 ris/delphi_nacc.sbatch ft    # 从 NACC 的权重开始，原配方在 ROSMAP 上微调 -> Delphi-ROSMAP-nojitter-nacc[-sNN]/
# 两步都是 nojitter 配方（--lifestyle_token_max=2），和对照 delphi-dedup-nojitter 一样。
#SBATCH --job-name=delphi-nacc
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/delphi_nacc_%j.out
#SBATCH --error=logs/delphi_nacc_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2
STAGE=$1
cd $BASE/delphi
mkdir -p logs
echo "host $(hostname)  stage $STAGE  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for s in 42 43 44 45 46; do
  if [ "$STAGE" = pre ]; then
    d=Delphi-NACC-nojitter; [ $s = 42 ] || d=$d-s$s
    [ -e $d/ckpt.pt ] && { echo "$d done already"; continue; }
    python -u train.py config/train_delphi_nacc.py --device=cuda --out_dir=$d --seed=$s --lifestyle_token_max=2 > logs/nacc_pre_$s.log 2>&1 &
  else
    d=Delphi-ROSMAP-nojitter-nacc; [ $s = 42 ] || d=$d-s$s
    python -c "import torch,sys; sys.exit(0 if torch.load('$d/ckpt.pt',weights_only=False)['iter_num']==0 else 1)" \
      || { echo "$d already fine-tuned (iter_num > 0), skipping"; continue; }
    python -u train.py config/train_delphi_rosmap.py --device=cuda --out_dir=$d --seed=$s --lifestyle_token_max=2 \
      --init_from=resume > logs/nacc_ft_$s.log 2>&1 &
  fi
done
wait
for s in 42 43 44 45 46; do echo "== seed $s"; tail -2 logs/nacc_${STAGE}_$s.log; grep "saving checkpoint" logs/nacc_${STAGE}_$s.log | tail -1 || true; done
