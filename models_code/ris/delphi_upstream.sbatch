#!/bin/bash
# 上游原版 Delphi（delphi_upstream.py）：在一块 GPU 上并行训练几个 seed。
#   sbatch ris/delphi_upstream.sbatch rosmap 42 43 44      # -> $AD_BASE/delphi_upstream/Delphi-upstream-rosmap[-sNN]/
#   sbatch ris/delphi_upstream.sbatch nacc 42 43 44
# 额外参数用环境变量 EXTRA 传给 train.py（例：EXTRA="--max_iters=300 --out_dir=speedtest"，只做测速）。
# 用上游原样的 15 万步日程，到 27 分钟就停：ckpt.pt 是 val loss 最低的那一步（上游只在 val 变好时存）。
# val 还在降就接着跑：EXTRA="--init_from=resume"（从 ckpt.pt 的步数和优化器状态继续）。
#SBATCH --job-name=delphi-up
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/delphi_upstream_%j.out
#SBATCH --error=logs/delphi_upstream_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2
DS=$1; shift
cd $BASE/delphi_upstream
echo "host $(hostname)  dataset $DS  seeds $*  $(nvidia-smi --query-gpu=name --format=csv,noheader)  $(cat UPSTREAM_COMMIT)"
md5sum model.py utils.py train.py configurator.py
for s in "$@"; do
  d=Delphi-upstream-$DS; [ $s = 42 ] || d=$d-s$s
  timeout ${TLIMIT:-27m} python -u train.py config/train_$DS.py --device=cuda --out_dir=$d --seed=$s ${EXTRA:-} >> logs/train_${DS}_$s.log 2>&1 &
done
wait
for s in "$@"; do echo "== seed $s"; grep "^step" logs/train_${DS}_$s.log | sort -t' ' -k7 -g | head -1; grep "^step" logs/train_${DS}_$s.log | tail -2; done
