#!/bin/bash
# Delphi 在 NACC 的一档训练量上从零训练，3 个 seed 在同一块 GPU 上并行，27 分钟停（只在 val 变好时存检查点 = 留下 val 最低的那一步）。
#   sbatch ris/delphi.sbatch delphi-upstream full     # 上游原版代码 + 配置（delphi_upstream/config/train_nacc.py）-> delphi_upstream/NS-full[-sNN]/
#   sbatch ris/delphi.sbatch delphi-zoo rosmap        # zoo 的 nojitter 版（delphi/config/train_delphi_nacc.py）-> delphi/NS-zoo-rosmap[-sNN]/
# 数据 delphi/data/nacc_scale_<档>/（nacc_scale.py prep）：train.bin = 这一档的人，val.bin = tune 的人。跑完写 DONE，CPU 工人再打分。
#SBATCH --job-name=ns-delphi
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/delphi_%j.out
#SBATCH --error=logs/delphi_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2
VAR=$1 SIZE=$2
LOG=$BASE/model_zoo/_runs/nacc_scale/delphi_logs
mkdir -p $LOG
if [ $VAR = delphi-upstream ]; then
  cd $BASE/delphi_upstream; base=NS-$SIZE
  cmd="python -u train.py config/train_nacc.py --device=cuda --dataset=nacc_scale_$SIZE"
else
  cd $BASE/delphi; base=NS-zoo-$SIZE
  cmd="python -u train.py config/train_delphi_nacc.py --device=cuda --dataset=nacc_scale_$SIZE --lifestyle_token_max=2"
fi
echo "host $(hostname)  $VAR $SIZE  $(nvidia-smi --query-gpu=name --format=csv,noheader)  $(pwd)"
for s in 42 43 44; do
  d=$base; [ $s = 42 ] || d=$d-s$s
  [ -e $d/DONE ] && { echo "$d done already"; continue; }
  # 锁：Slurm 不稳时同一档可能被交了两次，两个作业不能写同一个目录（超过 35 分钟的锁是死掉的作业留下的）
  [ -n "$(find $d.lock -maxdepth 0 -mmin +35 2>/dev/null)" ] && rmdir $d.lock
  mkdir $d.lock 2>/dev/null || { echo "$d locked by another job"; continue; }
  ( timeout 27m $cmd --out_dir=$d --seed=$s > $LOG/${VAR}_$d.log 2>&1 || true
    [ -e $d/ckpt.pt ] && touch $d/DONE; rmdir $d.lock ) &
done
wait
for s in 42 43 44; do
  d=$base; [ $s = 42 ] || d=$d-s$s
  [ -e $LOG/${VAR}_$d.log ] || continue
  echo "== $d: best $(grep '^step' $LOG/${VAR}_$d.log | sort -k8 -g | head -1)  last $(grep '^step' $LOG/${VAR}_$d.log | tail -1)"
done
