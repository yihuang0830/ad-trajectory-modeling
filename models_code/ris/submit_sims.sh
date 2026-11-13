#!/bin/bash
# 在 RIS 上（model_zoo/models_code 下）提交全部 rollout 版：每个模型一对作业（短分区 30 分钟 + 同名长作业兜底，
# 先跑完的算数，--redo_before 让后一个直接跳过），最后一个汇总作业等它们全部结束（afterany）。
#   bash ris/submit_sims.sh            # 打印并记下全部作业号到 logs/submit_sims_<时间>.txt
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(date +%s)
LOG=logs/submit_sims_$(date +%Y%m%d_%H%M).txt
IDS=()
sub() {   # sub <名字> <cpu|gpu> <长作业时限> [run.py 的参数...]
  local m=$1 kind=$2 tl=$3; shift 3
  local res_s res_l
  if [ "$kind" = gpu ]; then
    res_s="--partition=general-short --gres=gpu:1 --cpus-per-task=8"; res_l="--partition=general-gpu --gres=gpu:1 --cpus-per-task=8"
  else
    res_s="--partition=general-short --cpus-per-task=16"; res_l="--partition=general-cpu --cpus-per-task=16"
  fi
  local a b
  a=$(sbatch --parsable --job-name=bs-$m $res_s --time=00:30:00 ris/run.sbatch $m -- --redo_before $T "$@")
  b=$(sbatch --parsable --job-name=bs-$m $res_l --time=$tl --dependency=singleton ris/run.sbatch $m -- --redo_before $T "$@")
  IDS+=("$a" "$b"); echo "$m short=$a long=$b $*" | tee -a $LOG
}
# 线性 / 逻辑回归、Cox、临床评分、梯度提升、MLP：CPU，快
for m in lr-demo-sim lr-cur-sim bdsi-sim anu-adri-sim caide-sim cox-cur-sim cox-l2c-sim cox-cogn-sim cox-super-sim \
         xgb-l2c-sim xgbcox-l2c-sim survivalboost-l2c-sim deepsurv-l2c-sim deephit-l2c-sim logistichazard-l2c-sim \
         pmf-l2c-sim mtlr-l2c-sim; do
  sub $m cpu 06:00:00
done
# 随机森林、随机生存森林：预测慢，站点上 50 条、扰动上 30 条
sub rf-l2c-sim cpu 08:00:00 --n_mc_lm 50 --n_mc_pert 30
sub rsf-l2c-sim cpu 08:00:00 --n_mc_lm 50 --n_mc_pert 30
# 序列网络：GPU，每批 300 个站点（gru、lstm、grud、tlstm、retain、dynamic-deephit 2026-10-28 起本身就是自回归的，没有 -sim；
# transformer-sim 只留作 NACC 实验的对照）
for m in fnn-l2c-sim transformer-sim; do
  sub $m gpu 08:00:00 --lm_chunk 300
done
# TabPFN：每一步 54 个 TabPFN 模型都要推理，条数压到站点 10 条、扰动 10 条、第一次访视 30 条
sub tabpfn-l2c-sim gpu 08:00:00 --n_mc_lm 10 --n_mc_pert 10 --n_mc 30 --lm_chunk 1000
dep=$(IFS=:; echo "${IDS[*]}")
s=$(sbatch --parsable --dependency=afterany:$dep ris/sim_summary.sbatch)
echo "summary=$s (afterany all ${#IDS[@]} jobs)" | tee -a $LOG
