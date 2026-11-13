#!/bin/bash
# 登录节点上的看守（同 ../nacc_scale/ris/keeper.sh）：每 10 分钟按队列补交 W 个 CPU 工人、G 个 GPU 工人；全部做完交一次 collect 后退出。
# squeue 不通时（Slurm 控制器不稳）按"10 分钟内有更新的日志"估在跑的工人数，照样补交（工人重复了无害：有锁、做完的不重做）。
#   cd $AD_BASE/model_zoo/experiments/nacc_scaling_ar && nohup bash ris/keeper.sh [W=6] [G=3] > logs/keeper.log 2>&1 &
W=${1:-6} G=${2:-3}
BASE=/path/to/project
E=/path/to/env
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore MPLBACKEND=Agg
RES=$ZOO_RUN_DIR/packages/_nacc_scaling_ar/results.json
log() { echo "$(date +%m-%d\ %H:%M) $*"; }
sub() { log "submit $1 $2: $(timeout 90 sbatch --parsable $2 ris/$1.sbatch ${3:-} 2>&1 | tail -1)"; }
while true; do
  st=$(timeout 300 $E/venvs/ad-baselines/bin/python nacc_scaling_ar.py status 2>&1)
  log "$(echo "$st" | head -1)"
  done_all=$(echo "$st" | head -1 | grep -cE '^([0-9]+) / \1 units done')
  if q=$(timeout 90 squeue -u $USER -h -o "%j" 2>/dev/null); then
    nw=$(echo "$q" | grep -cx ns3-work); ng=$(echo "$q" | grep -cx ns3-gpu)
  else
    nw=$(find logs -name 'work_*.out' -mmin -10 | wc -l); ng=$(find logs -name 'gpu_*.out' -mmin -10 | wc -l)
    log "squeue down; by log: $nw cpu / $ng gpu workers"
  fi
  if [ $done_all = 1 ]; then
    [ -e $RES ] && { log "results.json written"; exit 0; }
    echo "${q:-}" | grep -qx ns3-collect || { [ $nw = 0 ] && sub worker --job-name=ns3-collect collect; }
  else
    # GPU 的活（transformer M / L）做完了就不再交 GPU 工人
    gl=$(echo "$st" | grep -cE "'trf-[ml]'")
    for i in $(seq $((nw + 1)) $W); do sub worker ""; done
    [ $gl -gt 0 ] && for i in $(seq $((ng + 1)) $G); do sub gpu ""; done
  fi
  sleep 600
done
