#!/bin/bash
# 登录节点上的看守（很轻：每 10 分钟查一次队列）。Slurm 控制器不稳时 sbatch 会超时、其实已经交上，重试就交重复了；
# 所以只在 squeue 能回话时，按队列里实际有的作业补交，每次只交一次、不重试。全部单元做完后交一次 collect 然后退出。
#   cd $AD_BASE/model_zoo/experiments/nacc_scale && nohup bash ris/keeper.sh [W=6] > logs/keeper.log 2>&1 &
W=${1:-6}
BASE=/path/to/project
E=/path/to/env
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore MPLBACKEND=Agg
RES=$ZOO_RUN_DIR/packages/_nacc_scale/results.json
declare -A tries
log() { echo "$(date +%m-%d\ %H:%M) $*"; }
while true; do
  if ! q=$(timeout 90 squeue -u $USER -h -o "%j" 2>/dev/null); then
    # squeue 不通：只补 CPU 工人（重复了也无害：有锁、做完的不重做）。在跑的工人数 = 10 分钟内有更新的工人日志
    n=$(find logs -name 'work_*.out' -mmin -10 | wc -l)
    log "squeue down; $n workers active by log"
    for i in $(seq $((n + 1)) $W); do log "submit worker (blind): $(timeout 90 sbatch --parsable ris/worker.sbatch 2>&1 | tail -1)"; done
    sleep 600; continue
  fi
  for v in delphi-upstream delphi-zoo; do
    for s in full rosmap 12k; do
      [ $v = delphi-upstream ] && b=$BASE/delphi_upstream/NS-$s || b=$BASE/delphi/NS-zoo-$s
      [ -e $b/DONE ] && [ -e $b-s43/DONE ] && [ -e $b-s44/DONE ] && continue
      echo "$q" | grep -qx "nsd-$v-$s" && continue
      k=$v-$s; tries[$k]=$(( ${tries[$k]:-0} + 1 ))
      [ ${tries[$k]} -gt 4 ] && { log "delphi $k: gave up after 4 submissions"; continue; }
      log "submit delphi $k: $(timeout 90 sbatch --parsable -J nsd-$k ris/delphi.sbatch $v $s 2>&1 | tail -1)"
    done
  done
  st=$(timeout 300 $E/venvs/ad-baselines/bin/python nacc_scale.py status 2>&1)
  log "$(echo "$st" | head -1)"
  if echo "$st" | head -1 | grep -qE '^([0-9]+) / \1 units done'; then
    echo "$q" | grep -qx ns-collect || log "submit collect: $(timeout 90 sbatch --parsable -J ns-collect ris/worker.sbatch collect 2>&1 | tail -1)"
    [ -e $RES ] && { log "results.json written"; exit 0; }
  else
    n=$(echo "$q" | grep -cx ns-work)
    for i in $(seq $((n + 1)) $W); do log "submit worker: $(timeout 90 sbatch --parsable ris/worker.sbatch 2>&1 | tail -1)"; done
  fi
  sleep 600
done
