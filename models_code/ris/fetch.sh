#!/bin/bash
# 把模型（Delphi 以外）的聚合产物（JSON + evaluate.py 画的 fig1 / fig2）拉回本地打包盒（models_self/、representative/ 或 models_not_usable/，见 registry.pkg_dir）。人级的打分表、轨迹不拉。
set -euo pipefail
ZOO="$(cd "$(dirname "$0")/../.." && pwd)"
R=/path/to/project/model_zoo/_runs/packages
pairs=$(cd "$ZOO/models_code" && python3 -c "import sys; sys.path.insert(0, '../unified_eval'); import models, registry; print(' '.join(f'{b[0]}:{registry.pkg_dir(b[0])}' for b in models.BASELINES))")
for p in $pairs; do
  n=${p%%:*}; dst=${p#*:}
  mkdir -p "$ZOO/$dst"
  rsync -a --include='metrics.json' --include='transitions.json' --include='trajectory.json' --include='baseline.json' --include='perturb.json' --include='perturb_v2.json' --include='timing.json' \
    --include='fig1_auc_2y.png' --include='fig2_calibration_2y.png' --exclude='*' \
    "cluster:$R/$n/" "$ZOO/$dst/" 2>/dev/null || echo "-- $n: nothing yet"
done
# NACC 预训练（表 5）：配对比较、预训练记录、数据的聚合描述
mkdir -p "$ZOO/models_self/_nacc"
rsync -a --include='*.json' --exclude='*' "cluster:$R/_nacc/" "$ZOO/models_self/_nacc/" 2>/dev/null || echo "-- _nacc: nothing yet"
# 表 5 的 Delphi（nacc_delphi.py）：两个打包盒在 unified_eval 那边评估，只拉表里用的聚合 JSON
for n in delphi-dedup-nojitter-nacc delphi-dedup-nojitter-nacc-zs; do
  mkdir -p "$ZOO/models_self/_nacc/delphi/$n"
  rsync -a --include='metrics.json' --include='transitions.json' --include='ckpt_info.json' --exclude='*' \
    "cluster:$R/$n/" "$ZOO/models_self/_nacc/delphi/$n/" 2>/dev/null || echo "-- $n: nothing yet"
done
# 9 模型图里 Delphi 的 rollout 版（unified_eval/rollout_landmarks.py）：和另外 8 个一起放在 representative/
mkdir -p "$ZOO/representative/delphi-dedup-sim"
rsync -a --include='metrics.json' --include='transitions.json' --include='timing.json' --include='perturb_v2.json' \
  --include='fig1_auc_2y.png' --include='fig2_calibration_2y.png' --exclude='*' \
  "cluster:$R/delphi-dedup-sim/" "$ZOO/representative/delphi-dedup-sim/" 2>/dev/null || echo "-- delphi-dedup-sim: nothing yet"
# 轨迹数值的比较（表 6）
mkdir -p "$ZOO/models_self/_traj"
rsync -a --include='*.json' --exclude='*' "cluster:$R/_traj/" "$ZOO/models_self/_traj/" 2>/dev/null || echo "-- _traj: nothing yet"
echo "fetched -> $ZOO/models_self"
