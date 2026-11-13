#!/bin/bash
# 推统一评估代码和模型代码（models_code/）到 RIS（只推代码；数据和 ckpt 早就在集群上，不走这里）
set -euo pipefail
ZOO="$(cd "$(dirname "$0")/../.." && pwd)"
BASE=/path/to/project
ssh -o BatchMode=yes cluster "mkdir -p $BASE/model_zoo/unified_eval/logs $BASE/model_zoo/models_code/logs $BASE/model_zoo/_runs && chmod 700 $BASE/model_zoo/_runs"
for d in unified_eval models_code; do
  rsync -a --exclude __pycache__ --exclude logs --exclude '*.csv' --exclude '*.png' "$ZOO/$d/" cluster:$BASE/model_zoo/$d/
done
echo "deployed -> $BASE/model_zoo/{unified_eval,models_code}"
