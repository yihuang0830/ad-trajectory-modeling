#!/bin/bash
# 只推 models_code/ 的代码到 RIS（要连 unified_eval 一起推，用 unified_eval/ris/deploy.sh）
set -euo pipefail
ZOO="$(cd "$(dirname "$0")/../.." && pwd)"
BASE=/path/to/project
ssh -o BatchMode=yes cluster "mkdir -p $BASE/model_zoo/models_code/logs"
rsync -a --exclude __pycache__ --exclude logs --exclude '*.csv' --exclude '*.png' \
  "$ZOO/models_code/" cluster:$BASE/model_zoo/models_code/
echo "deployed -> $BASE/model_zoo/models_code"
