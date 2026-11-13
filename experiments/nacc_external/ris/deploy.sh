#!/bin/bash
# 推本目录和 models_code/（模型代码）到 RIS；GIT_REV 记下部署时的提交，作业日志第一行会打印它
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ZOO="$(cd "$HERE/../.." && pwd)"
BASE=/path/to/project
D=$BASE/model_zoo/experiments/nacc_external
(cd "$ZOO" && git rev-parse --short HEAD && git status --porcelain -- experiments/nacc_external models_code | head -20) > "$HERE/GIT_REV"
ssh -o BatchMode=yes -o LogLevel=ERROR cluster "mkdir -p $D/logs && chmod 700 $D"
rsync -a --exclude __pycache__ --exclude logs --exclude results -e "ssh -o BatchMode=yes -o LogLevel=ERROR" "$HERE/" cluster:$D/
rsync -a --exclude __pycache__ --exclude logs --exclude '*.csv' --exclude '*.png' -e "ssh -o BatchMode=yes -o LogLevel=ERROR" \
  "$ZOO/models_code/" cluster:$BASE/model_zoo/models_code/
rm -f "$HERE/GIT_REV"
echo "deployed -> $D (+ models_code)"
