#!/bin/bash
# 本地：在 RIS 登录节点上跑 ext_status.py（只读、几秒钟）
ssh -o BatchMode=yes -o LogLevel=ERROR cluster 'B=/path/to/project; cd $B/model_zoo/experiments/nacc_external && source /path/to/env/miniforge3/etc/profile.d/conda.sh && conda activate ad-projection && AD_BASE=$B ZOO_RUN_DIR=$B/model_zoo/_runs python ext_status.py 2>&1 | grep -v Warning'
