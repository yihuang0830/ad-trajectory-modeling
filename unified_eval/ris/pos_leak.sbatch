#!/bin/bash
# 带位置编码的模型：位置下标漏不漏未来（每个 pos 模型的 seed 42 + 一个不带位置编码的对照）
#SBATCH --job-name=zoo-posleak
#SBATCH --partition=general-cpu
#SBATCH --account=your-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/posleak_%j.out
#SBATCH --error=logs/posleak_%j.err
set -euo pipefail
BASE="${AD_BASE:-/path/to/project}"
source /path/to/env/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE="$BASE" ZOO_RUN_DIR="$BASE/model_zoo/_runs" OMP_NUM_THREADS=8
cd "$BASE/model_zoo/unified_eval"
D=$BASE/delphi
python -u pos_leak_check.py --name delphi-nodedup-pos-split811 --ckpt $D/Delphi-ROSMAP-nodedup-pos-3w/ckpt.pt --dataset rosmap_nodedup_3w
python -u pos_leak_check.py --name delphi-nodedup-pos --ckpt $D/Delphi-ROSMAP-nodedup-pos/ckpt.pt --dataset rosmap_nodedup
python -u pos_leak_check.py --name delphi-nodedup --ckpt $D/Delphi-ROSMAP-nodedup/ckpt.pt --dataset rosmap_nodedup
python -u pos_leak_check.py --name delphi-snapshot-pos --ckpt $D/Delphi-ROSMAP-snapshot-pos/ckpt.pt --dataset rosmap_snapshot
