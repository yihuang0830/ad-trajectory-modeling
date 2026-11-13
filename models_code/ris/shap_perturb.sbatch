#!/bin/bash
# transformer 的 SHAP + 扰动（shap_perturb.py）：sbatch ris/shap_perturb.sbatch smoke | full
#   smoke：重训 + 8 人的 SHAP / 扰动（_runs/shap/transformer/_limit8/）；full：全队列（模型已存在就不重训）
#SBATCH --job-name=shap-pert
#SBATCH --partition=general-short
#SBATCH --account=your-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shap_pert_%j.out
#SBATCH --error=logs/shap_pert_%j.err
set -euo pipefail
umask 077
BASE=/path/to/project
E=/path/to/env
source $E/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export AD_BASE=$BASE ZOO_RUN_DIR=$BASE/model_zoo/_runs OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore MPLBACKEND=Agg
PY="$E/venvs/ad-baselines/bin/python -u shap_perturb.py"
cd $BASE/model_zoo/models_code
echo "host $(hostname)  mode: $1"
[ -f $ZOO_RUN_DIR/shap/transformer/model.pt ] || $PY train
if [ "$1" = smoke ]; then
  $PY shap --limit 8 --perms 4
  $PY perturb --limit 8 --n_mc 16
  $PY aggregate --limit 8
else
  [ -f $ZOO_RUN_DIR/shap/transformer/phi_landmark.npz ] || $PY shap
  $PY perturb
  $PY aggregate
fi
