"""replot.py -- 只用 metrics.json 在本地重画每个打包盒的两张统一图（不需要人级打分表）。

    python replot.py            # model_zoo/*/metrics.json -> fig1_auc_2y.png, fig2_calibration_2y.png
    python replot.py --auc-only # 只重画 fig1（fig2 不动）
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from evaluate import fig_auc, fig_cal  # noqa: E402
from registry import ARCHIVE_DIR, CAL_NOTES, REP_DIR, SELF_DIR  # noqa: E402

# 第三部分答题版的第一部分指标存成 metrics_part1.json（不进表 1 排名），图名带 _part1
for p in sorted(p for z in (SELF_DIR, ARCHIVE_DIR, REP_DIR) for f in ("metrics.json", "metrics_part1.json") for p in glob.glob(os.path.join(ZOO, z, "*", f))):
    d = os.path.dirname(p)
    name = os.path.basename(d)
    sfx = "_part1" if p.endswith("_part1.json") else ""
    m = json.load(open(p))
    fig_auc(m, os.path.join(d, f"fig1_auc_2y{sfx}.png"))
    if "--auc-only" not in sys.argv:
        fig_cal(m, os.path.join(d, f"fig2_calibration_2y{sfx}.png"), CAL_NOTES.get(name, m.get("calibration_note")))
    print("replotted", name)
