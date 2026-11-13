"""probe_nacc_cogn.py -- NACC 的复合认知分和 ROSMAP 的 cogn_global 是不是同一把尺（2026-10-29 诊断；只打印聚合数，在 RIS 上跑）。

    python probe_nacc_cogn.py

nacc_data.py 把 NACC 每项测验按"第一次做时认知正常（NACCUDSD = 1）的人"标准化再平均；ROSMAP 的 cogn_global 是相对整个
基线队列（含 MCI、痴呆）的 z 分数平均。两边随后用同一组 ROSMAP 均值 / 标准差归一化，"认知 < 0、< −1"两条线也直接套用。
这里用两边同一个刻度的 MMSE 当锚：第一次访视按 MMSE 分段，比两边复合分的均值、标准差和低于 0 / −1 的比例。
同一段 MMSE 上两边的复合分差得越多，同一个数在两边代表的认知水平越不一样。
"""
import json
import os

import numpy as np
import pandas as pd

import nacc_data as ND
from common import AD_BASE, RUN_DIR

BANDS = [(28, 30), (26, 27), (24, 25), (18, 23), (0, 17)]


def summ(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "sd": round(float(x.std()), 3),
            "lt0": round(float((x < 0).mean()), 3), "ltm1": round(float((x < -1).mean()), 3)}


def by_band(df, mmse, cogn):
    out = {}
    for lo, hi in BANDS:
        m = (df[mmse] >= lo) & (df[mmse] <= hi)
        out[f"mmse_{lo}_{hi}"] = summ(df.loc[m, cogn])
    out["all"] = summ(df[cogn])
    return out


def main():
    d, _ = ND.load_nacc()
    first = d[~d.duplicated("NACCID")]
    nacc = by_band(first, "mmse", "cogn")
    nacc_dx = {f"NACCUDSD_{k}": summ(first.loc[first.NACCUDSD == k, "cogn"]) for k in (1, 2, 3, 4)}
    raw = os.environ.get("ROSMAP_DATA_ROOT", os.path.join(AD_BASE, "raw"))
    lo = pd.read_excel(os.path.join(raw, "longitudinal_data_gk.xlsx")).sort_values(["projid", "fu_year"])
    base = lo[~lo.duplicated("projid")]
    ros = by_band(base, "cts_estmmse30", "cogn_global")
    out = {"nacc_first_visit_by_mmse": nacc, "nacc_first_visit_by_dx": nacc_dx, "rosmap_baseline_by_mmse": ros,
           "note": "cogn = 原始复合分（归一化之前）；lt0 / ltm1 = 低于 0 / −1 的比例（第三部分两条认知线）"}
    od = os.path.join(RUN_DIR, "packages", "_nacc")
    json.dump(out, open(os.path.join(od, "probe_cogn_scale.json"), "w"), indent=1)
    print(f"{'MMSE 段':12s} {'NACC n / 均值 / sd / <0 / <-1':40s} ROSMAP")
    for k in list(nacc):
        print(f"{k:12s} {str(nacc[k]):40s} {ros[k]}")
    print(json.dumps(nacc_dx))


if __name__ == "__main__":
    main()
