"""input_data.py -- 除 Delphi 外所有模型共用的输入（原来是 2026-11-01 删掉的 landmark 模型的输入，所以 RIS 上的文件名还叫 newmodel_data*）：逐访视的**连续值**（不分箱）+ 缺失掩码 + 状态 + 静态背景，和统一评估同一张时间网格。

    python input_data.py          # -> $ZOO_RUN_DIR/newmodel_data.npz（人级，只留在 RIS）

和 Delphi 分词的关系：
  * 原始表格、清洗规则（hba1c>20 剔除）、访视年龄 round(age_bl*365.25) + fu_year*365 与
    tokenization/build.py 完全相同，所以 (pid, t) 和 truth.py 的站点逐个对得上（下面有断言）。
  * 不分箱：MMSE、cogn_global、血压……直接用数值（按训练集均值/标准差标准化），缺失记 0 并给掩码。
  * 标签直接取 truth.py 的真值表（同一份定义），训练集也用同一份代码生成。
划分：train / test / val = rosmap_nodedup_3w 的三段（split811）。test 只用来挑 checkpoint，
val 只用于最终统一评估 —— 和 Delphi 的 val 是同一批人。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import AD_BASE, ENDPOINTS, HORIZONS, RUN_DIR, load_bin  # noqa: E402
import truth as truth_mod  # noqa: E402

RAW = os.environ.get("ROSMAP_DATA_ROOT", os.path.join(AD_BASE, "raw"))

CONT = ["cts_estmmse30", "cogn_global", "sbp_avg", "dbp_avg", "bmi", "glucose", "hba1c", "hdlchlstrl",
        "ldlchlstrl", "gfr_mdrs", "psqi_sum", "berlin_risk_class", "log_hcrp", "log_hil6", "log_htnfa"]
STATE = ["hypertension_cum", "dm_cum", "chf_cum", "claudication_cum", "heart_cum",
         "antihyp_rx", "statin_rx", "diabetes_rx", "ad_rx"]
EVER = ["r_stroke", "r_depres"]          # 逐访视"本次有记录" -> 累计"曾经有过"
PREVALENT = ["hypertension_cum", "dm_cum", "chf_cum", "claudication_cum", "heart_cum"]


def load_raw():
    cs = pd.read_excel(f"{RAW}/cross-sectional-data-gk.xlsx")
    lo = pd.read_excel(f"{RAW}/longitudinal_data_gk.xlsx").sort_values(["projid", "fu_year"])
    cl = pd.read_csv(f"{RAW}/ROSMAP_clinical.csv", dtype=str)
    cl["pid"] = pd.to_numeric(cl.projid, errors="coerce").astype("Int64")
    lo.loc[lo.hba1c > 20, "hba1c"] = np.nan                    # 与 build.py 相同
    cs = cs.set_index("projid")
    race = pd.to_numeric(cl.set_index("pid").race, errors="coerce").to_dict()
    span = pd.to_numeric(cl.set_index("pid").spanish, errors="coerce").to_dict()
    return cs, lo, race, span


def static_features(cs, race, span, pid):
    r = cs.loc[pid]
    f = [1.0 if r.msex == 1 else 0.0]
    educ = r.educ
    f += [0.0 if pd.isna(educ) else (educ - 16.0) / 4.0, float(pd.isna(educ))]
    rc = race.get(pid)
    f += [float(rc == 1), float(rc == 2), float(rc is not None and not pd.isna(rc) and rc not in (1, 2)),
          float(rc is None or pd.isna(rc))]
    sp = span.get(pid)
    f += [float(sp == 1), float(sp is None or pd.isna(sp))]
    g = r.apoe_genotype
    e4 = {22: 0, 23: 0, 33: 0, 24: 1, 34: 1, 44: 2}.get(int(g)) if pd.notna(g) else None
    f += [float(e4 == 1), float(e4 == 2), float(e4 is None),
          float(pd.notna(g) and int(g) in (22, 23, 24))]
    sm = r.smoking_bl
    f += [float(sm == 1), float(sm == 2), float(pd.isna(sm))]
    al = r.ldai_bl
    f += [0.0 if pd.isna(al) else np.log1p(max(al, 0.0)), float(pd.isna(al))]
    st = str(r.study).strip()
    f += [float(st == "MAP"), float(st == "LATC")]
    return f


STATIC_NAMES = ["male", "educ_z", "educ_na", "race_white", "race_black", "race_other", "race_na",
                "hisp_yes", "hisp_na", "apoe_e4x1", "apoe_e4x2", "apoe_na", "apoe_e2carrier",
                "smoke_former", "smoke_current", "smoke_na", "alcohol_log1p", "alcohol_na", "study_MAP",
                "study_LATC"] + [f"prevalent_{v}" for v in PREVALENT]


L2C_NAMES = [f"{c}_{k}" for c in CONT for k in ("last", "has", "since", "delta", "min", "max", "slope", "nobs")]


def l2c_features(hist, y):
    """longitudinal-to-cross-sectional 特征（TADPOLE 冠军 Frog / L2C-FNN 的做法），只用到当前访视为止。
    每个变量 8 个：最近值、是否测过、距最近一次测量的年数、最近一次变化、历史最小、历史最大、OLS 斜率（每年）、log(1+次数)。"""
    out = []
    for c in CONT:
        h = hist[c]
        if not h:
            out += [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            continue
        yrs = np.array([a for a, _ in h])
        val = np.array([b for _, b in h])
        slope = float(np.polyfit(yrs, val, 1)[0]) if len(h) >= 2 and np.ptp(yrs) > 0 else 0.0
        out += [float(val[-1]), 1.0, min(float(y - yrs[-1]), 10.0),
                float(val[-1] - val[-2]) if len(h) >= 2 else 0.0,
                float(val.min()), float(val.max()), float(np.clip(slope, -3, 3)), float(np.log1p(len(h)))]
    return out


def build():
    cs, lo, race, span = load_raw()
    split_pids = {s: set(np.unique(load_bin("rosmap_nodedup_3w", s)[:, 0]).tolist())
                  for s in ("train", "test", "val")}
    truth = {}
    for sp in ("train", "val"):                      # snapshot 是两分：train 里包含 3w 的 test 段
        f = os.path.join(RUN_DIR, f"truth_{sp}.csv")
        if not os.path.exists(f):
            t = truth_mod.build(sp)
            t.to_csv(f, index=False)
        truth[sp] = pd.read_csv(f)
    T = pd.concat([truth["train"], truth["val"]], ignore_index=True).set_index(["pid", "t"])
    ycols = [f"y_{e}_{h}y" for e in ENDPOINTS for h in HORIZONS]

    # 训练集上的标准化参数（只用 3w train 的人）
    lo_tr = lo[lo.projid.isin(split_pids["train"])]
    mu = lo_tr[CONT].mean()
    sd = lo_tr[CONT].std().replace(0, 1.0)

    lo_g = {p: g for p, g in lo.groupby("projid")}
    persons = {s: [] for s in split_pids}
    for s, pids in split_pids.items():
        for pid in sorted(pids):
            g = lo_g[pid].sort_values("fu_year")
            r = cs.loc[pid]
            ab = float(r.age_bl)
            years = sorted(float(y) for y in g.fu_year.dropna().unique())
            ad_k = max(int(round(float(r.age_first_ad_dx) - ab)), 0) if pd.notna(r.age_first_ad_dx) else None
            stat = static_features(cs, race, span, pid)
            b0 = g[g.fu_year == 0]
            stat += [float(len(b0) > 0 and b0[v].iloc[0] == 1) for v in PREVALENT]
            ever = {v: 0.0 for v in EVER}
            rows, ts, labels, l2c = [], [], [], []
            hist = {c: [] for c in CONT}           # (年, 标准化值)，只含到当前访视为止
            prev_y = None
            for y in years:
                v = g[g.fu_year == y].iloc[0]
                for c in CONT:
                    if pd.notna(v[c]):
                        hist[c].append((y, (v[c] - mu[c]) / sd[c]))
                l2c.append(l2c_features(hist, y))
                t = int(round(ab * 365.25)) + int(round(y)) * 365
                x_c = [(0.0 if pd.isna(v[c]) else (v[c] - mu[c]) / sd[c]) for c in CONT]
                m_c = [float(pd.isna(v[c])) for c in CONT]
                x_s = [0.0 if pd.isna(v[c]) else float(v[c] == 1) for c in STATE]
                m_s = [float(pd.isna(v[c])) for c in STATE]
                for c in EVER:
                    if pd.notna(v[c]) and v[c] in (1, 2, 3):
                        ever[c] = 1.0
                x_e = [ever[c] for c in EVER]
                ad_state = float(ad_k is not None and y >= ad_k)
                age = t / 365.25
                tf = [(age - 80.0) / 10.0, y / 10.0, 0.0 if prev_y is None else (y - prev_y)]
                rows.append(x_c + m_c + x_s + m_s + x_e + [ad_state] + tf)
                ts.append(t)
                key = (int(pid), t)
                assert key in T.index, f"站点 {key} 不在真值表里 —— 访视网格和分词对不上"
                labels.append(T.loc[key, ycols].to_numpy(dtype=np.int8))
                prev_y = y
            persons[s].append({"pid": int(pid), "x": np.array(rows, np.float32), "t": np.array(ts, np.int64),
                               "static": np.array(stat, np.float32), "y": np.array(labels, np.int8),
                               "l2c": np.array(l2c, np.float32)})
    # 每个人的站点数必须和真值表一致（不多不少）
    for s in ("val",):
        n_truth = len(truth["val"])
        n_here = sum(len(p["t"]) for p in persons[s])
        assert n_here == n_truth, f"val 站点数 {n_here} != 真值表 {n_truth}"
    feat_names = ([f"{c}" for c in CONT] + [f"{c}_missing" for c in CONT] + STATE + [f"{c}_missing" for c in STATE]
                  + [f"{c}_ever" for c in EVER] + ["ad_dx_state", "age_z", "years_since_bl", "gap_years"])
    return persons, feat_names, ycols, {"mu": mu.to_dict(), "sd": sd.to_dict()}


def pack(persons, max_v=None):
    n = len(persons)
    V = max_v or max(len(p["t"]) for p in persons)
    F = persons[0]["x"].shape[1]
    S = len(persons[0]["static"])
    K = persons[0]["y"].shape[1]
    X = np.zeros((n, V, F), np.float32)
    Lf = np.zeros((n, V, persons[0]["l2c"].shape[1]), np.float32)
    M = np.zeros((n, V), bool)
    Y = np.full((n, V, K), -1, np.int8)
    Tt = np.zeros((n, V), np.int64)
    St = np.zeros((n, S), np.float32)
    pid = np.zeros(n, np.int64)
    for i, p in enumerate(persons):
        L = len(p["t"])
        X[i, :L], M[i, :L], Y[i, :L], Tt[i, :L] = p["x"], True, p["y"], p["t"]
        Lf[i, :L] = p["l2c"]
        St[i], pid[i] = p["static"], p["pid"]
    return dict(X=X, M=M, Y=Y, T=Tt, S=St, pid=pid, L=Lf)


if __name__ == "__main__":
    persons, feat_names, ycols, norm = build()
    V = max(len(p["t"]) for s in persons for p in persons[s])
    out = {}
    for s in persons:
        for k, v in pack(persons[s], V).items():
            out[f"{s}_{k}"] = v
    os.makedirs(RUN_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(RUN_DIR, "newmodel_data.npz"), **out)
    with open(os.path.join(RUN_DIR, "newmodel_data_meta.json"), "w") as f:
        json.dump({"features": feat_names, "static": STATIC_NAMES, "l2c": L2C_NAMES, "labels": ycols, "norm": norm,
                   "n": {s: len(persons[s]) for s in persons}, "max_visits": V}, f, indent=1)
    print({s: (len(persons[s]), sum(len(p["t"]) for p in persons[s])) for s in persons}, "F", len(feat_names))
