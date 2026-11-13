"""shap_vars.py -- 按"变量"拆账的 SHAP 工具（models_code/shap_perturb.py 用它解释 representative/transformer）。

和 shap_delphi 同一批人、同一个站点、同一种单位：扰动队列（>= 4 次访视，站点 = 第 4 次访视；perturb_v2.cohort_table）。
players（拆账单位）= 变量，到站点为止的整段记录一起拿掉（remove）：
  15 个连续量（MMSE、认知、血压……）：每次访视都记"没测"（值 0、缺失掩码 1）；手工历史特征正好变成"从没测过"的默认值
      （最近值 0、是否测过 0、距最近测量 10 年、其余 0），和 input_data.l2c_features 对没测过的变量的写法相同；
  9 个状态（共病、用药）：每次访视记"状态缺失"；卒中、抑郁"曾经有过"：记 0；
  背景：教育、种族、西班牙裔、APOE、吸烟、饮酒记成"缺失"那一档；5 个基线即患病记 0；性别男女互换（和 shap_delphi 一样：
      人人都有性别，没有"缺失"）。年龄、随访时间、研究队列、AD 诊断状态不当 player；年龄另算"整段病史 +5 岁"。
Shapley 值（explain_person）：置换抽样（每人若干个随机顺序，再加各自的反序），每个顺序里各 player 的贡献之和正好等于
  f(全部) − f(全部拿掉)（效率性逐人核对）。
"看值还是看在不在"（aggregate_model）：连续量按这个人到站点为止最近一次测到的值分档（切点同 Delphi 分词 tokenization/spec.py；
  CRP / IL6 / TNFA 没有临床切点，用这批人的三分位），每档报这个变量的平均 φ，命名和 Delphi 的 token 一样（MMSE_impaired），
  所以 figures.shap_summary 能用同一个函数画。输出的 "family": "landmark" 是给 figures 认"按变量拆账"这种格式用的。
（2026-11-01 前叫 shap_landmark.py，同时解释 2026-11-01 删掉的 landmark 模型；只留下通用部分。）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RUN_DIR  # noqa: E402

MIN_CARRIERS = 5                                # 少于这么多人的档不报（聚合产物要离开 RIS）
N_BOOT = 2000
SHAP_DIR = os.path.join(RUN_DIR, "shap")

# 连续量（input_data.CONT 的顺序）-> (Delphi 的家族名, 切点, 档名)；"q3" = 这批人的三分位
CONT = [("cts_estmmse30", "MMSE", [-0.1, 24, 27, 30.1], ["impaired", "borderline", "normal"]),
        ("cogn_global", "COGN", [-99, -1.0, 0.0, 99], ["low", "mid", "high"]),
        ("sbp_avg", "SBP", [0, 120, 140, 999], ["normal", "elevated", "high"]),
        ("dbp_avg", "DBP", [0, 80, 90, 999], ["normal", "elevated", "high"]),
        ("bmi", "BMI", [0, 22, 28, 999], ["low", "mid", "high"]),
        ("glucose", "GLU", [0, 100, 126, 9999], ["normal", "pre", "diabetic"]),
        ("hba1c", "HBA1C", [0, 5.7, 6.5, 99], ["normal", "pre", "diabetic"]),
        ("hdlchlstrl", "HDL", [0, 40, 60, 999], ["low", "mid", "high"]),
        ("ldlchlstrl", "LDL", [0, 100, 160, 999], ["optimal", "borderline", "high"]),
        ("gfr_mdrs", "GFR", [0, 60, 90, 999], ["ckd", "mild", "normal"]),
        ("psqi_sum", "PSQI", [-0.1, 5.5, 10.5, 99], ["good", "poor", "verypoor"]),
        ("berlin_risk_class", "APNEA", [-0.5, 0.5, 1.5], ["low", "high"]),
        ("log_hcrp", "CRP", "q3", ["low", "mid", "high"]),
        ("log_hil6", "IL6", "q3", ["low", "mid", "high"]),
        ("log_htnfa", "TNFA", "q3", ["low", "mid", "high"])]
STATE = [("hypertension_cum", "HTN"), ("dm_cum", "DM"), ("chf_cum", "CHF"), ("claudication_cum", "CLAUD"),
         ("heart_cum", "HEART"), ("antihyp_rx", "ANTIHYP"), ("statin_rx", "STATIN"), ("diabetes_rx", "DIABRX"),
         ("ad_rx", "ADRX")]
EVER = [("r_stroke", "STROKE"), ("r_depres", "DEPRESSION")]
N_CONT, N_STATE = 15, 9
# 背景 player：(名字, 拿掉时要清零的 S 下标, 拿掉时置 1 的"缺失"下标)；性别单独处理（互换）
STATIC = [("EDU", (1,), 2), ("RACE", (3, 4, 5), 6), ("HISP", (7,), 8), ("APOE", (9, 10, 12), 11),
          ("SMOKING", (13, 14), 15), ("ALCOHOL", (16,), 17)] + [(f"PREV_{a}", (20 + k,), None)
                                                                 for k, (_, a) in enumerate(STATE[:5])]
GROUP_OF = {"MMSE": "mmse", "COGN": "cognition", "APOE": "apoe", "SEX": "sex", "EDU": "education",
            **{f: "vascular_metabolic" for f in ("SBP", "DBP", "BMI", "GLU", "HBA1C", "HDL", "LDL", "GFR")},
            **{f: "inflammation" for f in ("CRP", "IL6", "TNFA")}, "PSQI": "sleep", "APNEA": "sleep",
            **{a: "comorbidity" for _, a in STATE[:5]}, **{a: "medication" for _, a in STATE[5:]},
            "STROKE": "comorbidity", "DEPRESSION": "comorbidity", "RACE": "other_static", "HISP": "other_static",
            "SMOKING": "other_static", "ALCOHOL": "other_static",
            **{f"PREV_{a}": "comorbidity" for _, a in STATE[:5]}}


def players():
    """[(名字, 种类, 下标)]。"""
    P = [(fam, "cont", c) for c, (_, fam, _, _) in enumerate(CONT)]
    P += [(a, "state", k) for k, (_, a) in enumerate(STATE)]
    P += [(a, "ever", 48 + k) for k, (_, a) in enumerate(EVER)]
    P += [("SEX", "sex", 0)] + [(n, "static", (z, na)) for n, z, na in STATIC]
    return P


L2C_EMPTY = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)


def remove(P, keep, X, S, L):
    """keep: (B, n_players) bool；X/S/L 是一个人的 (V,54)/(25,)/(V,120)。返回 B 份改过的 (X, S, L)。"""
    B = keep.shape[0]
    Xb = np.repeat(X[None], B, 0)
    Sb = np.repeat(S[None], B, 0)
    Lb = np.repeat(L[None], B, 0)
    for j, (_, kind, ix) in enumerate(P):
        off = ~keep[:, j]
        if not off.any():
            continue
        if kind == "cont":
            Xb[off, :, ix] = 0.0
            Xb[off, :, N_CONT + ix] = 1.0
            Lb[off, :, ix * 8:(ix + 1) * 8] = L2C_EMPTY
        elif kind == "state":
            Xb[off, :, 2 * N_CONT + ix] = 0.0
            Xb[off, :, 2 * N_CONT + N_STATE + ix] = 1.0
        elif kind == "ever":
            Xb[off, :, ix] = 0.0
        elif kind == "sex":
            Sb[off, 0] = 1.0 - Sb[off, 0]
        else:
            z, na = ix
            for k in z:
                Sb[off, k] = 0.0
            if na is not None:
                Sb[off, na] = 1.0
    return Xb, Sb, Lb


def explain_person(f, P, X, M, S, L, perms, rng):
    """置换抽样 + 反序。返回 phi (n_players, n_q)、f(全部拿掉)、f(全部)、效率误差。"""
    n = len(P)
    orders = []
    for _ in range(perms):
        o = rng.permutation(n)
        orders += [o, o[::-1]]
    keep = np.zeros((len(orders) * (n + 1), n), bool)          # 每个顺序：先全拿掉，再一个个放回
    r = 0
    for o in orders:
        cur = np.zeros(n, bool)
        keep[r] = cur
        r += 1
        for j in o:
            cur = cur.copy()
            cur[j] = True
            keep[r] = cur
            r += 1
    vals = []
    for b0 in range(0, len(keep), 2048):
        Xb, Sb, Lb = remove(P, keep[b0:b0 + 2048], X, S, L)
        Mb = np.repeat(M[None], len(Xb), 0)
        vals.append(f(Xb, Mb, Sb, Lb))
    V = np.concatenate(vals)                                   # (rows, n_q)
    phi = np.zeros((n, V.shape[1]))
    r = 0
    for o in orders:
        prev = V[r]
        r += 1
        for j in o:
            phi[j] += V[r] - prev
            prev = V[r]
            r += 1
    phi /= len(orders)
    base, full = V[0], V[n]                                    # 第一个顺序的起点（全拿掉）和终点（全部）
    eff = float(np.abs(phi.sum(0) - (full - base)).max())
    return phi, base, full, eff


def _bins(vals, cut, names):
    """原始值 -> 档名（nan -> None）。"""
    if isinstance(cut, str):
        ok = vals[np.isfinite(vals)]
        cut = [-np.inf, *np.quantile(ok, [1 / 3, 2 / 3]), np.inf] if len(ok) else [-np.inf, np.inf]
    out = np.array([None] * len(vals), object)
    for k, nm in enumerate(names):
        lo, hi = cut[k], cut[k + 1]
        last = k == len(names) - 1
        m = np.isfinite(vals) & (vals >= lo) & ((vals <= hi) if last else (vals < hi))
        out[m] = nm
    return out


def aggregate_model(name):
    p = os.path.join(SHAP_DIR, name, "phi_landmark.npz")
    if not os.path.exists(p):
        return None
    Z = np.load(p, allow_pickle=False)
    info = json.load(open(os.path.join(SHAP_DIR, name, "run_landmark.json")))
    P, qs = [str(x) for x in Z["players"]], [str(x) for x in Z["questions"]]
    phi, risk, raw = Z["phi"], Z["risk"], Z["raw_last"]
    rng = np.random.default_rng(0)
    out = {"name": name, "family": "landmark", "n_persons": int(len(Z["pid"])),
           "settings": {"players": "variables (all visits up to the 4th visit removed together); see shap_vars.py",
                        "masking": "continuous: every visit recorded as not measured (history features = never measured); "
                                   "states: missing; ever-had: 0; background: the 'missing' level; prevalent comorbidity: 0; "
                                   "sex flipped",
                        "readout": "the model output f at the 4th visit (the caller overwrites this with its own readout)",
                        "units": "phi in log cumulative-hazard ratio (exp(phi) = ratio with vs without the variable)",
                        "estimator": f"permutation sampling, {info['perms']} random orders + their reverses per person",
                        "bins": "continuous variables binned by the last value measured up to the 4th visit, Delphi cut "
                                "points (tokenization/spec.py); CRP/IL6/TNFA tertiles of this cohort",
                        "min_carriers": MIN_CARRIERS},
           "checks": {"max_efficiency_err": info["max_efficiency_err"], "seeds": info["seeds"]},
           "questions": {}}
    for k, q in enumerate(qs):
        m = risk[:, k]
        ph = phi[m, :, k]                                       # (n_at_risk, n_players)
        tokens, families = [], []
        for c, (_, fam, cut, names) in enumerate(CONT):
            j = P.index(fam)
            b = _bins(raw[m, c], cut, names)
            bins = []
            for nm in names:
                sel = b == nm
                n = int(sel.sum())
                if n < MIN_CARRIERS:
                    continue
                v = float(ph[sel, j].mean())
                tokens.append({"token": f"{fam}_{nm}", "n_persons": n, "mean_person_phi": round(v, 5),
                               "frac_pos": round(float((ph[sel, j] > 0).mean()), 4)})
                bins.append({"token": f"{fam}_{nm}", "n_persons": n, "mean_person_phi": round(v, 5)})
            if len(bins) >= 2:
                vs = [x["mean_person_phi"] for x in bins]
                families.append({"family": fam, "bins": bins, "bin_spread": round(max(vs) - min(vs), 5),
                                 "mean_abs_all": round(float(np.abs(ph[:, j]).mean()), 5)})
        total = np.abs(ph).sum(1).mean()
        variables = sorted(({"variable": P[j], "group": GROUP_OF.get(P[j], "other"),
                             "mean_abs_all": round(float(np.abs(ph[:, j]).mean()), 5),
                             "share_abs": round(float(np.abs(ph[:, j]).mean() / total), 4) if total else None}
                            for j in range(len(P))), key=lambda x: -x["mean_abs_all"])
        a5 = Z["age5"][m, k]
        boot = [float(a5[rng.integers(0, len(a5), len(a5))].mean()) for _ in range(N_BOOT)]
        out["questions"][q] = {"n_at_risk": int(m.sum()), "base_value_mean": round(float(Z["base"][m, k].mean()), 5),
                               "f_full_mean": round(float(Z["full"][m, k].mean()), 5),
                               "tokens": sorted(tokens, key=lambda x: -x["mean_person_phi"]), "families": families,
                               "variables": variables,
                               "age5": {"mean": round(float(a5.mean()), 5),
                                        "ci": [round(float(np.quantile(boot, 0.025)), 5), round(float(np.quantile(boot, 0.975)), 5)],
                                        "rate_ratio_mean": round(float(np.exp(a5.mean())), 4)}}
    return out
