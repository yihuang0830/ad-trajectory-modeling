"""rollout_metrics.py -- fig3（模拟逼真度）/ fig4（扰动可信度）的汇总和画图。所有模型共用这一份代码。

    # RIS 上（读人级产物，写聚合 JSON + 图）
    python rollout_metrics.py --name delphi-nodedup --scores $ZOO_RUN_DIR/scores/delphi-nodedup__s42.csv
    # 本地（只用 JSON 重画）
    python rollout_metrics.py --replot ../models_self/delphi-nodedup

输入（RIS，$ZOO_RUN_DIR）：rollout/<name>/{fidelity_persons.json, perturb_persons.csv, run_info.json}、
（第二轮审查后，rollout_extra.py 补出的）rollout/<name>/{obs_persons.json, perturb_extra.csv, run_info_extra.json}、
traj_truth_val.json、perturb_cohort_val.csv、perturb_reference.json、truth_val.csv、主 seed 的打分表。
输出（packages/<name>/）：fidelity.json、perturb.json、fig3_rollout_fidelity.png、fig4_perturbation.png。
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ENDPOINTS, RUN_DIR  # noqa: E402
from evaluate import COLOR  # noqa: E402
from perturbations import DOSE, DOSE_VALUE, EARLY, EXPECTED, LABEL, MAIN, TOKENIZATION, stroke_early_stale  # noqa: E402
from registry import DELPHI  # noqa: E402

EP = list(ENDPOINTS)
DPY = 365.25
YEARS = [1, 2, 3, 4, 5]
RNG_SEED = 20261008
DEATH_GAP = 1.5     # 年。面板 A 的"对称删失"口径：死亡离最后一次访视超过这么久 = 失访之后才死，和失访后还活着的人
                    # 一样删失在最后一次访视（一个年度访视间隔 + 半年余量）。第二轮审查后给死亡加上；2026-10-12 代码审查后
                    # AD 也用（原口径里这些人在 AD 那条线上一直算"在风险中、没得 AD"直到去世，可失访那段时间根本观察不到 AD）。
                    # 原口径一并报
VISITHEADS = {m[0] for m in DELPHI if "visitheads" in m[4]}


def _aj(entries):
    """Aalen-Johansen（入组时间 0）。entries: [(时间(年), 类型)]，1 = 目标事件，2 = 竞争事件，0 = 删失。"""
    sys.path.insert(0, os.path.join(HERE, "..", "models_code"))
    from radc_delphi.engine import aalen_johansen
    t = np.array([e[0] for e in entries], float)
    k = np.array([e[1] for e in entries], int)
    return aalen_johansen(t, k, np.zeros(len(t)), YEARS)[0]


def _obs_entries(traj, pp, kind, rule="all"):
    """面板 A 真实一侧的输入（给 _aj）：每人一条 (距第一次访视的年数, 类型)。
    kind="ad"：AD 诊断 = 1，没诊断就死了 = 2（竞争事件），否则删失在最后一次访视 = 0；kind="death"：死亡 = 1，否则删失 = 0。
    rule="sym"（对称删失）：死在最后一次访视之后超过 DEATH_GAP 年的人当作失访，删失在最后一次访视。"""
    out = []
    for p in pp:
        d = traj[str(p)]
        t0, death, ad, last = d["t0"], d["death"], d["ad"], d["last"]
        if rule == "sym" and death is not None and death - last > DEATH_GAP * DPY:
            death = None
        if kind == "ad":
            if ad is not None and ad > t0 and (death is None or ad <= death):
                out.append(((ad - t0) / DPY, 1))
            elif death is not None:
                out.append(((death - t0) / DPY, 2))
            else:
                out.append(((last - t0) / DPY, 0))
        else:
            out.append((((death if death is not None else last) - t0) / DPY, 1 if death is not None else 0))
    return out


# ----------------------------------------------------------------------------- fig3 汇总

def _infu(d, k):
    """第 k 年那次访视（t0 + k·365 天）时还在随访里：那时或之后还有访视。"""
    return d["last"] >= d["t0"] + k * 365


def _gap_stats(df, pp):
    n = float(df.loc[pp, "gap_n"].sum())
    if n == 0:
        return None
    m = df.loc[pp, "gap_sum"].sum() / n
    sd = np.sqrt(max(df.loc[pp, "gap_sumsq"].sum() / n - m * m, 0.0))
    return {"n_gaps": int(n), "mean_days": float(m), "cv": float(sd / m), "share_gt_400d": float(df.loc[pp, "gap_gt400"].sum() / n)}


def fidelity(name, scores_csv, B=200):
    run = os.path.join(RUN_DIR, "rollout", name)
    rows = json.load(open(os.path.join(run, "fidelity_persons.json")))
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    fr = pd.DataFrame(rows).set_index("pid")
    obp = os.path.join(run, "obs_persons.json")
    ob = pd.DataFrame(json.load(open(obp))).set_index("pid") if os.path.exists(obp) else None
    pids = [int(p) for p in traj if int(p) in fr.index]          # 两边都有的人（正式运行时 = 全部 N 人）
    # ---- A：累计发生率
    no_ad0 = [p for p in pids if not (traj[str(p)]["ad"] is not None and traj[str(p)]["ad"] <= traj[str(p)]["t0"])]
    rng = np.random.default_rng(RNG_SEED)
    res = {"n_persons": len(pids), "n_missing_sim": len(traj) - len(pids), "n_mc": int(fr.n_mc.iloc[0])}
    for kind, pp in (("ad", no_ad0), ("death", pids)):
        obs = _aj(_obs_entries(traj, pp, kind))
        boots = np.array([_aj(_obs_entries(traj, rng.choice(pp, len(pp)), kind)) for _ in range(B)])
        sim = np.array([fr.loc[pp, f"cif_{kind}_{k}"].mean() for k in YEARS])
        sim_b = np.array([[fr.loc[rng.choice(pp, len(pp)), f"cif_{kind}_{k}"].mean() for k in YEARS] for _ in range(B)])
        res[f"cif_{kind}"] = {"obs": obs.tolist(), "obs_lo": np.percentile(boots, 2.5, 0).tolist(),
                              "obs_hi": np.percentile(boots, 97.5, 0).tolist(), "sim": sim.tolist(),
                              "sim_lo": np.percentile(sim_b, 2.5, 0).tolist(), "sim_hi": np.percentile(sim_b, 97.5, 0).tolist(),
                              "ratio_5y": float(sim[-1] / obs[-1]) if obs[-1] > 0 else None, "n": len(pp)}
    # 对称删失口径（见 DEATH_GAP；结论以它为准）。死亡沿用原来的随机数流；AD 是 2026-10-12 加的，用单独的随机数流，
    # 这样以前就有的字段（包括后面面板 B / C 的 bootstrap）重跑时逐位不变，可以拿来核对。
    for kind, pp, g in (("death", pids, rng), ("ad", no_ad0, np.random.default_rng(RNG_SEED + 1))):
        r = res[f"cif_{kind}"]
        obs = _aj(_obs_entries(traj, pp, kind, "sym"))
        boots = np.array([_aj(_obs_entries(traj, g.choice(pp, len(pp)), kind, "sym")) for _ in range(B)])
        r.update({"obs_sym": obs.tolist(), "obs_sym_lo": np.percentile(boots, 2.5, 0).tolist(),
                  "obs_sym_hi": np.percentile(boots, 97.5, 0).tolist(),
                  "ratio_5y_sym": float(r["sim"][-1] / obs[-1]) if obs[-1] > 0 else None,
                  "sym_rule": f"deaths more than {DEATH_GAP} y after the last visit are censored at the last visit"})
    # ---- B / C 的原口径（第二轮审查前；只作记录）：观测一侧 = 那一年确实测了的人 / 两年都测了的访视对
    for fam, col in (("mmse", 1), ("cogn", 2)):
        o, s, n = [], [], []
        for k in YEARS:
            ob_, sb_ = [], []
            for p in pids:
                d = traj[str(p)]
                v = [x for x in d["visits"] if x[0] == d["t0"] + k * 365 and x[col] is not None]
                den = fr.loc[p, f"{fam}_low_den_{k}"]
                if v and den > 0:
                    ob_.append(v[0][col] == 0)
                    sb_.append(fr.loc[p, f"{fam}_low_num_{k}"] / den)
            o.append(float(np.mean(ob_)) if ob_ else None)
            s.append(float(np.mean(sb_)) if sb_ else None)
            n.append(len(ob_))
        res[f"prev_{fam}_v1"] = {"obs": o, "sim": s, "n": n}
        num = den = 0
        for p in pids:
            d = traj[str(p)]
            by_age = {x[0]: x[col] for x in d["visits"] if x[col] is not None}
            for j in range(5):
                a0, a1 = d["t0"] + j * 365, d["t0"] + (j + 1) * 365
                if a0 in by_age and a1 in by_age:
                    den += 1
                    num += by_age[a0] != by_age[a1]
        obs_rate = num / den if den else None
        sim_rate = fr[f"{fam}_chg_num"].sum() / max(fr[f"{fam}_chg_den"].sum(), 1)
        res[f"change_{fam}_v1"] = {"obs": obs_rate, "sim": float(sim_rate), "obs_pairs": int(den),
                                   "ratio": float(sim_rate / obs_rate) if obs_rate else None}
    # ---- B / C：两边同一种读法（第二轮审查后）。观测一侧 = 这个人自己的观测流（同一份分词），用和生成轨迹
    # 完全相同的读出（沿用最近一次发射的箱）；只数那一年还在随访里的人 / 人年，模拟一侧只用同一批人、同一批年。
    if ob is not None and "notok_den_1" in fr.columns:
        res["readout"] = "same readout on both sides: carry-forward state in the model's own tokenization; person-years in follow-up"
        for fam in ("mmse", "cogn"):
            o, s, n = [], [], []
            for k in YEARS:
                ob_, sb_ = [], []
                for p in pids:
                    if _infu(traj[str(p)], k) and ob.loc[p, f"{fam}_low_den_{k}"] > 0 and fr.loc[p, f"{fam}_low_den_{k}"] > 0:
                        ob_.append(float(ob.loc[p, f"{fam}_low_num_{k}"]))
                        sb_.append(fr.loc[p, f"{fam}_low_num_{k}"] / fr.loc[p, f"{fam}_low_den_{k}"])
                o.append(float(np.mean(ob_)) if ob_ else None)
                s.append(float(np.mean(sb_)) if sb_ else None)
                n.append(len(ob_))
            res[f"prev_{fam}"] = {"obs": o, "sim": s, "n": n}

        def rate(cells):
            """cells: {pid: [(观测 0/1, 模拟比例), ...]} -> 合并的观测率、模拟率、比值，按人 bootstrap 的比值 CI。"""
            pp = [p for p in cells if cells[p]]
            o = np.array([sum(c[0] for c in cells[p]) for p in pp], float)
            sm = np.array([sum(c[1] for c in cells[p]) for p in pp], float)
            m = np.array([len(cells[p]) for p in pp], float)
            ro_, rs_ = o.sum() / m.sum(), sm.sum() / m.sum()
            bs = []
            for _ in range(B):
                ii = rng.integers(0, len(pp), len(pp))
                if o[ii].sum() > 0:
                    bs.append((sm[ii].sum() / m[ii].sum()) / (o[ii].sum() / m[ii].sum()))
            return {"obs": float(ro_), "sim": float(rs_), "ratio": float(rs_ / ro_) if ro_ > 0 else None,
                    "ratio_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
                    "person_years": int(m.sum()), "persons": len(pp)}
        for fam in ("mmse", "cogn"):
            cells = {p: [] for p in pids}
            for p in pids:
                for k in range(5):
                    if _infu(traj[str(p)], k + 1) and ob.loc[p, f"{fam}_chg_den_k{k}"] > 0 and fr.loc[p, f"{fam}_chg_den_k{k}"] > 0:
                        cells[p].append((ob.loc[p, f"{fam}_chg_num_k{k}"],
                                         fr.loc[p, f"{fam}_chg_num_k{k}"] / fr.loc[p, f"{fam}_chg_den_k{k}"]))
            res[f"change_{fam}"] = rate(cells)
        cells = {p: [] for p in pids}
        for p in pids:
            for k in YEARS:
                if _infu(traj[str(p)], k) and ob.loc[p, f"notok_den_{k}"] > 0 and fr.loc[p, f"notok_den_{k}"] > 0:
                    cells[p].append((ob.loc[p, f"notok_num_{k}"], fr.loc[p, f"notok_num_{k}"] / fr.loc[p, f"notok_den_{k}"]))
        res["no_token_year"] = rate(cells)
        res["gaps"] = {"obs": _gap_stats(ob, pids), "sim": _gap_stats(fr, pids)}
    else:
        res["readout"] = "v1 (before audit round 2): observed = measured at that year / both years measured"
        for fam in ("mmse", "cogn"):
            res[f"prev_{fam}"] = res[f"prev_{fam}_v1"]
            res[f"change_{fam}"] = res[f"change_{fam}_v1"]
    # ---- D：第一次访视站点上的 2 年 AUC：rollout vs 直接读出
    base = truth.merge(pd.DataFrame({"pid": pids, "t": [traj[str(p)]["t0"] for p in pids]}), on=["pid", "t"])
    sc = pd.read_csv(scores_csv)
    base = base.merge(sc, on=["pid", "t"], how="left")
    res["auc_baseline"] = {}
    for e in EP:
        y = base[f"y_{e}_2y"].to_numpy()
        ok = y >= 0
        roll = fr.loc[base.pid[ok], f"risk2y_{e}"].to_numpy()
        rcol = next((c for c in (f"rank_{e}", f"logit_{e}", f"risk_{e}_2y") if c in base.columns), None)
        direct = base.loc[ok, rcol].to_numpy()
        yy = y[ok]
        res["auc_baseline"][e] = {"n": int(ok.sum()), "n_pos": int(yy.sum()),
                                  "rollout": float(roc_auc_score(yy, roll)), "direct": float(roc_auc_score(yy, direct))}
    res["auc_baseline"]["macro_rollout"] = float(np.mean([res["auc_baseline"][e]["rollout"] for e in EP]))
    res["auc_baseline"]["macro_direct"] = float(np.mean([res["auc_baseline"][e]["direct"] for e in EP]))
    return res


# ----------------------------------------------------------------------------- fig4 汇总

EPS = 1e-12        # 风险用 -expm1 算，精度到 1e-300；碰到这个截断就说明读法饱和了（下面有断言）


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


NO_RESP = 0.05      # |平均 log OR| 小于它 = "无响应"（既不算对也不算错；2026-10-08 审查后加）


def perturbation(name, B=500):
    """跨家族可比的主读法 = **直接读出，每个训练 seed 单独算，再取平均**（2026-10-08 审查后修订：原来 Delphi 用单 seed 的
    rollout、当时的 landmark 模型（2026-11-01 已删）用 5 seed 集成，两种读法放在同一列里不公平）。rollout 的扰动响应单独算（rollout 字段，fig3 面板 E）。"""
    run = os.path.join(RUN_DIR, "rollout", name)
    df = pd.read_csv(os.path.join(run, "perturb_persons.csv"))
    info = json.load(open(os.path.join(run, "run_info.json")))
    n_pert = int(info.get("n_pert", 64))     # 每人每组的轨迹数（run_rollout_delphi --n_pert；risks() 里平滑成 (k+0.5)/(n+1)）
    cohort = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv")).set_index("pid")
    ref = json.load(open(os.path.join(RUN_DIR, "perturb_reference.json")))
    seeds = sorted({c.split("_")[1] for c in df.columns if c.startswith("dirrisk_s")}, key=lambda x: int(x[1:]))
    binned = name.startswith("delphi")
    rng = np.random.default_rng(RNG_SEED)
    arms = {p: g.set_index("pid") for p, g in df.groupby("pert")}
    orig = arms["orig"]
    ALL = [p for p, _, _ in MAIN] + [x for x in DOSE if x != "mmse25"] + [EARLY[0]]
    # visit_heads：lm_head 的 logit 不是速率，1 - exp(-exp(logit)·2 年) 饱和成 1，log OR 没有定义（第二轮审查：原来
    # 这里把饱和的风险截断后取 logit，死亡和 MMSE 的效应全是 0，表 2 里"大量无响应"描述的是读法、不是模型）。
    # 这种模型只描述原始 logit 的差（AUC 排序用的那个分数），不打分、不进表 2。
    defined = name not in VISITHEADS
    if defined:
        for c in [c for c in df.columns if c.startswith("dirrisk_")]:
            v = df[c].to_numpy()
            assert ((v > EPS) & (v < 1 - EPS)).all(), f"{name} {c}: 直接读出的风险碰到截断 -> 效应会被压成 0"

    def persons(pert, e):
        if pert not in arms:
            return []
        return [q for q in arms[pert].index if q in orig.index and bool(cohort.loc[q, f"risk_{e}"])]

    def dl(pert, e, s, pp):
        if not defined:
            c = f"dir_{s}_{e}"
            return arms[pert].loc[pp, c].to_numpy() - orig.loc[pp, c].to_numpy()
        c = f"dirrisk_{s}_{e}"
        return _logit(arms[pert].loc[pp, c].to_numpy()) - _logit(orig.loc[pp, c].to_numpy())

    eff = {}
    for pert in ALL:
        eff[pert] = {}
        for e in EP:
            pp = persons(pert, e)
            if not pp:
                eff[pert][e] = None
                continue
            per = np.array([dl(pert, e, s, pp) for s in seeds])               # (n_seed, n_person)
            vals = per.mean(1)
            m = per.mean(0)                                                   # 每人：各 seed 的平均
            bs = [m[rng.integers(0, len(m), len(m))].mean() for _ in range(B)]
            sg = np.sign(vals) * (np.abs(vals) >= NO_RESP)
            eff[pert][e] = {"dlogit": float(vals.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                            "seed_vals": vals.tolist(), "n": len(pp),
                            "seed_sign": (int(sg[0]) if len(vals) >= 2 and np.all(sg == sg[0]) else (0 if len(vals) >= 2 else None)),
                            "drisk": float(np.mean([(arms[pert].loc[pp, f"dirrisk_{s}_{e}"].to_numpy()
                                                     - orig.loc[pp, f"dirrisk_{s}_{e}"].to_numpy()).mean() for s in seeds])),
                            "risk_orig": float(np.mean([orig.loc[pp, f"dirrisk_{s}_{e}"].mean() for s in seeds]))}
    out = {"readout": "direct, per training seed, averaged", "seeds": seeds, "binned_mmse": binned,
           "no_response_threshold": NO_RESP, "effects": eff, "readout_defined": defined,
           "stroke_early_rule": info.get("stroke_early_rule")}
    if not defined:
        out["readout"] = "UNDEFINED (visit_heads: lm_head logits are not rates); effects = raw logit differences, not scored"
        out["reference"] = {e: ref["endpoints"][e]["perturb"] for e in EP}
        out["rollout"] = None
        return out

    # ---- 方向：每个 seed 单独打分（对 / 无响应 / 错），再平均
    EXP24 = {p: ({"ad": 1, "death": 1, "mmse": 1, "cogn": 1} if p == "educ" else dict(v)) for p, v in EXPECTED.items()}

    def datasign(pert, e):
        rv = ref["endpoints"][e]["perturb"].get(pert)
        if rv is None or not (rv["ci"][0] > 0 or rv["ci"][1] < 0):
            return None
        return 1 if rv["ci"][0] > 0 else -1

    def score(target):
        per_seed = []
        for k, s in enumerate(seeds):
            right = nore = wrong = cells = 0
            for pert, _, _ in MAIN:
                for e in EP:
                    sgn = target(pert, e)
                    v = eff[pert][e]
                    if sgn is None or v is None:
                        continue
                    cells += 1
                    x = v["seed_vals"][k]
                    if abs(x) < NO_RESP:
                        nore += 1
                    elif np.sign(x) == sgn:
                        right += 1
                    else:
                        wrong += 1
            per_seed.append({"right": right, "no_response": nore, "wrong": wrong, "cells": cells})
        allr = None
        if len(seeds) >= 2:
            allr = sum(1 for pert, _, _ in MAIN for e in EP
                       if target(pert, e) is not None and eff[pert][e] is not None and eff[pert][e]["seed_sign"] == target(pert, e))
        mean = {k: float(np.mean([d[k] for d in per_seed])) for k in ("right", "no_response", "wrong")}
        return {"cells": per_seed[0]["cells"] if per_seed else 0, **mean, "per_seed": per_seed, "all_seeds_right": allr}
    out["direction"] = {"prespec": score(lambda p, e: EXPECTED[p].get(e)),
                        "prespec_original24": score(lambda p, e: EXP24[p].get(e)),
                        "data": score(datasign)}

    # ---- 剂量反应：只算输入真的变了的台阶（分箱模型 29 和 27 同箱，那一步不算）
    def bin_(v):
        return 0 if v < 24 else (1 if v < 27 else 2)
    dose = {}
    tot = mono = 0
    for e in EP:
        levels = DOSE if e != "mmse" else DOSE[:3]
        pp = [q for q in persons(levels[0], e) if all(q in arms[x].index for x in levels)]
        if not pp:
            continue
        means = [float(np.mean([arms[x].loc[pp, f"dirrisk_{s}_{e}"].mean() for s in seeds])) for x in levels]
        steps = []
        for i in range(len(levels) - 1):
            va, vb = DOSE_VALUE[levels[i]], DOSE_VALUE[levels[i + 1]]
            changes = (bin_(va) != bin_(vb)) if binned else True
            steps.append({"from": va, "to": vb, "input_changes": changes, "delta": means[i + 1] - means[i]})
            if changes:
                tot += 1
                mono += means[i + 1] - means[i] > 0
        dose[e] = {"levels": [DOSE_VALUE[x] for x in levels], "mean_risk": means, "n": len(pp), "steps": steps}
    dose["monotone_frac"] = mono / tot if tot else None
    dose["input_changing_steps"] = tot
    out["dose"] = dose

    # ---- 早期 vs 近期
    er = {}
    ident = True
    for e in EP:
        a, b = eff[EARLY[0]][e], eff[EARLY[1]][e]
        if a is None or b is None:
            er[e] = None
            continue
        er[e] = {"early": a["dlogit"], "recent": b["dlogit"]}
        # 1e-6 以内 = 浮点误差（看不到卒中、或只看当前访视的 snapshot 模型只会差浮点误差；真有差别的模型远在门槛之上）
        ident &= max(abs(x - y) for x, y in zip(a["seed_vals"], b["seed_vals"])) < 1e-6
    out["early_vs_recent"] = er
    out["early_equals_recent_by_construction"] = bool(ident)

    # ---- 和数据对照（描述性，不排名）：每个 seed 单独算，再平均
    def agree_for(vals_of):
        mx, rx = [], []
        for pert, _, _ in MAIN:
            for e in EP:
                rv = ref["endpoints"][e]["perturb"].get(pert)
                if rv is None or eff[pert][e] is None:
                    continue
                mx.append(vals_of(eff[pert][e]))
                rx.append(rv["dlogit"])
        mx, rx = np.array(mx), np.array(rx)
        if len(mx) < 3 or np.std(mx) == 0:
            return None
        return {"r": float(np.corrcoef(mx, rx)[0, 1]), "slope": float(np.polyfit(rx, mx, 1)[0]), "cells": len(mx),
                "model": mx.tolist(), "data": rx.tolist()}
    per = [agree_for(lambda v, k=k: v["seed_vals"][k]) for k in range(len(seeds))]
    per = [x for x in per if x]
    meanag = agree_for(lambda v: v["dlogit"])
    out["agreement"] = None if not meanag else {**meanag, "r_seed_mean": float(np.mean([x["r"] for x in per])),
                                                "slope_seed_mean": float(np.mean([x["slope"] for x in per])),
                                                "r_seed_range": [float(min(x["r"] for x in per)), float(max(x["r"] for x in per))]}
    # ---- 跨 seed 稳定性
    ratios = [np.std(v["seed_vals"], ddof=1) / abs(np.mean(v["seed_vals"]))
              for pert in EXPECTED for v in eff[pert].values() if v and len(v["seed_vals"]) >= 2 and abs(np.mean(v["seed_vals"])) >= NO_RESP]
    out["seed_stability"] = {"median_sd_over_mean": float(np.median(ratios)) if ratios else None, "cells": len(ratios),
                             "n_seeds": len(seeds)}

    # ---- rollout 的扰动响应（只有能生成轨迹的模型）：每组把所有人的轨迹合并，logit(合并风险) 的差
    if "roll2_ad" in df.columns:
        s0 = seeds[0]                                   # = rollout 用的那个 checkpoint（seed 42）

        def hits(arm, pp, e):
            return np.rint(arm.loc[pp, f"roll2_{e}"].to_numpy() * (n_pert + 1) - 0.5)

        def pooled(hp, ho):
            n = len(hp) * n_pert
            return float(_logit((hp.sum() + 0.5) / (n + 1)) - _logit((ho.sum() + 0.5) / (n + 1)))

        def boot(hp, ho):
            bs = []
            for _ in range(B):
                ii = rng.integers(0, len(hp), len(hp))
                bs.append(pooled(hp[ii], ho[ii]))
            return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

        def pooled_direct(pert, e, pp, target):
            """同一个估计量用在直接读出上（第二轮审查后）：两组的速率乘同一个常数 c，使不扰动组的平均风险 = rollout
            不扰动组的合并风险，再取 logit(合并风险) 的差。一个把每个人的直接效应原样传下去的生成器，rollout 就该等于它。"""
            Ho = np.exp(orig.loc[pp, f"dir_{s0}_{e}"].to_numpy()) * 2 * DPY
            Hp = np.exp(arms[pert].loc[pp, f"dir_{s0}_{e}"].to_numpy()) * 2 * DPY
            lo, hi = 1e-9, 1e9
            for _ in range(200):
                c = np.sqrt(lo * hi)
                if (-np.expm1(-c * Ho)).mean() < target:
                    lo = c
                else:
                    hi = c
            return float(_logit((-np.expm1(-c * Hp)).mean()) - _logit((-np.expm1(-c * Ho)).mean()))
        ro = {}
        xs, xm, ys, cls, labs = [], [], [], [], []
        for pert, _, _ in MAIN:
            ro[pert] = {}
            for e in EP:
                pp = persons(pert, e)
                if not pp:
                    ro[pert][e] = None
                    continue
                hp, ho = hits(arms[pert], pp, e), hits(orig, pp, e)
                n = len(pp) * n_pert
                v = {"dlogit": pooled(hp, ho), "ci": boot(hp, ho), "p_orig": float(ho.sum() / n), "p_pert": float(hp.sum() / n),
                     "n_persons": len(pp), "direct_mean": eff[pert][e]["seed_vals"][0],
                     "direct_pooled": pooled_direct(pert, e, pp, (ho.sum() + 0.5) / (n + 1))}
                # 逐格：两边都有效应（rollout |log OR| >= 0.05 且 CI 不含 0；直接读出 |log OR| >= 0.05）-> 同号 / 反号；
                # 只有直接读出有 -> 丢失；只有 rollout 有 -> 凭空出现（生成器自己造出来的效应）；都没有 -> none
                rclear = abs(v["dlogit"]) >= NO_RESP and (v["ci"][0] > 0 or v["ci"][1] < 0)
                dclear = abs(v["direct_pooled"]) >= NO_RESP
                if rclear and dclear:
                    v["sign"] = "same" if np.sign(v["dlogit"]) == np.sign(v["direct_pooled"]) else "opposite"
                else:
                    v["sign"] = "rollout_only" if rclear else ("lost" if dclear else "none")
                ro[pert][e] = v
                xs.append(v["direct_pooled"])
                xm.append(v["direct_mean"])
                ys.append(v["dlogit"])
                cls.append(v["sign"])
                labs.append(f"{pert}->{e}")
        xs, xm, ys = np.array(xs), np.array(xm), np.array(ys)

        def rscore(target):
            right = nore = wrong = cells = 0
            for pert, _, _ in MAIN:
                for e in EP:
                    sgn = target(pert, e)
                    v = ro[pert][e]
                    if sgn is None or v is None:
                        continue
                    cells += 1
                    if abs(v["dlogit"]) < NO_RESP:
                        nore += 1
                    elif np.sign(v["dlogit"]) == sgn:
                        right += 1
                    else:
                        wrong += 1
            return {"cells": cells, "right": right, "no_response": nore, "wrong": wrong}
        out["rollout"] = {"effects": ro, "vs_direct_seed": s0, "direct_pooled": xs.tolist(), "direct": xm.tolist(),
                          "rollout": ys.tolist(), "cell_label": labs, "cell_sign": cls,
                          "slope_vs_direct": float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 2 else None,
                          "r_vs_direct": float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else None,
                          "slope_vs_direct_mean": float(np.polyfit(xm, ys, 1)[0]) if len(xs) > 2 else None,
                          "slope_perfect_transmitter_vs_direct_mean": float(np.polyfit(xm, xs, 1)[0]) if len(xs) > 2 else None,
                          "sign_agreement": {"cells": len(cls), **{k: cls.count(k) for k in ("same", "opposite", "lost", "rollout_only", "none")},
                                             "opposite_cells": [l for l, c in zip(labs, cls) if c == "opposite"],
                                             "rollout_only_cells": [l for l, c in zip(labs, cls) if c == "rollout_only"]},
                          "direction_prespec": rscore(lambda p, e: EXPECTED[p].get(e)),
                          "direction_data": rscore(datasign)}
        # 卒中反号的机制检验（rollout_extra.py）：不加卒中的对照组在生成时也禁采 STROKE
        ext = os.path.join(RUN_DIR, "rollout", name, "perturb_extra.csv")
        if os.path.exists(ext):
            ex = pd.read_csv(ext)
            nost = ex[ex.pert == "orig_nostroke"].set_index("pid")
            st = {}
            for e in EP:
                pp = [q for q in persons("stroke", e) if q in nost.index]
                hs, ho, hn = hits(arms["stroke"], pp, e), hits(orig, pp, e), hits(nost, pp, e)
                n = len(pp) * n_pert
                st[e] = {"vs_orig": pooled(hs, ho), "vs_orig_ci": boot(hs, ho),
                         "vs_nostroke": pooled(hs, hn), "vs_nostroke_ci": boot(hs, hn),
                         "p_stroke": float(hs.sum() / n), "p_orig": float(ho.sum() / n), "p_nostroke": float(hn.sum() / n),
                         "n_persons": len(pp)}
            # 第二个假设：两组都从站点后 1 天的查询 token 开始生成（rollout_extra.py --query_start）
            exq = os.path.join(RUN_DIR, "rollout", name, "perturb_extra_q.csv")
            if os.path.exists(exq):
                q = pd.read_csv(exq)
                oq, sq = q[q.pert == "orig_q"].set_index("pid"), q[q.pert == "stroke_q"].set_index("pid")
                for e in EP:
                    pp = [x for x in persons("stroke", e) if x in oq.index and x in sq.index]
                    hs, ho = hits(sq, pp, e), hits(oq, pp, e)
                    n = len(pp) * n_pert
                    st[e].update({"query_start": pooled(hs, ho), "query_start_ci": boot(hs, ho),
                                  "query_start_p": [float(hs.sum() / n), float(ho.sum() / n)],
                                  "direct_pooled": ro["stroke"][e]["direct_pooled"] if ro["stroke"][e] else None})
            ri = json.load(open(os.path.join(RUN_DIR, "rollout", name, "run_info_extra.json")))
            out["rollout"]["stroke_norepeat_test"] = {
                "effects": st, "comparison_arm": "no stroke added, STROKE blocked during generation (same random numbers)",
                "orig_arm_reproduced": [ri.get("orig_arm_reproduced_persons"), ri.get("orig_arm_persons")],
                "fidelity_v1_fields_reproduced": ri.get("fidelity_v1_fields_reproduced")}
    else:
        out["rollout"] = None
    out["reference"] = {e: ref["endpoints"][e]["perturb"] for e in EP}
    out["reference_spec"] = {k: ref.get(k) for k in ("fit_split", "encoding", "n_landmarks")}
    return out


# ----------------------------------------------------------------------------- 图

def no_generator_reason(name, info):
    """fig3 不适用时图上写的原因（英文：RIS 上的字体没有中文，原始报错直接画出来是一串方框；原文留在 rollout_info.json）。"""
    if TOKENIZATION.get(name) == "snapshot":
        return "snapshot tokenization: AD_DX / Death are re-emitted at every visit, which the figure-2 generator does not allow"
    if name in VISITHEADS:
        return "visit_heads: lm_head logits are not event rates, so the figure-2 generator cannot sample from them"
    if name not in TOKENIZATION:
        return "not a generative model: it predicts risks directly and has no trajectory generator"
    return info.get("generator_error", "").encode("ascii", "ignore").decode()[:120]


def fig3(name, fid, info, path, pert=None):
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.7))
    if fid is None:
        for ax in axes:
            ax.axis("off")
        axes[2].text(0.5, 0.5, f"{name}\nno trajectory generator -> rollout fidelity not applicable\n"
                     f"({no_generator_reason(name, info)})", ha="center", va="center", fontsize=10,
                     transform=axes[2].transAxes, wrap=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    ax = axes[0]
    for kind, c in (("ad", COLOR["ad"]), ("death", COLOR["death"])):
        r = fid[f"cif_{kind}"]
        Y0 = [0] + YEARS
        ax.plot(Y0, [0] + r["obs"], "-", color=c, lw=2, label=f"{kind} observed")
        ax.fill_between(Y0, [0] + r["obs_lo"], [0] + r["obs_hi"], color=c, alpha=0.15)
        ax.plot(Y0, [0] + r["sim"], "--o", color=c, ms=3.5, label=f"{kind} simulated")
        if "obs_sym" in r:
            ax.plot(Y0, [0] + r["obs_sym"], ":", color=c, lw=1.5, label=f"{kind} observed, symmetric censoring")
    ax.set_title("A  cumulative incidence from first visit", fontsize=9)
    ax.set_xlabel("years since first visit", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax = axes[1]
    for fam, c, lab in (("mmse", COLOR["mmse"], "MMSE<24"), ("cogn", COLOR["cogn"], "cognition<-1")):
        r = fid[f"prev_{fam}"]
        ax.plot(YEARS, [np.nan if v is None else v for v in r["obs"]], "-", color=c, lw=2, label=f"{lab} observed")
        ax.plot(YEARS, [np.nan if v is None else v for v in r["sim"]], "--o", color=c, ms=3.5, label=f"{lab} simulated")
    ax.set_title("B  share impaired, by year" + ("\n(same readout both sides; persons still in follow-up)"
                                               if "no_token_year" in fid else " (among measured)"), fontsize=8.5)
    ax.set_xlabel("years since first visit", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    if "no_token_year" in fid:
        keys = [("change_mmse", "MMSE bin\nchanged"), ("change_cogn", "cognition bin\nchanged"),
                ("no_token_year", "no token at all\nin the year")]
        xs = np.arange(len(keys))
        ov = [fid[k]["obs"] for k, _ in keys]
        sv = [fid[k]["sim"] for k, _ in keys]
        ax.bar(xs - 0.18, ov, 0.36, color="#4d4d4d", label="observed")
        ax.bar(xs + 0.18, sv, 0.36, color="#e08214", label="simulated")
        for i, (k, _) in enumerate(keys):
            r = fid[k]
            ci = r.get("ratio_ci")
            ax.text(i, max(ov[i], sv[i]) + 0.015, ("x—" if r["ratio"] is None else f"x{r['ratio']:.2f}")
                    + (f"\n[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs, [lab for _, lab in keys], fontsize=7.5)
        ax.set_ylim(0, max(ov + sv) * 1.45)
        ax.set_ylabel("share of person-years", fontsize=8)
        ax.set_title("C  year to year, same readout both sides\n(own tokenization; x = simulated / observed)", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    else:
        labs = ["MMSE bin", "cognition bin"]
        vals = [fid["change_mmse"]["ratio"], fid["change_cogn"]["ratio"]]
        ax.bar(labs, [np.nan if v is None else v for v in vals], color=[COLOR["mmse"], COLOR["cogn"]], width=0.5)
        ax.axhline(1, color="black", lw=1, ls=":")
        ax.set_title("C  year-to-year change rate: simulated / observed", fontsize=9)
    ax = axes[3]
    a = fid["auc_baseline"]
    xs = np.arange(len(EP))
    ax.bar(xs - 0.18, [a[e]["direct"] for e in EP], 0.36, color="#4d4d4d", label="direct readout")
    ax.bar(xs + 0.18, [a[e]["rollout"] for e in EP], 0.36, color="#e08214", label="from rollouts")
    ax.set_xticks(xs, ["AD", "Death", "MMSE<24", "Cogn<-1"], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="#cccccc", lw=0.8)
    ax.set_title(f"D  2-y AUC at first visit\n(macro: direct {a['macro_direct']:.3f}, rollout {a['macro_rollout']:.3f})",
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[4]
    ro = (pert or {}).get("rollout")
    if ro and ro.get("direct_pooled"):
        x, y = np.array(ro["direct_pooled"]), np.array(ro["rollout"])
        lim = max(0.5, float(np.max(np.abs(np.concatenate([x, y])))) * 1.15)
        ax.plot([-lim, lim], [-lim, lim], color="#bbbbbb", lw=1)
        ax.axhline(0, color="#eeeeee", lw=0.8)
        ax.axvline(0, color="#eeeeee", lw=0.8)
        col = {"same": "#e08214", "opposite": "#d7191c", "rollout_only": "#7b3294", "lost": "#92c5de", "none": "#cccccc"}
        nlab = 0
        for xi, yi, c, lab in zip(x, y, ro["cell_sign"], ro["cell_label"]):
            ax.scatter(xi, yi, s=20, color=col[c], zorder=3)
            if c == "opposite":                      # 只标反号的格子，标签错开避免重叠
                ax.annotate(lab.replace("->", "→"), (xi, yi), fontsize=5.5, color=col[c], xytext=(6, -10 - 8 * nlab),
                            textcoords="offset points", arrowprops=dict(arrowstyle="-", color=col[c], lw=0.4))
                nlab += 1
        for c, lab in (("same", "same sign"), ("opposite", "opposite sign"), ("lost", "lost"), ("rollout_only", "rollout only")):
            ax.scatter([], [], s=14, color=col[c], label=lab)
        ax.legend(fontsize=6, frameon=False, loc="upper left", handletextpad=0.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        sa = ro["sign_agreement"]
        t3 = ""
        stt = ro.get("stroke_norepeat_test")
        if stt:
            d = stt["effects"]["death"]
            t3 = f"\nstroke→death {d['vs_orig']:+.2f}; future strokes blocked {d['vs_nostroke']:+.2f}"
            if "query_start" in d:
                t3 += f"\nstart at query token {d['query_start']:+.2f} (direct readout {d['direct_pooled']:+.2f})"
        ax.set_title(f"E  effects through the generator (slope {ro['slope_vs_direct']:.2f})\nsame sign {sa['same']}, opposite "
                     f"{sa['opposite']}, lost {sa['lost']}, rollout-only {sa['rollout_only']} (of {sa['cells']})" + t3, fontsize=7.0)
        ax.set_xlabel(f"direct readout ({ro['vs_direct_seed']}), same pooled estimator, log OR", fontsize=7.5)
        ax.set_ylabel("from rollouts (pooled), log OR", fontsize=8)
    elif ro and ro.get("direct"):
        x, y = np.array(ro["direct"]), np.array(ro["rollout"])
        ax.scatter(x, y, s=20, color="#e08214")
        ax.set_title(f"E  perturbation effects through the generator (v1)\nslope vs direct {ro['slope_vs_direct']:.2f}", fontsize=8)
    else:
        ax.axis("off")
    fig.suptitle(f"{name} -- rollout fidelity (val, {fid['n_persons']} persons x {fid['n_mc']} trajectories, 5 years)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


SHORT = {"ad": "AD", "death": "Death", "mmse": "MMSE<24", "cogn": "Cogn<-1"}


def _fig4_undefined(name, pert, path):
    """读法没有定义的模型（visit_heads）：只画原始 logit 的差，灰色，不打分。"""
    eff = pert["effects"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.7, 1.3]})
    ax = axes[0]
    rows = [p for p, _, _ in MAIN]
    for i, p in enumerate(rows):
        for j, e in enumerate(EP):
            v = eff[p][e]
            if v is None:
                continue
            ax.scatter(j, i, s=40 + 220 * min(abs(v["dlogit"]), 1.5), color="#999999", zorder=3)
            ax.text(j + 0.3, i, f"{v['dlogit']:+.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [LABEL[p] for p in rows], fontsize=8)
    ax.set_xticks(range(len(EP)), [SHORT[e] for e in EP], fontsize=8)
    ax.set_xlim(-0.5, len(EP) - 0.05)
    ax.invert_yaxis()
    ax.set_title("A  raw lm_head logit difference (NOT a log OR; descriptive only)", fontsize=8)
    axes[1].axis("off")
    axes[1].text(0.02, 0.5, "Perturbation credibility is not scored for this model.\n\n"
                 "visit_heads: lm_head logits are not event rates, so the\nrisk readout 1 - exp(-exp(logit) x 2 y) saturates at 1\n"
                 "and every log OR collapses to 0. Direction, dose-response,\n"
                 "agreement with data and seed stability are undefined.", fontsize=9, va="center")
    fig.suptitle(f"{name} -- perturbation credibility: readout undefined", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig4(name, pert, path):
    if not pert.get("readout_defined", True):
        return _fig4_undefined(name, pert, path)
    eff = pert["effects"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.7, 1, 1, 1]})
    ax = axes[0]
    rows = [p for p, _, _ in MAIN]
    for i, p in enumerate(rows):
        for j, e in enumerate(EP):
            v = eff[p][e]
            if v is None:
                continue
            exp = EXPECTED[p].get(e)
            if abs(v["dlogit"]) < NO_RESP:
                col = "#bbbbbb"
            else:
                col = "#999999" if exp is None else ("#1a9641" if np.sign(v["dlogit"]) == exp else "#d7191c")
            filled = v["seed_sign"] is not None and v["seed_sign"] != 0 and v["seed_sign"] == np.sign(v["dlogit"])
            ax.scatter(j, i, s=40 + 220 * min(abs(v["dlogit"]), 1.5), color=col if filled else "white",
                       edgecolors=col, linewidths=1.5, zorder=3)
            ax.text(j + 0.3, i, f"{v['dlogit']:+.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [LABEL[p] for p in rows], fontsize=8)
    ax.set_xticks(range(len(EP)), [SHORT[e] for e in EP], fontsize=8)
    ax.set_xlim(-0.5, len(EP) - 0.05)
    ax.invert_yaxis()
    D = pert["direction"]

    def fmt(x):
        return f"{x['right']:.1f}/{x['cells']} (no resp. {x['no_response']:.1f}, wrong {x['wrong']:.1f})"
    ax.set_title(f"A  effect on 2-y risk (log OR; mean over {len(pert['seeds'])} seed(s), direct readout)\n"
                 f"pre-specified: {fmt(D['prespec'])}; original 24-cell: {D['prespec_original24']['right']:.1f}/"
                 f"{D['prespec_original24']['cells']}\nsame sign as data: {fmt(D['data'])}", fontsize=7.2)
    fig.text(0.01, 0.01, f"Panel A: green = expected direction, red = opposite, grey = not pre-specified, light grey = no response "
             f"(|log OR| < {NO_RESP}); filled = every training seed gives this sign. Circle size = |effect|.", fontsize=7)
    ax = axes[1]
    dz = pert["dose"]
    for e in EP:
        if e in dz:
            ax.plot(dz[e]["levels"], dz[e]["mean_risk"], "-o", color=COLOR[e], ms=4, label=ENDPOINTS[e][0])
    ax.invert_xaxis()
    ax.set_xlabel("MMSE set at the landmark visit", fontsize=8.5)
    ax.set_ylabel("mean 2-y risk", fontsize=8.5)
    mf = dz.get("monotone_frac")
    ax.set_title(f"B  dose-response: risk rises on {'n/a' if mf is None else f'{mf:.0%}'} of the "
                 f"{dz.get('input_changing_steps', 0)} steps where the input changes"
                 + ("\n(binned MMSE: 29 and 27 are the same input)" if pert.get("binned_mmse") else ""), fontsize=7.8)
    ax.legend(fontsize=6.5, frameon=False)
    ax = axes[2]
    ag = pert["agreement"]
    if ag:
        lim = max(0.5, float(np.max(np.abs(np.array(ag["model"] + ag["data"])))) * 1.15)
        ax.plot([-lim, lim], [-lim, lim], color="#bbbbbb", lw=1)
        ax.axhline(0, color="#eeeeee", lw=0.8)
        ax.axvline(0, color="#eeeeee", lw=0.8)
        ax.scatter(ag["data"], ag["model"], s=22, color="#2c7bb6")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"C  model vs data association (descriptive)\nr = {ag['r']:.2f}, slope = {ag['slope']:.2f}"
                     f"  [per seed r {ag['r_seed_range'][0]:.2f}-{ag['r_seed_range'][1]:.2f}]", fontsize=8.0)
    ax.set_xlabel("data: conditional log OR (held-out val, piecewise-linear LR)", fontsize=7.5)
    ax.set_ylabel("model: effect (log OR)", fontsize=8.5)
    ax = axes[3]
    er = pert["early_vs_recent"]
    stale = stroke_early_stale(name, pert)
    xs = np.arange(len(EP))
    ax.bar(xs - 0.18, [np.nan if er[e] is None else er[e]["recent"] for e in EP], 0.36, color="#4d4d4d",
           label="at landmark visit")
    if stale:
        # 修正前的"早期"结果翻译错了（见 perturbations.stroke_early_stale），不画，免得被误读；顶部留白放说明
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))
        ax.text(0.5, 0.97, "'at first visit' withheld: the edit was\nmistranslated for this tokenization\n"
                "(fixed 2026-10-12; re-run pending)", transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color="#d7191c", zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    else:
        ax.bar(xs + 0.18, [np.nan if er[e] is None else er[e]["early"] for e in EP], 0.36, color="#e08214",
               label="at first visit")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs, [SHORT[e] for e in EP], fontsize=8)
    ax.set_ylabel("effect of adding a stroke (log OR)", fontsize=8)
    ttl = "D  early vs recent (same edit)"
    if pert.get("early_equals_recent_by_construction"):
        ttl += "\nidentical by construction (no cross-visit memory)"
    ax.set_title(ttl, fontsize=8.0)
    ax.legend(fontsize=7, frameon=False, loc="best")
    st = pert["seed_stability"]["median_sd_over_mean"]
    fig.suptitle(f"{name} -- perturbation credibility (direct readout, per seed; seed sd/|effect| = "
                 f"{'n/a' if st is None else f'{st:.2f}'})", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--replot", default=None, help="打包盒目录：只用 fidelity.json / perturb.json 重画")
    args = ap.parse_args()
    if args.replot:
        d = args.replot
        name = os.path.basename(os.path.abspath(d))
        info = json.load(open(os.path.join(d, "rollout_info.json"))) if os.path.exists(os.path.join(d, "rollout_info.json")) else {}
        fid = json.load(open(os.path.join(d, "fidelity.json"))) if os.path.exists(os.path.join(d, "fidelity.json")) else None
        pt = json.load(open(os.path.join(d, "perturb.json"))) if os.path.exists(os.path.join(d, "perturb.json")) else None
        fig3(name, fid, info, os.path.join(d, "fig3_rollout_fidelity.png"), pt)
        if pt:
            fig4(name, pt, os.path.join(d, "fig4_perturbation.png"))
        return
    out = args.outdir or os.path.join(RUN_DIR, "packages", args.name)
    os.makedirs(out, exist_ok=True)
    info = json.load(open(os.path.join(RUN_DIR, "rollout", args.name, "run_info.json")))
    json.dump(info, open(os.path.join(out, "rollout_info.json"), "w"), indent=1, default=str)
    fid = None
    if os.path.exists(os.path.join(RUN_DIR, "rollout", args.name, "fidelity_persons.json")):
        fid = fidelity(args.name, args.scores)
        json.dump(fid, open(os.path.join(out, "fidelity.json"), "w"), indent=1)
    pert = perturbation(args.name)
    json.dump(pert, open(os.path.join(out, "perturb.json"), "w"), indent=1)
    fig3(args.name, fid, info, os.path.join(out, "fig3_rollout_fidelity.png"), pert)
    fig4(args.name, pert, os.path.join(out, "fig4_perturbation.png"))
    if not pert["readout_defined"]:
        print(f"{args.name}: perturbation readout undefined (visit_heads)")
        return
    d, dd = pert["direction"]["prespec"], pert["direction"]["data"]
    print(f"{args.name}: prespec right {d['right']:.1f}/{d['cells']} (no resp {d['no_response']:.1f}, wrong {d['wrong']:.1f}); "
          f"data-sign {dd['right']:.1f}/{dd['cells']}; dose {pert['dose']['monotone_frac']}"
          + (f"; rollout macro AUC {fid['auc_baseline']['macro_rollout']:.3f} vs direct {fid['auc_baseline']['macro_direct']:.3f}"
             f"; CIF ratio 5y AD {fid['cif_ad']['ratio_5y']}, death {fid['cif_death']['ratio_5y']} "
             f"(sym {fid['cif_death'].get('ratio_5y_sym')}); change MMSE {fid['change_mmse']['ratio']}, "
             f"no-token-year {fid.get('no_token_year', {}).get('ratio')}" if fid else "")
          + (f"; rollout slope {pert['rollout']['slope_vs_direct']:.2f}, signs {pert['rollout']['sign_agreement']}"
             if pert.get("rollout") else ""))


if __name__ == "__main__":
    main()
