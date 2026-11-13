"""perturb.py -- 统一输入的模型（Delphi 以外）的扰动评估，和 Delphi 同一批人、同一组改动、同一份打分代码。

两套扰动（都在统一输入的特征空间里改，perturb_features / perturb_v2 的改法原样复用）：
  第二部分（表 2b，perturb.json）：perturbations.PERTS × perturb_cohort_val.csv，改法 perturb_features.apply，
    读站点上 4 个终点的 2 年风险 -> $ZOO_RUN_DIR/rollout/<名字>/perturb_persons.csv -> rollout_metrics.perturbation
  第三部分（表 6，perturb_v2.json）：perturb_v2 的 ATOMS + COMBOS × cohort_table()，改法 perturb_v2.translate_features，
    读站点上 6 道题 × 1..10 年的风险 -> $ZOO_RUN_DIR/perturb_v2/<名字>/direct.npz -> perturb_v2.aggregate_model
每个基线训练完在这些"改过的人"上算一遍风险（records 当成额外的一批人传给各自的预测函数），不另存模型。
生成模型的风险本来就是从站点模拟出来的，所以它们的"直接读出"就是 rollout（每个站点 n_mc 条，平滑成 (k + 0.5) / (n + 1)）。
"""
import json
import os

import numpy as np
import pandas as pd

import bl_data as B
from common import ENDPOINTS, RUN_DIR
from transitions import SAME_AS_PART1

EP1 = list(ENDPOINTS)                          # ad, death, mmse, cogn


def _stack(D, i, Xs, Ss, Ls):
    n = len(Xs)
    V = D["X"].shape[1]
    return {"X": np.stack(Xs).astype(np.float32), "M": D["M"][i].copy(), "S": np.stack(Ss).astype(np.float32),
            "L": np.stack(Ls).astype(np.float32), "T": D["T"][i].copy(), "pid": D["pid"][i].copy(),
            "Y": np.full((n, V, B.NQ * B.NH), -1, np.int8)}


def records(Z, meta):
    """-> {"v1": (Dp, vL, recs), "v2": (Dp, vL, rows, C)}。Dp 是一批"人"（每条记录一个），站点 = vL。"""
    import perturb_features as PF
    import perturb_v2 as PV
    from perturbations import ELIG, PERTS
    D = B.split(Z, "val")
    norm = meta["norm"]
    idx = {int(p): i for i, p in enumerate(D["pid"])}
    out = {}
    # 第二部分
    cohort = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"))
    ii, Xs, Ss, Ls, vs, recs = [], [], [], [], [], []
    for _, c in cohort.iterrows():
        i = idx[int(c.pid)]
        vL = int(np.where(D["T"][i] == int(c.tL))[0][0])
        for pert in PERTS:
            if not bool(c[ELIG[pert]]):
                continue
            X, S, L = PF.apply(pert, D["X"][i], D["M"][i], D["S"][i], D["L"][i], vL, norm)
            ii.append(i); Xs.append(X); Ss.append(S); Ls.append(L); vs.append(vL)
            recs.append({"pid": int(c.pid), "pert": pert, "tland": int(c.tL)})
    out["v1"] = (_stack(D, np.array(ii), Xs, Ss, Ls), np.array(vs), recs)
    # 第三部分
    C, _ = PV.cohort_table()
    feats = meta["features"]
    ii, Xs, Ss, Ls, vs, rows = [], [], [], [], [], []
    for pid, r in C.iterrows():
        i, vL = int(r.lm_i), int(r.vL)
        for arm in PV.ARMS:
            if not r[f"elig__{arm}"]:
                continue
            X, S, L = PV.translate_features(arm, D["X"][i], D["M"][i], D["S"][i], D["L"][i], vL, norm, feats)
            ii.append(i); Xs.append(X); Ss.append(S); Ls.append(L); vs.append(vL)
            rows.append((int(pid), arm, int(r.tL)))
    out["v2"] = (_stack(D, np.array(ii), Xs, Ss, Ls), np.array(vs), rows, C)
    return out


PRS_ARMS = (("prs_p10", 0.10), ("prs_p90", 0.90))


def prs_records(Z, meta):
    """基因实验（run.py --variant geno）：PRS 设成 train 段有基因的人的第 10 / 第 90 百分位，别的一概不动。
    人 = 第三部分扰动队列（perturb_v2.cohort_table，站点 = 第 4 次访视）里有基因的人，每人两条。
    不放进 PERTS / ATOMS：那两张表和 Delphi、所有模型的对错计数共用。-> (Dp, vL, rows, 两个水平)"""
    import perturb_v2 as PV
    D = B.split(Z, "val")
    ip, ig = meta["static"].index("prs"), meta["static"].index("geno_na")
    ref = Z["train_S"][Z["train_S"][:, ig] < 0.5, ip]
    lev = {arm: float(np.quantile(ref, q)) for arm, q in PRS_ARMS}
    C, _ = PV.cohort_table()
    ii, Ss, vs, rows = [], [], [], []
    for pid, r in C.iterrows():
        i, vL = int(r.lm_i), int(r.vL)
        if D["S"][i, ig] > 0.5:
            continue
        for arm, _ in PRS_ARMS:
            S = D["S"][i].copy()
            S[ip] = lev[arm]
            ii.append(i); Ss.append(S); vs.append(vL); rows.append((int(pid), arm, int(r.tL)))
    ii = np.array(ii)
    return _stack(D, ii, list(D["X"][ii]), Ss, list(D["L"][ii])), np.array(vs), rows, lev


APOE_ARMS = {"apoe_33": (0, 0, 0), "apoe_23": (0, 0, 1), "apoe_34": (1, 0, 0), "apoe_44": (0, 1, 0)}   # e4x1, e4x2, e2carrier


def geno2_arms(Z, meta, interaction=False):
    """基因第二版（run.py --variant geno2 / geno2nx / geno2i）的扰动臂：{臂名: {S 的列: 值}}。每臂只改一个基因变量，别的照原样。
    通路 PRS：train 段有基因的人的第 10 / 第 90 百分位；Tier 1：风险（同向替代）等位基因 0 份 / 2 份（列存的是减过均值的剂量）；
    APOE：ε3/ε3、ε2/ε3、ε3/ε4、ε4/ε4（apoe_na 设 0；有 apoe_e2x2 这一列时一并设 0）。
    interaction（geno2i，2026-11-11）：另加 2 × 2 析因臂 <变量臂>__a33 / __a34 = 这个基因变量取低 / 高，同时 APOE 统一设成
    ε3/ε3（非携带者）或 ε3/ε4（携带者）；看同一批人里这个变量的效应随 APOE 怎么变（交互）。"""
    st = meta["static"]
    ig = st.index("geno_na")
    tr = Z["train_S"][Z["train_S"][:, ig] < 0.5]
    arms = {}
    for c in st:
        if c.startswith("prs_"):
            for tag, q in PRS_ARMS:
                arms[f"{c}__{tag.split('_')[1]}"] = {c: float(np.quantile(tr[:, st.index(c)], q))}
        elif c.startswith("t1_"):
            mu = meta["geno2"]["tier1_train_mean"][c]
            arms[f"{c}__0"] = {c: 0.0 - mu}
            arms[f"{c}__2"] = {c: 2.0 - mu}
    single = list(arms)
    for a, (x1, x2, e2) in APOE_ARMS.items():
        arms[a] = {"apoe_e4x1": x1, "apoe_e4x2": x2, "apoe_e2carrier": e2, "apoe_na": 0,
                   **({"apoe_e2x2": 0} if "apoe_e2x2" in st else {})}
    if interaction:
        for x in single:
            for g in ("33", "34"):
                arms[f"{x}__a{g}"] = {**arms[x], **arms[f"apoe_{g}"]}
    return arms


def geno2_ipairs(arms):
    """交互：每个基因变量 -> (低@ε3/ε3, 高@ε3/ε3, 低@ε3/ε4, 高@ε3/ε4)。"""
    out = {}
    for lo, hi in geno2_pairs(arms):
        if lo.startswith(("prs_", "t1_")) and f"{lo}__a33" in arms:
            out[lo.rsplit("__", 1)[0]] = (f"{lo}__a33", f"{hi}__a33", f"{lo}__a34", f"{hi}__a34")
    return out


def geno2_pairs(arms):
    """要比的 (低, 高) 臂：通路 PRS p10 → p90，Tier 1 0 → 2 份，APOE ε3/ε3 → ε2/ε3 / ε3/ε4 / ε4/ε4。"""
    out = []
    for x in arms:
        if x.endswith("__p10"):
            out.append((x, x.removesuffix("__p10") + "__p90"))
        elif x.endswith("__0"):
            out.append((x, x.removesuffix("__0") + "__2"))
    return out + [("apoe_33", "apoe_23"), ("apoe_33", "apoe_34"), ("apoe_33", "apoe_44")]


def geno2_records(Z, meta, interaction=False):
    """同 prs_records：人 = 第三部分扰动队列里有基因的人，每人每臂一条。-> (Dp, vL, rows, 臂定义)"""
    import perturb_v2 as PV
    D = B.split(Z, "val")
    st = meta["static"]
    ig = st.index("geno_na")
    arms = geno2_arms(Z, meta, interaction)
    C, _ = PV.cohort_table()
    ii, Ss, vs, rows = [], [], [], []
    for pid, r in C.iterrows():
        i, vL = int(r.lm_i), int(r.vL)
        if D["S"][i, ig] > 0.5:
            continue
        for arm, setv in arms.items():
            S = D["S"][i].copy()
            for c, v in setv.items():
                S[st.index(c)] = v
            ii.append(i); Ss.append(S); vs.append(vL); rows.append((int(pid), arm, int(r.tL)))
    ii = np.array(ii)
    return _stack(D, ii, list(D["X"][ii]), Ss, list(D["L"][ii])), np.array(vs), rows, arms


def write_prs(name, R, rec):
    """基因扰动的风险 (记录, 6, 10) -> $ZOO_RUN_DIR/geno/pert/<名字>.npz（人级；臂之间的对比在 geno_compare.py / geno2_compare.py 里做）。"""
    _, _, rows, lev = rec
    od = os.path.join(RUN_DIR, "geno", "pert")
    os.makedirs(od, exist_ok=True)
    arm = np.array([r[1] for r in rows])
    F = np.clip(R, 1e-6, 1 - 1e-6)
    np.savez_compressed(os.path.join(od, f"{name}.npz"), pid=np.array([r[0] for r in rows], np.int64), arm=arm,
                        tland=np.array([r[2] for r in rows], np.int64), F=F, q=np.array(B.Q), levels=json.dumps(lev))
    a = B.Q.index("ad")
    pairs = [("prs_p10", "prs_p90")] if "prs_p10" in arm else geno2_pairs(lev)
    for lo, hi in pairs:
        print(f"{name}: {lo} -> {hi}, mean AD risk 2 / 5 / 10 y:",
              [f"{F[arm == lo, a, H - 1].mean():.3f} -> {F[arm == hi, a, H - 1].mean():.3f}" for H in (2, 5, 10)], flush=True)


def write(name, R_v1, R_v2, recsets, mc=False):
    """R_v1 / R_v2：(记录, 6, 10) 的风险。写两份人级产物，再调现有的两个打分函数，聚合 JSON 进打包盒。"""
    import perturb_v2 as PV
    import rollout_metrics as RM
    pk = os.path.join(RUN_DIR, "packages", name)
    os.makedirs(pk, exist_ok=True)
    R_v1 = np.clip(R_v1, 1e-6, 1 - 1e-6)
    R_v2 = np.clip(R_v2, 1e-6, 1 - 1e-6)
    # 第二部分：4 个终点的 2 年风险
    _, _, recs = recsets["v1"]
    rec = pd.DataFrame(recs)
    for q, e in SAME_AS_PART1.items():
        rec[f"dirrisk_s0_{e}"] = R_v1[:, B.Q.index(q), 1]
    od = os.path.join(RUN_DIR, "rollout", name)
    os.makedirs(od, exist_ok=True)
    rec.to_csv(os.path.join(od, "perturb_persons.csv"), index=False)
    json.dump({"name": name, "members": [0], "generator_error": None if mc else "direct prediction only",
               "readout": "simulated from the landmark (share of trajectories, smoothed)" if mc else "direct risk"},
              open(os.path.join(od, "run_info.json"), "w"), indent=1)
    pert = RM.perturbation(name)
    json.dump(pert, open(os.path.join(pk, "perturb.json"), "w"), indent=1)
    # 第三部分：6 道题 × 10 年
    _, _, rows, C = recsets["v2"]
    od2 = os.path.join(PV.out_base(None), name)
    os.makedirs(od2, exist_ok=True)
    pid_a = np.array([r[0] for r in rows], np.int64)
    arm_a = np.array([r[1] for r in rows])
    np.savez_compressed(os.path.join(od2, "direct.npz"), pid=pid_a, arm=arm_a,
                        tland=np.array([r[2] for r in rows], np.int64), seeds=np.array([0]), F=R_v2[None])
    # 回归：原样臂 = val 打分表在 (pid, 站点) 上的值（确定性模型应逐位相同；模拟出来的只差随机数）
    sc = pd.read_csv(os.path.join(RUN_DIR, "scores_trans", f"{name}__s0.csv"))
    o = pd.DataFrame({"pid": pid_a, "arm": arm_a, "j": np.arange(len(pid_a)), "t": [r[2] for r in rows]})
    mm = o[o.arm == "orig"].merge(sc, on=["pid", "t"], how="inner")
    dmax = max(float(np.abs(R_v2[mm.j.to_numpy(), qi, H - 1] - np.clip(mm[f"risk_{q}_{H}y"].to_numpy(), 1e-6, 1 - 1e-6)).max())
               for qi, q in enumerate(B.Q) for H in B.HS)
    info = {"name": name, "seeds": [0], "status": "ok", "n_records": int(len(rows)),
            "regression_scores_trans": {"pass": bool(dmax < 1e-4) if not mc else None, "max_abs": dmax, "n": int(len(mm)),
                                        "note": "simulated: differs from the score table only by Monte Carlo noise" if mc else ""}}
    json.dump(info, open(os.path.join(od2, "direct_info.json"), "w"), indent=1)
    meta0 = PV.model_meta
    PV.model_meta = lambda n: ({"family": "unified input", "tokenization": None, "ranked": True}
                               if n == name else meta0(n))
    try:
        J = PV.aggregate_model(name, C, None)
    finally:
        PV.model_meta = meta0
    t = J["direct"].get("prespec_score") or {}
    print(f"{name}: perturbation v1 prespec {pert['direction']['prespec']}; v2 prespec {t}; regression max {dmax:.2e}",
          flush=True)
    return pert, J
