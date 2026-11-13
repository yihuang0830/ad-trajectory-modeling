"""geno2_compare.py -- 基因第二版（两层基因设计文档 的两层）的读法。RIS 上跑，读人级打分表和扰动风险。

    python geno2_compare.py     # -> $ZOO_RUN_DIR/packages/_geno2/compare.json + 几份 QC（聚合，可以拉回本地）

模型：VISTA（仓库里叫 transformer），run.py --variant geno2 / geno2nx --seed 0..4；对照用 2026-11-05 已有的 gflag / geno 各 5 个 seed。

读法在看 val 之前定下（2026-11-10），之后不改。用户的目的（2026-11-10）是让模型带上基因信息、做基因扰动，不是提分，所以：
 1. 主：预测没有明显变差。AD 的 2 / 5 / 10 年 AUC 平均（第三部分 "ad"），人 = val 里有芯片数据的人，
    geno2 − gflag（两边只差基因列；gflag 只留"有没有基因"的标志），风险 = 5 个 seed 的平均，按人配对 bootstrap 1000 次。
    预设门槛：95% CI 下限 > −0.02 就算"没有明显变差"（不是正式的非劣效检验，事先定的描述性门槛）。
 2. 次要（不做多重比较校正）：geno2 − geno（第一版：83 个先导变异的单个 PRS）、geno2nx − geno2（通路 PRS 不排除 Tier 1 区域，
    敏感性）；第一部分 4 个终点 2 年 AUC 平均、第三部分 6 道题 5 年 AUC 平均、AD 的 O/E；全部 val 的人上的同一组数。
 3. 扰动（geno2 主版本；geno2nx 同样算一份放旁边）：每个基因变量 低 → 高
      通路 PRS：train 段有基因的人的 p10 → p90；Tier 1：风险（或同向替代）等位基因 0 → 2 份；APOE：ε3/ε3 → ε2/ε3、ε3/ε4、ε4/ε4。
    读站点后 2 / 5 / 10 年的累积风险：AD、MMSE < 24、认知 < −1。每人先在 5 个 seed 上平均，log OR 和风险差按人 bootstrap；
    另报 5 个 seed 各自的 log OR（几个同号）。预设方向：AD、MMSE<24、认知<−1 的风险都升高；ε2/ε3 降低。
    数据一侧的量级参照：train 段有基因、有 APOE 的人，随访内 AD ~ 变量 + 年龄 + 性别 + ε4 个数（APOE 臂只调年龄、性别），
    换算到同一个对比（p10 → p90 / 0 → 2 份 / 基因型）。这是模型训练用过的数据，吻合不等于独立验证，只看量级。

2026-11-11 追加（看过第一轮结果之后、应用户要求加的；只用模型在扰动记录上的预测，不用 val 的标签）：
 a. 更正：每道题只用站点时"处在风险中"的人（perturb_v2.cohort_table 的 risk__<题>，和第三部分同一条规则：
    AD 站点时还没诊断；MMSE 从没 < 24；认知从没 < −1）。第一轮把站点时已经越线的人也算进去了，
    第一版（11-05，geno_compare.prs_effect）也一样，这里一并按同一规则重算（v1_prs_at_risk）。
 b. "提前几年"：处在风险中的人的平均累积风险曲线（5 个 seed、所有人平均；第 0 年 = 0，逐年线性插值），
    高的那条在哪一年达到低的那条第 10 年的风险，10 减去它 = 提前的年数；高的那条反而更低时，反过来算、记成负数（推迟）。
    另报 ΔRMST（10 年内平均无事件时间，高 − 低，年；和第三部分 perturb_v2 同一个梯形积分）。CI = 按人 bootstrap 重算曲线。
 c. 基因变量 × APOE 交互（geno2i：同一份数据、同一个 seed 重训，扰动另加 2 × 2 析因臂）：每个基因变量（4 个 Tier 1 + 6 个通路 PRS）
    低 → 高的效应，分别在"所有人 APOE 设成 ε3/ε3"和"设成 ε3/ε4"下算；交互 = 携带者里的效应 − 非携带者里的效应，
    在 log OR（相乘尺度）、风险差（相加尺度）、提前年数三种尺度上报，同一批人配对 bootstrap。不预设方向；
    10 个变量 × 3 道题，只作描述（不做检验校正），CI 不含 0 且 ≥ 4 个 seed 同号的才在报告里点名，并标明是探索性的。
    数据参照：train 段，随访内 AD ~ 变量 + ε4 携带 + 变量 × ε4 携带 + 年龄 + 性别。
 d. 核对：geno2i 和 geno2 同一个 seed 的 val 打分表、单变量扰动风险应该逐位相同（训练可复现）；不同就照实报差多少。
"""
import json
import os
import shutil

import numpy as np
import pandas as pd

import bl_data as B
from common import RUN_DIR
from geno_build import logit
from geno_compare import KEYS, N_BOOT, SEED, SEEDS, cells, metrics, prepare, tables
from perturb import geno2_ipairs, geno2_pairs

MODELS = ("transformer",)
V1_MODELS = ("transformer", "minimalrnn", "xgb-l2c-sim")
VARIANTS = ("gflag", "geno", "geno2", "geno2nx")
CONTRASTS = (("geno2", "gflag"), ("geno2", "geno"), ("geno2nx", "geno2"))
QS = ("ad", "mmse_lt24", "cogn_ltm1")
OUT = os.path.join(RUN_DIR, "packages", "_geno2")
MARGIN = -0.02


def compare(P, n, rng, variants):
    one = np.ones(n)
    full = {v: metrics(P, v, one) for v in variants}
    boot = {v: {k: [] for k in KEYS} for v in variants}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
        for v in variants:
            m = metrics(P, v, w)
            for k in KEYS:
                boot[v][k].append(m[k])
    per_seed = {v: [metrics(P, v, one, s) for s in SEEDS] for v in variants}
    out = {"value": {v: {k: full[v][k] for k in KEYS} for v in variants}, "contrasts": {}}
    for a, b in CONTRASTS:
        r = {}
        for k in KEYS:
            d = np.array(boot[a][k]) - np.array(boot[b][k])
            ds = [per_seed[a][s][k] - per_seed[b][s][k] for s in SEEDS]
            r[k] = {"diff": full[a][k] - full[b][k], "ci95": [float(np.nanquantile(d, q)) for q in (0.025, 0.975)],
                    "p_le_0": float(np.nanmean(d <= 0)), "per_seed": ds}
        out["contrasts"][f"{a}-{b}"] = r
    x = out["contrasts"]["geno2-gflag"]["ad"]
    out["primary_no_clear_loss"] = {"margin": MARGIN, "ci_low": x["ci95"][0], "pass": bool(x["ci95"][0] > MARGIN)}
    return out


def lg(x):
    return np.log(x / (1 - x))


def shift_years(Flo, Fhi):
    """两条平均累计风险曲线（第 1..10 年）-> 高的那条提前多少年达到低的那条第 10 年的风险（年）；
    高的那条反而更低时，低的那条提前多少年达到高的那条第 10 年的风险，记成负数（= 推迟）。第 0 年 = 0，逐年线性插值。"""
    t = np.arange(11.0)
    lo = np.maximum.accumulate(np.r_[0.0, Flo])           # 平均累计曲线本来就单调；防插值时的数值抖动
    hi = np.maximum.accumulate(np.r_[0.0, Fhi])
    if hi[-1] >= lo[-1]:
        return 10.0 - float(np.interp(lo[-1], hi, t))
    return -(10.0 - float(np.interp(hi[-1], lo, t)))


def rmst(F):
    S = 1.0 - np.r_[0.0, F]
    return float(0.5 * (S[:-1] + S[1:]).sum())


def risk_flags():
    import perturb_v2 as PV
    C, _ = PV.cohort_table()
    return {q: C[f"risk__{q}"].astype(bool) for q in QS}


def load_pert(name, variant):
    """5 个 seed -> (F: seed × 记录 × 题(QS) × 年, pid, arm, 臂定义)；缺 seed 返回 None。"""
    Fs, ref, lev = [], None, None
    for k in SEEDS:
        p = os.path.join(RUN_DIR, "geno", "pert", f"{name}-{variant}-s{k}.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        key = (z["pid"], z["arm"])
        assert ref is None or (np.array_equal(ref[0], key[0]) and np.array_equal(ref[1], key[1]))
        ref, lev = key, json.loads(str(z["levels"]))
        Fs.append(z["F"][:, [list(z["q"]).index(q) for q in QS]])
    return np.stack(Fs), ref[0], ref[1], lev


def _boot_w(n, rng):
    return np.stack([np.bincount(rng.integers(n, size=n), minlength=n) / n for _ in range(N_BOOT)])


def _ci(x):
    return [float(v) for v in np.quantile(x, (0.025, 0.975))]


def effect(L, H, W):
    """L / H：seed × 人 × 年（同一批人、两个臂）-> 10 / 5 / 2 年的 log OR、风险差，提前年数、ΔRMST，各带 CI 和逐 seed 值。"""
    Lm, Hm = L.mean(0), H.mean(0)                          # 每人先在 seed 上平均
    r = {"n": int(L.shape[1])}
    for h in (2, 5, 10):
        lor, rd = lg(Hm[:, h - 1]) - lg(Lm[:, h - 1]), Hm[:, h - 1] - Lm[:, h - 1]
        seeds = (lg(H[:, :, h - 1]) - lg(L[:, :, h - 1])).mean(1)
        r[f"{h}y"] = {"risk_lo": float(Lm[:, h - 1].mean()), "risk_hi": float(Hm[:, h - 1].mean()),
                      "log_or": float(lor.mean()), "log_or_ci95": _ci(W @ lor),
                      "risk_diff": float(rd.mean()), "risk_diff_ci95": _ci(W @ rd),
                      "per_seed_log_or": [round(float(v), 4) for v in seeds], "n_seeds_positive": int((seeds > 0).sum())}
    yrs_b = np.array([shift_years(w @ Lm, w @ Hm) for w in W])
    yrs_s = np.array([shift_years(L[k].mean(0), H[k].mean(0)) for k in range(L.shape[0])])
    drm = np.array([rmst(w @ Hm) - rmst(w @ Lm) for w in W])
    r["years_earlier"] = {"value": shift_years(Lm.mean(0), Hm.mean(0)), "ci95": _ci(yrs_b),
                          "per_seed": [round(float(v), 3) for v in yrs_s], "n_seeds_positive": int((yrs_s > 0).sum())}
    r["d_rmst_10y"] = {"value": rmst(Hm.mean(0)) - rmst(Lm.mean(0)), "ci95": _ci(drm)}
    return r


def perturbation(P, risk, rng):
    """单变量扰动（低 → 高），每道题只用处在风险中的人。"""
    if P is None:
        return None
    Fs, pid, arm, lev = P
    out = {"n_persons": int(len(np.unique(pid))), "arms": {k: v for k, v in lev.items() if "__a3" not in k}, "pairs": {}}
    for lo, hi in geno2_pairs(lev):
        a, b = arm == lo, arm == hi
        assert np.array_equal(pid[a], pid[b])
        r = {}
        for qi, q in enumerate(QS):
            m = risk[q].reindex(pid[a]).fillna(False).to_numpy(bool)
            r[q] = effect(Fs[:, a][:, m, qi], Fs[:, b][:, m, qi], _boot_w(int(m.sum()), rng))
        out["pairs"][f"{lo}->{hi}"] = r
    return out


def interaction(P, risk, rng):
    """基因变量 × APOE（geno2i 的 2 × 2 析因臂）：非携带者（ε3/ε3）和携带者（ε3/ε4）里各自的效应，及其差（配对 bootstrap）。"""
    if P is None:
        return None
    Fs, pid, arm, lev = P
    out = {}
    for var, (lo3, hi3, lo4, hi4) in geno2_ipairs(lev).items():
        masks = [arm == x for x in (lo3, hi3, lo4, hi4)]
        for mk in masks[1:]:
            assert np.array_equal(pid[masks[0]], pid[mk])
        r = {}
        for qi, q in enumerate(QS):
            m = risk[q].reindex(pid[masks[0]]).fillna(False).to_numpy(bool)
            X = [Fs[:, mk][:, m, qi] for mk in masks]          # 4 × (seed, 人, 年)
            n = int(m.sum())
            W = _boot_w(n, rng)
            e3, e4 = effect(X[0], X[1], W), effect(X[2], X[3], W)
            M = [x.mean(0) for x in X]                         # 每人 seed 平均
            h = 9                                              # 第 10 年
            dl = (lg(M[3][:, h]) - lg(M[2][:, h])) - (lg(M[1][:, h]) - lg(M[0][:, h]))
            dr = (M[3][:, h] - M[2][:, h]) - (M[1][:, h] - M[0][:, h])
            dy = np.array([shift_years(w @ M[2], w @ M[3]) - shift_years(w @ M[0], w @ M[1]) for w in W])
            seeds = [float((lg(X[3][k, :, h]) - lg(X[2][k, :, h]) - lg(X[1][k, :, h]) + lg(X[0][k, :, h])).mean())
                     for k in range(X[0].shape[0])]
            r[q] = {"n": n, "noncarrier_e33": e3, "carrier_e34": e4,
                    "interaction_10y": {"log_or_diff": float(dl.mean()), "log_or_diff_ci95": _ci(W @ dl),
                                        "risk_diff_diff": float(dr.mean()), "risk_diff_diff_ci95": _ci(W @ dr),
                                        "years_diff": e4["years_earlier"]["value"] - e3["years_earlier"]["value"],
                                        "years_diff_ci95": _ci(dy),
                                        "per_seed_log_or_diff": [round(v, 4) for v in seeds],
                                        "n_seeds_positive": int(sum(v > 0 for v in seeds))}}
        out[var] = r
    return out


def v1_prs_at_risk(risk, rng):
    """第一版（11-05）PRS p10 → p90 按同一条"处在风险中"规则重算（只看 AD）。"""
    out = {}
    for name in V1_MODELS:
        Fs = []
        for k in SEEDS:
            p = os.path.join(RUN_DIR, "geno", "pert", f"{name}-geno-s{k}.npz")
            if not os.path.exists(p):
                break
            z = np.load(p)
            Fs.append(z["F"][:, list(z["q"]).index("ad")])
            pid, arm = z["pid"], z["arm"]
        if len(Fs) < len(SEEDS):
            continue
        F = np.stack(Fs)
        a, b = arm == "prs_p10", arm == "prs_p90"
        m = risk["ad"].reindex(pid[a]).fillna(False).to_numpy(bool)
        out[name] = effect(F[:, a][:, m], F[:, b][:, m], _boot_w(int(m.sum()), rng))
        out[name]["n_all_persons"] = int(a.sum())
    return out


def reproducibility(name):
    """geno2i 和 geno2 同一个 seed：val 打分表、单变量扰动风险逐位比较（应当相同）。"""
    out = {}
    for k in SEEDS:
        r = {}
        for d in ("scores", "scores_trans"):
            pa = os.path.join(RUN_DIR, d, f"{name}-geno2-s{k}__s0.csv")
            pb = os.path.join(RUN_DIR, d, f"{name}-geno2i-s{k}__s0.csv")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                return None
            A, Bt = pd.read_csv(pa), pd.read_csv(pb)
            cols = [c for c in A.columns if c.startswith("risk_")]
            r[d] = float(np.abs(A[cols].to_numpy() - Bt[cols].to_numpy()).max())
        za = np.load(os.path.join(RUN_DIR, "geno", "pert", f"{name}-geno2-s{k}.npz"))
        zb = np.load(os.path.join(RUN_DIR, "geno", "pert", f"{name}-geno2i-s{k}.npz"))
        keep = np.array(["__a3" not in x for x in zb["arm"].astype(str)])
        assert np.array_equal(za["pid"], zb["pid"][keep]) and np.array_equal(za["arm"], zb["arm"][keep])
        r["single_arm_F"] = float(np.abs(za["F"] - zb["F"][keep]).max())
        out[f"s{k}"] = r
    return out


def data_reference(variant):
    """train 段有基因、有 APOE 的人：随访内 AD 的逻辑回归，换算成和扰动同一个对比的 log OR。只出聚合数。"""
    Z, meta = B.load(f"newmodel_data_trans_{variant}.npz")
    st, fi = meta["static"], meta["features"].index
    X, M, S = Z["train_X"], Z["train_M"], Z["train_S"]
    y = np.array([X[i, M[i], fi("ad_dx_state")].max() for i in range(len(S))]) > 0.5
    keep = (S[:, st.index("geno_na")] < 0.5) & (S[:, st.index("apoe_na")] < 0.5)
    S, y, age0 = S[keep], y[keep].astype(float), X[keep, 0, fi("age_z")]
    e4 = S[:, st.index("apoe_e4x1")] + 2 * S[:, st.index("apoe_e4x2")]
    cov = np.column_stack([age0, S[:, st.index("male")], e4])
    out = {"n": int(len(y)), "n_ad": int(y.sum()), "contrasts": {}}
    for c in st:
        if c.startswith(("prs_", "t1_")):
            v = S[:, st.index(c)]
            b, se = logit(np.column_stack([v, cov]), y)
            span = float(np.quantile(v, 0.9) - np.quantile(v, 0.1)) if c.startswith("prs_") else 2.0
            key = f"{c}__p10->{c}__p90" if c.startswith("prs_") else f"{c}__0->{c}__2"
            out["contrasts"][key] = {"log_or": float(b[0] * span), "ci95": [float((b[0] - 1.96 * se[0]) * span),
                                                                           float((b[0] + 1.96 * se[0]) * span)]}
    carrier = (e4 > 0).astype(float)
    out["interaction_with_e4_carrier"] = {}
    for c in st:
        if c.startswith(("prs_", "t1_")):
            v = S[:, st.index(c)]
            span = float(np.quantile(v, 0.9) - np.quantile(v, 0.1)) if c.startswith("prs_") else 2.0
            b, se = logit(np.column_stack([v, carrier, v * carrier, cov[:, :2]]), y)
            out["interaction_with_e4_carrier"][c] = {
                "noncarrier_log_or": float(b[0] * span), "carrier_log_or": float((b[0] + b[2]) * span),
                "diff_log_or": float(b[2] * span), "diff_ci95": [float((b[2] - 1.96 * se[2]) * span), float((b[2] + 1.96 * se[2]) * span)],
                "n_carrier": int(carrier.sum())}
    g = {"apoe_23": (S[:, st.index("apoe_e2carrier")] > 0.5) & (e4 == 0),
         "apoe_34": (S[:, st.index("apoe_e4x1")] > 0.5) & (S[:, st.index("apoe_e2carrier")] < 0.5),
         "apoe_44": S[:, st.index("apoe_e4x2")] > 0.5}
    ref = (e4 == 0) & (S[:, st.index("apoe_e2carrier")] < 0.5)
    for k, m in g.items():
        kk = m | ref
        b, se = logit(np.column_stack([m[kk].astype(float), cov[kk, :2]]), y[kk])
        out["contrasts"][f"apoe_33->{k}"] = {"log_or": float(b[0]), "ci95": [float(b[0] - 1.96 * se[0]), float(b[0] + 1.96 * se[0])],
                                            "n_group": int(m.sum())}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    truth1 = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv")).set_index(["pid", "t"])
    truth3 = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv")).set_index(["pid", "t"])
    C = cells(truth1, truth3)
    Z, meta = B.load("newmodel_data_trans_geno2.npz")
    gen = Z["val_pid"][Z["val_S"][:, meta["static"].index("geno_na")] < 0.5]
    all_p = np.unique(truth1.index.get_level_values(0))
    pops = {"genotyped": np.intersect1d(all_p, gen), "all_val": all_p}
    risk = risk_flags()
    res = {"n_persons": {k: int(len(v)) for k, v in pops.items()}, "n_boot": N_BOOT, "seeds": list(SEEDS),
           "v1_prs_at_risk": v1_prs_at_risk(risk, np.random.default_rng(SEED)),
           "static_added": meta["static"][25:], "models": {},
           "data_reference": {v: data_reference(v) for v in ("geno2", "geno2nx")}}
    for name in MODELS:
        tabs = {v: tables(name, v) for v in VARIANTS}
        miss = [v for v, t in tabs.items() if t is None]
        if miss:
            print(f"{name}: missing score tables for {miss}", flush=True)
            continue
        r = {pop: compare(prepare(C, tabs, pids), len(pids), np.random.default_rng(SEED), VARIANTS) for pop, pids in pops.items()}
        r["perturbation"] = {v: perturbation(load_pert(name, v), risk, np.random.default_rng(SEED)) for v in ("geno2", "geno2nx")}
        r["interaction"] = interaction(load_pert(name, "geno2i"), risk, np.random.default_rng(SEED))
        r["reproducibility_geno2i_vs_geno2"] = reproducibility(name)
        res["models"][name] = r
        x = r["genotyped"]["contrasts"]["geno2-gflag"]["ad"]
        print(f"{name}: AD AUC (2/5/10y mean) geno2 - gflag = {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]",
              flush=True)
    json.dump(res, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
    for f in ("tier1_qc.json", "geno2_qc.json", os.path.join("prscs", "prep.json")):
        shutil.copy(os.path.join(RUN_DIR, "geno", f), os.path.join(OUT, os.path.basename(f)))


if __name__ == "__main__":
    main()
