"""geno_compare.py -- 基因有没有用：同一结构的 -geno / -gflag / -base 各 5 个 seed（run.py --variant），按人配对 bootstrap。
RIS 上跑，读人级打分表。

    python geno_compare.py      # -> $ZOO_RUN_DIR/packages/_geno/{compare.json, geno_qc.json}（聚合，可以拉回本地）

读法在看 val 之前定下（2026-11-05），之后不改：
  主指标：AD 的 2 / 5 / 10 年 AUC 的平均（第三部分 "ad" 这道题），人 = val 里有芯片数据的人（他们的全部站点）。
  主对比：geno − gflag。两边同一个数据文件，只差 PRS 和 5 个主成分；gflag 只留"有没有基因"的标志，
          这个标志本身就带信息（有基因的人多是早入组的老队列，平均访视次数比其余人多）。
  次要：geno − base、gflag − base；第一部分 4 个终点 2 年 AUC 平均、第三部分 6 道题 5 年 AUC 平均；AD 的 O/E（2 / 5 / 10 年）；
        全部 val 人上的同一组数（看没基因的人有没有被带坏）。
  风险 = 5 个 seed 的平均（和 -x5 一个口径）；另报 5 对 seed 各自的差（种子方差）。
  配对 bootstrap：重抽人 1000 次，95% CI；3 个结构，主对比另给 Bonferroni 的 98.33% CI。
  PRS 扰动（只有 geno）：PRS 从 train 有基因的人的第 10 百分位调到第 90 百分位（perturb.prs_records），AD 风险的 log OR 和风险差
        （2 / 5 / 10 年）；每人先在 5 个 seed 上平均，再按人 bootstrap。预期方向：升高。
"""
import json
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bl_data as B
from common import ENDPOINTS, RUN_DIR

MODELS = ("transformer", "minimalrnn", "xgb-l2c-sim")
VARIANTS = ("base", "gflag", "geno")
CONTRASTS = (("geno", "gflag"), ("geno", "base"), ("gflag", "base"))
SEEDS = range(5)
N_BOOT = 1000
SEED = 20261105
OUT = os.path.join(RUN_DIR, "packages", "_geno")
KEYS = ["ad", "p1", "p3"] + [f"ad_{H}y{s}" for H in (2, 5, 10) for s in ("", "_oe")]


def tables(name, variant):
    """每个 seed 一对打分表（第一部分、第三部分），缺一个就返回 None。"""
    out = []
    for k in SEEDS:
        ps = [os.path.join(RUN_DIR, d, f"{name}-{variant}-s{k}__s0.csv") for d in ("scores", "scores_trans")]
        if not all(os.path.exists(p) for p in ps):
            return None
        out.append([pd.read_csv(p).set_index(["pid", "t"]) for p in ps])
    return out


def cells(truth1, truth3):
    C = [("ad", f"ad_{H}y", 1, f"y_ad_{H}y", f"risk_ad_{H}y") for H in (2, 5, 10)]
    C += [("p1", f"{e}_2y", 0, f"y_{e}_2y", f"risk_{e}_2y") for e in ENDPOINTS]
    C += [("p3", f"{q}_5y", 1, f"y_{q}_5y", f"risk_{q}_5y") for q in B.Q]
    out = []
    for g, nm, part, yc, rc in C:
        t = (truth1, truth3)[part]
        y = t[yc]
        k = y >= 0
        out.append({"group": g, "name": nm, "part": part, "idx": t.index[k], "y": y[k].to_numpy(), "col": rc})
    return out


def prepare(C, tabs, pids):
    """每格：参与的人的下标、标签、每个变体 × seed 的风险 (seed, 站点)。"""
    pos = {p: i for i, p in enumerate(pids)}
    P = []
    for c in C:
        pp = c["idx"].get_level_values(0)
        k = np.array([p in pos for p in pp])
        idx = c["idx"][k]
        risk = {v: np.stack([T[c["part"]].loc[idx, c["col"]].to_numpy() for T in tabs[v]]) for v in tabs}
        P.append({**c, "pi": np.array([pos[p] for p in pp[k]]), "y": c["y"][k], "risk": risk})
    return P


def _auc(y, s, w):
    k = w > 0
    if len(np.unique(y[k])) < 2:
        return np.nan
    return roc_auc_score(y[k], s[k], sample_weight=w[k])


def metrics(P, v, w, seed=None):
    """一个变体在人权重 w 下的各格 AUC、AD 的 O/E、分组平均。seed=None = 5 个 seed 的平均风险。"""
    res = {}
    for c in P:
        r = c["risk"][v].mean(0) if seed is None else c["risk"][v][seed]
        ww = w[c["pi"]]
        res[c["name"]] = _auc(c["y"], r, ww)
        if c["group"] == "ad":
            res[c["name"] + "_oe"] = float((ww * c["y"]).sum() / (ww * r).sum())
    for g in ("ad", "p1", "p3"):
        res[g] = float(np.nanmean([res[c["name"]] for c in P if c["group"] == g]))
    return res


def compare(P, n, rng):
    one = np.ones(n)
    full = {v: metrics(P, v, one) for v in VARIANTS}
    boot = {v: {k: [] for k in KEYS} for v in VARIANTS}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
        for v in VARIANTS:
            m = metrics(P, v, w)
            for k in KEYS:
                boot[v][k].append(m[k])
    per_seed = {v: [metrics(P, v, one, s) for s in SEEDS] for v in VARIANTS}
    out = {"value": {v: {k: full[v][k] for k in KEYS} for v in VARIANTS}, "contrasts": {}}
    for a, b in CONTRASTS:
        r = {}
        for k in KEYS:
            d = np.array(boot[a][k]) - np.array(boot[b][k])
            ds = [per_seed[a][s][k] - per_seed[b][s][k] for s in SEEDS]
            r[k] = {"diff": full[a][k] - full[b][k], "ci95": [float(np.nanquantile(d, q)) for q in (0.025, 0.975)],
                    "ci_bonf3": [float(np.nanquantile(d, q)) for q in (0.05 / 6, 1 - 0.05 / 6)],
                    "p_le_0": float(np.nanmean(d <= 0)), "per_seed": ds, "per_seed_sd": float(np.std(ds, ddof=1))}
        out["contrasts"][f"{a}-{b}"] = r
    return out


def prs_effect(name, rng):
    """PRS p10 -> p90：每人每个时间窗的 AD 风险先在 5 个 seed 上平均，再算 log OR / 风险差，按人 bootstrap。"""
    Fs, ref = [], None
    for k in SEEDS:
        p = os.path.join(RUN_DIR, "geno", "pert", f"{name}-geno-s{k}.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        key = (z["pid"], z["arm"])
        assert ref is None or (np.array_equal(ref[0], key[0]) and np.array_equal(ref[1], key[1]))
        ref = key
        Fs.append(z["F"][:, list(z["q"]).index("ad")])
    F = np.mean(Fs, 0)
    pid, arm = ref
    lo, hi = F[arm == "prs_p10"], F[arm == "prs_p90"]
    assert np.array_equal(pid[arm == "prs_p10"], pid[arm == "prs_p90"])
    lg = lambda x: np.log(x / (1 - x))    # noqa: E731
    out = {"n_persons": int(len(lo)), "levels": json.loads(str(z["levels"]))}
    n = len(lo)
    W = [np.bincount(rng.integers(n, size=n), minlength=n) / n for _ in range(N_BOOT)]
    for H in (2, 5, 10):
        lor, rd = lg(hi[:, H - 1]) - lg(lo[:, H - 1]), hi[:, H - 1] - lo[:, H - 1]
        out[f"{H}y"] = {"risk_p10": float(lo[:, H - 1].mean()), "risk_p90": float(hi[:, H - 1].mean()),
                        "log_or": float(lor.mean()), "log_or_ci95": [float(np.quantile([w @ lor for w in W], q)) for q in (0.025, 0.975)],
                        "risk_diff": float(rd.mean()), "risk_diff_ci95": [float(np.quantile([w @ rd for w in W], q)) for q in (0.025, 0.975)]}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    truth1 = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv")).set_index(["pid", "t"])
    truth3 = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv")).set_index(["pid", "t"])
    C = cells(truth1, truth3)
    Z, meta = B.load("newmodel_data_trans_geno.npz")
    gen = Z["val_pid"][Z["val_S"][:, meta["static"].index("geno_na")] < 0.5]
    all_p = np.unique(truth1.index.get_level_values(0))
    pops = {"genotyped": np.intersect1d(all_p, gen), "all_val": all_p}
    res = {"n_persons": {k: int(len(v)) for k, v in pops.items()}, "n_boot": N_BOOT, "seeds": list(SEEDS), "models": {}}
    for name in MODELS:
        tabs = {v: tables(name, v) for v in VARIANTS}
        if any(t is None for t in tabs.values()):
            print(f"{name}: missing score tables for {[v for v, t in tabs.items() if t is None]}", flush=True)
            continue
        r = {pop: compare(prepare(C, tabs, pids), len(pids), np.random.default_rng(SEED)) for pop, pids in pops.items()}
        r["prs_perturbation"] = prs_effect(name, np.random.default_rng(SEED))
        res["models"][name] = r
        x = r["genotyped"]["contrasts"]["geno-gflag"]["ad"]
        print(f"{name}: AD AUC (2/5/10y mean) geno - gflag = {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]",
              flush=True)
    json.dump(res, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
    shutil.copy(os.path.join(RUN_DIR, "geno", "geno_qc.json"), os.path.join(OUT, "geno_qc.json"))


if __name__ == "__main__":
    main()
