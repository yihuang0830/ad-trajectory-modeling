"""geno2_pert_compare.py -- 基因第二版扩大扰动（geno2_perturb.py）的读法。RIS 上跑。

    python geno2_pert_compare.py     # -> $ZOO_RUN_DIR/packages/_geno2/pert2.json（聚合，可以拉回本地）

读法在看结果之前定下（2026-11-11，用户定的范围），之后不改：
 人群：主 = val + test 里有芯片数据的人（模型没训练过）；敏感性 = train 段有芯片数据的人（模型见过）。
 站点：第 2、4、6 次访视；一个人最多 3 个站点。主表 = 三个站点合并，另按站点分开报（看病程阶段）。
 每道题只用站点时处在风险中的站点（AD、MMSE < 24、认知 < −1；规则同 perturb_v2.cohort_table）。
 单变量扰动（13 组，同 geno2_compare）：每条记录先在 5 个 seed 上平均，读 2 / 5 / 10 年 log OR、风险差、提前年数、ΔRMST。
 交互（只有主人群）：基因变量 × APOE（ε3/ε3 / ε3/ε4）的 2 × 2 析因，三种尺度上的差（同 geno2_compare.interaction）。
 CI：按人整体重抽（一个人的全部站点一起进出）bootstrap 1000 次。另报 5 个 seed 各自的值。
 train 对主：每组扰动的 AD 10 年 log OR 和提前年数，差 = train − 主，CI 用两边各自独立的 bootstrap 相减；
   另报 13 组里同号的组数。预设：train 和主"一致" = 同号，且差的 CI 含 0。不一致的照实报。
 预设方向同 geno2_compare：AD、MMSE < 24、认知 < −1 都升高；ε2/ε3 降低。
 核对：geno2s（存下来的模型）和 geno2 同一个 seed 的 val / test 打分表逐位相同（同一个模型）。
 这些只用模型在扰动记录上的预测，不用 val / test 的标签。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import bl_data  # noqa: F401  （先导入：它把 unified_eval 加进路径）
from common import RUN_DIR
from geno2_compare import lg, rmst, shift_years
from geno_compare import N_BOOT, SEED, SEEDS
from perturb import geno2_ipairs, geno2_pairs

QS = ("ad", "mmse_lt24", "cogn_ltm1")
VISITS = (2, 4, 6)
PERT = os.path.join(RUN_DIR, "geno", "pert2")
OUT = os.path.join(RUN_DIR, "packages", "_geno2")
MODEL = "transformer-geno2s-s{seed}"


def load(cohort):
    """5 个 seed × 各块 -> F (seed, 臂, 站点, 题, 年)、站点表、臂定义。各 seed 的站点按同一个顺序排好。"""
    Fs, tab, lev, arms = [], None, None, None
    for k in SEEDS:
        fs = sorted(glob.glob(os.path.join(PERT, f"{MODEL.format(seed=k)}-{cohort}-p*.npz")))
        if not fs:
            return None
        parts = [np.load(f) for f in fs]
        nparts = json.loads(str(parts[0]["info"]))["nparts"]
        assert len(parts) == nparts, f"{cohort} seed {k}: {len(parts)} / {nparts} 块"
        t = pd.concat([pd.DataFrame({"split": z["split"], "pid": z["pid"], "visit": z["visit"],
                                     **{f"risk__{q}": z[f"risk__{q}"] for q in QS}}) for z in parts], ignore_index=True)
        F = np.concatenate([z["F"] for z in parts], 1)
        o = np.lexsort((t.visit, t.pid, t.split))
        t, F = t.iloc[o].reset_index(drop=True), F[:, o]
        a = list(parts[0]["arm"])
        assert all(list(z["arm"]) == a for z in parts)
        if tab is None:
            tab, lev, arms = t, json.loads(str(parts[0]["levels"])), a
        assert tab[["split", "pid", "visit"]].equals(t[["split", "pid", "visit"]]) and arms == a
        qi = [list(parts[0]["q"]).index(q) for q in QS]
        Fs.append(F[:, :, qi])
    return np.stack(Fs), tab, lev, arms


def person_boot(tab, rng):
    key = tab.split.astype(str) + ":" + tab.pid.astype(str)
    codes, uniq = pd.factorize(key)
    n = len(uniq)
    W = np.stack([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(N_BOOT)]).astype(float)
    return codes, W


def _ci(x):
    return [float(v) for v in np.quantile(x, (0.025, 0.975))]


def effect(L, H, Wr):
    """L / H：seed × 站点 × 年；Wr：bootstrap × 站点（按人重抽得到的站点权重）。"""
    Lm, Hm = L.mean(0), H.mean(0)
    den = Wr.sum(1)
    r = {"n": int(L.shape[1])}
    for h in (2, 5, 10):
        lor, rd = lg(Hm[:, h - 1]) - lg(Lm[:, h - 1]), Hm[:, h - 1] - Lm[:, h - 1]
        seeds = (lg(H[:, :, h - 1]) - lg(L[:, :, h - 1])).mean(1)
        r[f"{h}y"] = {"risk_lo": float(Lm[:, h - 1].mean()), "risk_hi": float(Hm[:, h - 1].mean()),
                      "log_or": float(lor.mean()), "log_or_ci95": _ci(Wr @ lor / den), "log_or_boot": (Wr @ lor / den) if h == 10 else None,
                      "risk_diff": float(rd.mean()), "risk_diff_ci95": _ci(Wr @ rd / den),
                      "per_seed_log_or": [round(float(v), 4) for v in seeds], "n_seeds_positive": int((seeds > 0).sum())}
    BL, BH = (Wr @ Lm) / den[:, None], (Wr @ Hm) / den[:, None]
    yb = np.array([shift_years(BL[b], BH[b]) for b in range(len(Wr))])
    ys = np.array([shift_years(L[k].mean(0), H[k].mean(0)) for k in range(L.shape[0])])
    r["years_earlier"] = {"value": shift_years(Lm.mean(0), Hm.mean(0)), "ci95": _ci(yb), "boot": yb,
                          "per_seed": [round(float(v), 3) for v in ys], "n_seeds_positive": int((ys > 0).sum())}
    drm = np.array([rmst(BH[b]) - rmst(BL[b]) for b in range(len(Wr))])
    r["d_rmst_10y"] = {"value": rmst(Hm.mean(0)) - rmst(Lm.mean(0)), "ci95": _ci(drm)}
    return r


def strip(r):
    """去掉 bootstrap 样本（只在内部做 train − 主 的差时用）。"""
    if isinstance(r, dict):
        return {k: strip(v) for k, v in r.items() if k not in ("boot", "log_or_boot")}
    return r


def singles(P, rng):
    Fs, tab, lev, arms = P
    codes, W = person_boot(tab, rng)
    out = {"n_persons": int(codes.max() + 1), "n_landmarks": int(len(tab)),
           "n_landmarks_by_visit": {int(v): int((tab.visit == v).sum()) for v in VISITS}, "pooled": {}, "by_visit": {}}
    for lo, hi in geno2_pairs(lev):
        a, b = arms.index(lo), arms.index(hi)
        res, byv = {}, {}
        for qi, q in enumerate(QS):
            m = tab[f"risk__{q}"].to_numpy(bool)
            res[q] = effect(Fs[:, a][:, m, qi], Fs[:, b][:, m, qi], W[:, codes[m]])
            for v in VISITS:
                mv = m & (tab.visit == v).to_numpy()
                byv.setdefault(str(v), {})[q] = effect(Fs[:, a][:, mv, qi], Fs[:, b][:, mv, qi], W[:, codes[mv]])
        out["pooled"][f"{lo}->{hi}"] = res
        out["by_visit"][f"{lo}->{hi}"] = byv
    return out


def interactions(P, rng):
    Fs, tab, lev, arms = P
    codes, W = person_boot(tab, rng)
    out = {}
    for var, quad in geno2_ipairs(lev).items():
        ix = [arms.index(x) for x in quad]
        r = {}
        for qi, q in enumerate(QS):
            m = tab[f"risk__{q}"].to_numpy(bool)
            X = [Fs[:, k][:, m, qi] for k in ix]
            Wr = W[:, codes[m]]
            den = Wr.sum(1)
            e3, e4 = effect(X[0], X[1], Wr), effect(X[2], X[3], Wr)
            M = [x.mean(0) for x in X]
            h = 9
            dl = (lg(M[3][:, h]) - lg(M[2][:, h])) - (lg(M[1][:, h]) - lg(M[0][:, h]))
            dr = (M[3][:, h] - M[2][:, h]) - (M[1][:, h] - M[0][:, h])
            BM = [(Wr @ x) / den[:, None] for x in M]
            dy = np.array([shift_years(BM[2][b], BM[3][b]) - shift_years(BM[0][b], BM[1][b]) for b in range(len(Wr))])
            seeds = [float((lg(X[3][k, :, h]) - lg(X[2][k, :, h]) - lg(X[1][k, :, h]) + lg(X[0][k, :, h])).mean())
                     for k in range(X[0].shape[0])]
            r[q] = {"n": int(m.sum()), "noncarrier_e33": strip(e3), "carrier_e34": strip(e4),
                    "interaction_10y": {"log_or_diff": float(dl.mean()), "log_or_diff_ci95": _ci(Wr @ dl / den),
                                        "risk_diff_diff": float(dr.mean()), "risk_diff_diff_ci95": _ci(Wr @ dr / den),
                                        "years_diff": e4["years_earlier"]["value"] - e3["years_earlier"]["value"],
                                        "years_diff_ci95": _ci(dy), "per_seed_log_or_diff": [round(v, 4) for v in seeds],
                                        "n_seeds_positive": int(sum(v > 0 for v in seeds))}}
        out[var] = r
    return out


def train_vs_main(S_main, S_train):
    out, same = {}, 0
    for k in S_main["pooled"]:
        a, b = S_main["pooled"][k]["ad"], S_train["pooled"][k]["ad"]
        d_lor = b["10y"]["log_or_boot"] - a["10y"]["log_or_boot"]
        d_yr = b["years_earlier"]["boot"] - a["years_earlier"]["boot"]
        s = np.sign(a["10y"]["log_or"]) == np.sign(b["10y"]["log_or"])
        same += int(s)
        ci = _ci(d_lor)
        out[k] = {"main_log_or": a["10y"]["log_or"], "train_log_or": b["10y"]["log_or"],
                  "diff_log_or": b["10y"]["log_or"] - a["10y"]["log_or"], "diff_log_or_ci95": ci,
                  "main_years": a["years_earlier"]["value"], "train_years": b["years_earlier"]["value"],
                  "diff_years": b["years_earlier"]["value"] - a["years_earlier"]["value"], "diff_years_ci95": _ci(d_yr),
                  "same_sign": bool(s), "consistent": bool(s and ci[0] <= 0 <= ci[1])}
    x = np.array([[v["main_log_or"], v["train_log_or"]] for v in out.values()])
    return {"pairs": out, "n_same_sign": same, "n_pairs": len(out), "n_consistent": sum(v["consistent"] for v in out.values()),
            "corr_log_or": float(np.corrcoef(x.T)[0, 1])}


def reproducibility():
    out = {}
    for k in SEEDS:
        r = {}
        for d in ("scores", "scores_trans"):
            pa = os.path.join(RUN_DIR, d, f"transformer-geno2-s{k}__s0.csv")
            pb = os.path.join(RUN_DIR, d, f"transformer-geno2s-s{k}__s0.csv")
            if not (os.path.exists(pa) and os.path.exists(pb)):
                r[d] = None
                continue
            A, Bt = pd.read_csv(pa), pd.read_csv(pb)
            cols = [c for c in A.columns if c.startswith("risk_")]
            r[d] = float(np.abs(A[cols].to_numpy() - Bt[cols].to_numpy()).max())
        out[f"s{k}"] = r
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"n_boot": N_BOOT, "seeds": list(SEEDS), "reproducibility_geno2s_vs_geno2": reproducibility(), "cohorts": {}}
    S = {}
    for cohort in ("main", "train"):
        P = load(cohort)
        if P is None:
            print(f"{cohort}: missing", flush=True)
            continue
        infos = [json.loads(str(np.load(f)["info"])) for f in sorted(glob.glob(os.path.join(PERT, f"{MODEL.format(seed=0)}-{cohort}-p*.npz")))]
        S[cohort] = singles(P, np.random.default_rng(SEED))
        res["cohorts"][cohort] = {"info": infos, "single": strip(S[cohort])}
        if cohort == "main":
            res["cohorts"][cohort]["interaction"] = interactions(P, np.random.default_rng(SEED))
        x = S[cohort]["pooled"]["apoe_33->apoe_34"]["ad"]
        print(f"{cohort}: APOE e3/e3 -> e3/e4 AD 10y log OR {x['10y']['log_or']:+.3f}, years earlier {x['years_earlier']['value']:+.2f}",
              flush=True)
    if len(S) == 2:
        res["train_vs_main"] = train_vs_main(S["main"], S["train"])
    json.dump(res, open(os.path.join(OUT, "pert2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
