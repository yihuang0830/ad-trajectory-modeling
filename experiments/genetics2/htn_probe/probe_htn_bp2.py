"""probe_htn_bp2.py -- 高血压扰动补齐相关变量（probe_htn_bp.py 的第二版，2026-11-12）：按基因表（fig2）同一套读法出数。

    py.sbatch ../_probes_20261112/probe_htn_bp2.py --cohort main     # 人级 -> _runs/geno/htn_probe/htn_bp2_main.npz
    py.sbatch ../_probes_20261112/probe_htn_bp2.py --cohort train
    py.sbatch ../_probes_20261112/probe_htn_bp2.py --compare         # 聚合 -> _probes_20261112/probe_htn_bp2.json

和 probe_htn_bp.py 相同：5 个 VISTA（transformer-geno2s-s<seed>），站点 = 第 2 / 4 / 6 次访视、站点时 hypertension_cum = 0 且不缺失，
只改站点这次访视，每个 seed 所有臂同一个种子（CRN；main 的种子和第一版相同，结果应逐位复现）。
血压 "onset" = train 里真实新发高血压那次访视的均值（第一版 A_data.train.onset_visit：134.7 / 73.6）。
读法 = geno2_pert_compare.effect（原样 import）：每条记录先 5 个 seed 平均；AD 只用站点时还没 AD 的站点；
  提前年数 = 扰动那条平均累计曲线在第几年达到原样那条第 10 年的风险（负数 = 推迟）；CI = 按人 bootstrap（N_BOOT 次，种子 SEED）。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bl_data as B  # noqa: E402
import simulator as SIM  # noqa: E402
from common import RUN_DIR  # noqa: E402
from geno2_perturb import COHORTS, MODEL, landmarks, stack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTP = os.path.join(RUN_DIR, "geno", "htn_probe")
N_MC = 100
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("orig", "htn", "rx", "bp", "bp160", "htn_rx", "full", "full160")
VISITS = (2, 4, 6)


def simulate(cohort):
    import torch
    import perturb_v2 as PV
    import rollout_versions as RV
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    Z, meta = B.load("newmodel_data_trans_geno2.npz")
    feats = meta["features"]
    mu, sd = meta["norm"]["mu"], meta["norm"]["sd"]
    FI = {f: i for i, f in enumerate(feats)}
    on = json.load(open(os.path.join(HERE, "probe_htn_bp.json")))["A_data"]["train"]["onset_visit"]
    SBP_ON, DBP_ON = on["sbp_mean"], on["dbp_mean"]
    norm = SIM.norm_info(meta)
    rows = landmarks(Z, meta, COHORTS[cohort])
    D, vv = stack(Z, rows)
    xL = D["X"][np.arange(len(vv)), vv]
    obs = lambda x, f: x[:, FI[f + "_missing"]] < 0.5                       # noqa: E731
    ok = (xL[:, FI["hypertension_cum"]] < 0.5) & obs(xL, "hypertension_cum")
    ii = np.where(ok)[0]
    n = len(ii)
    xe = xL[ii]
    sbp = xe[:, FI["sbp_avg"]] * sd["sbp_avg"] + mu["sbp_avg"]
    clean = obs(xe, "antihyp_rx") & (xe[:, FI["antihyp_rx"]] < 0.5) & obs(xe, "sbp_avg") & (sbp < 140)
    v = vv[ii]
    k = np.arange(n)
    z = lambda f, val: (val - mu[f]) / sd[f]                                  # noqa: E731

    def edit(X, arm):
        X = X.copy()
        if "htn" in arm or arm.startswith("full"):
            X = np.stack([PV.translate_features("htn", X[j], D["M"][ii[j]], D["S"][ii[j]], D["L"][ii[j]], v[j], meta["norm"], feats)[0]
                          for j in range(n)])
        if "rx" in arm or arm.startswith("full"):
            X[k, v, FI["antihyp_rx"]] = 1.0
            X[k, v, FI["antihyp_rx_missing"]] = 0.0
        bp = (160.0, 90.0) if arm.endswith("160") else (SBP_ON, DBP_ON) if arm in ("bp", "full") else None
        if bp is not None:
            X[k, v, FI["sbp_avg"]] = z("sbp_avg", bp[0]); X[k, v, FI["sbp_avg_missing"]] = 0.0
            X[k, v, FI["dbp_avg"]] = z("dbp_avg", bp[1]); X[k, v, FI["dbp_avg_missing"]] = 0.0
        return X

    base = {kk: D[kk][ii].copy() for kk in ("X", "M", "S", "L", "T", "pid")}
    Xarm = {a: edit(base["X"], a) for a in ARMS}
    F = np.zeros((len(SEEDS), len(ARMS), n, B.NH), np.float32)
    iq = B.Q.index("ad")
    t0 = time.time()
    for si, seed in enumerate(SEEDS):
        gen = RV.load_state(os.path.join(RUN_DIR, "models", MODEL.format(seed=seed) + ".pt"))
        for ai, arm in enumerate(ARMS):
            Dp = {kk: val.copy() for kk, val in base.items()}
            Dp["X"] = Xarm[arm].copy()
            start = {"D": Dp, "ii": np.arange(n), "vv": v, "pid": Dp["pid"], "t": Dp["T"][k, v]}
            rng = np.random.default_rng(20261112 + 1000 * seed)
            F[si, ai] = SIM.risks(gen.sim(start, N_MC, rng, norm, full_draws=True), True)[:, iq]
        print(f"{cohort} seed {seed} done, {n} landmarks, {(time.time() - t0) / 60:.1f} min", flush=True)
    os.makedirs(OUTP, mode=0o700, exist_ok=True)
    f = os.path.join(OUTP, f"htn_bp2_{cohort}.npz")
    np.savez_compressed(f + ".tmp.npz", F=F, arms=np.array(ARMS), split=np.array([rows[j]["split"] for j in ii]),
                        pid=np.array([rows[j]["pid"] for j in ii], np.int64), visit=np.array([rows[j]["visit"] for j in ii], np.int64),
                        risk_ad=np.array([rows[j]["risk__ad"] for j in ii]), clean=clean, bp_onset=np.array([SBP_ON, DBP_ON]))
    os.replace(f + ".tmp.npz", f)


def compare():
    from geno2_pert_compare import effect, person_boot, strip
    from geno_compare import N_BOOT, SEED
    # 表的行：(行名, 臂, 只用"干净"的站点)
    ROWS = [("htn", "htn", False), ("htn_rx", "htn_rx", False), ("full", "full", False), ("full160", "full160", False),
            ("bp160", "bp160", False), ("rx", "rx", False), ("bp", "bp", False), ("htn_clean", "htn", True)]
    out = {"n_boot": N_BOOT, "seed": SEED, "rows": {}, "cohorts": {}}
    res = {}
    for cohort in ("main", "train"):
        z = np.load(os.path.join(OUTP, f"htn_bp2_{cohort}.npz"))
        F, arms = z["F"], list(z["arms"])
        tab = pd.DataFrame({"split": z["split"], "pid": z["pid"], "visit": z["visit"]})
        rng = np.random.default_rng(SEED)
        codes, W = person_boot(tab, rng)
        risk, clean = z["risk_ad"].astype(bool), z["clean"].astype(bool)
        out["cohorts"][cohort] = {"n_landmarks_eligible": int(len(tab)), "n_persons": int(codes.max() + 1),
                                  "n_at_risk_ad": int(risk.sum()), "n_clean_at_risk_ad": int((risk & clean).sum()),
                                  "n_at_risk_by_visit": {int(vi): int((risk & (tab.visit == vi).to_numpy()).sum()) for vi in VISITS},
                                  "bp_onset": [float(x) for x in z["bp_onset"]]}
        for name, arm, cl in ROWS:
            m = risk & clean if cl else risk
            a0, a1 = arms.index("orig"), arms.index(arm)
            r = {"pooled": strip(effect(F[:, a0][:, m], F[:, a1][:, m], W[:, codes[m]]))}
            if cohort == "main":
                r["by_visit"] = {}
                for vi in VISITS:
                    mv = m & (tab.visit == vi).to_numpy()
                    r["by_visit"][str(vi)] = strip(effect(F[:, a0][:, mv], F[:, a1][:, mv], W[:, codes[mv]]))
            res.setdefault(name, {})[cohort] = r
    out["rows"] = res
    json.dump(out, open(os.path.join(HERE, "probe_htn_bp2.json"), "w"), indent=1)
    for name in res:
        p = res[name]["main"]["pooled"]
        print(name, f"{p['10y']['risk_lo']:.3f} -> {p['10y']['risk_hi']:.3f}", "years", round(p["years_earlier"]["value"], 2),
              p["years_earlier"]["ci95"], p["years_earlier"]["per_seed"], "train", round(res[name]["train"]["pooled"]["years_earlier"]["value"], 2),
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("main", "train"))
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    compare() if a.compare else simulate(a.cohort)
