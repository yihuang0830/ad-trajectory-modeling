"""probe_htn_bp.py -- 2026-11-12 用户问："新发高血压让 AD 降低，是不是因为没把高血压相关的指标一起改 / 筛干净？"

G×E 和第三部分的"新发高血压"只把站点上的 hypertension_cum 设成 1；收缩压 / 舒张压（sbp_avg / dbp_avg）和降压药
（antihyp_rx）原样不动，资格也只看 hypertension_cum == 0。这里检查两件事：
  A（数据，只出聚合数）：真实"新发高血压"那次访视（上次 0 -> 这次 1）的血压、降压药是什么样；资格人群里已经在吃降压药 /
    血压 >= 140 的有多少；"只有高血压标记、没药、血压正常"这种组合在 train 里多不多；新发高血压和之后 AD 的数据关联。
  B（模型）：G×E 用的 5 个 VISTA（transformer-geno2s-s<seed>），同一批站点（val + test 有芯片数据的人，第 2 / 4 / 6 次访视，
    站点时 hypertension_cum = 0 且不缺失），每个 seed 所有臂共用随机数。臂（基因原样，只改站点这次访视）：
      orig；htn = 现在的改法；rx = 只改降压药；bp = 只把血压设成 train 里真实新发那次访视的平均值；bp160 = 160/90；
      htn_rx；full = htn + rx + bp（像一次真实的新发访视）；full160 = htn + rx + 160/90。
    读 10 年（和 5 年）AD 累积风险（只用站点时还没 AD 的）、10 年死亡；先每个 seed 算差，再按 seed 平均；CI = 按人 bootstrap。
    另把 htn 臂按"资格干净"（站点时降压药测过且为 0、收缩压测过且 < 140）拆开看。

    sbatch -p general-short --time=00:30:00 --cpus-per-task=8 --mem=32G ris/py.sbatch ../_probes_20261112/probe_htn_bp.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())                                   # models_code（py.sbatch 在那里跑）
import numpy as np  # noqa: E402

import bl_data as B  # noqa: E402
import simulator as SIM  # noqa: E402
from common import RUN_DIR  # noqa: E402
from geno2_perturb import COHORTS, MODEL, landmarks, stack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_MC = 100
N_BOOT = 1000
SEEDS = (0, 1, 2, 3, 4)
LIMIT = int(os.environ.get("PROBE_LIMIT", "0"))


def main():
    import torch
    import perturb_v2 as PV
    import rollout_versions as RV
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    Z, meta = B.load("newmodel_data_trans_geno2.npz")
    feats, st = meta["features"], meta["static"]
    mu, sd = meta["norm"]["mu"], meta["norm"]["sd"]
    FI = {f: i for i, f in enumerate(feats)}
    SI = {f: i for i, f in enumerate(st)}
    iq_ad, iq_d = B.Q.index("ad"), B.Q.index("death")
    out = {"note": __doc__.split("\n")[0]}

    def raw(x, f):
        return x[..., FI[f]] * sd[f] + mu[f]

    def obs(x, f):
        return x[..., FI[f + "_missing"]] < 0.5

    # ============================================================ A：数据
    def describe(x, label):
        rx_ok, sbp_ok, dbp_ok = obs(x, "antihyp_rx"), obs(x, "sbp_avg"), obs(x, "dbp_avg")
        sbp, dbp = raw(x, "sbp_avg"), raw(x, "dbp_avg")
        return {"label": label, "n_visits": int(len(x)),
                "rx_measured": float(rx_ok.mean()), "rx_1_among_measured": float((x[rx_ok, FI["antihyp_rx"]] > 0.5).mean()) if rx_ok.any() else None,
                "sbp_measured": float(sbp_ok.mean()), "sbp_mean": float(sbp[sbp_ok].mean()), "sbp_sd": float(sbp[sbp_ok].std()),
                "sbp_ge140": float((sbp[sbp_ok] >= 140).mean()), "dbp_mean": float(dbp[dbp_ok].mean()),
                "age_mean": float((x[:, FI["age_z"]] * 10 + 80).mean())}

    A = {}
    for split in ("train", "all"):
        Ds = [B.split(Z, s) for s in (("train",) if split == "train" else ("train", "test", "val"))]
        never0, onset_prev, onset_now, stay_now, ever1, flag_only = [], [], [], [], [], 0
        for D in Ds:
            X, M = D["X"], D["M"]
            for i in range(len(X)):
                v_ok = np.where(M[i] & obs(X[i], "hypertension_cum"))[0]
                for a, b in zip(v_ok[:-1], v_ok[1:]):
                    if X[i, a, FI["hypertension_cum"]] < 0.5:
                        if X[i, b, FI["hypertension_cum"]] > 0.5:
                            onset_prev.append(X[i, a]); onset_now.append(X[i, b])
                        else:
                            stay_now.append(X[i, b])
                for v in v_ok:
                    (ever1 if X[i, v, FI["hypertension_cum"]] > 0.5 else never0).append(X[i, v])
        never0, onset_prev, onset_now, stay_now, ever1 = map(np.array, (never0, onset_prev, onset_now, stay_now, ever1))
        x1 = ever1
        fo = (obs(x1, "antihyp_rx") & (x1[:, FI["antihyp_rx"]] < 0.5) & obs(x1, "sbp_avg") & (raw(x1, "sbp_avg") < 140))
        A[split] = {"htn0_visits": describe(never0, "hypertension_cum = 0"),
                    "htn1_visits": describe(ever1, "hypertension_cum = 1"),
                    "onset_visit": describe(onset_now, "new onset: this visit (prev 0 -> now 1)"),
                    "onset_prev_visit": describe(onset_prev, "new onset: the visit before"),
                    "stay0_visit": describe(stay_now, "prev 0 -> now 0"),
                    "htn1_flag_only_share": float(fo.mean()),
                    "htn1_flag_only_note": "share of htn=1 visits with antihyp_rx measured 0 and SBP measured < 140"}
    on = A["train"]["onset_visit"]
    SBP_ON, DBP_ON = on["sbp_mean"], on["dbp_mean"]
    out["A_data"] = A

    # 新发高血压 -> AD（数据关联；全部站点；上次访视 hypertension_cum = 0 的站点，暴露 = 这次变成 1）
    rows_x, rows_y5, rows_y10, rows_s, rows_p = [], [], [], [], []
    for s in ("train", "test", "val"):
        D = B.split(Z, s)
        X, M, Y, S = D["X"], D["M"], D["Y"], D["S"]
        for i in range(len(X)):
            v_ok = np.where(M[i] & obs(X[i], "hypertension_cum"))[0]
            for a, b in zip(v_ok[:-1], v_ok[1:]):
                if X[i, a, FI["hypertension_cum"]] > 0.5:
                    continue
                y = Y[i, b].reshape(B.NQ, B.NH)
                rows_x.append(X[i, b]); rows_s.append(S[i]); rows_p.append((s, int(D["pid"][i])))
                rows_y5.append(y[iq_ad, 4]); rows_y10.append(y[iq_ad, 9])
    Xa, Sa = np.array(rows_x), np.array(rows_s)
    pid = np.array([hash(p) for p in rows_p])
    expo = (Xa[:, FI["hypertension_cum"]] > 0.5).astype(float)
    mm_ok, cg_ok = obs(Xa, "cts_estmmse30"), obs(Xa, "cogn_global")
    cov = {
        "age": [Xa[:, FI["age_z"]]],
        "demo": [Sa[:, SI[c]] for c in ("male", "educ_z", "educ_na", "apoe_e4x1", "apoe_e4x2", "apoe_na") if c in SI],
        "cog": [np.where(mm_ok, Xa[:, FI["cts_estmmse30"]], 0), mm_ok.astype(float),
                np.where(cg_ok, Xa[:, FI["cogn_global"]], 0), cg_ok.astype(float)],
    }
    steps = [("crude", []), ("+age", ["age"]), ("+age/sex/educ/APOE", ["age", "demo"]),
             ("+age/sex/educ/APOE/MMSE/cognition", ["age", "demo", "cog"])]

    def fit(F, y):
        m, s_ = F.mean(0), F.std(0)
        s_ = np.where(s_ < 1e-9, 1.0, s_)
        return LogisticRegression(C=1e4, max_iter=5000).fit((F - m) / s_, y).coef_[0][0] / s_[0]

    rng = np.random.default_rng(20261112)
    assoc = {}
    for lab, yy in (("ad_5y", np.array(rows_y5)), ("ad_10y", np.array(rows_y10))):
        ok = yy >= 0
        res = {"n_landmarks": int(ok.sum()), "n_onset": int(expo[ok].sum()),
               "ad_rate_onset": float(yy[ok & (expo > 0.5)].mean()), "ad_rate_stay0": float(yy[ok & (expo < 0.5)].mean())}
        persons = np.unique(pid[ok])
        groups = {q: np.where(ok & (pid == q))[0] for q in persons}
        for name, keys in steps:
            F = np.column_stack([expo] + [c for k in keys for c in cov[k]])
            est = fit(F[ok], yy[ok].astype(int))
            bs = []
            for _ in range(100):
                idx = np.concatenate([groups[q] for q in rng.choice(persons, len(persons))])
                try:
                    bs.append(fit(F[idx], yy[idx].astype(int)))
                except ValueError:
                    pass
            res[name] = {"log_or": float(est), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        assoc[lab] = res
    out["A_onset_ad_association"] = assoc
    print(json.dumps(out, indent=1), flush=True)

    # ============================================================ B：模型
    norm = SIM.norm_info(meta)
    rows = landmarks(Z, meta, COHORTS["main"])
    D, vv = stack(Z, rows)
    xL = D["X"][np.arange(len(vv)), vv]
    ok = (xL[:, FI["hypertension_cum"]] < 0.5) & obs(xL, "hypertension_cum")
    ii = np.where(ok)[0]
    if LIMIT:
        ii = ii[:LIMIT]
    n = len(ii)
    xe = xL[ii]
    clean = obs(xe, "antihyp_rx") & (xe[:, FI["antihyp_rx"]] < 0.5) & obs(xe, "sbp_avg") & (raw(xe, "sbp_avg") < 140)
    risk_ad = np.array([rows[k]["risk__ad"] for k in ii])
    out["B_cohort"] = {"n_eligible": n, "eligible_profile": describe(xe, "eligible landmarks (htn = 0)"),
                       "n_clean": int(clean.sum()), "n_not_clean": int((~clean).sum()), "n_at_risk_ad": int(risk_ad.sum()),
                       "bp_onset_value": [SBP_ON, DBP_ON]}

    def z(f, v):
        return (v - mu[f]) / sd[f]

    def edit(X, arm):
        X = X.copy()
        k = np.arange(n)
        v = vv[ii]
        if "htn" in arm or arm.startswith("full"):
            X = np.stack([PV.translate_features("htn", X[j], D["M"][ii[j]], D["S"][ii[j]], D["L"][ii[j]], v[j], meta["norm"], feats)[0]
                          for j in range(n)])
        if "rx" in arm or arm.startswith("full"):
            X[k, v, FI["antihyp_rx"]] = 1.0
            X[k, v, FI["antihyp_rx_missing"]] = 0.0
        bp = (160.0, 90.0) if arm.endswith("160") else (SBP_ON, DBP_ON) if (arm in ("bp", "full")) else None
        if bp is not None:
            X[k, v, FI["sbp_avg"]] = z("sbp_avg", bp[0]); X[k, v, FI["sbp_avg_missing"]] = 0.0
            X[k, v, FI["dbp_avg"]] = z("dbp_avg", bp[1]); X[k, v, FI["dbp_avg_missing"]] = 0.0
        return X

    ARMS = ("orig", "htn", "rx", "bp", "bp160", "htn_rx", "full", "full160")
    base = {k: D[k][ii].copy() for k in ("X", "M", "S", "L", "T", "pid")}
    Xarm = {a: edit(base["X"], a) for a in ARMS}
    F = np.zeros((len(SEEDS), len(ARMS), n, 2, 3), np.float32)         # seed, 臂, 站点, (AD, 死亡), (5y, 10y, 2y)
    t0 = time.time()
    for si, seed in enumerate(SEEDS):
        gen = RV.load_state(os.path.join(RUN_DIR, "models", MODEL.format(seed=seed) + ".pt"))
        for ai, arm in enumerate(ARMS):
            Dp = {k: v.copy() for k, v in base.items()}
            Dp["X"] = Xarm[arm].copy()
            start = {"D": Dp, "ii": np.arange(n), "vv": vv[ii], "pid": Dp["pid"], "t": Dp["T"][np.arange(n), vv[ii]]}
            rng = np.random.default_rng(20261112 + 1000 * seed)                # 同一个 seed 的所有臂同一个种子 -> CRN
            R = SIM.risks(gen.sim(start, N_MC, rng, norm, full_draws=True), True)
            F[si, ai, :, 0] = R[:, iq_ad][:, [4, 9, 1]]
            F[si, ai, :, 1] = R[:, iq_d][:, [4, 9, 1]]
        print(f"seed {seed} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    persons = np.array([hash((rows[k]["split"], rows[k]["pid"])) for k in ii])
    up = np.unique(persons)
    groups = {q: np.where(persons == q)[0] for q in up}
    brng = np.random.default_rng(20261113)
    boots = [np.concatenate([groups[q] for q in brng.choice(up, len(up))]) for _ in range(N_BOOT)]

    def effect(arm, ep, h, mask):
        a = ARMS.index(arm)
        d = (F[:, a, :, ep, h] - F[:, 0, :, ep, h]) * 100           # 百分点
        dm = d.mean(0)
        sel = mask
        est = float(dm[sel].mean())
        bs = [dm[b][sel[b]].mean() for b in boots if sel[b].any()]
        per_seed = [float(d[s][sel].mean()) for s in range(len(SEEDS))]
        return {"pp": round(est, 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                "per_seed": [round(x, 3) for x in per_seed], "n": int(sel.sum()),
                "base_risk": round(float(F[:, 0, :, ep, h].mean(0)[sel].mean() * 100), 2)}

    allm = np.ones(n, bool)
    res = {}
    for arm in ARMS[1:]:
        res[arm] = {"ad_10y": effect(arm, 0, 1, risk_ad), "ad_5y": effect(arm, 0, 0, risk_ad),
                    "death_10y": effect(arm, 1, 1, allm)}
    res["htn_by_clean"] = {"clean": {"ad_10y": effect("htn", 0, 1, risk_ad & clean), "death_10y": effect("htn", 1, 1, clean)},
                           "not_clean": {"ad_10y": effect("htn", 0, 1, risk_ad & ~clean), "death_10y": effect("htn", 1, 1, ~clean)}}
    res["full_by_clean"] = {"clean": {"ad_10y": effect("full", 0, 1, risk_ad & clean)},
                            "not_clean": {"ad_10y": effect("full", 0, 1, risk_ad & ~clean)}}
    out["B_model"] = res
    out["minutes"] = (time.time() - t0) / 60
    f = os.path.join(HERE, "probe_htn_bp_limit.json" if LIMIT else "probe_htn_bp.json")
    json.dump(out, open(f, "w"), indent=1)
    print(json.dumps(out["B_cohort"], indent=1), json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
