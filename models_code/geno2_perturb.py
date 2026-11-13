"""geno2_perturb.py -- 基因第二版的扩大扰动（2026-11-11，用户定）：读存好的 VISTA 模型，在更多人、更多站点上做基因扰动。

    python geno2_perturb.py --seed 0 --cohort main  --part 0 --nparts 2
    python geno2_perturb.py --seed 0 --cohort train --part 0 --nparts 3
      -> $ZOO_RUN_DIR/geno/pert2/transformer-geno2s-s<seed>-<cohort>-p<part>.npz（人级，只留在 RIS）

模型：run.py --variant geno2s --save_model 存下的 $ZOO_RUN_DIR/models/transformer-geno2s-s<seed>.pt
（和 geno2 同一份数据、同一个 seed；它的 val 打分表和 -geno2- 逐位相同才算同一个模型，见 geno2_pert_compare.py）。

人和站点（用户 2026-11-11 定）：
  main  = val + test 里有芯片数据的人（模型没训练过：test 只用来挑停在哪一轮）；train = train 段有芯片数据的人（敏感性：模型见过）。
  站点 = 第 2、4、6 次访视（有这么多次访视的才算），一个人最多 3 个站点。
扰动臂：perturb.geno2_arms。main 做单变量臂 + 基因变量 × APOE 的 2 × 2 析因臂；train 只做单变量臂。
每道题"处在风险中"按站点算（和 perturb_v2.cohort_table 同一条规则）：AD = 到站点为止还没有 AD；
  MMSE < 24 / 认知 < −1 = 到站点为止测过、而且从没进过那一档；死亡都算。规则另在 val 第 4 次访视上和 cohort_table 逐人核对，结果记进输出。
模拟：每条记录 100 条 10 年轨迹（和 run.py 的 --n_mc_pert 一样），风险平滑成 (k + 0.5) / (n + 1)。
  共用随机数（CRN）：同一块里每个臂用同一个种子、同一批站点同一个顺序，simulate_visits(full_draws=True) 让第 j 条轨迹在各个臂里
  每一步用同一串随机数；臂之间的差只来自输入的改动（2026-10-31 记下的"扰动没有共用随机数"在这里补上）。
"""
import argparse
import json
import os
import time

import numpy as np

import bl_data as B
import simulator as SIM
from common import RUN_DIR
from perturb import geno2_arms

VISITS = (2, 4, 6)
COHORTS = {"main": ("val", "test"), "train": ("train",)}
N_MC = 100
QS_AT_RISK = ("ad", "mmse_lt24", "cogn_ltm1")
OUT = os.path.join(RUN_DIR, "geno", "pert2")
MODEL = "transformer-geno2s-s{seed}"


def landmarks(Z, meta, splits):
    """有芯片数据的人 × 第 2 / 4 / 6 次访视 -> 每个站点一行：数据段、段内下标、站点下标、pid、处在风险中的标记。"""
    norm = SIM.norm_info(meta)
    names, mu, sd = norm["names"], norm["mu"], norm["sd"]
    jm, jc = names.index("cts_estmmse30"), names.index("cogn_global")
    ig = meta["static"].index("geno_na")
    rows = []
    for s in splits:
        D = B.split(Z, s)
        for i in np.where(D["S"][:, ig] < 0.5)[0]:
            nv = int(D["M"][i].sum())
            assert D["M"][i, :nv].all(), "访视掩码不连续"
            for v in VISITS:
                if nv < v:
                    continue
                vL = v - 1
                X = D["X"][i, :vL + 1]
                worst = {}
                for f, j in (("mmse", jm), ("cogn", jc)):
                    ok = X[:, SIM.I_MC + j] < 0.5
                    # SIM.stage_of 的编码：0 = 最好、2 = 最差（MMSE < 24 / 认知 < −1），和真值表的编码相反；从没测过记 3（哪道题都不在风险中）
                    worst[f] = int(SIM.stage_of(X[ok, j] * sd[j] + mu[j], f).max()) if ok.any() else 3
                # MMSE < 27 / 认知 < 0：到站点为止测过、一直在最好那一档；MMSE < 24 / 认知 < −1：从没进过最差那一档（2026-11-12 加了前两道和死亡，
                # 原来三道的规则不变）
                rows.append({"split": s, "i": int(i), "vL": vL, "visit": v, "pid": int(D["pid"][i]), "t": int(D["T"][i, vL]),
                             "risk__ad": bool(X[:, SIM.I_AD].max() < 0.5),
                             "risk__mmse_lt27": worst["mmse"] == 0, "risk__mmse_lt24": worst["mmse"] <= 1,
                             "risk__cogn_lt0": worst["cogn"] == 0, "risk__cogn_ltm1": worst["cogn"] <= 1, "risk__death": True})
    return rows


def check_rule(rows):
    """同一条规则在 val 第 4 次访视上和 perturb_v2.cohort_table 逐人比（只有 val 有现成的表）。"""
    import perturb_v2 as PV
    C, _ = PV.cohort_table()
    mine = {r["pid"]: r for r in rows if r["split"] == "val" and r["visit"] == 4}
    both = [p for p in mine if p in C.index]
    out = {"n_compared": len(both)}
    for q in QS_AT_RISK:
        out[q] = float(np.mean([mine[p][f"risk__{q}"] == bool(C.loc[p, f"risk__{q}"]) for p in both])) if both else None
    return out


def stack(Z, rows):
    Ds = {s: B.split(Z, s) for s in {r["split"] for r in rows}}
    pick = lambda k: np.stack([Ds[r["split"]][k][r["i"]] for r in rows])   # noqa: E731
    D = {k: pick(k) for k in ("X", "M", "S", "L", "T", "pid")}
    vv = np.array([r["vL"] for r in rows])
    return D, vv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cohort", choices=tuple(COHORTS), required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只做前几个站点、前几个臂，写到 pert2/_limit/")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    Z, meta = B.load("newmodel_data_trans_geno2.npz")
    norm = SIM.norm_info(meta)
    import rollout_versions as RV
    gen = RV.load_state(os.path.join(RUN_DIR, "models", MODEL.format(seed=a.seed) + ".pt"))
    rows = landmarks(Z, meta, COHORTS[a.cohort])
    info = {"cohort": a.cohort, "splits": COHORTS[a.cohort], "n_landmarks_all": len(rows),
            "n_by_visit": {v: sum(r["visit"] == v for r in rows) for v in VISITS},
            "n_persons": len({(r["split"], r["pid"]) for r in rows})}
    if a.cohort == "main" and a.part == 0:
        info["rule_check_vs_cohort_table"] = check_rule(rows)
    rows = [r for k, r in enumerate(rows) if k % a.nparts == a.part]
    arms = geno2_arms(Z, meta, interaction=a.cohort == "main")
    names = list(arms)
    if a.limit:
        rows, names = rows[:a.limit], names[:4]
    st_names = meta["static"]
    D, vv = stack(Z, rows)
    n = len(rows)
    F = np.zeros((len(names), n, B.NQ, B.NH), np.float32)
    t0 = time.time()
    for k, arm in enumerate(names):
        Dp = dict(D)
        Dp["S"] = D["S"].copy()
        for c, v in arms[arm].items():
            Dp["S"][:, st_names.index(c)] = v
        start = {"D": Dp, "ii": np.arange(n), "vv": vv, "pid": D["pid"], "t": D["T"][np.arange(n), vv]}
        rng = np.random.default_rng(20261111 + 1000 * a.seed + 10 * a.part + (a.cohort == "train"))   # 每个臂同一个种子 -> CRN
        F[k] = SIM.risks(gen.sim(start, N_MC, rng, norm, full_draws=True), True)
        if k == 0:
            print(f"first arm: {n} landmarks in {(time.time() - t0) / 60:.2f} min", flush=True)
    info.update({"part": a.part, "nparts": a.nparts, "n_landmarks": n, "n_arms": len(names), "n_mc": N_MC,
                 "minutes": (time.time() - t0) / 60, "model": MODEL.format(seed=a.seed)})
    od = os.path.join(OUT, "_limit") if a.limit else OUT
    os.makedirs(od, mode=0o700, exist_ok=True)
    f = os.path.join(od, f"{MODEL.format(seed=a.seed)}-{a.cohort}-p{a.part}.npz")
    np.savez_compressed(f + ".tmp.npz", F=F, arm=np.array(names), q=np.array(B.Q), levels=json.dumps(arms),
                        split=np.array([r["split"] for r in rows]), pid=np.array([r["pid"] for r in rows], np.int64),
                        visit=np.array([r["visit"] for r in rows], np.int64), tland=np.array([r["t"] for r in rows], np.int64),
                        **{f"risk__{q}": np.array([r[f"risk__{q}"] for r in rows]) for q in QS_AT_RISK},
                        info=json.dumps(info))
    os.replace(f + ".tmp.npz", f)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
