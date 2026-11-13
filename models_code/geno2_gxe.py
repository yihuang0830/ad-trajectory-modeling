"""geno2_gxe.py -- 基因 × 合并症（G×E）"1+1>2"的测试（2026-11-12）。

    python geno2_gxe.py --seed 0 --comorb dm   # -> $ZOO_RUN_DIR/geno/gxe2/transformer-geno2s-s<seed>-<合并症>.npz（人级，只留在 RIS）

背景：想用模型挑出最高风险的"Genotype × Comorbidity"协同组合（例：APOE4 × 高血压），
协同 = 一起出现时加速超过各自单独的效应。在 VISTA 上做 2 × 2 析因扰动看看有没有。
  第一轮（用户："先简单测试一下"，相加、相乘两种尺度都报）：APOE ε3/ε4、免疫 PRS × 5 个新发合并症，输出在 geno/gxe/（一个 seed 一个文件）。
  第二轮（用户："更多基因变量（SORL1、BIN1、ε4/ε4 等）"，"不只看 AD，MMSE 啥的也测测"）：模型里全部 13 个基因变量 × 同样 5 个合并症，
    每道题都存（AD、MMSE < 27、MMSE < 24、认知 < 0、认知 < −1、死亡）；一个作业 = 一个 seed × 一个合并症，输出在 geno/gxe2/。

模型：run.py --variant geno2s --save_model 存下的 5 个模型（和 geno2 逐位相同）。
人和站点：同 geno2_perturb.py 的主分析：val + test 有芯片数据的人，第 2 / 4 / 6 次访视。
2 × 2 析因（其余输入一概不动）：
  基因 G（GENES，13 个，一律朝"风险更高"的方向改，所以交互 > 0 一律读作 1+1>2）：
    APOE：ε3/ε3 → ε3/ε4（第一轮）、ε3/ε3 → ε4/ε4、ε2/ε3 → ε3/ε3（= 去掉 ε2 的保护）；S 里 APOE 几列统一设成对应基因型。
    Tier 1：BIN1 / CLU / PICALM 风险等位基因 0 → 2 份（和主扰动相同）。SORL1 改成 1 → 2 份：rs11218343 的风险等位基因很常见，
      0 份（保护等位基因纯合）的人极少，主扰动的 0 → 2 份在交互里是外推；1 → 2 份 = 保护等位基因携带者 → 非携带者。
    通路 PRS 6 个：train 段有基因的人的 p10 → p90（免疫是第一轮的）。脂质、淀粉样 PRS 在模型里主效应是反的（README 第 2 节），照算照报。
  合并症 E：站点上新发高血压 / 糖尿病 / 心脏病 / 卒中 / 抑郁（perturb_v2 的新发原子，改法 perturb_v2.translate_features，和第三部分相同）。
    只用站点时还没有这个病的站点：特征为 0 且不缺失（val 第 4 次访视上另和 cohort_table 的 atom__<病> 逐人核对，结果记进输出）。
共用随机数（CRN）：同一个 seed、同一个合并症的所有臂（24 个基因臂 × 有无合并症）用同一个种子、同一批站点同一个顺序；种子只由 seed 和
  合并症定（和第一轮相同），所以第一轮的 APOE ε3/ε4、免疫 PRS 在这里会重算一遍，可以和第一轮逐位核对（geno2_gxe_compare.py 里做）。
模拟同 geno2_perturb.py：每条记录 100 条 10 年轨迹，风险平滑成 (k + 0.5) / (n + 1)。读法在 geno2_gxe_compare.py。
"""
import argparse
import json
import os
import time

import numpy as np

import bl_data as B
import simulator as SIM
from common import RUN_DIR
from geno2_perturb import COHORTS, MODEL, landmarks, stack
from perturb import geno2_arms

# 基因变量 -> (低风险臂, 高风险臂)
GENES = {
    "apoe": ("apoe_33", "apoe_34"),
    "apoe44": ("apoe_33", "apoe_44"),
    "apoe_noe2": ("apoe_23", "apoe_33"),
    "bin1": ("t1_BIN1__0", "t1_BIN1__2"),
    "clu": ("t1_CLU__0", "t1_CLU__2"),
    "picalm": ("t1_PICALM__0", "t1_PICALM__2"),
    "sorl1": ("t1_SORL1__1", "t1_SORL1__2"),
    "immune": ("prs_immune__p10", "prs_immune__p90"),
    "lipid": ("prs_lipid__p10", "prs_lipid__p90"),
    "endocytosis": ("prs_endocytosis__p10", "prs_endocytosis__p90"),
    "amyloid": ("prs_amyloid__p10", "prs_amyloid__p90"),
    "tau": ("prs_tau__p10", "prs_tau__p90"),
    "rest": ("prs_rest__p10", "prs_rest__p90"),
}
COMORB = ("htn", "dm", "heart", "stroke", "depression")
RISK_QS = ("ad", "mmse_lt27", "mmse_lt24", "cogn_lt0", "cogn_ltm1", "death")
N_MC = 100
OUT = os.path.join(RUN_DIR, "geno", "gxe2")


def gxe_arms(Z, meta):
    """GENES 用到的臂：perturb.geno2_arms 的那些，另加 SORL1 1 份（不加进 geno2_arms：geno2_perturb.py 会把那里的臂全跑一遍）。"""
    arms = geno2_arms(Z, meta)
    c = "t1_SORL1"
    arms[f"{c}__1"] = {c: 1.0 - meta["geno2"]["tier1_train_mean"][c]}
    need = list(dict.fromkeys(a for pair in GENES.values() for a in pair))
    miss = [a for a in need if a not in arms]
    assert not miss, f"没有这些臂：{miss}"
    return {a: arms[a] for a in need}


def eligible(D, vv, feats, at):
    """站点时还没有这个病：特征为 0，而且（有缺失掩码的）不缺失。"""
    import perturb_v2 as PV
    f = PV.NEW_ONSET[at][2]
    x = D["X"][np.arange(len(vv)), vv]
    ok = x[:, feats.index(f)] < 0.5
    if f"{f}_missing" in feats:
        ok &= x[:, feats.index(f"{f}_missing")] < 0.5
    return ok


def checks(rows, ok, at):
    """val 第 4 次访视（只有 val 有现成的表）：资格和 cohort_table 的 atom__<病>、六道题"处在风险中"和 risk__<题> 逐人核对。"""
    import perturb_v2 as PV
    C, _ = PV.cohort_table()
    idx = [k for k, r in enumerate(rows) if r["split"] == "val" and r["visit"] == 4 and r["pid"] in C.index]
    agree = lambda f: float(np.mean([f(k) for k in idx])) if idx else None   # noqa: E731
    return {"n": len(idx),
            "eligible": agree(lambda k: bool(ok[k]) == bool(C.loc[rows[k]["pid"], f"atom__{at}"])),
            "at_risk": {q: agree(lambda k: rows[k][f"risk__{q}"] == bool(C.loc[rows[k]["pid"], f"risk__{q}"])) for q in RISK_QS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--comorb", choices=COMORB, required=True)
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只做前几个站点、前 2 个臂，写到 gxe2/_limit/")
    a = ap.parse_args()
    import torch
    import perturb_v2 as PV
    import rollout_versions as RV
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    Z, meta = B.load("newmodel_data_trans_geno2.npz")
    norm, feats, st = SIM.norm_info(meta), meta["features"], meta["static"]
    gen = RV.load_state(os.path.join(RUN_DIR, "models", MODEL.format(seed=a.seed) + ".pt"))
    rows = landmarks(Z, meta, COHORTS["main"])
    D, vv = stack(Z, rows)
    arms = gxe_arms(Z, meta)
    at = a.comorb
    ok = eligible(D, vv, feats, at)
    info = {"comorb": at, "n_landmarks": len(rows), "eligible": int(ok.sum()), "check_vs_cohort_table": checks(rows, ok, at),
            "genes": GENES, "arms": arms, "n_mc": N_MC, "model": MODEL.format(seed=a.seed)}
    ii = np.where(ok)[0]
    names = list(arms)
    if a.limit:
        ii, names = ii[:a.limit], names[:2]
    n = len(ii)
    Xe = np.stack([PV.translate_features(at, D["X"][i], D["M"][i], D["S"][i], D["L"][i], vv[i], meta["norm"], feats)[0]
                   for i in ii])                                   # 和 perturb.records 一样传 meta["norm"]
    out, t0 = {}, time.time()
    for arm in names:
        F = np.zeros((2, n, B.NQ, B.NH), np.float32)               # [没有合并症, 有合并症]
        for ei in (0, 1):
            Dp = {k: D[k][ii].copy() for k in ("X", "M", "S", "L", "T", "pid")}
            for c, v in arms[arm].items():
                Dp["S"][:, st.index(c)] = v
            if ei:
                Dp["X"] = Xe.copy()
            start = {"D": Dp, "ii": np.arange(n), "vv": vv[ii], "pid": Dp["pid"], "t": Dp["T"][np.arange(n), vv[ii]]}
            rng = np.random.default_rng(20261112 + 1000 * a.seed + COMORB.index(at))      # 所有臂同一个种子 -> CRN（和第一轮同一个种子）
            F[ei] = SIM.risks(gen.sim(start, N_MC, rng, norm, full_draws=True), True)
        out[f"F__{arm}"] = F
        if arm == names[0]:
            print(f"first arm: {n} landmarks in {(time.time() - t0) / 60:.2f} min", flush=True)
    info["minutes"] = (time.time() - t0) / 60
    od = os.path.join(OUT, "_limit") if a.limit else OUT
    os.makedirs(od, mode=0o700, exist_ok=True)
    f = os.path.join(od, f"{MODEL.format(seed=a.seed)}-{at}.npz")
    np.savez_compressed(f + ".tmp.npz", **out, idx=ii, q=np.array(B.Q), split=np.array([r["split"] for r in rows]),
                        pid=np.array([r["pid"] for r in rows], np.int64), visit=np.array([r["visit"] for r in rows], np.int64),
                        **{f"risk__{q}": np.array([r[f"risk__{q}"] for r in rows]) for q in RISK_QS},
                        info=json.dumps(info))
    os.replace(f + ".tmp.npz", f)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
