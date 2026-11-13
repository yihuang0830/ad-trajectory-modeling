"""geno2_gxe_compare.py -- 基因 × 合并症（G×E）"1+1>2"测试（geno2_gxe.py）的读法。RIS 上跑。

    python geno2_gxe_compare.py     # -> $ZOO_RUN_DIR/packages/_geno2/gxe.json（聚合，可以拉回本地）

读法在看结果之前定下，之后不改。
第一轮（2026-11-12）：
 组合：基因 2 个（APOE ε3/ε3 → ε3/ε4；免疫 PRS p10 → p90）× 新发合并症 5 个（高血压、糖尿病、心脏病、卒中、抑郁）= 10 组。
 题目：AD、MMSE < 24、认知 < −1；只用站点时处在风险中、而且站点时还没有这个病的站点。
 每个站点先在 5 个 seed 上平均，读站点后 10 年的累积风险：R00 都没有、R10 只有基因风险、R01 只有合并症、R11 两者都有。
   相加尺度：IC = R11 − R10 − R01 + R00（百分点）；> 0 = 1+1>2。另报 RERI = IC / R00。
   相乘尺度：logit(R11) − logit(R10) − logit(R01) + logit(R00)（每个站点算、再平均）；> 0 = 1+1>2。
   时间尺度：相对 R00 那条平均累计曲线的提前年数（geno2_compare.shift_years），Y11 − Y10 − Y01；> 0 = 1+1>2。
   参照：log OR 上完全没有交互（纯相乘）时，相加尺度也会自动有的 IC_null = expit(logit R10 + logit R01 − logit R00) − R10 − R01 + R00。
     相加尺度的 1+1>2 里有多少是这一部分，要一起看。
 CI：按人整体重抽 1000 次（一个人的几个站点一起进出）。另报 5 个 seed 各自的值。
 判定"1+1>2"：这个尺度上 95% CI 下限 > 0，而且 ≥ 4 个 seed 为正（"1+1<2"反过来）。10 组 × 3 道题 × 3 种尺度，只作描述，
   不做多重比较校正；CI 只反映"换一批人"（5 个模型固定），训练的波动看 seed。
 这些只用模型在扰动记录上的预测，不用标签；扰动不是因果效应。
第二轮（2026-11-12，用户要更多基因变量、更多终点；在看第二轮任何结果之前定下）：
 组合：geno2_gxe.GENES 的 13 个基因变量（一律朝风险更高的方向改）× 同样 5 个合并症 = 65 组；第一轮的 10 组在里面逐位重算，
   和第一轮的原始风险逐位核对（v1_check）。
 题目：AD 是主读法；MMSE < 27、MMSE < 24、认知 < 0、认知 < −1 是次要读法；死亡只作背景（报基因、合并症各让 10 年死亡变多少；
   死亡的交互照算，但不当作协同来读）。"处在风险中"的规则见 geno2_perturb.landmarks（死亡都算）。
 每个格子的算法、CI、判定规则和第一轮完全相同。
 每组的归类（只看 AD；G = R10 − R00、E = R01 − R00 是基因、合并症单独的效应）：
   A "1+1>2"：相加、相乘两种尺度都判成 1+1>2，而且 G > 0、E > 0（两个风险因素叠加，超过纯相乘）—— 这是"候选"；
   B "只有一种尺度 1+1>2"：相加、相乘里只有一种判成 1+1>2，而且 G > 0、E > 0（多半是相加过线、相乘没有：
     没超过两个效应按比值相乘自然带出来的部分）；
   C "正交互，但不是字面的 1+1>2"：相加或相乘判成 1+1>2，但 G、E 里有 ≤ 0 的（比如合并症在模型里让 AD 变少）；
   D "1+1<2"：两种尺度都没有判成 1+1>2，至少一种判成 1+1<2；
   E "看不出"：其余。
   另数认知四道里相乘尺度判成 1+1>2 / 1+1<2 的各有几道，作为跨终点一致性的描述。
 13 × 5 × 6 × 3 = 1,170 个判定，不做多重比较校正，只作描述；单个格子过线可能是偶然，要看跨终点、跨 seed 一致不一致，以及量的大小。
"""
import json
import os

import numpy as np
import pandas as pd

import bl_data  # noqa: F401  （先导入：它把 unified_eval 加进路径）
from common import RUN_DIR
from geno2_compare import lg, shift_years
from geno2_gxe import COMORB, GENES, RISK_QS
from geno_compare import N_BOOT, SEED, SEEDS

COG = ("mmse_lt27", "mmse_lt24", "cogn_lt0", "cogn_ltm1")
GX1 = os.path.join(RUN_DIR, "geno", "gxe")
GX2 = os.path.join(RUN_DIR, "geno", "gxe2")
OUT = os.path.join(RUN_DIR, "packages", "_geno2")
H = 9                                                       # 第 10 年


def expit(x):
    return 1 / (1 + np.exp(-x))


def _ci(x):
    return [float(v) for v in np.quantile(x, (0.025, 0.975))]


def verdict(ci, seeds):
    pos = sum(s > 0 for s in seeds)
    if ci[0] > 0 and pos >= 4:
        return "1+1>2"
    if ci[1] < 0 and pos <= 1:
        return "1+1<2"
    return "≈ 相加 / 相乘（看不出协同）"


def cell(F, Wr):
    """F：seed, G, E, 站点, 年（已经只留处在风险中的站点）。和第一轮同一套算法。"""
    den = Wr.sum(1)
    M = F.mean(0)                                                       # G, E, 站点, 年
    R = M[..., H]
    ic = R[1, 1] - R[1, 0] - R[0, 1] + R[0, 0]
    mult = lg(R[1, 1]) - lg(R[1, 0]) - lg(R[0, 1]) + lg(R[0, 0])
    null = expit(lg(R[1, 0]) + lg(R[0, 1]) - lg(R[0, 0])) - R[1, 0] - R[0, 1] + R[0, 0]
    C = M.mean(2)                                                       # G, E, 年（平均累计曲线）
    yr = lambda c: (shift_years(c[0, 0], c[1, 0]), shift_years(c[0, 0], c[0, 1]), shift_years(c[0, 0], c[1, 1]))  # noqa: E731
    y10, y01, y11 = yr(C)
    BC = np.einsum("bn,geny->bgey", Wr, M) / den[:, None, None, None]
    yb = np.array([(lambda t: t[2] - t[0] - t[1])(yr(BC[b])) for b in range(len(Wr))])
    sd_ic = [float((f[1, 1, :, H] - f[1, 0, :, H] - f[0, 1, :, H] + f[0, 0, :, H]).mean()) for f in F]
    sd_mu = [float((lg(f[1, 1, :, H]) - lg(f[1, 0, :, H]) - lg(f[0, 1, :, H]) + lg(f[0, 0, :, H])).mean()) for f in F]
    sd_yr = [(lambda t: t[2] - t[0] - t[1])(yr(f.mean(2))) for f in F]
    R00 = float(R[0, 0].mean())
    c = {"n": int(F.shape[3]),
         "risk_10y": {"none": R00, "gene_only": float(R[1, 0].mean()), "comorb_only": float(R[0, 1].mean()),
                      "both": float(R[1, 1].mean())},
         "additive": {"ic_pp": float(ic.mean() * 100), "ci95_pp": [v * 100 for v in _ci(Wr @ ic / den)],
                      "reri": float(ic.mean() / R00), "ic_null_pp": float(null.mean() * 100),
                      "ic_beyond_null_pp": float((ic - null).mean() * 100),
                      "per_seed_pp": [round(v * 100, 3) for v in sd_ic]},
         "multiplicative": {"dlogor": float(mult.mean()), "ci95": _ci(Wr @ mult / den),
                            "ratio_of_or": float(np.exp(mult.mean())), "per_seed": [round(v, 4) for v in sd_mu]},
         "years": {"gene_only": y10, "comorb_only": y01, "both": y11, "synergy": y11 - y10 - y01,
                   "ci95": _ci(yb), "per_seed": [round(v, 3) for v in sd_yr]}}
    c["additive"]["verdict"] = verdict(c["additive"]["ci95_pp"], sd_ic)
    c["multiplicative"]["verdict"] = verdict(c["multiplicative"]["ci95"], sd_mu)
    c["years"]["verdict"] = verdict(c["years"]["ci95"], sd_yr)
    return c


def classify(r):
    a = r["ad"]
    g = a["risk_10y"]["gene_only"] - a["risk_10y"]["none"]
    e = a["risk_10y"]["comorb_only"] - a["risk_10y"]["none"]
    add, mul = a["additive"]["verdict"], a["multiplicative"]["verdict"]
    if "1+1>2" in (add, mul):
        k = ("A" if add == mul else "B") if g > 0 and e > 0 else "C"
    elif "1+1<2" in (add, mul):
        k = "D"
    else:
        k = "E"
    mc = [r[q]["multiplicative"]["verdict"] for q in COG]
    return {"class": k, "ad_gene_only_pp": 100 * g, "ad_comorb_only_pp": 100 * e,
            "cog_mult_gt": sum(v == "1+1>2" for v in mc), "cog_mult_lt": sum(v == "1+1<2" for v in mc),
            "death_gene_only_pp": 100 * (r["death"]["risk_10y"]["gene_only"] - r["death"]["risk_10y"]["none"]),
            "death_comorb_only_pp": 100 * (r["death"]["risk_10y"]["comorb_only"] - r["death"]["risk_10y"]["none"])}


def v1_check(Z2):
    """第一轮的原始风险（geno/gxe/）和这一轮重算的同一组臂逐位比。"""
    out = {}
    for k in SEEDS:
        f1 = os.path.join(GX1, f"transformer-geno2s-s{k}.npz")
        if not os.path.exists(f1):
            continue
        z1 = np.load(f1)
        for at in COMORB:
            z2 = Z2[k][at]
            assert np.array_equal(z1[f"idx__{at}"], z2["idx"]), f"s{k} {at}：站点不一样"
            for g in ("apoe", "immune"):
                lo, hi = GENES[g]
                d = np.abs(z1[f"F__{g}__{at}"] - np.stack([z2["F"][lo], z2["F"][hi]]))
                out[f"s{k}:{g}x{at}"] = {"max_abs": float(d.max()), "frac_equal": float((d == 0).mean())}
            for q in ("ad", "mmse_lt24", "cogn_ltm1"):
                assert np.array_equal(z1[f"risk__{q}"], z2[f"risk__{q}"]), f"s{k} {q}：处在风险中的标记不一样"
    if not out:
        return {"n": 0}
    return {"n": len(out), "max_abs": max(v["max_abs"] for v in out.values()),
            "frac_equal_min": min(v["frac_equal"] for v in out.values()), "by_pair": out}


def load_all():
    Z2 = {}
    for k in SEEDS:
        Z2[k] = {}
        for at in COMORB:
            z = np.load(os.path.join(GX2, f"transformer-geno2s-s{k}-{at}.npz"))
            arms = [a.removeprefix("F__") for a in z.files if a.startswith("F__")]
            Z2[k][at] = {"F": {a: z[f"F__{a}"] for a in arms}, "idx": z["idx"], "q": list(z["q"]),
                         "split": z["split"], "pid": z["pid"], "visit": z["visit"], "info": json.loads(str(z["info"])),
                         **{f"risk__{q}": z[f"risk__{q}"] for q in RISK_QS}}
    return Z2


def main():
    Z2 = load_all()
    z0 = Z2[SEEDS[0]][COMORB[0]]
    for k in SEEDS:
        for at in COMORB:
            z = Z2[k][at]
            for f in ("split", "pid", "visit", *[f"risk__{q}" for q in RISK_QS]):
                assert np.array_equal(z[f], z0[f]), f"s{k} {at} {f} 和别的文件不一样"
            assert np.array_equal(z["idx"], Z2[SEEDS[0]][at]["idx"])
    key = pd.Series(z0["split"].astype(str)) + ":" + pd.Series(z0["pid"]).astype(str)
    codes, uniq = pd.factorize(key)
    rng = np.random.default_rng(SEED)
    Wp = np.stack([np.bincount(rng.integers(len(uniq), size=len(uniq)), minlength=len(uniq)) for _ in range(N_BOOT)]).astype(float)
    qi = {q: z0["q"].index(q) for q in RISK_QS}
    res = {"info_seed0": {at: Z2[SEEDS[0]][at]["info"] for at in COMORB}, "n_boot": N_BOOT, "genes": GENES,
           "questions": list(RISK_QS), "v1_check": v1_check(Z2), "pairs": {}, "summary": {}}
    print("v1 check:", json.dumps({k: v for k, v in res["v1_check"].items() if k != "by_pair"}), flush=True)
    for g, (lo, hi) in GENES.items():
        for at in COMORB:
            ii = Z2[SEEDS[0]][at]["idx"]
            r = {}
            for q in RISK_QS:
                m = z0[f"risk__{q}"][ii]
                F = np.stack([np.stack([Z2[k][at]["F"][lo], Z2[k][at]["F"][hi]])[:, :, m, qi[q], :] for k in SEEDS])
                r[q] = cell(F, Wp[:, codes[ii][m]])
            p = f"{g}x{at}"
            res["pairs"][p] = r
            res["summary"][p] = s = classify(r)
            a = r["ad"]
            print(f"{g:11s} x {at:10s} [{s['class']}] AD n{a['n']}: G {s['ad_gene_only_pp']:+5.1f} E {s['ad_comorb_only_pp']:+5.1f} | "
                  f"IC {a['additive']['ic_pp']:+.2f}pp (null {a['additive']['ic_null_pp']:+.2f}) {a['additive']['verdict'][:5]} | "
                  f"ORr {a['multiplicative']['ratio_of_or']:.3f} {a['multiplicative']['verdict'][:5]} | "
                  f"yrs {a['years']['synergy']:+.2f} {a['years']['verdict'][:5]} | cog x>2 {s['cog_mult_gt']}/4 <2 {s['cog_mult_lt']}/4 | "
                  f"death G {s['death_gene_only_pp']:+.1f} E {s['death_comorb_only_pp']:+.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "gxe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
