"""geno2_prep.py -- 基因第二版第 2 步：定目标变异清单（PRS-CS 的 --bim_prefix）和 PRS-CS 的汇总统计输入。

    python geno2_prep.py      # 读 _runs/geno/hm3/chr*.npz + Bellenguez 2022 stage I 全基因组汇总统计
                              # -> _runs/geno/hm3/target.bim、_runs/geno/prscs/sst.txt、_runs/geno/prscs/prep.json（聚合）

质量门槛（看任何结局之前定）：推断 r² ≥ 0.8（和 两层基因设计文档 对 Tier 1 的常见变异要求一致），ROSMAP 次要等位基因频率 ≥ 0.01，
ROSMAP 的面板 A1 频率和 1000G 欧裔的差 ≤ 0.2（面板的 MAF 列就是 A1 的频率；这里先核对这一点，对不上就报错停下）。
权重来源：Bellenguez 2022 stage I（GWAS Catalog 的 GRCh38 版，有 rsID）。stage I 不含 ADGC（ROSMAP 在 ADGC 里，只在 stage II），
满足 两层基因设计文档 "权重要和我们的样本不重叠"。按 rsID 只留面板里的变异；同一个 rsID 多行的整个丢掉。
效应等位基因 = A1、BETA = log OR、SE（PRS-CS 的 BETA + SE 格式）；方向和面板对不上的交给 PRS-CS 自己翻转 / 丢弃。
GWAS 样本量（--n_gwas）= 这些变异的有效样本量 4 / (1/n_cases + 1/n_controls) 的中位数（stage I 的病例含 UK Biobank 的代理病例）。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geno_build import OUT  # noqa: E402

SUMSTATS = os.environ.get("AD_SUMSTATS",
                          "/path/to/gwas/AD_Bellenguez2022_GRCh38.tsv.gz")
HM3 = os.path.join(OUT, "hm3")
PRSCS = os.path.join(OUT, "prscs")
R2_MIN, MAF_MIN, FREQ_TOL = 0.8, 0.01, 0.2


def targets():
    rows, per_chr = [], {}
    for c in range(1, 23):
        f = os.path.join(HM3, f"chr{c}.npz")
        assert os.path.exists(f), f"缺 {f}（先跑 geno2_hm3.py --chr {c}）"
        z = np.load(f)
        t = pd.DataFrame({"CHR": c, "SNP": z["snp"], "BP": z["bp"], "A1": z["a1"], "A2": z["a2"],
                          "af": z["af"], "maf_ref": z["maf_ref"], "r2": z["r2"]})
        rows.append(t)
    T = pd.concat(rows, ignore_index=True)
    diff = (T.af - T.maf_ref).abs()
    diff_flip = (1 - T.af - T.maf_ref).abs()
    agree = float((diff < diff_flip).mean())
    assert agree > 0.95, f"面板 MAF 看起来不是 A1 的频率（只有 {agree:.1%} 更接近 A1）"
    maf = np.minimum(T.af, 1 - T.af)
    ok = (T.r2 >= R2_MIN) & (maf >= MAF_MIN) & (diff <= FREQ_TOL)
    for c in range(1, 23):
        m = T.CHR == c
        per_chr[c] = {"extracted": int(m.sum()), "kept": int((m & ok).sum())}
    info = {"extracted": int(len(T)), "kept": int(ok.sum()), "drop_r2": int((T.r2 < R2_MIN).sum()),
            "drop_maf": int((maf < MAF_MIN).sum()), "drop_freq": int((diff > FREQ_TOL).sum()),
            "maf_is_a1_freq_share": round(agree, 4), "per_chr": per_chr}
    return T[ok], info


def sumstats(snps):
    cols = ["variant_id", "effect_allele", "other_allele", "beta", "standard_error", "n_cases", "n_controls"]
    parts = []
    for ch in pd.read_csv(SUMSTATS, sep="\t", usecols=cols, chunksize=2_000_000,
                          dtype={"variant_id": str, "effect_allele": str, "other_allele": str}):
        parts.append(ch[ch.variant_id.isin(snps)])
    S = pd.concat(parts, ignore_index=True)
    dup = S.variant_id.duplicated(keep=False)
    S = S[~dup & S.beta.notna() & (S.standard_error > 0)]
    neff = 4 / (1 / S.n_cases + 1 / S.n_controls)
    out = pd.DataFrame({"SNP": S.variant_id, "A1": S.effect_allele.str.upper(), "A2": S.other_allele.str.upper(),
                        "BETA": S.beta, "SE": S.standard_error})
    info = {"rows_in_panel_and_target": int(len(S) + dup.sum()), "dropped_duplicate_rsid_rows": int(dup.sum()),
            "used": int(len(out)), "n_gwas_median_neff": int(round(float(neff.median()))),
            "neff_p10_p90": [int(neff.quantile(.1)), int(neff.quantile(.9))],
            "n_cases_median": int(S.n_cases.median()), "n_controls_median": int(S.n_controls.median())}
    return out, info


def main():
    os.makedirs(PRSCS, mode=0o700, exist_ok=True)
    T, tinfo = targets()
    T.assign(CM=0)[["CHR", "SNP", "CM", "BP", "A1", "A2"]].to_csv(os.path.join(HM3, "target.bim"), sep="\t",
                                                                    header=False, index=False)
    sst, sinfo = sumstats(set(T.SNP))
    sst.to_csv(os.path.join(PRSCS, "sst.txt"), sep="\t", index=False, float_format="%.6g")
    res = {"targets": tinfo, "sumstats": sinfo, "sumstats_file": SUMSTATS,
           "thresholds": {"r2_min": R2_MIN, "maf_min": MAF_MIN, "freq_tol": FREQ_TOL}}
    json.dump(res, open(os.path.join(PRSCS, "prep.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
