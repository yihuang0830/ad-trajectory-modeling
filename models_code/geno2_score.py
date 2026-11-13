"""geno2_score.py -- 基因第二版（两层基因设计文档 的 Tier 2）：用 PRS-CS 权重算每个人的通路 PRS。

    python geno2_score.py      # -> _runs/geno/geno2_table.csv（人级，只留在 RIS）+ _runs/geno/geno2_qc.json（聚合）

输入：PRS-CS-auto 的后验权重（_runs/geno/prscs/bell_pst_eff_a1_b0.5_phiauto_chr*.txt，每个变异一个面板 A1 的效应），
geno2_hm3.py 抽好的剂量（面板 A1 的个数），通路定义 geno/pathways_v1.json，GENCODE GRCh37 基因坐标。
分数 = Σ 权重 × 剂量，只用 geno2_prep.py 定的目标变异（PRS-CS 只给这些出权重）。

排除（两层基因设计文档，避免 Tier 1 的大位点主导通路分数）：APOE 扩展区 chr19:44.4–46.5 Mb，以及每个 Tier 1 先导变异 ±500 kb
（不论 Tier 1 最后留没留，见 pathways_v1.json）。排除 = 权重算好之后把这些区域的权重设成 0（PRS-CS 本身在全基因组上跑一次）；
敏感性分析的"不排除"版本用同一套权重，只差这一步（列名后缀 _noexcl）。
通路归属：变异落在通路里任一基因（GENCODE 基因记录）±10 kb 内就算；一个变异可以同时属于几条通路。
"rest" = 不属于任何一条通路的变异；"total" = 全部（排除之后）。

标准化（用户 2026-11-10 选的连续 z 分，不切十分位）：每个分数先在 train 段有基因的人里对 PC1–5 做线性回归，
所有人取残差（文档的"在遗传祖源内标准化"），再用 train 段有基因的人的均值 / SD 化成 z。列名 <分数>_z。
健全性检查只用 train：随访内 AD ~ 分数 z + 年龄 + 性别 + APOE ε4 个数（每个分数单独一个模型），以及 5 条通路 + rest 放在一起的模型。
"""
import glob
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import RUN_DIR  # noqa: E402
from geno_build import OUT, logit  # noqa: E402
from geno2_tier1 import leads  # noqa: E402

PW = json.load(open(os.path.join(HERE, "geno", "pathways_v1.json")))
GS = os.path.join(OUT, "ref", "genesets")
GMT = ["c5.go.bp.v2024.1.Hs.symbols.gmt", "c5.go.mf.v2024.1.Hs.symbols.gmt", "c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt"]
GTF = os.path.join(GS, "gencode.v46lift37.basic.annotation.gtf.gz")
HM3 = os.path.join(OUT, "hm3")
PRSCS = os.path.join(OUT, "prscs")
NAMES = list(PW["pathways"])


def gene_sets():
    out = {}
    for f in GMT:
        for ln in open(os.path.join(GS, f)):
            x = ln.rstrip("\n").split("\t")
            out[x[0]] = set(x[2:])
    return out


def gene_coords():
    """GENCODE 的 gene 记录 -> {基因名: [(chr, start, end), ...]}（常染色体）。"""
    out = {}
    with gzip.open(GTF, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            x = ln.split("\t", 9)
            if x[2] != "gene":
                continue
            c = x[0].removeprefix("chr")
            if not c.isdigit() or not 1 <= int(c) <= 22:
                continue
            g = re.search(r'gene_name "([^"]+)"', x[8]).group(1)
            out.setdefault(g, []).append((int(c), int(x[3]), int(x[4])))
    return out


def pathway_intervals():
    sets, coords = gene_sets(), gene_coords()
    win = PW["gene_window_bp"]
    iv, info = {}, {}
    for k, p in PW["pathways"].items():
        missing_sets = [s for s in p["sets"] if s not in sets]
        assert not missing_sets, f"{k}：MSigDB 里没有 {missing_sets}"
        genes = set().union(*(sets[s] for s in p["sets"]))
        found = [g for g in genes if g in coords]
        iv[k] = [(c, max(s - win, 0), e + win) for g in found for c, s, e in coords[g]]
        info[k] = {"label": p["label"], "n_genes": len(genes), "n_genes_with_coords": len(found),
                   "genes_without_coords": sorted(set(genes) - set(found))[:30]}
    return iv, info


def in_intervals(bp, ivs):
    """bp：一条染色体上排好序的位置；ivs：[(start, end)] -> 布尔掩码。"""
    m = np.zeros(len(bp), bool)
    for s, e in ivs:
        a, b = np.searchsorted(bp, s, "left"), np.searchsorted(bp, e, "right")
        m[a:b] = True
    return m


def exclusion():
    c0, lo, hi = PW["exclusion"]["apoe_region_grch37"]
    w = PW["exclusion"]["tier1_window_bp"]
    ex = {c0: [(lo, hi)]}
    for L in leads():
        ex.setdefault(L["chr"], []).append((L["pos"] - w, L["pos"] + w))
    return ex


def scores():
    iv, pinfo = pathway_intervals()
    ex = exclusion()
    cols = [*NAMES, "rest", "total"]
    out, counts, ids = None, {k: {"excl": 0, "noexcl": 0} for k in cols}, None
    for c in range(1, 23):
        f = glob.glob(os.path.join(PRSCS, f"bell_pst_eff_a1_b0.5_phi*_chr{c}.txt"))
        assert len(f) == 1, f"chr{c}：PRS-CS 输出 {f}"
        W = pd.read_csv(f[0], sep="\t", header=None, names=["CHR", "SNP", "BP", "A1", "A2", "BETA"])
        z = np.load(os.path.join(HM3, f"chr{c}.npz"))
        if ids is None:
            ids = z["ids"]
            out = {f"{k}{s}": np.zeros(len(ids)) for k in cols for s in ("", "_noexcl")}
        assert np.array_equal(ids, z["ids"])
        pos = {s: i for i, s in enumerate(z["snp"])}
        W = W[W.SNP.isin(pos)].sort_values("BP").reset_index(drop=True)
        ii = np.array([pos[s] for s in W.SNP])
        a1 = z["a1"][ii]
        same = W.A1.to_numpy() == a1
        flip = W.A1.to_numpy() == z["a2"][ii]
        assert (same | flip).all(), f"chr{c}：{int((~(same | flip)).sum())} 个变异的等位基因和面板对不上"
        # 换成"面板 A1 的效应"（剂量数的是面板 A1）；A2 的效应取负只差每人一样的常数，标准化后没有影响
        beta = np.where(same, W.BETA.to_numpy(), -W.BETA.to_numpy())
        D = z["D"][ii].astype(np.float32)
        bp = W.BP.to_numpy()
        excl = in_intervals(bp, ex.get(c, []))
        mem = {k: in_intervals(bp, [(s, e) for cc, s, e in iv[k] if cc == c]) for k in NAMES}
        anyp = np.logical_or.reduce([mem[k] for k in NAMES])
        masks = {**mem, "rest": ~anyp, "total": np.ones(len(bp), bool)}
        for k, m in masks.items():
            for suf, mm in (("", m & ~excl), ("_noexcl", m)):
                out[f"{k}{suf}"] += (beta * mm) @ D
                counts[k]["excl" if suf == "" else "noexcl"] += int(mm.sum())
        print(f"chr{c}: weights {len(W)}, excluded {int(excl.sum())}, "
              + ", ".join(f"{k} {int(mem[k].sum())}" for k in NAMES), flush=True)
    return ids, out, counts, pinfo


def main():
    ids, sc, counts, pinfo = scores()
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))
    split = {int(p): s for s in ("train", "test", "val") for p in Z[f"{s}_pid"]}
    T = pd.DataFrame({"projid": ids, **sc})
    T = T[T.projid.isin(split)].reset_index(drop=True)
    PC = pd.read_csv(os.path.join(OUT, "geno_table.csv")).set_index("projid")
    T = T[T.projid.isin(PC.index)].reset_index(drop=True)
    pcs = PC.loc[T.projid, [f"pc{k}" for k in range(1, 6)]].to_numpy()
    tr = (T.projid.map(split) == "train").to_numpy()
    X = np.column_stack([np.ones(len(T)), pcs])
    norm = {}
    for k in sc:
        b = np.linalg.lstsq(X[tr], T[k].to_numpy()[tr], rcond=None)[0]
        r = T[k].to_numpy() - X @ b
        mu, sd = float(r[tr].mean()), float(r[tr].std())
        T[f"{k}_z"] = (r - mu) / sd
        norm[k] = {"pc_coef": [round(float(v), 6) for v in b], "resid_mu": mu, "resid_sd": sd,
                   "r2_on_pcs_train": round(float(1 - r[tr].var() / T[k].to_numpy()[tr].var()), 4)}
    T.to_csv(os.path.join(OUT, "geno2_table.csv"), index=False)
    res = {"pathways": pinfo, "n_variants": counts, "n_people": int(len(T)),
           "n_by_split": T.projid.map(split).value_counts().to_dict(), "standardization": norm,
           "sanity_train": sanity(T, Z), "corr_train_z": corr(T, tr)}
    json.dump(res, open(os.path.join(OUT, "geno2_qc.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_variants", "n_people", "sanity_train", "corr_train_z")}, indent=1))


def corr(T, tr):
    ks = [f"{k}_z" for k in [*NAMES, "rest", "total"]]
    C = np.corrcoef(T.loc[tr, ks].to_numpy().T)
    return {"cols": ks, "matrix": [[round(float(v), 3) for v in row] for row in C]}


def sanity(T, Z):
    """train 段有基因、有 APOE 的人：随访内 AD ~ z + 年龄 + 性别 + ε4 个数。只出聚合数字。"""
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    fi, si = meta["features"].index, meta["static"].index
    X, M, S, pid = Z["train_X"], Z["train_M"], Z["train_S"], Z["train_pid"]
    ever_ad = np.array([X[i, M[i], fi("ad_dx_state")].max() for i in range(len(pid))]) > 0.5
    age0 = X[:, 0, fi("age_z")]
    e4 = S[:, si("apoe_e4x1")] + 2 * S[:, si("apoe_e4x2")]
    t = T.set_index("projid")
    has = np.isin(pid, t.index) & (S[:, si("apoe_na")] < 0.5)
    cov = np.column_stack([age0[has], S[has, si("male")], e4[has]])
    y = ever_ad[has].astype(float)
    out = {"n": int(has.sum()), "n_ad": int(y.sum()), "single": {}}
    fmt = lambda b, se: {"or_per_sd": round(float(np.exp(b)), 3), "ci95": [round(float(np.exp(b - 1.96 * se)), 3),  # noqa: E731
                                                                            round(float(np.exp(b + 1.96 * se)), 3)],
                         "z": round(float(b / se), 2)}
    for k in [*NAMES, "rest", "total"]:
        for suf in ("", "_noexcl"):
            v = t.loc[pid[has], f"{k}{suf}_z"].to_numpy()
            b, se = logit(np.column_stack([v, cov]), y)
            out["single"][f"{k}{suf}"] = fmt(b[0], se[0])
    V = t.loc[pid[has], [f"{k}_z" for k in [*NAMES, "rest"]]].to_numpy()
    b, se = logit(np.column_stack([V, cov]), y)
    out["joint"] = {k: fmt(b[j], se[j]) for j, k in enumerate([*NAMES, "rest"])}
    return out


if __name__ == "__main__":
    main()
