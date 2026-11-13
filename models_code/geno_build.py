"""geno_build.py -- 基因表：每个有芯片数据的人一行（AD 多基因风险分 + 5 个祖源主成分），人级，只留在 RIS。

    python geno_build.py [--refresh]
      -> _runs/geno/geno_table.csv     projid, prs83, prs50, pc1..pc5（原始尺度；标准化在 input_data_geno.py 里用 train 段做）
         _runs/geno/dosage_83.txt.gz   83 个变异的剂量行（缓存，免得每次重扫 40 GB；--refresh 重扫）
         _runs/geno/geno_qc.json       聚合：每个变异怎么匹配、留没留、为什么；PRS / 主成分的分布；train 段的健全性检查

数据：ROSMAP_genotype.zip（Synapse syn3157325，在共享存储上；不解压，按染色体流式读）。
- PRS：Imputed/ 下 Affymetrix 6.0 的 imputed dosage（GRCh37；每行 ID、A1、A2，然后每人一个 A1 的剂量，
  列顺序 = Imputed 的 .fam，和 Raw 的 .fam 不同）。权重 = geno/bellenguez2022_83_stageI.tsv：
  Bellenguez 2022 的 83 个先导变异（75 个位点；论文定位点前已去掉 APOE 区域），用 stage I 的 beta ——
  stage I 不含 ROSMAP（ROSMAP 在 ADGC 里，ADGC 只在 stage II）；论文表 1/2 的 OR 含 stage II，不能用。
  先按 rsID 找，找不到再按 GRCh37 的 chrN:pos 找（dosage 里很多 ID 是这种）。
  prs83 = 全部留下的；prs50 = 其中 stage I 自己就 P < 5e-8 的（另外 33 个要加上含 ROSMAP 的 stage II 才显著）。
- 没有 INFO 文件：剂量的 r² 用 var(d) / 2p(1−p) 估（MaCH 的做法），< 0.3 的不用。
- 主成分：Raw/ROSMAP_arrayGenotype.bed（原始芯片，SNP-major，坐标像 hg18）。常染色体、去掉长程 LD 区域、每 4 个取 1、
  call rate ≥ 0.98、MAF ≥ 0.05、窗口内 r² < 0.2 剪枝。用全部有基因的人拟合（不用任何结局；
  只用 train 拟合再投影，m >> n 时投影出来的分数会往 0 缩，train 和 val 的分布就不一样了）。
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import RUN_DIR  # noqa: E402

ZIP = os.environ.get("GENO_ZIP", "/path/to/raw/RADC/ROSMAP_genotype.zip")
IMP = "ROSMAP_genotype/Imputed/AMP-AD_ROSMAP_Rush-Broad_AffymetrixGenechip6_Imputed"
RAW = "ROSMAP_genotype/Raw/ROSMAP_arrayGenotype"
OUT = os.path.join(RUN_DIR, "geno")
WEIGHTS = os.path.join(HERE, "geno", "bellenguez2022_83_stageI.tsv")
R2_MIN = 0.3            # 剂量 r² 下限
RAF_TOL = 0.2           # ROSMAP 的风险等位基因频率和 GWAS 差这么多就当没对齐
AMBIG_MAF = 0.4         # A/T、C/G 位点：MAF 超过这个就分不清链，不用
N_PC = 5
# 长程 LD 区域（hg18 附近的坐标，边界放宽）：MHC、8p23 倒位、17q21 倒位、LCT、chr5 / chr11 着丝粒附近
LONG_LD = [(6, 24e6, 36e6), (8, 6e6, 14e6), (17, 39e6, 46e6), (2, 128e6, 138e6), (5, 44e6, 52e6), (11, 45e6, 58e6)]
COMP = str.maketrans("ACGT", "TGCA")


def fam(member):
    """IID（第 2 列，ROS/MAP + 8 位 projid）-> projid，按文件里的顺序。"""
    with zipfile.ZipFile(ZIP) as z:
        f = pd.read_csv(z.open(member), sep=r"\s+", header=None)
    return f[1].str[3:].astype(int).to_numpy()


def grab(c, keys):
    """一条染色体的 dosage 整个流过一遍，只留 ID 在 keys 里的行。"""
    member = f"{IMP}_chr{c}.dosage.gz"
    awk = 'BEGIN{n=split(K,a," "); for(i=1;i<=n;i++) k[a[i]]=1} ($1 in k)'
    p1 = subprocess.Popen(["unzip", "-p", ZIP, member], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["pigz", "-dc", "-p", "2"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["gawk", "-v", "K=" + " ".join(keys), awk], stdin=p2.stdout, stdout=subprocess.PIPE, text=True)
    p1.stdout.close()
    p2.stdout.close()
    out = p3.communicate()[0]
    rc = (p1.wait(), p2.wait(), p3.returncode)
    assert rc == (0, 0, 0), f"chr{c} 流读失败 {rc}"
    return out.splitlines()


def extract(W, refresh):
    cache = os.path.join(OUT, "dosage_83.txt.gz")
    if os.path.exists(cache) and not refresh:
        with gzip.open(cache, "rt") as f:
            return f.read().splitlines()
    keys = {c: sorted(set(g.rsid) | {f"chr{c}:{p}" for p in g.pos_grch37}) for c, g in W.groupby("chr")}
    with ThreadPoolExecutor(8) as ex:
        lines = [ln for part in ex.map(lambda c: grab(c, keys[c]), sorted(keys)) for ln in part]
    with gzip.open(cache + ".tmp", "wt") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(cache + ".tmp", cache)
    return lines


def orient(a1, a2, risk, other, f1, raf):
    """剂量数的是 A1。返回 (+1：A1 是风险等位基因 / −1：A2 是 / 0：对不上, 说明)。f1 = ROSMAP 里 A1 的频率。"""
    if {a1, a2} <= {"I", "D"}:                          # 插入缺失可能写成 I / D：长的那个是插入
        risk, other = ("I", "D") if len(risk) > len(other) else ("D", "I")
    ambig = a1 == a2.translate(COMP) and len(a1) == 1
    cands = []
    for s, (r, o) in ((+1, (a1, a2)), (-1, (a2, a1))):
        if (r, o) == (risk, other) or (len(r) == 1 and (r.translate(COMP), o.translate(COMP)) == (risk, other)):
            cands.append(s)
    if not cands:
        return 0, "alleles"
    if ambig:
        if min(raf, 1 - raf) > AMBIG_MAF:
            return 0, "ambiguous_maf"
        s = +1 if abs(f1 - raf) <= abs((1 - f1) - raf) else -1      # A/T、C/G：按频率定链
    else:
        s = cands[0]
    fr = f1 if s == +1 else 1 - f1
    if abs(fr - raf) > RAF_TOL:
        return 0, "freq"
    return s, "ambiguous_by_freq" if ambig else "ok"


def prs(W, lines, ids):
    rows = {}
    for ln in lines:
        f = ln.split()
        rows.setdefault(f[0], []).append(f)
    n = len(ids)
    dose = {}
    qc = []
    for _, w in W.iterrows():
        rec = {"rsid": w.rsid, "chr": int(w.chr), "locus": w.nearest_gene, "p_stageI": float(w.p_stageI)}
        cand = [("rsid", f) for f in rows.get(w.rsid, [])] + [("chrpos", f) for f in rows.get(f"chr{w.chr}:{w.pos_grch37}", [])]
        best = None
        for how, f in cand:
            assert len(f) == 3 + n, f"{w.rsid}：剂量列数 {len(f) - 3} != {n}"
            d = np.asarray(f[3:], np.float64)
            p = d.mean() / 2
            s, why = orient(f[1].upper(), f[2].upper(), w.risk_allele, w.other_allele, p, w.raf_stageI)
            if s:
                best = (how, f, d, p, s, why)
                break
            best = best or (how, f, d, p, 0, why)
        if best is None:
            qc.append({**rec, "used": False, "why": "not_found"})
            continue
        how, f, d, p, s, why = best
        r2 = float(d.var() / (2 * p * (1 - p))) if 0 < p < 1 else 0.0
        rec.update({"matched_by": how, "a1": f[1], "a2": f[2], "r2": round(r2, 3),
                    "raf_gwas": float(w.raf_stageI), "raf_rosmap": None if not s else round(p if s == 1 else 1 - p, 4)})
        if not s:
            qc.append({**rec, "used": False, "why": why})
            continue
        if r2 < R2_MIN:
            qc.append({**rec, "used": False, "why": "low_r2"})
            continue
        dose[w.rsid] = d if s == 1 else 2 - d
        qc.append({**rec, "used": True, "why": why})
    used = W[W.rsid.isin(dose)]
    D = np.stack([dose[r] for r in used.rsid], 1)                        # 人 × 变异，风险等位基因的剂量
    b = used.beta_stageI.to_numpy()
    sig = (used.p_stageI < 5e-8).to_numpy()
    return {"prs83": D @ b, "prs50": D[:, sig] @ b[sig]}, qc, {"n_used_83": int(len(used)), "n_used_50": int(sig.sum())}


def read_bed():
    with zipfile.ZipFile(ZIP) as z:
        bim = pd.read_csv(z.open(RAW + ".bim"), sep=r"\s+", header=None, names=["chr", "id", "cm", "pos", "a1", "a2"])
        bed = z.read(RAW + ".bed")
    assert bed[:3] == b"\x6c\x1b\x01", "不是 SNP-major 的 .bed"
    return bim, np.frombuffer(bed, np.uint8, offset=3)


def decode(B, n, idx):
    """.bed 的若干行 -> A2 的个数（0/1/2，缺失 = 3），uint8，变异 × 人。"""
    nb = (n + 3) // 4
    raw = B.reshape(-1, nb)[idx]
    lut = np.array([0, 3, 1, 2], np.uint8)          # 00 纯合 A1、01 缺失、10 杂合、11 纯合 A2
    g = np.stack([(raw >> (2 * k)) & 3 for k in range(4)], 2).reshape(len(idx), nb * 4)[:, :n]
    return lut[g]


def prune(G, pos_chr, r2_max=0.2, win=100, step=50):
    """窗口内贪心剪枝（同 plink --indep-pairwise 100 50 0.2）：一对 r² 超了就丢后面那个。G 是标准化后的 变异 × 人。"""
    keep = np.ones(len(G), bool)
    for c in np.unique(pos_chr):
        ii = np.where(pos_chr == c)[0]
        for a in range(0, len(ii), step):
            w = ii[a:a + win]
            w = w[keep[w]]
            if len(w) < 2:
                continue
            R = (G[w] @ G[w].T) / G.shape[1]
            R2 = R * R
            for i in range(len(w)):
                if keep[w[i]]:
                    drop = np.where(R2[i, i + 1:] > r2_max)[0] + i + 1
                    keep[w[drop]] = False
    return keep


def pcs(zoo_pids):
    bim, B = read_bed()
    ids = fam(RAW + ".fam")
    n = len(ids)
    auto = bim.chr.between(1, 22).to_numpy()
    for c, lo, hi in LONG_LD:
        auto &= ~((bim.chr == c) & bim.pos.between(lo, hi)).to_numpy()
    idx = np.where(auto)[0][::4]
    G = decode(B, n, idx)
    rows = np.isin(ids, list(zoo_pids))                 # 只要 zoo 里的人（芯片里有几个人不在 zoo 里）
    G = G[:, rows]
    miss = G == 3
    call = 1 - miss.mean(1)
    p = np.where(miss, 0, G).sum(1) / np.maximum((~miss).sum(1), 1) / 2
    ok = (call >= 0.98) & (np.minimum(p, 1 - p) >= 0.05)
    G, miss, p, chrs = G[ok], miss[ok], p[ok], bim.chr.to_numpy()[idx][ok]
    X = (G.astype(np.float32) - 2 * p[:, None].astype(np.float32)) / np.sqrt(2 * p * (1 - p))[:, None].astype(np.float32)
    X[miss] = 0.0
    keep = prune(X, chrs)
    X = X[keep]
    K = (X.T @ X) / X.shape[0]                            # 人 × 人
    ev, U = np.linalg.eigh(K)
    o = np.argsort(ev)[::-1]
    ev, U = ev[o], U[:, o]
    score = U[:, :N_PC] * np.sqrt(ev[:N_PC])
    info = {"n_people": int(rows.sum()), "snps_thinned": int(len(idx)), "snps_after_qc": int(ok.sum()),
            "snps_after_prune": int(keep.sum()), "var_explained_top5": [round(float(v), 5) for v in ev[:N_PC] / ev.sum()]}
    return ids[rows], score, info


def logit(X, y, iters=50):
    """无惩罚的逻辑回归（IRLS），返回系数和标准误。"""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b[1:], np.sqrt(np.diag(np.linalg.inv(H)))[1:]


def sanity(T):
    """train 段有基因的人：终身 AD（随访内）~ PRS（每 SD）+ 年龄 + 性别 + APOE ε4 个数。只出聚合数字。"""
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    fi, si = meta["features"].index, meta["static"].index
    X, M, S, pid = Z["train_X"], Z["train_M"], Z["train_S"], Z["train_pid"]
    ever_ad = np.array([X[i, M[i], fi("ad_dx_state")].max() for i in range(len(pid))]) > 0.5
    age0 = X[:, 0, fi("age_z")]
    e4 = S[:, si("apoe_e4x1")] + 2 * S[:, si("apoe_e4x2")]
    t = T.set_index("projid")
    has = np.isin(pid, t.index) & (S[:, si("apoe_na")] < 0.5)
    out = {"n": int(has.sum()), "n_ad": int(ever_ad[has].sum())}
    for k in ("prs83", "prs50"):
        v = t.loc[pid[has], k].to_numpy()
        z = (v - v.mean()) / v.std()
        b, se = logit(np.column_stack([z, age0[has], S[has, si("male")], e4[has]]), ever_ad[has].astype(float))
        out[k] = {"or_per_sd": round(float(np.exp(b[0])), 3), "ci95": [round(float(np.exp(b[0] - 1.96 * se[0])), 3),
                  round(float(np.exp(b[0] + 1.96 * se[0])), 3)], "z": round(float(b[0] / se[0]), 2),
                  "apoe_e4_or": round(float(np.exp(b[3])), 3)}
    pc = t.loc[pid[has], [f"pc{k}" for k in range(1, N_PC + 1)]].to_numpy()
    out["corr_prs83_pcs"] = [round(float(np.corrcoef(t.loc[pid[has], "prs83"], pc[:, k])[0, 1]), 3) for k in range(N_PC)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="重扫 dosage，不用缓存")
    a = ap.parse_args()
    os.makedirs(OUT, mode=0o700, exist_ok=True)
    W = pd.read_csv(WEIGHTS, sep="\t")
    assert len(W) == 83 and W.rsid.is_unique
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))
    split = {int(p): s for s in ("train", "test", "val") for p in Z[f"{s}_pid"]}
    assert sum(meta["n"].values()) == len(split)

    ids = fam(IMP + ".fam")
    P, qc, nu = prs(W, extract(W, a.refresh), ids)
    T = pd.DataFrame({"projid": ids, **P})
    T = T[T.projid.isin(split)]
    pid_pc, score, pcinfo = pcs(set(split))
    T = T.merge(pd.DataFrame({"projid": pid_pc, **{f"pc{k + 1}": score[:, k] for k in range(N_PC)}}), on="projid", how="inner")
    assert T.projid.is_unique
    T.to_csv(os.path.join(OUT, "geno_table.csv"), index=False)

    sp = T.projid.map(split)
    res = {"n_people": int(len(T)), "n_by_split": sp.value_counts().to_dict(), **nu,
           "variants": qc, "drop_reasons": pd.Series([q["why"] for q in qc if not q["used"]]).value_counts().to_dict(),
           "matched_by": pd.Series([q.get("matched_by") for q in qc if q["used"]]).value_counts().to_dict(),
           "prs_by_split": {s: {k: [round(float(T.loc[sp == s, k].mean()), 4), round(float(T.loc[sp == s, k].std()), 4)]
                                for k in ("prs83", "prs50")} for s in ("train", "test", "val")},
           "pcs": pcinfo, "sanity_train": sanity(T)}
    json.dump(res, open(os.path.join(OUT, "geno_qc.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "variants"}, indent=1))


if __name__ == "__main__":
    main()
