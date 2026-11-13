"""geno2_hm3.py -- 基因第二版（两层基因设计文档 的 Tier 2）第 1 步：从 ROSMAP 推断剂量里抽出 PRS-CS 用的 HapMap3 变异，一条染色体一个作业。

    python geno2_hm3.py --chr 1        # -> _runs/geno/hm3/chr1.npz（人级，只留在 RIS）

变异清单 = PRS-CS 的 1000G 欧裔 LD 参考面板（_runs/geno/ref/ldblk_1kg_eur/snpinfo_1kg_hm3，约 112 万个，坐标 GRCh37）。
剂量文件的 ID 一部分是 rsID、其余是 chrN:pos（GRCh37），两种都按面板去对。
方向一律换成"面板 A1 的个数"（PRS-CS 输出的权重就是面板 A1 的效应）：剂量数的是文件里的 A1，对不上就 2 − d；
等位基因对不上（含链翻转后）的不要；A/T、C/G 这种分不清链的一律不要（标准做法）。
每个变异记：ROSMAP 里面板 A1 的频率、推断 r²（没有 INFO 文件，和 geno_build.py 一样用 var(d) / 2p(1−p) 估）、面板的 MAF。
这里只去掉 r² < 0.3 的（留余地给敏感性分析）；正式的质量门槛在 geno2_prep.py 里统一定。
同一个面板变异在文件里出现两次（rsID 行和 chrN:pos 行）时留 r² 高的那个。
"""
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geno_build import COMP, IMP, OUT, ZIP, fam  # noqa: E402

REF = os.path.join(OUT, "ref", "ldblk_1kg_eur")
HM3 = os.path.join(OUT, "hm3")
R2_KEEP = 0.3


def panel(c):
    s = pd.read_csv(os.path.join(REF, "snpinfo_1kg_hm3"), sep=r"\s+")
    return s[s.CHR == c].reset_index(drop=True)


def stream(c, keyfile):
    """一条染色体的 dosage 流式读一遍，只吐出 ID 在 keyfile 里的行（逐行，不整块放进内存）。"""
    member = f"{IMP}_chr{c}.dosage.gz"
    p1 = subprocess.Popen(["unzip", "-p", ZIP, member], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["pigz", "-dc", "-p", "2"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(["gawk", "NR==FNR{k[$1]; next} ($1 in k)", keyfile, "-"], stdin=p2.stdout,
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    p1.stdout.close()
    p2.stdout.close()
    yield from p3.stdout
    rc = (p1.wait(), p2.wait(), p3.wait())
    assert rc == (0, 0, 0), f"chr{c} 流读失败 {rc}"


def orient(fa1, fa2, ra1, ra2):
    """+1：文件 A1 = 面板 A1；−1：文件 A1 = 面板 A2；0：对不上或分不清链。"""
    if len(fa1) != 1 or len(fa2) != 1 or fa1 == fa2.translate(COMP):
        return 0
    for a1, a2 in ((fa1, fa2), (fa1.translate(COMP), fa2.translate(COMP))):
        if (a1, a2) == (ra1, ra2):
            return 1
        if (a1, a2) == (ra2, ra1):
            return -1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chr", type=int, required=True)
    c = ap.parse_args().chr
    os.makedirs(HM3, mode=0o700, exist_ok=True)
    P = panel(c)
    by_rs = {r: i for i, r in enumerate(P.SNP)}
    by_pos = {}
    for i, bp in enumerate(P.BP):
        by_pos.setdefault(f"chr{c}:{bp}", []).append(i)
    ids = fam(IMP + ".fam")
    n = len(ids)
    with tempfile.NamedTemporaryFile("w", suffix=".keys", delete=False, dir=HM3) as f:
        f.write("\n".join(list(by_rs) + list(by_pos)) + "\n")
        keyfile = f.name
    best = {}                                   # 面板下标 -> (r², 剂量)
    seen = mism = ambig_or_bad = lowr2 = 0
    try:
        for ln in stream(c, keyfile):
            f = ln.split(maxsplit=3)
            seen += 1
            cand = [by_rs[f[0]]] if f[0] in by_rs else by_pos.get(f[0], [])
            fa1, fa2 = f[1].upper(), f[2].upper()
            hit = [(i, orient(fa1, fa2, P.A1[i], P.A2[i])) for i in cand]
            hit = [(i, s) for i, s in hit if s]
            if not hit:
                ambig_or_bad += 1
                continue
            i, s = hit[0]
            d = np.array(f[3].split(), np.float32)
            assert len(d) == n, f"{f[0]}：剂量列数 {len(d)} != {n}"
            if s < 0:
                d = 2 - d
            p = float(d.mean()) / 2
            r2 = float(d.var() / (2 * p * (1 - p))) if 0 < p < 1 else 0.0
            if r2 < R2_KEEP:
                lowr2 += 1
                continue
            if i in best:
                mism += 1
                if best[i][0] >= r2:
                    continue
            best[i] = (r2, d)
    finally:
        os.remove(keyfile)
    keep = np.array(sorted(best), dtype=np.int64)
    D = np.stack([best[i][1] for i in keep]).astype(np.float16) if len(keep) else np.zeros((0, n), np.float16)
    r2 = np.array([best[i][0] for i in keep], np.float32)
    af = D.astype(np.float32).mean(1) / 2
    np.savez(os.path.join(HM3, f"chr{c}.npz"), ids=ids, snp=P.SNP.to_numpy()[keep].astype(str),
             bp=P.BP.to_numpy()[keep], a1=P.A1.to_numpy()[keep].astype(str), a2=P.A2.to_numpy()[keep].astype(str),
             maf_ref=P.MAF.to_numpy()[keep].astype(np.float32), af=af, r2=r2, D=D)
    print(f"chr{c}: panel {len(P)}, matched rows {seen}, allele/strand dropped {ambig_or_bad}, r2<{R2_KEEP} {lowr2}, "
          f"duplicates {mism}, kept {len(keep)} ({len(keep) / max(len(P), 1):.1%} of panel), "
          f"r2>=0.8 {int((r2 >= 0.8).sum())}", flush=True)


if __name__ == "__main__":
    main()
