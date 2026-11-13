"""geno2_split.py -- PRS-CS 的大染色体在 30 分钟的 general-short 里跑不完时用：按 LD 块把一条染色体的汇总统计切成两半，各跑一次。

    python geno2_split.py split 1        # -> _runs/geno/prscs/sst_chr1_A.txt、sst_chr1_B.txt（各约一半变异，切在 LD 块边界上）
    #   然后各跑一次 PRS-CS：--sst_file=.../sst_chr1_A.txt --out_dir=.../bellA（B 同理）
    python geno2_split.py merge 1        # bellA + bellB 的 chr1 输出拼成 bell_pst_eff_a1_b0.5_phiauto_chr1.txt

PRS-CS 的模型里各 LD 块只通过全局收缩参数 phi 联系；auto 模式下两半各自估一个 phi（和"每条染色体各估一个"是同一类近似）。
"""
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geno_build import OUT  # noqa: E402

PRSCS = os.path.join(OUT, "prscs")
REF = os.path.join(OUT, "ref", "ldblk_1kg_eur")


def split(c):
    sst = pd.read_csv(os.path.join(PRSCS, "sst.txt"), sep="\t")
    with h5py.File(os.path.join(REF, f"ldblk_1kg_chr{c}.hdf5"), "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))       # blk_1, blk_2, ...（按染色体上的顺序）
        blocks = [[s.decode() if isinstance(s, bytes) else s for s in f[k]["snplist"][:]] for k in keys]
    blk = {s: b for b, ss in enumerate(blocks) for s in ss}
    s = sst[sst.SNP.isin(blk)].copy()
    s["blk"] = s.SNP.map(blk)
    n = s.groupby("blk").size().sort_index().cumsum()
    cut = int(n.index[np.searchsorted(n.to_numpy(), n.iloc[-1] / 2)])
    for half, m in (("A", s.blk <= cut), ("B", s.blk > cut)):
        s.loc[m].drop(columns="blk").to_csv(os.path.join(PRSCS, f"sst_chr{c}_{half}.txt"), sep="\t", index=False)
        print(f"chr{c} {half}: {int(m.sum())} SNPs, blocks {int(s.blk[m].min())}-{int(s.blk[m].max())}")


def merge(c):
    parts = []
    for h in ("A", "B"):
        f = glob.glob(os.path.join(PRSCS, f"bell{h}_pst_eff_a1_b0.5_phi*_chr{c}.txt"))
        assert len(f) == 1, f
        parts.append(pd.read_csv(f[0], sep="\t", header=None))
    out = pd.concat(parts).sort_values(2)
    assert out[1].is_unique
    out.to_csv(os.path.join(PRSCS, f"bell_pst_eff_a1_b0.5_phiauto_chr{c}.txt"), sep="\t", header=False, index=False)
    print(f"chr{c}: merged {len(out)}")


if __name__ == "__main__":
    {"split": split, "merge": merge}[sys.argv[1]](int(sys.argv[2]))
