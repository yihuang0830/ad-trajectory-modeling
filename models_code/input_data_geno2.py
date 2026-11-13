"""input_data_geno2.py -- 基因第二版的数据变体（两层基因设计文档 的两层）：和 newmodel_data_trans.npz **完全相同**，只在静态背景 S 末尾加列。

    python input_data_geno2.py      # -> newmodel_data_trans_geno2.npz   （通路 PRS 排除了 Tier 1 区域，主版本）
                                    #    newmodel_data_trans_geno2nx.npz （敏感性：通路 PRS 不排除，其余完全一样）

加的列（顺序固定，名字两份文件一样，扰动代码共用）：
  apoe_e2x2       ε2/ε2。原有的 apoe_e4x1 / e4x2 / e2carrier / na 已经分开了 ε2/ε3、ε2/ε4、ε3/ε3、ε3/ε4、ε4/ε4，
                  加这一列就是文档的 APOE 六类。全体里 ε2/ε2 少于 APOE_MIN 人时按文档并进 ε2/ε3（不加这一列，meta 里记下）。
  t1_<位点>       Tier 1（geno2_tier1.py 留下的位点）：风险（或同向的替代）等位基因的剂量，减去 train 段有基因的人的均值；
                  这样"0 = 平均水平"，和没基因的人记 0 + geno_na = 1 的约定一致；扰动 0 → 2 份 = 列值 −均值 → 2 − 均值。
  prs_<通路>      Tier 2：immune、lipid、endocytosis、amyloid、tau、rest 的 z 分（geno2_score.py：对 PC1–5 取残差再按 train 标准化）。
  pc1..pc5        祖源主成分，train 段有基因的人标准化（同 input_data_geno.py）。
  geno_na         没有芯片数据 = 1；没基因的人上面这些基因列都记 0。
自检：X / M / T / L / Y / pid 和原文件逐位相同，S 的前 25 列也相同。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import RUN_DIR  # noqa: E402

SRC = "newmodel_data_trans"
PATHWAYS = ["immune", "lipid", "endocytosis", "amyloid", "tau", "rest"]
APOE_MIN = 20
GENO = os.path.join(RUN_DIR, "geno")


def apoe22(pids):
    from input_data import load_raw
    cs = load_raw()[0]
    g = cs.apoe_genotype.reindex(pids)
    return (g == 22).to_numpy()


def main():
    Z = np.load(os.path.join(RUN_DIR, f"{SRC}.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, f"{SRC}_meta.json")))
    assert len(meta["static"]) == 25
    T1 = pd.read_csv(os.path.join(GENO, "tier1_table.csv")).set_index("projid")
    t1cols = [c for c in T1.columns if c.startswith("t1_")]
    G2 = pd.read_csv(os.path.join(GENO, "geno2_table.csv")).set_index("projid")
    PC = pd.read_csv(os.path.join(GENO, "geno_table.csv")).set_index("projid")
    genotyped = set(G2.index) & set(PC.index) & set(T1.index)
    allp = np.concatenate([Z[f"{s}_pid"] for s in ("train", "test", "val")])
    n22 = int(apoe22(allp).sum())
    use22 = n22 >= APOE_MIN
    trp = Z["train_pid"][np.isin(Z["train_pid"], list(genotyped))]
    t1_mu = {c: float(T1.loc[trp, c].mean()) for c in t1cols}
    pc_norm = {f"pc{k}": (float(PC.loc[trp, f"pc{k}"].mean()), float(PC.loc[trp, f"pc{k}"].std())) for k in range(1, 6)}
    names = (["apoe_e2x2"] if use22 else []) + t1cols + [f"prs_{p}" for p in PATHWAYS] + list(pc_norm) + ["geno_na"]
    for dst, suf in (("newmodel_data_trans_geno2", ""), ("newmodel_data_trans_geno2nx", "_noexcl")):
        out = {k: Z[k] for k in Z.files}
        for s in ("train", "test", "val"):
            pid, S = Z[f"{s}_pid"], Z[f"{s}_S"]
            has = np.isin(pid, list(genotyped))
            G = np.zeros((len(pid), len(names)), np.float32)
            j = 0
            if use22:
                G[:, 0] = apoe22(pid)
                j = 1
            for c in t1cols:
                G[has, j] = T1.loc[pid[has], c].to_numpy() - t1_mu[c]
                j += 1
            for p in PATHWAYS:
                G[has, j] = G2.loc[pid[has], f"{p}{suf}_z"].to_numpy()
                j += 1
            for k, (mu, sd) in pc_norm.items():
                G[has, j] = (PC.loc[pid[has], k].to_numpy() - mu) / sd
                j += 1
            G[~has, j] = 1.0
            assert j == len(names) - 1
            out[f"{s}_S"] = np.concatenate([S, G], 1).astype(S.dtype)
            print(dst, s, "persons", len(pid), "genotyped", int(has.sum()), flush=True)
        for k in Z.files:
            if k.endswith("_S"):
                assert np.array_equal(out[k][:, :25], Z[k]), k
            else:
                assert np.array_equal(out[k], Z[k]), k
        np.savez_compressed(os.path.join(RUN_DIR, f"{dst}.npz"), **out)
        m = {**meta, "static": meta["static"] + names,
             "geno2": {"pathway_version": "pathways_v1", "prs_columns": f"geno2_table.csv <通路>{suf}_z",
                       "exclusion": suf == "", "apoe_e2x2": {"n_all": n22, "own_column": use22, "min": APOE_MIN},
                       "tier1_columns": t1cols, "tier1_train_mean": t1_mu, "pc_norm_train_genotyped": pc_norm,
                       "n_train_genotyped": int(len(trp))}}
        json.dump(m, open(os.path.join(RUN_DIR, f"{dst}_meta.json"), "w"), indent=1)
        print("->", f"{dst}.npz", "static", len(m["static"]), "added", names, flush=True)


if __name__ == "__main__":
    main()
