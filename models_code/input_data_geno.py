"""input_data_geno.py -- 加基因的数据变体：和 newmodel_data_trans.npz **完全相同**，只在静态背景 S 的末尾加 7 列。

    python input_data_geno.py      # 读 newmodel_data_trans.npz + _runs/geno/geno_table.csv -> newmodel_data_trans_geno.npz（人级，只留在 RIS）

加的 7 列：prs（Bellenguez 2022 的 83 个变异、stage I 权重，不含 APOE 区域；geno_build.py 的 prs83）、pc1..pc5（祖源主成分）、
geno_na（没有芯片数据 = 1）。prs 和主成分用 train 段有基因的人标准化（均值 0、SD 1）；没有基因的人这 6 列记 0。
加在末尾：现有代码按下标读 S 的地方最大到 24，不受影响；S 是原样喂进网络的（没有另存的缩放），所以这里必须先标准化。
自检：X / M / T / L / Y / pid 和原文件逐位相同，S 的前 25 列也相同。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import RUN_DIR  # noqa: E402

SRC, DST = "newmodel_data_trans", "newmodel_data_trans_geno"
GENO = ["prs", "pc1", "pc2", "pc3", "pc4", "pc5"]
COLS = {"prs": "prs83", **{f"pc{k}": f"pc{k}" for k in range(1, 6)}}


def main():
    Z = np.load(os.path.join(RUN_DIR, f"{SRC}.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, f"{SRC}_meta.json")))
    T = pd.read_csv(os.path.join(RUN_DIR, "geno", "geno_table.csv")).set_index("projid")
    tr = np.isin(Z["train_pid"], T.index)
    ref = T.loc[Z["train_pid"][tr]]
    norm = {g: (float(ref[c].mean()), float(ref[c].std())) for g, c in COLS.items()}
    out = {k: Z[k] for k in Z.files}
    for s in ("train", "test", "val"):
        pid, S = Z[f"{s}_pid"], Z[f"{s}_S"]
        assert S.shape[1] == len(meta["static"]) == 25
        has = np.isin(pid, T.index)
        G = np.zeros((len(pid), len(GENO) + 1), np.float32)
        for j, g in enumerate(GENO):
            mu, sd = norm[g]
            G[has, j] = (T.loc[pid[has], COLS[g]].to_numpy() - mu) / sd
        G[~has, -1] = 1.0
        out[f"{s}_S"] = np.concatenate([S, G], 1).astype(S.dtype)
        print(s, "persons", len(pid), "genotyped", int(has.sum()), flush=True)
    for k in Z.files:                                       # 自检：只有 S 变了，而且只多了末尾 7 列
        if k.endswith("_S"):
            assert np.array_equal(out[k][:, :25], Z[k]), k
        else:
            assert out[k] is Z[k] or np.array_equal(out[k], Z[k]), k
    np.savez_compressed(os.path.join(RUN_DIR, f"{DST}.npz"), **out)
    meta = {**meta, "static": meta["static"] + GENO + ["geno_na"],
            "geno": {"norm_train_genotyped": norm, "n_train_genotyped": int(tr.sum()), "source": "geno_build.py prs83, pc1-5"}}
    json.dump(meta, open(os.path.join(RUN_DIR, f"{DST}_meta.json"), "w"), indent=1)
    print("->", f"{DST}.npz", "static", len(meta["static"]), flush=True)


if __name__ == "__main__":
    main()
