"""input_data_trans.py -- 第三部分（疾病分期转换）变体的训练数据：特征和原版**完全相同**，只换标签。

    python input_data_trans.py      # 读 newmodel_data.npz + truth_trans_{train,val}.csv -> newmodel_data_trans.npz（人级，只留在 RIS）

为什么不重跑 data.py：特征（逐访视输入、手工历史特征、静态背景、标准化参数）一个字都不该变，
变体和原版的差别只能是"训练、回答哪一套题"。这里原样拷贝 X / M / T / S / L / pid，
只把 Y 换成 transitions.TRANS × TRUTH_HORIZONS（逐年 1..10）的 60 列（题目外层、时间窗内层，和 train.py 的读法一致）。

自检：和第一部分同一道题、同一个时间窗的列（MMSE<24、认知<−1、AD、死亡 × 1/2/3/5 年）必须和原版的 Y 逐格相同。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import ENDPOINTS, HORIZONS, RUN_DIR  # noqa: E402
from transitions import SAME_AS_PART1, TRANS, TRUTH_HORIZONS, label_col  # noqa: E402


def main():
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_meta.json")))
    old_cols = meta["labels"]
    assert old_cols == [f"y_{e}_{h}y" for e in ENDPOINTS for h in HORIZONS], "原版标签列的顺序变了"
    new_cols = [label_col(q, H) for q in TRANS for H in TRUTH_HORIZONS]
    # 两分 snapshot：train 里包含 split811 的 test 段（和 data.py 同一个做法）
    T = pd.concat([pd.read_csv(os.path.join(RUN_DIR, f"truth_trans_{sp}.csv")) for sp in ("train", "val")],
                  ignore_index=True)
    assert not T.duplicated(["pid", "t"]).any()
    lab = T.set_index(["pid", "t"])[new_cols]
    out = {k: Z[k] for k in Z.files}
    n_checked = 0
    for s in ("train", "test", "val"):
        M, Tt, pid, Y_old = Z[f"{s}_M"], Z[f"{s}_T"], Z[f"{s}_pid"], Z[f"{s}_Y"]
        n, V = M.shape
        Y = np.full((n, V, len(new_cols)), -1, np.int8)
        ii, vv = np.where(M)
        keys = list(zip(pid[ii].tolist(), Tt[ii, vv].tolist()))
        missing = [k for k in keys if k not in lab.index]
        # 只报个数：Slurm 日志不在 _runs 里，不能把人和年龄写进去
        assert not missing, f"{s}: {len(missing)} 个站点不在第三部分的真值表里"
        Y[ii, vv] = lab.loc[keys].to_numpy(dtype=np.int8)
        # 和原版标签逐格对照
        for q, e in SAME_AS_PART1.items():
            for H in HORIZONS:
                a = Y[ii, vv, new_cols.index(label_col(q, H))]
                b = Y_old[ii, vv, old_cols.index(f"y_{e}_{H}y")]
                bad = int((a != b).sum())
                assert bad == 0, f"{s}: {q} {H}y 和原版 y_{e}_{H}y 有 {bad} 格不一致"
                n_checked += len(a)
        out[f"{s}_Y"] = Y
        print(f"{s}: {n} 人、{len(ii)} 个站点；标签 {len(new_cols)} 列，已知 {int((Y[ii, vv] >= 0).sum())} 格")
    np.savez_compressed(os.path.join(RUN_DIR, "newmodel_data_trans.npz"), **out)
    meta_t = dict(meta, labels=new_cols, questions=list(TRANS), horizons=list(TRUTH_HORIZONS))
    with open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json"), "w") as f:
        json.dump(meta_t, f, indent=1)
    print(f"和原版标签逐格对照：{n_checked} 格，0 处不一致 -> newmodel_data_trans.npz")


if __name__ == "__main__":
    main()
