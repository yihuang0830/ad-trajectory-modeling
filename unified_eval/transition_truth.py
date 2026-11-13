"""transition_truth.py -- 第三部分（疾病分期转换）的真值：站点 × 题目 × 时间窗。只从数据推，不碰任何模型。

    python transition_truth.py --split val      # -> $ZOO_RUN_DIR/truth_trans_val.csv（人级，只留在 RIS）
    python transition_truth.py --split train    # 当前分期基线的发生率、直接预测模型的训练标签

规则和 truth.py 完全相同（同一份数据源 rosmap_snapshot、同一张 365 天网格、同一套"只数答案确实知道的"），
只是推广到 transitions.TRANS 的 6 道题和逐年 1..10 年（TRUTH_HORIZONS；评估只用其中的 1/2/3/5/10）：
  threshold 题（目标 = 一组分箱）
    不计入：到站点为止没测过这一族，或已经测到过目标区间（只问"首次"）
    阳性  ：(t, W] 内第一次测到目标区间
    阴性  ：(t, W] 内没测到，且 W 时或之后的**第一次**测量不在目标区间
  event 题（AD、死亡）：和 truth.py 一字不差
  其余一律不计入。W = t + H*365。

另外每个站点带上**当前分期**（到站点为止最近一次测量的分箱：0 最差 … 2 最好，没测过 = -1），
给热图的行、"同一分期内的 AUC"和当前分期基线用。

自检（每次都跑，不是可选项）：和第一部分同一道题的格子（MMSE<24、认知<−1、AD、死亡 × 1/2/3/5 年）
必须和 truth_<split>.csv **逐格相同**。两份代码各写一遍同一条规则，对上了才说明推广没有改掉原来的口径。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import GRID, HORIZONS, RUN_DIR, TRUTH_DATASET, label_names, load_bin, pre_id
from transitions import SAME_AS_PART1, STAGE_BINS, TRANS, TRUTH_HORIZONS, label_col


def first_age(arr, tok):
    m = arr[:, 2] == tok
    return pd.DataFrame(arr[m][:, :2], columns=["pid", "age"]).groupby("pid").age.min().to_dict()


def build(split="val"):
    names = label_names(TRUTH_DATASET)
    pre = pre_id(names)
    s = load_bin(TRUTH_DATASET, split)

    # 访视 = snapshot 里性别 token 出现的年龄（--snapshot 每次访视都重发背景块）—— 和 truth.py 同一条规则
    sex_tok = {pre["Female"]: 0, pre["Male"]: 1}
    is_sex = np.isin(s[:, 2], list(sex_tok))
    vis = pd.DataFrame({"pid": s[is_sex, 0], "age": s[is_sex, 1], "male": [sex_tok[t] for t in s[is_sex, 2]]})
    vis = vis.drop_duplicates(["pid", "age"]).sort_values(["pid", "age"])

    ev_age = {"ad": first_age(s, pre["AD_DX"]), "death": first_age(s, pre["Death"])}

    # 逐访视测量：(pid) -> (年龄, 分箱序号 0 最差..2 最好)，按年龄排序
    meas = {}
    for fam, bins in STAGE_BINS.items():
        ids = [pre[n] for n in bins]
        lvl = {pre[n]: k for k, n in enumerate(bins)}
        m = np.isin(s[:, 2], ids)
        df = pd.DataFrame({"pid": s[m, 0], "age": s[m, 1], "lvl": [lvl[t] for t in s[m, 2]]})
        assert not df.duplicated(["pid", "age"]).any(), f"{fam}: 同一次访视出现两个分箱"
        meas[fam] = {p: (g.age.to_numpy(), g.lvl.to_numpy()) for p, g in df.sort_values("age").groupby("pid")}
    # 每道 threshold 题的目标分箱序号
    target = {}
    for q, (_, fam, toks) in TRANS.items():
        if fam != "event":
            target[q] = (fam, np.array(sorted(STAGE_BINS[fam].index(n) for n in toks)))

    rows = []
    empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    for pid, g in vis.groupby("pid"):
        ages = g.age.to_numpy()
        last_vis = ages.max()
        death = ev_age["death"].get(pid, np.inf)
        ad = ev_age["ad"].get(pid, np.inf)
        assert death > last_vis, f"pid {pid}: 死亡不在最后一次访视之后"
        for i, t in enumerate(ages):
            r = {"pid": int(pid), "t": int(t), "age": t / 365.25, "visit_idx": i, "male": int(g.male.iloc[0]),
                 "years_since_first": (t - ages[0]) / 365.25}
            for fam in STAGE_BINS:
                a_, l_ = meas[fam].get(pid, empty)
                past = a_ <= t
                r[f"{fam}_stage"] = int(l_[past][-1]) if past.any() else -1        # 最近一次测量
                r[f"{fam}_measured_at_t"] = int((a_ == t).any())
                r[f"{fam}_worst_so_far"] = int(l_[past].min()) if past.any() else -1
            for H in TRUTH_HORIZONS:
                W = t + H * GRID
                # 死亡（truth.py 原样）
                if death <= W:
                    y = 1
                elif np.isfinite(death) or last_vis >= W:
                    y = 0
                else:
                    y = -1
                r[label_col("death", H)] = y
                # AD（truth.py 原样）
                if ad <= t:
                    y = -1
                elif ad <= W:
                    y = 1
                elif ((ages >= W) & (ages < ad)).any():
                    y = 0
                else:
                    y = -1
                r[label_col("ad", H)] = y
                # threshold 题
                for q, (fam, tgt) in target.items():
                    a_, l_ = meas[fam].get(pid, empty)
                    hit = np.isin(l_, tgt)
                    past = a_ <= t
                    if not past.any() or hit[past].any():
                        y = -1
                    else:
                        win = (a_ > t) & (a_ <= W)
                        after = a_ >= W
                        if hit[win].any():
                            y = 1
                        elif after.any() and not hit[after][0]:
                            y = 0
                        else:
                            y = -1
                    r[label_col(q, H)] = y
            rows.append(r)
    return pd.DataFrame(rows)


def check_against_part1(tt, split):
    """和第一部分的真值逐格对照。不一致就停：说明推广改掉了原来的口径。"""
    p = os.path.join(RUN_DIR, f"truth_{split}.csv")
    # 不是可选项：没有第一部分的真值就停，不许悄悄跳过对照（先跑 truth.py --split <split>）
    assert os.path.exists(p), f"没有 {p}：先跑 truth.py --split {split}，第三部分的真值必须和第一部分逐格对照"
    t1 = pd.read_csv(p)
    # 两边的列名会撞（AD、死亡两道题的列名在两份真值里完全相同），所以两边都显式加后缀；
    # 只给一边加后缀的话，撞名的那几列会变成"自己和自己比"，永远 0 处不一致（2026-10-14 审查发现，已逐格补核）
    m = tt.merge(t1, on=["pid", "t"], how="inner", suffixes=("_p3", "_p1"), validate="one_to_one")
    assert len(m) == len(t1) == len(tt), f"站点集合不同：第三部分 {len(tt)}，第一部分 {len(t1)}，交集 {len(m)}"
    n = 0
    for q, e in SAME_AS_PART1.items():
        for H in HORIZONS:
            c3, c1 = label_col(q, H), f"y_{e}_{H}y"
            if c3 == c1:
                c3, c1 = c3 + "_p3", c1 + "_p1"
            assert c3 in m.columns and c1 in m.columns and c3 != c1, (c3, c1)
            a, b = m[c3].to_numpy(), m[c1].to_numpy()
            bad = int((a != b).sum())
            assert bad == 0, f"{q} {H}y 和第一部分的 y_{e}_{H}y 有 {bad} 格不一致"
            n += len(a)
    print(f"和第一部分逐格对照：{n} 格，0 处不一致")
    return n


def summarize(tt):
    out = {"n_persons": int(tt.pid.nunique()), "n_landmarks": int(len(tt)),
           "stage_counts": {fam: tt[f"{fam}_stage"].value_counts().sort_index().to_dict() for fam in STAGE_BINS}}
    for q in TRANS:
        for H in TRUTH_HORIZONS:
            y = tt[label_col(q, H)]
            out[f"{q}_{H}y"] = {"pos": int((y == 1).sum()), "neg": int((y == 0).sum()),
                                "excluded": int((y == -1).sum()), "persons_pos": int(tt.pid[y == 1].nunique())}
    out["stage_counts"] = {f: {str(k): int(v) for k, v in d.items()} for f, d in out["stage_counts"].items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    tt = build(args.split)
    n_checked = check_against_part1(tt, args.split)
    tt.to_csv(os.path.join(RUN_DIR, f"truth_trans_{args.split}.csv"), index=False)
    summ = summarize(tt)
    summ["cells_checked_against_part1"] = n_checked
    with open(os.path.join(RUN_DIR, f"truth_trans_{args.split}_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: summ[k] for k in ("n_persons", "n_landmarks", "stage_counts", "cells_checked_against_part1")},
                     indent=1))
    for q in TRANS:
        print(q, "  ".join(f"{H}y +{summ[f'{q}_{H}y']['pos']}/-{summ[f'{q}_{H}y']['neg']}" for H in TRUTH_HORIZONS))
