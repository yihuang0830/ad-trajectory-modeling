"""truth.py -- 站点 × 终点 × 时间窗的真值表。只从数据推，不碰任何模型。

    python truth.py --split val        # -> $ZOO_RUN_DIR/truth_val.csv + truth_val_summary.json

真值取自 rosmap_snapshot：它每次访视都发射完整状态（背景块、每个测到的量表分箱），所以
"有哪些访视""每次 MMSE / cogn_global 落在哪个箱"都能直接读出来。AD 诊断和死亡的年龄在 6 份
分词里逐人相同（已核对，下面再断言一次），所以这张表对所有模型都是同一份答案。

每个 (人, 站点 t, 终点 e, 窗口 H)：y = 1 阳性 / 0 阴性 / -1 不计入。W = t + H*365。
  event（AD、死亡）
    不计入：站点时已发生（AD 已诊断）
    阳性  ：t < 事件 <= W
    阴性  ：AD —— 窗口内没诊断，且在 [W, AD 诊断) 之间有一次访视（W 时确实被观察到、还没诊断）
            死亡 —— 死在 W 之后，或活着且 W 时或之后还有访视
  threshold（MMSE<24、cogn_global<-1）
    不计入：到站点为止没测过，或已经测到过目标箱（只问"第一次"）
    阳性  ：(t, W] 内第一次测到目标箱
    阴性  ：(t, W] 内没测到目标箱，且 W 时或之后的**第一次**测量不在目标箱里
  其余（窗口没走完就失访 / 死亡；或 W 之后的第一次观测就是事件本身 —— 越过 W 的时刻不知道）
  一律不计入 -> 不把"没来得及观察"当成"没发生"。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import (ENDPOINTS, FAMILY, GRID, HORIZONS, RUN_DIR, TRUTH_DATASET, label_names,
                    load_bin, pre_id)


def first_age(arr, tok):
    m = arr[:, 2] == tok
    return pd.DataFrame(arr[m][:, :2], columns=["pid", "age"]).groupby("pid").age.min().to_dict()


def build(split="val"):
    names = label_names(TRUTH_DATASET)
    pre = pre_id(names)
    s = load_bin(TRUTH_DATASET, split)

    # 访视 = snapshot 里性别 token 出现的年龄（--snapshot 每次访视都重发背景块）
    sex_tok = {pre["Female"]: 0, pre["Male"]: 1}
    is_sex = np.isin(s[:, 2], list(sex_tok))
    vis = pd.DataFrame({"pid": s[is_sex, 0], "age": s[is_sex, 1], "male": [sex_tok[t] for t in s[is_sex, 2]]})
    vis = vis.drop_duplicates(["pid", "age"]).sort_values(["pid", "age"])

    # 事件年龄，并和去重版逐人对一遍（6 份分词同一个 RNG 抽取序列，必须完全相同）
    ev_age = {e: first_age(s, pre[tok]) for e, (_, tok, kind) in ENDPOINTS.items() if kind == "event"}
    d = load_bin("rosmap", split)
    for e, (_, tok, kind) in ENDPOINTS.items():
        if kind == "event":
            assert ev_age[e] == first_age(d, pre[tok]), f"{e} 的年龄在 snapshot 和 dedup 之间不一致"

    # 逐访视测量：(pid, age) -> 是否落在目标箱
    meas = {}
    for e, fam in FAMILY.items():
        ids = [pre[n] for n in fam]
        m = np.isin(s[:, 2], ids)
        df = pd.DataFrame({"pid": s[m, 0], "age": s[m, 1], "hit": s[m, 2] == pre[ENDPOINTS[e][1]]})
        assert not df.duplicated(["pid", "age"]).any(), f"{e}: 同一次访视出现两个分箱"
        meas[e] = {p: (g.age.to_numpy(), g.hit.to_numpy()) for p, g in df.sort_values("age").groupby("pid")}

    rows = []
    for pid, g in vis.groupby("pid"):
        ages = g.age.to_numpy()
        last_vis = ages.max()
        death = ev_age["death"].get(pid, np.inf)
        assert death > last_vis, f"pid {pid}: 死亡不在最后一次访视之后"
        for i, t in enumerate(ages):
            r = {"pid": int(pid), "t": int(t), "age": t / 365.25, "visit_idx": i, "male": int(g.male.iloc[0])}
            for H in HORIZONS:
                W = t + H * GRID
                # 死亡
                if death <= W:
                    y = 1
                elif np.isfinite(death) or last_vis >= W:
                    y = 0
                else:
                    y = -1
                r[f"y_death_{H}y"] = y
                # AD
                ad = ev_age["ad"].get(pid, np.inf)
                if ad <= t:
                    y = -1
                elif ad <= W:
                    y = 1
                elif ((ages >= W) & (ages < ad)).any():      # W 时确实还在、且还没诊断
                    y = 0
                else:
                    y = -1
                r[f"y_ad_{H}y"] = y
                # 阈值类
                for e in FAMILY:
                    a_, h_ = meas[e].get(pid, (np.array([]), np.array([], bool)))
                    past = a_ <= t
                    if not past.any() or h_[past].any():
                        y = -1
                    else:
                        win = (a_ > t) & (a_ <= W)
                        after = a_ >= W
                        if h_[win].any():
                            y = 1
                        elif after.any() and not h_[after][0]:   # W 之后的第一次测量仍不在目标箱
                            y = 0
                        else:
                            y = -1
                    r[f"y_{e}_{H}y"] = y
            rows.append(r)
    truth = pd.DataFrame(rows)
    return truth


def summarize(truth):
    out = {"n_persons": int(truth.pid.nunique()), "n_landmarks": int(len(truth))}
    for e in ENDPOINTS:
        for H in HORIZONS:
            y = truth[f"y_{e}_{H}y"]
            out[f"{e}_{H}y"] = {"pos": int((y == 1).sum()), "neg": int((y == 0).sum()),
                                "excluded": int((y == -1).sum()),
                                "persons_pos": int(truth.pid[y == 1].nunique())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    truth = build(args.split)
    truth.to_csv(os.path.join(RUN_DIR, f"truth_{args.split}.csv"), index=False)
    summ = summarize(truth)
    with open(os.path.join(RUN_DIR, f"truth_{args.split}_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
