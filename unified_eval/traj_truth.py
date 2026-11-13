"""traj_truth.py -- fig3 / fig4 的"真实一侧"：观测轨迹 + 扰动队列。只从数据推，不碰任何模型。

    python traj_truth.py               # -> $ZOO_RUN_DIR/traj_truth_val.json, perturb_cohort_val.csv（人级，只留在 RIS）
    python traj_truth.py --split test  # -> traj_truth_test.json（只有观测轨迹）：split811 的 test 段的人，
                                       #    给模拟器调参用（val 只在最后报一次数）

数据源和 truth.py 一样是 rosmap_snapshot（每次访视发射完整状态）。

traj_truth_val.json，每人一条：
  t0        第一次访视的年龄（天）= fig3 模拟的出发点
  death/ad  死亡 / AD 诊断的年龄（天），没有为 null
  last      最后一次访视的年龄（天）
  visits    [[年龄, MMSE 箱, 认知箱], ...]，箱 = 0 低 / 1 中 / 2 高（没测为 null）

perturb_cohort_val.csv，每个至少有 4 次访视的人一行（站点 = 第 4 次访视）：
  pid, t0, tL, t_first（第一次访视）, 资格列 elig_<扰动>, 以及 4 个终点在站点上的 2 年真值 y_<终点>_2y
  资格规则见 EVAL_PROTOCOL.md 第 7 节；全部从 snapshot 的 token 推出，所有模型共用。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import RUN_DIR, TRUTH_DATASET, label_names, load_bin, pre_id

LANDMARK_VISIT = 3          # 第 4 次访视（0 起）
BIN = {"MMSE": ("MMSE_impaired", "MMSE_borderline", "MMSE_normal"),
       "COGN": ("COGN_low", "COGN_mid", "COGN_high")}


def build(split="val"):
    names = label_names(TRUTH_DATASET)
    pre = pre_id(names)
    if split == "val":
        s = load_bin(TRUTH_DATASET, "val")
        truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    else:
        # snapshot 是两分：split811 的 test 段在 snapshot 的 train 里，按 split811 的人名单取出来
        keep = np.unique(load_bin("rosmap_nodedup_3w", split)[:, 0])
        s = load_bin(TRUTH_DATASET, "train")
        s = s[np.isin(s[:, 0], keep)]
        assert len(np.unique(s[:, 0])) == len(keep), f"{split} 段有人不在 snapshot 的 train 里"
        truth = None
    sex = {pre["Female"], pre["Male"]}
    bin_of = {}
    for fam, toks in BIN.items():
        for k, n in enumerate(toks):
            bin_of[pre[n]] = (fam, k)
    persons, rows = {}, []
    for pid, g in pd.DataFrame(s, columns=["pid", "age", "tok"]).groupby("pid", sort=True):
        g = g.sort_values("age", kind="stable")
        ages = sorted(g.age[g.tok.isin(sex)].unique().tolist())          # 访视 = 性别 token 的年龄
        death = g.age[g.tok == pre["Death"]].min() if (g.tok == pre["Death"]).any() else None
        ad = g.age[g.tok == pre["AD_DX"]].min() if (g.tok == pre["AD_DX"]).any() else None
        vis = []
        for a in ages:
            gv = g[g.age == a]
            m = [bin_of[t][1] for t in gv.tok if t in bin_of and bin_of[t][0] == "MMSE"]
            c = [bin_of[t][1] for t in gv.tok if t in bin_of and bin_of[t][0] == "COGN"]
            vis.append([int(a), m[0] if m else None, c[0] if c else None])
        persons[int(pid)] = {"t0": int(ages[0]), "death": None if death is None else int(death),
                             "ad": None if ad is None else int(ad), "last": int(ages[-1]), "visits": vis}
        if truth is None or len(ages) <= LANDMARK_VISIT:        # 扰动队列只给 val
            continue
        tL = ages[LANDMARK_VISIT]
        h = g[g.age <= tL]
        toks = set(h.tok.tolist())
        mm = [v[1] for v in vis if v[0] <= tL and v[1] is not None]
        cg = [v[2] for v in vis if v[0] <= tL and v[2] is not None]
        mm_at = vis[LANDMARK_VISIT][1]
        cg_at = vis[LANDMARK_VISIT][2]
        r = {"pid": int(pid), "t0": int(ages[0]), "tL": int(tL), "t_first": int(ages[0]),
             "elig_apoe": pre["APOE_e4_0"] in toks,
             "elig_mmse": mm_at == 2 and all(x == 2 for x in mm),          # 到此为止都 >=27，且站点测了
             "elig_cogn": cg_at == 2 and all(x == 2 for x in cg),          # 到此为止都 >=0，且站点测了
             "elig_stroke": pre["STROKE"] not in toks,
             "elig_age": True,
             "elig_sex": pre["Female"] in toks,
             "elig_educ": bool({pre["EDU_16to19"], pre["EDU_ge20"]} & toks)}
        # 纯粹的"处在风险中"（不要求结局已知）：算扰动效应只需要这个
        r["risk_ad"] = ad is None or ad > tL
        r["risk_death"] = True
        r["risk_mmse"] = len(mm) > 0 and min(mm) >= 1           # 到此为止从没 <24
        r["risk_cogn"] = len(cg) > 0 and min(cg) >= 1           # 到此为止从没 <-1
        tr = truth[(truth.pid == pid) & (truth.t == tL)]
        assert len(tr) == 1, f"pid {pid} 的第 4 次访视不在真值表里"
        for e in ("ad", "death", "mmse", "cogn"):
            r[f"y_{e}_2y"] = int(tr[f"y_{e}_2y"].iloc[0])
        rows.append(r)
    return persons, pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"))
    split = ap.parse_args().split
    persons, cohort = build(split)
    with open(os.path.join(RUN_DIR, f"traj_truth_{split}.json"), "w") as f:
        json.dump(persons, f)
    if split != "val":
        print(f"persons {len(persons)} -> traj_truth_{split}.json")
        raise SystemExit
    cohort.to_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"), index=False)
    el = [c for c in cohort.columns if c.startswith("elig_")]
    print(f"persons {len(persons)}; perturbation cohort {len(cohort)} (>= {LANDMARK_VISIT + 1} visits)")
    print({c: int(cohort[c].sum()) for c in el})
    print({c: int(cohort[c].sum()) for c in cohort.columns if c.startswith("risk_")})
    print({e: {"pos": int((cohort[f'y_{e}_2y'] == 1).sum()), "neg": int((cohort[f'y_{e}_2y'] == 0).sum())}
           for e in ("ad", "death", "mmse", "cogn")})
