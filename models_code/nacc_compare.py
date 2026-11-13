"""nacc_compare.py -- NACC 预训练有没有用：同一结构的"预训练后微调 / 不微调"减"只在 ROSMAP 上训练"，按人配对 bootstrap（RIS 上跑，读人级打分表）。

    python nacc_compare.py      # -> $ZOO_RUN_DIR/packages/_nacc/{compare.json, data_summary.json, pretrain.json}（聚合，可以拉回本地）

两个数（和 MODELS.md 表 1 同一种定义）：第一部分 4 个终点 2 年 AUC 的平均；第三部分 6 道题 5 年 AUC 的平均。
配对：每次重抽 val 的人（有放回，1000 次），两个模型在同一批人、同一批站点上各算一次，差值的 2.5% / 97.5% 分位数 = 95% CI。
另报去掉 LATC（Rush 拉丁裔核心队列）的人之后的同一组数：NACC 的中心是匿名的，这批人可能也在 NACC 里（nacc_data.py 开头）。
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import bl_data as B
import models as MZ
from common import ENDPOINTS, RUN_DIR

N_BOOT = 1000
SEED = 20261022
OUT = os.path.join(RUN_DIR, "packages", "_nacc")
# Delphi（nacc_delphi.py）：和自己的对照（同一配方只在 ROSMAP 上训练）比；只在 NACC 上训练的 Delphi 和只在 NACC 上训练的别的结构比
DELPHI_PAIRS = [("delphi-dedup-nojitter-nacc", "delphi-dedup-nojitter"), ("delphi-dedup-nojitter-nacc-zs", "delphi-dedup-nojitter"),
                ("delphi-dedup-nojitter-nacc-zs", "transformer-nacc-zs"),
                ("delphi-dedup-nojitter-nacc", "transformer-nacc"), ("delphi-dedup-nojitter", "transformer-x5")]
# 上游原版 Delphi（delphi_upstream.py）：和 zoo 改过的 Delphi（有 / 没有背景抖动）、和 transformer 比
UPSTREAM_PAIRS = [("delphi-upstream", "delphi-dedup-nojitter"), ("delphi-upstream", "delphi-dedup"),
                  ("delphi-upstream", "transformer-x5")]


def _seed_mean(d, name):
    """Delphi 的打分表每个 seed 一份（<name>__s42.csv ...）：风险按 seed 平均 = 和基线的 __s0（5 seed 集成）同一个口径。"""
    ps = sorted(glob.glob(os.path.join(RUN_DIR, d, f"{name}__s[0-9]*.csv")))
    if not ps:
        return None
    T = [pd.read_csv(p).set_index(["pid", "t"]) for p in ps]
    cols = [c for c in T[0].columns if c.startswith("risk_")]
    return sum(t[cols] for t in T) / len(T)


def tables(name):
    p1 = os.path.join(RUN_DIR, "scores", f"{name}__s0.csv")
    p3 = os.path.join(RUN_DIR, "scores_trans", f"{name}__s0.csv")
    if name.startswith("delphi-"):
        t1, t3 = _seed_mean("scores", name), _seed_mean("scores_trans", name)
        return None if t1 is None or t3 is None else (t1, t3)
    if not (os.path.exists(p1) and os.path.exists(p3)):
        return None
    return pd.read_csv(p1).set_index(["pid", "t"]), pd.read_csv(p3).set_index(["pid", "t"])


def cells(truth1, truth3):
    """每道题一组 (站点, 标签)：第一部分 4 个终点 × 2 年，第三部分 6 道题 × 5 年。"""
    out = []
    for e in ENDPOINTS:
        y = truth1[f"y_{e}_2y"]
        k = y >= 0
        out.append(("p1", e, truth1.index[k], y[k].to_numpy(), f"risk_{e}_2y"))
    for q in B.Q:
        y = truth3[f"y_{q}_5y"]
        k = y >= 0
        out.append(("p3", q, truth3.index[k], y[k].to_numpy(), f"risk_{q}_5y"))
    return out


def _auc(y, s, w):
    k = w > 0
    if len(np.unique(y[k])) < 2:
        return np.nan
    return roc_auc_score(y[k], s[k], sample_weight=w[k])


def compare(a, b, C, pids, rng):
    """a 减 b 的配对 bootstrap。pids：参与的人（全部 val，或去掉 LATC）。每个格子先取出两边的风险，重抽只改每个人的权重。"""
    Ta, Tb = tables(a), tables(b)
    if Ta is None or Tb is None:
        return None
    pos = {p: i for i, p in enumerate(pids)}
    cel = []
    for part, _, idx, y, col in C:
        pp = idx.get_level_values(0)
        k = np.array([p in pos for p in pp])
        T = (lambda T_: T_[0] if part == "p1" else T_[1])
        cel.append((part, np.array([pos[p] for p in pp[k]]), y[k], T(Ta).loc[idx[k], col].to_numpy(),
                    T(Tb).loc[idx[k], col].to_numpy()))

    names = [f"{part}_{q}" for part, q, *_ in C]

    def both(cnt):
        """两个部分的平均 AUC，以及每道题（每个格子）各自的 AUC：{键: (a, b)}。"""
        out, per = {"p1": ([], []), "p3": ([], [])}, {}
        for nm, (part, pi, y, sa, sb) in zip(names, cel):
            w = cnt[pi]
            x, z = _auc(y, sa, w), _auc(y, sb, w)
            out[part][0].append(x)
            out[part][1].append(z)
            per[nm] = (x, z)
        res = {p: (float(np.nanmean(v[0])), float(np.nanmean(v[1]))) for p, v in out.items()}
        return res, per
    full, full_per = both(np.ones(len(pids)))
    D, Dper = {"p1": [], "p3": []}, {nm: [] for nm in names}
    for _ in range(N_BOOT):
        r, per = both(np.bincount(rng.integers(len(pids), size=len(pids)), minlength=len(pids)).astype(float))
        for p in D:
            D[p].append(r[p][0] - r[p][1])
        for nm in names:
            Dper[nm].append(per[nm][0] - per[nm][1])
    ci = lambda d: [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]   # noqa: E731
    out = {}
    for p, key in (("p1", "part1_2y_4endpoints"), ("p3", "part3_5y_6questions")):
        d = np.array(D[p])
        ma, mb = full[p]
        out[key] = {"a": ma, "b": mb, "diff": ma - mb, "ci": ci(d), "p_diff_le_0": float((d <= 0).mean())}
    out["per_question"] = {nm: {"a": full_per[nm][0], "b": full_per[nm][1], "diff": full_per[nm][0] - full_per[nm][1],
                                "ci": ci(np.array(Dper[nm]))} for nm in names}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    truth1 = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv")).set_index(["pid", "t"])
    truth3 = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv")).set_index(["pid", "t"])
    C = cells(truth1, truth3)
    Z, _ = B.load()
    S, pid = Z["val_S"], Z["val_pid"]
    latc = pid[S[:, 19] > 0.5]
    all_p = np.unique(truth1.index.get_level_values(0))
    no_latc = np.setdiff1d(all_p, latc)
    pairs = [(a, b) for a, b in MZ.NACC_CONTROL.items()] + DELPHI_PAIRS + UPSTREAM_PAIRS
    res = {"n_val_persons": int(len(all_p)), "n_latc_val_persons": int(len(np.intersect1d(all_p, latc))), "n_boot": N_BOOT,
           "pairs": []}
    for a, b in pairs:
        r = compare(a, b, C, all_p, np.random.default_rng(SEED))
        if r is None:
            print(f"{a} vs {b}: missing score tables", flush=True)
            continue
        r_n = compare(a, b, C, no_latc, np.random.default_rng(SEED))
        res["pairs"].append({"a": a, "b": b, "all": r, "no_latc": r_n})
        x = r["part1_2y_4endpoints"]
        print(f"{a} - {b}: part1 {x['a']:.4f} - {x['b']:.4f} = {x['diff']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]; "
              f"part3 {r['part3_5y_6questions']['diff']:+.4f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "compare.json"), "w"), indent=1)
    # 预训练本身：每个检查点在 NACC test 段上的损失、轮数、用时
    pre = {}
    for pj in sorted(glob.glob(os.path.join(RUN_DIR, "nacc", "ckpt", "*.json"))):
        pre[os.path.basename(pj)[:-5]] = json.load(open(pj))
    json.dump(pre, open(os.path.join(OUT, "pretrain.json"), "w"), indent=1)
    s = json.load(open(os.path.join(RUN_DIR, "nacc", "nacc_data_summary.json")))
    json.dump(s, open(os.path.join(OUT, "data_summary.json"), "w"), indent=1)
    print("->", OUT, flush=True)


if __name__ == "__main__":
    main()
