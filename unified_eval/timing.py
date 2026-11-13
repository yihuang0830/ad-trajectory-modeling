"""timing.py -- 时间预测得准不准：真的发生了的人，模型说的"什么时候"和实际差几年（旧 figure 2 的 panel b，搬进统一评估）。

    python timing.py                    # 全部有 transitions.json 的打包盒 -> packages/<名字>/timing.json
    python timing.py --name delphi-dedup hmm

站点 = 第一次访视（和 rollout_trans、fig4 / fig5 同一个起点），窗口 10 年。
  实际时间：第一次访视时在风险中（rollout_trans.at_risk_flags），10 年内（t0, t0+10*365] 真的发生的人；
            threshold 题 = 第一次测到目标分箱的那次访视，AD / 死亡 = 记录的日期。单位：数据网格的年（365 天）
  预测时间：模型给这个人的累计风险曲线 F(1..10)（F(0) = 0）上，F 第一次到达 F(10) 的一半的整年（实际时间都在整年上，不插值）
            = 10 年内会发生的那些未来里，发生时间的中位数。中位数是让绝对误差最小的单点预测；
            只看 10 年内发生的未来，和"只评估 10 年内真的发生的人"对齐，所以这里只考时间，不考会不会发生（那是 AUC 的事）。
            F(10) = 0（没有一条未来在 10 年内发生）的人没有预测时间，计数记在 n_no_prediction。
  曲线从哪来：能生成未来的模型 = rollout_trans/<名字>/persons.npz（每人 100 条模拟未来，逐年首次越线的条数）；
            其余 = scores_trans/<名字>__<主 seed>.csv 在第一次访视那一行的 risk_<题>_1y..10y（不单调的地方按累计最大值抹平）。
  尺子：所有人都猜同一个数 = 这道题实际时间的中位数（偷看了答案的常数），它的平均绝对误差。
            比它小 = 模型分得出谁早谁晚；早发生的人预测偏晚、晚发生的人预测偏早，常数也会这样，看的是比常数好多少。

产出：packages/<名字>/timing.json（只有聚合数：每个时间段的箱线五数、平均绝对误差、常数的平均绝对误差、人数）。
图在本地按它画（figures.py 的 fig8_timing.png）。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import GRID, RUN_DIR  # noqa: E402
from transitions import STAGE_BINS, TRANS  # noqa: E402
from rollout_trans import at_risk_flags  # noqa: E402

YEARS = 10
Q = list(TRANS)
TIME_BINS = ((0, 2), (2, 5), (5, 10))      # 实际时间（年），左开右闭
MIN_BIN = 10                               # 一个时间段少于这么多人就不报
N_BOOT = 1000
BOOT_SEED = 20261026


def observed_time(d, q):
    """第一次访视之后多少年真的发生（10 年内）；没发生、第一次访视时已不在风险中 = nan。"""
    t0 = int(d["t0"])
    fam = TRANS[q][1]
    if fam == "event":
        a = d["death"] if q == "death" else d["ad"]
    else:
        j = 1 if fam == "mmse" else 2
        a = next((int(v[0]) for v in d["visits"]
                  if int(v[0]) > t0 and v[j] is not None and STAGE_BINS[fam][int(v[j])] in TRANS[q][2]), None)
    if a is None or not (t0 < a <= t0 + YEARS * GRID):
        return np.nan
    return (a - t0) / GRID


def predicted_time(F):
    """(n, 10) 累计风险 F(1..10) -> F 第一次到达 F(10)/2 的整年；F(10) = 0 的行 = nan。
    实际时间全部落在整年上（越线 = 测到它的那次年度访视；AD 吸附到访视；死亡 = 末次访视 + 向上取整的年数），
    F(k) 正好是"实际时间 <= k"的概率，所以中位数是阶梯函数的中位数，不插值。2026-10-31 以前是年内线性插值：
    把离散的时间当成年内均匀，预测系统性地偏早（逐年模拟的模型 bias 偏负，Delphi 的偏正被压小）。"""
    F = np.maximum.accumulate(np.clip(np.asarray(F, float), 0, 1), axis=1)
    F = np.hstack([np.zeros((len(F), 1)), F])                  # F(0) = 0
    half = F[:, -1] / 2
    out = np.full(len(F), np.nan)
    for i in np.where(F[:, -1] > 0)[0]:
        out[i] = int(np.argmax(F[i] >= half[i]))               # 第一个到达一半的整年（k >= 1）
    return out


def curves(name, pids, t0s):
    """每人 6 道题的 10 年累计风险曲线 (n, 6, 10)，和来源。"""
    rp = os.path.join(RUN_DIR, "rollout_trans", name, "persons.npz")
    if os.path.exists(rp):
        z = np.load(rp)
        idx = {int(p): i for i, p in enumerate(z["pid"])}
        assert all(int(p) in idx for p in pids), "rollout 里少了人"
        ii = [idx[int(p)] for p in pids]
        assert (z["t0"][ii] == t0s).all(), "rollout 的起点不是第一次访视"
        return z["cum"][ii] / float(z["n_mc"]), "rollout", rp
    main = json.load(open(os.path.join(RUN_DIR, "packages", name, "transitions.json")))["main_seed"]
    sp = os.path.join(RUN_DIR, "scores_trans", f"{name}__{main}.csv")
    cols = [f"risk_{q}_{k}y" for q in Q for k in range(1, YEARS + 1)]
    sc = pd.read_csv(sp, usecols=lambda c: c in {"pid", "t", *cols})
    have = [q for q in Q if all(f"risk_{q}_{k}y" in sc.columns for k in range(1, YEARS + 1))]
    m = pd.DataFrame({"pid": pids, "t": t0s}).merge(sc, on=["pid", "t"], how="left", validate="one_to_one")
    C = np.full((len(pids), len(Q), YEARS), np.nan)
    for qi, q in enumerate(Q):
        if q in have:
            C[:, qi] = m[[f"risk_{q}_{k}y" for k in range(1, YEARS + 1)]].to_numpy(float)
    return C, "direct", sp


def box(e):
    """matplotlib 的箱线规则（须 = 1.5 倍四分位距以内最远的点），给本地 ax.bxp 用。"""
    q1, med, q3 = np.percentile(e, [25, 50, 75])
    iqr = q3 - q1
    inl = e[(e >= q1 - 1.5 * iqr) & (e <= q3 + 1.5 * iqr)]
    return {"q1": q1, "med": med, "q3": q3, "whislo": float(inl.min()), "whishi": float(inl.max()), "mean": float(e.mean())}


def boot_mae(o, p, c, rng):
    """按人重抽样：模型的平均绝对误差、它减常数的平均绝对误差，各自的 95% CI。"""
    n = len(o)
    ii = rng.integers(0, n, (N_BOOT, n))
    a = np.abs(p[ii] - o[ii]).mean(1)
    d = a - np.abs(c - o[ii]).mean(1)
    return np.percentile(a, [2.5, 97.5]), np.percentile(d, [2.5, 97.5])


def _r(x, k=3):
    if isinstance(x, dict):
        return {a: _r(b, k) for a, b in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(b, k) for b in x]
    if not isinstance(x, (float, np.floating)):
        return x                                               # 字符串、布尔、计数照写
    return None if not np.isfinite(x) else round(float(x), k)


def run(name, traj):
    pids = np.array(sorted(traj, key=int), dtype=np.int64)
    t0s = np.array([int(traj[str(p)]["t0"]) for p in pids], dtype=np.int64)
    C, source, path = curves(name, pids, t0s)
    if np.isnan(C).all():
        return {"name": name, "applicable": False,
                "reason": f"no question has a yearly risk curve out to {YEARS} years"}
    risk = np.array([at_risk_flags(traj[str(p)]) for p in pids], bool)
    # at_risk_flags 把"第一次访视没测这一族"也算在风险中（真值标签会把它们卡成 -1，这里没有标签，自己卡）：
    # 第一次访视没测 MMSE / 认知的人，不知道起点是不是已经越线，threshold 题不算（EVAL_PROTOCOL 第 9 节）
    v0 = [traj[str(p)]["visits"][0] for p in pids]
    for qi, q in enumerate(Q):
        fam = TRANS[q][1]
        if fam != "event":
            risk[:, qi] &= np.array([(v[1] if fam == "mmse" else v[2]) is not None for v in v0])
    rng = np.random.default_rng(BOOT_SEED)
    res = {"name": name, "applicable": True, "source": source, "curves": path, "n_persons": int(len(pids)),
           "years": YEARS, "bins": [list(b) for b in TIME_BINS], "min_bin": MIN_BIN, "questions": {}}
    for qi, q in enumerate(Q):
        if not np.isfinite(C[:, qi]).all(1).any():
            continue
        o = np.array([observed_time(traj[str(p)], q) for p in pids])
        p = predicted_time(np.nan_to_num(C[:, qi]))
        p[~np.isfinite(C[:, qi]).all(1)] = np.nan
        ev = risk[:, qi] & np.isfinite(o)
        ok = ev & np.isfinite(p)
        oo, pp = o[ok], p[ok]
        r = {"n_events": int(ev.sum()), "n_no_prediction": int((ev & ~np.isfinite(p)).sum()), "n": int(ok.sum())}
        if ok.sum() >= MIN_BIN:
            c = float(np.median(oo))
            ci, dci = boot_mae(oo, pp, c, rng)
            r.update(constant=c, mae=float(np.abs(pp - oo).mean()), mae_ci=ci, constant_mae=float(np.abs(c - oo).mean()),
                     mae_minus_constant_ci=dci, bias=float((pp - oo).mean()),
                     spearman=float(pd.Series(pp).corr(pd.Series(oo), method="spearman")), by_bin=[])
            for lo, hi in TIME_BINS:
                k = (oo > lo) & (oo <= hi)
                if k.sum() < MIN_BIN:
                    r["by_bin"].append({"bin": [lo, hi], "n": int(k.sum())})
                    continue
                e = pp[k] - oo[k]
                r["by_bin"].append({"bin": [lo, hi], "n": int(k.sum()), "mae": float(np.abs(e).mean()),
                                    "constant_mae": float(np.abs(c - oo[k]).mean()), "box": box(e)})
        res["questions"][q] = r
    return _r(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", nargs="*", help="打包盒名字；不给 = packages/ 下全部有 transitions.json 的")
    args = ap.parse_args()
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    pk = os.path.join(RUN_DIR, "packages")
    names = args.name or sorted(n for n in os.listdir(pk) if os.path.exists(os.path.join(pk, n, "transitions.json")))
    for n in names:
        res = run(n, traj)
        with open(os.path.join(pk, n, "timing.json"), "w") as f:
            json.dump(res, f, indent=1)
        if not res["applicable"]:
            print(f"{n:45s} n/a  {res['reason']}", flush=True)
            continue
        s = "  ".join(f"{q} {v['mae']:.2f}/{v['constant_mae']:.2f} (n {v['n']})"
                      for q, v in res["questions"].items() if q in ("ad", "mmse_lt24", "death") and "mae" in v)
        print(f"{n:45s} {res['source']:6s}  MAE model/constant  {s}", flush=True)


if __name__ == "__main__":
    main()
