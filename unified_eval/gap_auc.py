"""gap_auc.py -- 只用 N 年以前的数据，AUC 掉多少（仿 Delphi 论文的 time-gap 实验）。不重新训练、不重新打分。

    python gap_auc.py      # -> packages/GAP_AUC.json（聚合数）

题目不变：每个验证集站点 t、每个终点照旧是"t 之后 2 年内首次发生"（同一份真值 truth_val.csv）。
gap = N 年：分数改用这个人在 t − N 年及以前**最后一次访视**上的那次预测（那次预测本来就只用到那次访视为止的数据），
所以模型看到的最新信息离窗口起点至少 N 年、离结局 N 到 N+2 年。
- 每个 N 只在"有 N 年前访视"的站点上算，并在**同一批站点**上和 gap 0 配对比较（按人 bootstrap）——
  N 越大，留下的越是随访很久的人，不配对的话人群变了，掉多少说不清。
- 同时报年龄分层 AUC（同一 5 岁年龄段内比较，年龄按站点 t 算）：只用旧数据时模型仍然知道年龄，年龄本身就能排出
  一大部分风险；分层之后剩下的才是"旧病史本身"的信号。
- 只看人口学的 Delphi 一起算，作为地板。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ENDPOINTS, GRID, RUN_DIR  # noqa: E402

EP = list(ENDPOINTS)
GAPS = [0, 1, 2, 3, 5, 7, 10]
MODELS = [("delphi-dedup", "delphi-dedup__s42.csv"),
          ("delphi-nodedup", "delphi-nodedup__s42.csv"),
          ("delphi-nodedup-pos-split811", "delphi-nodedup-pos-split811__s42.csv"),
          ("delphi-fullvisit", "delphi-fullvisit__s42.csv"),
          ("delphi-nodedup-staticonly-split811 (demographics only)", "delphi-nodedup-staticonly-split811__s42.csv")]
B = 300


def strat_auc(y, s, band):
    num = den = 0.0
    for b in np.unique(band):
        m = band == b
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos and nneg:
            num += roc_auc_score(y[m], s[m]) * npos * nneg
            den += npos * nneg
    return num / den if den else np.nan


def main():
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv")).sort_values(["pid", "t"]).reset_index(drop=True)
    band = np.clip(((truth.age - 70) // 5).astype(int), 0, 6).to_numpy()
    visits = truth.groupby("pid").t.apply(lambda s: np.sort(s.to_numpy())).to_dict()
    # 每个站点、每个 gap：t − N 年及以前最后一次访视的年龄（没有 -> NaN）
    src = {}
    for g in GAPS:
        u = []
        for p, t in zip(truth.pid, truth.t):
            a = visits[p][visits[p] <= t - g * GRID]
            u.append(a[-1] if len(a) else np.nan)
        src[g] = np.array(u)
    pid = truth.pid.to_numpy()
    out = {"gaps_years": GAPS, "n_landmarks_total": int(len(truth)), "models": {}}
    rng = np.random.default_rng(20261008)
    for label, f in MODELS:
        sc = pd.read_csv(os.path.join(RUN_DIR, "scores", f))
        cols = [next(c for c in (f"rank_{e}", f"logit_{e}", f"risk_{e}_2y") if c in sc.columns) for e in EP]
        look = {(int(p), int(t)): i for i, (p, t) in enumerate(zip(sc.pid, sc.t))}
        S = sc[cols].to_numpy()

        def scores_at(g):
            idx = np.array([look.get((int(p), int(u)), -1) if not np.isnan(u) else -1 for p, u in zip(pid, src[g])])
            s = np.full((len(idx), len(EP)), np.nan)
            s[idx >= 0] = S[idx[idx >= 0]]
            return s
        S0 = scores_at(0)
        res = {}
        for g in GAPS:
            Sg = scores_at(g)
            keep = ~np.isnan(Sg[:, 0])

            def macro(rows, Sx, strat=False):
                v = []
                for k, e in enumerate(EP):
                    y = truth[f"y_{e}_2y"].to_numpy()[rows]
                    ok = y >= 0
                    if not (0 < y[ok].sum() < ok.sum()):
                        return np.nan
                    s = Sx[rows][ok, k]
                    v.append(strat_auc(y[ok], s, band[rows][ok]) if strat else roc_auc_score(y[ok], s))
                return float(np.mean(v))
            rows = np.where(keep)[0]
            r = {"n_landmarks": int(keep.sum()), "n_persons": int(len(np.unique(pid[keep]))),
                 "n_pos": {e: int((truth[f"y_{e}_2y"].to_numpy()[keep] == 1).sum()) for e in EP},
                 "auc_gap": macro(rows, Sg), "auc_gap0_same_landmarks": macro(rows, S0),
                 "auc_gap_agestrat": macro(rows, Sg, True), "auc_gap0_same_landmarks_agestrat": macro(rows, S0, True),
                 "per_endpoint_gap": {}, "per_endpoint_gap0": {}}
            for k, e in enumerate(EP):
                y = truth[f"y_{e}_2y"].to_numpy()[rows]
                ok = y >= 0
                if 0 < y[ok].sum() < ok.sum():
                    r["per_endpoint_gap"][e] = float(roc_auc_score(y[ok], Sg[rows][ok, k]))
                    r["per_endpoint_gap0"][e] = float(roc_auc_score(y[ok], S0[rows][ok, k]))
            if g > 0:
                persons = np.unique(pid[keep])
                groups = {p: rows[pid[rows] == p] for p in persons}
                d, ds = [], []
                for _ in range(B):
                    ii = np.concatenate([groups[p] for p in rng.choice(persons, len(persons))])
                    a, b = macro(ii, S0), macro(ii, Sg)
                    if not (np.isnan(a) or np.isnan(b)):
                        d.append(a - b)
                    a, b = macro(ii, S0, True), macro(ii, Sg, True)
                    if not (np.isnan(a) or np.isnan(b)):
                        ds.append(a - b)
                r["drop"] = r["auc_gap0_same_landmarks"] - r["auc_gap"]
                r["drop_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else None
                r["drop_agestrat"] = r["auc_gap0_same_landmarks_agestrat"] - r["auc_gap_agestrat"]
                r["drop_agestrat_ci"] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if ds else None
                r["bootstrap_reps_used"] = len(d)
            res[str(g)] = r
            print(f"{label:52s} gap {g:2d}y  n={r['n_landmarks']:4d} ({r['n_persons']} persons, pos {r['n_pos']})  "
                  f"AUC {r['auc_gap']:.3f} vs gap0 {r['auc_gap0_same_landmarks']:.3f}"
                  + (f"  drop {r['drop']:+.3f} {np.round(r['drop_ci'], 3).tolist() if r['drop_ci'] else ''}" if g else "")
                  + f" | age-strat {r['auc_gap_agestrat']:.3f} vs {r['auc_gap0_same_landmarks_agestrat']:.3f}", flush=True)
        out["models"][label] = res
    json.dump(out, open(os.path.join(RUN_DIR, "packages", "GAP_AUC.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
