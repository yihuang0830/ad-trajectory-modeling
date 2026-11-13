"""sensitivity.py -- 防偏倚的敏感性分析（EVAL_PROTOCOL.md 第 8 节）。所有模型同一份代码。

    python sensitivity.py            # 读 $ZOO_RUN_DIR 下的真值和每个模型的打分 -> packages/<name>/sensitivity.json + SENSITIVITY.json

主结果的口径选择可能让 AUC 偏高或偏向某类模型。每个都换一种口径重算 2 年 AUC：
  main            主结果（只数答案已知的站点，所有站点合并）
  ipcw            失访加权（第二轮审查后重写）。删失模型 = 逻辑回归（年龄、性别、访视序号、当前 MMSE / 认知箱），
                  只在**窗口内没发生事件**的、处在风险中的站点上拟合 P(窗口走完仍被观察到) = p。
                  对照权重 1/p；病例要在事件那一刻被观察到，权重 1/G(T)，G(T) 介于 1 和 p 之间：主口径按窗口内删失速率
                  恒定插值 G(T) = p^(T/W)，两端（病例权重 1 / 1/p）另报成区间。死亡不需要来访视就能被记录，病例权重 = 1。
                  每个终点各拟合一个；权重在第 99 百分位截断。原版把病例也放进"答案已知"的模型里（病例总是已知），
                  权重和事件风险缠在一起，几乎复现不加权的 AUC，等于没检验失访偏倚。
  age_strat       只在同一个 5 岁年龄段内比较（病例和对照必须同年龄段）
  person_weight   每人权重相同（站点权重 = 1 / 这个人在这个终点上可计入的站点数）
  first_visit     每人只取第一次访视
  ad_cr           只对 AD：窗口内先死亡、没诊断 AD 的站点算阴性（累计发生率口径），而不是排除
另外：每个模型和参照模型（delphi-dedup）的**配对** bootstrap 差值（同一批人重抽样，1000 次）；README 里"分词排序"的第二步（nodedup 对 fullvisit）单独配对检验。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ENDPOINTS, GRID, RUN_DIR  # noqa: E402
from registry import DELPHI  # noqa: E402

EP = list(ENDPOINTS)
H = 2
REF_MODEL = "delphi-dedup"
PAIRS = [("delphi-nodedup", "delphi-fullvisit"), ("delphi-nodedup-split811", "delphi-fullvisit-split811"),
         ("delphi-dedup", "delphi-nodedup")]                                # 分词之间的配对比较（第二轮审查后加）
IPCW_RULES = ("ipcw", "ipcw_case1", "ipcw_casep")                           # 主口径、病例权重 1、病例权重 1/p


def landmark_table():
    """真值 + 每个站点的协变量和"处在风险中"标记（从 traj_truth 推，不碰模型）。"""
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    rows = []
    for r in truth.itertuples():
        d = traj[str(r.pid)]
        past = [v for v in d["visits"] if v[0] <= r.t]
        mm = [v[1] for v in past if v[1] is not None]
        cg = [v[2] for v in past if v[2] is not None]
        W = r.t + H * GRID
        ad, death = d["ad"], d["death"]
        fut = [v for v in d["visits"] if v[0] > r.t]
        low = {e: next((v[0] for v in fut if v[c] == 0), None) for e, c in (("mmse", 1), ("cogn", 2))}
        T = {"ad": None if ad is None or ad <= r.t else ad - r.t, "death": None if death is None else death - r.t,
             "mmse": None if low["mmse"] is None else low["mmse"] - r.t, "cogn": None if low["cogn"] is None else low["cogn"] - r.t}
        rows.append({"pid": r.pid, "t": r.t, **{f"T_{e}": (np.nan if v is None else float(v)) for e, v in T.items()},
                     "mmse_now": mm[-1] if mm else -1, "cogn_now": cg[-1] if cg else -1,
                     "n_vis": len(past),
                     "risk_ad": ad is None or ad > r.t, "risk_death": True,
                     "risk_mmse": bool(mm) and min(mm) >= 1, "risk_cogn": bool(cg) and min(cg) >= 1,
                     # AD 竞争风险口径：窗口内先死、且没诊断 AD -> 阴性
                     "died_first_in_window": death is not None and death <= W and (ad is None or ad > death)})
    lm = truth.merge(pd.DataFrame(rows), on=["pid", "t"], validate="one_to_one")
    lm["age_band"] = np.clip(((lm.age - 70) // 5).astype(int), 0, 6)       # <75, 75-80, ..., 95-100, >=100
    return lm


def ipcw_weights(lm, e):
    """-> {口径: 权重}。见文件头 ipcw 一条。"""
    y = lm[f"y_{e}_{H}y"].to_numpy()
    at = lm[f"risk_{e}"].to_numpy()
    fit = at & (y != 1)                                   # 只用窗口内没发生事件的站点拟合删失
    # n_vis（到此为止的访视数）恒等于 visit_idx + 1，是同一个量；L2 正则下多一列无害，留着是为了已发表的数字能逐位复现
    X = np.column_stack([lm.age, lm.male, lm.visit_idx, lm.n_vis,
                         *[(lm.mmse_now == k) for k in (-1, 0, 1, 2)], *[(lm.cogn_now == k) for k in (-1, 0, 1, 2)]]).astype(float)
    X = (X - X[fit].mean(0)) / (X[fit].std(0) + 1e-9)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit], (y == 0)[fit])
    p = np.clip(m.predict_proba(X)[:, 1], 1e-3, 1)
    frac = np.clip(np.nan_to_num(lm[f"T_{e}"].to_numpy(), nan=H * GRID) / (H * GRID), 0, 1)
    case = {"ipcw": 1.0 / p ** frac, "ipcw_case1": np.ones_like(p), "ipcw_casep": 1.0 / p}
    out = {}
    for k, cw in case.items():
        if e == "death":
            cw = np.ones_like(p)                          # 死亡不靠来访视记录
        w = np.where(at & (y == 0), 1.0 / p, np.where(at & (y == 1), cw, 0.0))
        cap = np.percentile(w[w > 0], 99)
        out[k] = np.minimum(w, cap)
    return out


def strat_auc(y, s, band):
    num = den = 0.0
    for b in np.unique(band):
        m = band == b
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos and nneg:
            a = roc_auc_score(y[m], s[m])
            num += a * npos * nneg
            den += npos * nneg
    return num / den if den else float("nan")


def variants(lm, score, W):
    """一个模型、一个打分表 -> 每个口径、每个终点的 AUC。score: (n,4) 按 EP 顺序，越大越危险。"""
    out = {}
    for k, e in enumerate(EP):
        y = lm[f"y_{e}_{H}y"].to_numpy()
        s = score[:, k]
        ok = y >= 0
        r = {"main": roc_auc_score(y[ok], s[ok])}
        for k_, w in W[e].items():
            r[k_] = roc_auc_score(y[ok], s[ok], sample_weight=w[ok])
        r["age_strat"] = strat_auc(y[ok], s[ok], lm.age_band.to_numpy()[ok])
        cnt = lm[ok].groupby("pid").pid.transform("size").to_numpy()
        r["person_weight"] = roc_auc_score(y[ok], s[ok], sample_weight=1.0 / cnt)
        fv = ok & (lm.visit_idx.to_numpy() == 0)
        r["first_visit"] = roc_auc_score(y[fv], s[fv]) if 0 < y[fv].sum() < fv.sum() else float("nan")
        if e == "ad":
            ycr = y.copy()
            ycr[(y == -1) & lm.risk_ad.to_numpy() & lm.died_first_in_window.to_numpy()] = 0
            okc = ycr >= 0
            r["ad_cr"] = roc_auc_score(ycr[okc], s[okc])
            r["n_added_by_cr"] = int(okc.sum() - ok.sum())
        out[e] = r
    keys = ["main", *IPCW_RULES, "age_strat", "person_weight", "first_visit"]
    out["macro"] = {k: float(np.mean([out[e][k] for e in EP])) for k in keys}
    return out


def load_scores(lm, path):
    sc = pd.read_csv(path)
    m = lm[["pid", "t"]].merge(sc, on=["pid", "t"], how="left", validate="one_to_one")
    cols = []
    for e in EP:
        c = next((c for c in (f"rank_{e}", f"logit_{e}", f"risk_{e}_{H}y") if c in m.columns), None)
        cols.append(m[c].to_numpy())
    S = np.column_stack(cols)
    assert not np.isnan(S).any(), f"{path}: 有站点没有打分"
    return S


def paired_diff(lm, S_a, S_b, B=1000, seed=20261008, W=None):
    """macro AUC(a) - macro AUC(b)，同一批人 bootstrap。W：{终点: 站点权重}（IPCW 口径；权重在全样本上拟合一次）。"""
    pid = lm.pid.to_numpy()
    persons = np.unique(pid)
    groups = {p: np.where(pid == p)[0] for p in persons}
    rng = np.random.default_rng(seed)
    Y = [lm[f"y_{e}_{H}y"].to_numpy() for e in EP]

    def macro(idx, S):
        v = []
        for k in range(len(EP)):
            y = Y[k][idx]
            ok = y >= 0
            if not (0 < y[ok].sum() < ok.sum()):
                return np.nan
            v.append(roc_auc_score(y[ok], S[idx][ok, k], sample_weight=None if W is None else W[EP[k]][idx][ok]))
        return np.mean(v)
    allidx = np.arange(len(lm))
    point = macro(allidx, S_a) - macro(allidx, S_b)
    bs = []
    for _ in range(B):
        idx = np.concatenate([groups[p] for p in rng.choice(persons, len(persons))])
        d = macro(idx, S_a) - macro(idx, S_b)
        if not np.isnan(d):
            bs.append(d)
    return {"diff": float(point), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_le_0": float(np.mean(np.array(bs) <= 0))}


def main():
    lm = landmark_table()
    W = {e: ipcw_weights(lm, e) for e in EP}
    models = [(m[0], os.path.join(RUN_DIR, "scores", f"{m[0]}__s42.csv")) for m in DELPHI]
    S = {n: load_scores(lm, p) for n, p in models if os.path.exists(p)}
    ref = S[REF_MODEL]
    summary = {"ipcw_weight_max": {e: float(W[e]["ipcw"].max()) for e in EP},
               "n_landmarks": int(len(lm)), "models": {}, "pairs": {}}
    for n, Sm in S.items():
        res = variants(lm, Sm, W)
        res["paired_vs_" + REF_MODEL] = None if n == REF_MODEL else paired_diff(lm, Sm, ref)
        out = os.path.join(RUN_DIR, "packages", n)
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, "sensitivity.json"), "w"), indent=1)
        summary["models"][n] = res
        print(n, {k: round(v, 3) for k, v in res["macro"].items()},
              "" if n == REF_MODEL else f"diff vs {REF_MODEL} {res['paired_vs_' + REF_MODEL]['diff']:+.3f} "
                                        f"{[round(x, 3) for x in res['paired_vs_' + REF_MODEL]['ci']]}", flush=True)
    for a, b in PAIRS:
        if a in S and b in S:
            summary["pairs"][f"{a} - {b}"] = paired_diff(lm, S[a], S[b])
            print("paired", a, "-", b, summary["pairs"][f"{a} - {b}"], flush=True)
    json.dump(summary, open(os.path.join(RUN_DIR, "packages", "SENSITIVITY.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
