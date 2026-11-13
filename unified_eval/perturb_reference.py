"""perturb_reference.py -- fig4 面板 C 的"数据一侧"：验证集上的条件关联（和任何模型无关，只算一次）。

    python perturb_reference.py      # -> $ZOO_RUN_DIR/perturb_reference.json（聚合数）

每个终点在**验证集**全部站点（答案已知的）上拟合一个逻辑回归（没有任何模型训练过这批人 -> 和所有模型的训练数据独立；
训练段上的同一个回归另存在 endpoints_train 里备查——它和在训练段上训练的模型是同一份数据上拟合的，吻合是必然的，不能拿来比）：
  年龄、男、教育（含缺失）、APOE e4 1/2 个（含缺失）、最近一次 MMSE 和认知（含"测过没有"）、卒中、抑郁、糖尿病、高血压
MMSE 和认知用**分段线性**（拐点放在分箱切点：MMSE 24、27；认知 −1、0），同时容纳"分箱"和"线性"两种形状，
不偏向任何一种模型的输入表示（2026-10-08 审查后修订：原来是在验证集上、用线性形状拟合，形状一换 MMSE→AD 那一格就会变号）。
然后对扰动队列（验证集）里每个有资格、处在风险中的人，把扰动代进特征（perturb_features.apply，和统一输入的模型同一个修改），
效应 = 平均的 Δ(线性预测) = 平均的 log 优势比。CI = 按人 bootstrap 200 次（重拟合回归）。
这是**关联**不是因果，只作量级参照；面板 C 只是描述，不参与排名。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ENDPOINTS, RUN_DIR  # noqa: E402
from perturbations import ELIG, PERTS  # noqa: E402
from perturb_features import apply  # noqa: E402

EP = list(ENDPOINTS)
FEATS = ["age", "male", "educ", "educ_na", "e4x1", "e4x2", "apoe_na", "mmse", "mmse_gt24", "mmse_gt27", "mmse_has",
         "cogn", "cogn_gt_m1", "cogn_gt0", "cogn_has", "stroke", "depression", "diabetes", "hypertension"]


def feats(X, S, L, norm):
    """(..., F) 特征；X/L 是某次访视那一行（或多行），S 是静态（已广播）。MMSE / 认知换回原始单位做分段线性。"""
    mm = L[..., 0] * norm["sd"]["cts_estmmse30"] + norm["mu"]["cts_estmmse30"]
    cg = L[..., 8] * norm["sd"]["cogn_global"] + norm["mu"]["cogn_global"]
    mm_has, cg_has = L[..., 1], L[..., 9]
    mm = np.where(mm_has > 0.5, mm, 0.0)
    cg = np.where(cg_has > 0.5, cg, 0.0)
    sh = X[..., 51].shape
    b = lambda v: np.broadcast_to(v, sh)  # noqa: E731
    return np.stack([X[..., 51], b(S[..., 0]), b(S[..., 1]), b(S[..., 2]), b(S[..., 9]), b(S[..., 10]), b(S[..., 11]),
                     mm, np.maximum(mm - 24, 0) * mm_has, np.maximum(mm - 27, 0) * mm_has, mm_has,
                     cg, np.maximum(cg + 1, 0) * cg_has, np.maximum(cg, 0) * cg_has, cg_has,
                     X[..., 48], X[..., 49],
                     np.where(X[..., 40] > 0.5, 0, X[..., 31]), np.where(X[..., 39] > 0.5, 0, X[..., 30])], -1)


def fit_reference(F, Yl, pid, ok_all, j, dF, cohort, e, B=200, seed=20261008):
    rng = np.random.default_rng(seed)
    ok = Yl[:, j] >= 0
    mu, sd = F[ok].mean(0), F[ok].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    fit = lambda ii: LogisticRegression(C=1e4, max_iter=5000).fit((F[ii] - mu) / sd, Yl[ii, j])  # noqa: E731
    beta = fit(np.where(ok)[0]).coef_[0] / sd
    persons = np.unique(pid)
    groups = {p: np.where(pid == p)[0] for p in persons}
    coefs = []
    for _ in range(B):
        pick = np.concatenate([groups[p] for p in rng.choice(persons, len(persons))])
        pick = pick[ok[pick]]
        if 0 < Yl[pick, j].sum() < len(pick):
            coefs.append(fit(pick).coef_[0] / sd)
    coefs = np.array(coefs)
    res = {"n": int(ok.sum()), "n_pos": int(Yl[ok, j].sum()), "beta_per_unit": dict(zip(FEATS, beta.round(4).tolist())),
           "perturb": {}}
    at = set(cohort.pid[cohort[f"risk_{e}"]].astype(int))
    for pert, rows in dF.items():
        d = np.array([r[1] for r in rows if r[0] in at])
        if len(d) == 0:
            continue
        boot = (d @ coefs.T).mean(0)
        res["perturb"][pert] = {"dlogit": float((d @ beta).mean()),
                                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                "n_persons": int(len(d))}
    return res


def design(Z, split, norm):
    X, M, Y, S, P, L = (Z[f"{split}_{k}"] for k in ("X", "M", "Y", "S", "pid", "L"))
    Sb = np.broadcast_to(S[:, None, :], (S.shape[0], X.shape[1], S.shape[1]))
    return feats(X, Sb, L, norm)[M], Y[M], np.repeat(P, M.sum(1))


def main():
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_meta.json")))
    norm = meta["norm"]
    ycols = meta["labels"]
    # 扰动：验证集的扰动队列（特征差和回归系数无关）
    Xv, Mv, Tv, Sv, Pv, Lv = (Z[f"val_{k}"] for k in ("X", "M", "T", "S", "pid", "L"))
    cohort = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"))
    idx = {int(p): i for i, p in enumerate(Pv)}
    dF = {}
    for pert in PERTS:
        if pert == "orig":
            continue
        rows = []
        for _, c in cohort.iterrows():
            if not bool(c[ELIG[pert]]):
                continue
            i = idx[int(c.pid)]
            vL = int(np.where(Tv[i] == int(c.tL))[0][0])
            X1, S1, L1 = apply(pert, Xv[i], Mv[i], Sv[i], Lv[i], vL, norm)
            rows.append((int(c.pid), feats(X1[vL], S1, L1[vL], norm) - feats(Xv[i][vL], Sv[i], Lv[i][vL], norm)))
        dF[pert] = rows
    out = {"features": FEATS, "fit_split": "val (held out from every model)",
           "encoding": "piecewise-linear MMSE (knots 24, 27) and cognition (knots -1, 0)", "endpoints": {}, "endpoints_train": {}}
    for split, key in (("val", "endpoints"), ("train", "endpoints_train")):
        F, Yl, pid = design(Z, split, norm)
        if split == "val":
            out["n_landmarks"] = int(len(F))
        for e in EP:
            out[key][e] = fit_reference(F, Yl, pid, None, ycols.index(f"y_{e}_2y"), dF, cohort, e)
    json.dump(out, open(os.path.join(RUN_DIR, "perturb_reference.json"), "w"), indent=1)
    for e in EP:
        print(e, {p: (round(v["dlogit"], 2), [round(x, 2) for x in v["ci"]]) for p, v in out["endpoints"][e]["perturb"].items()
                  if p in ("apoe", "mmse25", "cogn", "stroke", "age5", "male", "educ")})


if __name__ == "__main__":
    main()
