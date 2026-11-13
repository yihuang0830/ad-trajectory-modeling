"""nacc_tabular.py -- 逻辑回归、XGBoost 只在 NACC 上训练，在 NACC test 段上评估（2026-10-28 加）。

为什么：看训练量多 N 倍以后，Delphi 能不能追上只看当前访视的逻辑回归。和 nacc_delphi.nacc_test 同一批站点、同一套标签。

    python nacc_tabular.py lr      # lr-cur 的配方：当前访视 54 维 + 静态背景 25 维，60 个二分类器，C ∈ {0.01, 0.1, 1}
    python nacc_tabular.py xgb     # xgb-l2c 的配方：再加手工历史特征 120 维，depth ∈ {3, 5}，早停
    -> $ZOO_RUN_DIR/packages/_nacc/nacc_tabular_<kind>.json

调参：NACC 的 train 段按人随机留 10% 做调参 / 早停（固定种子），其余 90% 训练；test 段只用来报数。
（pretrain.py 的网络是在 test 段上早停的，这里更严一点。）
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import bl_data as B
import pretrain as P
import tabular as TB
from common import RUN_DIR

FEAT = {"lr": "cur", "xgb": "l2c"}
NAME = {"lr": "lr-cur", "xgb": "xgb-l2c"}
SEED = 20261028


def aucs(Yv, R):
    """nacc_delphi.nacc_test 的同一个口径。"""
    out = {}
    for qi, q in enumerate(B.Q):
        for H in (1, 2, 3, 5, 10):
            y, r = Yv[:, qi, H - 1], R[:, qi, H - 1]
            k = y >= 0
            out[f"{q}_{H}y"] = float(roc_auc_score(y[k], r[k])) if len(np.unique(y[k])) == 2 else None
            if H == 2:
                out[f"{q}_2y_oe"] = float(y[k].sum() / r[k].sum())
    return {"part1_2y_4endpoints": float(np.mean([out[f"{q}_2y"] for q in ("ad", "death", "mmse_lt24", "cogn_ltm1")])),
            "part3_5y_6questions": float(np.mean([out[f"{q}_5y"] for q in B.Q])), "cells": out}


def main(kind):
    D = P.load_nacc()
    tr_all = D["train"]
    rng = np.random.default_rng(SEED)
    hold = rng.random(len(tr_all["pid"])) < 0.1
    sub = lambda m: {k: v[m] for k, v in tr_all.items()}        # noqa: E731
    tr, tu, te = (B.landmarks(d, FEAT[kind]) for d in (sub(~hold), sub(hold), D["test"]))
    M = D["test"]["M"].astype(bool)
    assert len(te["F"]) == M.sum()                               # 行的顺序 = nacc_test 里的 Y[M]
    log, best = [], None
    for hp in TB.CLF_GRID[kind]:
        Rtu, Rte = TB.per_target(kind, hp, tr, tu, te)
        score = B.macro_auc(tu["Y"], Rtu)
        log.append({"hp": hp, "tune_macro_auc": score})
        print(f"{NAME[kind]} {hp}: tune macro AUC {score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, hp, Rte)
    res = {"name": NAME[kind], "n_train_persons": int((~hold).sum()), "n_tune_persons": int(hold.sum()),
           "n_train_landmarks": int(len(tr["F"])), "n_test_landmarks": int(len(te["F"])), "chosen": best[1], "grid": log,
           "test": aucs(te["Y"], best[2])}
    print(f"{NAME[kind]} NACC test: part1 2y {res['test']['part1_2y_4endpoints']:.4f}  part3 5y {res['test']['part3_5y_6questions']:.4f}",
          flush=True)
    out = os.path.join(RUN_DIR, "packages", "_nacc", f"nacc_tabular_{kind}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(FEAT))
    main(ap.parse_args().kind)
