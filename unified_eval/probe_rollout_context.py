"""probe_rollout_context.py -- rollout 的第一步读到的上下文，和直接读出是不是同一个（2026-10-29 诊断；只打印聚合数，在 RIS 上跑）。

    python probe_rollout_context.py --name delphi-dedup [--limit 500]

直接读出（score_delphi_trans）：站点为止的流 + 查询 token，经 utils.get_batch(padding='regular') 排布 ——
  get_batch 会在固定年龄（每 5 年）插 No-event token，和训练时（padding='random'，同一个速率）一样。
rollout（rollout_trans.simulate_from -> engine.simulate）：站点为止的原始流 + 查询 token，没有 No-event 插值，
  engine.simulate 默认 mask_ties=False。
这里对 val 的每个站点，用 engine._logits 在 rollout 的上下文上读查询位置的 logit（mask_ties 关 / 开各一遍），
换成和直接读出同一个排序量（各题目标 token 的 logsumexp），和打分表里的直接读出比：
  逐站点 log 速率差的分布、Spearman 相关、2 / 5 年 AUC（同一张真值表、同一批站点）。
如果 rollout 上下文的 AUC 明显低于直接读出，rollout 版（-sim）的 AUC 掉的那一截里有一部分来自上下文，而不是生成本身。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DAYS_PER_YEAR, DELPHI_DIR, RUN_DIR  # noqa: E402
from registry import DELPHI  # noqa: E402
from score_delphi_trans import READ_TOKENS  # noqa: E402
from transitions import TRANS, label_col, risk_col  # noqa: E402

import torch  # noqa: E402  （numpy / pandas 先于 torch：RIS 上 libstdc++ 的加载顺序）

import rollout_delphi as RD  # noqa: E402

Q = list(TRANS)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    m = y >= 0
    return float(roc_auc_score(y[m], s[m])) if m.any() and len(np.unique(y[m])) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="delphi-dedup")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    reg = {m[0]: m for m in DELPHI}
    _, dirs, dataset, *_ = reg[args.name]
    main_seed = json.load(open(os.path.join(RUN_DIR, "packages", args.name, "metrics.json")))["main_seed"]
    sc = pd.read_csv(os.path.join(RUN_DIR, "scores_trans", f"{args.name}__{main_seed}.csv"))
    tr = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv"))
    sc = sc.merge(tr[["pid", "t"] + [label_col(q, H) for q in Q for H in (2, 5)]], on=["pid", "t"], how="inner",
                  validate="one_to_one")
    if args.limit:
        sc = sc.iloc[: args.limit]
    eng = RD.make_engine(os.path.join(DELPHI_DIR, dirs[0], "ckpt.pt"), dataset, "cpu")
    streams = RD.person_streams(dataset)
    cols = [eng.tok[t] for t in READ_TOKENS]
    B = eng.block_size
    lg = {False: np.zeros((len(sc), len(READ_TOKENS))), True: np.zeros((len(sc), len(READ_TOKENS)))}
    n_cut = 0
    with torch.no_grad():
        for i, (p, t) in enumerate(zip(sc.pid.to_numpy(), sc.t.to_numpy())):
            ages, toks = streams[int(p)]
            m = ages <= t
            a2 = np.append(ages[m], t + 1.0)[-B:]
            k2 = np.append(toks[m], int(eng.res.NO_EVENT))[-B:]
            n_cut += int(m.sum() + 1 > B)
            idx = torch.as_tensor(k2[None], dtype=torch.long)
            age = torch.as_tensor(a2[None], dtype=torch.float32)
            for mt in (False, True):
                lg[mt][i] = eng._logits(idx, age, mt)[0, cols].double().numpy()
    out = {"name": args.name, "seed": main_seed, "n_landmarks": int(len(sc)), "prefix_truncated_to_block": n_cut,
           "questions": {}}
    for qi, q in enumerate(Q):
        j = [READ_TOKENS.index(x) for x in TRANS[q][2]]
        # 直接读出的排序量：从 2 年风险反解 log Σλ（风险 = −expm1(−Σλ·H·365.25)，单调）
        r2 = np.clip(sc[risk_col(q, 2)].to_numpy(), 1e-12, 1 - 1e-12)
        lse_dir = np.log(-np.log1p(-r2) / (2 * DAYS_PER_YEAR))
        row = {"auc_direct": {}, "auc_rollout_ctx": {}, "auc_rollout_ctx_masktie": {}}
        for mt, key in ((False, "rollout_ctx"), (True, "rollout_ctx_masktie")):
            lse = np.logaddexp.reduce(lg[mt][:, j], axis=1)
            d = lse - lse_dir
            row[f"dlog_rate_{key}"] = {"mean": float(d.mean()), "sd": float(d.std()),
                                       "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
                                       "spearman": float(pd.Series(lse).corr(pd.Series(lse_dir), method="spearman"))}
            for H in (2, 5):
                row[f"auc_{key}"][f"{H}y"] = auc(sc[label_col(q, H)].to_numpy(), lse)
        for H in (2, 5):
            row["auc_direct"][f"{H}y"] = auc(sc[label_col(q, H)].to_numpy(), sc[risk_col(q, H)].to_numpy())
        out["questions"][q] = row
    for k in ("auc_direct", "auc_rollout_ctx", "auc_rollout_ctx_masktie"):
        out[f"macro_{k}"] = {f"{H}y": float(np.nanmean([out["questions"][q][k][f"{H}y"] for q in Q])) for H in (2, 5)}
    od = os.path.join(RUN_DIR, "packages", "_probe_rollout_context")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"{args.name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "questions"}, indent=1))
    for q in Q:
        r = out["questions"][q]
        print(q, {k: r[k] for k in ("auc_direct", "auc_rollout_ctx", "auc_rollout_ctx_masktie")},
              "dlog", {k: round(r[f"dlog_rate_{k}"]["mean"], 3) for k in ("rollout_ctx", "rollout_ctx_masktie")})


if __name__ == "__main__":
    main()
