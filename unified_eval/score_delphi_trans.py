"""score_delphi_trans.py -- 第三部分（疾病分期转换）里 Delphi 家族的答题适配器。

    python score_delphi_trans.py --ckpt .../Delphi-ROSMAP-nodedup/ckpt.pt --dataset rosmap_nodedup \
        --out $ZOO_RUN_DIR/scores_trans/delphi-nodedup__s42.csv [--part1 $ZOO_RUN_DIR/scores/delphi-nodedup__s42.csv]

读法和第一部分**完全相同**（同一个函数：score_delphi.pseudo_records / query_logits / pos_align_length）：
截到站点的记录 + 站点后 1 天的 no-event 查询 token，读查询位置的 logit。只是多读几个 token：
  一道题的目标是一组分箱时（"首次 <27" = MMSE_borderline 或 MMSE_impaired），速率可加 ——
  Delphi 的等待时间是指数竞赛，"这一组里任意一个先发生"的速率 = 各自速率之和。
  风险 = 1 − exp(−Σλ · H · 365.25)（−expm1 计算）；排序分数 = log Σλ（logsumexp，不会在浮点上饱和）。

自检：和第一部分同一道题（MMSE<24、认知<−1、AD、死亡）读的是同一个 token、同一个位置，logit 必须和
第一部分的打分表逐站点相同（给了 --part1 就对，差 > 1e-4 就停）。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DAYS_PER_YEAR, RUN_DIR, label_names, load_bin, post_id  # noqa: E402
from score_delphi import load_model, pos_align_length, pseudo_records, query_logits  # noqa: E402
from transitions import SAME_AS_PART1, TRANS, TRUTH_HORIZONS, risk_col  # noqa: E402

# 读这些 token 的 logit：6 道题用到的全部目标 token（顺序固定，列名用 token 名）
READ_TOKENS = sorted({t for _, _, toks in TRANS.values() for t in toks})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--truth", default=None, help="默认 $ZOO_RUN_DIR/truth_trans_<split>.csv（站点集合和第一部分相同）")
    ap.add_argument("--part1", default=None, help="第一部分同一个 ckpt 的打分表：用来逐站点核对 logit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pos_q", type=int, default=50)
    args = ap.parse_args()

    truth = pd.read_csv(args.truth or os.path.join(RUN_DIR, f"truth_trans_{args.split}.csv"))
    names = label_names(args.dataset)
    assert names == label_names("rosmap"), "这份数据的词表和交付版不同"
    post = post_id(names)
    tok_post = [post[n] for n in READ_TOKENS]

    model, ck = load_model(args.ckpt, args.device)
    cfg = ck.get("config", {})
    assert cfg.get("dataset", args.dataset) == args.dataset, \
        f"ckpt 是在 {cfg.get('dataset')} 上训的，这里给的是 {args.dataset}"
    block_size = ck["model_args"]["block_size"]
    lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))

    arr = load_bin(args.dataset, args.split)
    data, hist_len = pseudo_records(arr, truth)
    q_age = truth.t.to_numpy() + 1
    pos_len = None
    if ck["model_args"].get("pos_embedding", False):
        pos_len, pos_pct = pos_align_length(args.dataset, cfg, block_size, lr, q=args.pos_q)
        print(f"  pos_embedding：训练 batch 的 x 长度 p5/p50/p95 = {pos_pct}，按第 {args.pos_q} 百分位 {pos_len} 对齐")
    logit = query_logits(model, data, hist_len, q_age, block_size, tok_post, args.device,
                         lifestyle_range=lr, pos_len=pos_len)
    logit = logit.astype(np.float64)

    res = truth[["pid", "t"]].copy()
    for k, n in enumerate(READ_TOKENS):
        res[f"logit_{n}"] = logit[:, k]
    for q, (_, _, toks) in TRANS.items():
        L = np.stack([res[f"logit_{n}"].to_numpy() for n in toks], 1)
        lse = np.logaddexp.reduce(L, axis=1)                     # log Σλ（每天）
        res[f"rank_{q}"] = lse
        rate = np.exp(lse)
        for H in TRUTH_HORIZONS:
            res[risk_col(q, H)] = -np.expm1(-rate * H * DAYS_PER_YEAR)
    res["hist_len"] = hist_len

    if args.part1 and os.path.exists(args.part1):
        p1 = pd.read_csv(args.part1)
        m = res.merge(p1, on=["pid", "t"], how="inner", suffixes=("", "_p1"), validate="one_to_one")
        assert len(m) == len(res), f"站点对不上：第三部分 {len(res)}，交集 {len(m)}"
        for q, e in SAME_AS_PART1.items():
            tok = TRANS[q][2][0]
            d = np.abs(m[f"logit_{tok}"].to_numpy() - m[f"logit_{e}"].to_numpy()).max()
            assert d < 1e-4, f"{q}（{tok}）和第一部分的 logit_{e} 最多差 {d:.2e}：读分位置或排布不一致"
        print(f"  和第一部分逐站点核对 logit：{len(m)} 个站点 × {len(SAME_AS_PART1)} 个 token，最大差 < 1e-4")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"scored {len(res)} landmarks  ckpt={args.ckpt}  iter={ck.get('iter_num')}  -> {args.out}")


if __name__ == "__main__":
    main()
