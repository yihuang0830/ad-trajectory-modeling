"""pos_leak_check.py -- 带位置编码的模型有没有从"位置下标"里读到未来？

    python pos_leak_check.py --name delphi-nodedup-pos-split811 --ckpt ... --dataset rosmap_nodedup_3w

同一个 ckpt、同一批站点、同一个查询 token（站点后 1 天的 no-event），只换输入排布：
  honest ：只给到站点为止的记录（统一评估的做法），查询 token 在记录末尾那一格。
  leaky  ：给**完整记录**（含站点之后的全部 token），把查询 token 按年龄插进去。因果注意力保证
           查询 token 看不到之后的 token；但记录是右对齐的，查询 token 的位置下标 = 对齐长度 − 1 −
           "它后面还有几个 token"，于是未来的长度通过位置漏了进来（以前在完整记录上读分就是这种排布）。
两种排布都按**训练时的方式**对齐：训练用 cut_batch=True，每个 batch 右对齐到这一批最长的那条，
所以这里右对齐到 max(自身长度, L)，L 取模拟训练 batch 长度的第 5/50/95 百分位各跑一遍（敏感性）。
不按训练方式对齐的话，查询 token 会落在训练里从没占用过的位置上，测到的是"没训练过的位置"的噪声而不是泄漏
（2026-10-07 审查发现的问题，第一版就是这样错的）。
不带位置编码的模型两种排布的输出应当逐位相同（对照）。

输出 $ZOO_RUN_DIR/packages/<name>/pos_leak.json + fig5_pos_leak.png（这个模型独有的图）。
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ENDPOINTS, MAIN_H, RUN_DIR, label_names, load_bin, post_id  # noqa: E402
from score_delphi import align_right, load_model, pos_align_length, pseudo_records  # noqa: E402
from utils import get_batch, get_p2i  # noqa: E402  (score_delphi 已把 delphi/ 放进 sys.path)


def leaky_records(arr, landmarks):
    """完整记录 + 在 t+1 天插一个查询 token（按年龄排序后它落在站点访视之后、下一次访视之前）。"""
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    pids, first = np.unique(arr[:, 0], return_index=True)
    counts = np.diff(np.append(first, len(arr)))
    starts = {int(p): (f, c) for p, f, c in zip(pids, first, counts)}
    chunks, n_future = [], np.zeros(len(landmarks), np.int64)
    for j, (pid, t) in enumerate(zip(landmarks.pid.to_numpy(), landmarks.t.to_numpy())):
        f, c = starts[int(pid)]
        h = arr[f: f + c]
        k = int((h[:, 1] <= t).sum())
        # 末尾再接一个哑目标：窗口最后一行只当目标、不进输入；站点是最后一次访视时查询 token
        # 本来就是最后一行。统一加一行，所有站点的排布规则一致。
        rec = np.empty((len(h) + 2, 3), np.int64)
        rec[:, 0] = j
        rec[:k, 1:] = h[:k, 1:]
        rec[k] = (j, t + 1, 0)
        rec[k + 1:-1, 1:] = h[k:, 1:]
        rec[-1] = (j, max(int(h[:, 1].max()), t + 1) + 1, 0)
        n_future[j] = len(h) - k
        chunks.append(rec)
    return np.concatenate(chunks), n_future


@torch.no_grad()
def read_query(model, data, q_age, block_size, tok_post, lr, pos_len, bs=64):
    p2i = get_p2i(data)
    n = len(q_age)
    out = np.zeros((n, len(tok_post)))
    qpos = np.zeros(n, np.int64)
    for s0 in range(0, n, bs):
        ix = list(range(s0, min(n, s0 + bs)))
        x, a, y, b = get_batch(ix, data, p2i, select="left", block_size=block_size, device="cpu",
                               padding="regular", no_event_token_rate=5, lifestyle_augmentations=False,
                               lifestyle_token_range=lr)
        if pos_len is not None:
            # 逐行按训练方式对齐：每行单独裁，免得同一批里长记录把短记录挤到别的位置上
            rows = []
            for r in range(len(ix)):
                rows.append(align_right(x[r:r + 1], a[r:r + 1], y[r:r + 1], b[r:r + 1], pos_len))
            W = max(t[0].shape[1] for t in rows)
            pad = lambda t, v: torch.cat([torch.full((1, W - t.shape[1]), v, dtype=t.dtype), t], 1)  # noqa: E731
            x = torch.cat([pad(t[0], 0) for t in rows]); a = torch.cat([pad(t[1], -10000.0) for t in rows])
            y = torch.cat([pad(t[2], 0) for t in rows]); b = torch.cat([pad(t[3], -10000.0) for t in rows])
        qa = torch.from_numpy(q_age[ix].astype(np.float32))
        is_q = (x == 1) & (a == qa[:, None])
        assert (is_q.sum(1) >= 1).all()
        qi = torch.argmax(is_q.int(), 1)
        if pos_len is not None:
            # 逐行前向：每行的位置下标从它自己的对齐起点算（pad 到同宽会改变 pos 模型的位置）
            for r in range(len(ix)):
                n_tok = int((x[r] != 0).sum()); Lr = max(pos_len, n_tok); st = x.shape[1] - Lr
                lg = model(x[r:r + 1, st:], a[r:r + 1, st:], y[r:r + 1, st:], b[r:r + 1, st:])[0]
                out[ix[r]] = lg[0, int(qi[r]) - st, tok_post].numpy()
                qpos[ix[r]] = int(qi[r]) - st
        else:
            logits = model(x, a, y, b)[0]
            out[ix] = logits[torch.arange(len(ix)), qi][:, tok_post].numpy()
            qpos[ix] = qi.numpy()
    return out, qpos


def plot(res, name, path):
    """只用 pos_leak.json 的内容画图（本地可重画：python pos_leak_check.py --replot <pos_leak.json> --name <名字>）。"""
    from matplotlib.patches import Patch
    is_pos = res["pos_embedding"]
    pct = res.get("align_lengths_p5_p50_p95") or [None, None, None]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xs = np.arange(len(ENDPOINTS))
    cols = ("#4d4d4d", "#e08214", "#cccccc")
    labs = ("record ends at landmark (unified eval)", "full-record positions (old readout)", "# tokens after landmark, alone")
    hv = [res["endpoints"][e]["honest"] for e in ENDPOINTS]
    lv = [res["endpoints"][e]["leaky"] for e in ENDPOINTS]
    fv = [res["endpoints"][e]["future_length_alone"] for e in ENDPOINTS]
    for off, vals, c in zip((-0.25, 0.0, 0.25), (hv, lv, fv), cols):
        ax.bar(xs + off, vals, 0.25, color=c)
    for k, e in enumerate(ENDPOINTS):
        ds = [res["by_align"][L]["endpoints"][e]["leaky"] - res["by_align"][L]["endpoints"][e]["honest"]
              for L in res["by_align"]]
        txt = f"{lv[k] - hv[k]:+.3f}" + (f"\n[{min(ds):+.3f},{max(ds):+.3f}]" if len(ds) > 1 else "")
        ax.text(xs[k], max(hv[k], lv[k]) + 0.01, txt, ha="center", fontsize=7.5)
    ax.set_xticks(xs, [ENDPOINTS[e][0] for e in ENDPOINTS], fontsize=8.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel(f"AUC, {MAIN_H}-year risk")
    sub = (f"aligned like training (batch length p50={pct[1]}; [..] = range over p5/p50/p95)" if is_pos
           else "no positional embedding")
    ax.set_title(f"{name}: does position leak the future?\n{sub}", fontsize=8.5)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labs)], fontsize=7, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--replot", default=None, help="只用已有的 pos_leak.json 重画图")
    args = ap.parse_args()
    if args.replot:
        plot(json.load(open(args.replot)), args.name, os.path.join(os.path.dirname(args.replot), "fig5_pos_leak.png"))
        return
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    model, ck = load_model(args.ckpt, "cpu")
    bsz = ck["model_args"]["block_size"]
    cfg = ck.get("config", {})
    lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
    post = post_id(label_names(args.dataset))
    tok_post = [post[ENDPOINTS[e][1]] for e in ENDPOINTS]
    arr = load_bin(args.dataset, "val")
    q_age = truth.t.to_numpy() + 1
    d_h, _ = pseudo_records(arr, truth)
    d_l, n_future = leaky_records(arr, truth)
    is_pos = bool(ck["model_args"].get("pos_embedding", False))
    pct = pos_align_length(args.dataset, cfg, bsz, lr)[1] if is_pos else [None]
    res = {"pos_embedding": is_pos, "align_lengths_p5_p50_p95": pct, "by_align": {}}
    for L in pct:
        honest, qh = read_query(model, d_h, q_age, bsz, tok_post, lr, L)
        leaky, ql = read_query(model, d_l, q_age, bsz, tok_post, lr, L)
        r = {"max_abs_logit_diff": float(np.abs(honest - leaky).max()),
             "query_pos_honest_median": float(np.median(qh)), "query_pos_leaky_median": float(np.median(ql)),
             "endpoints": {}}
        for k, e in enumerate(ENDPOINTS):
            y = truth[f"y_{e}_{MAIN_H}y"].to_numpy()
            m = y >= 0
            r["endpoints"][e] = {"honest": float(roc_auc_score(y[m], honest[m, k])),
                                 "leaky": float(roc_auc_score(y[m], leaky[m, k])),
                                 "future_length_alone": float(roc_auc_score(y[m], -n_future[m]))}
        res["by_align"][str(L)] = r
    main_L = str(pct[1] if is_pos else None)
    res["max_abs_logit_diff"] = res["by_align"][main_L]["max_abs_logit_diff"]
    res["endpoints"] = res["by_align"][main_L]["endpoints"]
    out = os.path.join(RUN_DIR, "packages", args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "pos_leak.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res, args.name, os.path.join(out, "fig5_pos_leak.png"))
    print(json.dumps({L: {"maxdiff": round(v["max_abs_logit_diff"], 2), "qpos": (v["query_pos_honest_median"], v["query_pos_leaky_median"]),
                          **{e: (round(x["honest"], 3), round(x["leaky"], 3)) for e, x in v["endpoints"].items()}}
                      for L, v in res["by_align"].items()}, indent=1))


if __name__ == "__main__":
    main()
