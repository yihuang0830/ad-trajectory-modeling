"""delphi_upstream.py -- 用 Delphi 的上游原版代码（一行不改）分别在 ROSMAP、NACC 上从零训练，各自在自己的留出集上评估（2026-10-26 加）。

为什么：zoo 里的 Delphi 是改过的版本（加了位置嵌入、判别头、访视头、注意力范围等开关，背景 token 的年龄抖动范围改成可配，
排序改成稳定排序），配方也缩小了（6 层 × 96 宽、dropout 0.1、12,000 步）。这里排除"是我们改坏了"这个可能。

    python delphi_upstream.py configs [--max_iters N]   # 上游 config/train_delphi.py -> delphi_upstream/config/train_{rosmap,nacc}.py
    python delphi_upstream.py score                     # 打分：ROSMAP val -> scores/、scores_trans/（统一评估的格式）；NACC test -> AUC
                                                        #   -> $ZOO_RUN_DIR/packages/_nacc/delphi_upstream.json

代码：$AD_BASE/delphi_upstream/ 里的 model.py / utils.py / train.py / configurator.py 是 github.com/gerstung-lab/Delphi
fb72166 原样拷过去的（md5 和克隆一致，UPSTREAM_COMMIT 里有记录），data/ 链到 delphi/data。训练和打分都 import 这份代码。
配置：上游的 config/train_delphi.py（Delphi-2M：12 层 × 12 头 × 120 宽、dropout 0、学习率 6e-4、warmup 1000、weight decay 0.2），
只改和数据有关的：dataset、out_dir、vocab_size（129）、ignore_tokens（padding + 性别 + 背景块，和上游"padding, sex and lifestyle"
同一个意思）。步数：上游的 150,000 步日程原样不动（学习率的余弦衰减也按 15 万步算），ris/delphi_upstream.sbatch 跑到 27 分钟停；
上游只在 val loss 变好时存 ckpt.pt，所以留下的是 val 最低的那一步。ROSMAP 最低点在 N 步、NACC 在 N 步，
之后一路上升到停的时候，没有必要接着跑。
分词：直接用已有的两份（delphi/data/rosmap = 交付版 dedup；delphi/data/nacc = nacc_delphi.py，同一个词表）。上游代码把位移前
3..11 号 token 的年龄随机挪动（UKB 的生活方式块）；在这份词表里 3..11 是子研究、教育、种族（白、黑），APOE 等其余背景 token
不挪 —— 这是"上游代码 + 我们的分词"的字面结果，没有改。
评估：读法和 zoo 的 Delphi 一样（截到站点的记录 + 站点后 1 天的 no-event 查询 token，读速率，风险 = 1 − exp(−λH)），
只是排布和前向都用上游的 get_batch / Delphi。ROSMAP：val，统一评估的第一、三部分真值；NACC：test 段，nacc_data 的标签。
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import AD_BASE, DATA_DIR, DAYS_PER_YEAR, ENDPOINTS, HORIZONS, RUN_DIR, label_names, load_bin, post_id  # noqa: E402

UP = os.path.join(AD_BASE, "delphi_upstream")
SEEDS = (42, 43, 44)
NAME = {"rosmap": "delphi-upstream", "nacc": "delphi-upstream-nacc"}
OUT_DIR = {"rosmap": "Delphi-upstream-rosmap", "nacc": "Delphi-upstream-nacc"}
IGNORE = [0] + list(range(2, 34))          # 位移后：padding、性别（2、3）、背景块（4..33）


def dname(ds, s):
    return OUT_DIR[ds] if s == 42 else f"{OUT_DIR[ds]}-s{s}"


def configs(max_iters):
    src = open(os.path.join(UP, "config", "train_delphi.py")).read()
    for ds in ("rosmap", "nacc"):
        assert label_names(ds) == label_names("rosmap"), f"{ds} 的词表和交付版不同"
        rep = [("out_dir = 'Delphi'", f"out_dir = '{OUT_DIR[ds]}'"), ("dataset = 'ukb_real_data'", f"dataset = '{ds}'"),
               ("vocab_size = 1270", f"vocab_size = {len(label_names(ds))}"),
               ("ignore_tokens = [0, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12]", f"ignore_tokens = {IGNORE}")]
        if max_iters:
            rep += [("max_iters = 150000", f"max_iters = {max_iters}"), ("lr_decay_iters = 150000", f"lr_decay_iters = {max_iters}")]
        cfg = src
        for a, b in rep:
            assert cfg.count(a) == 1, f"上游配置里找不到 {a!r}"
            cfg = cfg.replace(a, b)
        cfg = (f"# 由 models_code/delphi_upstream.py 从上游 config/train_delphi.py 生成，只改了：{[a for a, _ in rep]}\n" + cfg)
        open(os.path.join(UP, "config", f"train_{ds}.py"), "w").write(cfg)
        print(f"-> config/train_{ds}.py", flush=True)


# ============================================================================== 上游代码的读法


def _upstream():
    """上游的 model.py / utils.py 用自己的模块名载入（和 delphi/ 里改过的那两个同名）。"""
    mods = {}
    for n in ("model", "utils"):
        spec = importlib.util.spec_from_file_location(f"delphi_upstream_{n}", os.path.join(UP, f"{n}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"delphi_upstream_{n}"] = m
        spec.loader.exec_module(m)
        mods[n] = m
    return mods["model"], mods["utils"]


def load(ckpt, device):
    import torch
    M, _ = _upstream()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = M.Delphi(M.DelphiConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    return model.eval().to(device), ck


def query_logits(model, data, hist_len, q_age, block_size, tok_post, device, batch_size=256):
    """score_delphi.query_logits 的同一个算法（不带位置编码的那一支），排布用上游的 get_batch。"""
    import torch
    _, U = _upstream()
    p2i = U.get_p2i(data)
    assert len(p2i) == len(hist_len)
    out = np.zeros((len(hist_len), len(tok_post)))
    order = np.argsort(hist_len, kind="stable")
    with torch.no_grad():
        for s0 in range(0, len(order), batch_size):
            ix = order[s0: s0 + batch_size].tolist()
            x, a, y, b = U.get_batch(ix, data, p2i, select="left", block_size=block_size, device="cpu", padding="regular",
                                     lifestyle_augmentations=False, no_event_token_rate=5, cut_batch=True)
            assert ((x > 1).sum(1).numpy() == hist_len[ix]).all(), "有站点的历史被窗口裁掉了"
            is_q = (x == 1) & (a == torch.from_numpy(q_age[ix].astype(np.float32))[:, None])
            assert (is_q.sum(1) >= 1).all(), "找不到查询 token"
            qi = torch.argmax(is_q.int(), dim=1)
            lg = model(x.to(device), a.to(device), y.to(device), b.to(device))[0]
            out[ix] = lg[torch.arange(len(ix), device=lg.device), qi.to(lg.device)][:, tok_post].float().cpu().numpy()
    return out


def score_records(ckpt, arr, lm, device, tokens):
    from score_delphi import pseudo_records
    model, ck = load(ckpt, device)
    data, hist_len = pseudo_records(arr, lm)
    post = post_id(label_names("rosmap"))
    lg = query_logits(model, data, hist_len, lm.t.to_numpy() + 1, ck["model_args"]["block_size"], [post[n] for n in tokens], device)
    return lg, hist_len, {"iter_num": int(ck["iter_num"]), "best_val_loss": float(ck["best_val_loss"])}


# ============================================================================== 打分


def score():
    import torch
    import bl_data as B
    from score_delphi_trans import READ_TOKENS
    from transitions import TRANS, TRUTH_HORIZONS, risk_col
    from sklearn.metrics import roc_auc_score
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"code": open(os.path.join(UP, "UPSTREAM_COMMIT")).read().strip(), "rosmap": {}, "nacc": {}}

    # ROSMAP val：统一评估的站点和真值；第一、三部分的打分表写成 zoo 的格式（nacc_compare.py 读）
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    arr = load_bin("rosmap", "val")
    for s in SEEDS:
        ck = os.path.join(UP, dname("rosmap", s), "ckpt.pt")
        if not os.path.exists(ck):
            continue
        lg, hl, info = score_records(ck, arr, truth[["pid", "t"]], dev, READ_TOKENS)
        L = {n: lg[:, k] for k, n in enumerate(READ_TOKENS)}
        p1 = truth[["pid", "t"]].copy()
        for e, (_, tok, _) in ENDPOINTS.items():
            p1[f"logit_{e}"] = L[tok]
            p1[f"rank_{e}"] = L[tok]
            for H in HORIZONS:
                p1[f"risk_{e}_{H}y"] = -np.expm1(-np.exp(L[tok]) * H * DAYS_PER_YEAR)
        p3 = truth[["pid", "t"]].copy()
        for q, (_, _, toks) in TRANS.items():
            lse = np.logaddexp.reduce(np.stack([L[n] for n in toks], 1), axis=1)
            p3[f"rank_{q}"] = lse
            for H in TRUTH_HORIZONS:
                p3[risk_col(q, H)] = -np.expm1(-np.exp(lse) * H * DAYS_PER_YEAR)
        for d, t in (("scores", p1), ("scores_trans", p3)):
            t.assign(hist_len=hl).to_csv(os.path.join(RUN_DIR, d, f"{NAME['rosmap']}__s{s}.csv"), index=False)
        res["rosmap"][f"s{s}"] = info
        print(f"rosmap s{s}: iter {info['iter_num']}, val loss {info['best_val_loss']:.4f}", flush=True)

    # NACC test：nacc_delphi.nacc_test 的同一批站点、同一套标签
    import pretrain as P
    Dn = P.load_nacc()["test"]
    M = Dn["M"].astype(bool)
    Yv = Dn["Y"].reshape(*M.shape, B.NQ, B.NH)[M]
    lm = pd.DataFrame({"pid": np.repeat(Dn["pid"] + 1, M.sum(1)), "t": Dn["T"][M]})
    arr = np.fromfile(os.path.join(DATA_DIR, "nacc", "val.bin"), np.uint32).reshape(-1, 3)
    Rs = []
    for s in SEEDS:
        ck = os.path.join(UP, dname("nacc", s), "ckpt.pt")
        if not os.path.exists(ck):
            continue
        lg, _, info = score_records(ck, arr, lm, dev, READ_TOKENS)
        R = np.zeros((len(lm), B.NQ, B.NH))
        for qi, (q, (_, _, toks)) in enumerate(TRANS.items()):
            lse = np.logaddexp.reduce(np.stack([lg[:, READ_TOKENS.index(n)] for n in toks], 1), axis=1)
            for hi, H in enumerate(B.HS):
                R[:, qi, hi] = -np.expm1(-np.exp(lse) * H * DAYS_PER_YEAR)
        Rs.append(R)
        res["nacc"][f"s{s}"] = info
    if Rs:
        R = np.mean(Rs, 0)
        cells = {}
        for qi, q in enumerate(B.Q):
            for H in (1, 2, 3, 5, 10):
                y, r = Yv[:, qi, H - 1], R[:, qi, H - 1]
                k = y >= 0
                cells[f"{q}_{H}y"] = float(roc_auc_score(y[k], r[k]))
                if H == 2:
                    cells[f"{q}_2y_oe"] = float(y[k].sum() / r[k].sum())
        res["nacc"]["ensemble"] = {"part1_2y_4endpoints": float(np.mean([cells[f"{q}_2y"] for q in ("ad", "death", "mmse_lt24", "cogn_ltm1")])),
                                   "part3_5y_6questions": float(np.mean([cells[f"{q}_5y"] for q in B.Q])), "cells": cells}
        print("nacc test:", json.dumps({k: round(v, 4) for k, v in res["nacc"]["ensemble"].items() if k != "cells"}), flush=True)
    out = os.path.join(RUN_DIR, "packages", "_nacc", "delphi_upstream.json")
    json.dump(res, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["configs", "score"])
    ap.add_argument("--max_iters", type=int, default=0)
    a = ap.parse_args()
    configs(a.max_iters) if a.what == "configs" else score()
