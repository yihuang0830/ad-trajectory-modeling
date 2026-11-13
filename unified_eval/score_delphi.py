"""score_delphi.py -- Delphi 家族的答题适配器：ckpt + 它自己那份分词 -> 统一风险表。

    python score_delphi.py --ckpt .../Delphi-ROSMAP-nodedup/ckpt.pt --dataset rosmap_nodedup \
                           --out $ZOO_RUN_DIR/scores/delphi-nodedup__s42.csv

每个站点 (pid, t)：取这个人在**该模型自己的 val.bin** 里年龄 <= t 的全部 token，在 t+1 天接一个
no-event 查询 token、t+2 天再接一个哑目标，拼成一个"在这里结束的记录"，交给上游
`utils.get_batch` 按训练时的规矩排布（右对齐、no-event 插值、稳定排序、+1 位移），再前向一次，
读查询位置的 logits。

  * 为什么用 get_batch 而不是手搓：pos_embedding 的模型对排布敏感。训练时 get_batch(cut_batch=True)
    把每个 batch 右对齐到**这一批最长的那条**（不是 block_size），所以每条记录的最后一个输入 token
    落在 L_batch-1 号位置，L_batch 是这一批的长度（block 144）。block 末尾那几格
    在训练里几乎从来没被占过。这里对 pos 模型模拟训练 batch 的长度分布（pos_align_length），
    把伪记录右对齐到 max(自身长度, 中位数)，查询 token 就落在训练里最常见的"记录末尾"那一格。
    只喂到站点为止的记录 -> 位置不会把未来的长度漏给模型。不带位置编码的模型和排布无关（已验证逐位相同）。
  * 为什么要哑目标：get_batch 的 x = tokens[:-1]，窗口里最后一个 token 只当目标。哑目标在
    t+2 天，查询 token 的 mask_ties 只遮"和目标同龄"的 key，没有任何 key 在 t+2 天，
    所以查询 token 看得到站点那次访视的全部 token。
  * padding='regular'：no-event 插在固定年龄（每 5 年），所有模型、所有站点一样，没有随机性。
  * lifestyle_augmentations=False：背景块放在真实年龄上（训练时的随机偏移只是增广）。

风险 = 1 - exp(-exp(logit) * H * 365.25)（用 -expm1 算，速率很小时不丢精度）：Delphi 的 logit 是每天的速率（loss_dt 的 dt 以天计）。
visit_heads 的模型 logit 不再是速率（见 model.py），AUC 照算（只看排序），校准不可解释。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DELPHI_DIR, DAYS_PER_YEAR, ENDPOINTS, HORIZONS, RUN_DIR, label_names, load_bin, post_id  # noqa: E402

sys.path.insert(0, DELPHI_DIR)
from model import Delphi, DelphiConfig  # noqa: E402
from utils import get_batch, get_p2i  # noqa: E402

NO_EVENT_POST = 1


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**ck["model_args"]))
    model.load_state_dict(ck["model"])
    return model.eval().to(device), ck


def pseudo_records(arr, landmarks):
    """每个站点一条"截到站点为止"的伪记录。pid 换成站点序号 j（相邻不同 -> get_p2i 分得开）。"""
    order = np.argsort(arr[:, 0], kind="stable")          # val.bin 本来就按人连续、人内按年龄排
    arr = arr[order]
    starts = {}
    pids, first = np.unique(arr[:, 0], return_index=True)
    counts = np.diff(np.append(first, len(arr)))
    for p, f, c in zip(pids, first, counts):
        starts[int(p)] = (f, c)
    chunks, hist_len = [], np.zeros(len(landmarks), np.int64)
    for j, (pid, t) in enumerate(zip(landmarks.pid.to_numpy(), landmarks.t.to_numpy())):
        f, c = starts[int(pid)]
        h = arr[f: f + c]
        assert (np.diff(h[:, 1]) >= 0).all(), f"pid {pid} 的 token 不是按年龄排的"
        h = h[h[:, 1] <= t]
        rec = np.empty((len(h) + 2, 3), np.int64)
        rec[:, 0] = j
        rec[: len(h), 1:] = h[:, 1:]
        rec[len(h)] = (j, t + 1, 0)       # 查询：pre-shift 0 = No event
        rec[len(h) + 1] = (j, t + 2, 0)   # 哑目标
        chunks.append(rec)
        hist_len[j] = len(h)
    return np.concatenate(chunks), hist_len


def pos_align_length(dataset, cfg, block_size, lifestyle_range, q=50, n_batches=200, seed=0):
    """模拟训练时的 get_batch（padding='random'、lifestyle 增广、cut_batch=True、batch_size 同训练），
    返回每个 batch 输入 x 长度的第 q 百分位。pos 模型按它对齐，位置下标才落在训练用过的范围里。"""
    train = load_bin(dataset, "train")
    p2i = get_p2i(train)
    g = torch.Generator().manual_seed(seed)
    bs = int(cfg.get("batch_size", 128))
    L = []
    for _ in range(n_batches):
        ix = torch.randint(len(p2i), (bs,), generator=g)
        x, _a, _y, _b = get_batch(ix, train, p2i, select="left", block_size=block_size, device="cpu",
                                  padding="random", lifestyle_augmentations=True,
                                  lifestyle_token_range=lifestyle_range,
                                  no_event_token_rate=int(cfg.get("no_event_token_rate", 5)), cut_batch=True)
        L.append(x.shape[1])
    return int(np.percentile(L, q)), [int(np.percentile(L, k)) for k in (5, 50, 95)]


def align_right(x, a, y, b, Lx):
    """把 get_batch(cut_batch=False) 的结果（右对齐到 block_size）裁成右对齐到 Lx：只裁左侧整列 padding。"""
    n_tok = (x != 0).sum(1)
    Lx = max(int(Lx), int(n_tok.max()))
    drop = x.shape[1] - Lx
    if drop > 0:
        assert (x[:, :drop] == 0).all(), "裁掉的列里有真实 token"
        x, a, y, b = x[:, drop:], a[:, drop:], y[:, drop:], b[:, drop:]
    return x, a, y, b


@torch.no_grad()
def query_logits(model, data, hist_len, q_age, block_size, tok_post, device, batch_size=256,
                 lifestyle_range=(3, 32), pos_len=None):
    p2i = get_p2i(data)
    n = len(hist_len)
    assert len(p2i) == n, (len(p2i), n)
    assert (hist_len + 2 <= block_size + 1).all(), \
        f"最长的伪记录 {hist_len.max() + 2} 行放不进 block_size+1={block_size + 1} 的窗口"
    out = np.zeros((n, len(tok_post)), np.float64)
    n_multi = 0
    # 按历史长度排序后分批：不带位置编码的模型可以裁掉整批共有的左侧 padding（padding 在注意力里
    # 被遮掉、又没有位置下标，裁不裁输出逐位相同）；带位置编码的必须保持 block_size 的完整排布。
    # model.forward 会把每层注意力图都存下来 (n_layer, b, heads, t, t)，block 736 时批量要小。
    pos = bool(getattr(model.config, "pos_embedding", False))
    order = np.argsort(hist_len, kind="stable")
    cfg = model.config
    s0 = 0
    while s0 < n:
        t_est = max(pos_len or 0, int(hist_len[order[min(n - 1, s0 + batch_size - 1)]]) + 40)
        bs = int(np.clip(1.5e8 / (cfg.n_layer * cfg.n_head * t_est * t_est), 4, batch_size))
        ix = order[s0: s0 + bs].tolist()
        s0 += bs
        x, a, y, b = get_batch(ix, data, p2i, select="left", block_size=block_size, device="cpu",
                               padding="regular", no_event_token_rate=5, cut_batch=not pos,
                               lifestyle_augmentations=False, lifestyle_token_range=lifestyle_range)
        if pos:
            assert pos_len is not None, "pos 模型必须给对齐长度（pos_align_length）"
        # 历史没被裁掉：x 里真实 token（post-shift > 1）的个数 = 历史行数
        n_real = (x > NO_EVENT_POST).sum(1).numpy()
        assert (n_real == hist_len[ix]).all(), "有站点的历史被窗口裁掉了"
        qa = torch.from_numpy(q_age[ix].astype(np.float32))
        is_q = (x == NO_EVENT_POST) & (a == qa[:, None])
        cnt = is_q.sum(1)
        assert (cnt >= 1).all(), "找不到查询 token"
        n_multi += int((cnt > 1).sum())
        qi = torch.argmax(is_q.int(), dim=1)                # 第一个匹配 = 我们放进去的那一行（稳定排序）
        if pos:
            # 逐行对齐到 max(pos_len, 自身长度)：同一个 Lr 的行一起前向。结果不依赖分批方式。
            n_tok = (x != 0).sum(1)
            Lr = torch.clamp(n_tok, min=pos_len)
            for L in torch.unique(Lr).tolist():
                rows = torch.where(Lr == L)[0]
                st = x.shape[1] - L
                assert (x[rows, :st] == 0).all(), "裁掉的列里有真实 token"
                lg = model(*(v[rows, st:].to(device) for v in (x, a, y, b)))[0]
                lq = lg[torch.arange(len(rows), device=lg.device), (qi[rows] - st).to(lg.device)]
                out[[ix[r] for r in rows.tolist()]] = lq[:, tok_post].float().cpu().numpy()
            continue
        logits = model(x.to(device), a.to(device), y.to(device), b.to(device))[0]
        lq = logits[torch.arange(len(ix), device=logits.device), qi.to(logits.device)]
        out[ix] = lq[:, tok_post].float().cpu().numpy()
    if n_multi:
        print(f"  注意：{n_multi} 个站点的查询年龄恰好撞上 regular no-event（取第一个）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--truth", default=None, help="默认 $ZOO_RUN_DIR/truth_<split>.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pos_q", type=int, default=50, help="pos 模型的对齐长度取训练 batch 长度的第几百分位")
    args = ap.parse_args()

    truth = pd.read_csv(args.truth or os.path.join(RUN_DIR, f"truth_{args.split}.csv"))
    names = label_names(args.dataset)
    assert names == label_names("rosmap"), "这份数据的词表和交付版不同"
    post = post_id(names)
    tok_names = [ENDPOINTS[e][1] for e in ENDPOINTS]
    tok_post = [post[n] for n in tok_names]

    model, ck = load_model(args.ckpt, args.device)
    cfg = ck.get("config", {})
    assert cfg.get("dataset", args.dataset) == args.dataset, \
        f"ckpt 是在 {cfg.get('dataset')} 上训的，这里给的是 {args.dataset}"
    block_size = ck["model_args"]["block_size"]
    lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))

    arr = load_bin(args.dataset, args.split)
    data, hist_len = pseudo_records(arr, truth)
    q_age = truth.t.to_numpy() + 1
    pos_len, pos_pct = (None, None)
    if ck["model_args"].get("pos_embedding", False):
        pos_len, pos_pct = pos_align_length(args.dataset, cfg, block_size, lr, q=args.pos_q)
        print(f"  pos_embedding：训练 batch 的 x 长度 p5/p50/p95 = {pos_pct}，按第 {args.pos_q} 百分位 {pos_len} 对齐")
    logit = query_logits(model, data, hist_len, q_age, block_size, tok_post, args.device,
                         lifestyle_range=lr, pos_len=pos_len)

    res = truth[["pid", "t"]].copy()
    rate = np.exp(logit)                                     # 每天
    for k, e in enumerate(ENDPOINTS):
        res[f"logit_{e}"] = logit[:, k]
        res[f"rank_{e}"] = logit[:, k]                       # 排序用（常数速率假设下各 H 的排序相同）
        for H in HORIZONS:
            res[f"risk_{e}_{H}y"] = -np.expm1(-rate[:, k] * H * DAYS_PER_YEAR)
    res["hist_len"] = hist_len
    res["pos_align_len"] = -1 if pos_len is None else pos_len
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"scored {len(res)} landmarks  ckpt={args.ckpt}  iter={ck.get('iter_num')}  "
          f"best_val={ck.get('best_val_loss')}  -> {args.out}")


if __name__ == "__main__":
    main()
