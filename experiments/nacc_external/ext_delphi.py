"""ext_delphi.py -- delphi-dedup（直接读出）和 delphi-dedup-sim（rollout）在外部验证的记录集上打分。

    python ext_delphi.py direct --set rosmap|H|E          # 直接读出（score_delphi / score_delphi_trans 的同一组函数）
    python ext_delphi.py sim --set rosmap|H|E --part k --n_parts n     # rollout（rollout_landmarks 的同一个生成器和读出）
    python ext_delphi.py sim --set E --merge --n_parts n

检查点：已有的 delphi-dedup 主 seed（registry.DELPHI 的第一个目录，s42），只在 ROSMAP 上训练过。
每个站点一条截到站点为止的伪记录（pid = 站点序号 + 1）：E 的伪记录在 ext_data.py 里就截好了（E_tok.bin，文件里没有站点之后的 token）；
ROSMAP val、H 这里用同样的规则从整条记录截出来，所以三个集合走同一条路径。
--set rosmap 是核对：截断伪记录的打分必须和已发表的打分表（scores_trans/delphi-dedup[-sim]__s42.csv）逐位相同。
rollout 的种子：ROSMAP = crn_seed(真 pid) + 3 + t（和 rollout_landmarks 一样，才能逐位对照）；NACC = crn_seed(站点序号 + 1) + 3 + t。
输出和 ext_score.py 同一种格式：scores/<集>/<名字>.npz（ii = 段内的人、vv = 第几次访视、R = (站点, 6, 10)）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import ext_common as C
import bl_data as B
from common import DAYS_PER_YEAR, DELPHI_DIR, label_names, load_bin, post_id  # noqa: E402
from registry import DELPHI  # noqa: E402
from transitions import TRANS, TRUTH_HORIZONS, risk_col  # noqa: E402

NAME = "delphi-dedup"
REG = {m[0]: m for m in DELPHI}
CKPT = os.path.join(DELPHI_DIR, REG[NAME][1][0], "ckpt.pt")
DATASET = REG[NAME][2]
SEED_OFFSET = 3
assert list(TRANS) == list(B.Q) and list(TRUTH_HORIZONS) == list(B.HS)


def truncate_tokens(arr, pid, t):
    """整条记录 (pid, 年龄, token) -> 每个站点一条截到 t 为止的伪记录，pid = 站点序号 + 1。"""
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    u, first = np.unique(arr[:, 0], return_index=True)
    end = np.r_[first[1:], len(arr)]
    span = {int(p): (f, e) for p, f, e in zip(u, first, end)}
    out = []
    for q, (p, tt) in enumerate(zip(pid, t)):
        f, e = span[int(p)]
        h = arr[f:e]
        h = h[h[:, 1] <= tt]
        out.append(np.column_stack([np.full(len(h), q + 1), h[:, 1:]]))
    return np.concatenate(out).astype(np.int64)


def landmark_set(s):
    """-> (伪记录 (q+1, 年龄, token), 站点表 DataFrame[q, ii, vv, t, pid_real], 种子用的 pid)。只取 primary 的人。"""
    if s == "rosmap":
        tr = pd.read_csv(os.path.join(C.RUN_DIR, "truth_trans_val.csv"), usecols=["pid", "t"])
        Z = np.load(os.path.join(C.RUN_DIR, "newmodel_data_trans.npz"))
        P, T, M = Z["val_pid"], Z["val_T"], Z["val_M"]
        idx = {int(p): i for i, p in enumerate(P)}
        ii = np.array([idx[int(p)] for p in tr.pid])
        vv = np.array([int(np.where(M[i] & (T[i] == t))[0][0]) for i, t in zip(ii, tr.t)])
        arr = truncate_tokens(load_bin(DATASET, "val"), tr.pid.to_numpy(), tr.t.to_numpy())
        lm = pd.DataFrame({"ii": ii, "vv": vv, "t": tr.t.to_numpy(), "pid_real": tr.pid.to_numpy()})
        return arr, lm, lm.pid_real.to_numpy()
    if s == "H":
        with np.load(os.path.join(C.OUT, "data", "H.npz")) as z:
            M, T, prim = z["M"], z["T"], z["primary"]
        ii, vv = np.where(M & prim[:, None])
        full = np.fromfile(os.path.join(C.OUT, "data", "H_tok.bin"), np.uint32).reshape(-1, 3).astype(np.int64)
        arr = truncate_tokens(full, ii + 1, T[ii, vv])                  # H_tok 的 pid = H.npz 的人 + 1
        lm = pd.DataFrame({"ii": ii, "vv": vv, "t": T[ii, vv]})
        return arr, lm, np.arange(len(lm)) + 1
    # E：伪记录已经截好；只读特征文件里的 primary 标记
    with np.load(os.path.join(C.OUT, "data", "E_X.npz")) as z:
        assert not any(k.startswith("Y") for k in z.files)
        prim = z["primary"]
    L = np.load(os.path.join(C.OUT, "data", "E_tok_lm.npz"))
    full = np.fromfile(os.path.join(C.OUT, "data", "E_tok.bin"), np.uint32).reshape(-1, 3).astype(np.int64)
    keep = prim[L["pid"]]
    q_keep = L["q"][keep]
    sel = np.isin(full[:, 0] - 1, q_keep)
    arr = full[sel]
    remap = np.full(len(L["q"]), -1)
    remap[q_keep] = np.arange(len(q_keep))
    arr[:, 0] = remap[arr[:, 0] - 1] + 1
    lm = pd.DataFrame({"ii": L["pid"][keep], "vv": L["v"][keep], "t": L["t"][keep]})
    assert (arr[:, 1] <= lm.t.to_numpy()[arr[:, 0] - 1]).all()
    return arr, lm, np.arange(len(lm)) + 1


def direct(s, od):
    import torch
    from score_delphi import load_model, pos_align_length, pseudo_records, query_logits
    from score_delphi_trans import READ_TOKENS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arr, lm, _ = landmark_set(s)
    names = label_names(DATASET)
    post = post_id(names)
    model, ck = load_model(CKPT, dev)
    cfg = ck.get("config", {})
    block_size = ck["model_args"]["block_size"]
    lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
    qtab = pd.DataFrame({"pid": np.arange(len(lm)) + 1, "t": lm.t.to_numpy()})
    data, hist_len = pseudo_records(arr, qtab)
    pos_len = None
    if ck["model_args"].get("pos_embedding", False):
        pos_len, _ = pos_align_length(DATASET, cfg, block_size, lr)
    logit = query_logits(model, data, hist_len, qtab.t.to_numpy() + 1, block_size, [post[n] for n in READ_TOKENS], dev,
                         lifestyle_range=lr, pos_len=pos_len).astype(np.float64)
    R = np.zeros((len(lm), B.NQ, B.NH))
    for qi, (q, (_, _, toks)) in enumerate(TRANS.items()):
        lse = np.logaddexp.reduce(logit[:, [READ_TOKENS.index(n) for n in toks]], axis=1)
        for hi, H in enumerate(TRUTH_HORIZONS):
            R[:, qi, hi] = -np.expm1(-np.exp(lse) * H * DAYS_PER_YEAR)
    return lm, R


_S = {}


def _init(ckpt, stream_path):
    import rollout_trans as RT
    from radc_delphi.batching import get_p2i, patient_stream
    RT._init_worker(ckpt, DATASET)
    # 引擎找不到经验访视大小（visit_sizes.npy）会悄悄改成逐 token 采样（2026-11-01 重组后就是这样，和已发表的 rollout 不同）
    assert RT._W["eng"].visit_sizes_path is not None, "Delphi 引擎没找到 visit_sizes.npy"
    a = np.load(stream_path)["arr"]
    p2i = get_p2i(a)
    RT._W["streams"] = {int(a[int(s), 0]): patient_stream(a, int(s), int(n)) for s, n in p2i[:, :2]}


def sim(s, od, part, n_parts, workers):
    import multiprocessing as mp
    import rollout_trans as RT
    from run_rollout_delphi import crn_seed
    arr, lm, seed_pid = landmark_set(s)
    idx = np.array_split(np.arange(len(lm)), n_parts)[part]
    q1 = idx + 1                                                        # 伪记录的 pid
    sp = os.path.join(od, f"_streams_{NAME}-sim_part{part}of{n_parts}.npz")
    np.savez(sp, arr=arr[np.isin(arr[:, 0], q1)])
    tt = lm.t.to_numpy()
    jobs = [(int(q1[k]), int(tt[j]), RT.N_MC, (crn_seed(seed_pid[j]) + SEED_OFFSET + int(tt[j])) % 2_147_483_647)
            for k, j in enumerate(idx)]
    cum = np.zeros((len(idx), B.NQ, B.NH))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(CKPT, sp)) as pool:
        for k, (_, r) in enumerate(pool.imap(RT._run_one, jobs, chunksize=1)):
            cum[k] = r["cum"]
            if (k + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {k + 1}/{len(jobs)}  {el / 60:.1f} min  ~{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min left", flush=True)
    os.remove(sp)
    return lm.iloc[idx].reset_index(drop=True), cum / float(RT.N_MC), idx


def published(name, lm):
    from transitions import label_col  # noqa: F401
    tab = pd.read_csv(os.path.join(C.RUN_DIR, "scores_trans", f"{name}__s42.csv")).set_index(["pid", "t"])
    cols = [risk_col(q, H) for q in B.Q for H in B.HS]
    return tab.loc[list(zip(lm.pid_real.tolist(), lm.t.tolist())), cols].to_numpy().reshape(-1, B.NQ, B.NH)


def save(od, name, lm, R, extra=None):
    pid_seg = lm.ii.to_numpy()
    path = os.path.join(od, f"{name}.npz")
    np.savez_compressed(path, ii=pid_seg, vv=lm.vv.to_numpy(), pid=pid_seg, t=lm.t.to_numpy(), R=R,
                        model_sha256=C.sha256(CKPT), **(extra or {}))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["direct", "sim"])
    ap.add_argument("--set", required=True, choices=["rosmap", "H", "E"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n_parts", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = ap.parse_args()
    od = os.path.join(C.SCORE_DIR, args.set)
    os.makedirs(od, exist_ok=True)
    name = NAME if args.what == "direct" else f"{NAME}-sim"
    t0 = time.time()
    if args.what == "direct":
        lm, R = direct(args.set, od)
        path = save(od, name, lm, np.clip(R, 1e-6, 1 - 1e-6))
    elif args.merge:
        parts = [np.load(os.path.join(od, f"{name}.part{k}of{args.n_parts}.npz")) for k in range(args.n_parts)]
        out = {k: np.concatenate([p[k] for p in parts]) for k in ("ii", "vv", "pid", "t", "R")}
        path = os.path.join(od, f"{name}.npz")
        np.savez_compressed(path, **out, model_sha256=parts[0]["model_sha256"])
        print(f"merged -> {path} ({len(out['ii'])})  sha256 {C.sha256(path)}")
        lm = pd.DataFrame({"ii": out["ii"], "vv": out["vv"], "t": out["t"]})
        R = out["R"]
    else:
        lm, R, _ = sim(args.set, od, args.part, args.n_parts, args.workers)
        R = np.clip(R, 1e-6, 1 - 1e-6)
        if args.n_parts == 1:
            path = save(od, name, lm, R)
        else:
            path = os.path.join(od, f"{name}.part{args.part}of{args.n_parts}.npz")
            np.savez_compressed(path, ii=lm.ii.to_numpy(), vv=lm.vv.to_numpy(), pid=lm.ii.to_numpy(), t=lm.t.to_numpy(),
                                R=R, model_sha256=C.sha256(CKPT))
    rep = {"name": name, "set": args.set, "ckpt": CKPT, "n": int(len(lm)), "minutes": (time.time() - t0) / 60, "out": path}
    if args.set == "rosmap" and "pid_real" in lm and (args.what == "direct" or args.n_parts == 1):
        P = published(name, lm)
        d = np.abs(R - P)
        rep["vs_published"] = {"identical_1e-9": bool(d.max() < 1e-9), "max_abs_diff": float(d.max())}
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(os.path.join(od, f"{name}.info.json" if not args.n_parts > 1 or args.merge else f"{name}.part{args.part}.info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
