"""nacc_scale.py -- ROSMAP 上"最简单的模型最好"是不是因为 ROSMAP 太小？同一组模型换到 NACC 上、按训练量重跑一遍（2026-10-28）。

设计（README.md 有完整说明）：
  数据   NACC（nacc_data.py：和 ROSMAP 逐位同一种输入格式、同一套 6 题 × 10 年标签；Delphi 用 nacc_delphi.py 的分词，同一个词表）。
         NACC 的 train 段按人随机留 10% 做调参 / 早停（tune），其余的人按固定顺序取嵌套子集当训练集：
           rosmap = 和 ROSMAP 的训练人数一样、12k = 12,000 人、full = 全部。
         三档都在同一个 NACC test 段上报数；tune 在三档之间也一样。
  模型   和 ROSMAP 上同一份代码、同一组超参 / 调参网格，只换数据：
           表格：lr-cur、lr-l2c、dt-l2c（单棵决策树，这里新加）、rf-l2c、xgb-l2c、lgbm-l2c（tabular.py；网格在 tune 上挑）
           网络：transformer、grud、tlstm（pretrain.py 的构造 / 训练循环，ROSMAP 的超参，在 tune 上早停；3 个 seed 集成）
           Delphi：上游原版代码 + 配置（delphi_upstream.py）、zoo 的 nojitter 版（nacc_delphi.py 的配置）；3 个 seed 集成
  指标   和表 5 一样：4 个终点（AD、死亡、MMSE<24、认知<−1）2 年 AUC 的平均；6 道题 5 年 AUC 的平均。
         每个模型减 lr-cur（同一档训练量），按人配对 bootstrap。ROSMAP val 上同一组模型、同一个指标（现成的打分表；dt-l2c 在这里补训）。

    python nacc_scale.py prep            # 划分 + Delphi 的数据（delphi/data/nacc_scale_<档>/）
    python nacc_scale.py work --budget 27   # 一个工人：一直领还没做的单元做到预算用完（ris/worker.sbatch；多个工人并行、可续）
    python nacc_scale.py status          # 还剩哪些单元
    python nacc_scale.py collect         # 汇总 -> $ZOO_RUN_DIR/packages/_nacc_scale/results.json（聚合，可以拉回本地）

人级的东西（划分、每个单元的风险）只留在 RIS 的 $ZOO_RUN_DIR/nacc_scale/。
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ZOO, "models_code"))
sys.path.insert(0, os.path.join(ZOO, "unified_eval"))
import bl_data as B  # noqa: E402
from common import AD_BASE, DATA_DIR, DELPHI_DIR, RUN_DIR  # noqa: E402

OUT = os.path.join(RUN_DIR, "nacc_scale")
RES = os.path.join(RUN_DIR, "packages", "_nacc_scale")
UP = os.path.join(AD_BASE, "delphi_upstream")
SEED = 20261028
SIZES = ("rosmap", "12k", "full")
N_12K = 12_000
NET_SEEDS = (0, 1, 2)
DELPHI_SEEDS = (42, 43, 44)
N_BOOT = 500

TAB = {"lr-cur": ("lr", "cur"), "lr-l2c": ("lr", "l2c"), "dt-l2c": ("dt", "l2c"), "rf-l2c": ("rf", "l2c"),
       "xgb-l2c": ("xgb", "l2c"), "lgbm-l2c": ("lgbm", "l2c")}
DT_GRID = [{"depth": d} for d in (4, 6, 8)]                 # 单棵树：深度在 tune 上挑，叶子至少 50 个站点
NETS = {"transformer": ("seq", "transformer"), "grud": ("seq", "grud"), "tlstm": ("seq", "tlstm")}
DELPHI = ("delphi-upstream", "delphi-zoo")
# ROSMAP val 上同一组模型用哪份现成的打分表（transformer / grud / tlstm 是 5 seed 集成的那一份）
ROSMAP_SRC = {"lr-cur": "lr-cur", "lr-l2c": "lr-l2c", "rf-l2c": "rf-l2c", "xgb-l2c": "xgb-l2c", "lgbm-l2c": "lgbm-l2c",
              "transformer": "transformer-x5", "grud": "grud-x5",
              "tlstm": "tlstm-x5", "delphi-upstream": "delphi-upstream", "delphi-zoo": "delphi-dedup-nojitter"}
P1 = ("ad", "death", "mmse_lt24", "cogn_ltm1")


def grid(kind):
    import tabular as TB
    return DT_GRID if kind == "dt" else TB.CLF_GRID[kind]


def ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


# ============================================================================== 数据


_C = {}


def nacc():
    if not _C:
        import pretrain as P
        _C["D"] = P.load_nacc()
        _C["sp"] = dict(np.load(os.path.join(OUT, "split.npz")))
    return _C["D"], _C["sp"]


def sub(D, idx):
    return {k: v[idx] for k, v in D.items()}


def parts(size):
    """(训练, tune, test) 三段的人级数组。"""
    D, sp = nacc()
    return sub(D["train"], sp[size]), sub(D["train"], sp["hold"]), D["test"]


def lms(size, feat):
    k = (size, feat)
    if k not in _C:
        _C[k] = tuple(B.landmarks(d, feat) for d in parts(size))
    return _C[k]


def prep():
    """划分（固定种子）+ Delphi 的三档数据：NACC 的 train.bin 按人过滤，tune 的人当 val.bin（早停用）。"""
    import pretrain as P
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    D = P.load_nacc()["train"]
    n = len(D["pid"])
    rng = np.random.default_rng(SEED)
    hold = np.where(rng.random(n) < 0.1)[0]
    rest = rng.permutation(np.setdiff1d(np.arange(n), hold))
    n_ros = len(B.load()[0]["train_pid"])
    sp = {"hold": hold, "rosmap": np.sort(rest[:n_ros]), "12k": np.sort(rest[:N_12K]), "full": np.sort(rest)}
    np.savez(os.path.join(OUT, "split.npz"), **sp)
    summ = {k: int(len(v)) for k, v in sp.items()}
    summ["n_landmarks"] = {k: int(D["M"][v].sum()) for k, v in sp.items()}
    print("split:", summ, flush=True)
    src = os.path.join(DATA_DIR, "nacc")
    arr = np.fromfile(os.path.join(src, "train.bin"), np.uint32).reshape(-1, 3)
    dpid = arr[:, 0].astype(np.int64)
    assert set(np.unique(dpid)) == set(D["pid"] + 1), "delphi/data/nacc/train.bin 的人和 nacc_data 的 train 段对不上"
    val = arr[np.isin(dpid, D["pid"][hold] + 1)]
    summ["delphi_rows"] = {}
    for size in SIZES:
        d = os.path.join(DATA_DIR, f"nacc_scale_{size}")
        os.makedirs(d, exist_ok=True)
        tr = arr[np.isin(dpid, D["pid"][sp[size]] + 1)]
        tr.tofile(os.path.join(d, "train.bin"))
        val.tofile(os.path.join(d, "val.bin"))
        for f in ("labels.csv", "meta.json"):
            shutil.copy(os.path.join(src, f), os.path.join(d, f))
        summ["delphi_rows"][size] = {"train": int(len(tr)), "val": int(len(val)),
                                     "train_persons": int(len(np.unique(tr[:, 0]))), "val_persons": int(len(np.unique(val[:, 0])))}
        print(f"-> {d}: {summ['delphi_rows'][size]}", flush=True)
    json.dump(summ, open(os.path.join(RES, "split_summary.json"), "w"), indent=1)


# ============================================================================== 单元


def units():
    """按优先级排好的全部单元：(种类, 路径, 参数)。先做 rosmap 和 full 两档，最后补 12k。"""
    U = []
    order = [("rosmap", "full"), ("12k",)]
    for sizes in order:
        for name in ("lr-cur",):
            U += [("tab", name, s, h, q) for s in sizes for h in range(len(grid(TAB[name][0]))) for q in range(B.NQ)]
        if sizes[0] == "rosmap":
            U += [("rdt", "dt-l2c", "rosmap-val", h, q) for h in range(len(DT_GRID)) for q in range(B.NQ)]
        U += [("net", name, s, seed) for name in ("transformer", "grud", "tlstm")
              for s in sizes for seed in NET_SEEDS]
        for name in ("xgb-l2c", "lgbm-l2c", "dt-l2c", "lr-l2c", "rf-l2c"):
            U += [("tab", name, s, h, q) for s in sizes for h in range(len(grid(TAB[name][0]))) for q in range(B.NQ)]
        U += [("dscore", v, s, seed) for v in DELPHI for s in sizes for seed in DELPHI_SEEDS]
    return U


def upath(u):
    kind, name, size, *rest = u
    if kind in ("tab", "rdt"):
        return os.path.join(OUT, kind, f"{name}__{size}__h{rest[0]}__q{rest[1]}.npz")
    return os.path.join(OUT, kind, f"{name}__{size}__s{rest[0]}.npz")


def delphi_dir(variant, size, seed):
    base = (os.path.join(UP, f"NS-{size}") if variant == "delphi-upstream" else os.path.join(DELPHI_DIR, f"NS-zoo-{size}"))
    return base if seed == 42 else f"{base}-s{seed}"


def ready(u):
    """Delphi 的打分要等 GPU 上的训练跑完（ris/delphi.sbatch 跑完会写 DONE）。"""
    return u[0] != "dscore" or os.path.exists(os.path.join(delphi_dir(u[1], u[2], u[3]), "DONE"))


def _fit(kind, hp, inner, Ftr, ytr, Ftu, ytu, Fs):
    if kind == "dt":
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(max_depth=hp["depth"], min_samples_leaf=50, random_state=0).fit(Ftr, ytr)
        return [m.predict_proba(F)[:, 1] for F in Fs]
    import tabular as TB
    return TB._fit_one(kind, hp, 0, inner, Ftr, ytr, Ftu, ytu, Fs)


def run_tab(kind, hp, qi, tr, tu, te):
    """tabular.per_target 的同一个算法，只做一道题的 10 个时间窗：训练段上标准化，tune 段早停（树）。"""
    from joblib import Parallel, delayed
    import tabular as TB
    prep_ = TB.Prep(tr["F"])
    Ftr, Ftu, Fte = prep_(tr["F"]), prep_(tu["F"]), prep_(te["F"])
    n = ncpu()
    outer, inner = (min(n, B.NH), 1) if kind in ("lr", "dt") else (max(1, n // 4), 4)
    jobs = []
    for hi in range(B.NH):
        a, b = tr["Y"][:, qi, hi], tu["Y"][:, qi, hi]
        ka, kb = a >= 0, b >= 0
        jobs.append((Ftr[ka], a[ka], Ftu[kb], b[kb]))
    res = Parallel(n_jobs=outer)(delayed(_fit)(kind, hp, inner, a, ya, b, yb, (Ftu, Fte)) for a, ya, b, yb in jobs)
    return np.stack([r[0] for r in res], 1).astype(np.float32), np.stack([r[1] for r in res], 1).astype(np.float32)


def do_unit(u, lock):
    import pretrain as P
    path = upath(u)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    tmp = f"{path}.{ME}.tmp.npz"                                    # 每个作业自己的临时文件：两个工人撞上同一个单元也不会写坏
    kind = u[0]
    if kind == "tab":
        _, name, size, hi, qi = u
        k, feat = TAB[name]
        tr, tu, te = lms(size, feat)
        Rtu, Rte = run_tab(k, grid(k)[hi], qi, tr, tu, te)
        np.savez(tmp, tu=Rtu, te=Rte)
    elif kind == "rdt":
        _, name, _, hi, qi = u
        if "ros" not in _C:
            Z, _ = B.load()
            _C["ros"] = tuple(B.landmarks(B.split(Z, s), "l2c") for s in ("train", "test", "val"))
        tr, tu, va = _C["ros"]
        Rtu, Rva = run_tab("dt", DT_GRID[hi], qi, tr, tu, va)
        np.savez(tmp, tu=Rtu, te=Rva)
    elif kind == "net":
        import torch
        _, name, size, seed = u
        fam, key = NETS[name]
        torch.set_num_threads(ncpu())
        trD, tuD, teD = parts(size)
        tr, tu, te = (P.tensors(fam, D, "cpu") for D in (trD, tuD, teD))
        torch.manual_seed(seed)
        model = P.build(fam, key, P.x_mean_of(trD))
        lr, wd, bs, epochs, patience, cosine = P.hp_of(fam, key)
        info = P.train_loop(fam, model, tr, tu, lr, wd, bs, epochs, patience, seed, cosine, bucket=True,
                            partial=path + ".partial", lock=lock)
        R = P.risk_of(fam, model, te)[teD["M"].astype(bool)]
        info.update(n_params=sum(q.numel() for q in model.parameters()), bs=bs, lr=lr)
        np.savez(tmp, te=R.astype(np.float32), info=json.dumps(info))
    elif kind == "dscore":
        _, variant, size, seed = u
        R, info = delphi_risk(variant, os.path.join(delphi_dir(variant, size, seed), "ckpt.pt"))
        np.savez(tmp, te=R.astype(np.float32), info=json.dumps(info))
    os.replace(tmp, path)
    if os.path.exists(path + ".partial"):
        os.remove(path + ".partial")
    print(f"  done {u} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def delphi_risk(variant, ck):
    """NACC test 段上的 (站点, 6, 10) 风险：和 nacc_delphi.nacc_test / delphi_upstream.score 同一个读法。"""
    import torch
    from transitions import TRANS
    from score_delphi_trans import READ_TOKENS
    from common import DAYS_PER_YEAR, label_names, post_id
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(ncpu())
    D, _ = nacc()
    Dn = D["test"]
    M = Dn["M"].astype(bool)
    lm = pd.DataFrame({"pid": np.repeat(Dn["pid"] + 1, M.sum(1)), "t": Dn["T"][M]})
    arr = np.fromfile(os.path.join(DATA_DIR, "nacc", "val.bin"), np.uint32).reshape(-1, 3)
    if variant == "delphi-upstream":
        import delphi_upstream as DU
        lg, _, info = DU.score_records(ck, arr, lm, dev, READ_TOKENS)
    else:
        from score_delphi import load_model, pseudo_records, query_logits
        model, c = load_model(ck, dev)
        cfg = c["config"]
        post = post_id(label_names("nacc"))
        data, hist_len = pseudo_records(arr, lm)
        lg = query_logits(model, data, hist_len, lm.t.to_numpy() + 1, c["model_args"]["block_size"],
                          [post[n] for n in READ_TOKENS], dev,
                          lifestyle_range=(cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32)))
        info = {"iter_num": int(c["iter_num"]), "best_val_loss": float(c["best_val_loss"])}
    R = np.zeros((len(lm), B.NQ, B.NH))
    for qi, (q, (_, _, toks)) in enumerate(TRANS.items()):
        lse = np.logaddexp.reduce(np.stack([lg[:, READ_TOKENS.index(n)] for n in toks], 1), axis=1)
        for hi, H in enumerate(B.HS):
            R[:, qi, hi] = -np.expm1(-np.exp(lse) * H * DAYS_PER_YEAR)
    log = os.path.join(OUT, "delphi_logs", f"{variant}_{os.path.basename(os.path.dirname(ck))}.log")
    if os.path.exists(log):                                        # 训练是不是已经过了 val 的最低点
        steps = [ln.split() for ln in open(log) if ln.startswith("step ")]
        if steps:
            info["last_step"] = int(steps[-1][1].rstrip(":"))
    return R, info


ME = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")


def _holder(lock):
    try:
        return open(lock).read().strip()
    except FileNotFoundError:
        return None


def release(lock):
    if _holder(lock) == ME:
        os.remove(lock)


def claim(P, lock, u):
    """pretrain._claim 外面再包两层：两个工人同时接手同一把死锁时，等 1 秒再读，锁里是谁就归谁；
    拿到锁时单元已经做完（别的工人刚做完、刚删锁，我的待办表是旧的）就放掉。"""
    try:
        if not P._claim(lock):
            return False
    except FileNotFoundError:                                      # 别的工人刚删锁
        return False
    time.sleep(1)
    if _holder(lock) != ME:
        return False
    if os.path.exists(upath(u)):
        release(lock)
        return False
    return True


def work(budget):
    import pretrain as P
    t0 = time.time()
    while True:
        left = budget - (time.time() - t0) / 60
        todo = [u for u in units() if not os.path.exists(upath(u))]
        if not todo:
            print("ALL DONE", flush=True)
            return
        took = False
        for u in todo:
            if not ready(u):
                continue
            big = u[0] == "tab" and u[2] == "full" and u[1] in ("rf-l2c", "xgb-l2c", "lgbm-l2c")
            if u[0] in ("tab", "rdt", "dscore") and left < (14 if big else 6):
                continue                                           # 做不完的不领（网络每个 epoch 存一次，可以领）
            os.makedirs(os.path.dirname(upath(u)), exist_ok=True)
            lock = upath(u) + ".lock"
            if not claim(P, lock, u):
                continue
            try:
                do_unit(u, lock)
            finally:
                release(lock)
            took = True
            break
        if not took:
            print(f"nothing I can take now ({len(todo)} left: locked by other workers, waiting on Delphi, or too little time)",
                  flush=True)
            return


def status():
    U = units()
    todo = [u for u in U if not os.path.exists(upath(u))]
    print(f"{len(U) - len(todo)} / {len(U)} units done")
    from collections import Counter
    c = Counter((u[0], u[1], u[2]) for u in todo)
    for k, v in sorted(c.items()):
        lk = any(os.path.exists(upath(u) + ".lock") for u in todo if (u[0], u[1], u[2]) == k)
        print(f"  {k}: {v} left{' (in progress)' if lk else ''}{'' if all(ready(u) for u in todo if (u[0], u[1], u[2]) == k) else ' (waiting on GPU training)'}")


# ============================================================================== 汇总


def _auc(y, s, w):
    from sklearn.metrics import roc_auc_score
    k = w > 0
    if len(np.unique(y[k])) < 2:
        return np.nan
    return roc_auc_score(y[k], s[k], sample_weight=w[k])


def evaluate(Y, pidx, n_person, risks, ref="lr-cur"):
    """Y (站点, 6, 10)、pidx 站点 -> 人的序号、risks {模型: (站点, 6, 10)}。每个模型：4 终点 2 年、6 题 5 年、每个终点；减 ref 的配对 bootstrap。"""
    from joblib import Parallel, delayed
    cells = [(qi, 1, f"p1_{q}") for qi, q in enumerate(B.Q) if q in P1] + [(qi, 4, f"p3_{q}") for qi, q in enumerate(B.Q)]
    C = []
    for qi, hi, nm in cells:
        k = Y[:, qi, hi] >= 0
        C.append((nm, Y[k, qi, hi], pidx[k], {m: R[k, qi, hi] for m, R in risks.items()}))

    def score(w_person, m):
        out = {}
        for nm, y, pi, rs in C:
            out[nm] = _auc(y, rs[m], w_person[pi])
        p1 = np.nanmean([v for k, v in out.items() if k.startswith("p1_")])
        p3 = np.nanmean([v for k, v in out.items() if k.startswith("p3_")])
        return p1, p3, out

    one = np.ones(n_person)
    full = {m: score(one, m) for m in risks}
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(n_person, size=n_person), minlength=n_person).astype(float) for _ in range(N_BOOT)]

    def boot(m):
        return np.array([score(w, m)[:2] for w in W])
    names = list(risks)
    bs = dict(zip(names, Parallel(n_jobs=min(len(names), ncpu()))(delayed(boot)(m) for m in names)))
    res = {}
    for m in names:
        p1, p3, per = full[m]
        r = {"p1": p1, "p3": p3, "cells": per}
        if ref in bs and m != ref:
            d = bs[m] - bs[ref]
            for j, key in enumerate(("p1", "p3")):
                r[f"{key}_vs_lr"] = {"diff": float(full[m][j] - full[ref][j]), "ci": [float(np.nanquantile(d[:, j], 0.025)),
                                                                                       float(np.nanquantile(d[:, j], 0.975))]}
        res[m] = r
    return res


def tab_risk(name, size, n_rows, kind_dir="tab", tuY=None):
    """表格模型：在 tune 上挑网格（tabular.run_classifier 的同一个准则：6 题 × 2/5/10 年平均 AUC），拼回 (站点, 6, 10)。"""
    k = TAB[name][0] if name in TAB else "dt"
    best = None
    for hi in range(len(grid(k))):
        ps = [os.path.join(OUT, kind_dir, f"{name}__{size}__h{hi}__q{qi}.npz") for qi in range(B.NQ)]
        if not all(os.path.exists(p) for p in ps):
            return None
        Z = [np.load(p) for p in ps]
        Rtu, Rte = np.stack([z["tu"] for z in Z], 1), np.stack([z["te"] for z in Z], 1)
        s = B.macro_auc(tuY, Rtu)
        if best is None or s > best[0]:
            best = (s, grid(k)[hi], Rte)
    assert best[2].shape[0] == n_rows
    return best[2], {"chosen": best[1], "tune_macro_auc": best[0]}


def collect():
    D, sp = nacc()
    Dn = D["test"]
    M = Dn["M"].astype(bool)
    Y = Dn["Y"].reshape(*M.shape, B.NQ, B.NH)[M]
    pidx = np.repeat(np.arange(len(M)), M.sum(1))
    out = {"split": json.load(open(os.path.join(RES, "split_summary.json"))), "n_test_persons": int(len(M)),
           "n_test_landmarks": int(M.sum()), "n_boot": N_BOOT, "nacc": {}, "rosmap": {}}
    for size in SIZES:
        risks, meta = {}, {}
        tuY = lms(size, "cur")[1]["Y"]
        for name in TAB:
            r = tab_risk(name, size, len(Y), tuY=tuY)
            if r:
                risks[name], meta[name] = r
        for name in NETS:
            ps = [upath(("net", name, size, s)) for s in NET_SEEDS]
            ps = [p for p in ps if os.path.exists(p)]
            if ps:
                Z = [np.load(p) for p in ps]
                risks[name] = np.mean([z["te"] for z in Z], 0)
                meta[name] = {"n_seeds": len(ps), "runs": [json.loads(str(z["info"])) for z in Z]}
        for v in DELPHI:
            ps = [upath(("dscore", v, size, s)) for s in DELPHI_SEEDS]
            ps = [p for p in ps if os.path.exists(p)]
            if ps:
                Z = [np.load(p) for p in ps]
                risks[v] = np.mean([z["te"] for z in Z], 0)
                meta[v] = {"n_seeds": len(ps), "runs": [json.loads(str(z["info"])) for z in Z]}
        print(f"{size}: {len(risks)} models", flush=True)
        if not risks:
            continue
        ev = evaluate(Y, pidx, len(M), risks)
        out["nacc"][size] = {"n_train_persons": int(len(sp[size])), "models": {m: {**ev[m], **meta[m]} for m in ev}}
        for m in sorted(ev, key=lambda m: -ev[m]["p1"]):
            print(f"  {m:>16s}  p1 {ev[m]['p1']:.4f}  p3 {ev[m]['p3']:.4f}  "
                  f"{('vs lr ' + format(ev[m]['p1_vs_lr']['diff'], '+.4f')) if 'p1_vs_lr' in ev[m] else ''}", flush=True)
    # ROSMAP val：同一个指标（第三部分的标签：4 个终点 2 年、6 题 5 年），现成的打分表
    from transitions import label_col, risk_col
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv")).set_index(["pid", "t"])
    Yr = np.stack([np.stack([truth[label_col(q, H)].to_numpy() for H in B.HS], 1) for q in B.Q], 1)
    risks = {}
    for m, src in ROSMAP_SRC.items():
        ps = sorted(glob.glob(os.path.join(RUN_DIR, "scores_trans", f"{src}__s[0-9]*.csv")))
        if not src.startswith("delphi-"):
            ps = [p for p in ps if p.endswith("__s0.csv")]
        if not ps:
            continue
        T = [pd.read_csv(p).set_index(["pid", "t"]).loc[truth.index] for p in ps]
        risks[m] = np.mean([np.stack([np.stack([t[risk_col(q, H)].to_numpy() for H in B.HS], 1) for q in B.Q], 1) for t in T], 0)
    Z, _ = B.load()
    lv = B.landmarks(B.split(Z, "val"), "l2c")
    r = tab_risk("dt-l2c", "rosmap-val", len(lv["Y"]), kind_dir="rdt", tuY=B.landmarks(B.split(Z, "test"), "l2c")["Y"])
    if r:
        R = pd.DataFrame(r[0].reshape(len(r[0]), -1),                 # bl_data.write_scores 的同一种键（pid、t 取整）
                         index=pd.MultiIndex.from_arrays([lv["pid"].astype(int), lv["t"].astype(int)], names=["pid", "t"]))
        risks["dt-l2c"] = R.loc[truth.index].to_numpy().reshape(-1, B.NQ, B.NH)
    pids = truth.index.get_level_values(0).to_numpy()
    up, pidx_r = np.unique(pids, return_inverse=True)
    ev = evaluate(Yr, pidx_r, len(up), risks)
    out["rosmap"] = {"n_val_persons": int(len(up)), "n_train_persons": int(len(Z["train_pid"])), "models": ev}
    if r:
        out["rosmap"]["models"]["dt-l2c"].update(r[1])
    print("rosmap val:", flush=True)
    for m in sorted(ev, key=lambda m: -ev[m]["p1"]):
        print(f"  {m:>16s}  p1 {ev[m]['p1']:.4f}  p3 {ev[m]['p3']:.4f}", flush=True)
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=1)
    print("->", os.path.join(RES, "results.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prep", "work", "status", "collect"])
    ap.add_argument("--budget", type=float, default=27.0)
    a = ap.parse_args()
    {"prep": prep, "status": status, "collect": collect}.get(a.what, lambda: work(a.budget))()
