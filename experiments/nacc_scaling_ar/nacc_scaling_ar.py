"""nacc_scaling_ar.py -- 训练量的学习曲线，全部用自回归（只学下一步、答案全从 rollout 来）的模型（2026-10-28）。

问题：zoo 的序列网络改成了自回归（rollout_versions.SeqARModel）之后，"数据越多神经网络越占优"有没有出现？
设计（README.md 有完整说明）：
  数据   NACC（nacc_data.py），tune 段（早停 / 挑学习率）和 test 段都和 ../nacc_scale 同一批人。
         其余的人：每次抽样一个随机排列，取前 1k / 2k / 3,542 / 7k / 12k / 25k 人（同一次抽样里嵌套）+ 全部；抽 3 次。
  模型   全部只学"下一年"：这一年里死亡 / 得 AD 的概率 + 下一次访视的分布，然后一年一年往后模拟。
           ar-lr、ar-lgbm   参数化 g-formula（generators.ARModels，zoo 的原配方）；
           trf-s / m / l    自回归 transformer（rollout_versions.SeqARModel 的结构：sequence.Causal + 2 个风险输出 + 下一次访视头；
                            S = d64 × 2 层 = zoo 里那个，M = d128 × 4 层，L = d256 × 6 层；学习率 {5e-4, 2e-4} 按 tune 上的下一步损失挑）。
         网络每次抽样一个 seed（= 抽样号），不集成。
  打分   test 的每个站点模拟 100 条未来、走 5 年，数 H 年内出事的比例 k，风险 = (k + 0.5) / 101（和 run.py 扰动打分同一个平滑）。
         10 个格子（4 个终点 2 年 + 6 题 5 年）的 log-loss（主指标）、Brier、AUC；幂律在本地拟合（make_fig.py）。

    python nacc_scaling_ar.py prep              # 划分（7 档 × 3 次抽样）+ 手工历史特征逐步回放的自检
    python nacc_scaling_ar.py work --budget 27  # CPU 工人（ris/worker.sbatch）
    python nacc_scaling_ar.py work --budget 27 --gpu   # GPU 工人：transformer M / L 优先（ris/gpu.sbatch）
    python nacc_scaling_ar.py status
    python nacc_scaling_ar.py collect           # -> $ZOO_RUN_DIR/packages/_nacc_scaling_ar/results.json（聚合）
冒烟：NS3_SMOKE=1（只做 1k、第 0 次抽样、300 个 test 站点、20 条轨迹、网络 2 个 epoch，写到 *_smoke 目录）。

人级的东西只留在 RIS 的 $ZOO_RUN_DIR/nacc_scaling_ar/。
"""
import argparse
import json
import os
import pickle
import sys
import time

SMOKE = os.environ.get("NS3_SMOKE", "0") == "1"
if SMOKE:
    os.environ.setdefault("NACC_SMOKE_EPOCHS", "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ZOO, "models_code"))
sys.path.insert(0, os.path.join(ZOO, "unified_eval"))
sys.path.insert(0, os.path.join(HERE, "..", "nacc_scale"))
import bl_data as B  # noqa: E402
from common import RUN_DIR  # noqa: E402
import generators as G  # noqa: E402
import nacc_scale as NS  # noqa: E402  tune 段、AUC
import simulator as SIM  # noqa: E402

OUT = os.path.join(RUN_DIR, "nacc_scaling_ar" + ("_smoke" if SMOKE else ""))
RES = os.path.join(RUN_DIR, "packages", "_nacc_scaling_ar" + ("_smoke" if SMOKE else ""))
SEED = 20261029                                   # 和之前的学习曲线同一组抽样
SIZES = {"1k": 1000, "2k": 2000, "3.5k": 3542, "7k": 7000, "12k": 12000, "25k": 25000, "full": None}
DRAWS = (0, 1, 2)
if SMOKE:
    SIZES, DRAWS = {"1k": 1000}, (0,)
AR = {"ar-lr": "lr", "ar-lgbm": "lgbm"}
TRF = {"trf-s": (64, 2, 4), "trf-m": (128, 4, 4), "trf-l": (256, 6, 8)}   # (d, 层, 头)
TRF_LR = (5e-4, 2e-4)
NETS = ("trf-s", "trf-m", "trf-l")
MODELS = tuple(AR) + NETS
# 模拟随机数里的模型序号：固定成删除 landmark 模型（nm-ar，原来是 2）之前的值，重跑能复现 results/ 里的数
SEED_IDX = {"ar-lr": 0, "ar-lgbm": 1, "trf-s": 3, "trf-m": 4, "trf-l": 5}
GPU_ONLY = ("trf-m", "trf-l")
N_MC = 20 if SMOKE else 100
SIM_YEARS = 5                                     # 只读到 5 年（10 个格子最长 5 年）
CHUNK = 1000                                      # 一块 = 这么多个站点 × N_MC 条轨迹，每块存一次
N_LM_SMOKE = 300
P1 = NS.P1
ME = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")


# ============================================================================== 数据


_C = {}


def nacc():
    if "D" not in _C:
        import pretrain as P
        _C["D"] = P.load_nacc()
        _C["sp"] = dict(np.load(os.path.join(OUT, "split.npz")))
    return _C["D"], _C["sp"]


def norm():
    if "norm" not in _C:
        import pretrain as P
        _C["norm"] = SIM.norm_info(json.load(open(os.path.join(P.NACC_DIR, "nacc_meta.json"))))
    return _C["norm"]


def idx_of(lab, r):
    sp = nacc()[1]
    return sp["full"] if lab == "full" else sp[f"{lab}_r{r}"]


def parts(lab, r):
    """(训练, tune, test) 三段的人级数组。"""
    D, sp = nacc()
    return NS.sub(D["train"], idx_of(lab, r)), NS.sub(D["train"], sp["hold"]), D["test"]


def test_landmarks():
    """test 段的站点（和 B.landmarks、SIM.landmark_start 同一个顺序）；冒烟只取前 N_LM_SMOKE 个。"""
    ii, vv = np.where(nacc()[0]["test"]["M"])
    return (ii[:N_LM_SMOKE], vv[:N_LM_SMOKE]) if SMOKE else (ii, vv)


def prep():
    import pretrain as P
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    Dall = P.load_nacc()
    D = Dall["train"]
    hold = np.load(os.path.join(NS.OUT, "split.npz"))["hold"]     # 和 nacc_scale 同一批 tune 的人
    rest = np.setdiff1d(np.arange(len(D["pid"])), hold)
    sp = {"hold": hold, "full": rest}
    for r in (0, 1, 2):
        perm = np.random.default_rng([SEED, r]).permutation(rest)
        for lab, n in SIZES.items():
            if n is not None:
                sp[f"{lab}_r{r}"] = np.sort(perm[:n])
    np.savez(os.path.join(OUT, "split.npz"), **sp)
    # 自检：站点起点的手工历史特征（模拟时逐步回放）= 数据里存的那份（NACC 的 L）
    Dt = Dall["test"]
    ii, vv = np.where(Dt["M"])
    k = np.random.default_rng(0).choice(len(ii), min(3000, len(ii)), replace=False)
    st = SIM.l2c_replay(Dt["X"][ii[k]], Dt["M"][ii[k]], vv[k])
    Lr = st.features(np.round(Dt["X"][ii[k], vv[k], B.I_TF + 1] * 10.0))
    l2c_diff = float(np.abs(Lr - Dt["L"][ii[k], vv[k]]).max())
    summ = {"n_hold": int(len(hold)), "n_rest": int(len(rest)),
            "sizes": {lab: int(len(sp["full"] if n is None else sp[f"{lab}_r0"])) for lab, n in SIZES.items()},
            "landmarks_r0": {lab: int(D["M"][sp["full"] if n is None else sp[f"{lab}_r0"]].sum()) for lab, n in SIZES.items()},
            "n_test_landmarks": int(len(ii)), "l2c_replay_max_abs_diff": l2c_diff}
    json.dump(summ, open(os.path.join(RES, "split_summary.json"), "w"), indent=1)
    print("split:", summ, flush=True)
    assert l2c_diff < 1e-3, "逐步回放的手工历史特征和数据里的对不上"


# ============================================================================== 单元


def fits_of(name, lab, r):
    """一个 (模型, 训练量, 抽样) 要训练的几份：transformer 每个学习率一份。"""
    return [("fit", name, lab, r, h) for h in (range(len(TRF_LR)) if name in TRF else (0,))]


def units():
    """先把第 0 次抽样的整条曲线做完，再做第 1、2 次；全部人那一档 ar 模型是确定的，只做一次。"""
    U = []
    for r in DRAWS:
        for lab in SIZES:
            for name in MODELS:
                if name in AR and lab == "full" and r > 0:
                    continue
                U += fits_of(name, lab, r) + [("roll", name, lab, r, 0)]
    return U


def upath(u):
    kind, name, lab, r, h = u
    return os.path.join(OUT, kind, f"{name}__{lab}__r{r}" + (f"__h{h}.json" if kind == "fit" else ".npz"))


def mpath(u):
    """训练好的模型（fit 单元的 json 是"做完了"的标记，模型先写）。"""
    return upath(u)[:-5] + (".pkl" if u[1] in AR else ".pt")


def deps_done(u):
    return u[0] == "fit" or all(os.path.exists(upath(f)) for f in fits_of(*u[1:4]))


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------ 网络


def build_trf(name):
    """rollout_versions.build_seqsim 的结构（风险头只有"下一年内死亡 / AD"两个输出），换 transformer 的大小。"""
    import torch.nn as nn
    import rollout_versions as RV
    import sequence as SQ
    d, nl, nh = TRF[name]
    hp = SQ.SEQ_HP["transformer"]

    class SeqAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = SQ.Causal(d, hp["dropout"], layers=nl, heads=nh)
            self.risk = nn.Linear(d, 2)
            self.next = nn.Linear(d, RV.N_NEXT)

        def forward(self, b):
            h = self.enc(b)
            return {"risk": self.risk(h), **RV.split_next(self.next(h))}

    return SeqAR()


def seq_tensors(P, D, dev):
    """序列网络的张量，标签只留"1 年内死亡 / AD"两列（rollout_versions.seqsim_loss 用全部列）。"""
    import rollout_versions as RV
    T = P.tensors("seqsim", D, dev)
    T["Y"] = T["Y"][..., [RV.IDX_D1, RV.IDX_A1]].contiguous()
    return T


def fit_net(u, lock):
    import torch
    import pretrain as P
    _, name, lab, r, h = u
    dev = device()
    torch.set_num_threads(NS.ncpu())
    trD, tuD, _ = parts(lab, r)
    torch.manual_seed(r)
    fam = "seqsim"
    tr, tu = seq_tensors(P, trD, dev), seq_tensors(P, tuD, dev)
    model = build_trf(name)
    _, wd, bs, epochs, patience, cosine = P.hp_of("seqsim", "transformer")
    lr = TRF_LR[h]
    model = model.to(dev)
    info = P.train_loop(fam, model, tr, tu, lr, wd, bs, epochs, patience, r, cosine, bucket=True,
                        partial=mpath(u) + ".partial", lock=lock)
    info.update(n_params=sum(q.numel() for q in model.parameters()), lr=lr, bs=bs, device=dev)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, mpath(u) + f".{ME}.tmp")
    os.replace(mpath(u) + f".{ME}.tmp", mpath(u))
    if os.path.exists(mpath(u) + ".partial"):
        os.remove(mpath(u) + ".partial")
    return info


class _Zero:
    """NACC 从来没测过的连续量（化验、睡眠、炎症）：没有样本可学；模拟时它们的"缺失"概率是 1，这个值不会被用到。"""

    def predict(self, F):
        return np.zeros(len(F))


class ARNacc(G.ARModels):
    """generators.ARModels，加一条：一个变量在训练段里一个观测都没有（NACC 缺的那些）就给常数，不去拟合 0 个样本。"""

    def _fit(self, m, F, y, Fe, ye, clf=True):
        if len(y) == 0:
            return ("const", 0.0) if clf else _Zero()
        return super()._fit(m, F, y, Fe, ye, clf)

    def fit(self, Ztr, Zte):
        super().fit(Ztr, Zte)
        self.sd = np.nan_to_num(self.sd, nan=1.0)
        return self


def fit_ar(u):
    _, name, lab, r, _ = u
    trD, tuD, _ = parts(lab, r)
    m = ARNacc(AR[name]).fit(trD, tuD)
    with open(mpath(u) + f".{ME}.tmp", "wb") as f:
        pickle.dump(m, f)
    os.replace(mpath(u) + f".{ME}.tmp", mpath(u))
    return {"minutes_fit": m.minutes, "n_train_rows": int(B.landmarks(trD, "cur")["Y"].shape[0])}


# ------------------------------------------------------------------------------ 模拟


def chosen_fit(name, lab, r):
    """transformer：tune 上下一步损失最低的那个学习率；其它模型只有一份。"""
    F = fits_of(name, lab, r)
    infos = [json.load(open(upath(f))) for f in F]
    j = int(np.argmin([i.get("best_test_loss", 0.0) for i in infos])) if name in TRF else 0
    return F[j], infos[j]


def stepper_factory(name, lab, r):
    import torch
    f, _ = chosen_fit(name, lab, r)
    D = nacc()[0]["test"]
    if name in AR:
        m = pickle.load(open(mpath(f), "rb"))
        return lambda ii, vv, y0: G.ARStepper(m, SIM.l2c_replay(D["X"][ii], D["M"][ii], vv), y0, N_MC)
    dev = device()
    torch.set_num_threads(NS.ncpu())
    net = build_trf(name)
    net.load_state_dict(torch.load(mpath(f), map_location="cpu"))
    net = net.to(dev).eval()
    import rollout_versions as RV
    m = RV.SeqARModel("transformer", dev)
    m.net = net
    return lambda ii, vv, y0: RV.SeqStepper(m, {"D": D, "ii": ii, "vv": vv}, N_MC, m.chunk)


def roll(u, lock, deadline):
    """test 的全部站点，每块 CHUNK 个站点（按访视位置排好，缓冲区短）存一次；时间不够就停，下一个工人接着做。
    存的是计数 k (站点, 6 题, 10 年)：N_MC 条轨迹里 H 年内出事的条数。"""
    _, name, lab, r, _ = u
    D = nacc()[0]["test"]
    ii, vv = test_landmarks()
    order = np.argsort(vv, kind="stable")
    blocks = [order[s: s + CHUNK] for s in range(0, len(order), CHUNK)]
    cdir = upath(u) + ".chunks"
    os.makedirs(cdir, exist_ok=True)
    make, took = None, []
    for c, idx in enumerate(blocks):
        cf = os.path.join(cdir, f"c{c:03d}.npy")
        if os.path.exists(cf):
            continue
        need = 1.5 * max(took) + 0.5 if took else 3.0
        if (deadline - time.time()) / 60 < need:
            print(f"    stop at block {c}/{len(blocks)}: {(deadline - time.time()) / 60:.1f} min left", flush=True)
            return False
        t0 = time.time()
        if make is None:
            make = stepper_factory(name, lab, r)
        a, b = ii[idx], vv[idx]
        y0 = np.round(D["X"][a, b, B.I_TF + 1] * 10.0)
        rng = np.random.default_rng([SEED, r, c, SEED_IDX[name], list(SIZES).index(lab)])
        sim = SIM.simulate_visits(make(a, b, y0), D["X"][a, b], D["S"][a], y0, N_MC, rng, norm(), years=SIM_YEARS)
        k = np.rint(SIM.risks(sim) * N_MC).astype(np.uint8)
        np.save(cf + f".{ME}.tmp.npy", k)
        os.replace(cf + f".{ME}.tmp.npy", cf)
        took.append((time.time() - t0) / 60)
        if lock:
            os.utime(lock)
        print(f"    block {c + 1}/{len(blocks)} ({len(idx)} landmarks) {took[-1]:.2f} min", flush=True)
    K = np.zeros((len(ii), B.NQ, B.NH), np.uint8)
    for c, idx in enumerate(blocks):
        K[idx] = np.load(os.path.join(cdir, f"c{c:03d}.npy"))
    tmp = upath(u) + f".{ME}.tmp.npz"
    np.savez(tmp, k=K, n_mc=N_MC, years=SIM_YEARS)
    os.replace(tmp, upath(u))
    for c in range(len(blocks)):
        os.remove(os.path.join(cdir, f"c{c:03d}.npy"))
    os.rmdir(cdir)
    return True


def do_unit(u, lock, deadline):
    t0 = time.time()
    if u[0] == "fit":
        info = fit_ar(u) if u[1] in AR else fit_net(u, lock)
        info["minutes"] = (time.time() - t0) / 60
        tmp = upath(u) + f".{ME}.tmp"
        json.dump(info, open(tmp, "w"), indent=1)
        os.replace(tmp, upath(u))
        done = True
    else:
        done = roll(u, lock, deadline)
    print(f"  {'done' if done else 'paused'} {u} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ============================================================================== 工人


def _holder(lock):
    try:
        return open(lock).read().strip()
    except FileNotFoundError:
        return None


def release(lock):
    if _holder(lock) == ME:
        os.remove(lock)


def claim(P, lock, u):
    """同 nacc_scale.claim：接手死锁后等 1 秒再读一遍、拿到锁时单元已经做完就放掉。"""
    try:
        if not P._claim(lock):
            return False
    except FileNotFoundError:
        return False
    time.sleep(1)
    if _holder(lock) != ME:
        return False
    if os.path.exists(upath(u)):
        release(lock)
        return False
    return True


def mine(u, gpu):
    """GPU 工人只做网络（大的 transformer 优先）；CPU 工人什么都做，大的 transformer 排最后（GPU 排不上队的时候帮着训，
    每个 epoch / 每块模拟都存，GPU 工人来了接着做）。"""
    return u[1] in NETS if gpu else True


def need_minutes(u):
    """开始一个单元至少要剩的时间：ar 的训练不能续，按训练量留够；网络每个 epoch、模拟每块都存，可以领。"""
    if u[0] == "fit" and u[1] == "ar-lgbm":
        return {"full": 22, "25k": 16, "12k": 10}.get(u[2], 6)
    if u[0] == "fit" and u[1] == "ar-lr":
        return {"full": 8, "25k": 6}.get(u[2], 3)
    return 3


def work(budget, gpu=False):
    import pretrain as P
    t0 = time.time()
    deadline = t0 + budget * 60
    while True:
        left = (deadline - time.time()) / 60
        todo = [u for u in units() if not os.path.exists(upath(u)) and mine(u, gpu)]
        todo.sort(key=lambda u: (u[1] not in GPU_ONLY) if gpu else (u[1] in GPU_ONLY))
        if not todo:
            print("ALL DONE (for this kind of worker)", flush=True)
            return
        took = False
        for u in todo:
            if left < need_minutes(u) or not deps_done(u):
                continue
            os.makedirs(os.path.dirname(upath(u)), exist_ok=True)
            lock = upath(u) + ".lock"
            if not claim(P, lock, u):
                continue
            try:
                do_unit(u, lock, deadline)
            finally:
                release(lock)
            took = True
            break
        if not took:
            print(f"nothing I can take now ({len(todo)} left: locked, waiting on fits, or too little time)", flush=True)
            return


def status():
    U = units()
    todo = [u for u in U if not os.path.exists(upath(u))]
    print(f"{len(U) - len(todo)} / {len(U)} units done")
    from collections import Counter
    c = Counter((u[0], u[1], u[3]) for u in todo)
    for k, v in sorted(c.items()):
        lk = sum(os.path.exists(upath(u) + ".lock") for u in todo if (u[0], u[1], u[3]) == k)
        print(f"  {k}: {v} left{f' ({lk} in progress)' if lk else ''}")


# ============================================================================== 汇总


def cells(Y):
    """10 个格子：4 个终点 2 年、6 题 5 年。(名字, 题号, 年号, 有标签的站点)"""
    out = [(f"p1_{q}", qi, 1) for qi, q in enumerate(B.Q) if q in P1] + [(f"p3_{q}", qi, 4) for qi, q in enumerate(B.Q)]
    return [(nm, qi, hi, Y[:, qi, hi] >= 0) for nm, qi, hi in out]


def metrics(Y, R, C):
    """每个格子的 log-loss、Brier、AUC，和 p1 / p3 / 全部 10 格的平均。"""
    per = {}
    for nm, qi, hi, k in C:
        y, p = Y[k, qi, hi], np.clip(R[k, qi, hi].astype(np.float64), 1e-6, 1 - 1e-6)
        per[nm] = {"ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "brier": float(np.mean((p - y) ** 2)),
                   "auc": float(NS._auc(y, p, np.ones(len(y)))), "n": int(k.sum()), "prev": float(y.mean()),
                   "mean_p": float(p.mean())}
    m = {}
    for key in ("ll", "brier", "auc"):
        for part, pre in (("p1", "p1_"), ("p3", "p3_"), ("all", "")):
            m[f"{key}_{part}"] = float(np.mean([v[key] for nm, v in per.items() if nm.startswith(pre)]))
    m["cells"] = per
    return m


def collect():
    D, sp = nacc()
    Dn = D["test"]
    ii, vv = test_landmarks()
    Y = Dn["Y"].reshape(*Dn["M"].shape, B.NQ, B.NH)[ii, vv]
    C = cells(Y)
    runs = []
    for r in DRAWS:
        for lab in SIZES:
            n_tr = int(len(idx_of(lab, r)))
            for name in MODELS:
                u = ("roll", name, lab, r, 0)
                if not os.path.exists(upath(u)):                 # 全部人那一档 ar 模型只有第 0 次
                    continue
                Z = np.load(upath(u))
                R = (Z["k"].astype(np.float64) + 0.5) / (int(Z["n_mc"]) + 1)
                m = metrics(Y, R, C)
                f, info = chosen_fit(name, lab, r)
                if name in AR:
                    info = dict(info, n_params=None)
                row = {"model": name, "size": lab, "n_train": n_tr, "draw": r, "info": info, **m}
                runs.append(row)
                print(f"{name:>8s} {lab:>5s} r{r}  ll {m['ll_all']:.4f}  auc_p1 {m['auc_p1']:.4f}", flush=True)
    out = {"split": json.load(open(os.path.join(RES, "split_summary.json"))), "n_test_persons": int(len(Dn["M"])),
           "n_test_landmarks": int(len(ii)), "n_mc": N_MC, "sim_years": SIM_YEARS, "risk": "(k + 0.5) / (n_mc + 1)",
           "trf_sizes": {k: dict(zip(("d", "layers", "heads"), v)) for k, v in TRF.items()}, "trf_lr_grid": TRF_LR,
           "runs": runs}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "results.json"), "w"), indent=1)
    print("->", os.path.join(RES, "results.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prep", "work", "status", "collect"])
    ap.add_argument("--budget", type=float, default=27.0)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "status": status, "collect": collect}.get(a.what, lambda: work(a.budget, a.gpu))()
