"""ext_common.py -- 外部验证共用：路径、模型名单、站点截断、用存盘的模型打分。

截断（PROTOCOL.md 第 4 节第 5 条）：每个站点一条记录，站点之后的访视在交给模型之前物理删掉（X、L、T 置 0，M 置 False）；
记录里只有 X / M / S / L / T / pid，没有 Y。ROSMAP 复现核对和 NACC 打分走这同一段代码。
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "models_code"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "unified_eval"))
sys.path.insert(0, os.path.join(HERE, "_vendor"))            # cloudpickle（读写按值序列化的模型）
import bl_data as B  # noqa: E402
from common import RUN_DIR  # noqa: E402

OUT = os.path.join(RUN_DIR, "nacc_ext")
MODEL_DIR = os.path.join(OUT, "models")
SCORE_DIR = os.path.join(OUT, "scores")
REP = ["xgb-l2c-sim", "minimalrnn", "transformer", "fnn-l2c-sim", "ar-lr", "cox-cur-sim", "ethos", "bdsi-sim"]
DIRECT = ["xgb-l2c", "lr-cur", "lr-demo"]
NONDELPHI = REP + DIRECT                                      # 11 个，ext_train.py 训练
DELPHI = ["delphi-dedup", "delphi-dedup-sim"]                 # 已有的检查点，ext_delphi.py
ALL13 = NONDELPHI + DELPHI
# 和打包盒那次运行一样的参数（RIS 日志 from_baselines_code/bl_*.out 的命令行）
OVERRIDES = {"ethos": {"n_mc_lm": 50}, "fnn-l2c-sim": {"lm_chunk": 300}}
GPU_MODELS = {"minimalrnn", "transformer", "fnn-l2c-sim", "ethos", "cox-cur-sim"}
KEYS = ("X", "M", "S", "L", "T", "pid")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def features_only(D):
    """一段人级数组 -> 只留特征。打分进程里只有这个版本。"""
    return {k: D[k] for k in KEYS}


def truncate(D, ii, vv):
    """第 j 条记录 = 第 ii[j] 个人到第 vv[j] 次访视为止（之后的访视删掉）。"""
    ii, vv = np.asarray(ii), np.asarray(vv)
    after = np.arange(D["M"].shape[1])[None, :] > vv[:, None]
    X, M, L, T = (D[k][ii].copy() for k in ("X", "M", "L", "T"))
    X[after] = 0
    L[after] = 0
    T[after] = 0
    M[after] = False
    assert M[np.arange(len(ii)), vv].all(), "站点那次访视不存在"
    return {"X": X, "M": M, "S": D["S"][ii].copy(), "L": L, "T": T, "pid": D["pid"][ii].copy()}


def risk_fn(obj, norm):
    """存盘的模型 -> f(截断的记录, vv, rng, n_mc) -> (n, 6, 10) 累计风险（和 run.train_and_score 同一种读法）。"""
    if obj["kind"] == "clf":
        prep, models, fs = obj["prep"], obj["models"], obj["featset"]

        def f(Dc, vv, rng, n_mc):
            F = prep(B.featrows(Dc, np.arange(len(vv)), vv, fs))
            R = np.zeros((len(vv), B.NQ, B.NH))
            for qi, hi, m in models:
                R[:, qi, hi] = m.predict_proba(F)[:, 1]
            return R
        return f
    import generators as G
    import simulator as SIM
    impl, gen = obj["impl"], obj["gen"]

    def f(Dc, vv, rng, n_mc):
        st = {"D": Dc, "ii": np.arange(len(vv)), "vv": vv, "pid": Dc["pid"], "t": Dc["T"][np.arange(len(vv)), vv]}
        if impl == "ar":
            return SIM.risks(G.ar_sim(gen, st, n_mc, rng, norm))
        if impl == "minimalrnn":
            return SIM.risks(G.minimalrnn_sim(gen[0], st, n_mc, rng, norm, gen[1]))
        if impl in ("msm", "ctmsm"):
            return gen.landmark_risks(st, norm)
        return SIM.risks(gen.sim(st, n_mc, rng, norm))
    return f


def score(obj, D, ii, vv, rng, norm, n_mc=None, chunk=None, log=None):
    """D 只能是 features_only 的结果（断言没有 Y）。按 chunk 个站点一批（和 run.py 的 lm_chunk 一样分批，随机数顺序才一致）。"""
    assert "Y" not in D, "打分进程拿到了标签"
    f = risk_fn(obj, norm)
    n_mc = n_mc or obj.get("n_mc", 100)
    chunk = chunk or obj.get("chunk", 1000)
    ii, vv = np.asarray(ii), np.asarray(vv)
    R = np.zeros((len(ii), B.NQ, B.NH))
    for a in range(0, len(ii), chunk):
        Dc = truncate(D, ii[a: a + chunk], vv[a: a + chunk])
        R[a: a + chunk] = f(Dc, vv[a: a + chunk], rng, n_mc)
        if log:
            log(a + len(Dc["pid"]), len(ii))
    return R


def load_model(name, device=None):
    import torch
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    obj = torch.load(path, map_location=device, weights_only=False)
    return obj, path
