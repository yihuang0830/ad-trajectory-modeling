"""bl_data.py -- 统一输入的模型（Delphi 以外的全部）共用的输入和输出。

输入：这些模型共用的一份数据（models_code/input_data.py -> input_data_trans.py 的产物 $ZOO_RUN_DIR/newmodel_data_trans.npz），
不重新清洗、不重新标准化，所以每个模型看到的信息一样多，差别只在模型。
  X (人, 访视, 54)：15 个连续量（训练段 z 分数，缺失记 0）+ 15 个缺失掩码 + 9 个状态 + 9 个状态缺失 + 2 个"曾经有过" + AD 状态
                   + 年龄 z、随访年数 / 10、距上次访视的年数
  S (人, 25)：静态背景；L (人, 访视, 120)：手工历史特征（每个连续量 8 个，TADPOLE Frog / L2C 的做法）
  Y (人, 访视, 60)：第三部分的 6 道题 × 逐年 1..10 年（题目外层、时间窗内层），-1 = 答案未知或不在风险中
划分：split811。train 训练，test 调参 / 早停 / 挑版本，val（和所有模型同一批）只在最后报一次数。

输出：和 Delphi 同一种打分表（人级，只留在 RIS）
  $ZOO_RUN_DIR/scores_trans/<name>__<tag>.csv   pid, t, risk_<题>_<H>y（6 道题 × 1..10 年）
  $ZOO_RUN_DIR/scores/<name>__<tag>.csv         第一部分的列名（4 个终点 × 1/2/3/5 年），同一份风险改名
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import HORIZONS, RUN_DIR  # noqa: E402
from transitions import SAME_AS_PART1, TRANS, TRUTH_HORIZONS, label_col, risk_col  # noqa: E402

Q = list(TRANS)
HS = tuple(TRUTH_HORIZONS)                 # 1..10
NQ, NH = len(Q), len(HS)
I_DEATH, I_ADQ = Q.index("death"), Q.index("ad")
THRESH = {"mmse_lt27": ("mmse", 27.0), "mmse_lt24": ("mmse", 24.0), "cogn_lt0": ("cogn", 0.0),
          "cogn_ltm1": ("cogn", -1.0)}

# 一次访视 54 维里各段的起点（models_code/input_data.py 的布局）
N_CONT, N_STATE, N_EVER = 15, 9, 2
I_MC, I_XS, I_MS = N_CONT, 2 * N_CONT, 2 * N_CONT + N_STATE
I_EVER = 2 * N_CONT + 2 * N_STATE
I_AD = I_EVER + N_EVER
I_TF = I_AD + 1
N_FEAT = I_TF + 3
N_CUM = 5                                  # STATE 的前 5 个是累计标志（*_cum），只增不减
N_DEMO_STATIC = 13                         # 静态背景的前 13 维：性别、教育、种族、西语裔、APOE

DATA_FILE = "newmodel_data_trans.npz"
N_S = 25                                   # 静态背景的列数；load() 读加基因的变体（32 列）时改掉，网络的输入宽度按它建


def load(data_file=None):
    """data_file：None = 原版；"newmodel_data_trans_geno.npz" = S 末尾多 7 列基因（input_data_geno.py）。"""
    global DATA_FILE, N_S
    DATA_FILE = data_file or DATA_FILE
    Z = np.load(os.path.join(RUN_DIR, DATA_FILE))
    meta = json.load(open(os.path.join(RUN_DIR, DATA_FILE.replace(".npz", "_meta.json"))))
    assert meta["labels"] == [label_col(q, H) for q in Q for H in HS], "标签列的顺序和 transitions 对不上"
    assert Z["val_X"].shape[-1] == N_FEAT
    N_S = Z["val_S"].shape[-1]
    assert N_S == len(meta["static"])
    return {k: Z[k] for k in Z.files}, meta


def split(Z, s):
    """一个段的人级数组。"""
    return {k: Z[f"{s}_{k}"] for k in ("X", "M", "S", "L", "Y", "T", "pid")}


def landmarks(D, featset):
    """人级 -> 站点表：每个 (人, 访视) 一行。featset：
       demo = 年龄 + 人口学（性别、教育、种族、西语裔、APOE）
       cur  = 当前这次访视的全部 54 维 + 静态背景 25 维（站点 Cox / LOCF 的输入：不看更早的访视）
       l2c  = cur + 手工历史特征 120 维（TADPOLE Frog、L2C-FNN 的输入）"""
    ii, vv = np.where(D["M"])
    return {"F": featrows(D, ii, vv, featset), "Y": D["Y"][ii, vv].reshape(-1, NQ, NH), "pid": D["pid"][ii],
            "t": D["T"][ii, vv], "ii": ii, "vv": vv}


def at(D, vL, featset):
    """扰动记录用：第 i 个"人"只取第 vL[i] 次访视那一行。"""
    return featrows(D, np.arange(len(vL)), np.asarray(vL), featset)


def featrows(D, ii, vv, featset):
    x = D["X"][ii, vv]
    if featset == "demo":
        F = np.concatenate([x[:, [I_TF]], D["S"][ii, :N_DEMO_STATIC]], 1)
    elif featset == "cur":
        F = np.concatenate([x, D["S"][ii]], 1)
    elif featset == "l2c":
        F = np.concatenate([x, D["S"][ii], D["L"][ii, vv]], 1)
    elif featset == "cogn":
        # Yang et al. 2022（ROS/MAP）的"只看全局认知"：最近一次的 cogn_global（没测过 = 0）+ 是否测过
        L = D["L"][ii, vv]
        F = np.stack([L[:, 1 * 8 + 0], L[:, 1 * 8 + 1]], 1)
    elif featset == "super":
        # landmark supermodel（van Houwelingen 2007）：当前访视 + 静态背景，再加它们和站点时间 s、s² 的交互（s = 随访年数 / 10）
        c = np.concatenate([x, D["S"][ii]], 1)
        sl = x[:, [I_TF + 1]]
        F = np.concatenate([c, c * sl, c * sl ** 2], 1)
    else:
        raise ValueError(featset)
    return F.astype(np.float32)


def tte(yq):
    """一道题的 10 个逐年标签 (n, 10) -> 离散时间的 (T, E, ok)。
    发生：T = 第一个 y=1 的 H（事件落在 (T-1, T] 年）；没发生：T = 最后一个 y=0 的 H（删失在 T 年）；
    一个已知格都没有（不在风险中或一年都没观察到）-> ok=False，不进训练。"""
    pos = yq == 1
    neg = yq == 0
    has_pos = pos.any(1)
    t_pos = np.where(has_pos, pos.argmax(1) + 1, 0)
    t_neg = np.where(neg.any(1), NH - neg[:, ::-1].argmax(1), 0)
    E = has_pos.astype(np.int64)
    T = np.where(has_pos, t_pos, t_neg).astype(np.int64)
    ok = T > 0
    return T, E, ok


def write_scores(name, tag, pid, t, R, split_name="val"):
    """R (站点, 6, 10) 的累计风险 -> 第三部分和第一部分两份打分表。val 以外的段写进 _<段> 子目录（调参用，不进打包盒）。"""
    assert R.shape[1:] == (NQ, NH) and np.isfinite(R).all()
    R = np.clip(R, 1e-6, 1 - 1e-6)
    tab = pd.DataFrame({"pid": pid.astype(int), "t": t.astype(int)})
    for qi, q in enumerate(Q):
        for hi, H in enumerate(HS):
            tab[risk_col(q, H)] = R[:, qi, hi]
    sub = "" if split_name == "val" else f"_{split_name}"
    d3 = os.path.join(RUN_DIR, f"scores_trans{sub}")
    d1 = os.path.join(RUN_DIR, f"scores{sub}")
    os.makedirs(d3, exist_ok=True)
    os.makedirs(d1, exist_ok=True)
    tab.to_csv(os.path.join(d3, f"{name}__{tag}.csv"), index=False)
    p1 = tab[["pid", "t"]].copy()
    for q, e in SAME_AS_PART1.items():
        for H in HORIZONS:
            p1[f"risk_{e}_{H}y"] = tab[risk_col(q, H)]
    p1.to_csv(os.path.join(d1, f"{name}__{tag}.csv"), index=False)
    return os.path.join(d1, f"{name}__{tag}.csv"), os.path.join(d3, f"{name}__{tag}.csv")


def masked_auc(y, p):
    """y ∈ {-1, 0, 1}：只数已知格。阳性或阴性没有时返回 nan。"""
    from sklearn.metrics import roc_auc_score
    ok = y >= 0
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], p[ok]))


def macro_auc(Y, R, hs=(2, 5, 10)):
    """调参用的一个数：6 道题 × 给定时间窗的 AUC 平均（test 段，按站点，不 bootstrap）。"""
    a = [masked_auc(Y[:, qi, H - 1], R[:, qi, H - 1]) for qi in range(NQ) for H in hs]
    return float(np.nanmean(a))


def masked_bce(Y, R):
    ok = Y >= 0
    p = np.clip(R[ok], 1e-6, 1 - 1e-6)
    y = Y[ok]
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
