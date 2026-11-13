"""perturb_features.py -- 在统一输入的特征空间里做扰动（models_code/input_data.py 的 npz 格式），以及从特征重算手工历史特征。

统一输入的模型（models_code/perturb.py）和数据对照（perturb_reference.py）共用这一份修改函数，保证两边改的是同一件事。
特征布局见 model_zoo/models_code/input_data.py：
  X[..., 0:15]  15 个连续量（训练集标准化），顺序 = data.CONT（0 = MMSE，1 = cogn_global）
  X[..., 15:30] 缺失掩码（1 = 没测）
  X[..., 30:39] 9 个状态；X[..., 39:48] 状态缺失掩码
  X[..., 48] 卒中曾经有过；X[..., 49] 抑郁曾经有过；X[..., 50] AD 诊断状态
  X[..., 51] (年龄-80)/10；X[..., 52] 随访年数/10；X[..., 53] 距上次访视的年数
  S[0] 男；S[1] (教育-16)/4；S[2] 教育缺失；S[9] APOE e4 1 个；S[10] e4 2 个；S[11] APOE 缺失
  L = 每个连续量 8 个历史特征（最近值、是否测过、距最近测量年数、最近变化、最小、最大、斜率、log 次数）
"""
import numpy as np

N_CONT = 15
I_MMSE, I_COGN, I_STROKE, I_AGE, I_YEARS = 0, 1, 48, 51, 52


def l2c_from_x(X, M):
    """从 X 重算 L（和 data.l2c_features 一样的定义，只用到当前访视为止）。X: (V, F) 单人，M: (V,)。"""
    V = X.shape[0]
    L = np.zeros((V, N_CONT * 8), np.float32)
    years = X[:, I_YEARS] * 10.0
    for v in range(V):
        if not M[v]:
            continue
        y = years[v]
        out = []
        for c in range(N_CONT):
            obs = [(years[u], X[u, c]) for u in range(v + 1) if M[u] and X[u, N_CONT + c] < 0.5]
            if not obs:
                out += [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            yrs = np.array([a for a, _ in obs])
            val = np.array([b for _, b in obs])
            slope = float(np.polyfit(yrs, val, 1)[0]) if len(obs) >= 2 and np.ptp(yrs) > 0 else 0.0
            out += [float(val[-1]), 1.0, min(float(y - yrs[-1]), 10.0),
                    float(val[-1] - val[-2]) if len(obs) >= 2 else 0.0,
                    float(val.min()), float(val.max()), float(np.clip(slope, -3, 3)), float(np.log1p(len(obs)))]
        L[v] = out
    return L


def apply(pert, X, M, S, Lf, vL, norm):
    """单人（X: (V,F), M: (V,), S: (Sdim,), Lf: (V, 120)），站点 = 第 vL 次访视。返回改过的 (X, S, L) 副本。
    norm: data 的 {"mu": {...}, "sd": {...}}，用来把 MMSE / cogn 的原始值换算成标准化值。"""
    X, S, Lf = X.copy(), S.copy(), Lf.copy()
    recompute = False
    if pert == "orig":
        pass
    elif pert == "apoe":
        S[9], S[10], S[11] = 1.0, 0.0, 0.0
    elif pert == "male":
        S[0] = 1.0
    elif pert == "educ":
        S[1], S[2] = (10.0 - 16.0) / 4.0, 0.0
    elif pert == "age5":
        X[M, I_AGE] += 0.5
    elif pert == "stroke":
        X[vL:, I_STROKE] = np.where(M[vL:], 1.0, X[vL:, I_STROKE])
    elif pert == "stroke_early":
        X[:, I_STROKE] = np.where(M, 1.0, X[:, I_STROKE])
    elif pert.startswith("mmse"):
        v = float(pert[4:])
        X[vL, I_MMSE] = (v - norm["mu"]["cts_estmmse30"]) / norm["sd"]["cts_estmmse30"]
        X[vL, N_CONT + I_MMSE] = 0.0
        recompute = True
    elif pert == "cogn":
        X[vL, I_COGN] = (-0.5 - norm["mu"]["cogn_global"]) / norm["sd"]["cogn_global"]
        X[vL, N_CONT + I_COGN] = 0.0
        recompute = True
    else:
        raise KeyError(pert)
    if recompute:
        Lf = l2c_from_x(X, M)
    return X, S, Lf
