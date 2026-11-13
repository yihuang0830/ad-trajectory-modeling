"""ext_metrics.py -- 第 5、6 节的指标（纯函数，本地可测）。

auc            并列取平均秩的 AUC
wauc_prep/wauc 带权 AUC（bootstrap 用：权重 = 每个站点被重抽的次数），同分数并列的处理和 auc 一样
strat_auc      分层 AUC：只比同一层里的阳性-阴性对（= 各层 AUC 按 n⁺n⁻ 加权；unified_eval/evaluate_transitions.auc_within 同一个定义）
calib          O/E、校准截距（logit(p) 作偏移，calibration-in-the-large）、校准斜率（y ~ a + b·logit(p)）、Brier
recal_fit      logit(p') = α + β·logit(p)（H 上拟合，E 上套用）
boot_weights   两级聚类 bootstrap：先有放回地抽中心，再在每个抽到的中心里有放回地抽人 -> (重抽次数, 人) 的次数
reml_meta      随机效应汇总（REML），在 logit(AUC) 上做：汇总值、I²、τ²、95% 预测区间（t 分布，k − 2 自由度）
"""
import numpy as np
from scipy import stats


def auc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wauc_prep(y, s):
    u, g = np.unique(s, return_inverse=True)
    return {"g": g, "y": np.asarray(y).astype(float), "ng": len(u)}


def wauc(P, w):
    """w：(站点,) 或 (重抽, 站点) 的权重。"""
    w = np.atleast_2d(w)
    out = np.empty(len(w))
    for b, wb in enumerate(w):
        pw = np.bincount(P["g"], wb * P["y"], P["ng"])
        nw = np.bincount(P["g"], wb * (1 - P["y"]), P["ng"])
        below = np.cumsum(nw) - nw
        den = pw.sum() * nw.sum()
        out[b] = (pw * (below + 0.5 * nw)).sum() / den if den > 0 else np.nan
    return out


def strat_auc(y, s, k):
    y = np.asarray(y).astype(bool)
    num = den = 0.0
    for key in np.unique(k):
        m = k == key
        n1, n0 = y[m].sum(), (~y[m]).sum()
        if n1 and n0:
            num += auc(y[m], s[m]) * n1 * n0
            den += n1 * n0
    return float(num / den) if den else np.nan


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _logistic(y, X, offset=None, iters=50):
    """Newton–Raphson 的逻辑回归（不加惩罚）。"""
    beta = np.zeros(X.shape[1])
    off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def calib(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = logit(p)
    one = np.ones((len(y), 1))
    a_citl = float(_logistic(y, one, offset=z)[0])
    a, b = _logistic(y, np.column_stack([one, z]))
    return {"n": int(len(y)), "n_pos": int(y.sum()), "obs_rate": float(y.mean()), "mean_pred": float(p.mean()),
            "o_e": float(y.sum() / p.sum()), "cal_intercept": a_citl, "cal_slope": float(b),
            "brier": float(np.mean((p - y) ** 2))}


def recal_fit(y, p):
    a, b = _logistic(np.asarray(y, float), np.column_stack([np.ones(len(y)), logit(p)]))
    return float(a), float(b)


def recal_apply(p, a, b):
    return 1 / (1 + np.exp(-(a + b * logit(p))))


def boot_weights(center_of_person, n_boot, rng):
    """-> (n_boot, 人数) 的 int16 次数。"""
    cop = np.asarray(center_of_person)
    cs = np.unique(cop)
    members = {c: np.where(cop == c)[0] for c in cs}
    W = np.zeros((n_boot, len(cop)), np.int16)
    for b in range(n_boot):
        for c in rng.choice(cs, len(cs)):
            m = members[c]
            np.add.at(W[b], m[rng.integers(0, len(m), len(m))], 1)
    return W


def reml_meta(est, se, iters=200):
    """est、se：每个中心的 logit(AUC) 和标准误。返回 AUC 尺度的汇总值、95% CI、95% 预测区间、I²、τ²、k。"""
    y, v = np.asarray(est, float), np.asarray(se, float) ** 2
    k = len(y)
    w = 1 / v
    th_fe = (w * y).sum() / w.sum()
    Q = float((w * (y - th_fe) ** 2).sum())
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))      # DL 起点
    for _ in range(iters):
        wr = 1 / (v + tau2)
        th = (wr * y).sum() / wr.sum()
        new = max(0.0, float((wr ** 2 * ((y - th) ** 2 - v)).sum() / (wr ** 2).sum() + 1 / wr.sum()))
        if abs(new - tau2) < 1e-12:
            tau2 = new
            break
        tau2 = new
    wr = 1 / (v + tau2)
    th = float((wr * y).sum() / wr.sum())
    se_th = float(np.sqrt(1 / wr.sum()))
    t = stats.t.ppf(0.975, max(k - 2, 1))
    ex = lambda x: float(1 / (1 + np.exp(-x)))          # noqa: E731
    return {"k": k, "pooled": ex(th), "ci": [ex(th - 1.96 * se_th), ex(th + 1.96 * se_th)],
            "pi": [ex(th - t * np.sqrt(tau2 + se_th ** 2)), ex(th + t * np.sqrt(tau2 + se_th ** 2))],
            "tau2_logit": tau2, "I2": float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0}


def spearman(a, b):
    return float(stats.spearmanr(a, b).correlation)
