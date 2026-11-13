"""tabular.py -- 站点表上的基线：每个 (人, 访视) 一行，只用到这次访视为止的信息（landmarking，van Houwelingen 2007）。

两类：
  分类器（每道题 × 每个时间窗单独一个二分类器，和 TADPOLE Frog 的 L2C-XGBoost 一样逐个目标训练）：
    逻辑回归、随机森林、XGBoost、LightGBM。60 个模型 = 6 道题 × 1..10 年，只在"答案已知"的站点上训练。
    不保证对 H 单调（照原文的做法）。
  生存模型（每道题一个，给出整条生存曲线 S(H | 站点)，风险 = 1 − S(H)，天然对 H 单调）：
    Cox（lifelines，岭惩罚）、随机生存森林（scikit-survival）、梯度提升 Cox（XGBoost survival:cox + Breslow 基线）、
    DeepSurv、DeepHit、Logistic-Hazard（Nnet-survival）、PMF、N-MTLR（pycox 的损失函数，网络和训练循环在这里）。
    事件时间是逐年标签推出来的离散时间（bl_data.tte）；死亡对其它题是删失（cause-specific），和主口径
    "窗口内先死亡的站点不计入"一致。
都在 train 上拟合，在 test 上挑超参 / 早停，val 只预测一次。
"""
import time

import numpy as np
from joblib import Parallel, delayed

import bl_data as B

# ----------------------------------------------------------------------------- 预处理


class Prep:
    """训练段上定的标准化：去掉常数列，z 分数。"""

    def __init__(self, F):
        sd = F.std(0)
        self.keep = sd > 1e-8
        self.mu = F[:, self.keep].mean(0)
        self.sd = sd[self.keep]

    def __call__(self, F):
        return ((F[:, self.keep] - self.mu) / self.sd).astype(np.float32)


# ----------------------------------------------------------------------------- 分类器


def make_clf(kind, hp, seed, n_jobs):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=hp["C"], max_iter=3000)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=hp["leaf"], max_features="sqrt",
                                      n_jobs=n_jobs, random_state=seed)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=2000, learning_rate=hp["lr"], max_depth=hp["depth"], subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, tree_method="hist",
                                 early_stopping_rounds=50, eval_metric="logloss", n_jobs=n_jobs, random_state=seed)
    if kind == "tabpfn":
        # TabPFN v2（Hollmann et al., Nature 2025）：默认设置、不调参（原文的用法），只在本地 GPU 上推理（不用云端接口）
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cuda", random_state=seed)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=2000, learning_rate=hp["lr"], num_leaves=hp["leaves"],
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                                  n_jobs=n_jobs, random_state=seed, verbose=-1)
    raise ValueError(kind)


def _fit_one(kind, hp, seed, n_jobs, Ftr, ytr, Fte, yte, Fs, ret_model=False):
    m = make_clf(kind, hp, seed, n_jobs)
    if kind == "xgb":
        m.fit(Ftr, ytr, eval_set=[(Fte, yte)], verbose=False)
    elif kind == "lgbm":
        import lightgbm as lgb
        m.fit(Ftr, ytr, eval_set=[(Fte, yte)], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif kind == "tabpfn":
        # v2 的训练集上限是 1 万行：超过就按固定种子随机抽 1 万行（不看标签分层）
        if len(Ftr) > TABPFN_MAX:
            k = np.random.default_rng(seed).choice(len(Ftr), TABPFN_MAX, replace=False)
            Ftr, ytr = Ftr[k], ytr[k]
        m.fit(Ftr, ytr)
    else:
        m.fit(Ftr, ytr)
    preds = [m.predict_proba(F)[:, 1] for F in Fs]
    return (preds, m) if ret_model else preds


def per_target(kind, hp, tr, te, va, seed=0, n_jobs=16, extra=(), keep=None):
    """60 个二分类器 -> test、val（和 extra 里每一批行）上的 (行, 6, 10) 风险。
    keep（dict）：另外交回训练好的模型（keep["prep"]、keep["models"] = [(题, 时间窗, 模型)]），给外部验证存盘用。"""
    prep = Prep(tr["F"])
    Ftr, Fte, Fva = prep(tr["F"]), prep(te["F"]), prep(va["F"])
    Fex = [prep(F) for F in extra]
    jobs = []
    for qi in range(B.NQ):
        for hi in range(B.NH):
            a, b = tr["Y"][:, qi, hi], te["Y"][:, qi, hi]
            ka, kb = a >= 0, b >= 0
            jobs.append((qi, hi, Ftr[ka], a[ka], Fte[kb], b[kb]))
    # 树模型自己多线程；逻辑回归按目标并行；TabPFN 一块 GPU 逐个来
    outer, inner = (n_jobs, 1) if kind == "lr" else ((1, 1) if kind == "tabpfn" else (max(1, n_jobs // 4), 4))
    res = Parallel(n_jobs=outer)(delayed(_fit_one)(kind, hp, seed, inner, a, ya, b, yb, (Fte, Fva, *Fex), keep is not None)
                                 for (_, _, a, ya, b, yb) in jobs)
    if keep is not None:
        keep["prep"] = prep
        keep["models"] = [(qi, hi, m) for (qi, hi, *_), (_, m) in zip(jobs, res)]
        res = [p for p, _ in res]
    out = [np.zeros((len(F), B.NQ, B.NH)) for F in (Fte, Fva, *Fex)]
    for (qi, hi, *_), preds in zip(jobs, res):
        for R, p in zip(out, preds):
            R[:, qi, hi] = p
    return out


TABPFN_MAX = 10_000
CLF_GRID = {"lr": [{"C": c} for c in (0.01, 0.1, 1.0)],
            "tabpfn": [{}],
            "rf": [{"leaf": k} for k in (5, 20, 50)],
            "xgb": [{"lr": 0.03, "depth": d} for d in (3, 5)],
            "lgbm": [{"lr": 0.03, "leaves": k} for k in (15, 63)]}


def run_classifier(kind, featset, Z, log, extras=(), keep=None):
    """keep（dict）：选中那组超参的模型交回到 keep 里（per_target 的 keep + hp、featset）。"""
    tr, te, va = (B.landmarks(B.split(Z, s), featset) for s in ("train", "test", "val"))
    Fex = [B.at(Dp, vL, featset) for Dp, vL in extras]
    best = None
    for hp in CLF_GRID[kind]:
        t0 = time.time()
        kp = {} if keep is not None else None
        Rte, Rva, *Rex = per_target(kind, hp, tr, te, va, extra=Fex, keep=kp)
        score = B.macro_auc(te["Y"], Rte)
        log.append({"hp": hp, "test_macro_auc": score, "test_bce": B.masked_bce(te["Y"], Rte),
                    "minutes": (time.time() - t0) / 60})
        print(f"  {kind}-{featset} {hp}: test macro AUC {score:.4f}", flush=True)
        if best is None or score > best[0]:
            best = (score, hp, Rte, Rva, Rex, kp)
    if keep is not None:
        keep.update(best[5], hp=best[1], featset=featset)
    return {"val": (va, best[3]), "test": (te, best[2]), "chosen": best[1], "extras": best[4]}


# ----------------------------------------------------------------------------- 生存模型


def breslow(T, E, lp, times=B.HS):
    """Breslow 累计基线风险 H0(t)，t = 1..10（离散整年时间，事件按年并列）。lp：线性预测值（log 风险比）。"""
    r = np.exp(lp - lp.max())
    H0, acc = [], 0.0
    for t in times:
        d = ((T == t) & (E == 1)).sum()
        at = r[T >= t].sum()
        acc += d / at if at > 0 else 0.0
        H0.append(acc)
    return np.array(H0), lp.max()


def cox_surv(H0, shift, lp):
    return np.exp(-np.exp(lp - shift)[:, None] * H0[None, :])


def cox_ridge(F, T, E, pen):
    """岭惩罚 Cox：平均 Breslow 部分似然 + 0.5·pen·‖β‖²，L-BFGS。（lifelines 的 Newton 在这批特征上溢出成 nan，
    所以自己写；离散整年时间只有 10 个，风险集逐年算。）"""
    from scipy.optimize import minimize
    F = F.astype(np.float64)
    N = len(F)
    times = [t for t in B.HS if ((T == t) & (E == 1)).any()]
    R = [T >= t for t in times]
    Dm = [(T == t) & (E == 1) for t in times]
    dsum = [F[d].sum(0) for d in Dm]
    nd = [d.sum() for d in Dm]

    def f(beta):
        lp = F @ beta
        m = lp.max()
        w = np.exp(lp - m)
        loss, grad = 0.0, np.zeros_like(beta)
        for r, d, ds, k in zip(R, Dm, dsum, nd):
            s0 = w[r].sum()
            s1 = F[r].T @ w[r]
            loss -= lp[d].sum() - k * (np.log(s0) + m)
            grad -= ds - k * s1 / s0
        return loss / N + 0.5 * pen * beta @ beta, grad / N + pen * beta

    res = minimize(f, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return res.x


def surv_cox(F_tr, T, E, Fs, hp):
    beta = cox_ridge(F_tr, T, E, hp["pen"])
    lp_tr = F_tr @ beta
    H0, shift = breslow(T, E, lp_tr)
    return [cox_surv(H0, shift, F @ beta) for F in Fs]


def surv_rsf(F_tr, T, E, Fs, hp, seed=0, n_jobs=16):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    m = RandomSurvivalForest(n_estimators=hp["trees"], min_samples_leaf=hp["leaf"], max_features="sqrt",
                             max_samples=hp.get("max_samples"), n_jobs=n_jobs, random_state=seed, low_memory=False)
    m.fit(F_tr, Surv.from_arrays(E.astype(bool), T.astype(float)))
    ut = m.unique_times_
    out = []
    for F in Fs:
        S = m.predict_survival_function(F, return_array=True)       # (n, len(ut))
        # 第 H 年的生存 = 最后一个 <= H 的事件时间上的值（阶梯函数）
        idx = np.searchsorted(ut, np.array(B.HS, float), side="right") - 1
        out.append(np.where(idx[None, :] >= 0, S[:, np.maximum(idx, 0)], 1.0))
    return out


def surv_xgbcox(F_tr, T, E, F_te_T_E, Fs, hp, seed=0, n_jobs=16):
    import xgboost as xgb
    y = np.where(E == 1, T, -T).astype(float)
    Fte, Tte, Ete = F_te_T_E
    dtr = xgb.DMatrix(F_tr, label=y)
    dte = xgb.DMatrix(Fte, label=np.where(Ete == 1, Tte, -Tte).astype(float))
    p = {"objective": "survival:cox", "eta": hp["lr"], "max_depth": hp["depth"], "subsample": 0.8,
         "colsample_bytree": 0.8, "eval_metric": "cox-nloglik", "nthread": n_jobs, "seed": seed}
    b = xgb.train(p, dtr, 3000, evals=[(dte, "test")], early_stopping_rounds=50, verbose_eval=False)
    rng = (0, b.best_iteration + 1)
    lp_tr = np.log(b.predict(dtr, iteration_range=rng))
    H0, shift = breslow(T, E, lp_tr)
    return [cox_surv(H0, shift, np.log(b.predict(xgb.DMatrix(F), iteration_range=rng))) for F in Fs]


# --- 深度生存（pycox 的损失；网络 = pycox 示例的默认结构：MLP 两层 32，BatchNorm + ReLU + dropout 0.1，batch 256，lr 0.01）


def _mlp(n_in, n_out, width=32, depth=2, dropout=0.1):
    import torch.nn as nn
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def _surv_from_out(kind, out):
    import torch
    if kind == "logistichazard":
        h = torch.sigmoid(out)
        return torch.cumprod(1 - h, 1)
    # pycox 的 PMF 类损失在最后补一列 0（"10 年之后"），这里同样补上再 softmax
    from pycox.models.utils import cumsum_reverse, pad_col
    if kind in ("pmf", "deephit"):
        pmf = torch.softmax(pad_col(out), 1)[:, :B.NH]
        return 1 - torch.cumsum(pmf, 1)
    if kind == "mtlr":
        pmf = torch.softmax(pad_col(cumsum_reverse(out, dim=1)), 1)[:, :B.NH]
        return 1 - torch.cumsum(pmf, 1)
    raise ValueError(kind)


def surv_deep(kind, F_tr, T, E, te, Fs, hp, seed=0, device="cpu", max_epochs=200, patience=10):
    """kind ∈ deepsurv / logistichazard / pmf / mtlr / deephit。te = (F, T, E) 早停用。"""
    import torch
    from pycox.models import loss as PL
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_out = 1 if kind == "deepsurv" else B.NH
    net = _mlp(F_tr.shape[1], n_out, dropout=hp.get("dropout", 0.1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp.get("lr", 1e-2))
    if kind == "deephit":
        crit = PL.DeepHitSingleLoss(alpha=hp.get("alpha", 0.2), sigma=hp.get("sigma", 0.1))

    def loss_fn(o, t, e):
        idx = t - 1                                       # 事件落在第 t 年 = 第 t-1 个区间
        if kind == "deepsurv":
            return PL.cox_ph_loss(o[:, 0], t.float(), e.float())
        if kind == "logistichazard":
            return PL.nll_logistic_hazard(o, idx, e.float())
        if kind == "pmf":
            return PL.nll_pmf(o, idx, e.float())
        if kind == "mtlr":
            return PL.nll_mtlr(o, idx, e.float())
        from pycox.models.data import pair_rank_mat
        rm = torch.from_numpy(pair_rank_mat(idx.cpu().numpy(), e.cpu().numpy())).to(o.device)
        return crit(o, idx, e.float(), rm)

    to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)   # noqa: E731
    Xtr, Ttr, Etr = to(F_tr), to(T, torch.long), to(E, torch.long)
    Xte, Tte, Ete = to(te[0]), to(te[1], torch.long), to(te[2], torch.long)
    bs = 256
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        net.train()
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm), bs):
            b = torch.as_tensor(perm[s: s + bs], device=device)
            if len(b) < 2:
                continue
            l = loss_fn(net(Xtr[b]), Ttr[b], Etr[b])
            opt.zero_grad()
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            # 早停看 test 段的损失（DeepSurv 的部分似然在全 test 段上算；其余分批平均）
            if kind == "deepsurv":
                vl = float(loss_fn(net(Xte), Tte, Ete))
            else:
                vl = float(np.mean([float(loss_fn(net(Xte[s: s + 2048]), Tte[s: s + 2048], Ete[s: s + 2048]))
                                    for s in range(0, len(Xte), 2048)]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        if kind == "deepsurv":
            H0, shift = breslow(T, E, net(Xtr)[:, 0].cpu().numpy().astype(float))

    @torch.no_grad()
    def surv_of(F):
        if kind == "deepsurv":
            return cox_surv(H0, shift, net(to(F))[:, 0].cpu().numpy().astype(float))
        return _surv_from_out(kind, net(to(F))).cpu().numpy()

    if Fs is None:                        # rollout 版要的是模型本身：返回 F -> S(1..10) 的函数
        return surv_of, ep + 1
    return [surv_of(F) for F in Fs], ep + 1


def cr_targets(Yl, q):
    """站点级的竞争风险标签：原因 1 = 这道题，2 = 死亡先发生（和 sequence.ddh_targets 同一条规则）。"""
    qi = B.Q.index(q)
    Tq, Eq, ok = B.tte(Yl[:, qi])
    if q == "death":
        return Tq, Eq, ok
    Td, Ed, _ = B.tte(Yl[:, B.I_DEATH])
    dfirst = (Eq == 0) & (Ed == 1) & ok & (Td <= Tq + 1)
    return np.where(dfirst, Td, Tq), np.where(Eq == 1, 1, np.where(dfirst, 2, 0)), ok


def surv_survivalboost(F_tr, T, C, Fs, hp, seed=0):
    """SurvivalBoost（Alberge et al., AISTATS 2025；hazardous 包，默认超参）：竞争风险的梯度提升，直接给每个原因的累计发生函数。"""
    import pandas as pd
    from hazardous import SurvivalBoost
    m = SurvivalBoost(n_iter=hp.get("n_iter", 100), learning_rate=hp.get("lr", 0.05), show_progressbar=False,
                      random_state=seed)
    m.fit(F_tr, pd.DataFrame({"event": C.astype(int), "duration": T.astype(float)}))
    out = []
    for F in Fs:
        cif = np.asarray(m.predict_cumulative_incidence(F, times=np.array(B.HS, float)))
        out.append(1 - cif[:, 1, :] if cif.ndim == 3 else 1 - cif)       # 返回"生存"（1 − 原因 1 的累计发生），和其它生存模型一样
    return out


def km_censor(T, C):
    """删失分布的 Kaplan–Meier：G[k] = P(删失时刻 > k)，k = 0..10（离散整年；同一年里事件排在删失之前）。"""
    G = [1.0]
    for t in B.HS:
        at = (T >= t).sum()
        c = ((T == t) & (C == 0)).sum()
        G.append(G[-1] * (1 - c / at) if at > 0 else G[-1])
    return np.array(G)


def fg_weights(T, C):
    """Fine–Gray 的风险集权重：第 t 年（有原因 1 发生的年份）的风险集 = 还没出事的人（权重 1）+ 更早死于竞争事件的人
    （IPCW 权重 G(t−) / G(T_j−)，Fine & Gray 1999 式 (5)）。返回 [(t, 权重向量, 这一年原因 1 发生的人)]。"""
    G = km_censor(T, C)
    out = []
    for t in B.HS:
        d = (T == t) & (C == 1)
        if not d.any():
            continue
        w = (T >= t).astype(np.float64)
        comp = (C == 2) & (T < t)
        w[comp] = G[t - 1] / np.maximum(G[T[comp] - 1], 1e-12)
        out.append((t, w, d))
    return out


def fg_ridge(F, T, C, pen):
    """岭惩罚的 Fine–Gray 部分似然（Breslow 并列），L-BFGS；和 cox_ridge 同一种写法，风险集换成加权的子分布风险集。"""
    from scipy.optimize import minimize
    F = F.astype(np.float64)
    N = len(F)
    WD = fg_weights(T, C)
    dsum = [F[d].sum(0) for _, _, d in WD]

    def f(beta):
        lp = F @ beta
        m = lp.max()
        e = np.exp(lp - m)
        loss, grad = 0.0, np.zeros_like(beta)
        for (_, w, d), ds in zip(WD, dsum):
            we = w * e
            s0 = we.sum()
            k = d.sum()
            loss -= lp[d].sum() - k * (np.log(s0) + m)
            grad -= ds - k * (F.T @ we) / s0
        return loss / N + 0.5 * pen * beta @ beta, grad / N + pen * beta

    beta = minimize(f, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 2000}).x
    lp = F @ beta
    shift = lp.max()
    e = np.exp(lp - shift)
    by_t = {t: (w, d) for t, w, d in WD}
    H0, acc = [], 0.0
    for t in B.HS:                                   # 子分布的累计基线风险（Breslow 型估计）
        if t in by_t:
            w, d = by_t[t]
            acc += d.sum() / (w * e).sum()
        H0.append(acc)
    return beta, np.array(H0), shift


def surv_fg(F_tr, T, C, Fs, hp):
    """返回"1 − 累计发生函数"（和其它生存模型同一种输出：风险 = 1 − 它）。CIF(t | x) = 1 − exp(−Λ₁(t) e^{xβ})。"""
    beta, H0, shift = fg_ridge(F_tr, T, C, hp["pen"])
    return [cox_surv(H0, shift, F @ beta) for F in Fs]


def _cox_grad_hess(y, eta):
    """Cox 部分似然（Breslow 并列）对每个人线性预测值的梯度和 Hessian 对角，和 XGBoost survival:cox 同一个式子。
    y 的编码也照 XGBoost：发生 = +T，删失 = −T（T = 离散整年 1..10）。"""
    T, E = np.abs(y).astype(int), (y > 0)
    e = np.exp(eta - eta.max())
    grad, hess = -E.astype(float), np.zeros(len(y))
    for t in B.HS:
        d = ((T == t) & E).sum()
        if not d:
            continue
        risk = T >= t
        p = np.where(risk, e / e[risk].sum(), 0.0)
        grad += d * p
        hess += d * (p - p ** 2)
    return grad, np.maximum(hess, 1e-12)


def _cox_nll(y, eta):
    T, E = np.abs(y).astype(int), (y > 0)
    m = eta.max()
    ll = 0.0
    for t in B.HS:
        d = (T == t) & E
        if d.any():
            ll += eta[d].sum() - d.sum() * (np.log(np.exp(eta[T >= t] - m).sum()) + m)
    return -ll / max(E.sum(), 1)


def surv_lgbmcox(F_tr, T, E, F_te_T_E, Fs, hp, seed=0, n_jobs=16, return_model=False):
    """LightGBM 没有内置的生存目标：用自定义目标函数做梯度提升 Cox（_cox_grad_hess），test 段的部分似然早停，Breslow 基线。"""
    import lightgbm as lgb
    Fte, Tte, Ete = F_te_T_E
    y, yte = np.where(E == 1, T, -T).astype(float), np.where(Ete == 1, Tte, -Tte).astype(float)
    m = lgb.LGBMRegressor(objective=lambda yt, yp: _cox_grad_hess(yt, yp), n_estimators=3000, learning_rate=hp["lr"],
                          num_leaves=hp["leaves"], subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                          metric="None", n_jobs=n_jobs, random_state=seed, verbose=-1)    # 只用下面的部分似然早停
    m.fit(F_tr, y, eval_set=[(Fte, yte)], eval_metric=lambda yt, yp: ("cox_nll", _cox_nll(yt, yp), False),
          callbacks=[lgb.early_stopping(50, verbose=False)])
    lp = lambda F: m.predict(F, raw_score=True)                        # noqa: E731
    H0, shift = breslow(T, E, lp(F_tr))
    surv_of = lambda F: cox_surv(H0, shift, lp(F))                     # noqa: E731
    if return_model:
        return surv_of
    return [surv_of(F) for F in Fs]


def xgbaft_dmatrix(F, T, E):
    """离散整年的事件时间 -> 区间删失：第 T 年内发生 = (T−1, T]；到第 T 年还没发生 = [T, ∞)。"""
    import xgboost as xgb
    dm = xgb.DMatrix(F)
    dm.set_float_info("label_lower_bound", np.where(E == 1, T - 1, T).astype(float))
    dm.set_float_info("label_upper_bound", np.where(E == 1, T, np.inf).astype(float))
    return dm


def aft_surv(mu, dist, scale, times=B.HS):
    """AFT：log T = μ(x) + σ ε。S(t) = 1 − F_ε((log t − μ) / σ)；extreme = Gumbel（Weibull AFT）。"""
    from scipy.stats import norm
    z = (np.log(np.asarray(times, float))[None, :] - mu[:, None]) / scale
    cdf = {"normal": norm.cdf(z), "logistic": 1 / (1 + np.exp(-z)), "extreme": 1 - np.exp(-np.exp(np.clip(z, -30, 30)))}[dist]
    return 1 - cdf


def surv_xgbaft(F_tr, T, E, F_te_T_E, Fs, hp, seed=0, n_jobs=16, return_model=False):
    import xgboost as xgb
    Fte, Tte, Ete = F_te_T_E
    dtr, dte = xgbaft_dmatrix(F_tr, T, E), xgbaft_dmatrix(Fte, Tte, Ete)
    p = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": hp["dist"],
         "aft_loss_distribution_scale": hp["scale"], "eta": hp.get("lr", 0.03), "max_depth": hp.get("depth", 3),
         "subsample": 0.8, "colsample_bytree": 0.8, "nthread": n_jobs, "seed": seed}
    b = xgb.train(p, dtr, 3000, evals=[(dte, "test")], early_stopping_rounds=50, verbose_eval=False)
    rng = (0, b.best_iteration + 1)
    surv_of = lambda F: aft_surv(b.predict(xgb.DMatrix(F), iteration_range=rng, output_margin=True),   # noqa: E731
                                 hp["dist"], hp["scale"])
    if return_model:
        return surv_of
    return [surv_of(F) for F in Fs]


SURV_GRID = {"survivalboost": [{}], "cox": [{"pen": p} for p in (1e-3, 1e-2, 1e-1)],
             "fg": [{"pen": p} for p in (1e-3, 1e-2, 1e-1)],
             "lgbmcox": [{"lr": 0.03, "leaves": k} for k in (15, 63)],
             "xgbaft": [{"dist": d, "scale": s} for d in ("normal", "logistic", "extreme") for s in (0.5, 1.0)],
             "rsf": [{"trees": 1000, "leaf": 16}],                 # de Swart 2025 的设置
             "xgbcox": [{"lr": 0.03, "depth": d} for d in (3, 5)],
             "deepsurv": [{"lr": 1e-2}], "logistichazard": [{"lr": 1e-2}], "pmf": [{"lr": 1e-2}],
             "mtlr": [{"lr": 1e-2}], "deephit": [{"lr": 1e-2, "alpha": 0.2, "sigma": 0.1}]}


def run_survival(kind, featset, Z, log, device="cpu", extras=()):
    tr, te, va = (B.landmarks(B.split(Z, s), featset) for s in ("train", "test", "val"))
    prep = Prep(tr["F"])
    Ftr, Fte, Fva = prep(tr["F"]), prep(te["F"]), prep(va["F"])
    Fex = [prep(B.at(Dp, vL, featset)) for Dp, vL in extras]
    Rte = np.zeros((len(Fte), B.NQ, B.NH))
    Rva = np.zeros((len(Fva), B.NQ, B.NH))
    Rex = [np.zeros((len(F), B.NQ, B.NH)) for F in Fex]
    chosen = {}
    for qi, q in enumerate(B.Q):
        T, E, ok = B.tte(tr["Y"][:, qi])
        Tt, Et, okt = B.tte(te["Y"][:, qi])
        best = None
        for hp in SURV_GRID[kind]:
            t0 = time.time()
            if kind == "cox" or (kind == "fg" and q == "death"):           # Fine–Gray：死亡这道题本身没有竞争事件，就是 Cox
                S_te, S_va, *S_ex = surv_cox(Ftr[ok], T[ok], E[ok], (Fte, Fva, *Fex), hp)
                extra = {}
            elif kind == "fg":
                Tc, Cc, okc = cr_targets(tr["Y"], q)
                S_te, S_va, *S_ex = surv_fg(Ftr[okc], Tc[okc], Cc[okc], (Fte, Fva, *Fex), hp)
                extra = {"n_competing": int((Cc[okc] == 2).sum())}
            elif kind == "xgbaft":
                S_te, S_va, *S_ex = surv_xgbaft(Ftr[ok], T[ok], E[ok], (Fte[okt], Tt[okt], Et[okt]), (Fte, Fva, *Fex), hp)
                extra = {}
            elif kind == "lgbmcox":
                S_te, S_va, *S_ex = surv_lgbmcox(Ftr[ok], T[ok], E[ok], (Fte[okt], Tt[okt], Et[okt]), (Fte, Fva, *Fex), hp)
                extra = {}
            elif kind == "survivalboost":
                Tc, Cc, okc = cr_targets(tr["Y"], q)
                S_te, S_va, *S_ex = surv_survivalboost(Ftr[okc], Tc[okc], Cc[okc], (Fte, Fva, *Fex), hp)
                extra = {}
            elif kind == "rsf":
                S_te, S_va, *S_ex = surv_rsf(Ftr[ok], T[ok], E[ok], (Fte, Fva, *Fex), hp)
                extra = {}
            elif kind == "xgbcox":
                S_te, S_va, *S_ex = surv_xgbcox(Ftr[ok], T[ok], E[ok], (Fte[okt], Tt[okt], Et[okt]), (Fte, Fva, *Fex), hp)
                extra = {}
            else:
                (S_te, S_va, *S_ex), n_ep = surv_deep(kind, Ftr[ok], T[ok], E[ok], (Fte[okt], Tt[okt], Et[okt]),
                                                      (Fte, Fva, *Fex), hp, device=device)
                extra = {"epochs": n_ep}
            r_te = 1 - S_te
            sc = np.nanmean([B.masked_auc(te["Y"][:, qi, H - 1], r_te[:, H - 1]) for H in (2, 5, 10)])
            log.append({"q": q, "hp": hp, "test_auc_2_5_10": float(sc), "minutes": (time.time() - t0) / 60, **extra})
            print(f"  {kind}-{featset} {q} {hp}: test AUC {sc:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if best is None or sc > best[0]:
                best = (sc, hp, r_te, 1 - S_va, [1 - x for x in S_ex])
        chosen[q] = best[1]
        Rte[:, qi], Rva[:, qi] = best[2], best[3]
        for R, x in zip(Rex, best[4]):
            R[:, qi] = x
    return {"val": (va, Rva), "test": (te, Rte), "chosen": chosen, "extras": Rex}


# ----------------------------------------------------------------------------- 固定权重的风险评分（不训练）


def _raw_last(lm_D, j, norm):
    """最近一次测到的原始值（手工历史特征的"最近值"还原），没测过 = nan。"""
    D, ii, vv = lm_D
    L = D["L"][ii, vv]
    raw = L[:, j * 8] * norm["sd"][j] + norm["mu"][j]
    return np.where(L[:, j * 8 + 1] > 0.5, raw, np.nan)


def risk_score(kind, D, norm):
    """站点上的评分（越高风险越高）。缺失的项记 0 分（文献验证里通用的做法）。原文变量和这里的对应见 README。"""
    ii, vv = np.where(D["M"])
    x, S = D["X"][ii, vv], D["S"][ii]
    age = 80.0 + 10.0 * x[:, B.I_TF]
    male = S[:, 0] > 0.5
    educ = np.where(S[:, 2] > 0.5, np.nan, 16.0 + 4.0 * S[:, 1])
    dm = np.maximum(np.where(x[:, B.I_MS + 1] < 0.5, x[:, B.I_XS + 1], 0.0), S[:, 21]) > 0.5
    dep = x[:, B.I_EVER + 1] > 0.5
    stroke = x[:, B.I_EVER + 0] > 0.5
    names = norm["names"]
    lmD = (D, ii, vv)
    bmi = _raw_last(lmD, names.index("bmi"), norm)
    sbp = _raw_last(lmD, names.index("sbp_avg"), norm)
    tc = _raw_last(lmD, names.index("ldlchlstrl"), norm) + _raw_last(lmD, names.index("hdlchlstrl"), norm)
    f = lambda c: np.nan_to_num(c.astype(float), nan=0.0)      # noqa: E731
    if kind == "bdsi":
        # Barnes et al. 2014：年龄每岁 1 分（65 岁起；> 79 岁照样逐岁加，CHAP 验证的做法）、教育 < 12 年 9、卒中 6、
        # 糖尿病 3、BMI < 18.5 8、抑郁 6；"需要别人帮忙管钱 / 管药"10 分 ROSMAP 这份数据里没有，记 0
        return (np.maximum(age - 65, 0) + 9 * f(educ < 12) + 6 * stroke + 3 * dm + 8 * f(bmi < 18.5) + 6 * dep)
    if kind == "anuadri":
        # Anstey et al. 2014 表 2，MAP 验证用的"共同变量"（年龄 × 性别、教育、糖尿病、吸烟、饮酒）+ 抑郁
        bands = np.array([65, 70, 75, 80, 85, 90])
        pm = np.array([0, 1, 12, 18, 26, 33, 38])
        pw = np.array([0, 5, 14, 21, 29, 35, 41])
        k = np.searchsorted(bands, age, side="right")
        pts = np.where(male, pm[k], pw[k]).astype(float)
        pts += np.where(educ < 8, 6, np.where(educ <= 11, 3, 0)) * np.isfinite(educ)
        pts += 3 * dm + 2 * dep + 1 * (S[:, 13] > 0.5) + 4 * (S[:, 14] > 0.5)
        pts -= 3 * ((S[:, 16] > 0) & (S[:, 17] < 0.5))          # 少量到中等饮酒（这批人几乎没有重度饮酒）
        return pts
    if kind == "caide":
        # Kivipelto et al. 2006（中年 20 年痴呆风险）：年龄 > 53 一律 4 分（这批人都是），教育 0–6 / 7–9 / ≥ 10 年 = 3 / 2 / 0、
        # 男 1、收缩压 > 140 2、BMI > 30 2、总胆固醇 > 6.5 mmol/L（251 mg/dL；用 LDL + HDL 近似）2；没有体力活动这一项
        return (4 + np.where(educ <= 6, 3, np.where(educ <= 9, 2, 0)) * np.isfinite(educ) + 1 * male
                + 2 * f(sbp > 140) + 2 * f(bmi > 30) + 2 * f(tc > 251))
    raise ValueError(kind)


def run_score(kind, Z, meta, log, extras=()):
    """固定权重 -> 每道题 × 时间窗一个单变量逻辑回归（只把分数换算成概率：截距和斜率在 train 上定，斜率限定 ≥ 0，
    排序 = 原评分的排序，AUC 不受换算影响）。"""
    from sklearn.linear_model import LogisticRegression
    import simulator as SIM
    norm = SIM.norm_info(meta)
    out = {}
    sc = {s: risk_score(kind, B.split(Z, s), norm) for s in ("train", "test", "val")}
    for k, (Dp, vL) in enumerate(extras):
        one = dict(Dp, M=np.eye(Dp["M"].shape[1], dtype=bool)[vL])      # 只取站点那一行（np.where 的顺序 = 记录顺序）
        sc[f"ex{k}"] = risk_score(kind, one, norm)
    lms = {s: B.landmarks(B.split(Z, s), "cur") for s in ("train", "test", "val")}
    mu, sd = sc["train"].mean(), sc["train"].std() + 1e-8
    R = {s: np.zeros((len(sc[s]), B.NQ, B.NH)) for s in sc if s != "train"}
    slopes = []
    for qi in range(B.NQ):
        for hi in range(B.NH):
            y = lms["train"]["Y"][:, qi, hi]
            k = y >= 0
            z = ((sc["train"] - mu) / sd)[k, None]
            m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y[k])
            a, b = float(m.intercept_[0]), max(float(m.coef_[0, 0]), 1e-6)
            slopes.append(float(m.coef_[0, 0]))
            for s in R:
                R[s][:, qi, hi] = 1 / (1 + np.exp(-(a + b * (sc[s] - mu) / sd)))
    log.append({"score_mean_train": float(mu), "score_sd_train": float(sd),
                "n_negative_slopes": int(sum(v < 0 for v in slopes))})
    out["val"], out["test"] = (lms["val"], R["val"]), (lms["test"], R["test"])
    out["chosen"] = {"score": kind}
    out["extras"] = [R[f"ex{k}"] for k in range(len(extras))]
    return out
