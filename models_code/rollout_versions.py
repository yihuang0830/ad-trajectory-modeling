"""rollout_versions.py -- 把不能 rollout 的模型，各做一个能 rollout 的版本（参数化 g-formula / 微观模拟的做法）。

一句话：不让模型直接回答"H 年内会不会发生"，让它学"明年会怎样"，再一年一年往后滚（simulator.simulate_visits，
和 ar-lr、MinimalRNN 同一个步进规则）。每一步要两样东西：
  1. 这一年里死亡、得 AD 的概率（"风险"）；
  2. 活过这一年的话，明年那次访视的分布（"随访指标"：15 个连续量各一个高斯 + 是否缺失；9 个状态 + 是否缺失；卒中 / 抑郁新出现）。
原模型是哪一类，这两样就用哪一类来学（超参照原模型在 test 段上选中的那组，从 packages/<原模型>/baseline.json 读）：

  分类器家族（lr / rf / xgb / tabpfn）  风险 = 同一种分类器在"1 年内死亡 / AD"上（就是原模型那 60 个分类器里的两个）；
                                          指标 = 同一种学习器的回归 / 分类，每个指标一个
  生存家族                              风险 = 原模型自己的 1 − S(1 年)；指标 = 同族的学习器：
                                          Cox -> 线性 / 逻辑回归；随机生存森林 -> 随机森林；XGBoost-Cox -> XGBoost；
                                          SurvivalBoost -> sklearn 的直方图梯度提升（它本身就建在这个上面）；
                                          DeepSurv / DeepHit / Logistic-Hazard / PMF / N-MTLR -> pycox 默认结构的 MLP（两层 32）
  临床评分（BDSI / ANU-ADRI / CAIDE）    风险 = 评分换算成 1 年概率（train 上的单变量逻辑回归，斜率 ≥ 0）；指标 = 线性 / 逻辑回归
  L2C-FNN                                原结构 + 一个"下一次访视分布"的输出头，和原来 60 个输出的风险头一起训练；
                                         风险 = 风险头的"1 年内死亡 / AD"
  transformer-sim                        同上（只留作 NACC 预训练实验 transformer-sim-nacc 的对照）

序列网络（GRU、LSTM、Transformer、GRU-D、T-LSTM、RETAIN、Dynamic-DeepHit；2026-10-28 改）本身就是自回归的，不再是"rollout 版"：
只学下一步——风险头只有"下一年内死亡 / AD"两个输出 + 下一次访视的分布，不加任何"H 年内会不会"的题目头；
把抽出来的访视接到序列后面，再过一遍网络。所有答案（各题各时间窗的风险、轨迹、扰动）都从 rollout 来（SeqARModel、DDHARModel）。

输入特征和原模型一样（demo / cur / l2c / cogn / super）；手工历史特征随模拟出来的访视逐步更新（simulator.L2C）。
"""
import json
import os
import time

import numpy as np

import bl_data as B
import simulator as SIM
import tabular as TB
from bl_data import I_AD, I_EVER, I_MC, I_MS, I_TF, I_XS, N_CONT, N_STATE
from common import RUN_DIR
from tabular import Prep

N_EV = I_AD - I_EVER
NEXT_SPLIT = (N_CONT, N_CONT, N_CONT, N_STATE, N_STATE, N_EV)     # mu, log_sd, miss_c, xs, miss_s, ever
NEXT_KEYS = ("mu", "log_sd", "miss_c", "xs", "miss_s", "ever")
N_NEXT = sum(NEXT_SPLIT)
SIG = lambda z: 1.0 / (1.0 + np.exp(-z))      # noqa: E731
IDX_D1 = B.I_DEATH * B.NH                     # 60 个风险输出里"1 年内死亡"的位置（题目外层、时间窗内层）
IDX_A1 = B.I_ADQ * B.NH


def chosen_hp(orig):
    p = os.path.join(RUN_DIR, "packages", orig, "baseline.json")
    return json.load(open(p))["chosen"] if os.path.exists(p) else {}


def feats(featset, x, S, L):
    """一批"当前访视"（模拟出来的也行）-> 和原模型同一种特征行。"""
    n = len(x)
    D = {"X": x[:, None], "S": S, "L": L[:, None]}
    return B.featrows(D, np.arange(n), np.zeros(n, int), featset)


def pairs(D, featset):
    """训练行：站点特征 + 下一次访视（间隔正好 1 年才算）。返回 (站点表, 下一次访视, 当前访视, 有下一次)。"""
    lm = B.landmarks(D, featset)
    X, M = D["X"], D["M"]
    ii, vv = lm["ii"], lm["vv"]
    V = X.shape[1]
    nxt = np.minimum(vv + 1, V - 1)
    ok = (vv + 1 < V) & M[ii, nxt] & (np.abs(X[ii, nxt, I_TF + 2] - 1.0) < 1e-6)
    return lm, X[ii, nxt], X[ii, vv], ok


def nd_from_logits(o):
    """next 头的原始输出 (n, 65) 或 dict -> simulator 要的分布。"""
    if not isinstance(o, dict):
        o = dict(zip(NEXT_KEYS, np.split(o, np.cumsum(NEXT_SPLIT)[:-1], axis=-1)))
    return {"mu": o["mu"], "sd": np.exp(o["log_sd"]).clip(1e-2), "miss_c": SIG(o["miss_c"]), "xs": SIG(o["xs"]),
            "miss_s": SIG(o["miss_s"]), "ever": SIG(o["ever"])}


def next_nll(o, nxt, cur, valid):
    """下一次访视的负对数似然（torch）：连续量高斯（只算测到的）+ 是否缺失 + 状态（只算测到的）+ 新出现（只算还没有的）。"""
    import torch
    import torch.nn.functional as F
    bce = lambda z, y, m: F.binary_cross_entropy_with_logits(z[m], y[m]) if m.any() else z.sum() * 0   # noqa: E731
    v = valid[..., None]
    oc = (nxt[..., I_MC:I_XS] < 0.5) & v
    sd = torch.exp(o["log_sd"]).clamp(min=1e-2)
    l = (((o["mu"] - nxt[..., :N_CONT]) / sd) ** 2 / 2 + torch.log(sd))[oc].mean() if oc.any() else o["mu"].sum() * 0
    l = l + bce(o["miss_c"], nxt[..., I_MC:I_XS], v.expand_as(o["miss_c"]))
    l = l + bce(o["miss_s"], nxt[..., I_MS:I_EVER], v.expand_as(o["miss_s"]))
    l = l + bce(o["xs"], nxt[..., I_XS:I_MS], (nxt[..., I_MS:I_EVER] < 0.5) & v)
    l = l + bce(o["ever"], nxt[..., I_EVER:I_AD], (cur[..., I_EVER:I_AD] < 0.5) & v)
    return l


def split_next(z):
    import torch
    return dict(zip(NEXT_KEYS, torch.split(z, list(NEXT_SPLIT), -1)))


# ================================================================ 学习器（分类 / 回归），和原模型同一种


def make_learner(kind, hp, task, seed=0, n_jobs=8):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression, Ridge
        return LogisticRegression(C=hp.get("C", 0.1), max_iter=3000) if task == "clf" else Ridge(alpha=1.0)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task == "clf" else RandomForestRegressor
        return cls(n_estimators=hp.get("trees", 300), min_samples_leaf=hp.get("leaf", 20), max_features="sqrt",
                   n_jobs=n_jobs, random_state=seed)
    if kind == "xgb":
        import xgboost as xgb
        cls = xgb.XGBClassifier if task == "clf" else xgb.XGBRegressor
        return cls(n_estimators=2000, learning_rate=hp.get("lr", 0.03), max_depth=hp.get("depth", 3), subsample=0.8,
                   colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50, n_jobs=n_jobs, random_state=seed)
    if kind == "lgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
        return cls(n_estimators=2000, learning_rate=hp.get("lr", 0.03), num_leaves=hp.get("leaves", 15), subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, n_jobs=n_jobs, random_state=seed, verbose=-1)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        cls = HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor
        return cls(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=seed)
    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        cls = TabPFNClassifier if task == "clf" else TabPFNRegressor
        return cls(device="cuda", n_estimators=hp.get("n_estimators", 4), random_state=seed)   # 4 = v2 默认（原模型用的）
    raise ValueError(kind)


def fit_learner(kind, hp, task, F, y, Fe, ye, seed=0):
    if task == "clf" and len(np.unique(y)) < 2:
        return ("const", float(y.mean()))
    if task == "reg" and len(y) == 0:                         # 从没测过的量（只在 NACC 视图上训练时出现）：模拟时永远缺失，值无所谓
        return ("const", 0.0)
    m = make_learner(kind, hp, task, seed)
    if kind == "xgb":
        m.fit(F, y, eval_set=[(Fe, ye)], verbose=False)
    elif kind == "lgbm":
        import lightgbm as lgb
        m.fit(F, y, eval_set=[(Fe, ye)], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif kind == "tabpfn":
        ctx = hp.get("ctx", 10_000)                            # v2 的上限（原模型用的）；随访指标用 5000 省时间
        if len(F) > ctx:
            k = np.random.default_rng(seed).choice(len(F), ctx, replace=False)
            F, y = F[k], y[k]
        m.fit(F, y)
    else:
        m.fit(F, y)
    return m


def pred(m, F, task, chunk=50_000):
    if isinstance(m, tuple):
        return np.full(len(F), m[1])
    if type(m).__name__.startswith("TabPFN"):
        chunk = 5000                                          # TabPFN 测试行和上下文一起过网络：分小块，免得显存不够
    f = (lambda G: m.predict_proba(G)[:, 1]) if task == "clf" else m.predict
    return np.concatenate([f(F[s: s + chunk]) for s in range(0, len(F), chunk)]) if len(F) else np.zeros(0)


# ================================================================ 风险（死亡、AD 在下一年里）


class ClfHazard:
    """分类器家族：同一种分类器在 1 年标签上（原模型 60 个分类器里的 (死亡, 1 年)、(AD, 1 年) 那两个）。"""

    def __init__(self, kind, hp, seed=0):
        self.kind, self.hp, self.seed = kind, hp, seed

    def fit(self, lm, lme, prep):
        F, Fe = prep(lm["F"]), prep(lme["F"])
        self.m = {}
        for key, qi in (("death", B.I_DEATH), ("ad", B.I_ADQ)):
            y, ye = lm["Y"][:, qi, 0], lme["Y"][:, qi, 0]
            k, ke = y >= 0, ye >= 0
            self.m[key] = fit_learner(self.kind, self.hp, "clf", F[k], y[k], Fe[ke], ye[ke], self.seed)
        return self

    def __call__(self, F):
        return pred(self.m["death"], F, "clf"), pred(self.m["ad"], F, "clf")


class CoxOneYear:
    """1 − S(1 年 | G) = 1 − cox_surv(H0, shift, G·β)。原来是闭包 lambda；写成类才能整个模型存盘（外部验证，2026-11-01），算的是同一个数。"""

    def __init__(self, H0, shift, beta):
        self.H0, self.shift, self.beta = H0, shift, beta

    def __call__(self, G):
        return 1 - TB.cox_surv(self.H0, self.shift, G @ self.beta)[:, 0]


class SurvHazard:
    """生存家族：原模型自己（同一组超参）在死亡、AD 两道题上的 1 − S(1 年)。"""

    def __init__(self, kind, hp_by_q, device="cpu"):
        self.kind, self.hp, self.device = kind, hp_by_q, device

    def _fit_one(self, q, lm, lme, F, Fe):
        qi = B.Q.index(q)
        hp = self.hp.get(q, {}) if isinstance(self.hp, dict) else {}
        T, E, ok = B.tte(lm["Y"][:, qi])
        Tt, Et, okt = B.tte(lme["Y"][:, qi])
        if self.kind == "cox" or (self.kind == "fg" and q == "death"):
            beta = TB.cox_ridge(F[ok], T[ok], E[ok], hp.get("pen", 0.01))
            H0, shift = TB.breslow(T[ok], E[ok], F[ok] @ beta)
            return CoxOneYear(H0, shift, beta)
        if self.kind == "fg":                        # AD：Fine–Gray 的 1 年累计发生（死亡是竞争事件）
            Tc, Cc, okc = TB.cr_targets(lm["Y"], q)
            beta, H0, shift = TB.fg_ridge(F[okc], Tc[okc], Cc[okc], hp.get("pen", 0.01))
            return CoxOneYear(H0, shift, beta)
        if self.kind == "lgbmcox":
            surv_of = TB.surv_lgbmcox(F[ok], T[ok], E[ok], (Fe[okt], Tt[okt], Et[okt]), None,
                                      hp or {"lr": 0.03, "leaves": 15}, return_model=True)
            return lambda G: 1 - surv_of(G)[:, 0]
        if self.kind == "xgbaft":
            surv_of = TB.surv_xgbaft(F[ok], T[ok], E[ok], (Fe[okt], Tt[okt], Et[okt]), None,
                                     hp or {"dist": "normal", "scale": 1.0}, return_model=True)
            return lambda G: 1 - surv_of(G)[:, 0]
        if self.kind == "rsf":
            from sksurv.ensemble import RandomSurvivalForest
            from sksurv.util import Surv
            m = RandomSurvivalForest(n_estimators=hp.get("trees", 1000), min_samples_leaf=hp.get("leaf", 16),
                                     max_features="sqrt", n_jobs=16, random_state=0, low_memory=False)
            m.fit(F[ok], Surv.from_arrays(E[ok].astype(bool), T[ok].astype(float)))
            j = int(np.searchsorted(m.unique_times_, 1.0, side="right") - 1)

            def f(G, m=m, j=j):
                if j < 0:
                    return np.zeros(len(G))
                return np.concatenate([1 - m.predict_survival_function(G[s: s + 20000], return_array=True)[:, j]
                                       for s in range(0, len(G), 20000)]) if len(G) else np.zeros(0)
            return f
        if self.kind == "xgbcox":
            import xgboost as xgb
            yv = np.where(E[ok] == 1, T[ok], -T[ok]).astype(float)
            dtr = xgb.DMatrix(F[ok], label=yv)
            dte = xgb.DMatrix(Fe[okt], label=np.where(Et[okt] == 1, Tt[okt], -Tt[okt]).astype(float))
            p = {"objective": "survival:cox", "eta": hp.get("lr", 0.03), "max_depth": hp.get("depth", 3), "subsample": 0.8,
                 "colsample_bytree": 0.8, "eval_metric": "cox-nloglik", "nthread": 16, "seed": 0}
            b = xgb.train(p, dtr, 3000, evals=[(dte, "test")], early_stopping_rounds=50, verbose_eval=False)
            rng = (0, b.best_iteration + 1)
            H0, shift = TB.breslow(T[ok], E[ok], np.log(b.predict(dtr, iteration_range=rng)))
            return lambda G: 1 - TB.cox_surv(H0, shift, np.log(b.predict(xgb.DMatrix(G), iteration_range=rng)))[:, 0]
        if self.kind == "survivalboost":
            Tc, Cc, okc = TB.cr_targets(lm["Y"], q)
            import pandas as pd
            from hazardous import SurvivalBoost
            m = SurvivalBoost(n_iter=100, learning_rate=0.05, show_progressbar=False, random_state=0)
            m.fit(F[okc], pd.DataFrame({"event": Cc[okc].astype(int), "duration": Tc[okc].astype(float)}))

            def f(G, m=m):
                if not len(G):
                    return np.zeros(0)
                c = np.asarray(m.predict_cumulative_incidence(G, times=np.array([1.0])))
                return c[:, 1, 0] if c.ndim == 3 else c[:, 0]
            return f
        # 深度生存：pycox 的损失，网络两层 32（和原模型同一个函数）
        surv_of, _ = TB.surv_deep(self.kind, F[ok], T[ok], E[ok], (Fe[okt], Tt[okt], Et[okt]), None, hp, device=self.device)
        return lambda G: 1 - surv_of(G)[:, 0]

    def fit(self, lm, lme, prep):
        F, Fe = prep(lm["F"]), prep(lme["F"])
        self.f = {q: self._fit_one(q, lm, lme, F, Fe) for q in ("death", "ad")}
        return self

    def __call__(self, F):
        return self.f["death"](F), self.f["ad"](F)


class ScoreHazard:
    """临床评分：原文的分数 -> 1 年概率（train 上的单变量逻辑回归，斜率 ≥ 0；排序就是原评分）。"""

    def __init__(self, kind, norm):
        self.kind, self.norm = kind, norm

    def fit_score(self, Dtr):
        from sklearn.linear_model import LogisticRegression
        sc = TB.risk_score(self.kind, Dtr, self.norm)
        lm = B.landmarks(Dtr, "cur")
        self.mu, self.sd = sc.mean(), sc.std() + 1e-8
        self.ab = {}
        for key, qi in (("death", B.I_DEATH), ("ad", B.I_ADQ)):
            y = lm["Y"][:, qi, 0]
            k = y >= 0
            m = LogisticRegression(C=1e6, max_iter=1000).fit(((sc - self.mu) / self.sd)[k, None], y[k])
            self.ab[key] = (float(m.intercept_[0]), max(float(m.coef_[0, 0]), 1e-6))
        return self

    def from_visits(self, x, S, L):
        n = len(x)
        D = {"X": x[:, None], "M": np.ones((n, 1), bool), "S": S, "L": L[:, None]}
        z = (TB.risk_score(self.kind, D, self.norm) - self.mu) / self.sd
        return tuple(SIG(self.ab[k][0] + self.ab[k][1] * z) for k in ("death", "ad"))


# ================================================================ 随访指标：同族的学习器，每个指标一个；或一个 MLP 一次给全部


class MLPNext:
    """pycox 默认结构的 MLP（两层 32，BatchNorm + ReLU + dropout 0.1），一次输出下一次访视的全部分布参数。"""

    def __init__(self, device="cpu", width=32, lr=1e-2, seed=0):
        self.device, self.width, self.lr, self.seed = device, width, lr, seed

    def fit(self, F, nx, cur, Fe, nxe, cure):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = TB._mlp(F.shape[1], N_NEXT, width=self.width).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)   # noqa: E731
        Ft, Nt, Ct, Fet, Net, Cet = (t(a) for a in (F, nx, cur, Fe, nxe, cure))
        ones = lambda n: torch.ones(n, dtype=torch.bool, device=self.device)   # noqa: E731
        best, state, bad = np.inf, None, 0
        for ep in range(200):
            self.net.train()
            perm = rng.permutation(len(Ft))
            for s in range(0, len(perm), 256):
                b = torch.as_tensor(perm[s: s + 256], device=self.device)
                if len(b) < 2:
                    continue
                l = next_nll(split_next(self.net(Ft[b])), Nt[b], Ct[b], ones(len(b)))
                opt.zero_grad()
                l.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(next_nll(split_next(self.net(Fet)), Net, Cet, ones(len(Fet))))
            if vl < best - 1e-4:
                best, bad, state = vl, 0, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        self.info = {"next_test_nll": best, "epochs": ep + 1}
        return self

    def __call__(self, F):
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(F), 50000):
                out.append(self.net(torch.as_tensor(F[s: s + 50000], dtype=torch.float32, device=self.device)).cpu().numpy())
        return nd_from_logits(np.concatenate(out))


class PerTargetNext:
    """每个指标一个同族学习器（g-formula 的写法）：连续量回归 + 高斯残差，缺失 / 状态 / 新出现各一个分类器。"""

    def __init__(self, kind, hp, seed=0):
        self.kind, self.hp, self.seed = kind, hp, seed

    def fit(self, F, nx, cur, Fe, nxe, cure):
        k, hp, sd = self.kind, self.hp, self.seed
        self.m, self.sd = {}, np.zeros(N_CONT)
        for j in range(N_CONT):
            self.m[f"miss_c{j}"] = fit_learner(k, hp, "clf", F, nx[:, I_MC + j], Fe, nxe[:, I_MC + j], sd)
            o, oe = nx[:, I_MC + j] < 0.5, nxe[:, I_MC + j] < 0.5
            r = fit_learner(k, hp, "reg", F[o], nx[o, j], Fe[oe], nxe[oe, j], sd)
            self.m[f"mu{j}"] = r
            # 噪声 = 训练段上的残差标准差。2026-10-29 试过改用 test 段残差（树 / 提升模型在训练段上的残差偏小）：
            # 5 年转移矩阵反而更不像（ar-lgbm 5.9 -> 7.2、rsf 6.7 -> 7.9）—— 噪声每年独立地抽，加大只会多出随机的换分期，没有采用
            sub = np.where(o)[0][:20000]
            self.sd[j] = float(np.std(nx[sub, j] - pred(r, F[sub], "reg"))) if len(sub) else 0.0
        for j in range(N_STATE):
            self.m[f"miss_s{j}"] = fit_learner(k, hp, "clf", F, nx[:, I_MS + j], Fe, nxe[:, I_MS + j], sd)
            o, oe = nx[:, I_MS + j] < 0.5, nxe[:, I_MS + j] < 0.5
            self.m[f"xs{j}"] = fit_learner(k, hp, "clf", F[o], nx[o, I_XS + j], Fe[oe], nxe[oe, I_XS + j], sd)
        for j in range(N_EV):
            o, oe = cur[:, I_EVER + j] < 0.5, cure[:, I_EVER + j] < 0.5
            self.m[f"ever{j}"] = fit_learner(k, hp, "clf", F[o], nx[o, I_EVER + j], Fe[oe], nxe[oe, I_EVER + j], sd)
        self.info = {"resid_sd": self.sd.tolist()}
        return self

    def __call__(self, F):
        P = lambda key, task: pred(self.m[key], F, task)     # noqa: E731
        return {"mu": np.stack([P(f"mu{j}", "reg") for j in range(N_CONT)], 1),
                "sd": np.broadcast_to(self.sd, (len(F), N_CONT)),
                "miss_c": np.stack([P(f"miss_c{j}", "clf") for j in range(N_CONT)], 1),
                "xs": np.stack([P(f"xs{j}", "clf") for j in range(N_STATE)], 1),
                "miss_s": np.stack([P(f"miss_s{j}", "clf") for j in range(N_STATE)], 1),
                "ever": np.stack([P(f"ever{j}", "clf") for j in range(N_EV)], 1)}


# ================================================================ 表格类的 rollout 版：风险 + 随访指标，逐年滚动


class ARVersion:
    def __init__(self, kw, norm, device="cpu", seed=0):
        self.kw, self.norm, self.device, self.seed = kw, norm, device, seed
        self.featset = kw["feat"]

    def fit(self, Dtr, Dte):
        t0 = time.time()
        kw, fs = self.kw, self.featset
        hp = chosen_hp(kw["orig"])
        lm, nx, cur, ok = pairs(Dtr, fs)
        lme, nxe, cure, oke = pairs(Dte, fs)
        self.prep = Prep(lm["F"])
        F, Fe = self.prep(lm["F"]), self.prep(lme["F"])
        haz, hk = kw["haz"].split(":") if ":" in kw["haz"] else (kw["haz"], kw["cov"])
        if haz == "clf":
            self.hazard = ClfHazard(hk, hp, self.seed).fit(lm, lme, self.prep)
        elif haz == "surv":
            self.hazard = SurvHazard(hk, hp, self.device).fit(lm, lme, self.prep)
        else:
            self.hazard = ScoreHazard(hk, self.norm).fit_score(Dtr)
        cov_hp = {**(hp if haz == "clf" else {}), **(kw.get("cov_hp") or {})}
        if kw["cov"] == "mlp":
            self.cov = MLPNext(self.device, seed=self.seed).fit(F[ok], nx[ok], cur[ok], Fe[oke], nxe[oke], cure[oke])
        else:
            self.cov = PerTargetNext(kw["cov"], cov_hp, self.seed).fit(F[ok], nx[ok], cur[ok], Fe[oke], nxe[oke], cure[oke])
        self.info = {"minutes": (time.time() - t0) / 60, "hp_from": kw["orig"], "hp": hp, "cov": kw["cov"],
                     "hazard": kw["haz"], "featset": fs, "n_pairs": int(ok.sum()), **getattr(self.cov, "info", {})}
        return self

    def step(self, x, S, L):
        F = self.prep(feats(self.featset, x, S, L))
        if isinstance(self.hazard, ScoreHazard):
            p_d, p_a = self.hazard.from_visits(x, S, L)
        else:
            p_d, p_a = self.hazard(F)
        return np.clip(p_d, 0, 1), np.clip(p_a, 0, 1), self.cov(F)

    def sim(self, start, n_mc, rng, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        l0 = SIM.l2c_replay(D["X"][ii], D["M"][ii], vv)
        y0 = np.round(D["X"][ii, vv, I_TF + 1] * 10.0)
        return SIM.simulate_visits(ARVStepper(self, l0, y0), D["X"][ii, vv], D["S"][ii], y0, n_mc, rng, norm)


class ARVStepper:
    def __init__(self, model, l2c_state, y0, chunk=100_000):
        self.model, self.l0, self.y0, self.chunk = model, l2c_state, y0, chunk

    def init(self, n_mc):
        rep = np.repeat(np.arange(len(self.y0)), n_mc)
        return {"l2c": self.l0.take(rep), "year": np.repeat(self.y0, n_mc).astype(float)}

    def step(self, st, x, S, rows):
        L = st["l2c"].take(rows).features(st["year"][rows])
        outs = [self.model.step(x[rows[s: s + self.chunk]], S[rows[s: s + self.chunk]], L[s: s + self.chunk])
                for s in range(0, len(rows), self.chunk)]
        nd = {k: np.concatenate([o[2][k] for o in outs]) for k in outs[0][2]}
        return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]), nd

    def update(self, st, rows, xn, year):
        st["l2c"].update(rows, xn[:, :N_CONT].astype(float), xn[:, I_MC:I_XS] < 0.5, year)
        st["year"][rows] = year


# ================================================================ 序列网络：原结构 + 下一次访视的头


def build_seqsim(kind, x_mean, n_risk=B.NQ * B.NH):
    import torch.nn as nn
    import sequence as SQ
    hp = SQ.SEQ_HP[kind]

    class SeqSim(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = SQ.make_encoder(kind, hp["d"], hp["dropout"], x_mean)
            self.risk = nn.Linear(hp["d"], n_risk)
            self.next = nn.Linear(hp["d"], N_NEXT)

        def forward(self, b):
            h = self.enc(b)
            return {"risk": self.risk(h), **split_next(self.next(h))}

    return SeqSim(), hp


def seqsim_loss(b, out, cols=None):
    """风险头（cols = 只训练标签的这几列，None = 60 个全部）+ 下一次访视的似然。"""
    import torch.nn.functional as F
    y = b["Y"] if cols is None else b["Y"][..., cols]
    ok = (y >= 0) & b["M"][..., None]
    l = F.binary_cross_entropy_with_logits(out["risk"][ok], y[ok])
    X, M = b["X"], b["M"]
    gap1 = (abs(X[:, 1:, I_TF + 2] - 1.0) < 1e-6) & M[:, 1:]
    o = {k: out[k][:, :-1] for k in NEXT_KEYS}
    return l + next_nll(o, X[:, 1:], X[:, :-1], gap1)


class SeqSimModel:
    chunk = 20_000                                # 一次过网络的轨迹数
    cols = None                                   # 风险头训练标签的哪几列（None = 6 道题 × 10 年全部）
    idx = (IDX_D1, IDX_A1)                        # 模拟时读风险头的哪两列：1 年内死亡、1 年内 AD

    def __init__(self, kind, device):
        self.kind, self.device = kind, device

    def fit(self, Dtr, Dte, seed=0, max_epochs=150, patience=15):
        import torch
        import sequence as SQ
        t0 = time.time()
        torch.manual_seed(seed)
        tr, te = SQ.tensors(Dtr, self.device), SQ.tensors(Dte, self.device)
        val, m = SQ.obs_values(tr["X"])
        x_mean = ((val * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)).cpu().numpy()
        self.x_mean = x_mean                                       # 存模型时要（save_state / load_state 重建网络用）
        self.net, hp = build_seqsim(self.kind, x_mean, B.NQ * B.NH if self.cols is None else len(self.cols))
        self.net = self.net.to(self.device)
        best, n_ep = SQ.fit(self.net, tr, te, lambda b, o: seqsim_loss(b, o, self.cols), hp["lr"], hp["wd"], 64,
                            max_epochs, patience, seed)
        self.net.eval()
        self.info = {"best_test_loss": best, "epochs": n_ep, "minutes": (time.time() - t0) / 60, "hp": hp}
        return self

    def outputs(self, Xb, Mb, Sb, Lb, pos):
        """一批轨迹（缓冲区）在各自当前位置 pos 上的 (p_death_1y, p_ad_1y, 下一次访视分布)。"""
        import torch
        import sequence as SQ
        V = int(pos.max()) + 1
        X, M = Xb[:, :V], Mb[:, :V]
        L = Lb[:, :V] if Lb is not None else np.zeros((len(X), V, 1), np.float32)   # 只有 L2C-FNN 读手工历史特征
        d = {"X": X, "M": M, "S": Sb, "L": L, "Y": np.zeros((len(X), V, 1), np.float32)}
        with torch.no_grad():
            T = SQ.tensors(d, self.device)
            o = self.net(T)
            ar = torch.arange(len(X), device=self.device)
            pt = torch.as_tensor(pos, device=self.device)
            r = torch.sigmoid(o["risk"][ar, pt])
            nd = {k: o[k][ar, pt].cpu().numpy() for k in NEXT_KEYS}
        r = r.cpu().numpy()
        return r[:, self.idx[0]], r[:, self.idx[1]], nd_from_logits(nd)

    def sim(self, start, n_mc, rng, norm, full_draws=False):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        y0 = np.round(D["X"][ii, vv, I_TF + 1] * 10.0)
        return SIM.simulate_visits(SeqStepper(self, start, n_mc, self.chunk), D["X"][ii, vv], D["S"][ii], y0, n_mc, rng, norm,
                                   full_draws=full_draws)


def save_state(model, path):
    """SeqSim / SeqAR 模型 -> 权重 + 重建网络要的设置（网络类定义在 build_seqsim 里面，整个对象 pickle 不了）。"""
    import torch
    torch.save({"cls": type(model).__name__, "kind": model.kind, "x_mean": model.x_mean, "info": model.info,
                "state_dict": model.net.state_dict()}, path)


def load_state(path, device="cpu"):
    import torch
    d = torch.load(path, map_location=device, weights_only=False)
    m = {"SeqSimModel": SeqSimModel, "SeqARModel": SeqARModel}[d["cls"]](d["kind"], device)
    m.x_mean, m.info = d["x_mean"], d["info"]
    m.net, _ = build_seqsim(m.kind, m.x_mean, B.NQ * B.NH if m.cols is None else len(m.cols))
    m.net.load_state_dict(d["state_dict"])
    m.net = m.net.to(device)
    m.net.eval()
    return m


class SeqStepper:
    """每条轨迹一个访视缓冲区：历史（到起点）+ 模拟出来的访视；每一步把整条序列重新过一遍网络（因果结构，只读当前位置）。"""

    def __init__(self, model, start, n_mc, chunk=20_000):
        self.model, self.start, self.n_mc, self.chunk = model, start, n_mc, chunk

    def init(self, n_mc):
        D, ii, vv = self.start["D"], self.start["ii"], self.start["vv"]
        rep = np.repeat(np.arange(len(ii)), n_mc)
        Vt = int(vv.max()) + 1 + SIM.YEARS
        n = len(rep)
        st = {"X": np.zeros((n, Vt, B.N_FEAT), np.float32), "M": np.zeros((n, Vt), bool), "L": None,
              "S": D["S"][ii][rep].astype(np.float32), "pos": vv[rep].astype(int).copy()}
        V0 = int(vv.max()) + 1
        keep = np.arange(V0)[None, :] <= vv[:, None]
        st["X"][:, :V0] = (D["X"][ii, :V0] * keep[..., None])[rep]
        st["M"][:, :V0] = (D["M"][ii, :V0] & keep)[rep]
        if self.model.kind == "fnn":                              # 只有 L2C-FNN 读手工历史特征：逐步更新
            st["L"] = np.zeros((n, Vt, D["L"].shape[-1]), np.float32)
            st["L"][:, :V0] = (D["L"][ii, :V0] * keep[..., None])[rep]
            st["l2c"] = SIM.l2c_replay(D["X"][ii], D["M"][ii], vv).take(rep)
            st["year"] = np.round(D["X"][ii, vv, I_TF + 1] * 10.0)[rep]
        return st

    def step(self, st, x, S, rows):
        outs = []
        for s in range(0, len(rows), self.chunk):
            r = rows[s: s + self.chunk]
            outs.append(self.model.outputs(st["X"][r], st["M"][r], st["S"][r],
                                           None if st["L"] is None else st["L"][r], st["pos"][r]))
        nd = {k: np.concatenate([o[2][k] for o in outs]) for k in outs[0][2]}
        return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs]), nd

    def update(self, st, rows, xn, year):
        st["pos"][rows] += 1
        p = st["pos"][rows]
        st["X"][rows, p] = xn
        st["M"][rows, p] = True
        if "l2c" in st:
            st["l2c"].update(rows, xn[:, :N_CONT].astype(float), xn[:, I_MC:I_XS] < 0.5, year)
            st["year"][rows] = year
            st["L"][rows, p] = st["l2c"].take(rows).features(year)


# ================================================================ 自回归的序列网络（2026-10-28）：只学下一步


class SeqARModel(SeqSimModel):
    """GRU / LSTM / Transformer / GRU-D / T-LSTM / RETAIN：同一个编码器（sequence.make_encoder，同一组超参），
    风险头只有"下一年内死亡 / AD"两个输出，和下一次访视的分布一起训练（next-step 似然）；没有"H 年内会不会"的题目头。"""
    cols = [IDX_D1, IDX_A1]
    idx = (0, 1)


def build_ddhar():
    """Dynamic-DeepHit 的结构（官方默认：2 层 LSTM、以当前隐状态为查询对过去隐状态的注意力、每个原因一个 2 层子网络），
    原因 = 死亡、AD，每个原因输出下一年内发生的 logit；另加下一次访视分布的头（读 LSTM 隐状态，和原来的逐步预测 L3 同一个位置）。"""
    import torch
    import torch.nn as nn
    import sequence as SQ
    hp = SQ.SEQ_HP["ddh"]
    K = 2

    class DDHAR(nn.Module):
        def __init__(self, d, dropout):
            super().__init__()
            self.inp = nn.Linear(B.N_FEAT + B.N_S, d)
            self.rnn = nn.LSTM(d, d, num_layers=2, batch_first=True, dropout=dropout)
            self.att = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
            self.cs = nn.ModuleList([nn.Sequential(nn.Linear(d + B.N_FEAT, d), nn.ReLU(), nn.Dropout(dropout),
                                                   nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout)) for _ in range(K)])
            self.out = nn.Linear(K * d, K)
            self.next = nn.Linear(d, N_NEXT)
            self.drop = nn.Dropout(dropout)

        def forward(self, b):
            n, V, _ = b["X"].shape
            x = torch.relu(self.inp(torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1)], -1)))
            h, _ = self.rnn(self.drop(x))
            # 第 t 个站点：对 j <= t 的隐状态做注意力（以 h_t 为查询）
            q = h[:, :, None].expand(-1, -1, V, -1)
            k = h[:, None].expand(-1, V, -1, -1)
            e = self.att(torch.cat([q, k], -1)).squeeze(-1)
            causal = torch.triu(torch.ones(V, V, dtype=torch.bool, device=h.device), 1)
            e = e.masked_fill(causal[None] | ~b["M"][:, None, :], -1e9)
            ctx = torch.einsum("ntj,njd->ntd", torch.softmax(e, -1), h)
            z = torch.cat([self.cs[c](torch.cat([ctx, b["X"]], -1)) for c in range(K)], -1)
            return {"risk": self.out(self.drop(z)), **split_next(self.next(self.drop(h)))}

    return DDHAR(hp["d"], hp["dropout"]), hp


def ddhar_loss(b, out, alpha, sigma=0.1):
    """DDH 的损失落到"下一年"这一格：两个原因的对数似然（= BCE）+ 下一次访视的似然（换掉原来只算连续量的 L3）
    + alpha × 每个原因的排序损失（同一批里下一年发生的 i、没发生的 j：要 F(i) > F(j)，exp(−(F_i − F_j)/σ)）。"""
    import torch
    cols = [IDX_D1, IDX_A1]
    l = seqsim_loss(b, out, cols)
    y = b["Y"][..., cols]
    p = torch.sigmoid(out["risk"])
    ok = (y >= 0) & b["M"][..., None]
    for c in range(len(cols)):
        pc, yc = p[..., c][ok[..., c]], y[..., c][ok[..., c]]
        sub = torch.randperm(len(pc), device=pc.device)[:512]
        pc, yc = pc[sub], yc[sub]
        A = ((yc[:, None] > 0.5) & (yc[None, :] < 0.5)).float()
        l = l + alpha * (A * torch.exp(-(pc[:, None] - pc[None, :]) / sigma)).sum() / A.sum().clamp(min=1)
    return l


class DDHARModel(SeqSimModel):
    """Dynamic-DeepHit 的自回归版：一个模型同时给死亡、AD 下一年的风险和下一次访视的分布；超参是官方默认（SEQ_HP["ddh"]）。"""
    chunk = 2_000                                 # 注意力是 (轨迹, 访视, 访视, 2d)：分小块
    idx = (0, 1)

    def __init__(self, device):
        super().__init__("ddh", device)

    def fit(self, Dtr, Dte, seed=0, max_epochs=150, patience=15):
        import torch
        import sequence as SQ
        t0 = time.time()
        torch.manual_seed(seed)
        tr, te = SQ.tensors(Dtr, self.device), SQ.tensors(Dte, self.device)
        self.net, hp = build_ddhar()
        self.net = self.net.to(self.device)
        best, n_ep = SQ.fit(self.net, tr, te, lambda b, o: ddhar_loss(b, o, hp["alpha"]), hp["lr"], hp["wd"], hp["bs"],
                            max_epochs, patience, seed)
        self.net.eval()
        self.info = {"best_test_loss": best, "epochs": n_ep, "minutes": (time.time() - t0) / 60, "hp": hp}
        return self
