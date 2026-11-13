"""joint.py -- 轨迹 + 事件的联合模型（2026-10-22 加），都能生成未来：
  jm             共享随机效应的联合模型，**联合估计**（Rizopoulos 2012；R 包 JM / JMbayes2 的模型结构）
  gpboost-joint  轨迹 = GPBoost（树提升的固定效应 + 每人随机截距、斜率；Sigrist, JMLR 2022），风险两阶段
  merf-joint     轨迹 = MERF（混合效应随机森林；Hajjem, Bellavance & Larocque 2014 的 EM），风险两阶段
  gp-joint       轨迹 = 分层高斯过程（总体均值 + 每人一条 GP，核 = 随机截距 + 随机斜率 + 平方指数；边际似然定超参），风险两阶段
和 lmm-joint（generators.LMMJoint，两阶段的线性混合模型）同一个框架，差别只在"真实轨迹"怎么来、风险是不是一起估计：
  两条真实轨迹 m(t)：MMSE、全局认知（原始分，t = 距基线的年数）。
  风险：死亡、AD 每一年的离散时间风险，logit h = γ·w + α·[m_MMSE(t), m'_MMSE(t), m_认知(t), m'_认知(t)]（当前真值 + 当前斜率，
        Rizopoulos 的 value + slope 关联），w = 截距、年龄、性别、教育、APOE ε4 一个 / 两个。
  从站点起：轨迹的随机部分按到站点为止的观测求后验、抽 n_mc 组，然后逐年走：先抽这一年的死亡 / AD，活着就在年底"测一次"
            （真值 + 测量误差 -> 分期）。和 lmm-joint 同一种读出。

jm 的似然（每人）：p(y, T) = p(y) · ∫ p(T | b) p(b | y) db。p(y) 是双变量线性混合模型的边际（随机效应 b = 两个标记各一对截距、
斜率，4 维、协方差 G 不受限，所以两条轨迹的个体差异可以相关）；p(b | y) 是它的高斯后验；积分用固定的 K 个标准正态抽样做重参数化
蒙特卡洛（common random numbers），所有参数（β、σ、G、风险的 γ、α）一起用 L-BFGS 最大化。事件部分：
  死亡 = 从基线到死亡（或删失）的每一个人年（死亡年份由最后一次访视的标签推出：死在之后第 k 年内）；
  AD   = 每次没有 AD 的访视到下一次访视之间（间隔 g 年 = g 个人年，区间删失：下一次才知道有没有）。
站点上的后验另外用"到站点为止还活着、AD 的历史"给抽样加权（重要性重抽样），所以既用了标记的历史、也用了事件的历史。
"""
import time

import numpy as np
import torch

import bl_data as B
import simulator as SIM
from bl_data import I_AD, I_MC, I_TF
from generators import _stage_measured

FAMS = ("mmse", "cogn")
J = {"mmse": 0, "cogn": 1}
LOG2PI = float(np.log(2 * np.pi))


def design(t, age0, S):
    """线性混合模型的固定效应（和 lmm-joint 一样）：截距、时间、基线年龄、性别、教育、ε4 个数，以及时间 × 基线年龄、时间 × ε4。"""
    e4 = S[..., 9] + 2 * S[..., 10]
    return np.stack([np.ones_like(t), t, age0, S[..., 0], S[..., 1], e4, t * age0, t * e4], -1)


def hf(vals, age_z, S):
    """风险的输入：MMSE 真值、斜率，认知真值、斜率，年龄 z，性别，教育，ε4 一个，两个（和 lmm-joint 的 _hf 同一顺序）。"""
    return np.stack([vals["mmse"][0], vals["mmse"][1], vals["cogn"][0], vals["cogn"][1], age_z,
                     S[..., 0], S[..., 1], S[..., 9], S[..., 10]], -1)


def raw_value(D, fam, norm, ii, vv):
    j = J[fam]
    x = D["X"][ii, vv]
    return x[:, j] * norm["sd"][j] + norm["mu"][j], x[:, I_MC + j] < 0.5


def years(D, ii, vv):
    return D["X"][ii, vv, I_TF + 1] * 10.0


def simulate(model, start, n_mc, rng, norm, n_years=SIM.YEARS):
    """共用的逐年模拟。model.latent(start, n_mc, rng, norm) -> fam: (真值, 斜率)，形状 (n·n_mc, n_years + 1)，第 k 列 = 站点后第 k 年；
    model.hazard_probs(vals, 年龄 z, 静态) -> (p 死亡, p AD)；model.noise_sd[fam] = 测量误差的标准差。"""
    D, ii, vv = start["D"], start["ii"], start["vv"]
    n = len(ii)
    N = n * n_mc
    rep = np.repeat(np.arange(n), n_mc)
    lat = model.latent(start, n_mc, rng, norm)
    agez0 = D["X"][ii, vv, I_TF][rep]
    S = D["S"][ii][rep]
    stage = {f: np.full((N, n_years + 1), -1, np.int64) for f in FAMS}
    for f in FAMS:
        stage[f][:, 0] = np.repeat(_stage_measured(D, ii, vv, norm, f), n_mc)
    t_death = np.full(N, np.inf)
    t_ad = np.where(np.repeat(D["X"][ii, vv, I_AD] > 0.5, n_mc), -np.inf, np.inf)
    for k in range(1, n_years + 1):
        r = np.where(~np.isfinite(t_death))[0]
        if not len(r):
            break
        vals = {f: (lat[f][0][r, k - 1], lat[f][1][r, k - 1]) for f in FAMS}
        p_d, p_a = model.hazard_probs(vals, agez0[r] + (k - 1) * SIM.AGE_STEP_Z, S[r])
        dies = rng.random(len(r)) < p_d
        td = np.where(dies, k - 1 + rng.random(len(r)), np.inf)
        gets = (t_ad[r] == np.inf) & (rng.random(len(r)) < p_a)
        ta = np.where(gets, k - 1 + rng.random(len(r)), np.inf)
        gets &= ta < td
        t_ad[r[gets]] = ta[gets]
        t_death[r] = td
        al = r[~dies]
        if not len(al):
            continue
        for f in FAMS:
            y = lat[f][0][al, k] + model.noise_sd[f] * rng.standard_normal(len(al))
            if f == "mmse":
                y = np.clip(np.round(y), 0, 30)
            stage[f][al, k] = SIM.stage_of(y, f)
    rs = lambda a: a.reshape(n, n_mc, *a.shape[1:])            # noqa: E731
    nvis = np.minimum(np.where(np.isfinite(t_death), np.floor(t_death), n_years), n_years).astype(int)
    return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
            "n_visits": rs(nvis)}


def landmark_obs(D, ii, vv, fam, norm, f_obs=None):
    """每个站点（第 ii 个人到第 vv 次访视为止）的观测，排成 (站点, 访视) 的格子：时间、值（或减去固定效应的残差）、是否观测到。"""
    V = int(vv.max()) + 1
    T = np.zeros((len(ii), V))
    Y = np.zeros((len(ii), V))
    O = np.zeros((len(ii), V), bool)
    for v in range(V):
        y, obs = raw_value(D, fam, norm, ii, np.full(len(ii), v))
        ok = (v <= vv) & D["M"][ii, v] & obs
        T[:, v] = years(D, ii, np.full(len(ii), v))
        Y[:, v] = np.where(ok, y - (f_obs[:, v] if f_obs is not None else 0.0), 0.0)
        O[:, v] = ok
    return T, Y, O


def rint_post(T, R, O, G, s2):
    """随机截距 + 随机斜率的高斯后验（先验 N(0, G)、误差方差 s2）：b | 残差 ~ N(m, V)。T、R、O 是 landmark_obs 的格子。"""
    z0, z1 = O.astype(float), T * O
    ZtZ = np.stack([np.stack([z0.sum(1), z1.sum(1)], -1), np.stack([z1.sum(1), (z1 * T).sum(1)], -1)], -2) / s2
    Ztr = np.stack([(z0 * R).sum(1), (z1 * R).sum(1)], -1) / s2
    V = np.linalg.inv(np.linalg.inv(G)[None] + ZtZ)
    return np.einsum("nij,nj->ni", V, Ztr), V


def draw(m, V, n_mc, rng):
    """N(m, V) 各抽 n_mc 个。GP 后验在半年间隔上几乎共线：用对称特征分解（负的小特征值截成 0），不用 Cholesky。"""
    w, Q = np.linalg.eigh(0.5 * (V + np.swapaxes(V, -1, -2)))
    L = Q * np.sqrt(np.clip(w, 0, None))[:, None, :]
    rep = np.repeat(np.arange(len(m)), n_mc)
    return m[rep] + np.einsum("nij,nj->ni", L[rep], rng.standard_normal((len(rep), m.shape[1])))


class TwoStageHazard:
    """两阶段：每个训练站点上，用到站点为止的观测求轨迹的后验均值 -> 当前真值、斜率 -> 死亡、AD 的 1 年逻辑回归（lmm-joint 的做法）。"""

    def fit_hazard(self, Dtr, norm):
        from sklearn.linear_model import LogisticRegression
        lm = B.landmarks(Dtr, "cur")
        st = {"D": Dtr, "ii": lm["ii"], "vv": lm["vv"]}
        mean = self.latent_mean(st, norm)
        F = hf(mean, Dtr["X"][lm["ii"], lm["vv"], I_TF], Dtr["S"][lm["ii"]])
        self.hmu, self.hsd = F.mean(0), F.std(0) + 1e-8
        self.haz = {}
        for key, qi in (("death", B.I_DEATH), ("ad", B.I_ADQ)):
            y = lm["Y"][:, qi, 0]
            k = y >= 0
            self.haz[key] = LogisticRegression(C=1.0, max_iter=3000).fit((F[k] - self.hmu) / self.hsd, y[k])
        return {key: m.coef_[0].round(3).tolist() for key, m in self.haz.items()}

    def hazard_probs(self, vals, age_z, S):
        Fh = (hf(vals, age_z, S) - self.hmu) / self.hsd
        return self.haz["death"].predict_proba(Fh)[:, 1], self.haz["ad"].predict_proba(Fh)[:, 1]

    def sim(self, start, n_mc, rng, norm):
        return simulate(self, start, n_mc, rng, norm)


# ============================================================================== GPBoost / MERF 轨迹 + 两阶段风险


class MixedJoint(TwoStageHazard):
    """y = f(t, 基线) + b0 + b1 t + ε。f：GPBoost 的树提升 / MERF 的随机森林；(b0, b1) ~ N(0, G)。
    固定效应的输入：距基线的年数、基线年龄、25 维静态背景。网格（GPBoost 的轮数 / MERF 的叶子大小）按 test 段的
    "下一次测量的预测误差"挑：用到上一次为止的观测求随机效应后验，预测这一次的值。"""

    GRID = {"gpboost": [{"rounds": 150}, {"rounds": 500}], "merf": [{"leaf": 20}, {"leaf": 100}]}

    def __init__(self, kw, device="cpu"):
        self.kind = kw["kind"]

    @staticmethod
    def fe_inputs(t, age0, S):
        return np.concatenate([t[:, None], age0[:, None], S], 1).astype(np.float64)

    def long_table(self, D, fam, norm):
        ii, vv = np.where(D["M"])
        y, obs = raw_value(D, fam, norm, ii, vv)
        ii, vv, y = ii[obs], vv[obs], y[obs]
        t = years(D, ii, vv)
        return ii, t, y, self.fe_inputs(t, D["X"][ii, 0, I_TF], D["S"][ii])

    def fit_one(self, fam, hp, ii, t, y, Xf):
        if self.kind == "gpboost":
            import gpboost as gpb
            gm = gpb.GPModel(group_data=ii, group_rand_coef_data=t[:, None], ind_effect_group_rand_coef=[1],
                             likelihood="gaussian")
            params = {"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50, "verbose": 0, "num_threads": 16}
            bst = gpb.train(params=params, train_set=gpb.Dataset(Xf, y), gp_model=gm, num_boost_round=hp["rounds"])
            cp = np.asarray(gm.get_cov_pars(), float).ravel()             # 误差方差、随机截距方差、随机斜率方差
            s2, G = float(cp[0]), np.diag([cp[1], cp[2]])

            def fe(X, bst=bst):
                p = bst.predict(data=X, ignore_gp_model=True, pred_latent=True)
                return np.asarray(p["fixed_effect"] if isinstance(p, dict) else p, float)
            return fe, G, s2
        import pandas as pd
        from merf import MERF
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=hp["leaf"], max_features=0.5, n_jobs=16, random_state=0)
        m = MERF(fixed_effects_model=rf, max_iterations=10)
        m.fit(pd.DataFrame(Xf), np.stack([np.ones_like(t), t], 1), pd.Series(ii), y)
        return (lambda X, f=m.trained_fe_model: f.predict(X)), np.asarray(m.D_hat_history[-1], float), float(m.sigma2_hat_history[-1])

    def f_obs(self, fam, D, ii):
        """站点所属的人每次访视上的固定效应 f(t_v, 基线)：(站点, 访视)。"""
        V = D["X"].shape[1]
        t = D["X"][ii, :, I_TF + 1] * 10.0
        X = self.fe_inputs(t.ravel(), np.repeat(D["X"][ii, 0, I_TF], V), np.repeat(D["S"][ii], V, 0))
        return self.fe[fam](X).reshape(len(ii), V)

    def posterior(self, fam, start, norm, f_obs=None):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        f_obs = self.f_obs(fam, D, ii) if f_obs is None else f_obs
        T, R, O = landmark_obs(D, ii, vv, fam, norm, f_obs)
        p = self.lmm[fam]
        return rint_post(T, R, O, p["G"], p["s2"])

    def one_step_rmse(self, fam, Dte, norm):
        """test 段：每个（第 2 次以后的）测量，用之前的观测预测它；返回 RMSE。"""
        ii, vv = np.where(Dte["M"])
        y, obs = raw_value(Dte, fam, norm, ii, vv)
        keep = obs & (vv >= 1)
        ii, vv, y = ii[keep], vv[keep], y[keep]
        fo = self.f_obs(fam, Dte, ii)
        m, _ = self.posterior(fam, {"D": Dte, "ii": ii, "vv": vv - 1}, norm, fo)
        t = years(Dte, ii, vv)
        pred = fo[np.arange(len(ii)), vv] + m[:, 0] + m[:, 1] * t
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    def fit(self, Dtr, Dte, norm):
        t0 = time.time()
        self.fe, self.lmm, info = {}, {}, {"grid": []}
        for fam in FAMS:
            tab = self.long_table(Dtr, fam, norm)
            best = None
            for hp in self.GRID[self.kind]:
                fe, G, s2 = self.fit_one(fam, hp, *tab)
                self.fe[fam], self.lmm[fam] = fe, {"G": G, "s2": s2}
                rmse = self.one_step_rmse(fam, Dte, norm)
                info["grid"].append({"fam": fam, "hp": hp, "test_one_step_rmse": rmse, "G": G.round(4).tolist(), "s2": s2,
                                     "minutes": (time.time() - t0) / 60})
                print(f"  {self.kind} {fam} {hp}: test one-step RMSE {rmse:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if best is None or rmse < best[0]:
                    best = (rmse, hp, fe, G, s2)
            _, hp, fe, G, s2 = best
            self.fe[fam], self.lmm[fam] = fe, {"G": G, "s2": s2}
            info[f"chosen_{fam}"] = hp
        self.noise_sd = {f: float(np.sqrt(self.lmm[f]["s2"])) for f in FAMS}
        info["hazard_coef"] = self.fit_hazard(Dtr, norm)
        info["minutes"] = (time.time() - t0) / 60
        self.info = info
        return self

    def _fe_grid(self, fam, D, ii, vv, offsets):
        t0 = years(D, ii, vv)
        t = (t0[:, None] + offsets[None, :]).ravel()
        k = len(offsets)
        X = self.fe_inputs(t, np.repeat(D["X"][ii, 0, I_TF], k), np.repeat(D["S"][ii], k, 0))
        return self.fe[fam](X).reshape(len(ii), k), t0

    def latent_mean(self, start, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        out = {}
        for fam in FAMS:
            m, _ = self.posterior(fam, start, norm)
            f3, t0 = self._fe_grid(fam, D, ii, vv, np.array([-0.5, 0.0, 0.5]))
            out[fam] = (f3[:, 1] + m[:, 0] + m[:, 1] * t0, f3[:, 2] - f3[:, 0] + m[:, 1])
        return out

    def latent(self, start, n_mc, rng, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        k = np.arange(SIM.YEARS + 1, dtype=float)
        out = {}
        for fam in FAMS:
            m, V = self.posterior(fam, start, norm)
            b = draw(m, V, n_mc, rng)
            fk, t0 = self._fe_grid(fam, D, ii, vv, np.concatenate([k, k + 0.5, k - 0.5]))
            K = len(k)
            rep = np.repeat(np.arange(len(ii)), n_mc)
            tt = t0[rep][:, None] + k[None, :]
            val = fk[rep, :K] + b[:, :1] + b[:, 1:] * tt
            slope = fk[rep, K:2 * K] - fk[rep, 2 * K:] + b[:, 1:]
            out[fam] = (val, slope)
        return out


# ============================================================================== 分层高斯过程


class GPJoint(TwoStageHazard):
    """y_i(t) = X(t) β + g_i(t) + ε，g_i ~ GP(0, k)，k(t, t') = σ0² + σ1² t t' + σse² exp(−(t − t')² / 2ℓ²)（前两项 = 随机截距 + 斜率，
    第三项让个体轨迹可以弯）。β 和 (σ0, σ1, σse, ℓ, σε) 一起最大化训练段的边际似然（每人独立 -> 分块对角，torch 批量 Cholesky）。
    站点：GP 后验（到站点为止的观测）在站点前后 −0.5..10.5 年、每半年一个点上联合抽样；斜率 = 前后半年的差。"""

    def __init__(self, kw, device="cpu"):
        self.device = device
        self.dt = torch.float64

    def _k(self, a, b, p):
        s0, s1, sse, ell = (torch.exp(p[i]) for i in range(4))
        return s0 ** 2 + s1 ** 2 * a[..., :, None] * b[..., None, :] + sse ** 2 * torch.exp(
            -(a[..., :, None] - b[..., None, :]) ** 2 / (2 * ell ** 2))

    def _obs(self, fam, D, ii, vv, norm):
        T, Y, O = landmark_obs(D, ii, vv, fam, norm)
        X = design(T, np.repeat(D["X"][ii, 0, I_TF][:, None], T.shape[1], 1),
                   np.repeat(D["S"][ii][:, None, :], T.shape[1], 1))
        t = lambda a: torch.as_tensor(a, dtype=self.dt, device=self.device)   # noqa: E731
        return t(T), t(Y), torch.as_tensor(O, device=self.device), t(X)

    def _Kinv_r(self, fam, T, Y, O, X):
        p = self.par[fam]
        r = (Y - X @ p["beta"]) * O
        K = self._k(T, T, p["log"]) * (O[..., :, None] & O[..., None, :])
        n, V = O.shape
        eye = torch.eye(V, dtype=self.dt, device=self.device)
        K = K + eye * torch.where(O, torch.exp(2 * p["log"][4]), torch.ones((), dtype=self.dt, device=self.device))[..., None]
        L = torch.linalg.cholesky(K)
        return L, r

    def nll(self, fam, T, Y, O, X):
        L, r = self._Kinv_r(fam, T, Y, O, X)
        a = torch.cholesky_solve(r[..., None], L)[..., 0]
        logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        nobs = O.sum(-1)
        return 0.5 * ((r * a).sum(-1) + logdet + nobs * LOG2PI).sum() / nobs.sum()

    def fit(self, Dtr, Dte, norm):
        t0 = time.time()
        lm = {"ii": np.arange(len(Dtr["X"])), "vv": Dtr["M"].sum(1) - 1}
        self.par, info = {}, {}
        for fam in FAMS:
            T, Y, O, X = self._obs(fam, Dtr, lm["ii"], lm["vv"], norm)
            o = O.cpu().numpy()
            Xn, Yn = X.cpu().numpy()[o], Y.cpu().numpy()[o]
            beta0 = np.linalg.lstsq(Xn, Yn, rcond=None)[0]
            sd = float(np.std(Yn - Xn @ beta0))
            init = np.log([0.7 * sd, 0.1 * sd, 0.3 * sd, 3.0, 0.5 * sd])
            self.par[fam] = {"beta": torch.tensor(beta0, dtype=self.dt, device=self.device, requires_grad=True),
                             "log": torch.tensor(init, dtype=self.dt, device=self.device, requires_grad=True)}
            opt = torch.optim.LBFGS(list(self.par[fam].values()), lr=1.0, max_iter=300, line_search_fn="strong_wolfe",
                                    tolerance_change=1e-10)

            def closure(fam=fam, T=T, Y=Y, O=O, X=X):
                opt.zero_grad()
                l = self.nll(fam, T, Y, O, X)
                l.backward()
                return l
            opt.step(closure)
            self.par[fam] = {k: v.detach() for k, v in self.par[fam].items()}
            Tt, Yt, Ot, Xt = self._obs(fam, Dte, np.arange(len(Dte["X"])), Dte["M"].sum(1) - 1, norm)
            with torch.no_grad():
                info[fam] = {"train_nll_per_obs": float(self.nll(fam, T, Y, O, X)),
                             "test_nll_per_obs": float(self.nll(fam, Tt, Yt, Ot, Xt)),
                             "sigma0_sigma1_sigmaSE_ell_noise": torch.exp(self.par[fam]["log"]).cpu().numpy().round(4).tolist()}
            print(f"  gp {fam}: {info[fam]} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        self.noise_sd = {f: float(torch.exp(self.par[f]["log"][4])) for f in FAMS}
        info["hazard_coef"] = self.fit_hazard(Dtr, norm)
        info["minutes"] = (time.time() - t0) / 60
        self.info = info
        return self

    @torch.no_grad()
    def post(self, fam, start, norm, offsets):
        """站点后 offsets 年（相对站点时刻）上真实轨迹的后验：均值 (n, k)、协方差 (n, k, k)。"""
        D, ii, vv = start["D"], start["ii"], start["vv"]
        T, Y, O, X = self._obs(fam, D, ii, vv, norm)
        p = self.par[fam]
        L, r = self._Kinv_r(fam, T, Y, O, X)
        t0 = torch.as_tensor(years(D, ii, vv), dtype=self.dt, device=self.device)
        off = torch.as_tensor(offsets, dtype=self.dt, device=self.device)
        Ts = t0[:, None] + off[None, :]
        age0 = np.repeat(D["X"][ii, 0, I_TF][:, None], len(offsets), 1)
        Xs = torch.as_tensor(design(Ts.cpu().numpy(), age0, np.repeat(D["S"][ii][:, None, :], len(offsets), 1)),
                             dtype=self.dt, device=self.device)
        Kso = self._k(Ts, T, p["log"]) * O[:, None, :]
        Kss = self._k(Ts, Ts, p["log"])
        A = torch.cholesky_solve(Kso.transpose(1, 2), L)                     # (n, V, k)
        mean = Xs @ p["beta"] + (A.transpose(1, 2) @ r[..., None])[..., 0]
        cov = Kss - Kso @ A
        return mean.cpu().numpy(), cov.cpu().numpy()

    def latent_mean(self, start, norm):
        out = {}
        for fam in FAMS:
            mean, _ = self.post(fam, start, norm, np.array([-0.5, 0.0, 0.5]))
            out[fam] = (mean[:, 1], mean[:, 2] - mean[:, 0])
        return out

    def latent(self, start, n_mc, rng, norm):
        off = -0.5 + 0.5 * np.arange(2 * SIM.YEARS + 3)                     # −0.5, 0, 0.5, …, 10.5
        out = {}
        for fam in FAMS:
            mean, cov = self.post(fam, start, norm, off)
            g = draw(mean, cov, n_mc, rng)
            out[fam] = (g[:, 1::2], g[:, 2::2] - g[:, 0:-1:2])               # 第 k 年的值、第 k 年前后半年之差
        return out


# ============================================================================== 联合估计的联合模型


class JointModel:
    K_FIT = 48              # 拟合时每人几个随机效应抽样（固定的标准正态，重参数化）
    K_POST = 256            # 站点上重要性重抽样的候选数

    def __init__(self, kw, device="cpu"):
        self.device = device
        self.dt = torch.float64

    def t(self, a, dt=None):
        return torch.as_tensor(a, dtype=dt or self.dt, device=self.device)

    # ------------------------------------------------------------------ 数据
    def long_rows(self, D, norm, ii, vv):
        """每个"单元"（训练时是人，站点上是站点）一组测量：单元下标、标记（0 MMSE / 1 认知）、时间、值、固定效应设计。"""
        rows = {"u": [], "f": [], "t": [], "y": [], "X": []}
        for fam in FAMS:
            T, Y, O = landmark_obs(D, ii, vv, fam, norm)
            u, v = np.where(O)
            age0 = D["X"][ii, 0, I_TF][u]
            rows["u"].append(u); rows["f"].append(np.full(len(u), J[fam]))
            rows["t"].append(T[u, v]); rows["y"].append(Y[u, v]); rows["X"].append(design(T[u, v], age0, D["S"][ii][u]))
        return {k: np.concatenate(v) for k, v in rows.items()}

    def event_rows(self, D, ii, vv, after_last=True):
        """死亡：从基线到站点（训练时到最后一次访视之后的死亡 / 删失）的每个人年；AD：每次没有 AD 的访视到下一次（区间删失）。
        返回每一行的单元、时间（这一年起点，距基线年数）、年龄 z、类型（0 死亡 / 1 AD）、所属的组、组的结局。"""
        X, Y = D["X"], D["Y"]
        out = {"u": [], "t": [], "agez": [], "kind": [], "g": [], "gd": [], "gu": [], "gk": []}
        gid = 0
        age0 = X[ii, 0, I_TF]
        for u, (i, v) in enumerate(zip(ii, vv)):
            kv = int(round(X[i, v, I_TF + 1] * 10))
            n_alive, death_at = kv, None
            if after_last:
                yd = Y[i, v].reshape(B.NQ, B.NH)[B.I_DEATH]
                if (yd == 1).any():
                    death_at = kv + int(np.argmax(yd == 1)) + 1
                elif (yd == 0).any():
                    n_alive = kv + int(B.NH - np.argmax(yd[::-1] == 0))
            last = death_at if death_at is not None else n_alive
            for j in range(1, last + 1):                                     # 死亡：每个人年一组
                out["u"].append(u); out["t"].append(j - 1.0); out["agez"].append(age0[u] + (j - 1) * SIM.AGE_STEP_Z)
                out["kind"].append(0); out["g"].append(gid)
                out["gd"].append(int(death_at is not None and j == death_at)); out["gu"].append(u); out["gk"].append(0)
                gid += 1
            for w in range(v):                                               # AD：访视 w（没有 AD）到 w+1
                if X[i, w, I_AD] > 0.5 or not D["M"][i, w + 1]:
                    continue
                k0 = int(round(X[i, w, I_TF + 1] * 10))
                k1 = int(round(X[i, w + 1, I_TF + 1] * 10))
                for j in range(k0, k1):
                    out["u"].append(u); out["t"].append(float(j)); out["agez"].append(age0[u] + j * SIM.AGE_STEP_Z)
                    out["kind"].append(1); out["g"].append(gid)
                out["gd"].append(int(X[i, w + 1, I_AD] > 0.5)); out["gu"].append(u); out["gk"].append(1)
                gid += 1
        return {k: np.asarray(v) for k, v in out.items()}

    def pack(self, D, norm, ii, vv, after_last=True):
        L, E = self.long_rows(D, norm, ii, vv), self.event_rows(D, ii, vv, after_last)
        S = D["S"][ii]
        d = {"n": len(ii), "age0": self.t(D["X"][ii, 0, I_TF]), "e4": self.t(S[:, 9] + 2 * S[:, 10]), "S": self.t(S)}
        d.update({f"L_{k}": self.t(v, torch.long if k in ("u", "f") else None) for k, v in L.items()})
        d.update({f"E_{k}": self.t(v, torch.long if k in ("u", "kind", "g", "gu", "gk") else None) for k, v in E.items()})
        d["n_groups"] = int(E["gd"].shape[0])
        return d

    # ------------------------------------------------------------------ 模型
    def G(self):
        L = torch.tril(self.Lraw, -1) + torch.diag(torch.exp(torch.diagonal(self.Lraw)))
        return L @ L.T

    def long_post(self, d):
        """双变量线性混合模型：每个单元的对数边际似然 log p(y) 和随机效应后验 N(m, V)（Woodbury，4 × 4）。"""
        u, f, t, y, X = d["L_u"], d["L_f"], d["L_t"], d["L_y"], d["L_X"]
        n = d["n"]
        s2 = torch.exp(2 * self.logsig)[f]
        r = y - (X * self.beta[f]).sum(-1)
        z = torch.zeros(len(u), 4, dtype=self.dt, device=self.device)
        z = z.index_put((torch.arange(len(u), device=self.device), 2 * f), torch.ones_like(t))
        z = z.index_put((torch.arange(len(u), device=self.device), 2 * f + 1), t)
        ZtZ = torch.zeros(n, 4, 4, dtype=self.dt, device=self.device).index_add(0, u, z[:, :, None] * z[:, None, :] / s2[:, None, None])
        Ztr = torch.zeros(n, 4, dtype=self.dt, device=self.device).index_add(0, u, z * (r / s2)[:, None])
        rr = torch.zeros(n, dtype=self.dt, device=self.device).index_add(0, u, r ** 2 / s2)
        lr = torch.zeros(n, dtype=self.dt, device=self.device).index_add(0, u, torch.log(s2))
        no = torch.zeros(n, dtype=self.dt, device=self.device).index_add(0, u, torch.ones_like(t))
        G = self.G()
        Gi = torch.linalg.inv(G)
        A = Gi[None] + ZtZ
        LA = torch.linalg.cholesky(A)
        V = torch.cholesky_inverse(LA)
        m = (V @ Ztr[..., None])[..., 0]
        quad = rr - (Ztr * m).sum(-1)
        ll = -0.5 * (no * LOG2PI + lr + torch.logdet(G) + 2 * torch.log(torch.diagonal(LA, dim1=-2, dim2=-1)).sum(-1) + quad)
        return ll, m, V

    def traj(self, b, u, t, d):
        """b (单元, K, 4) -> 行上的 [MMSE 真值, 斜率, 认知真值, 斜率]，(行, K, 4)。"""
        age0, e4, S = d["age0"][u], d["e4"][u], d["S"][u]
        out = []
        for f in (0, 1):
            Xd = torch.stack([torch.ones_like(t), t, age0, S[:, 0], S[:, 1], e4, t * age0, t * e4], -1)
            be = self.beta[f]
            val = (Xd @ be)[:, None] + b[u, :, 2 * f] + b[u, :, 2 * f + 1] * t[:, None]
            slope = (be[1] + be[6] * age0 + be[7] * e4)[:, None] + b[u, :, 2 * f + 1]
            out += [val, slope]
        return torch.stack(out, -1)

    def eta(self, feats, agez, S, g):
        """logit h = γ0 + γ·标准化([4 个轨迹量, 年龄 z, 性别, 教育, ε4 一个, 两个])。feats (行, K, 4)；g (行, 10) = 这一行
        用的风险参数（死亡或 AD 那一组）。"""
        w = torch.stack([agez, S[:, 0], S[:, 1], S[:, 9], S[:, 10]], -1)[:, None, :].expand(-1, feats.shape[1], -1)
        F = (torch.cat([feats, w], -1) - self.hmu) / self.hsd
        return g[:, None, 0] + (F * g[:, None, 1:]).sum(-1)

    def event_ll(self, b, d):
        """b (单元, K, 4) -> 每个单元的事件对数似然 (单元, K)。"""
        u, t, agez, kind, g = d["E_u"], d["E_t"], d["E_agez"], d["E_kind"], d["E_g"]
        if len(u) == 0:
            return torch.zeros(d["n"], b.shape[1], dtype=self.dt, device=self.device)
        feats = self.traj(b, u, t, d)
        et = self.eta(feats, agez, d["S"][u], self.gam_p[kind])
        s = torch.zeros(d["n_groups"], b.shape[1], dtype=self.dt, device=self.device).index_add(
            0, g, torch.nn.functional.logsigmoid(-et))                         # 每组 Σ log(1 − h)
        gd = d["E_gd"][:, None]
        lg = gd * torch.log(-torch.expm1(s.clamp(max=-1e-12))) + (1 - gd) * s
        return torch.zeros(d["n"], b.shape[1], dtype=self.dt, device=self.device).index_add(0, d["E_gu"], lg)

    def nll(self, d, eps):
        ll, m, V = self.long_post(d)
        Lv = torch.linalg.cholesky(V + 1e-9 * torch.eye(4, dtype=self.dt, device=self.device))
        b = m[:, None, :] + (Lv[:, None] @ eps[None, :, :, None])[..., 0]       # (n, K, 4)
        ev = self.event_ll(b, d)
        lme = torch.logsumexp(ev, 1) - float(np.log(eps.shape[0]))
        return -(ll + lme).mean(), -ll.mean(), -lme.mean()

    # ------------------------------------------------------------------ 拟合
    def fit(self, Dtr, Dte, norm):
        import statsmodels.api as sm
        t0 = time.time()
        n = len(Dtr["X"])
        last = Dtr["M"].sum(1) - 1
        d = self.pack(Dtr, norm, np.arange(n), last)
        # 起点：两条标记各一个线性混合模型（lmm-joint 那一步），G 取两块对角拼起来
        beta, logsig, Gb = [], [], []
        for fam in FAMS:
            k = d["L_f"].cpu().numpy() == J[fam]
            X, y = d["L_X"].cpu().numpy()[k], d["L_y"].cpu().numpy()[k]
            tt, uu = d["L_t"].cpu().numpy()[k], d["L_u"].cpu().numpy()[k]
            res = sm.MixedLM(y, X, groups=uu, exog_re=np.stack([np.ones_like(tt), tt], 1)).fit(reml=True, method=["lbfgs"])
            beta.append(np.asarray(res.fe_params)); logsig.append(0.5 * np.log(res.scale)); Gb.append(np.asarray(res.cov_re))
        G0 = np.zeros((4, 4))
        G0[:2, :2], G0[2:, 2:] = Gb[0], Gb[1]
        L0 = np.linalg.cholesky(G0 + 1e-6 * np.eye(4))
        self.beta = self.t(np.stack(beta)).requires_grad_(True)
        self.logsig = self.t(np.array(logsig)).requires_grad_(True)
        self.Lraw = self.t(np.tril(L0, -1) + np.diag(np.log(np.diag(L0)))).requires_grad_(True)
        # 风险输入的标准化常数：起点参数下的后验均值轨迹
        with torch.no_grad():
            _, m, _ = self.long_post(d)
            feats = self.traj(m[:, None, :], d["E_u"], d["E_t"], d)[:, 0]
            w = torch.stack([d["E_agez"], d["S"][d["E_u"], 0], d["S"][d["E_u"], 1], d["S"][d["E_u"], 9], d["S"][d["E_u"], 10]], -1)
            F = torch.cat([feats, w], -1)
            self.hmu, self.hsd = F.mean(0), F.std(0) + 1e-8
        rate = [float(d["E_gd"][d["E_gk"] == k].float().mean()) for k in (0, 1)]
        g0 = np.zeros((2, 10))
        g0[:, 0] = np.log(np.array(rate) / (1 - np.array(rate)))
        self.gam_p = self.t(g0).requires_grad_(True)                          # (死亡 / AD, 截距 + 9 个输入)
        gen = torch.Generator().manual_seed(0)
        eps = torch.randn(self.K_FIT, 4, generator=gen, dtype=self.dt).to(self.device)
        params = [self.beta, self.logsig, self.Lraw, self.gam_p]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=400, line_search_fn="strong_wolfe", tolerance_change=1e-10,
                                history_size=30)

        def closure():
            opt.zero_grad()
            l, _, _ = self.nll(d, eps)
            l.backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            tot, lo, ev = (float(x) for x in self.nll(d, eps))
            dte = self.pack(Dte, norm, np.arange(len(Dte["X"])), Dte["M"].sum(1) - 1)
            te = [float(x) for x in self.nll(dte, eps)]
        for p in params:
            p.requires_grad_(False)
        self.noise_sd = {f: float(torch.exp(self.logsig[J[f]])) for f in FAMS}
        G = self.G().cpu().numpy()
        sdv = np.sqrt(np.diag(G))
        self.info = {"train_nll": tot, "train_nll_long": lo, "train_nll_event": ev, "test_nll": te[0],
                     "test_nll_long": te[1], "test_nll_event": te[2], "minutes": (time.time() - t0) / 60,
                     "beta": self.beta.cpu().numpy().round(4).tolist(), "sigma": [self.noise_sd[f] for f in FAMS],
                     "G_corr": (G / np.outer(sdv, sdv)).round(3).tolist(), "G_sd": sdv.round(4).tolist(),
                     "hazard_death_ad": self.gam_p.cpu().numpy().round(3).tolist(),
                     "hazard_inputs": ["intercept", "mmse_value", "mmse_slope", "cogn_value", "cogn_slope", "age_z", "male",
                                       "educ_z", "e4x1", "e4x2"]}
        print(f"  jm: train nll {tot:.4f} (long {lo:.4f}, event {ev:.4f}); test {te[0]:.4f} "
              f"({self.info['minutes']:.1f} min)", flush=True)
        return self

    # ------------------------------------------------------------------ 预测
    @torch.no_grad()
    def post_draws(self, start, n_mc, rng, norm):
        """站点：b ~ p(b | 到站点为止的测量) 抽 K_POST 个候选，按到站点为止的事件历史（活着、AD 的区间）加权，重抽 n_mc 个。"""
        D, ii, vv = start["D"], start["ii"], start["vv"]
        d = self.pack(D, norm, ii, vv, after_last=False)
        _, m, V = self.long_post(d)
        Lv = torch.linalg.cholesky(V + 1e-9 * torch.eye(4, dtype=self.dt, device=self.device))
        g = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        eps = torch.randn(d["n"], self.K_POST, 4, generator=g, dtype=self.dt).to(self.device)
        b = m[:, None, :] + (Lv[:, None] @ eps[..., None])[..., 0]
        lw = self.event_ll(b, d)
        w = torch.softmax(lw, 1).cpu().numpy()
        b = b.cpu().numpy()
        idx = np.stack([rng.choice(self.K_POST, n_mc, p=w[i] / w[i].sum()) for i in range(d["n"])])
        return b[np.arange(d["n"])[:, None], idx].reshape(-1, 4)

    def latent(self, start, n_mc, rng, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        b = self.post_draws(start, n_mc, rng, norm)
        rep = np.repeat(np.arange(len(ii)), n_mc)
        t0 = years(D, ii, vv)[rep]
        k = np.arange(SIM.YEARS + 1, dtype=float)
        tt = t0[:, None] + k[None, :]
        age0 = D["X"][ii, 0, I_TF][rep]
        S = D["S"][ii][rep]
        be = self.beta.cpu().numpy()
        out = {}
        for fam in FAMS:
            f = J[fam]
            Xd = design(tt, age0[:, None].repeat(len(k), 1), S[:, None, :].repeat(len(k), 1))
            val = Xd @ be[f] + b[:, None, 2 * f] + b[:, None, 2 * f + 1] * tt
            e4 = S[:, 9] + 2 * S[:, 10]
            slope = (be[f, 1] + be[f, 6] * age0 + be[f, 7] * e4 + b[:, 2 * f + 1])[:, None].repeat(len(k), 1)
            out[fam] = (val, slope)
        return out

    @torch.no_grad()
    def hazard_probs(self, vals, age_z, S):
        feats = self.t(np.stack([vals["mmse"][0], vals["mmse"][1], vals["cogn"][0], vals["cogn"][1]], -1))[:, None, :]
        St, az = self.t(S), self.t(age_z)
        out = [torch.sigmoid(self.eta(feats, az, St, self.gam_p[k][None].expand(len(St), -1)))[:, 0].cpu().numpy()
               for k in (0, 1)]
        return out[0], out[1]

    def sim(self, start, n_mc, rng, norm):
        return simulate(self, start, n_mc, rng, norm)
