"""hmm.py -- 隐 Markov 模型（latent Markov），照 Yu et al., Neurobiology of Aging 2015（ROS/MAP）的结构：
3 个潜在的认知分期（只能往前走：1 -> 2 -> 3，不回退）+ 死亡（吸收态，死亡时间观测到）。原文用 msm 的连续时间 HMM、
全局认知做发射；这里在 1 年网格上做离散时间版本（访视本来就是一年一次），发射加上 MMSE 和"这次访视新诊断 AD"，
这样模拟出来的轨迹才有 MMSE 分期和 AD。

  转移：每个起始分期一个多项逻辑回归（年龄、性别、教育、APOE ε4 一个 / 两个），只允许 停留 / 往前 / 死亡
  发射：每个分期 MMSE ~ N(μ, σ)、全局认知 ~ N(μ, σ)（各自只在测了的访视上算），新诊断 AD ~ Bernoulli(p)（只在还没有 AD 的访视上算）
  起始分布：多项逻辑回归（第一次访视的年龄等同一组协变量）
  拟合：EM（前向-后向 + 加权回归），30 轮
  时间网格：第一次访视起每年一格；没来的年份 = 活着但没观测；最后一次访视之后死亡的，按死亡标签放在第 T 格。
模拟：用到起点为止的观测做前向滤波，得到起点那一年潜在分期的后验，抽一个，往后逐年转移、发射。
"""
import time

import numpy as np

import bl_data as B
import simulator as SIM
from bl_data import I_AD, I_MC, I_TF

K = 4                                         # 3 个潜在分期 + 死亡
DEAD = 3
ALLOWED = {0: (0, 1, 2, 3), 1: (1, 2, 3), 2: (2, 3)}


def covars(age_z, S):
    return np.stack([age_z, S[..., 0], S[..., 1], S[..., 9], S[..., 10]], -1)


class Grid:
    """人级访视 -> 逐年网格上的观测。只用到第 upto 次访视为止（None = 全部 + 死亡）。"""

    def __init__(self, D, norm, upto=None, ii=None, with_death=True, extra=0):
        ii = np.arange(len(D["pid"])) if ii is None else ii
        X, M = D["X"][ii], D["M"][ii]
        n = len(ii)
        upto = (M.sum(1) - 1) if upto is None else upto
        yrs = np.round(X[..., I_TF + 1] * 10.0).astype(int)
        last_y = yrs[np.arange(n), upto]
        dT = np.zeros(n, int)
        if with_death:
            Y = D["Y"][ii].reshape(n, X.shape[1], B.NQ, B.NH)[np.arange(n), upto, B.I_DEATH]
            has = (Y == 1).any(1)
            dT = np.where(has, (Y == 1).argmax(1) + 1, 0)
        self.Tn = last_y + dT + 1 + extra                    # 每人的格数
        T = int(self.Tn.max())
        self.mmse = np.full((n, T), np.nan)
        self.cogn = np.full((n, T), np.nan)
        self.visit = np.zeros((n, T), bool)
        self.ad_new = np.zeros((n, T), np.int8)              # 1 = 这次新诊断；0 = 这次还没有；-1 = 不算（已经有了 / 没访视）
        self.ad_new[:] = -1
        self.dead_at = np.where(dT > 0, last_y + dT, -1)
        had = np.zeros(n, bool)
        for v in range(int(upto.max()) + 1):
            ok = (v <= upto) & M[:, v]
            r = np.where(ok)[0]
            y = yrs[r, v]
            x = X[r, v]
            self.visit[r, y] = True
            for arr, j in ((self.mmse, 0), (self.cogn, 1)):
                m = x[:, I_MC + j] < 0.5
                arr[r[m], y[m]] = x[m, j] * norm["sd"][j] + norm["mu"][j]
            ad = x[:, I_AD] > 0.5
            self.ad_new[r, y] = np.where(had[r], -1, ad.astype(np.int8))
            had[r] |= ad
        self.had_ad = had
        self.age0 = X[:, 0, I_TF]
        self.S = D["S"][ii]
        self.last_y = last_y


class HMM:
    def __init__(self, n_iter=30, C=1.0):
        self.n_iter, self.C = n_iter, C

    # --- 参数
    def _trans(self, cov):
        """cov (..., 5) -> (..., 4, 4) 一年转移矩阵。"""
        sh = cov.shape[:-1]
        A = np.zeros(sh + (K, K))
        A[..., DEAD, DEAD] = 1.0
        if A.size == 0:                                       # 只有一格的网格（从第一次访视起步）：没有转移
            return A
        Cz = (cov - self.cmu) / self.csd
        for i, allowed in ALLOWED.items():
            if self.tr_models is None:
                p = self.tr_init[i]
            else:
                m = self.tr_models[i]
                pr = m.predict_proba(Cz.reshape(-1, Cz.shape[-1])).reshape(sh + (len(m.classes_),))
                p = np.zeros(sh + (K,))
                p[..., m.classes_] = pr
            A[..., i, :] = p
        return A

    def _pi(self, cov0):
        if self.pi_model is None:
            return np.broadcast_to(np.array([0.7, 0.2, 0.1, 0.0]), cov0.shape[:-1] + (K,))
        pr = self.pi_model.predict_proba((cov0 - self.cmu) / self.csd)
        p = np.zeros((len(cov0), K))
        p[:, self.pi_model.classes_] = pr
        return p

    def _emis(self, g):
        """(n, T, 4) 的发射似然。"""
        n, T = g.mmse.shape
        L = np.ones((n, T, K))
        for arr, mu, sd in ((g.mmse, self.mu_m, self.sd_m), (g.cogn, self.mu_c, self.sd_c)):
            ok = np.isfinite(arr)
            z = (np.nan_to_num(arr)[..., None] - mu[None, None, :3]) / sd[None, None, :3]
            dens = np.exp(-0.5 * z ** 2) / (sd[None, None, :3] * np.sqrt(2 * np.pi))
            L[..., :3] *= np.where(ok[..., None], dens, 1.0)
        a = g.ad_new
        pa = self.p_ad[None, None, :3]
        L[..., :3] *= np.where((a == 1)[..., None], pa, np.where((a == 0)[..., None], 1 - pa, 1.0))
        # 活着的格（访视、没访视但死亡之前）不能是死亡；死亡那格只能是死亡
        t = np.arange(T)[None, :]
        inside = t < g.Tn[:, None]
        is_death = t == g.dead_at[:, None]
        L[..., DEAD] = np.where(is_death, 1.0, 0.0)
        L[..., :3] *= ~is_death[..., None]
        L[~inside] = 1.0
        return L

    def _cov_grid(self, g):
        T = g.mmse.shape[1]
        age = g.age0[:, None] + np.arange(T)[None, :] * SIM.AGE_STEP_Z
        return covars(age, g.S[:, None, :].repeat(T, 1))

    def forward_backward(self, g):
        n, T = g.mmse.shape
        L = self._emis(g)
        cov = self._cov_grid(g)
        A = self._trans(cov[:, :-1])                                  # (n, T-1, 4, 4)
        alpha = np.zeros((n, T, K))
        c = np.zeros((n, T))
        a = self._pi(cov[:, 0]) * L[:, 0]
        c[:, 0] = a.sum(1)
        alpha[:, 0] = a / c[:, :1]
        inside = np.arange(T)[None, :] < g.Tn[:, None]
        for t in range(1, T):
            a = np.einsum("ni,nij->nj", alpha[:, t - 1], A[:, t - 1]) * L[:, t]
            s = a.sum(1)
            a = np.where(inside[:, t:t + 1], a / np.where(s > 0, s, 1)[:, None], alpha[:, t - 1])
            c[:, t] = np.where(inside[:, t], s, 1.0)
            alpha[:, t] = a
        beta = np.ones((n, T, K))
        for t in range(T - 2, -1, -1):
            b = np.einsum("nij,nj->ni", A[:, t], L[:, t + 1] * beta[:, t + 1]) / c[:, t + 1:t + 2]
            beta[:, t] = np.where(inside[:, t + 1:t + 2], b, 1.0)
        gamma = alpha * beta
        gamma /= gamma.sum(-1, keepdims=True)
        xi = (alpha[:, :-1, :, None] * A * (L[:, 1:] * beta[:, 1:])[:, :, None, :]) / c[:, 1:, None, None]
        xi *= inside[:, 1:, None, None]
        ll = float(np.log(np.where(inside, c, 1.0)).sum())
        return gamma, xi, alpha, ll, cov

    def fit(self, Dtr, norm):
        from sklearn.linear_model import LogisticRegression
        t0 = time.time()
        g = Grid(Dtr, norm)
        self.mu_m, self.sd_m = np.array([29.0, 26.5, 18.0, 0]), np.array([1.5, 2.5, 6.0, 1])
        self.mu_c, self.sd_c = np.array([0.3, -0.4, -1.8, 0]), np.array([0.5, 0.6, 0.9, 1])
        self.p_ad = np.array([0.002, 0.03, 0.25, 0])
        self.tr_init = {0: np.array([0.85, 0.1, 0.0, 0.05]), 1: np.array([0, 0.8, 0.12, 0.08]),
                        2: np.array([0, 0, 0.85, 0.15])}
        self.tr_models, self.pi_model = None, None
        cov = self._cov_grid(g)
        inside = np.arange(cov.shape[1])[None, :] < g.Tn[:, None]
        allc = cov[inside]
        self.cmu, self.csd = allc.mean(0), allc.std(0) + 1e-8
        self.ll_trace = []
        for it in range(self.n_iter):
            gamma, xi, _, ll, cov = self.forward_backward(g)
            self.ll_trace.append(ll)
            w = gamma[..., :3] * inside[..., None]
            for arr, mu, sd in ((g.mmse, self.mu_m, self.sd_m), (g.cogn, self.mu_c, self.sd_c)):
                ok = np.isfinite(arr)[..., None]
                ww = w * ok
                x = np.nan_to_num(arr)[..., None]
                mu[:3] = (ww * x).sum((0, 1)) / ww.sum((0, 1))
                sd[:3] = np.sqrt((ww * (x - mu[None, None, :3]) ** 2).sum((0, 1)) / ww.sum((0, 1))) + 1e-3
            a = g.ad_new[..., None]
            wa = w * (a >= 0)
            self.p_ad[:3] = np.clip((wa * (a == 1)).sum((0, 1)) / wa.sum((0, 1)), 1e-4, 0.99)
            # 转移：每个起始分期一个加权多项逻辑回归
            Cz = (cov[:, :-1] - self.cmu) / self.csd
            models = {}
            for i, allowed in ALLOWED.items():
                Xr, yr, wr = [], [], []
                for j in allowed:
                    wt = xi[:, :, i, j]
                    m = wt > 1e-6
                    Xr.append(Cz[m])
                    yr.append(np.full(m.sum(), j))
                    wr.append(wt[m])
                Xr, yr, wr = np.concatenate(Xr), np.concatenate(yr), np.concatenate(wr)
                models[i] = LogisticRegression(C=self.C, max_iter=500).fit(Xr, yr, sample_weight=wr)
            self.tr_models = models
            g0 = gamma[:, 0, :3]
            Xp = np.repeat((cov[:, 0] - self.cmu) / self.csd, 3, 0)
            self.pi_model = LogisticRegression(C=self.C, max_iter=500).fit(
                Xp, np.tile(np.arange(3), len(g0)), sample_weight=g0.reshape(-1) + 1e-9)
            if it > 2 and abs(self.ll_trace[-1] - self.ll_trace[-2]) < 1e-3 * abs(self.ll_trace[-1]) * 1e-3:
                break
        self.minutes = (time.time() - t0) / 60
        self.info = {"iterations": it + 1, "log_lik": self.ll_trace[-1], "ll_first": self.ll_trace[0],
                     "mmse_mu": self.mu_m[:3].tolist(), "mmse_sd": self.sd_m[:3].tolist(),
                     "cogn_mu": self.mu_c[:3].tolist(), "cogn_sd": self.sd_c[:3].tolist(),
                     "p_new_ad": self.p_ad[:3].tolist(), "minutes": self.minutes}
        return self

    def sim(self, start, n_mc, rng, norm, years=SIM.YEARS):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        n = len(ii)
        g = Grid(D, norm, upto=vv, ii=ii, with_death=False)
        _, _, alpha, _, _ = self.forward_backward(g)
        post = alpha[np.arange(n), g.last_y]                      # 起点那一年的滤波分布
        post = post[:, :3] / post[:, :3].sum(1, keepdims=True)
        N = n * n_mc
        rep = np.repeat(np.arange(n), n_mc)
        s = (post[rep].cumsum(1) < rng.random(N)[:, None]).sum(1).clip(max=2)
        stage = {f: np.full((N, years + 1), -1, np.int64) for f in ("mmse", "cogn")}
        for f, j in (("mmse", 0), ("cogn", 1)):
            x = D["X"][ii, vv]
            raw = x[:, j] * norm["sd"][j] + norm["mu"][j]
            stage[f][:, 0] = np.repeat(np.where(x[:, I_MC + j] < 0.5, SIM.stage_of(raw, f), -1), n_mc)
        t_death = np.full(N, np.inf)
        has_ad = np.repeat(D["X"][ii, vv, I_AD] > 0.5, n_mc)
        t_ad = np.where(has_ad, -np.inf, np.inf)
        age0 = D["X"][ii, vv, I_TF][rep]
        S = D["S"][ii][rep]
        for k in range(1, years + 1):
            r = np.where(~np.isfinite(t_death))[0]
            if not len(r):
                break
            A = self._trans(covars(age0[r] + (k - 1) * SIM.AGE_STEP_Z, S[r]))[np.arange(len(r)), s[r]]
            nxt = (A.cumsum(1) < rng.random(len(r))[:, None]).sum(1).clip(max=DEAD)
            dies = nxt == DEAD
            t_death[r[dies]] = k - 1 + rng.random(dies.sum())
            al = r[~dies]
            s[al] = nxt[~dies]
            st = s[al]
            mm = np.clip(np.round(self.mu_m[st] + self.sd_m[st] * rng.standard_normal(len(al))), 0, 30)
            cg = self.mu_c[st] + self.sd_c[st] * rng.standard_normal(len(al))
            stage["mmse"][al, k] = SIM.stage_of(mm, "mmse")
            stage["cogn"][al, k] = SIM.stage_of(cg, "cogn")
            new = (t_ad[al] == np.inf) & (rng.random(len(al)) < self.p_ad[st])
            t_ad[al[new]] = k
        rs = lambda a: a.reshape(n, n_mc, *a.shape[1:])            # noqa: E731
        nvis = np.minimum(np.where(np.isfinite(t_death), np.floor(t_death), years), years).astype(int)
        return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
                "n_visits": rs(nvis)}
