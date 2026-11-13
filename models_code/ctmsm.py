"""ctmsm.py -- 连续时间多状态 Markov 模型（Kalbfleisch & Lawless 1985；R 包 msm，Jackson 2011 的做法），用 torch 写（RIS 上没有 R）。

和离散时间的 markov / msm-cov（generators.MSM）同一组状态：(AD 有无, MMSE 分期, 认知分期) 的 18 个活着的组合 + 死亡。不同在于：
  * 连续时间：转移强度矩阵 Q，Δt 年后的转移概率 P(Δt) = expm(Q Δt)。一瞬间只能走一步（MMSE 或认知分期 ±1、出现 AD、死亡），
    一年里走两步（比如 MMSE 正常 -> 受损）由 expm 自动算进去；间隔 2、3 年的访视对照样能用（离散版只能用间隔正好 1 年的配对）。
  * 面板数据似然：访视之间的状态没观察到，只在访视上看到。某次访视 MMSE 或认知没测 = "删失状态"（msm 的 censor）：
    这次的观测只说明状态落在一组可能的状态里；前向算法把它们加起来，所以所有访视都能用（离散版要求两边分期都测了）。
  * 死亡：死亡年份已知（死在最后一次访视之后的第 k 年内）-> 似然 = Σ_r P(最后一次的状态 -> r, k−1 年) · P(r -> 死亡, 1 年)。
  * 协变量（msm 的 covariates + constraint）：log q_rs = 基线 + β_type · z，z = 年龄（每个区间起点的年龄，随时间变）、性别、教育、
    APOE ε4 一个 / 两个；β 按转移类型共享（MMSE 变差 / 好转、认知变差 / 好转、出现 AD、死亡，6 类）。
    基线强度每个转移一个；为了稀少的转移不跑到 0，对"每个转移偏离它那一类的平均"加一个小的岭惩罚。
  * 起点：第一次访视的状态按训练段第一次访视的经验分布、限制在和观测相容的那几个状态上。
站点上的风险：先用到站点为止的全部访视做前向滤波，得到站点时状态的后验分布（分期没测的站点不再用"最近一次测量"顶替），
再按年把一年转移矩阵连乘、目标状态在整年上吸收（和离散版、和"在年度访视上测到"的题意一致），精确算，不抽样。
"""
import time

import numpy as np
import torch

import bl_data as B
import simulator as SIM
from bl_data import I_AD, I_TF
from generators import _stage_measured

NS, DEAD = 19, 18
N_TYPE = 6                                     # 0 MMSE 变差 1 MMSE 好转 2 认知变差 3 认知好转 4 出现 AD 5 死亡


def _transitions():
    src, dst, typ = [], [], []
    for a in (0, 1):
        for sm in range(3):
            for sc in range(3):
                r = a * 9 + sm * 3 + sc
                for d, t in ((1, 0), (-1, 1)):
                    if 0 <= sm + d <= 2:
                        src.append(r); dst.append(a * 9 + (sm + d) * 3 + sc); typ.append(t)
                for d, t in ((1, 2), (-1, 3)):
                    if 0 <= sc + d <= 2:
                        src.append(r); dst.append(a * 9 + sm * 3 + sc + d); typ.append(t)
                if a == 0:
                    src.append(r); dst.append(9 + sm * 3 + sc); typ.append(4)
                src.append(r); dst.append(DEAD); typ.append(5)
    return np.array(src), np.array(dst), np.array(typ)


SRC, DST, TYP = _transitions()
ST = np.arange(18)
A_OF, SM_OF, SC_OF = ST // 9, (ST // 3) % 3, ST % 3


def covars(X, S):
    """(…, 54) 的访视 + (…, 25) 静态 -> 5 个协变量：年龄 z、男、教育 z、ε4 一个、两个（和 msm-cov 一样）。"""
    return np.stack([X[..., I_TF], S[..., 0], S[..., 1], S[..., 9], S[..., 10]], -1)


def compat(D, ii, vv, norm):
    """访视上的观测 -> 和它相容的状态（18 个活着的状态的 0/1 向量）。AD 状态总是知道；分期没测 = 三期都可能。"""
    sm = _stage_measured(D, ii, vv, norm, "mmse")
    sc = _stage_measured(D, ii, vv, norm, "cogn")
    a = (D["X"][ii, vv, I_AD] > 0.5).astype(int)
    E = (A_OF[None] == a[:, None]) & ((sm[:, None] < 0) | (SM_OF[None] == sm[:, None])) & \
        ((sc[:, None] < 0) | (SC_OF[None] == sc[:, None]))
    return E.astype(np.float64)


class CTMSM:
    def __init__(self, device="cpu", pens=(1.0,)):
        self.device, self.pens = device, pens
        self.dt = torch.float64

    # ------------------------------------------------------------------ 数据
    def person_data(self, D, norm, upto=None, death=True):
        """每个人一行：E (n, V, 18) 相容向量、z (n, V, 5)、gap (n, V)、有没有这次访视 (n, V)；死亡信息 kd（最后一次访视之后
        第 kd 年内死亡，0 = 不知道）、alive_to（最后一次之后至少活到第几年，0 = 不知道）。upto：每人只用到第 upto 次访视。"""
        X, M = D["X"], D["M"].copy()
        n, V = M.shape
        if upto is not None:
            M &= np.arange(V)[None, :] <= np.asarray(upto)[:, None]
        E = np.zeros((n, V, 18))
        ii, vv = np.where(M)
        E[ii, vv] = compat(D, ii, vv, norm)
        z = (covars(X, D["S"][:, None, :].repeat(V, 1)) - self.zmu) / self.zsd
        gap = X[..., I_TF + 2].astype(np.float64)
        last = M.sum(1) - 1
        kd = np.zeros(n, int)
        alive_to = np.zeros(n, int)
        if death:
            yd = D["Y"][np.arange(n), last].reshape(n, B.NQ, B.NH)[:, B.I_DEATH]
            has = (yd == 1).any(1)
            kd = np.where(has, (yd == 1).argmax(1) + 1, 0)
            neg = yd == 0
            alive_to = np.where(~has & neg.any(1), B.NH - neg[:, ::-1].argmax(1), 0)
        t = lambda a: torch.as_tensor(a, dtype=self.dt, device=self.device)    # noqa: E731
        return {"E": t(E), "z": t(z), "gap": t(gap), "M": torch.as_tensor(M, device=self.device), "last": last,
                "kd": kd, "alive_to": alive_to, "zlast": t(z[np.arange(n), last])}

    # ------------------------------------------------------------------ 模型
    def Q(self, z):
        """z (m, 5) -> (m, 19, 19) 强度矩阵。"""
        lq = self.base[None, :] + (z @ self.beta.T)[:, self.typ]                    # (m, 转移数)
        Qf = lq.new_zeros(len(z), NS * NS)
        Qf[:, self.flat] = torch.exp(lq)
        Q = Qf.view(-1, NS, NS)
        return Q - torch.diag_embed(Q.sum(-1))

    def P(self, z, dt):
        return torch.linalg.matrix_exp(self.Q(z) * dt.reshape(-1, 1, 1))

    def P_year(self, zk):
        """一年的转移矩阵；zk 是这一年起点的协变量。"""
        return torch.linalg.matrix_exp(self.Q(zk))

    def _age_shift(self, z, k):
        z = z.clone()
        z[:, 0] = z[:, 0] + k * SIM.AGE_STEP_Z / self.zsd[0]
        return z

    def forward_filter(self, d, keep_all=False):
        """前向算法：返回每人的对数似然（访视部分）和最后一次（或 upto）访视上的状态后验 α (n, 19)。"""
        E, z, gap, M = d["E"], d["z"], d["gap"], d["M"]
        n, V, _ = E.shape
        pad = E.new_zeros(n, 1)
        a = self.pi0[None, :18] * E[:, 0]
        c = a.sum(1, keepdim=True).clamp(min=1e-300)
        alpha = torch.cat([a / c, pad], 1)
        ll = E.new_zeros(n)
        hist = [alpha] if keep_all else None
        for v in range(1, V):
            on = M[:, v]
            if not bool(on.any()):
                break
            idx = torch.where(on)[0]
            Pm = self.P(z[idx, v - 1], gap[idx, v])
            a = torch.einsum("ns,nst->nt", alpha[idx], Pm)[:, :18] * E[idx, v]
            c = a.sum(1, keepdim=True).clamp(min=1e-300)
            ll = ll.index_add(0, idx, torch.log(c[:, 0]))
            new = alpha.clone()
            new[idx] = torch.cat([a / c, pad[idx]], 1)
            alpha = new
            if keep_all:
                hist.append(alpha)
        return ll, alpha, hist

    def death_ll(self, d, alpha):
        """最后一次访视之后的死亡信息：第 kd 年内死亡，或者至少活到第 alive_to 年。逐年走（年龄每年加一岁）。"""
        n = len(alpha)
        ll = alpha.new_zeros(n)
        kd = torch.as_tensor(d["kd"], device=self.device)
        at = torch.as_tensor(d["alive_to"], device=self.device)
        need = (kd > 0) | (at > 0)
        if not bool(need.any()):
            return ll
        idx = torch.where(need)[0]
        p = alpha[idx]
        zl = d["zlast"][idx]
        out = alpha.new_zeros(len(idx))
        kdi, ati = kd[idx], at[idx]
        for k in range(1, B.NH + 1):
            Pk = self.P_year(self._age_shift(zl, k - 1))
            p_new = torch.einsum("ns,nst->nt", p, Pk)
            hit = kdi == k
            if bool(hit.any()):
                out = torch.where(hit, torch.log(p_new[:, DEAD].clamp(min=1e-300)), out)
            al = ati == k
            if bool(al.any()):
                out = torch.where(al, torch.log(p_new[:, :18].sum(1).clamp(min=1e-300)), out)
            p = torch.cat([p_new[:, :18], torch.zeros_like(p_new[:, 18:])], 1)        # 只往后推活着的部分
        return ll.index_add(0, idx, out)

    def nll(self, d, pen):
        ll, alpha, _ = self.forward_filter(d)
        ll = ll + self.death_ll(d, alpha)
        dev = self.base - self.base_type_mean()
        return -ll.mean() + pen * (dev ** 2).sum() / len(ll)

    def base_type_mean(self):
        m = torch.zeros(N_TYPE, dtype=self.dt, device=self.device).index_add(0, self.typ, self.base)
        cnt = torch.zeros(N_TYPE, dtype=self.dt, device=self.device).index_add(0, self.typ, torch.ones_like(self.base))
        return (m / cnt)[self.typ]

    # ------------------------------------------------------------------ 拟合
    def fit(self, Dtr, Dte, norm):
        t0 = time.time()
        n0 = len(Dtr["X"])
        C = covars(Dtr["X"][:, 0], Dtr["S"])
        self.zmu, self.zsd = C.mean(0), C.std(0) + 1e-8
        # 起点分布：训练段第一次访视、分期都测了的人的经验分布（+0.5 平滑）
        E0 = compat(Dtr, np.arange(n0), np.zeros(n0, int), norm)
        full = E0.sum(1) == 1
        cnt = E0[full].sum(0) + 0.5
        self.pi0 = torch.as_tensor(np.r_[cnt / cnt.sum(), 0.0], dtype=self.dt, device=self.device)
        self.typ = torch.as_tensor(TYP, device=self.device)
        self.flat = torch.as_tensor(SRC * NS + DST, device=self.device)
        dtr = self.person_data(Dtr, norm)
        dte = self.person_data(Dte, norm)
        init = np.log(np.array([0.15, 0.10, 0.15, 0.10, 0.03, 0.05]))[TYP]
        best, log = None, []
        for pen in self.pens:
            self.base = torch.tensor(init, dtype=self.dt, device=self.device, requires_grad=True)
            self.beta = torch.zeros(N_TYPE, 5, dtype=self.dt, device=self.device, requires_grad=True)
            opt = torch.optim.LBFGS([self.base, self.beta], lr=1.0, max_iter=300, tolerance_grad=1e-7,
                                    tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                l = self.nll(dtr, pen)
                l.backward()
                return l
            opt.step(closure)
            with torch.no_grad():
                tr_ll = -float(self.nll(dtr, 0.0))
                te_ll = -float(self.nll(dte, 0.0))
            log.append({"pen": pen, "train_ll_per_person": tr_ll, "test_ll_per_person": te_ll})
            print(f"  msm-ct pen {pen}: train ll {tr_ll:.4f}, test ll {te_ll:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if best is None or te_ll > best[0]:
                best = (te_ll, pen, self.base.detach().clone(), self.beta.detach().clone())
        _, pen, self.base, self.beta = best
        q = torch.exp(self.base).cpu().numpy()
        self.info = {"grid": log, "chosen_pen": pen, "n_transitions": int(len(SRC)), "minutes": (time.time() - t0) / 60,
                     "beta_by_type": dict(zip(["mmse_worse", "mmse_better", "cogn_worse", "cogn_better", "ad_onset", "death"],
                                              self.beta.cpu().numpy().round(3).tolist())),
                     "covariates": ["age_z", "male", "educ_z", "e4x1", "e4x2"],
                     "mean_baseline_intensity_by_type": [float(q[TYP == k].mean()) for k in range(N_TYPE)]}
        return self

    # ------------------------------------------------------------------ 预测
    @torch.no_grad()
    def start_alpha(self, start, norm):
        """每个站点：用第 0..vv 次访视滤波出来的状态后验 (n, 19)，和站点访视上的协变量 (n, 5)。"""
        D, ii, vv = start["D"], start["ii"], start["vv"]
        Ds = {k: D[k][ii] for k in ("X", "M", "S", "Y")}
        d = self.person_data(Ds, norm, upto=vv, death=False)
        _, alpha, _ = self.forward_filter(d)
        return alpha, d["z"][np.arange(len(ii)), vv]

    @torch.no_grad()
    def landmark_risks(self, start, norm):
        alpha, z0 = self.start_alpha(start, norm)
        n = len(alpha)
        tgt = {"mmse_lt27": SM_OF >= 1, "mmse_lt24": SM_OF == 2, "cogn_lt0": SC_OF >= 1, "cogn_ltm1": SC_OF == 2, "ad": A_OF == 1}
        Ps = [self.P_year(self._age_shift(z0, k)) for k in range(B.NH)]
        R = np.zeros((n, B.NQ, B.NH))
        for qi, q in enumerate(B.Q):
            p = alpha.clone()
            hit = torch.zeros(n, dtype=self.dt, device=self.device)
            m = None if q == "death" else torch.as_tensor(np.r_[tgt[q], False], device=self.device)
            for k in range(B.NH):
                p = torch.einsum("ns,nst->nt", p, Ps[k])
                if q == "death":
                    R[:, qi, k] = p[:, DEAD].cpu().numpy()
                    continue
                hit = hit + p[:, m].sum(1)
                p = p.masked_fill(m[None], 0.0)
                R[:, qi, k] = hit.cpu().numpy()
        return np.clip(R, 0, 1)

    @torch.no_grad()
    def sim(self, start, n_mc, rng, norm, years=SIM.YEARS):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        n = len(ii)
        N = n * n_mc
        alpha, z0 = self.start_alpha(start, norm)
        a = alpha[:, :18].cpu().numpy()
        a = a / a.sum(1, keepdims=True)
        rep = np.repeat(np.arange(n), n_mc)
        u = rng.random(N)
        s = (np.cumsum(a[rep], 1) < u[:, None]).sum(1).clip(max=17)
        stage = {f: np.full((N, years + 1), -1, np.int64) for f in ("mmse", "cogn")}
        for f in ("mmse", "cogn"):
            stage[f][:, 0] = np.repeat(_stage_measured(D, ii, vv, norm, f), n_mc)
        t_death = np.full(N, np.inf)
        t_ad = np.where(s >= 9, -np.inf, np.inf)
        for k in range(1, years + 1):
            P = self.P_year(self._age_shift(z0, k - 1)).cpu().numpy()                     # (n, 19, 19)
            r = np.where(~np.isfinite(t_death))[0]
            if not len(r):
                break
            row = P[rep[r], s[r]]
            nxt = (np.cumsum(row, 1) < rng.random(len(r))[:, None]).sum(1).clip(max=DEAD)
            dies = nxt == DEAD
            t_death[r[dies]] = k - 1 + rng.random(int(dies.sum()))
            al, ns = r[~dies], nxt[~dies]
            new_ad = (ns >= 9) & (s[al] < 9)
            t_ad[al[new_ad]] = k - 1 + rng.random(int(new_ad.sum()))
            s[al] = ns
            stage["mmse"][al, k] = (ns // 3) % 3
            stage["cogn"][al, k] = ns % 3
        rs = lambda x: x.reshape(n, n_mc, *x.shape[1:])            # noqa: E731
        nvis = np.minimum(np.where(np.isfinite(t_death), np.floor(t_death), years), years).astype(int)
        return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
                "n_visits": rs(nvis)}
