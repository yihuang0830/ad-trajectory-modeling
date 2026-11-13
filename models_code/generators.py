"""generators.py -- 能生成未来轨迹的模型（Delphi 以外）。每个都给两样东西：
  landmark_risks(start)  从每个站点起模拟（或精确算）-> (站点, 6, 10) 的首次发生概率（第一部分、第三部分的打分表）
  first_visit_sim(start) 从第一次访视起的 10 年轨迹（trajectory.json）

  ar-lr / ar-lgbm   逐访视的自回归微观模拟（dynamic microsimulation）：一组"上一次访视 -> 下一次访视"的条件模型
                    （每个连续量一个回归 + 高斯残差、每个缺失 / 状态 / 新出现一个二分类）+ 1 年死亡、AD 风险，
                    输入 = 当前访视 + 手工历史特征 + 静态背景（和直接预测的 l2c 特征同一份）。模型分别是线性 / 逻辑回归和 LightGBM。
  minimalrnn        MinimalRNN（Nguyen et al., NeuroImage 2020，TADPOLE 第 2）：f = σ(U_h h + U_u u)，h = f⊙h + (1−f)⊙u，
                    缺失的输入用模型自己上一步的预测补（论文的 model filling），输出下一次访视 + 1 年死亡、AD 风险；
                    原文只预测诊断和连续量，这里加了死亡头（没有它就没法模拟死亡）。
  markov / msm-cov  离散时间（1 年）多状态 Markov 模型：状态 = (AD 有无, MMSE 分期, 认知分期) 的 18 个活着的组合 + 死亡。
                    markov：每个起始状态一行经验转移概率（加 0.5 平滑），不看协变量；
                    msm-cov：每个起始状态一个多项逻辑回归（年龄、性别、教育、APOE ε4），即转移特异的协变量效应（msm 包的做法）。
                    站点上的风险按转移矩阵连乘精确算（目标状态吸收），不用抽样。
  lmm-joint         两阶段联合模型：MMSE、全局认知各一个线性混合模型（随机截距 + 斜率，固定效应 = 时间、基线年龄、性别、教育、
                    APOE ε4 及其和时间的交互），死亡、AD 的 1 年离散时间风险用"当前真值 + 当前斜率"（Rizopoulos 的
                    value + slope 关联）+ 人口学；模拟时从到起点为止的观测求随机效应的后验、抽一组，往后走。
"""
import time

import numpy as np

import bl_data as B
import simulator as SIM
from bl_data import I_AD, I_EVER, I_MC, I_MS, I_TF, I_XS, N_CONT, N_STATE
from tabular import Prep

J_MMSE, J_COGN = 0, 1                         # CONT 的前两个
SIG = lambda z: 1.0 / (1.0 + np.exp(-z))      # noqa: E731


def _rows_l2c(D):
    """站点表（l2c 特征）+ 下一次访视（间隔正好 1 年）的配对。"""
    lm = B.landmarks(D, "l2c")
    X, M = D["X"], D["M"]
    ii, vv = lm["ii"], lm["vv"]
    V = X.shape[1]
    has_next = (vv + 1 < V)
    nxt = np.where(has_next, vv + 1, vv)
    ok = has_next & M[ii, nxt] & (np.abs(X[ii, nxt, I_TF + 2] - 1.0) < 1e-6)
    return lm, X[ii, nxt], ok


# ----------------------------------------------------------------------------- 自回归微观模拟（ar-lr、ar-lgbm）


class _ZeroReg:
    def predict(self, F):
        return np.zeros(len(F))


class ARModels:
    """52 个条件模型。kind = lr / lgbm。"""

    def __init__(self, kind):
        self.kind = kind

    def _clf(self):
        if self.kind == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.1, max_iter=3000)
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, n_jobs=8, verbose=-1)

    def _reg(self):
        if self.kind == "lr":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, n_jobs=8, verbose=-1)

    def _fit(self, m, F, y, Fe, ye, clf=True):
        if len(np.unique(y)) < 2 and clf:
            return ("const", float(y.mean()))
        if not clf and len(y) == 0:                           # 从没测过的量（只在 NACC 视图上训练时出现）：模拟时永远缺失，值无所谓
            return _ZeroReg()
        if self.kind == "lgbm":
            import lightgbm as lgb
            m.fit(F, y, eval_set=[(Fe, ye)], callbacks=[lgb.early_stopping(30, verbose=False)])
        else:
            m.fit(F, y)
        return m

    @staticmethod
    def _p(m, F):
        if isinstance(m, tuple):
            return np.full(len(F), m[1])
        return m.predict_proba(F)[:, 1]

    def fit(self, Ztr, Zte):
        t0 = time.time()
        lm, nx, ok = _rows_l2c(Ztr)
        lme, nxe, oke = _rows_l2c(Zte)
        self.prep = Prep(lm["F"])
        F, Fe = self.prep(lm["F"]), self.prep(lme["F"])
        self.m = {}
        # 1 年风险：死亡、AD（只在答案已知的站点上）
        for key, qi in (("death", B.I_DEATH), ("ad", B.I_ADQ)):
            y, ye = lm["Y"][:, qi, 0], lme["Y"][:, qi, 0]
            k, ke = y >= 0, ye >= 0
            self.m[key] = self._fit(self._clf(), F[k], y[k], Fe[ke], ye[ke])
        # 下一次访视
        Fn, Fne = F[ok], Fe[oke]
        xn, xne = nx[ok], nxe[oke]
        self.sd = np.zeros(N_CONT)
        for j in range(N_CONT):
            self.m[f"miss_c{j}"] = self._fit(self._clf(), Fn, xn[:, I_MC + j], Fne, xne[:, I_MC + j])
            o, oe = xn[:, I_MC + j] < 0.5, xne[:, I_MC + j] < 0.5
            r = self._fit(self._reg(), Fn[o], xn[o, j], Fne[oe], xne[oe, j], clf=False)
            self.m[f"mu{j}"] = r
            self.sd[j] = float(np.std(xn[o, j] - r.predict(Fn[o]))) if o.any() else 0.0   # 训练段残差（为什么不用 test 段见 rollout_versions.PerTargetNext）
        for j in range(N_STATE):
            self.m[f"miss_s{j}"] = self._fit(self._clf(), Fn, xn[:, I_MS + j], Fne, xne[:, I_MS + j])
            o, oe = xn[:, I_MS + j] < 0.5, xne[:, I_MS + j] < 0.5
            self.m[f"xs{j}"] = self._fit(self._clf(), Fn[o], xn[o, I_XS + j], Fne[oe], xne[oe, I_XS + j])
        xcur = lm["F"][ok][:, :B.N_FEAT]
        xcure = lme["F"][oke][:, :B.N_FEAT]
        for j in range(I_AD - I_EVER):
            o, oe = xcur[:, I_EVER + j] < 0.5, xcure[:, I_EVER + j] < 0.5
            self.m[f"ever{j}"] = self._fit(self._clf(), Fn[o], xn[o, I_EVER + j], Fne[oe], xne[oe, I_EVER + j])
        self.minutes = (time.time() - t0) / 60
        return self

    def predict(self, F):
        """F：未标准化的 l2c 特征行 -> (p_death, p_ad, 下一次访视分布)。"""
        F = self.prep(F)
        nd = {"mu": np.stack([self.m[f"mu{j}"].predict(F) for j in range(N_CONT)], 1),
              "sd": np.broadcast_to(self.sd, (len(F), N_CONT)),
              "miss_c": np.stack([self._p(self.m[f"miss_c{j}"], F) for j in range(N_CONT)], 1),
              "xs": np.stack([self._p(self.m[f"xs{j}"], F) for j in range(N_STATE)], 1),
              "miss_s": np.stack([self._p(self.m[f"miss_s{j}"], F) for j in range(N_STATE)], 1),
              "ever": np.stack([self._p(self.m[f"ever{j}"], F) for j in range(I_AD - I_EVER)], 1)}
        return self._p(self.m["death"], F), self._p(self.m["ad"], F), nd


class ARStepper:
    def __init__(self, models, l2c_state, y0, n_mc, chunk=200_000):
        self.models, self.l0, self.y0, self.n_mc, self.chunk = models, l2c_state, y0, n_mc, chunk

    def init(self, n_mc):
        n = len(self.y0)
        rep = np.repeat(np.arange(n), n_mc)
        return {"l2c": self.l0.take(rep), "year": np.repeat(self.y0, n_mc).astype(float)}

    def step(self, st, x, S, rows):
        L = st["l2c"].take(rows).features(st["year"][rows])
        F = np.concatenate([x[rows], S[rows], L], 1)
        out = [self.models.predict(F[s: s + self.chunk]) for s in range(0, len(F), self.chunk)]
        p_d = np.concatenate([o[0] for o in out])
        p_a = np.concatenate([o[1] for o in out])
        nd = {k: np.concatenate([o[2][k] for o in out]) for k in out[0][2]}
        return p_d, p_a, nd

    def update(self, st, rows, xn, year):
        st["l2c"].update(rows, xn[:, :N_CONT].astype(float), xn[:, I_MC:I_XS] < 0.5, year)
        st["year"][rows] = year


def ar_sim(models, start, n_mc, rng, norm):
    D, ii, vv = start["D"], start["ii"], start["vv"]
    l0 = SIM.l2c_replay(D["X"][ii], D["M"][ii], vv)
    y0 = np.round(D["X"][ii, vv, I_TF + 1] * 10.0)
    stp = ARStepper(models, l0, y0, n_mc)
    return SIM.simulate_visits(stp, D["X"][ii, vv], D["S"][ii], y0, n_mc, rng, norm)


# ----------------------------------------------------------------------------- MinimalRNN


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


# 默认超参 = minimalrnn 打包盒用的那组。论文（Table 2，HORD）搜的是 in_drop、rec_drop 0–0.5，L2 1e-7–1e-5，学习率 1e-5–1e-2，
# 1–3 层，隐层 128–512（tune_minimalrnn.py）；论文的 L2 是 Adam 的 weight decay（opt="adam"），默认这组是 AdamW。
MRNN_HP = {"d": 128, "n_layers": 1, "dropout": 0.2, "in_drop": 0.0, "rec_drop": 0.0, "lr": 1e-3, "wd": 1e-4, "opt": "adamw"}


def minimalrnn_hp(kw):
    """models.py 的 kw -> 超参。kw["hp"] = "tuned"：用 tune_minimalrnn.py 选出来的那组（$ZOO_RUN_DIR/tune_minimalrnn/chosen.json）。"""
    import json
    import os
    hp = kw.get("hp", {})
    if hp == "tuned":
        hp = json.load(open(os.path.join(B.RUN_DIR, "tune_minimalrnn", "chosen.json")))["hp"]
    return {**MRNN_HP, **hp}


def build_minimalrnn(d=128, dropout=0.2, n_layers=1, in_drop=0.0, rec_drop=0.0):
    torch, nn = _torch()
    n_in = N_CONT + N_STATE + (B.N_FEAT - I_EVER) + B.N_S          # 补过的连续量 + 状态 + 其余 6 维 + 静态；掩码不进输入

    class MinimalRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.d, self.n_layers, self.rec_drop = d, n_layers, rec_drop
            self.phi = nn.Sequential(nn.Linear(n_in, d), nn.Tanh())
            self.uh = nn.Linear(d, d)
            self.uu = nn.Linear(d, d, bias=False)
            if n_layers > 1:                        # 叠层：第 k 层的输入是第 k−1 层的 h
                self.deep = nn.ModuleList(nn.ModuleDict({"phi": nn.Sequential(nn.Linear(d, d), nn.Tanh()), "uh": nn.Linear(d, d),
                                                         "uu": nn.Linear(d, d, bias=False)}) for _ in range(n_layers - 1))
            self.in_drop = nn.Dropout(in_drop)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(d, 3 * N_CONT + 2 * N_STATE + 2 + 2)

        def inp(self, x, S, fill_c, fill_s):
            obs_c = x[:, I_MC:I_XS] < 0.5
            obs_s = x[:, I_MS:I_EVER] < 0.5
            c = torch.where(obs_c, x[:, :N_CONT], fill_c)
            s = torch.where(obs_s, x[:, I_XS:I_MS], fill_s)
            return torch.cat([c, s, x[:, I_EVER:], S], -1)

        def cell(self, h, u, mask=None):
            """h：各层的隐状态拼在一起 (n, n_layers·d)。mask：recurrent dropout，乘在进门控的 h 上，整条序列共用一个。"""
            layers = [(self.phi, self.uh, self.uu)] + [(m["phi"], m["uh"], m["uu"]) for m in getattr(self, "deep", [])]
            hs, x = [], self.in_drop(u)
            for k, (phi, uh, uu) in enumerate(layers):
                hk = h[:, k * self.d:(k + 1) * self.d]
                z = phi(x)
                hg = hk if mask is None else hk * mask[:, k * self.d:(k + 1) * self.d]
                f = torch.sigmoid(uh(hg) + uu(z))
                x = f * hk + (1 - f) * z
                hs.append(x)
            return torch.cat(hs, -1) if len(hs) > 1 else x

        def heads(self, h):
            o = self.out(self.drop(h[:, -self.d:]))
            sp = torch.split(o, [N_CONT, N_CONT, N_CONT, N_STATE, N_STATE, 2, 2], -1)
            return dict(zip(("mu", "log_sd", "miss_c", "xs", "miss_s", "ever", "haz"), sp))

        def run(self, X, M, S):
            """整条历史（训练 / 回放）：返回每次访视之后的隐状态和头。缺失输入用上一步预测补。"""
            n, V, _ = X.shape
            h = torch.zeros(n, n_layers * d, device=X.device)
            mask = None
            if self.training and self.rec_drop > 0:
                mask = torch.bernoulli(torch.full_like(h, 1 - self.rec_drop)) / (1 - self.rec_drop)
            fill_c = torch.zeros(n, N_CONT, device=X.device)
            fill_s = torch.full((n, N_STATE), 0.5, device=X.device)
            H, outs = [], []
            for v in range(V):
                h_new = self.cell(h, self.inp(X[:, v], S, fill_c, fill_s), mask)
                h = torch.where(M[:, v:v + 1], h_new, h)
                o = self.heads(h)
                fill_c = torch.where(M[:, v:v + 1], o["mu"], fill_c)
                fill_s = torch.where(M[:, v:v + 1], torch.sigmoid(o["xs"]), fill_s)
                H.append(h)
                outs.append(o)
            return H, {k: torch.stack([o[k] for o in outs], 1) for k in outs[0]}

    return MinimalRNN()


def minimalrnn_loss(out, X, M, Y):
    torch, nn = _torch()
    import torch.nn.functional as F
    gap1 = (torch.abs(X[:, 1:, I_TF + 2] - 1.0) < 1e-6) & M[:, 1:]
    nxt = X[:, 1:]
    o = {k: v[:, :-1] for k, v in out.items()}
    oc = (nxt[..., I_MC:I_XS] < 0.5) & gap1[..., None]
    sd = torch.exp(o["log_sd"]).clamp(min=1e-2)
    l_c = (((o["mu"] - nxt[..., :N_CONT]) / sd) ** 2 / 2 + torch.log(sd))[oc].mean()
    bce = lambda z, y, m: F.binary_cross_entropy_with_logits(z[m], y[m]) if m.any() else z.sum() * 0   # noqa: E731
    g = gap1[..., None].expand(-1, -1, N_CONT)
    l_mc = bce(o["miss_c"], nxt[..., I_MC:I_XS], g)
    gs = gap1[..., None].expand(-1, -1, N_STATE)
    os_ = (nxt[..., I_MS:I_EVER] < 0.5) & gs
    l_ms = bce(o["miss_s"], nxt[..., I_MS:I_EVER], gs)
    l_s = bce(o["xs"], nxt[..., I_XS:I_MS], os_)
    oe = (X[:, :-1, I_EVER:I_AD] < 0.5) & gap1[..., None]
    l_e = bce(o["ever"], nxt[..., I_EVER:I_AD], oe)
    # 1 年风险：每个站点（含最后一次访视）
    y = torch.stack([Y[..., B.I_DEATH * B.NH], Y[..., B.I_ADQ * B.NH]], -1)
    oh = (y >= 0) & M[..., None]
    l_h = bce(out["haz"], y, oh)
    return l_c + l_mc + l_ms + l_s + l_e + 2 * l_h


def fit_minimalrnn(Ztr, Zte, device, seed=0, bs=64, max_epochs=150, patience=15, hp=None):
    torch, nn = _torch()
    hp = {**MRNN_HP, **(hp or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = lambda D, k, dt=torch.float32: torch.as_tensor(D[k], dtype=dt, device=device)   # noqa: E731
    tr = {k: T(Ztr, k) for k in ("X", "S")}
    tr["M"] = T(Ztr, "M", torch.bool)
    tr["Y"] = T(Ztr, "Y")
    te = {k: T(Zte, k) for k in ("X", "S")}
    te["M"] = T(Zte, "M", torch.bool)
    te["Y"] = T(Zte, "Y")
    model = build_minimalrnn(hp["d"], hp["dropout"], hp["n_layers"], hp["in_drop"], hp["rec_drop"]).to(device)
    opt = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[hp["opt"]](model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(len(tr["X"]))
        for s in range(0, len(perm), bs):
            b = torch.as_tensor(perm[s: s + bs], device=device)
            _, out = model.run(tr["X"][b], tr["M"][b], tr["S"][b])
            l = minimalrnn_loss(out, tr["X"][b], tr["M"][b], tr["Y"][b])
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            _, out = model.run(te["X"], te["M"], te["S"])
            vl = float(minimalrnn_loss(out, te["X"], te["M"], te["Y"]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_test_loss": best, "epochs": ep + 1, "hp": hp,
                   "n_params": sum(p.numel() for p in model.parameters())}


class RNNStepper:
    def __init__(self, model, h0, fc0, fs0, device, chunk=100_000):
        self.model, self.h0, self.fc0, self.fs0, self.device, self.chunk = model, h0, fc0, fs0, device, chunk

    def init(self, n_mc):
        rep = np.repeat(np.arange(len(self.h0)), n_mc)
        return {"h": self.h0[rep].copy(), "fc": self.fc0[rep].copy(), "fs": self.fs0[rep].copy()}

    def step(self, st, x, S, rows):
        torch, _ = _torch()
        outs = []
        with torch.no_grad():
            for s in range(0, len(rows), self.chunk):
                r = rows[s: s + self.chunk]
                o = self.model.heads(torch.as_tensor(st["h"][r], device=self.device))
                outs.append({k: v.cpu().numpy() for k, v in o.items()})
        o = {k: np.concatenate([u[k] for u in outs]) for k in outs[0]}
        nd = {"mu": o["mu"], "sd": np.exp(o["log_sd"]).clip(1e-2), "miss_c": SIG(o["miss_c"]), "xs": SIG(o["xs"]),
              "miss_s": SIG(o["miss_s"]), "ever": SIG(o["ever"])}
        return SIG(o["haz"][:, 0]), SIG(o["haz"][:, 1]), nd

    def update(self, st, rows, xn, year):
        torch, _ = _torch()
        with torch.no_grad():
            for s in range(0, len(rows), self.chunk):
                r = rows[s: s + self.chunk]
                t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)   # noqa: E731
                h = t(st["h"][r])
                u = self.model.inp(t(xn[s: s + self.chunk]), t(self.S_of(r)), t(st["fc"][r]), t(st["fs"][r]))
                h = self.model.cell(h, u)
                o = self.model.heads(h)
                st["h"][r] = h.cpu().numpy()
                st["fc"][r] = o["mu"].cpu().numpy()
                st["fs"][r] = torch.sigmoid(o["xs"]).cpu().numpy()

    def S_of(self, r):
        return self.S_all[r]


def minimalrnn_sim(model, start, n_mc, rng, norm, device):
    torch, _ = _torch()
    D, ii, vv = start["D"], start["ii"], start["vv"]
    h0, fc0, fs0 = [], [], []
    with torch.no_grad():
        for s in range(0, len(ii), 2048):
            i, v = ii[s: s + 2048], vv[s: s + 2048]
            V = int(v.max()) + 1
            X = torch.as_tensor(D["X"][i, :V], device=device)
            M = torch.as_tensor(D["M"][i, :V] & (np.arange(V)[None, :] <= v[:, None]), device=device)
            H, out = model.run(X, M, torch.as_tensor(D["S"][i], device=device))
            ar = torch.arange(len(i), device=device)
            vt = torch.as_tensor(v, device=device)
            h0.append(torch.stack(H, 1)[ar, vt].cpu().numpy())
            fc0.append(out["mu"][ar, vt].cpu().numpy())
            fs0.append(torch.sigmoid(out["xs"][ar, vt]).cpu().numpy())
    stp = RNNStepper(model, np.concatenate(h0), np.concatenate(fc0), np.concatenate(fs0), device)
    stp.S_all = np.repeat(D["S"][ii], n_mc, 0)
    y0 = np.round(D["X"][ii, vv, I_TF + 1] * 10.0)
    return SIM.simulate_visits(stp, D["X"][ii, vv], D["S"][ii], y0, n_mc, rng, norm)


# ----------------------------------------------------------------------------- 多状态 Markov

N_ALIVE, DEAD = 18, 18
SURVIVED = 19                       # MSM.pairs：活过这一年，但下一次访视没测全（只用来估死亡）


def _stage_locf(D, ii, vv, norm, fam):
    """站点上的当前分期：这次测了用这次，没测用最近一次（手工历史特征里的"最近值"），从来没测过 = -1。"""
    j = J_MMSE if fam == "mmse" else J_COGN
    L = D["L"][ii, vv]
    has = L[:, j * 8 + 1] > 0.5
    raw = L[:, j * 8 + 0] * norm["sd"][j] + norm["mu"][j]
    return np.where(has, SIM.stage_of(raw, fam), -1)


def _stage_measured(D, ii, vv, norm, fam):
    j = J_MMSE if fam == "mmse" else J_COGN
    x = D["X"][ii, vv]
    raw = x[:, j] * norm["sd"][j] + norm["mu"][j]
    return np.where(x[:, I_MC + j] < 0.5, SIM.stage_of(raw, fam), -1)


def _state(a, sm, sc):
    return a * 9 + sm * 3 + sc


def _covars(D, ii, vv, dyear=0.0):
    x, s = D["X"][ii, vv], D["S"][ii]
    age = x[:, I_TF] + dyear * SIM.AGE_STEP_Z
    return np.stack([age, s[:, 0], s[:, 1], s[:, 9], s[:, 10]], 1)      # 年龄 z、男、教育 z、ε4 一个、两个


class MSM:
    def __init__(self, cov):
        self.cov = cov

    def pairs(self, D, norm):
        lm = B.landmarks(D, "cur")
        ii, vv = lm["ii"], lm["vv"]
        V = D["X"].shape[1]
        sm, sc = _stage_locf(D, ii, vv, norm, "mmse"), _stage_locf(D, ii, vv, norm, "cogn")
        a = (D["X"][ii, vv, I_AD] > 0.5).astype(int)
        cur = np.where((sm >= 0) & (sc >= 0), _state(a, np.maximum(sm, 0), np.maximum(sc, 0)), -1)
        nv = np.minimum(vv + 1, V - 1)
        ok_next = (vv + 1 < V) & D["M"][ii, nv] & (np.abs(D["X"][ii, nv, I_TF + 2] - 1.0) < 1e-6)
        sm1, sc1 = _stage_measured(D, ii, nv, norm, "mmse"), _stage_measured(D, ii, nv, norm, "cogn")
        a1 = np.maximum(a, (D["X"][ii, nv, I_AD] > 0.5).astype(int))
        alive_next = np.where(ok_next & (sm1 >= 0) & (sc1 >= 0), _state(a1, np.maximum(sm1, 0), np.maximum(sc1, 0)), -1)
        died = lm["Y"][:, B.I_DEATH, 0]
        # 一年里死没死已知的都留下（活过这一年、但下一次访视没测全的 = SURVIVED：只进死亡那一部分）。
        # 2026-10-29 改：原来只留"死了"和"活着且下一次访视测全了"，活着的人被选掉一部分，死亡转移偏多（死亡 O/E 约 0.9）
        nxt = np.where(died == 1, DEAD, np.where(died == 0, np.where(alive_next >= 0, alive_next, SURVIVED), -1))
        keep = (cur >= 0) & (nxt != -1)
        return cur[keep], nxt[keep], _covars(D, ii[keep], vv[keep])

    def _part(self, X, y, n_cls):
        """一个起点的一部分（死 / 活，或活着时下一个状态）：协变量模型且数据够用多项逻辑回归，否则按计数 + 0.5。"""
        from sklearn.linear_model import LogisticRegression
        if self.cov and len(y) >= 50 and len(np.unique(y)) >= 2:
            return ("lr", LogisticRegression(C=1.0, max_iter=3000).fit(X, y))
        cnt = np.bincount(y, minlength=n_cls).astype(float) + 0.5
        return ("emp", cnt / cnt.sum())

    @staticmethod
    def _prob(part, Cz, n_cls):
        kind, m = part
        if kind == "emp":
            return np.broadcast_to(m, (len(Cz), n_cls))
        P = np.zeros((len(Cz), n_cls))
        P[:, m.classes_] = m.predict_proba(Cz)
        return P

    def fit(self, Dtr, norm):
        cur, nxt, C = self.pairs(Dtr, norm)
        self.mu, self.sd = C.mean(0), C.std(0) + 1e-8
        Cz = (C - self.mu) / self.sd
        self.rows = {}
        self.n_pairs = int(len(cur))
        # 一年转移 = P(死) × 死亡 + (1 − P(死)) × P(下一个状态 | 活着)；两部分各用自己能用的那批人
        for o in range(N_ALIVE):
            m = cur == o
            al = m & (nxt >= 0) & (nxt < N_ALIVE)
            self.rows[o] = (self._part(Cz[m], (nxt[m] == DEAD).astype(int), 2), self._part(Cz[al], nxt[al], N_ALIVE))
        return self

    def matrices(self, C):
        """C (n, 5) 协变量 -> (n, 18, 19) 的一年转移概率。"""
        n = len(C)
        P = np.zeros((n, N_ALIVE, N_ALIVE + 1))
        Cz = (C - self.mu) / self.sd
        for o, (pd_, pa_) in self.rows.items():
            pd = self._prob(pd_, Cz, 2)[:, 1]
            pa = np.array(self._prob(pa_, Cz, N_ALIVE))
            if o >= 9:                                         # 有 AD 的状态回不到没有 AD
                pa[:, :9] = 0.0
            pa /= pa.sum(1, keepdims=True)
            P[:, o, :N_ALIVE] = (1.0 - pd)[:, None] * pa
            P[:, o, DEAD] = pd
        return P

    def start_state(self, D, ii, vv, norm):
        sm, sc = _stage_locf(D, ii, vv, norm, "mmse"), _stage_locf(D, ii, vv, norm, "cogn")
        a = (D["X"][ii, vv, I_AD] > 0.5).astype(int)
        return _state(a, np.maximum(sm, 0), np.maximum(sc, 0))   # 从来没测过：按最好的分期起步（val 里极少）

    def landmark_risks(self, start, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        n = len(ii)
        s0 = self.start_state(D, ii, vv, norm)
        st = np.arange(N_ALIVE)
        a_of, sm_of, sc_of = st // 9, (st // 3) % 3, st % 3
        tgt = {"mmse_lt27": sm_of >= 1, "mmse_lt24": sm_of == 2, "cogn_lt0": sc_of >= 1, "cogn_ltm1": sc_of == 2,
               "ad": a_of == 1}
        Ps = [self.matrices(_covars(D, ii, vv, k)) for k in range(B.NH)]
        R = np.zeros((n, B.NQ, B.NH))
        for qi, q in enumerate(B.Q):
            p = np.zeros((n, N_ALIVE + 1))
            p[np.arange(n), s0] = 1.0
            hit = np.zeros(n)
            for k in range(B.NH):
                p = np.einsum("ns,nst->nt", p[:, :N_ALIVE], Ps[k]) + np.concatenate(
                    [np.zeros((n, N_ALIVE)), p[:, N_ALIVE:]], 1)
                if q == "death":
                    R[:, qi, k] = p[:, DEAD]
                    continue
                m = tgt[q]
                hit += p[:, :N_ALIVE][:, m].sum(1)
                p[:, :N_ALIVE][:, m] = 0.0
                R[:, qi, k] = hit
        return R

    def sim(self, start, n_mc, rng, norm, years=SIM.YEARS):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        n = len(ii)
        N = n * n_mc
        s = np.repeat(self.start_state(D, ii, vv, norm), n_mc)
        stage = {f: np.full((N, years + 1), -1, np.int64) for f in ("mmse", "cogn")}
        for f in ("mmse", "cogn"):
            stage[f][:, 0] = np.repeat(_stage_measured(D, ii, vv, norm, f), n_mc)
        t_death = np.full(N, np.inf)
        t_ad = np.where(s >= 9, -np.inf, np.inf)
        rep = np.repeat(np.arange(n), n_mc)
        for k in range(1, years + 1):
            alive = np.isfinite(t_death) == False      # noqa: E712
            r = np.where(alive)[0]
            if not len(r):
                break
            P = self.matrices(_covars(D, ii[rep[r]], vv[rep[r]], k - 1))[np.arange(len(r)), s[r]]   # (len, 19)
            u = rng.random(len(r))
            nxt = (P.cumsum(1) < u[:, None]).sum(1).clip(max=N_ALIVE)
            dies = nxt == DEAD
            t_death[r[dies]] = k - 1 + rng.random(dies.sum())
            al = r[~dies]
            ns = nxt[~dies]
            new_ad = (ns >= 9) & (s[al] < 9)
            t_ad[al[new_ad]] = k - 1 + rng.random(new_ad.sum())
            s[al] = ns
            stage["mmse"][al, k] = (ns // 3) % 3
            stage["cogn"][al, k] = ns % 3
        rs = lambda a: a.reshape(n, n_mc, *a.shape[1:])            # noqa: E731
        nvis = np.minimum(np.where(np.isfinite(t_death), np.floor(t_death), years), years).astype(int)   # 每年一次"访视"
        return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
                "n_visits": rs(nvis)}


# ----------------------------------------------------------------------------- 两阶段联合模型


class LMMJoint:
    """MMSE、全局认知各一个线性混合模型 + 死亡、AD 的 1 年风险（当前真值 + 斜率 + 人口学）。"""

    FIX = ("1", "t", "age0", "male", "educ", "e4", "t_age0", "t_e4")

    def _design(self, t, age0, s):
        e4 = s[:, 9] + 2 * s[:, 10]
        return np.stack([np.ones_like(t), t, age0, s[:, 0], s[:, 1], e4, t * age0, t * e4], -1)

    def _long(self, D, norm):
        """测量表：每个测量一行（人下标、随访年数、值）。"""
        X, M = D["X"], D["M"]
        out = {}
        for fam, j in (("mmse", J_MMSE), ("cogn", J_COGN)):
            ii, vv = np.where(M & (X[..., I_MC + j] < 0.5))
            out[fam] = (ii, X[ii, vv, I_TF + 1] * 10.0, X[ii, vv, j] * norm["sd"][j] + norm["mu"][j])
        return out

    def fit(self, Dtr, Dte, norm):
        import statsmodels.api as sm
        from sklearn.linear_model import LogisticRegression
        t0 = time.time()
        self.lmm = {}
        age0 = Dtr["X"][:, 0, I_TF]
        for fam, (ii, t, y) in self._long(Dtr, norm).items():
            Xf = self._design(t, age0[ii], Dtr["S"][ii])
            Zr = np.stack([np.ones_like(t), t], 1)
            res = sm.MixedLM(y, Xf, groups=ii, exog_re=Zr).fit(reml=True, method=["lbfgs"])
            self.lmm[fam] = {"beta": np.asarray(res.fe_params), "G": np.asarray(res.cov_re), "s2": float(res.scale)}
        # 风险模型：每个训练站点上，用到这个站点为止的测量求随机效应的后验均值 -> 当前真值和斜率
        lm = B.landmarks(Dtr, "cur")
        F = self.hazard_features(Dtr, lm["ii"], lm["vv"], norm)
        lme = B.landmarks(Dte, "cur")
        self.haz = {}
        self.hmu, self.hsd = F.mean(0), F.std(0) + 1e-8
        for key, qi in (("death", B.I_DEATH), ("ad", B.I_ADQ)):
            y = lm["Y"][:, qi, 0]
            k = y >= 0
            m = LogisticRegression(C=1.0, max_iter=3000).fit((F[k] - self.hmu) / self.hsd, y[k])
            self.haz[key] = m
        self.minutes = (time.time() - t0) / 60
        self.n_test_landmarks = len(lme["ii"])
        return self

    def posterior(self, D, ii, vv, norm):
        """每个站点：用第 0..vv 次访视的测量求随机效应后验 N(m, V)（固定 G、σ²）。返回 {fam: (m (n,2), V (n,2,2))}。"""
        out = {}
        age0 = D["X"][ii, 0, I_TF]
        for fam, j in (("mmse", J_MMSE), ("cogn", J_COGN)):
            p = self.lmm[fam]
            Gi = np.linalg.inv(p["G"])
            ZtZ = np.zeros((len(ii), 2, 2))
            Ztr = np.zeros((len(ii), 2))
            for v in range(int(vv.max()) + 1):
                x = D["X"][ii, v]
                ok = (v <= vv) & D["M"][ii, v] & (x[:, I_MC + j] < 0.5)
                t = x[:, I_TF + 1] * 10.0
                y = x[:, j] * norm["sd"][j] + norm["mu"][j]
                r = y - self._design(t, age0, D["S"][ii]) @ p["beta"]
                z = np.stack([np.ones_like(t), t], 1) * ok[:, None]
                ZtZ += z[:, :, None] * z[:, None, :] / p["s2"]
                Ztr += z * (r * ok)[:, None] / p["s2"]
            Vp = np.linalg.inv(Gi[None] + ZtZ)
            out[fam] = (np.einsum("nij,nj->ni", Vp, Ztr), Vp)
        return out

    def true_value(self, fam, b, t, age0, S):
        p = self.lmm[fam]
        Xf = self._design(t, age0, S)
        val = Xf @ p["beta"] + b[:, 0] + b[:, 1] * t
        slope = p["beta"][1] + p["beta"][6] * age0 + p["beta"][7] * (S[:, 9] + 2 * S[:, 10]) + b[:, 1]
        return val, slope

    def _hf(self, vals, age_z, S):
        return np.stack([vals["mmse"][0], vals["mmse"][1], vals["cogn"][0], vals["cogn"][1], age_z,
                         S[:, 0], S[:, 1], S[:, 9], S[:, 10]], 1)

    def hazard_features(self, D, ii, vv, norm):
        post = self.posterior(D, ii, vv, norm)
        t = D["X"][ii, vv, I_TF + 1] * 10.0
        age0 = D["X"][ii, 0, I_TF]
        vals = {fam: self.true_value(fam, post[fam][0], t, age0, D["S"][ii]) for fam in post}
        return self._hf(vals, D["X"][ii, vv, I_TF], D["S"][ii])

    def sim(self, start, n_mc, rng, norm, years=SIM.YEARS):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        n = len(ii)
        N = n * n_mc
        rep = np.repeat(np.arange(n), n_mc)
        post = self.posterior(D, ii, vv, norm)
        b = {}
        for fam, (m, V) in post.items():
            Lc = np.linalg.cholesky(V + 1e-9 * np.eye(2))
            b[fam] = m[rep] + np.einsum("nij,nj->ni", Lc[rep], rng.standard_normal((N, 2)))
        t0 = (D["X"][ii, vv, I_TF + 1] * 10.0)[rep]
        age0 = D["X"][ii, 0, I_TF][rep]
        agez0 = D["X"][ii, vv, I_TF][rep]
        S = D["S"][ii][rep]
        stage = {f: np.full((N, years + 1), -1, np.int64) for f in ("mmse", "cogn")}
        for f in ("mmse", "cogn"):
            stage[f][:, 0] = np.repeat(_stage_measured(D, ii, vv, norm, f), n_mc)
        t_death = np.full(N, np.inf)
        t_ad = np.where(np.repeat(D["X"][ii, vv, I_AD] > 0.5, n_mc), -np.inf, np.inf)
        for k in range(1, years + 1):
            r = np.where(~np.isfinite(t_death))[0]
            if not len(r):
                break
            tk = t0[r] + (k - 1)
            vals = {f: self.true_value(f, b[f][r], tk, age0[r], S[r]) for f in b}
            Fh = (self._hf(vals, agez0[r] + (k - 1) * SIM.AGE_STEP_Z, S[r]) - self.hmu) / self.hsd
            p_d = self.haz["death"].predict_proba(Fh)[:, 1]
            p_a = self.haz["ad"].predict_proba(Fh)[:, 1]
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
            t1 = t0[al] + k
            for f in ("mmse", "cogn"):
                v, _ = self.true_value(f, b[f][al], t1, age0[al], S[al])
                y = v + np.sqrt(self.lmm[f]["s2"]) * rng.standard_normal(len(al))
                if f == "mmse":
                    y = np.clip(np.round(y), 0, 30)
                stage[f][al, k] = SIM.stage_of(y, f)
        rs = lambda a: a.reshape(n, n_mc, *a.shape[1:])            # noqa: E731
        nvis = np.minimum(np.where(np.isfinite(t_death), np.floor(t_death), years), years).astype(int)
        return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
                "n_visits": rs(nvis)}
