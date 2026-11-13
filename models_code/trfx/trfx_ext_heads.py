"""trfx_ext_heads.py -- TRFX extension: output distributions (heads), next-visit likelihood, simulator sampling.

Config name prefix "h-". Everything is opt-in by config; trfx_models.DEFAULT behaviour is unchanged (new keys: sim.temp = 1.0,
read only by the simulators below; model.skip_hidden, read only by ar_next_mlpskip).

  LOSS_TERMS  next_nll_dcg     next_nll, but observed MMSE gets the likelihood of exactly what the simulator draws
                               (raw = clip(round(z*sd+mu), 0, 30), meta norm of cts_estmmse30): P(k) = Gaussian mass on
                               [k-.5, k+.5] for 0<k<30, P(30) = mass above 29.5, P(0) = mass below 0.5 (discretised,
                               floor/ceiling-censored Gaussian). Sampling unchanged (shared simulator).
  HEADS       ar_next_cat31    ar_next + 31-way MMSE logits "extra_mmse_logits" (passed to the simulator); the Gaussian
                               MMSE mu/log_sd outputs are overwritten with the categorical's mean / log sd in z units
                               (only so that test_loss_std and other simulators see a moment-matched Gaussian)
  LOSS_TERMS  next_nll_cat31   next_nll with the MMSE Gaussian term replaced by the 31-way cross-entropy
  SIMULATORS  cat31            simulator.simulate_visits with MMSE drawn from the categorical by inverse CDF at
                               Phi(the MMSE normal draw) => same random stream as the shared simulator
  HEADS       ar_next_resid    mu = last observed value of the variable (causal, from X/M; training mean if never
                               observed) + predicted change (GraphCast / Pangu residual); sd and the rest as ar_next
  SIMULATORS  temp             Gaussian noise of the 15 continuous variables scaled by sim.temp at sampling only
  SIMULATORS  lhs              stratified (Latin-hypercube) uniforms across the n_mc futures of each start point for
                               every draw (death / AD and their in-year times, missingness, states, ever; normals =
                               ndtri of stratified uniforms). Every marginal stays U(0, 1), so risks are still the share
                               of n_mc simulated futures; only the Monte-Carlo noise of the shares goes down.
  HEADS       ar_next_mlpskip  risk logits = 2-layer MLP of [encoder state, raw current visit vector] (Dynamic-DeepHit
                               style skip); next-visit outputs as ar_next
  HEADS       ar_next_aux      ar_next + TRAIN-ONLY auxiliary outputs: aux_risk (death / AD within 2 and 3 y) and aux_mu2
                               (+2-year means of the 15 continuous). No "extra_" prefix => never passed to the simulator,
                               never read out.
  LOSS_TERMS  aux_horizons     BCE of aux_risk + unit-variance Gaussian NLL of aux_mu2 (use with a small weight, e.g. 0.3)
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr, ndtri

import trfx_models as TM
import bl_data as B
import rollout_versions as RV
import simulator as SIM
from bl_data import I_AD, I_EVER, I_MC, I_MS, I_TF, I_XS, N_CONT, N_CUM, N_STATE

TM.add_defaults(("model",), {"skip_hidden": 64})     # ar_next_mlpskip: hidden width of the risk MLP
TM.add_defaults(("sim",), {"temp": 1.0})             # simulators temp / lhs / cat31: scale of the continuous noise

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
N_MMSE = 31                                          # MMSE 0..30
_NORM = {}


def mmse_norm():
    """(column of MMSE among the 15 continuous, meta mu, meta sd) = what SIM.norm_info(meta) gives the simulator."""
    if not _NORM:
        with open(os.path.join(B.RUN_DIR, "newmodel_data_trans_meta.json")) as f:
            nm = SIM.norm_info(json.load(f))
        j = nm["names"].index("cts_estmmse30")
        _NORM.update(j=j, mu=float(nm["mu"][j]), sd=float(nm["sd"][j]))
    return _NORM["j"], _NORM["mu"], _NORM["sd"]


# ============================================================================ next-visit likelihoods

def next_nll_mmse(o, nxt, cur, valid, mmse_nll):
    """= rollout_versions.next_nll, except that observed MMSE elements get mmse_nll(o, nxt, sel) (one value per selected
    element, on the scale of the Gaussian term) instead of the Gaussian term; the mean still runs over all observed
    continuous elements, and the four BCE means are unchanged."""
    j = mmse_norm()[0]
    bce = lambda z, y, m: F.binary_cross_entropy_with_logits(z[m], y[m]) if m.any() else z.sum() * 0   # noqa: E731
    v = valid[..., None]
    oc = (nxt[..., I_MC:I_XS] < 0.5) & v
    if oc.any():
        sd = torch.exp(o["log_sd"]).clamp(min=1e-2)
        g = ((o["mu"] - nxt[..., :N_CONT]) / sd) ** 2 / 2 + torch.log(sd)
        sel = oc[..., j]
        oth = oc.clone()
        oth[..., j] = False
        tot = g[oth].sum()
        if sel.any():
            tot = tot + mmse_nll(o, nxt, sel).sum()
        l = tot / oc.sum()
    else:
        l = o["mu"].sum() * 0
    l = l + bce(o["miss_c"], nxt[..., I_MC:I_XS], v.expand_as(o["miss_c"]))
    l = l + bce(o["miss_s"], nxt[..., I_MS:I_EVER], v.expand_as(o["miss_s"]))
    l = l + bce(o["xs"], nxt[..., I_XS:I_MS], (nxt[..., I_MS:I_EVER] < 0.5) & v)
    l = l + bce(o["ever"], nxt[..., I_EVER:I_AD], (cur[..., I_EVER:I_AD] < 0.5) & v)
    return l


def _log1mexp(x):
    """log(1 - exp(x)) for x < 0, accurate on both sides of -log 2."""
    return torch.where(x > -math.log(2.0), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def log_mass(lo, hi):
    """log(Phi(hi) - Phi(lo)) for hi > lo; an interval right of 0 is reflected so both CDFs are lower-tail values."""
    flip = lo > 0
    a = torch.where(flip, -hi, lo)
    b = torch.where(flip, -lo, hi)
    la, lb = torch.special.log_ndtr(a), torch.special.log_ndtr(b)
    return lb + _log1mexp(la - lb)


def dcg_logp(mu_r, s_r, k):
    """log P(clip(round(N(mu_r, s_r^2)), 0, 30) = k) in raw MMSE points (float64 tensors, k integer-valued 0..30)."""
    lo, hi = (k - 0.5 - mu_r) / s_r, (k + 0.5 - mu_r) / s_r
    return torch.where(k >= 29.5, torch.special.log_ndtr(-lo),          # P(raw >= 29.5)
                       torch.where(k <= 0.5, torch.special.log_ndtr(hi),  # P(raw < 0.5)
                                   log_mass(lo, hi)))


def mmse_dcg_nll(o, nxt, sel):
    """-log P(k) of the simulator's MMSE draw (sd floor 1e-2 in z units, as in rollout_versions.nd_from_logits), shifted by
    -log(sd_j) - log sqrt(2 pi) so that it is on the scale of the z-space Gaussian term (same gradients)."""
    j, mu_j, sd_j = mmse_norm()
    mu_r = o["mu"][..., j][sel].double() * sd_j + mu_j
    s_r = torch.exp(o["log_sd"][..., j][sel].double()).clamp(min=1e-2) * sd_j
    k = torch.round(nxt[..., j][sel].double() * sd_j + mu_j).clamp(0.0, 30.0)
    return (-dcg_logp(mu_r, s_r, k) - math.log(sd_j) - LOG_SQRT_2PI).float()


def mmse_cat_nll(o, nxt, sel):
    """31-way cross-entropy of round(raw MMSE) (same scale shift as mmse_dcg_nll)."""
    j, mu_j, sd_j = mmse_norm()
    k = torch.round(nxt[..., j][sel] * sd_j + mu_j).clamp(0, 30).long()
    return F.cross_entropy(o["extra_mmse_logits"][sel], k, reduction="none") - math.log(sd_j) - LOG_SQRT_2PI


def _gap1_pairs(b, out, keys):
    X, M = b["X"], b["M"]
    gap1 = (abs(X[:, 1:, I_TF + 2] - 1.0) < 1e-6) & M[:, 1:]
    return {k: out[k][:, :-1] for k in keys}, X[:, 1:], X[:, :-1], gap1


@TM.register(TM.LOSS_TERMS, "next_nll_dcg")
def loss_next_nll_dcg(b, out, ctx):
    """next_nll on 1-year pairs with the MMSE term = discretised, floor/ceiling-censored Gaussian likelihood."""
    o, nxt, cur, ok = _gap1_pairs(b, out, RV.NEXT_KEYS)
    return next_nll_mmse(o, nxt, cur, ok, mmse_dcg_nll)


@TM.register(TM.LOSS_TERMS, "next_nll_cat31")
def loss_next_nll_cat31(b, out, ctx):
    """next_nll on 1-year pairs with the MMSE term = 31-way cross-entropy (needs head ar_next_cat31)."""
    o, nxt, cur, ok = _gap1_pairs(b, out, tuple(RV.NEXT_KEYS) + ("extra_mmse_logits",))
    return next_nll_mmse(o, nxt, cur, ok, mmse_cat_nll)


# ============================================================================ heads

@TM.register(TM.HEADS, "ar_next_cat31")
class ARNextCat31(TM.ARNextHead):
    """ar_next (same modules, same init order) + Linear(d, 31) MMSE logits."""

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        self.mmse = nn.Linear(cfg["model"]["d"], N_MMSE)
        self.j, mu_j, sd_j = mmse_norm()
        self.register_buffer("zk", torch.tensor((np.arange(N_MMSE) - mu_j) / sd_j, dtype=torch.float32), persistent=False)

    def forward(self, h, b):
        o = super().forward(h, b)
        lg = self.mmse(h)
        p = torch.softmax(lg, -1)
        m = (p * self.zk).sum(-1, keepdim=True)
        ls = 0.5 * torch.log((p * (self.zk - m) ** 2).sum(-1, keepdim=True).clamp(min=1e-6))
        j = self.j
        o["mu"] = torch.cat([o["mu"][..., :j], m, o["mu"][..., j + 1:]], -1)
        o["log_sd"] = torch.cat([o["log_sd"][..., :j], ls, o["log_sd"][..., j + 1:]], -1)
        o["extra_mmse_logits"] = lg
        return o


def last_observed(X, M, fill):
    """(n, V, 15): value of each continuous variable at its latest observed visit <= v (causal); fill if none yet."""
    V = X.shape[1]
    obs = (X[..., I_MC:I_XS] < 0.5) & M[..., None]
    idx = torch.arange(V, device=X.device).view(1, V, 1).expand_as(obs)
    last = torch.cummax(torch.where(obs, idx, torch.full_like(idx, -1)), 1).values
    xl = torch.gather(X[..., :N_CONT], 1, last.clamp(min=0))
    return torch.where(last >= 0, xl, fill.to(xl.dtype))


@TM.register(TM.HEADS, "ar_next_resid")
class ARNextResid(TM.ARNextHead):
    """ar_next (same modules, same init) with mu = last observed value (or the training mean) + predicted change."""

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        fill = np.asarray(ctx["x_mean"], np.float32)[:N_CONT]
        self.register_buffer("fill", torch.as_tensor(fill.copy()), persistent=False)

    def forward(self, h, b):
        o = super().forward(h, b)
        o["mu"] = last_observed(b["X"], b["M"], self.fill) + o["mu"]
        return o


@TM.register(TM.HEADS, "ar_next_mlpskip")
class ARNextMLPSkip(nn.Module):
    """risk logits = Linear(Dropout(ReLU(Linear([h, x_v])))) (Dynamic-DeepHit: the raw current measurement skips the
    encoder into the event sub-network); next-visit outputs = Linear(d, 65) as ar_next."""

    def __init__(self, cfg, ctx):
        super().__init__()
        m = cfg["model"]
        self.risk = nn.Sequential(nn.Linear(m["d"] + ctx["n_feat"], m["skip_hidden"]), nn.ReLU(),
                                  nn.Dropout(m["dropout"]), nn.Linear(m["skip_hidden"], ctx["n_risk"]))
        self.next = nn.Linear(m["d"], ctx["n_next"])

    def forward(self, h, b):
        return {"risk": self.risk(torch.cat([h, b["X"]], -1)), **RV.split_next(self.next(h))}


AUX_COLS = [B.I_DEATH * B.NH + 1, B.I_DEATH * B.NH + 2, B.I_ADQ * B.NH + 1, B.I_ADQ * B.NH + 2]   # 2y, 3y


@TM.register(TM.HEADS, "ar_next_aux")
class ARNextAux(TM.ARNextHead):
    """ar_next + train-only auxiliary heads on the encoder state (keys without "extra_": never reach the simulator)."""

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        d = cfg["model"]["d"]
        self.aux_risk = nn.Linear(d, len(AUX_COLS))
        self.aux_mu2 = nn.Linear(d, N_CONT)

    def forward(self, h, b):
        o = super().forward(h, b)
        o["aux_risk"] = self.aux_risk(h)
        o["aux_mu2"] = self.aux_mu2(h)
        return o


@TM.register(TM.LOSS_TERMS, "aux_horizons")
def loss_aux_horizons(b, out, ctx):
    """BCE of death / AD within 2 and 3 y (labels Y, known cells at observed visits) + 0.5 * squared error of the
    continuous values at the visit exactly 2 years later (observed elements only). Train-time only."""
    X, M = b["X"], b["M"]
    y = b["Y"][..., AUX_COLS]
    ok = (y >= 0) & M[..., None]
    l = F.binary_cross_entropy_with_logits(out["aux_risk"][ok], y[ok]) if ok.any() else out["aux_risk"].sum() * 0
    t = torch.round(X[..., I_TF + 1] * 10.0)
    pair = ((t[:, None, :] - t[:, :, None]) == 2) & M[:, :, None] & M[:, None, :]    # [v, u]: u is 2 years after v
    has = pair.any(-1)
    idx = pair.float().argmax(-1)[..., None].expand(-1, -1, N_CONT)
    x2 = torch.gather(X[..., :N_CONT], 1, idx)
    o2 = (torch.gather(X[..., I_MC:I_XS], 1, idx) < 0.5) & has[..., None]
    if o2.any():
        l = l + ((out["aux_mu2"] - x2) ** 2 / 2)[o2].mean()
    return l


# ============================================================================ simulators

def strat_uniform(rng, g, c=None):
    """Stratified uniforms. Rows are grouped by g (non-decreasing; here the start point of each trajectory). Within a
    group of n_g rows each column gets exactly one value in every stratum [i/n_g, (i+1)/n_g) (independent random
    permutation per column + uniform jitter), so every value is marginally U(0, 1). Shape (L,) or (L, c)."""
    L = len(g)
    C = 1 if c is None else int(c)
    if L == 0:
        return np.zeros((0,) if c is None else (0, C))
    new = np.ones(L, bool)
    new[1:] = g[1:] != g[:-1]
    first = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    within = np.arange(L) - first[gid]
    cnt = np.diff(np.append(first, L))[gid]
    order = np.argsort(gid[:, None] + 0.5 * rng.random((L, C)), axis=0)      # sort within group by a random key
    rank = np.empty((L, C), np.int64)
    np.put_along_axis(rank, order, np.broadcast_to(within[:, None], (L, C)), axis=0)
    u = (rank + rng.random((L, C))) / cnt[:, None]
    return u[:, 0] if c is None else u


def cat_sample(logits, eps):
    """Inverse-CDF draw from softmax(logits) (n, 31) at u = Phi(eps): returns k in 0..30 (float)."""
    lg = logits.astype(np.float64)
    p = np.exp(lg - lg.max(1, keepdims=True))
    cdf = np.cumsum(p / p.sum(1, keepdims=True), 1)
    return np.minimum((cdf < ndtr(eps)[:, None]).sum(1), N_MMSE - 1).astype(np.float64)


def simulate_visits_x(stepper, x0, s0, y0, n_mc, rng, norm, years=SIM.YEARS, temp=1.0, mmse_cat=False, lhs=False):
    """Copy of simulator.simulate_visits (same draws in the same order: with temp = 1, mmse_cat = lhs = False it is
    identical, checked by wip_heads_check.py) with three switches: temp scales the Gaussian noise of the continuous
    variables; mmse_cat draws MMSE from nd["extra_mmse_logits"] at Phi(its normal draw); lhs replaces every uniform /
    normal draw by stratified ones across the n_mc futures of each start point."""
    names = norm["names"]
    mu, sd = norm["mu"], norm["sd"]
    j_mmse, j_cogn = names.index("cts_estmmse30"), names.index("cogn_global")
    n = len(x0)
    N = n * n_mc
    x = np.repeat(x0, n_mc, 0).astype(np.float32)
    S = np.repeat(s0, n_mc, 0).astype(np.float32)
    yy = np.repeat(y0, n_mc).astype(float)
    st = stepper.init(n_mc)
    cum_known = x[:, I_XS:I_XS + N_CUM].copy() * (x[:, I_MS:I_MS + N_CUM] < 0.5)
    alive = np.ones(N, bool)
    t_death = np.full(N, np.inf)
    t_ad = np.where(x[:, I_AD] > 0.5, -np.inf, np.inf)
    stage = {f: np.full((N, years + 1), -1, np.int64) for f in ("mmse", "cogn")}

    if lhs:
        def unif(rows, c=None):
            return strat_uniform(rng, rows // n_mc, c)

        def normal(rows, c):
            return ndtri(np.clip(strat_uniform(rng, rows // n_mc, c), 1e-15, 1.0 - 1e-15))
    else:
        def unif(rows, c=None):
            return rng.random(len(rows)) if c is None else rng.random((len(rows), c))

        def normal(rows, c):
            return rng.standard_normal((len(rows), c))

    def record(k, rows, xv):
        for f, j in (("mmse", j_mmse), ("cogn", j_cogn)):
            ok = xv[:, I_MC + j] < 0.5
            raw = xv[:, j] * sd[j] + mu[j]
            stage[f][rows[ok], k] = SIM.stage_of(raw[ok], f)

    record(0, np.arange(N), x)
    n_visits = np.zeros(N, np.int64)
    for k in range(1, years + 1):
        rows = np.where(alive)[0]
        if not len(rows):
            break
        p_d, p_a, nd = stepper.step(st, x, S, rows)
        if mmse_cat and "extra_mmse_logits" not in nd:
            raise KeyError("simulator cat31 needs a head that outputs extra_mmse_logits (ar_next_cat31)")
        dies = unif(rows) < p_d
        td = np.where(dies, k - 1 + unif(rows), np.inf)
        can_ad = t_ad[rows] == np.inf
        gets = can_ad & (unif(rows) < p_a)
        ta = np.where(gets, k - 1 + unif(rows), np.inf)
        gets &= ta < td
        t_ad[rows[gets]] = ta[gets]
        t_death[rows] = td
        alive[rows[dies]] = False
        surv = ~dies
        r = rows[surv]
        if not len(r):
            break
        nd = {key: v[surv] for key, v in nd.items()}
        m_c = unif(r, N_CONT) < nd["miss_c"]
        eps = normal(r, N_CONT)
        v_c = nd["mu"] + (nd["sd"] if temp == 1.0 else nd["sd"] * temp) * eps
        if mmse_cat:
            raw = cat_sample(nd["extra_mmse_logits"], eps[:, j_mmse])
        else:
            raw = np.clip(np.round(v_c[:, j_mmse] * sd[j_mmse] + mu[j_mmse]), 0.0, 30.0)
        v_c[:, j_mmse] = (raw - mu[j_mmse]) / sd[j_mmse]
        v_c = np.clip(v_c, -6.0, 6.0)
        m_s = unif(r, N_STATE) < nd["miss_s"]
        v_s = (unif(r, N_STATE) < nd["xs"]).astype(np.float32)
        v_s[:, :N_CUM] = np.maximum(v_s[:, :N_CUM], cum_known[r])
        cum_known[r] = np.where(m_s[:, :N_CUM], cum_known[r], v_s[:, :N_CUM])
        ever = np.maximum(x[r, I_EVER:I_AD], (unif(r, I_AD - I_EVER) < nd["ever"]))
        xn = x[r].copy()
        xn[:, :N_CONT] = np.where(m_c, 0.0, v_c)
        xn[:, I_MC:I_XS] = m_c
        xn[:, I_XS:I_MS] = np.where(m_s, 0.0, v_s)
        xn[:, I_MS:I_EVER] = m_s
        xn[:, I_EVER:I_AD] = ever
        xn[:, I_AD] = np.maximum(xn[:, I_AD], (t_ad[r] <= k).astype(np.float32))
        xn[:, I_TF] += SIM.AGE_STEP_Z
        xn[:, I_TF + 1] = (yy[r] + k) / 10.0
        xn[:, I_TF + 2] = 1.0
        x[r] = xn
        stepper.update(st, r, xn, yy[r] + k)
        record(k, r, xn)
        n_visits[r] += 1
    rs = lambda a: a.reshape(n, n_mc, *a.shape[1:])            # noqa: E731
    return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
            "n_visits": rs(n_visits)}


def _sim_cfg(stepper):
    cfg = getattr(getattr(stepper, "model", None), "cfg", None)
    if cfg is None:
        raise RuntimeError("trfx_ext_heads simulators read cfg['sim'] from stepper.model.cfg (TRFXModel)")
    return cfg["sim"]


@TM.register(TM.SIMULATORS, "temp")
def simulator_temp(stepper, x0, s0, y0, n_mc, rng, norm):
    """Shared sampling rule with the continuous Gaussian noise scaled by sim.temp (inference-time knob)."""
    return simulate_visits_x(stepper, x0, s0, y0, n_mc, rng, norm, temp=float(_sim_cfg(stepper)["temp"]))


@TM.register(TM.SIMULATORS, "lhs")
def simulator_lhs(stepper, x0, s0, y0, n_mc, rng, norm):
    """Shared sampling rule with stratified (Latin-hypercube) draws across each start point's n_mc futures."""
    return simulate_visits_x(stepper, x0, s0, y0, n_mc, rng, norm, temp=float(_sim_cfg(stepper)["temp"]), lhs=True)


@TM.register(TM.SIMULATORS, "cat31")
def simulator_cat31(stepper, x0, s0, y0, n_mc, rng, norm):
    """Shared sampling rule with MMSE drawn from the 31-way categorical of ar_next_cat31."""
    return simulate_visits_x(stepper, x0, s0, y0, n_mc, rng, norm, temp=float(_sim_cfg(stepper)["temp"]),
                             mmse_cat=True)
