"""trfx_ext_r2.py -- round-2 TRFX extensions (configs prefixed "r2-"; 2026-11-05).

Developed as code/wip_r2.py, installed atomically with `mv wip_r2.py trfx_ext_r2.py`; trfx_models auto-imports it.
Nothing here touches the default (baseline) path: only new registry entries and new config keys that only the
components below read (baseline-preserving defaults, so no existing config hash changes). No randomness is consumed at
import time. Every answer is still counted from simulated futures: the auxiliary heads are training-only (keys without
"extra_" never reach the simulator), and no component reads a label at rollout time.

Idea 1 -- categorical (histogram) head for global cognition on top of the cat31 MMSE head
  HEADS       ar_next_cat31_cog   ar_next_cat31 (same modules, same init order) + Linear(d, model.cog_bins) logits
                                  "extra_cogn_logits" over EQUAL-WIDTH bins of raw cognition (z-score units of the
                                  cohort) on [model.cog_lo, model.cog_hi] = [-4, 2] by default: 48 bins of 0.125, so
                                  the stage cuts -1 and 0 are bin edges (observed values outside
                                  the range go to the end bins). Equal width rather than
                                  quantile bins: the same resolution everywhere (quantile bins would be narrow in
                                  the centre but much wider in the tails, where a uniform jitter would make
                                  low-cognition trajectories jump). The Gaussian mu/log_sd of cognition are overwritten
                                  with the histogram's mean / log sd in z units (only so that test_loss_std and other
                                  simulators see a moment-matched Gaussian).
  LOSS_TERMS  next_nll_cat31_cog  next_nll with MMSE = 31-way cross-entropy (as next_nll_cat31) and observed cognition =
                                  histogram density NLL in z units: CE(bin) + log(bin width in z) - log sqrt(2 pi)
                                  (on the scale of the z-space Gaussian term; the shift has no gradient). Bin of a
                                  value: [e_k, e_k+1), raw rounded to 4 decimals as simulator.stage_of does.
  SIMULATORS  cat31_cog           simulator.simulate_visits (copy of trfx_ext_heads.simulate_visits_x without lhs) with
                                  MMSE from the 31-way categorical and cognition from the histogram, both by inverse
                                  CDF at Phi(the variable's own normal draw) => the same random stream as the shared
                                  simulator. The histogram inverse CDF = bin + uniform jitter inside the bin
                                  (piecewise-uniform density), no extra random numbers.
Idea 2 -- scheduled sampling / pushforward
  FITTERS     sched_sampling      fit_loop (same batch order, same AdamW / clipping / early stopping) where every
                                  training batch first runs the model WITHOUT gradient in eval mode (as in rollouts),
                                  draws a synthetic version of every next visit from the predicted next-visit
                                  distribution with the simulator's rules (Gaussian continuous values, MMSE rounded and
                                  clipped or from extra_mmse_logits, cognition from extra_cogn_logits when present,
                                  independent missingness, states with the cumulative-state rule, new-event flags), and
                                  replaces the INPUT of each visit v >= 1 that is exactly 1 year after an observed visit
                                  by that sample with probability p = train.ss_p * min(1, epoch / train.ss_ramp)
                                  (own torch.Generator, seed 20000 + seed). Kept real in the sample: AD state and the 3
                                  time features of the visit. Only visits with gap 1 are replaced (the head predicts
                                  the visit 1 year ahead). Targets (next-visit values,
                                  event labels) and the loss read the real b["X"]; the input layer reads b["X_in"]
                                  (model.input concat_linear_xin). Early stopping on the plain test loss (no sampling).
Idea 3 -- MinimalRNN model filling
  FITTERS     model_fill          fit_loop where every batch (training AND test-loss batches) first runs
                                  train.fill_passes passes without gradient in eval mode; after each pass the missing
                                  continuous inputs of visit v >= 1 are set to the model's predicted mean for that visit
                                  made at visit v-1 (clipped to [-6, 6] like simulated values); the last fill is the
                                  training input b["X_in"]. Missing flags stay inputs (the value changes from 0 to the
                                  prediction); states and the first visit are not filled. Losses read the real b["X"].
                                  Early stopping on the test loss with filled inputs. (test_loss_std is computed by the
                                  core on UNFILLED inputs, so it is not comparable for these configs.)
  STEPPERS    fill                rollout_versions.SeqStepper whose buffer holds filled visits: at init the start
                                  history is filled exactly as in training (same passes, once per start point); each
                                  step keeps the predicted means per trajectory, and every simulated visit is written to
                                  the buffer with its missing continuous values replaced by them (recursive filling as
                                  in MinimalRNN's RNNStepper). The simulator's own state (stages, events) is untouched.
Idea 4 -- hybrid encoder
  ENCODERS    trf_gru             the causal_transformer layers (same modules, built first) followed by a 1-layer GRU
                                  (hidden d) over the visit sequence: h = T(x); out = h + Dropout(GRU(LayerNorm(h)))
                                  (model.gru_resid; false = GRU state only). Transformer first, GRU last: the state the
                                  heads read is recurrent (recency weighting and smooth state updates of the recurrent
                                  rivals), while the residual keeps the transformer path intact so the GRU learns a
                                  correction. Causal (forward GRU; visits are left-aligned with no holes, padded
                                  positions come after the valid ones and cannot influence them).
Idea 5 -- cat31 + auxiliary horizons
  HEADS       ar_next_cat31_aux   ar_next_cat31 + the TRAIN-ONLY aux_risk / aux_mu2 heads of ar_next_aux (use with the
                                  loss term aux_horizons of trfx_ext_heads, e.g. weight 0.3)
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr

import trfx_models as TM
import trfx_ext_heads as H
import trfx_ext_train as TT          # noqa: F401  (registers model.input concat_linear_xin, read by the fitters below)
import bl_data as B
import rollout_versions as RV
import sequence as SQ
import simulator as SIM
from bl_data import I_AD, I_EVER, I_MC, I_MS, I_TF, I_XS, N_CONT, N_CUM, N_STATE

N_EV = I_AD - I_EVER
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

TM.add_defaults(("model",), {
    "cog_bins": 48, "cog_lo": -4.0, "cog_hi": 2.0,     # ar_next_cat31_cog: equal-width bins of raw cognition
    "gru_resid": True,                                  # trf_gru: out = h + GRU (true) or GRU only (false)
})
TM.add_defaults(("train",), {
    "ss_p": 0.0, "ss_ramp": 20,                         # fitter sched_sampling: replacement probability, ramp epochs
    "fill_passes": 1,                                   # fitter model_fill / stepper fill: no-grad filling passes
})

_NORM = {}


def col_norm(name):
    """(column among the 15 continuous, meta mu, meta sd) = what SIM.norm_info(meta) gives the simulator."""
    if name not in _NORM:
        with open(os.path.join(B.RUN_DIR, "newmodel_data_trans_meta.json")) as f:
            nm = SIM.norm_info(json.load(f))
        j = nm["names"].index(name)
        _NORM[name] = (j, float(nm["mu"][j]), float(nm["sd"][j]))
    return _NORM[name]


# ============================================================================ idea 1: cognition histogram

def hist_inv_cdf_np(logits, eps, edges):
    """Inverse CDF of the piecewise-uniform density softmax(logits) on bins [edges_k, edges_k+1) at u = Phi(eps):
    bin by the categorical CDF, position inside the bin = the remaining probability share (= uniform jitter)."""
    lg = logits.astype(np.float64)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    cdf = np.cumsum(p, 1)
    u = ndtr(eps)
    nb = p.shape[1]
    k = np.minimum((cdf < u[:, None]).sum(1), nb - 1)
    ar = np.arange(len(k))
    lo = np.where(k > 0, cdf[ar, np.maximum(k - 1, 0)], 0.0)
    pk = p[ar, k]
    frac = np.clip(np.where(pk > 0, (u - lo) / np.where(pk > 0, pk, 1.0), 0.5), 0.0, 1.0)
    return edges[k] + frac * (edges[k + 1] - edges[k])


def hist_inv_cdf_torch(logits, u, edges):
    """Torch version of hist_inv_cdf_np (any leading shape); u = uniforms; edges float64 (nb + 1,)."""
    p = torch.softmax(logits.double(), -1)
    cdf = torch.cumsum(p, -1)
    nb = p.shape[-1]
    k = (cdf < u.double()[..., None]).sum(-1).clamp(max=nb - 1)
    lo = torch.where(k > 0, cdf.gather(-1, (k - 1).clamp(min=0)[..., None])[..., 0], torch.zeros_like(u, dtype=p.dtype))
    pk = p.gather(-1, k[..., None])[..., 0]
    frac = torch.where(pk > 0, (u.double() - lo) / torch.where(pk > 0, pk, torch.ones_like(pk)),
                       torch.full_like(pk, 0.5)).clamp(0.0, 1.0)
    return edges[k] + frac * (edges[k + 1] - edges[k])


def cat_inv_cdf_torch(logits, u):
    """Torch version of trfx_ext_heads.cat_sample: k in 0..n-1 (float64) at the uniforms u."""
    p = torch.softmax(logits.double(), -1)
    cdf = torch.cumsum(p, -1)
    return (cdf < u.double()[..., None]).sum(-1).clamp(max=p.shape[-1] - 1).double()


@TM.register(TM.HEADS, "ar_next_cat31_cog")
class ARNextCat31Cog(H.ARNextCat31):
    """ar_next_cat31 (same modules, same init order) + Linear(d, cog_bins) cognition histogram logits."""

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        m = cfg["model"]
        nb, lo, hi = int(m["cog_bins"]), float(m["cog_lo"]), float(m["cog_hi"])
        assert nb >= 2 and hi > lo
        self.cog = nn.Linear(m["d"], nb)
        self.jc, self.mu_c, self.sd_c = col_norm("cogn_global")
        e_raw = np.linspace(lo, hi, nb + 1)
        e_z = (e_raw - self.mu_c) / self.sd_c
        assert e_z[0] >= -6.0 and e_z[-1] <= 6.0, "cognition bins must lie inside the simulator clip [-6, 6] (z)"
        self.edges_z_np = e_z
        w_z = np.diff(e_z)
        self.register_buffer("inner_raw", torch.as_tensor(e_raw[1:-1], dtype=torch.float64), persistent=False)
        self.register_buffer("edges_z", torch.as_tensor(e_z, dtype=torch.float64), persistent=False)
        self.register_buffer("c_z", torch.as_tensor(0.5 * (e_z[1:] + e_z[:-1]), dtype=torch.float32), persistent=False)
        self.register_buffer("w2_12", torch.as_tensor(w_z ** 2 / 12.0, dtype=torch.float32), persistent=False)
        self.register_buffer("log_w_z", torch.as_tensor(np.log(w_z), dtype=torch.float32), persistent=False)

    def forward(self, h, b):
        o = super().forward(h, b)
        lg = self.cog(h)
        p = torch.softmax(lg, -1)
        m = (p * self.c_z).sum(-1, keepdim=True)
        ls = 0.5 * torch.log((p * ((self.c_z - m) ** 2 + self.w2_12)).sum(-1, keepdim=True).clamp(min=1e-6))
        j = self.jc
        o["mu"] = torch.cat([o["mu"][..., :j], m, o["mu"][..., j + 1:]], -1)
        o["log_sd"] = torch.cat([o["log_sd"][..., :j], ls, o["log_sd"][..., j + 1:]], -1)
        o["extra_cogn_logits"] = lg
        return o

    def cogn_bin(self, z):
        """Bin index of z-scored cognition values: [e_k, e_k+1) on the raw value rounded to 4 decimals."""
        raw = torch.round((z.double() * self.sd_c + self.mu_c) * 1e4) / 1e4
        return torch.bucketize(raw, self.inner_raw, right=True)


def next_nll_cols(o, nxt, cur, valid, col_nll):
    """= trfx_ext_heads.next_nll_mmse generalised to several columns: observed elements of column j get col_nll[j](o,
    nxt, sel) instead of the Gaussian term (same order of operations, so {j_mmse: mmse_cat_nll} gives exactly
    next_nll_mmse); the mean runs over all observed continuous elements; the four BCE means are unchanged."""
    bce = lambda z, y, m: F.binary_cross_entropy_with_logits(z[m], y[m]) if m.any() else z.sum() * 0   # noqa: E731
    v = valid[..., None]
    oc = (nxt[..., I_MC:I_XS] < 0.5) & v
    if oc.any():
        sd = torch.exp(o["log_sd"]).clamp(min=1e-2)
        g = ((o["mu"] - nxt[..., :N_CONT]) / sd) ** 2 / 2 + torch.log(sd)
        oth = oc.clone()
        for j in col_nll:
            oth[..., j] = False
        tot = g[oth].sum()
        for j in sorted(col_nll):
            sel = oc[..., j]
            if sel.any():
                tot = tot + col_nll[j](o, nxt, sel).sum()
        l = tot / oc.sum()
    else:
        l = o["mu"].sum() * 0
    l = l + bce(o["miss_c"], nxt[..., I_MC:I_XS], v.expand_as(o["miss_c"]))
    l = l + bce(o["miss_s"], nxt[..., I_MS:I_EVER], v.expand_as(o["miss_s"]))
    l = l + bce(o["xs"], nxt[..., I_XS:I_MS], (nxt[..., I_MS:I_EVER] < 0.5) & v)
    l = l + bce(o["ever"], nxt[..., I_EVER:I_AD], (cur[..., I_EVER:I_AD] < 0.5) & v)
    return l


def _head_of(ctx, cls):
    hd = ctx["net"].head
    if not isinstance(hd, cls):
        raise RuntimeError("loss term needs model.head %s" % cls.__name__)
    return hd


@TM.register(TM.LOSS_TERMS, "next_nll_cat31_cog")
def loss_next_nll_cat31_cog(b, out, ctx):
    """next_nll on 1-year pairs: MMSE = 31-way cross-entropy, cognition = histogram density NLL (head ar_next_cat31_cog)."""
    hd = _head_of(ctx, ARNextCat31Cog)

    def cogn_nll(o, nxt, sel):
        k = hd.cogn_bin(nxt[..., hd.jc][sel])
        return F.cross_entropy(o["extra_cogn_logits"][sel], k, reduction="none") + hd.log_w_z[k] - LOG_SQRT_2PI

    o, nxt, cur, ok = H._gap1_pairs(b, out, tuple(RV.NEXT_KEYS) + ("extra_mmse_logits", "extra_cogn_logits"))
    return next_nll_cols(o, nxt, cur, ok, {H.mmse_norm()[0]: H.mmse_cat_nll, hd.jc: cogn_nll})


def simulate_visits_r2(stepper, x0, s0, y0, n_mc, rng, norm, years=SIM.YEARS, temp=1.0, mmse_cat=False, cogn_edges=None):
    """Copy of trfx_ext_heads.simulate_visits_x (lhs branch removed; same draws in the same order): with cogn_edges=None
    it is identical to simulate_visits_x(..., lhs=False) (checked by wip_r2_test.py). cogn_edges (z units, nb + 1):
    cognition drawn from nd["extra_cogn_logits"] at Phi(its normal draw) (histogram inverse CDF)."""
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

    def unif(rows, c=None):
        return rng.random(len(rows)) if c is None else rng.random((len(rows), c))

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
            raise KeyError("simulator needs a head that outputs extra_mmse_logits (ar_next_cat31*)")
        if cogn_edges is not None and "extra_cogn_logits" not in nd:
            raise KeyError("simulator cat31_cog needs a head that outputs extra_cogn_logits (ar_next_cat31_cog)")
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
        eps = rng.standard_normal((len(r), N_CONT))
        v_c = nd["mu"] + (nd["sd"] if temp == 1.0 else nd["sd"] * temp) * eps
        if mmse_cat:
            raw = H.cat_sample(nd["extra_mmse_logits"], eps[:, j_mmse])
        else:
            raw = np.clip(np.round(v_c[:, j_mmse] * sd[j_mmse] + mu[j_mmse]), 0.0, 30.0)
        v_c[:, j_mmse] = (raw - mu[j_mmse]) / sd[j_mmse]
        if cogn_edges is not None:
            v_c[:, j_cogn] = hist_inv_cdf_np(nd["extra_cogn_logits"], eps[:, j_cogn], cogn_edges)
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


@TM.register(TM.SIMULATORS, "cat31_cog")
def simulator_cat31_cog(stepper, x0, s0, y0, n_mc, rng, norm):
    """Shared sampling rule with MMSE from the 31-way categorical and cognition from the histogram head."""
    hd = stepper.model.net.head
    if not isinstance(hd, ARNextCat31Cog):
        raise RuntimeError("simulator cat31_cog needs model.head ar_next_cat31_cog")
    return simulate_visits_r2(stepper, x0, s0, y0, n_mc, rng, norm, temp=float(H._sim_cfg(stepper)["temp"]),
                              mmse_cat=True, cogn_edges=hd.edges_z_np)


# ============================================================================ ideas 2 + 3: two-pass fitters

def _fit_two_pass(model, tr, te, loss_of, tc, seed, log_every, ctx, train_xin, eval_xin):
    """= trfx_models.fit_loop (same batch order, optimiser, clipping, early stopping) with b = train_xin(model, b, ep,
    gen) on training batches and b = eval_xin(model, b) on test-loss batches (both may add b["X_in"])."""
    if not (ctx or {}).get("reads_X_in"):
        raise RuntimeError("this fitter writes b['X_in']; use model.input concat_linear_xin")
    if tc.get("batch_transform", "none") != "none":
        raise RuntimeError("this fitter does not combine with train.batch_transform")
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=tr["X"].device)
    gen.manual_seed(20_000 + int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    best, best_state, bad, ep = np.inf, None, 0, 0
    n = len(tr["X"])
    hist = []
    for ep in range(tc["max_epochs"]):
        model.train()
        for idx in SQ._batches(n, tc["batch_size"], rng):
            b = SQ._sub(tr, torch.as_tensor(idx, device=tr["X"].device))
            b = train_xin(model, b, ep, gen)
            l = loss_of(b, model(b))
            opt.zero_grad()
            l.backward()
            if tc["clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            opt.step()
        model.eval()
        with torch.no_grad():
            ls = []
            for s in range(0, len(te["X"]), tc["eval_bs"]):
                b = SQ._sub(te, torch.arange(s, min(s + tc["eval_bs"], len(te["X"])), device=te["X"].device))
                b = eval_xin(model, b)
                ls.append(float(loss_of(b, model(b))))
            vl = float(np.mean(ls))
        hist.append(vl)
        if log_every and ep % log_every == 0:
            print(f"    epoch {ep}: test loss {vl:.4f}", flush=True)
        if vl < best - tc["min_delta"]:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    return best, ep + 1, hist


@torch.no_grad()
def _eval_forward(model, b):
    """Forward without gradient in eval mode (no dropout, as in rollouts); restores the previous mode."""
    was = model.training
    model.eval()
    try:
        return model(b)
    finally:
        model.train(was)


@torch.no_grad()
def sample_next_visits(o, X, gen, head=None):
    """(n, V-1, 54): a draw of visit v+1 from the next-visit distribution predicted at visit v (o[:, v]), v = 0..V-2,
    with the rules of simulator.simulate_visits (Gaussian continuous values with MMSE rounded / clipped to 0..30 or drawn
    from extra_mmse_logits, cognition from extra_cogn_logits when present, clip to [-6, 6] z, independent missingness,
    states with the cumulative rule max(draw, observed cumulative state of visit v), new-event flags max(visit v, draw)).
    AD state and time features are those of the real visit v+1."""
    j_m, mu_m, sd_m = H.mmse_norm()
    cur, nxt = X[:, :-1], X[:, 1:]
    shp = tuple(cur.shape[:2])
    dev = X.device
    u_mc = torch.rand(shp + (N_CONT,), generator=gen, device=dev)
    eps = torch.randn(shp + (N_CONT,), generator=gen, device=dev)
    u_ms = torch.rand(shp + (N_STATE,), generator=gen, device=dev)
    u_xs = torch.rand(shp + (N_STATE,), generator=gen, device=dev)
    u_ev = torch.rand(shp + (N_EV,), generator=gen, device=dev)
    p = {k: o[k][:, :-1] for k in tuple(RV.NEXT_KEYS) + ("extra_mmse_logits", "extra_cogn_logits") if k in o}
    m_c = u_mc < torch.sigmoid(p["miss_c"])
    v_c = p["mu"] + torch.exp(p["log_sd"]).clamp(min=1e-2) * eps
    if "extra_mmse_logits" in p:
        raw = cat_inv_cdf_torch(p["extra_mmse_logits"], torch.special.ndtr(eps[..., j_m]))
    else:
        raw = torch.round(v_c[..., j_m].double() * sd_m + mu_m).clamp(0.0, 30.0)
    v_c[..., j_m] = ((raw - mu_m) / sd_m).to(v_c.dtype)
    if "extra_cogn_logits" in p:
        hd = head
        if not isinstance(hd, ARNextCat31Cog):
            raise RuntimeError("extra_cogn_logits needs head ar_next_cat31_cog")
        v_c[..., hd.jc] = hist_inv_cdf_torch(p["extra_cogn_logits"], torch.special.ndtr(eps[..., hd.jc]),
                                             hd.edges_z).to(v_c.dtype)
    v_c = v_c.clamp(-6.0, 6.0)
    m_s = u_ms < torch.sigmoid(p["miss_s"])
    v_s = (u_xs < torch.sigmoid(p["xs"])).to(X.dtype)
    cum_known = cur[..., I_XS:I_XS + N_CUM] * (cur[..., I_MS:I_MS + N_CUM] < 0.5).to(X.dtype)
    v_s[..., :N_CUM] = torch.maximum(v_s[..., :N_CUM], cum_known)
    ever = torch.maximum(cur[..., I_EVER:I_AD], (u_ev < torch.sigmoid(p["ever"])).to(X.dtype))
    xn = nxt.clone()
    xn[..., :N_CONT] = torch.where(m_c, torch.zeros_like(v_c), v_c)
    xn[..., I_MC:I_XS] = m_c.to(X.dtype)
    xn[..., I_XS:I_MS] = torch.where(m_s, torch.zeros_like(v_s), v_s)
    xn[..., I_MS:I_EVER] = m_s.to(X.dtype)
    xn[..., I_EVER:I_AD] = ever
    return xn


def ss_inputs(model, b, p, gen):
    """b["X_in"] = X with each visit v >= 1 that is 1 year after an observed visit replaced, with probability p, by a
    draw from the model's own prediction made at visit v-1 (no gradient, eval mode)."""
    o = _eval_forward(model, b)
    X, M = b["X"], b["M"]
    xs = sample_next_visits(o, X, gen, getattr(model, "head", None))
    u = torch.rand(tuple(X.shape[:1]) + (X.shape[1] - 1,), generator=gen, device=X.device)
    gap1 = (torch.abs(X[:, 1:, I_TF + 2] - 1.0) < 1e-6) & M[:, 1:] & M[:, :-1]
    rep = gap1 & (u < p)
    Xi = X.clone()
    Xi[:, 1:] = torch.where(rep[..., None], xs, X[:, 1:])
    out = dict(b)
    out["X_in"] = Xi
    return out


@TM.register(TM.FITTERS, "sched_sampling")
def fit_sched_sampling(model, tr, te, loss_of, tc, seed, log_every=0, ctx=None):
    p_max, ramp = float(tc["ss_p"]), float(tc["ss_ramp"])

    def train_xin(model, b, ep, gen):
        p = p_max * (min(1.0, ep / ramp) if ramp > 0 else 1.0)
        return ss_inputs(model, b, p, gen) if p > 0 else b

    return _fit_two_pass(model, tr, te, loss_of, tc, seed, log_every, ctx, train_xin, lambda model, b: b)


def fill_from(X, M, mu):
    """X with the missing continuous values of every valid visit v >= 1 replaced by mu[:, v-1] (the prediction made at
    the previous visit), clipped to [-6, 6]; observed values, flags and everything else unchanged."""
    prev = torch.cat([torch.zeros_like(mu[:, :1]), mu[:, :-1]], 1).clamp(-6.0, 6.0).to(X.dtype)
    miss = (X[..., I_MC:I_XS] > 0.5) & M[..., None]
    miss[:, 0] = False
    Xf = X.clone()
    Xf[..., :N_CONT] = torch.where(miss, prev, X[..., :N_CONT])
    return Xf


@torch.no_grad()
def fill_inputs(model, b, passes):
    """b["X_in"] after `passes` no-gradient eval-mode passes (pass k reads the fill of pass k-1; pass 1 the raw X)."""
    X, M = b["X"], b["M"]
    out = dict(b)
    for _ in range(int(passes)):
        o = _eval_forward(model, out)
        out["X_in"] = fill_from(X, M, o["mu"])
    return out


@TM.register(TM.FITTERS, "model_fill")
def fit_model_fill(model, tr, te, loss_of, tc, seed, log_every=0, ctx=None):
    passes = int(tc["fill_passes"])
    assert passes >= 1
    return _fit_two_pass(model, tr, te, loss_of, tc, seed, log_every, ctx,
                         lambda model, b, ep, gen: fill_inputs(model, b, passes),
                         lambda model, b: fill_inputs(model, b, passes))


class FillStepper(RV.SeqStepper):
    """SeqStepper whose buffer holds filled visits (history filled as in training; simulated visits filled with the
    predicted means of the step that produced them)."""

    def init(self, n_mc):
        st = super().init(n_mc)
        m = self.model
        if m._ctx["needs"]:
            raise NotImplementedError("stepper fill: input layers that read %s are not supported" % sorted(m._ctx["needs"]))
        D, ii, vv = self.start["D"], self.start["ii"], self.start["vv"]
        V0 = int(vv.max()) + 1
        if V0 > 1:
            keep = np.arange(V0)[None, :] <= vv[:, None]
            Xh = (D["X"][ii, :V0] * keep[..., None]).astype(np.float32)
            Mh = D["M"][ii, :V0] & keep
            Xf = fill_history(m, Xh, Mh, D["S"][ii].astype(np.float32), int(m.cfg["train"]["fill_passes"]))
            rep = np.repeat(np.arange(len(ii)), n_mc)
            for s in range(0, len(rep), self.chunk):
                st["X"][s: s + self.chunk, :V0] = Xf[rep[s: s + self.chunk]]
        st["last_mu"] = np.zeros((len(st["pos"]), N_CONT), np.float32)
        return st

    def step(self, st, x, S, rows):
        p_d, p_a, nd = super().step(st, x, S, rows)
        st["last_mu"][rows] = nd["mu"]
        return p_d, p_a, nd

    def update(self, st, rows, xn, year):
        xf = xn.copy()
        miss = xf[:, I_MC:I_XS] > 0.5
        xf[:, :N_CONT] = np.where(miss, np.clip(st["last_mu"][rows], -6.0, 6.0), xf[:, :N_CONT])
        super().update(st, rows, xf, year)


@torch.no_grad()
def fill_history(model, X, M, S, passes, bs=2000):
    """Filled copies of start histories (numpy, (n, V0, 54)), exactly as fill_inputs does in training."""
    net, dev = model.net, model.device
    assert not net.training
    out = np.empty_like(X)
    for s in range(0, len(X), bs):
        T = {"X": torch.as_tensor(X[s: s + bs], dtype=torch.float32, device=dev),
             "M": torch.as_tensor(M[s: s + bs], dtype=torch.bool, device=dev),
             "S": torch.as_tensor(S[s: s + bs], dtype=torch.float32, device=dev)}
        out[s: s + bs] = fill_inputs(net, T, passes)["X_in"].cpu().numpy()
    return out


@TM.register(TM.STEPPERS, "fill")
def stepper_fill(model, start, n_mc):
    if model.cfg["train"]["fitter"] != "model_fill":
        raise RuntimeError("stepper fill expects a model trained with train.fitter model_fill (filled inputs)")
    return FillStepper(model, start, n_mc, model.chunk)


# ============================================================================ idea 4: transformer + GRU encoder

@TM.register(TM.ENCODERS, "trf_gru")
class TrfGRU(nn.Module):
    """causal_transformer (same modules, built first) -> h; out = h + Dropout(GRU(LayerNorm(h))) (or the GRU state only)."""

    def __init__(self, cfg, ctx):
        super().__init__()
        m = cfg["model"]
        self.trf = TM.CausalTransformer(cfg, ctx)
        self.ln = nn.LayerNorm(m["d"])
        self.gru = nn.GRU(m["d"], m["d"], num_layers=1, batch_first=True)
        self.drop = nn.Dropout(m["dropout"])
        self.resid = bool(m["gru_resid"])

    def forward(self, x, b):
        h = self.trf(x, b)
        g, _ = self.gru(self.ln(h))
        g = self.drop(g)
        return h + g if self.resid else g


# ============================================================================ idea 5: cat31 + auxiliary horizons

@TM.register(TM.HEADS, "ar_next_cat31_aux")
class ARNextCat31Aux(H.ARNextCat31):
    """ar_next_cat31 + the train-only auxiliary heads of ar_next_aux (aux_risk: death / AD within 2 and 3 y; aux_mu2:
    +2-year means of the 15 continuous). No "extra_" prefix => never passed to the simulator, never read out."""

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)
        d = cfg["model"]["d"]
        self.aux_risk = nn.Linear(d, len(H.AUX_COLS))
        self.aux_mu2 = nn.Linear(d, N_CONT)

    def forward(self, h, b):
        o = super().forward(h, b)
        o["aux_risk"] = self.aux_risk(h)
        o["aux_mu2"] = self.aux_mu2(h)
        return o
