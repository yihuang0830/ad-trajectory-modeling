"""trfx_ext_train.py -- training / loss / optimisation extensions for TRFX (configs prefixed "t-").

Nothing here runs unless a config selects it; every new config key defaults to a value that keeps the baseline
bit for bit (the baseline config diff stays empty). Registered components:

  INPUT_LAYERS
    concat_linear_xin   = concat_linear (same module, same parameters, same init); reads b["X_in"] when a training batch
                          transform provides it, else b["X"] (eval batches and rollouts never carry X_in). It also
                          publishes the run config as ctx["cfg"]: the core ctx has no cfg, and the loss terms / batch
                          transforms of this module read their config keys there.
  BATCH_TRANSFORMS (training batches only; targets b["X"], b["Y"] are never changed)
    noise_xin           b["X_in"] = X + N(0, train.noise_sd) on observed continuous inputs of valid visits, then a share
                          train.noise_mask_p of the observed inputs (15 continuous + 9 states) re-masked: value 0, missing
                          flag 1 (= how missing inputs are encoded). Needs model.input concat_linear_xin.
  LOSS_TERMS
    next_nll_w          next_nll with per-variable weights loss.var_w on the Gaussian part: mean over observed elements of
                          w_j * nll_ij (all weights 1 = next_nll exactly). Needs ctx["cfg"] (model.input concat_linear_xin).
    rank_1y             Dynamic-DeepHit-style pairwise ranking on the 1-year death and AD risks within the batch:
                          for each cause, mean over (event in the coming year, known non-event) pairs of
                          exp(-(p_pos - p_neg) / loss.rank_sigma), averaged over the 2 causes; p = sigmoid(logit), i.e.
                          DDH's CIF scale (sigma 0.1 is meant for probabilities; on raw logits exp(-dlogit/0.1) overflows).
  FITTERS
    refit_event         standard fit_loop (bit-identical phase 1), then the encoder and the next-visit head stay frozen and
                          only head.risk (the 2 event logits) continues training on the 1-y event BCE over frozen eval-mode
                          features, with its own early stopping on the TEST event BCE (element-pooled over the split).
                          train.refit_max_epochs 0 (default) = standard.
    ema_cosine          linear warmup (train.warmup_epochs) + cosine decay of the LR from train.lr to train.lr_min over
                          max_epochs (per step), EMA of the weights (train.ema_decay, ramped min(decay, (1+t)/(10+t)));
                          early stopping and the returned model use the EMA weights.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import trfx_models as TM

B, RV, SQ = TM.B, TM.RV, TM.SQ

# the 15 continuous variables of the visit vector, in order (= meta["features"][:15]; checked against the data meta below)
CONT_NAMES = ["cts_estmmse30", "cogn_global", "sbp_avg", "dbp_avg", "bmi", "glucose", "hba1c", "hdlchlstrl",
              "ldlchlstrl", "gfr_mdrs", "psqi_sum", "berlin_risk_class", "log_hcrp", "log_hil6", "log_htnfa"]


def _check_names():
    import json
    import os
    p = os.path.join(B.RUN_DIR, "newmodel_data_trans_meta.json")
    if os.path.exists(p):
        with open(p) as f:
            got = json.load(f)["features"][:B.N_CONT]
        assert got == CONT_NAMES, "trfx_ext_train: continuous feature names differ from the data meta"


_check_names()

TM.add_defaults(("loss",), {
    "var_w": {n: 1.0 for n in CONT_NAMES},   # next_nll_w only: per-variable weight on the Gaussian NLL
    "rank_sigma": 0.1,                         # rank_1y only
})
TM.add_defaults(("train",), {
    "noise_sd": 0.0, "noise_mask_p": 0.0,      # batch_transform noise_xin only
    "warmup_epochs": 0, "lr_min": 0.0, "ema_decay": 0.0,                 # fitter ema_cosine only
    "refit_max_epochs": 0, "refit_lr": 1e-3, "refit_bs": 512, "refit_patience": 20,   # fitter refit_event only
})


def _cfg_get(ctx, path, required=False):
    cfg = (ctx or {}).get("cfg")
    if cfg is None:
        if required:
            raise RuntimeError("trfx_ext_train: %s is read from ctx['cfg'], which the core does not provide; use "
                               "model.input concat_linear_xin (it publishes the run config)" % ".".join(path))
        cfg = TM.DEFAULT
    for k in path:
        cfg = cfg[k]
    return cfg


# ============================================================================ input layer

@TM.register(TM.INPUT_LAYERS, "concat_linear_xin")
class ConcatLinearXin(TM.ConcatLinear):
    """concat_linear reading b["X_in"] when present (training augmentation), else b["X"]; publishes ctx["cfg"]."""

    def __init__(self, cfg, ctx):
        super().__init__(cfg, ctx)          # identical construction => identical parameters / init
        ctx["cfg"] = cfg
        ctx["reads_X_in"] = True

    def forward(self, b):
        X = b["X_in"] if "X_in" in b else b["X"]
        n, V, _ = X.shape
        return self.lin(torch.cat([X, b["S"][:, None].expand(-1, V, -1)], -1))


# ============================================================================ batch transform

@TM.register(TM.BATCH_TRANSFORMS, "noise_xin")
def bt_noise_xin(b, ctx, gen):
    if not ctx.get("reads_X_in"):
        raise RuntimeError("batch_transform noise_xin writes b['X_in']; use model.input concat_linear_xin")
    tc = ctx["cfg"]["train"]
    sd, p = float(tc["noise_sd"]), float(tc["noise_mask_p"])
    X, M = b["X"], b["M"]
    dev = X.device
    nc = B.N_CONT
    valid = M[..., None]
    obs_c = (X[..., B.I_MC:B.I_XS] < 0.5) & valid
    obs_s = (X[..., B.I_MS:B.I_EVER] < 0.5) & valid
    z = torch.randn(obs_c.shape, generator=gen, device=dev)      # fixed RNG use whatever sd / p are
    u_c = torch.rand(obs_c.shape, generator=gen, device=dev)
    u_s = torch.rand(obs_s.shape, generator=gen, device=dev)
    Xi = X.clone()
    cont = torch.where(obs_c, Xi[..., :nc] + sd * z, Xi[..., :nc])
    rm_c = obs_c & (u_c < p)
    rm_s = obs_s & (u_s < p)
    Xi[..., :nc] = torch.where(rm_c, torch.zeros_like(cont), cont)
    Xi[..., B.I_MC:B.I_XS] = torch.where(rm_c, torch.ones_like(cont), Xi[..., B.I_MC:B.I_XS])
    xs = Xi[..., B.I_XS:B.I_MS]
    Xi[..., B.I_XS:B.I_MS] = torch.where(rm_s, torch.zeros_like(xs), xs)
    Xi[..., B.I_MS:B.I_EVER] = torch.where(rm_s, torch.ones_like(xs), Xi[..., B.I_MS:B.I_EVER])
    out = dict(b)
    out["X_in"] = Xi
    return out


# ============================================================================ loss terms

def _var_w_vec(ctx, device):
    v = ctx.get("_t_var_w")
    if v is None:
        w = _cfg_get(ctx, ("loss", "var_w"), required=True)
        v = torch.tensor([float(w[n]) for n in CONT_NAMES], dtype=torch.float32, device=device)
        ctx["_t_var_w"] = v
    return v


def next_nll_weighted(o, nxt, cur, valid, w):
    """rollout_versions.next_nll with per-variable weights w (15,) on the Gaussian part (mean of w_j * nll over the
    observed elements; w = 1 reproduces next_nll exactly). The 4 Bernoulli parts are unchanged."""
    bce = lambda z, y, m: F.binary_cross_entropy_with_logits(z[m], y[m]) if m.any() else z.sum() * 0   # noqa: E731
    v = valid[..., None]
    oc = (nxt[..., B.I_MC:B.I_XS] < 0.5) & v
    sd = torch.exp(o["log_sd"]).clamp(min=1e-2)
    g = ((o["mu"] - nxt[..., :B.N_CONT]) / sd) ** 2 / 2 + torch.log(sd)
    l = (g * w)[oc].mean() if oc.any() else o["mu"].sum() * 0
    l = l + bce(o["miss_c"], nxt[..., B.I_MC:B.I_XS], v.expand_as(o["miss_c"]))
    l = l + bce(o["miss_s"], nxt[..., B.I_MS:B.I_EVER], v.expand_as(o["miss_s"]))
    l = l + bce(o["xs"], nxt[..., B.I_XS:B.I_MS], (nxt[..., B.I_MS:B.I_EVER] < 0.5) & v)
    l = l + bce(o["ever"], nxt[..., B.I_EVER:B.I_AD], (cur[..., B.I_EVER:B.I_AD] < 0.5) & v)
    return l


@TM.register(TM.LOSS_TERMS, "next_nll_w")
def loss_next_nll_w(b, out, ctx):
    X, M = b["X"], b["M"]
    gap1 = (abs(X[:, 1:, B.I_TF + 2] - 1.0) < 1e-6) & M[:, 1:]
    o = {k: out[k][:, :-1] for k in RV.NEXT_KEYS}
    return next_nll_weighted(o, X[:, 1:], X[:, :-1], gap1, _var_w_vec(ctx, X.device))


@TM.register(TM.LOSS_TERMS, "rank_1y")
def loss_rank_1y(b, out, ctx):
    y = b["Y"][..., TM.RISK_COLS]
    valid = b["M"][..., None] & (y >= 0)
    p = torch.sigmoid(out["risk"])
    sigma = float(_cfg_get(ctx, ("loss", "rank_sigma")))
    terms = []
    for k in range(y.shape[-1]):
        vk, yk, pk = valid[..., k], y[..., k], p[..., k]
        pos, neg = pk[vk & (yk > 0.5)], pk[vk & (yk < 0.5)]
        if pos.numel() and neg.numel():
            terms.append(torch.exp(-(pos[:, None] - neg[None, :]) / sigma).mean())
    if not terms:
        return out["risk"].sum() * 0
    return torch.stack(terms).mean()


# ============================================================================ fitters

def _bt(tc, seed, device):
    name = tc.get("batch_transform", "none")
    gen = None
    if name != "none":
        gen = torch.Generator(device=device)
        gen.manual_seed(10_000 + int(seed))
    return TM.BATCH_TRANSFORMS[name], gen


@torch.no_grad()
def _event_feats(model, T, bs):
    """Eval-mode encoder states of every valid visit with >= 1 known 1-y event label -> (H (N, d), Y (N, 2), -1 unknown)."""
    Hs, Ys = [], []
    for s in range(0, len(T["X"]), bs):
        b = SQ._sub(T, torch.arange(s, min(s + bs, len(T["X"])), device=T["X"].device))
        h = model.encoder(model.pos(model.input(b), b), b)
        y = b["Y"][..., TM.RISK_COLS]
        ok = b["M"] & (y >= 0).any(-1)
        Hs.append(h[ok])
        Ys.append(y[ok])
    return torch.cat(Hs), torch.cat(Ys)


@TM.register(TM.FITTERS, "refit_event")
def fit_refit_event(model, tr, te, loss_of, tc, seed, log_every=0, ctx=None):
    best, n_ep, hist = TM.fit_loop(model, tr, te, loss_of, tc, seed, log_every, ctx)
    if int(tc["refit_max_epochs"]) <= 0:
        return best, n_ep, hist
    lin = model.head.risk
    assert isinstance(lin, nn.Linear), "refit_event needs a head with a Linear .risk (ar_next)"
    model.eval()
    with torch.no_grad():         # the frozen-feature refit is only valid if risk logits = head.risk(encoder state)
        b = SQ._sub(te, torch.arange(0, min(tc["eval_bs"], len(te["X"])), device=te["X"].device))
        h = model.encoder(model.pos(model.input(b), b), b)
        d = (lin(h) - model(b)["risk"])[b["M"]].abs().max().item()
        assert d < 1e-5, f"refit_event: head.risk(h) != model risk logits (max diff {d})"
    Htr, Ytr = _event_feats(model, tr, tc["eval_bs"])
    Hte, Yte = _event_feats(model, te, tc["eval_bs"])

    def bce(H, Y):
        ok = Y >= 0
        return F.binary_cross_entropy_with_logits(lin(H)[ok], Y[ok])

    for p in model.parameters():
        p.requires_grad_(False)
    for p in lin.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        b0 = float(bce(Hte, Yte))
    best_ev, bad, ep = b0, 0, -1
    best_state = {k: v.detach().clone() for k, v in lin.state_dict().items()}
    opt = torch.optim.AdamW(lin.parameters(), lr=tc["refit_lr"], weight_decay=tc["wd"])
    rng = np.random.default_rng(1_000_003 + int(seed))
    N = len(Htr)
    for ep in range(int(tc["refit_max_epochs"])):
        for idx in SQ._batches(N, int(tc["refit_bs"]), rng):
            idx = torch.as_tensor(idx, device=Htr.device)
            l = bce(Htr[idx], Ytr[idx])
            opt.zero_grad()
            l.backward()
            opt.step()
        with torch.no_grad():
            v = float(bce(Hte, Yte))
        if v < best_ev - tc["min_delta"]:
            best_ev, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in lin.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tc["refit_patience"]):
                break
    lin.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    print(f"    refit_event: test event BCE {b0:.5f} -> {best_ev:.5f} ({ep + 1} refit epochs, {N} train visits)",
          flush=True)
    best = TM.eval_loss(model, te, loss_of, tc["eval_bs"])     # training objective on test with the returned weights
    return best, n_ep, hist


@TM.register(TM.FITTERS, "ema_cosine")
def fit_ema_cosine(model, tr, te, loss_of, tc, seed, log_every=0, ctx=None):
    """fit_loop with a per-step warmup + cosine LR and EMA weights (selection and returned model use the EMA)."""
    rng = np.random.default_rng(seed)
    bt, gen = _bt(tc, seed, tr["X"].device)
    lr0, lr_min = float(tc["lr"]), float(tc["lr_min"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=tc["wd"])
    n = len(tr["X"])
    spe = math.ceil(n / tc["batch_size"])
    total, warm = int(tc["max_epochs"]) * spe, int(tc["warmup_epochs"]) * spe
    decay = float(tc["ema_decay"])

    def lr_at(s):
        if s < warm:
            return lr0 * (s + 1) / warm
        prog = min(1.0, (s - warm) / max(1, total - warm))
        return lr_min + (lr0 - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))

    live = list(model.state_dict().values())            # share storage with the parameters / buffers
    ema = [t.detach().clone() for t in live]
    best, best_state, bad, ep, step = np.inf, None, 0, 0, 0
    hist = []
    for ep in range(tc["max_epochs"]):
        model.train()
        for idx in SQ._batches(n, tc["batch_size"], rng):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            b = SQ._sub(tr, torch.as_tensor(idx, device=tr["X"].device))
            if gen is not None:
                b = bt(b, ctx, gen)
            l = loss_of(b, model(b))
            opt.zero_grad()
            l.backward()
            if tc["clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), tc["clip"])
            opt.step()
            d = min(decay, (1.0 + step) / (10.0 + step))
            with torch.no_grad():
                for e, t in zip(ema, live):
                    if e.is_floating_point():
                        e.mul_(d).add_(t, alpha=1.0 - d)
                    else:
                        e.copy_(t)
            step += 1
        with torch.no_grad():
            keep = [t.clone() for t in live]
            for t, e in zip(live, ema):
                t.copy_(e)
            vl = TM.eval_loss(model, te, loss_of, tc["eval_bs"])
            for t, k in zip(live, keep):
                t.copy_(k)
        hist.append(vl)
        if log_every and ep % log_every == 0:
            print(f"    epoch {ep}: test loss (EMA) {vl:.4f}  lr {lr_at(step - 1):.2e}", flush=True)
        if vl < best - tc["min_delta"]:
            best, bad = vl, 0
            best_state = [e.clone() for e in ema]
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    if best_state is None:
        raise RuntimeError("ema_cosine: no finite test loss (NaN from the first epoch)")
    with torch.no_grad():
        for t, e in zip(live, best_state):
            t.copy_(e)
    return best, ep + 1, hist
