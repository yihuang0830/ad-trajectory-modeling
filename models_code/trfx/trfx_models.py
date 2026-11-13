"""trfx_models.py -- config-driven autoregressive visit transformer (TRFXModel).

The DEFAULT config reproduces rollout_versions.SeqARModel("transformer") (models.py "transformer", impl "seqar") exactly:
same modules built in the same order (=> identical initial parameters for the same torch seed), same loss
(1-year death/AD BCE + next-visit NLL), same training loop (AdamW, batch 64, grad-clip 1.0, early stopping on the test
split with patience 15, min improvement 1e-5), same rollout (rollout_versions.SeqStepper + simulator.simulate_visits:
every step re-encodes the whole visit buffer and reads the current position). trfx_equiv_test.py checks this.

Everything is generative: the model only predicts the NEXT year; every answer (risks at 1..10 y, trajectories) is
counted from simulated futures (simulator.risks = share of the n_mc trajectories with the event).

Extension points -- add a function/class to a registry and select it by name in the config (keep defaults unchanged):
  INPUT_LAYERS[name](cfg, ctx) -> nn.Module   forward(b) -> visit tokens (n, V, d); b = dict(X (n,V,54), M (n,V) bool,
                                              S (n,25) static, L (n,V,120) l2c history feats, delta (n,V,24), Y)
  POSITIONS[name](cfg, ctx)    -> nn.Module   forward(x, b) -> x with position information added
  ENCODERS[name](cfg, ctx)     -> nn.Module   forward(x, b) -> h (n, V, d); MUST be causal (h_v sees visits <= v only)
  HEADS[name](cfg, ctx)        -> nn.Module   forward(h, b) -> dict(risk (n,V,2) logits [death 1y, AD 1y],
                                              mu, log_sd, miss_c (15 each), xs, miss_s (9 each), ever (2)) (raw logits)
  LOSS_TERMS[name](b, out, ctx) -> scalar     total loss = sum_k w_k * term_k over cfg["loss"]["terms"] (ordered dict)
  STEPPERS[name](model, start, n_mc) -> obj   init/step/update protocol of simulator.simulate_visits
  READOUTS[name](sim) -> (n, 6, 10)           risks counted from simulated futures (default: share with the event)
  BATCH_TRANSFORMS[name](b, ctx, gen) -> b    training-only batch transform (augmentation, dropout of visits, ...);
                                              gen = torch.Generator seeded from the run seed (default "none" = identity,
                                              consumes no randomness)
  ctx = {"x_mean", "n_feat", "n_static", "n_l2c", "n_next", "n_risk", "needs": set of batch keys the modules read,
         "net": the TRFXNet (set after construction; loss terms may run extra forward passes with it)}
Components that need extra batch tensors add their names to ctx["needs"] in __init__ ("L", "delta"); only those are
built during rollouts (speed).

Config (JSON; merged onto DEFAULT; unknown keys are an error so typos cannot silently fall back to defaults;
"loss.terms" is replaced as a whole, every other dict is merged key by key). To add a switch: add its key to DEFAULT with
the value that keeps the baseline behaviour, read it in your component, and add a test config under configs/.
"""
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "models_code"), os.path.join(HERE, "unified_eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bl_data as B  # noqa: E402
import rollout_versions as RV  # noqa: E402
import sequence as SQ  # noqa: E402
import simulator as SIM  # noqa: E402

DEFAULT = {
    "name": "baseline",
    "notes": "",
    "model": {
        "input": "concat_linear",          # Linear(54 visit features + 25 static -> d)
        "pos": "visit_index",              # learned embedding of the visit index (0 = first visit), max_v entries
        "encoder": "causal_transformer",   # nn.TransformerEncoder, causal mask + padding mask
        "head": "ar_next",                 # 2 risk logits (death / AD within 1 y) + next-visit distribution (65)
        "d": 64, "layers": 2, "heads": 4, "ff_mult": 4, "dropout": 0.2, "max_v": 64, "norm_first": True,
    },
    "loss": {"terms": {"risk_bce_1y": 1.0, "next_nll": 1.0}},
    "train": {"lr": 5e-4, "wd": 1e-2, "batch_size": 64, "max_epochs": 150, "patience": 15, "min_delta": 1e-5,
              "clip": 1.0, "eval_bs": 256,
              "threads": 8,                  # fixed: CPU results are bit-reproducible only at a fixed thread count
                                             # (0 = $SLURM_CPUS_PER_TASK); perf-only knob, not in the config hash
              "batch_transform": "none",
              "fitter": "standard"},           # FITTERS registry (standard = fit_loop)
    "sim": {"stepper": "full_recompute", "readout": "share", "n_mc_lm": 100, "n_mc_traj": 100, "lm_chunk": 1000,
            "net_chunk": 20000,
            "simulator": "shared"},          # SIMULATORS registry (shared = simulator.simulate_visits)
}
# knobs that change speed only (not results beyond float noise): excluded from the config hash
PERF_ONLY = {("train", "threads"), ("sim", "net_chunk")}
REPLACE_WHOLE = {("loss", "terms")}


def merge_config(user, base=DEFAULT, path=()):
    out = copy.deepcopy(base)
    for k, v in (user or {}).items():
        if k not in base:
            raise KeyError(f"unknown config key {'.'.join(path + (k,))} (add it to trfx_models.DEFAULT first)")
        if isinstance(base[k], dict) and isinstance(v, dict) and (path + (k,)) not in REPLACE_WHOLE:
            out[k] = merge_config(v, base[k], path + (k,))
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_diff(cfg, base=DEFAULT, path=()):
    """The part of a merged config that differs from DEFAULT (name/notes/perf-only knobs excluded)."""
    out = {}
    for k, v in cfg.items():
        p = path + (k,)
        if p in (("name",), ("notes",)) or p in PERF_ONLY:
            continue
        b = base.get(k, None) if isinstance(base, dict) else None
        if isinstance(v, dict) and isinstance(b, dict) and p not in REPLACE_WHOLE:
            d = config_diff(v, b, p)
            if d:
                out[k] = d
        elif v != b:
            out[k] = v
    return out


def config_hash(cfg):
    """Hash of the overrides relative to DEFAULT: adding a new switch to DEFAULT (with its baseline-preserving value)
    does not change the hash of existing configs; the baseline config hashes to the empty diff."""
    return hashlib.sha1(json.dumps(config_diff(cfg), sort_keys=True).encode()).hexdigest()[:10]


def load_config(path_or_dict):
    user = path_or_dict
    if isinstance(path_or_dict, str):
        with open(path_or_dict) as f:
            user = json.load(f)
    return merge_config(user)


# ============================================================================ registries

INPUT_LAYERS, POSITIONS, ENCODERS, HEADS, LOSS_TERMS, STEPPERS, READOUTS = {}, {}, {}, {}, {}, {}, {}
BATCH_TRANSFORMS = {}
SIMULATORS, FITTERS = {}, {}     # sim.simulator, train.fitter (2026-11-05: extension points for trfx_ext_* modules)


def register(reg, name):
    def deco(f):
        assert name not in reg, f"duplicate registry name {name}"
        reg[name] = f
        return f
    return deco


# ---------------------------------------------------------------- input layers

@register(INPUT_LAYERS, "concat_linear")
class ConcatLinear(nn.Module):
    """Original Causal.inp: one Linear over [visit vector (54), static background (25)] -> d."""

    def __init__(self, cfg, ctx):
        super().__init__()
        self.lin = nn.Linear(ctx["n_feat"] + ctx["n_static"], cfg["model"]["d"])

    def forward(self, b):
        n, V, _ = b["X"].shape
        return self.lin(torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1)], -1))


# ---------------------------------------------------------------- positions

@register(POSITIONS, "visit_index")
class VisitIndexPos(nn.Module):
    """Original Causal.pos: learned embedding of the visit index (first visit = 0; not right-aligned)."""

    def __init__(self, cfg, ctx):
        super().__init__()
        self.emb = nn.Embedding(cfg["model"]["max_v"], cfg["model"]["d"])

    def forward(self, x, b):
        return x + self.emb(torch.arange(x.shape[1], device=x.device))[None]


@register(POSITIONS, "none")
class NoPos(nn.Module):
    def __init__(self, cfg, ctx):
        super().__init__()

    def forward(self, x, b):
        return x


# ---------------------------------------------------------------- encoders

@register(ENCODERS, "causal_transformer")
class CausalTransformer(nn.Module):
    """Original Causal.enc: pre-LN TransformerEncoder (ff = ff_mult * d), causal mask + key padding mask ~M."""

    def __init__(self, cfg, ctx):
        super().__init__()
        m = cfg["model"]
        el = nn.TransformerEncoderLayer(m["d"], m["heads"], m["ff_mult"] * m["d"], m["dropout"], batch_first=True,
                                        norm_first=m["norm_first"])
        self.enc = nn.TransformerEncoder(el, m["layers"])

    def forward(self, x, b):
        V = x.shape[1]
        causal = torch.triu(torch.ones(V, V, dtype=torch.bool, device=x.device), 1)
        return self.enc(x, mask=causal, src_key_padding_mask=~b["M"])


# ---------------------------------------------------------------- heads

@register(HEADS, "ar_next")
class ARNextHead(nn.Module):
    """Original SeqSim heads: risk Linear(d, 2) [death 1y, AD 1y] then next Linear(d, 65) split by rollout_versions.split_next."""

    def __init__(self, cfg, ctx):
        super().__init__()
        d = cfg["model"]["d"]
        self.risk = nn.Linear(d, ctx["n_risk"])
        self.next = nn.Linear(d, ctx["n_next"])

    def forward(self, h, b):
        return {"risk": self.risk(h), **RV.split_next(self.next(h))}


# ---------------------------------------------------------------- loss terms

RISK_COLS = [RV.IDX_D1, RV.IDX_A1]          # label columns (of the 60) the 2 risk logits are trained on


@register(LOSS_TERMS, "risk_bce_1y")
def loss_risk_bce_1y(b, out, ctx):
    """BCE of the 2 risk logits vs 'death within 1 y' / 'AD within 1 y' at every observed visit with a known label."""
    y = b["Y"][..., RISK_COLS]
    ok = (y >= 0) & b["M"][..., None]
    return F.binary_cross_entropy_with_logits(out["risk"][ok], y[ok])


@register(LOSS_TERMS, "next_nll")
def loss_next_nll(b, out, ctx):
    """Next-visit NLL (rollout_versions.next_nll) on consecutive visit pairs exactly 1 year apart."""
    X, M = b["X"], b["M"]
    gap1 = (abs(X[:, 1:, B.I_TF + 2] - 1.0) < 1e-6) & M[:, 1:]
    o = {k: out[k][:, :-1] for k in RV.NEXT_KEYS}
    return RV.next_nll(o, X[:, 1:], X[:, :-1], gap1)


# ---------------------------------------------------------------- rollout

@register(STEPPERS, "full_recompute")
def stepper_full_recompute(model, start, n_mc):
    """rollout_versions.SeqStepper: per trajectory a visit buffer; each year re-encodes the buffer, reads position pos."""
    return RV.SeqStepper(model, start, n_mc, model.chunk)


@register(READOUTS, "share")
def readout_share(sim):
    """simulator.risks: share of the n_mc simulated futures with a first passage / event within H years."""
    return SIM.risks(sim)


@register(BATCH_TRANSFORMS, "none")
def bt_none(b, ctx, gen):
    return b


# ============================================================================ network + training loop

class TRFXNet(nn.Module):
    """input -> pos -> encoder -> head. Construction order = original SeqSim(Causal) => same init for the same seed."""

    def __init__(self, cfg, ctx):
        super().__init__()
        m = cfg["model"]
        self.input = INPUT_LAYERS[m["input"]](cfg, ctx)
        self.pos = POSITIONS[m["pos"]](cfg, ctx)
        self.encoder = ENCODERS[m["encoder"]](cfg, ctx)
        self.head = HEADS[m["head"]](cfg, ctx)

    def forward(self, b):
        x = self.pos(self.input(b), b)
        return self.head(self.encoder(x, b), b)


def make_loss(cfg, ctx):
    terms = list(cfg["loss"]["terms"].items())
    for name, _ in terms:
        assert name in LOSS_TERMS, f"unknown loss term {name}"

    def loss_of(b, out):
        total = None
        for name, w in terms:
            t = LOSS_TERMS[name](b, out, ctx)
            t = t if w == 1.0 else w * t
            total = t if total is None else total + t
        return total
    return loss_of


def fit_loop(model, tr, te, loss_of, tc, seed, log_every=0, ctx=None):
    """= sequence.fit (same RNG use, same order of operations) + per-epoch test-loss history. Returns (best, epochs, hist).
    train.batch_transform (training batches only) gets its own torch.Generator so the default path is untouched."""
    rng = np.random.default_rng(seed)
    bt_name = tc.get("batch_transform", "none")
    bt = BATCH_TRANSFORMS[bt_name]
    gen = None
    if bt_name != "none":
        gen = torch.Generator(device=tr["X"].device)
        gen.manual_seed(10_000 + int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    best, best_state, bad, ep = np.inf, None, 0, 0
    n = len(tr["X"])
    hist = []
    for ep in range(tc["max_epochs"]):
        model.train()
        for idx in SQ._batches(n, tc["batch_size"], rng):
            b = SQ._sub(tr, torch.as_tensor(idx, device=tr["X"].device))
            if gen is not None:
                b = bt(b, ctx, gen)
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
def eval_loss(model, te, loss_of, bs=256):
    model.eval()
    ls = []
    for s in range(0, len(te["X"]), bs):
        b = SQ._sub(te, torch.arange(s, min(s + bs, len(te["X"])), device=te["X"].device))
        ls.append(float(loss_of(b, model(b))))
    return float(np.mean(ls))


# ============================================================================ model

FITTERS["standard"] = fit_loop


@register(SIMULATORS, "shared")
def simulator_shared(stepper, x0, s0, y0, n_mc, rng, norm):
    return SIM.simulate_visits(stepper, x0, s0, y0, n_mc, rng, norm)


class TRFXModel(RV.SeqSimModel):
    """Drop-in for rollout_versions.SeqARModel("transformer") with a config. Methods used by the driver:
    fit(Dtr, Dte, seed) -> self (self.info), sim(start, n_mc, rng, norm) -> trajectories, readout(sim) -> (n, 6, 10)."""

    cols = RISK_COLS
    idx = (0, 1)

    def __init__(self, cfg=None, device="cpu"):
        self.cfg = cfg if (cfg is not None and "model" in cfg and "sim" in cfg) else merge_config(cfg)
        self.kind = "transformer"             # SeqStepper only special-cases "fnn"
        self.device = device
        self.chunk = int(self.cfg["sim"]["net_chunk"])

    def ctx(self, x_mean):
        return {"x_mean": x_mean, "n_feat": B.N_FEAT, "n_static": 25, "n_l2c": 120, "n_next": RV.N_NEXT,
                "n_risk": len(self.cols), "needs": set()}

    def build(self, x_mean):
        self._ctx = self.ctx(x_mean)
        return TRFXNet(self.cfg, self._ctx)

    def fit(self, Dtr, Dte, seed=0, log_every=0):
        t0 = time.time()
        torch.manual_seed(seed)
        tr, te = SQ.tensors(Dtr, self.device), SQ.tensors(Dte, self.device)
        val, m = SQ.obs_values(tr["X"])
        x_mean = ((val * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)).cpu().numpy()
        self.net = self.build(x_mean).to(self.device)
        self._ctx["net"] = self.net
        self.loss_of = make_loss(self.cfg, self._ctx)
        tc = self.cfg["train"]
        best, n_ep, hist = FITTERS[tc.get("fitter", "standard")](self.net, tr, te, self.loss_of, tc, seed, log_every, self._ctx)
        self.net.eval()
        # the baseline objective on test with the selected weights: comparable across configs whose loss differs
        std = make_loss(DEFAULT, self._ctx)
        self.info = {"best_test_loss": best, "epochs": n_ep, "minutes": (time.time() - t0) / 60,
                     "test_loss_std": eval_loss(self.net, te, std), "loss_hist": hist,
                     "n_params": int(sum(p.numel() for p in self.net.parameters()))}
        return self

    def outputs(self, Xb, Mb, Sb, Lb, pos):
        """= SeqSimModel.outputs without building unused tensors (delta / L / Y) unless a module needs them."""
        V = int(pos.max()) + 1
        X, M = Xb[:, :V], Mb[:, :V]
        dev = self.device
        need = self._ctx["needs"]
        with torch.no_grad():
            T = {"X": torch.as_tensor(X, dtype=torch.float32, device=dev),
                 "M": torch.as_tensor(M, dtype=torch.bool, device=dev),
                 "S": torch.as_tensor(Sb, dtype=torch.float32, device=dev)}
            if "L" in need:
                L = Lb[:, :V] if Lb is not None else np.zeros((len(X), V, 120), np.float32)
                T["L"] = torch.as_tensor(L, dtype=torch.float32, device=dev)
            if "delta" in need:
                T["delta"] = torch.as_tensor(SQ.obs_delta(X, M), dtype=torch.float32, device=dev)
            o = self.net(T)
            ar = torch.arange(len(X), device=dev)
            pt = torch.as_tensor(pos, device=dev)
            r = torch.sigmoid(o["risk"][ar, pt])
            nd = {k: o[k][ar, pt].cpu().numpy() for k in RV.NEXT_KEYS}
            extra = {k: o[k][ar, pt].cpu().numpy() for k in o if k.startswith("extra_")}   # custom heads -> custom simulators
        r = r.cpu().numpy()
        out = RV.nd_from_logits(nd)
        out.update(extra)
        return r[:, self.idx[0]], r[:, self.idx[1]], out

    def sim(self, start, n_mc, rng, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        y0 = np.round(D["X"][ii, vv, B.I_TF + 1] * 10.0)
        stepper = STEPPERS[self.cfg["sim"]["stepper"]](self, start, n_mc)
        return SIMULATORS[self.cfg["sim"].get("simulator", "shared")](stepper, D["X"][ii, vv], D["S"][ii], y0, n_mc, rng, norm)

    def readout(self, sim):
        return READOUTS[self.cfg["sim"]["readout"]](sim)


# ============================================================================ extensions (2026-11-05)
# Each experiment family lives in its own module code/trfx_ext_<topic>.py, so several agents can work in parallel without
# editing this file. A module does `import trfx_models as TM`, registers components with TM.register(TM.<REGISTRY>, name)
# and adds its new config keys with TM.add_defaults(("model",), {...}) using BASELINE-PRESERVING default values
# (the config hash is a diff against DEFAULT, so existing hashes do not change).


def add_defaults(path, entries):
    d = DEFAULT
    for k in path:
        d = d[k]
    for k, v in entries.items():
        assert k not in d, "config key %s already exists" % ".".join(tuple(path) + (k,))
        d[k] = v


def _load_extensions():
    import glob
    import importlib
    for f in sorted(glob.glob(os.path.join(HERE, "trfx_ext_*.py"))):
        mod = os.path.splitext(os.path.basename(f))[0]
        try:                       # a broken extension must not break other experiments' runs
            importlib.import_module(mod)
        except Exception as e:     # noqa: BLE001
            print(f"[trfx] WARNING: extension {mod} failed to import: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


_load_extensions()
