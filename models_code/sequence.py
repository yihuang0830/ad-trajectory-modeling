"""sequence.py -- 看整条访视历史的深度序列模型：每次访视都是一个站点，因果结构保证只用到这次访视为止的信息，
一次前向给出所有站点的预测（和 STraTS 的训练方式一样）。

编码器（输入 = 每次访视 54 维 + 静态背景 25 维）：
  fnn       L2C-FNN（Zhang et al., HBM 2025）：不跨访视，每次访视 = 当前值 + 手工历史特征 + 静态背景，过前馈网络
  gru/lstm  标准 RNN
  transformer  因果 transformer（只看自己和更早的访视；位置 = 第几次访视，从第一次访视起数，不右对齐）
  grud      GRU-D（Che et al., Sci Rep 2018）：输入向"最近一次观测值 / 训练均值"按距上次观测的时间衰减，隐状态同样衰减
  tlstm     T-LSTM（Baytas et al., KDD 2017）：把记忆拆成短期 / 长期，短期部分按访视间隔 g(Δt) = 1 / log(e + Δt) 打折
  retain    RETAIN（Choi et al., NeurIPS 2016）：两个逆时间 GRU 给访视级 α、变量级 β 注意力，c = Σ α β ⊙ v
基线 gru / lstm / transformer / grud / tlstm / retain / dynamic-deephit（2026-10-28 起）用这里的编码器做自回归模型：
只学下一步，所有答案从 rollout 来（rollout_versions.SeqARModel、DDHARModel；Dynamic-DeepHit 的结构也在那里）。
这里的多任务头（6 道题 × 1..10 年 = 60 个 sigmoid，只对"答案已知"的格子算 BCE，不保证对 H 单调）
只剩 fnn-l2c 和 NACC 预训练实验（pretrain.py）在用。
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bl_data as B

N_OBS = B.N_CONT + B.N_STATE            # 有缺失掩码的变量：15 个连续量 + 9 个状态


def tensors(D, device):
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)   # noqa: E731
    out = {"X": t(D["X"]), "M": t(D["M"], torch.bool), "S": t(D["S"]), "L": t(D["L"]),
           "Y": t(D["Y"].astype(np.float32))}
    out["delta"] = t(obs_delta(D["X"], D["M"]))
    return out


def obs_delta(X, M):
    """GRU-D 的 δ：每个有掩码的变量，距上一次观测到它过了几年（第一次访视 = 0）。用随访年数（X 的 years_since_bl × 10）。"""
    n, V, _ = X.shape
    yrs = X[..., B.I_TF + 1] * 10.0
    obs = np.concatenate([X[..., B.I_MC:B.I_XS] < 0.5, X[..., B.I_MS:B.I_EVER] < 0.5], -1)   # (n, V, 24)
    d = np.zeros((n, V, N_OBS), np.float32)
    last = np.repeat(yrs[:, :1, None], N_OBS, 2)
    for v in range(V):
        if v:
            d[:, v] = yrs[:, v, None] - last[:, 0]
        last[:, 0] = np.where(obs[:, v], yrs[:, v, None], last[:, 0])
    return np.clip(d, 0, 20) * M[..., None]


def obs_values(X):
    """有掩码的 24 个变量的值和"观测到"掩码。"""
    val = torch.cat([X[..., :B.N_CONT], X[..., B.I_XS:B.I_MS]], -1)
    m = torch.cat([X[..., B.I_MC:B.I_XS] < 0.5, X[..., B.I_MS:B.I_EVER] < 0.5], -1).float()
    return val, m


# ----------------------------------------------------------------------------- 编码器


class FNN(nn.Module):
    """L2C-FNN 仓库的默认配置：3 层 [128, 128, 256]，LeakyReLU(0.01)，dropout 0.5。"""

    def __init__(self, d, dropout, widths=(128, 128, 256)):
        super().__init__()
        layers, n_in = [], B.N_FEAT + B.N_S + 120
        for w in widths:
            layers += [nn.Linear(n_in, w), nn.LeakyReLU(0.01), nn.Dropout(dropout)]
            n_in = w
        assert n_in == d
        self.net = nn.Sequential(*layers)

    def forward(self, b):
        V = b["X"].shape[1]
        x = torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1), b["L"]], -1)
        return self.net(x)


class RNN(nn.Module):
    def __init__(self, d, dropout, cell="gru"):
        super().__init__()
        self.inp = nn.Linear(B.N_FEAT + B.N_S, d)
        self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, b):
        V = b["X"].shape[1]
        x = torch.relu(self.inp(torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1)], -1)))
        h, _ = self.rnn(self.drop(x))
        return self.drop(h)


class Causal(nn.Module):
    def __init__(self, d, dropout, layers=2, heads=4, max_v=64):
        super().__init__()
        self.inp = nn.Linear(B.N_FEAT + B.N_S, d)
        self.pos = nn.Embedding(max_v, d)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers)

    def forward(self, b):
        n, V, _ = b["X"].shape
        x = self.inp(torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1)], -1))
        x = x + self.pos(torch.arange(V, device=x.device))[None]
        causal = torch.triu(torch.ones(V, V, dtype=torch.bool, device=x.device), 1)
        return self.enc(x, mask=causal, src_key_padding_mask=~b["M"])


class GRUD(nn.Module):
    """GRU-D：x̂ = m·x + (1−m)·(γx·x_last + (1−γx)·x̄)，γx = exp(−relu(Wγ δ + bγ))（逐变量）；h ← γh ⊙ h，
    γh = exp(−relu(Wh δ + bh))；GRU 的输入是 [x̂, m, 其余特征, 静态背景]。x̄ = 训练段均值（z 分数下连续量 = 0）。"""

    def __init__(self, d, dropout, x_mean):
        super().__init__()
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.gx = nn.Parameter(torch.zeros(N_OBS))
        self.bx = nn.Parameter(torch.zeros(N_OBS))
        self.gh = nn.Linear(N_OBS, d)
        n_rest = B.N_FEAT - B.I_EVER
        self.cell = nn.GRUCell(2 * N_OBS + n_rest + B.N_S, d)
        self.drop = nn.Dropout(dropout)
        self.d = d

    def forward(self, b):
        n, V, _ = b["X"].shape
        val, m = obs_values(b["X"])
        rest = b["X"][..., B.I_EVER:]
        h = torch.zeros(n, self.d, device=val.device)
        x_last = self.x_mean.expand(n, -1).clone()
        out = []
        for v in range(V):
            dl = b["delta"][:, v]
            gx = torch.exp(-torch.relu(self.gx * dl + self.bx))
            xh = m[:, v] * val[:, v] + (1 - m[:, v]) * (gx * x_last + (1 - gx) * self.x_mean)
            h = torch.exp(-torch.relu(self.gh(dl))) * h
            h_new = self.cell(torch.cat([xh, m[:, v], rest[:, v], b["S"]], -1), h)
            keep = b["M"][:, v:v + 1]
            h = torch.where(keep, h_new, h)
            x_last = torch.where(m[:, v] > 0, val[:, v], x_last)
            out.append(h)
        return self.drop(torch.stack(out, 1))


class TLSTM(nn.Module):
    """T-LSTM：C_s = tanh(W_d C + b_d)，Ĉ_s = C_s · g(Δt)，C* = (C − C_s) + Ĉ_s，再走标准 LSTM 门。"""

    def __init__(self, d, dropout):
        super().__init__()
        self.inp = nn.Linear(B.N_FEAT + B.N_S, d)
        self.wd = nn.Linear(d, d)
        self.gates = nn.Linear(2 * d, 4 * d)
        self.drop = nn.Dropout(dropout)
        self.d = d

    def forward(self, b):
        n, V, _ = b["X"].shape
        x = torch.relu(self.inp(torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1)], -1)))
        gap = b["X"][..., B.I_TF + 2]
        h = torch.zeros(n, self.d, device=x.device)
        c = torch.zeros_like(h)
        out = []
        for v in range(V):
            cs = torch.tanh(self.wd(c))
            c_star = (c - cs) + cs * (1.0 / torch.log(math.e + gap[:, v:v + 1]))
            i, f, o, g = self.gates(torch.cat([x[:, v], h], -1)).chunk(4, -1)
            c_new = torch.sigmoid(f) * c_star + torch.sigmoid(i) * torch.tanh(g)
            h_new = torch.sigmoid(o) * torch.tanh(c_new)
            keep = b["M"][:, v:v + 1]
            h, c = torch.where(keep, h_new, h), torch.where(keep, c_new, c)
            out.append(h)
        return self.drop(torch.stack(out, 1))


class RETAIN(nn.Module):
    """RETAIN：v_j = W_emb x_j；逆时间 GRU_α -> e_j -> α = softmax；逆时间 GRU_β -> β_j = tanh(W_β g_j)；c = Σ α_j β_j ⊙ v_j。
    第 i 个站点只看第 1..i 次访视（逆时间 = 从第 i 次往回走），每个站点单独算一遍。"""

    def __init__(self, d, dropout):
        super().__init__()
        self.emb = nn.Linear(B.N_FEAT + B.N_S, d, bias=False)
        self.ga = nn.GRU(d, d, batch_first=True)
        self.gb = nn.GRU(d, d, batch_first=True)
        self.wa = nn.Linear(d, 1)
        self.wb = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, b):
        n, V, _ = b["X"].shape
        v_all = self.drop(self.emb(torch.cat([b["X"], b["S"][:, None].expand(-1, V, -1)], -1)))
        out = []
        for i in range(V):
            vv = v_all[:, : i + 1].flip(1)                        # 第 i 次 -> 第 1 次
            mm = b["M"][:, : i + 1].flip(1)
            e = self.wa(self.ga(vv)[0]).squeeze(-1).masked_fill(~mm, -1e9)
            a = torch.softmax(e, 1)
            beta = torch.tanh(self.wb(self.gb(vv)[0]))
            out.append((a[..., None] * beta * vv).sum(1))
        return self.drop(torch.stack(out, 1))


class MultiTask(nn.Module):
    def __init__(self, enc, d):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(d, B.NQ * B.NH)

    def forward(self, b):
        return self.head(self.enc(b))                          # (n, V, 60) logits

    def loss(self, b, out):
        y = b["Y"]
        ok = (y >= 0) & b["M"][..., None]
        return F.binary_cross_entropy_with_logits(out[ok], y[ok])

    def risk(self, out):
        return torch.sigmoid(out).reshape(*out.shape[:2], B.NQ, B.NH)


def make_encoder(kind, d, dropout, x_mean):
    if kind == "fnn":
        return FNN(d, dropout)
    if kind in ("gru", "lstm"):
        return RNN(d, dropout, kind)
    if kind == "transformer":
        return Causal(d, dropout)
    if kind == "grud":
        return GRUD(d, dropout, x_mean)
    if kind == "tlstm":
        return TLSTM(d, dropout)
    if kind == "retain":
        return RETAIN(d, dropout)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- 训练


def _batches(n, bs, rng):
    p = rng.permutation(n)
    return [p[s: s + bs] for s in range(0, n, bs)]


def _sub(T, idx):
    return {k: v[idx] for k, v in T.items()}


def fit(model, tr, te, loss_of, lr, wd, bs, max_epochs, patience, seed, log_every=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad, ep = np.inf, None, 0, 0
    n = len(tr["X"])
    for ep in range(max_epochs):
        model.train()
        for idx in _batches(n, bs, rng):
            b = _sub(tr, torch.as_tensor(idx, device=tr["X"].device))
            l = loss_of(b, model(b))
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ls = []
            for s in range(0, len(te["X"]), 256):
                b = _sub(te, torch.arange(s, min(s + 256, len(te["X"])), device=te["X"].device))
                ls.append(float(loss_of(b, model(b))))
            vl = float(np.mean(ls))
        if log_every and ep % log_every == 0:
            print(f"    epoch {ep}: test loss {vl:.4f}", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, ep + 1


@torch.no_grad()
def predict_at(model, D, vL, risk_of, bs=256):
    """扰动记录：第 i 个"人"只取第 vL[i] 次访视的风险（序列模型看到的是改过的整条历史）。"""
    model.eval()
    R = []
    for s in range(0, len(D["X"]), bs):
        b = _sub(D, torch.arange(s, min(s + bs, len(D["X"])), device=D["X"].device))
        r = risk_of(model(b)).cpu().numpy()
        R.append(r[np.arange(len(r)), vL[s: s + bs]])
    return np.concatenate(R)


@torch.no_grad()
def predict(model, D, risk_of, bs=256):
    model.eval()
    R = []
    for s in range(0, len(D["X"]), bs):
        b = _sub(D, torch.arange(s, min(s + bs, len(D["X"])), device=D["X"].device))
        R.append(risk_of(model(b)).cpu().numpy())
    R = np.concatenate(R)
    M = D["M"].cpu().numpy()
    return R[M]                                    # 和 bl_data.landmarks 同一顺序（np.where(M) 的行优先）


SEQ_HP = {"fnn": {"d": 256, "dropout": 0.5, "lr": 1e-3, "wd": 1e-4},
          "gru": {"d": 64, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4},
          "lstm": {"d": 64, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4},
          "transformer": {"d": 64, "dropout": 0.2, "lr": 5e-4, "wd": 1e-2},
          "grud": {"d": 64, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4},
          "tlstm": {"d": 64, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4},
          "retain": {"d": 64, "dropout": 0.2, "lr": 1e-3, "wd": 1e-4},
          # Dynamic-DeepHit 官方默认：keep_prob 0.6、lr 1e-4、batch 32、排序损失 0.1（rollout_versions.ddhar_loss）
          "ddh": {"d": 100, "dropout": 0.4, "lr": 1e-4, "wd": 1e-5, "alpha": 0.1, "bs": 32}}


def run_sequence(kind, Z, log, device="cpu", seeds=(0,), bs=64, max_epochs=150, patience=15, extras=()):
    hp = SEQ_HP[kind]
    Dtr, Dte, Dva = (B.split(Z, s) for s in ("train", "test", "val"))
    tr, te, va = (tensors(D, device) for D in (Dtr, Dte, Dva))
    exs = [(tensors(Dp, device), np.asarray(vL)) for Dp, vL in extras]
    val, m = obs_values(tr["X"])
    x_mean = ((val * m).sum((0, 1)) / m.sum((0, 1)).clamp(min=1)).cpu().numpy()
    lm_te, lm_va = B.landmarks(Dte, "cur"), B.landmarks(Dva, "cur")
    R_te, R_va = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        t0 = time.time()
        model = MultiTask(make_encoder(kind, hp["d"], hp["dropout"], x_mean), hp["d"]).to(device)
        best, n_ep = fit(model, tr, te, model.loss, hp["lr"], hp["wd"], bs, max_epochs, patience, seed)
        rte = predict(model, te, model.risk)
        rva = predict(model, va, model.risk)
        rex = [predict_at(model, Et, vL, model.risk) for Et, vL in exs]
        log.append({"seed": seed, "best_test_bce": best, "epochs": n_ep,
                    "n_params": sum(p.numel() for p in model.parameters())})
        R_te.append(rte)
        R_va.append(rva)
        R_ex = rex
        print(f"  {kind} seed {seed}: test macro AUC {B.macro_auc(lm_te['Y'], rte):.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"val": (lm_va, np.mean(R_va, 0)), "test": (lm_te, np.mean(R_te, 0)), "chosen": hp, "extras": R_ex}
