"""ethos.py -- ETHOS（Renc et al., npj Digital Medicine 2024；ETHOS-ARES, GigaScience 2025）在 ROSMAP 上从头训练。

和 Delphi 同属"把病历写成 token、GPT 逐个生成"，但两处设计不同（ETHOS 原文的做法，原样照搬）：
  时间：不是连续的等待时间，而是**时间间隔 token**。原文 13 个间隔 token，最长一档 6 个月，更长的间隔重复发；
        这份数据的访视在整年网格上，所以只会用到"6 个月"这一档：间隔 1 年 = 2 个、2 年 = 4 个。
  数值：每个测量 = 变量 token + **分位数 token**（10 个分位数 token Q0–Q9 所有变量共用，切点在 train 上定）。
  风险：从前缀往后生成 N 条未来，H 年内事件的比例 = 风险（原文 rep_num 次重复生成）。
序列：静态块（性别、种族、西语裔、教育五分位、APOE、吸烟、饮酒、研究、第一次访视的年龄十分位）
      -> 第 0 次访视的 token -> 间隔 token -> 下一次访视 ……；死亡 = 最后一次访视之后若干个间隔 token + DEATH。
  一次访视（固定顺序）：15 个连续量（测到的才发）各 [变量, 分位数]；9 个状态里 5 个累计病 + 卒中 / 抑郁在第一次为真时发一次"新发"，
  4 个用药当次在用就发；AD 状态第一次为真时发 AD_DX。
模型：GPT（可学习绝对位置，从时间线起点数，不右对齐；4 层、d 128、4 头、dropout 0.1），下一个 token 的交叉熵，test 段早停。
生成：KV 缓存逐 token 采样（温度 1）；前缀之后第一个 token 只能是间隔或 DEATH（前缀已经是完整的一次访视），
      累计间隔超过 10 年或出 DEATH 就停。读出：第 k 年那次访视的分期 = 那次 MMSE / 认知分位数 token 对应的 train 值里随机抽一个
      再按 24 / 27、−1 / 0 切；AD、死亡的时间 = 生成到它时累计的间隔（死亡取那半年的中点）。
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bl_data as B
import simulator as SIM
from bl_data import I_AD, I_EVER, I_MC, I_MS, I_TF, I_XS, N_CONT, N_STATE

N_Q = 10
MED = list(range(5, 9))                      # STATE 里 5..8 是用药
CUM = list(range(5))                         # 0..4 是累计病
MAX_LEN = 1600


class Vocab:
    def __init__(self, meta):
        self.cont = meta["features"][:N_CONT]
        self.state = meta["features"][B.I_XS:B.I_MS]
        toks = ["PAD", "T_6M", "DEATH", "AD_DX"] + [f"Q{k}" for k in range(N_Q)]
        toks += [f"LAB_{c}" for c in self.cont]
        toks += [f"DX_{self.state[j]}" for j in CUM] + [f"RX_{self.state[j]}" for j in MED]
        toks += ["DX_stroke", "DX_depression"]
        toks += ["SEX_M", "SEX_F", "RACE_white", "RACE_black", "RACE_other", "RACE_na", "HISP_yes", "HISP_no", "HISP_na"]
        toks += [f"EDU_{k}" for k in range(5)] + ["EDU_na", "APOE_0", "APOE_1", "APOE_2", "APOE_na"]
        toks += ["SMOKE_never", "SMOKE_former", "SMOKE_current", "SMOKE_na", "ALC_no", "ALC_yes", "ALC_na"]
        toks += ["STUDY_ROS", "STUDY_MAP", "STUDY_LATC"] + [f"AGE_{k}" for k in range(10)]
        self.toks = toks
        self.id = {t: i for i, t in enumerate(toks)}

    def __len__(self):
        return len(self.toks)


class Tokenizer:
    """train 段上定分位数切点；每个分位数箱里存 train 的原始值（MMSE、认知读出用）。"""

    def __init__(self, Dtr, meta, norm):
        self.V = Vocab(meta)
        self.norm = norm
        X, M = Dtr["X"], Dtr["M"]
        self.edges, self.pool = [], []
        for j in range(N_CONT):
            v = X[..., j][M & (X[..., I_MC + j] < 0.5)]
            # 从没测过的量（只在 NACC 视图上训练时出现）：没有切点，也永远不会发这个量的 token
            e = np.unique(np.quantile(v, np.linspace(0.1, 0.9, N_Q - 1))) if len(v) else np.zeros(0)
            self.edges.append(e)
            raw = v * norm["sd"][j] + norm["mu"][j]
            b = np.digitize(v, e)
            self.pool.append([np.sort(raw[b == k]) for k in range(N_Q)])
        a0 = X[:, 0, I_TF][M[:, 0]]
        self.age_edges = np.quantile(a0, np.linspace(0.1, 0.9, 9))
        e = Dtr["S"][:, 1][Dtr["S"][:, 2] < 0.5]
        self.edu_edges = np.quantile(e, [0.2, 0.4, 0.6, 0.8])

    def static(self, S, x0):
        V = self.V.id
        t = [V["SEX_M"] if S[0] > 0.5 else V["SEX_F"]]
        race = ["RACE_white", "RACE_black", "RACE_other", "RACE_na"][int(np.argmax(S[3:7]))]
        t.append(V[race])
        t.append(V["HISP_na"] if S[8] > 0.5 else (V["HISP_yes"] if S[7] > 0.5 else V["HISP_no"]))
        t.append(V["EDU_na"] if S[2] > 0.5 else V[f"EDU_{int(np.digitize(S[1], self.edu_edges))}"])
        t.append(V["APOE_na"] if S[11] > 0.5 else V[f"APOE_{1 if S[9] > 0.5 else (2 if S[10] > 0.5 else 0)}"])
        t.append(V["SMOKE_na"] if S[15] > 0.5 else V["SMOKE_current" if S[14] > 0.5 else ("SMOKE_former" if S[13] > 0.5 else "SMOKE_never")])
        t.append(V["ALC_na"] if S[17] > 0.5 else V["ALC_yes" if S[16] > 0 else "ALC_no"])
        t.append(V["STUDY_MAP"] if S[18] > 0.5 else (V["STUDY_LATC"] if S[19] > 0.5 else V["STUDY_ROS"]))
        t.append(V[f"AGE_{int(np.digitize(x0[I_TF], self.age_edges))}"])
        return t

    def visit(self, x, prev):
        """一次访视的 token；prev = 上一次访视（None = 第一次），用来判断"新发"。"""
        V = self.V.id
        t = []
        for j in range(N_CONT):
            if x[I_MC + j] < 0.5:
                t += [V[f"LAB_{self.V.cont[j]}"], V[f"Q{int(np.digitize(x[j], self.edges[j]))}"]]
        for j in CUM:
            on = x[I_MS + j] < 0.5 and x[I_XS + j] > 0.5
            was = prev is not None and prev["cum"][j]
            if on and not was:
                t.append(V[f"DX_{self.V.state[j]}"])
        for j in MED:
            if x[I_MS + j] < 0.5 and x[I_XS + j] > 0.5:
                t.append(V[f"RX_{self.V.state[j]}"])
        for k, nm in enumerate(("DX_stroke", "DX_depression")):
            if x[I_EVER + k] > 0.5 and not (prev is not None and prev["ever"][k]):
                t.append(V[nm])
        if x[I_AD] > 0.5 and not (prev is not None and prev["ad"]):
            t.append(V["AD_DX"])
        cum = [(prev["cum"][j] if prev else False) or (x[I_MS + j] < 0.5 and x[I_XS + j] > 0.5) for j in CUM]
        return t, {"cum": cum, "ever": [x[I_EVER + k] > 0.5 for k in range(2)], "ad": x[I_AD] > 0.5}

    def person(self, D, i, upto=None, with_death=True):
        """第 i 个人的序列（到第 upto 次访视为止；None = 全部，再加死亡）。"""
        V = self.V.id
        M = D["M"][i]
        nv = int(M.sum())
        last = nv - 1 if upto is None else int(upto)
        seq = self.static(D["S"][i], D["X"][i, 0])
        prev = None
        for v in range(last + 1):
            x = D["X"][i, v]
            if v:
                gap = int(round(x[I_TF + 2]))
                seq += [V["T_6M"]] * (2 * max(gap, 1))
            tv, prev = self.visit(x, prev)
            seq += tv
        if upto is None and with_death:
            yd = D["Y"][i, last].reshape(B.NQ, B.NH)[B.I_DEATH]
            if (yd == 1).any():
                T = int(np.argmax(yd == 1)) + 1                     # 死在最后一次访视之后的第 T 年内
                seq += [V["T_6M"]] * (2 * T - 1) + [V["DEATH"]]
        return seq


# ----------------------------------------------------------------------------- GPT（带 KV 缓存）


class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))
        self.h, self.p = h, p

    def forward(self, x, key_mask, cache=None):
        n, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, -1)
        sh = lambda a: a.view(n, -1, self.h, d // self.h).transpose(1, 2)    # noqa: E731
        q, k, v = sh(q), sh(k), sh(v)
        if cache is not None and "k" in cache:
            k = torch.cat([cache["k"], k], 2)
            v = torch.cat([cache["v"], v], 2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        S = k.shape[2]
        ones = torch.ones(L, S, dtype=torch.bool, device=x.device)
        causal = ones.tril(S - L)
        diag = causal & ~ones.tril(S - L - 1)                 # 每个位置至少能看自己：左填充的位置整行被遮会算出 nan
        mask = (causal[None, None] & key_mask[:, None, None, :S]) | diag[None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        x = x + F.dropout(self.proj(y.transpose(1, 2).reshape(n, L, d)), self.p, self.training)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, n_vocab, d=128, layers=4, heads=4, p=0.1, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(n_vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, p) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_vocab, bias=False)
        self.drop = nn.Dropout(p)

    def forward(self, idx, pos, key_mask, caches=None):
        x = self.drop(self.tok(idx) + self.pos(pos))
        for i, b in enumerate(self.blocks):
            x = b(x, key_mask, None if caches is None else caches[i])
        return self.head(self.ln(x))


def _pad_right(seqs):
    L = max(len(s) for s in seqs)
    a = np.zeros((len(seqs), L), np.int64)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


def fit(tok, Dtr, Dte, device, seed=0, bs=32, max_epochs=200, patience=8, lr=1e-3, init=None):
    """init：起点权重（NACC 预训练的，pretrain.EthosNACC）；None = 从头训练。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = [tok.person(Dtr, i) for i in range(len(Dtr["pid"]))]
    te = [tok.person(Dte, i) for i in range(len(Dte["pid"]))]
    tr = [s[:MAX_LEN] for s in tr]
    te = [s[:MAX_LEN] for s in te]
    model = GPT(len(tok.V)).to(device)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    def loss_of(batch):
        a = torch.as_tensor(_pad_right(batch), device=device)
        km = a > 0
        pos = torch.arange(a.shape[1], device=device)[None].expand_as(a)
        out = model(a[:, :-1], pos[:, :-1], km[:, :-1])
        tgt = a[:, 1:].masked_fill(~km[:, 1:], -100)
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), tgt.reshape(-1), ignore_index=-100)

    best, state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm), bs):
            l = loss_of([tr[k] for k in perm[s: s + bs]])
            opt.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_of(te[s: s + 64])) for s in range(0, len(te), 64)]))
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model, {"best_test_loss": best, "epochs": ep + 1, "n_params": sum(p.numel() for p in model.parameters()),
                   "train_tokens": int(sum(len(s) for s in tr))}


# ----------------------------------------------------------------------------- 生成和读出


@torch.no_grad()
def generate(model, tok, prefixes, rng_t, device, years=SIM.YEARS, max_new=700, bs=1024):
    """prefixes：每条轨迹的前缀 token 列表。返回每条轨迹生成的 token（numpy，0 = 停止后的填充）。"""
    V = tok.V.id
    out = []
    allowed_first = torch.zeros(len(tok.V), dtype=torch.bool, device=device)
    allowed_first[[V["T_6M"], V["DEATH"]]] = True
    for s in range(0, len(prefixes), bs):
        P = prefixes[s: s + bs]
        n = len(P)
        L = max(len(p) for p in P)
        a = np.zeros((n, L), np.int64)
        for i, p in enumerate(P):
            a[i, L - len(p):] = p                                  # 左填充
        a = torch.as_tensor(a, device=device)
        km = a > 0
        pos = (km.long().cumsum(1) - 1).clamp(min=0)
        caches = [dict() for _ in model.blocks]
        keym = torch.cat([km, torch.ones(n, max_new, dtype=torch.bool, device=device)], 1)
        logits = model(a, pos, keym, caches)[:, -1]
        nxt_pos = km.sum(1)
        n_t = torch.zeros(n, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        gen = torch.zeros(n, max_new, dtype=torch.long, device=device)
        for step in range(max_new):
            if step == 0:
                logits = logits.masked_fill(~allowed_first, -1e9)
            nt = torch.multinomial(torch.softmax(logits.float(), -1), 1, generator=rng_t)[:, 0]
            nt = nt.masked_fill(done, 0)
            gen[:, step] = nt
            n_t += (nt == V["T_6M"]).long()
            done |= (nt == V["DEATH"]) | (n_t > 2 * years) | (nxt_pos + 1 >= model.pos.num_embeddings)
            if bool(done.all()):
                break
            logits = model(nt[:, None], nxt_pos[:, None], keym[:, :L + step + 1], caches)[:, -1]
            nxt_pos = nxt_pos + 1
        out.append(gen.cpu().numpy())
    W = max(o.shape[1] for o in out)
    return np.concatenate([np.pad(o, ((0, 0), (0, W - o.shape[1]))) for o in out])


def decode(tok, gen, base_stage, ad0, rng, years=SIM.YEARS):
    """生成的 token (N, L) -> simulator 的轨迹字典（向量化）。base_stage：起点访视的 (mmse, cogn) 分期（-1 = 没测）。"""
    V = tok.V.id
    N, L = gen.shape
    e2 = np.cumsum(gen == V["T_6M"], 1)                         # 到这个位置为止的半年数
    k = np.minimum(np.ceil(e2 / 2).astype(int), years)          # 落在第几年那次访视
    live = gen > 0
    stage = {f: np.full((N, years + 1), -1, np.int64) for f in ("mmse", "cogn")}
    stage["mmse"][:, 0], stage["cogn"][:, 0] = base_stage[:, 0], base_stage[:, 1]
    is_d = gen == V["DEATH"]
    has_d = is_d.any(1)
    pd_ = is_d.argmax(1)
    t_death = np.where(has_d, e2[np.arange(N), pd_] / 2 + 0.25, np.inf)
    is_a = (gen == V["AD_DX"]) & (k >= 1)
    has_a = is_a.any(1)
    t_ad = np.where(has_a, k[np.arange(N), is_a.argmax(1)].astype(float), np.inf)
    t_ad = np.where(ad0, -np.inf, t_ad)
    t_ad = np.where(np.isfinite(t_ad) & (t_ad > 0) & (t_ad >= t_death), np.inf, t_ad)   # AD 必须早于死亡
    q0 = V["Q0"]
    isq = (gen[:, 1:] >= q0) & (gen[:, 1:] < q0 + N_Q)
    for fam, j in (("mmse", 0), ("cogn", 1)):
        r, p = np.where((gen[:, :-1] == V[f"LAB_{tok.V.cont[j]}"]) & isq & (k[:, :-1] >= 1))
        q = gen[r, p + 1] - q0
        raw = np.full(len(r), np.nan)
        for b in range(N_Q):
            m = q == b
            pool = tok.pool[j][b]
            if m.any() and len(pool):
                raw[m] = pool[rng.integers(len(pool), size=int(m.sum()))]
        ok = np.isfinite(raw)
        stage[fam][r[ok], k[r[ok], p[ok]]] = SIM.stage_of(raw[ok], fam)
    vis = np.zeros((N, years + 1), bool)
    rr, pp = np.where(live & (k >= 1) & (gen != V["T_6M"]) & (gen != V["DEATH"]))
    vis[rr, k[rr, pp]] = True
    return {"mmse": stage["mmse"], "cogn": stage["cogn"], "t_death": t_death, "t_ad": t_ad, "n_visits": vis.sum(1)}


class Ethos:
    def __init__(self, meta, device):
        self.meta, self.device = meta, device
        self.norm = SIM.norm_info(meta)

    def fit(self, Dtr, Dte, max_epochs=200):
        t0 = time.time()
        self.tok = Tokenizer(Dtr, self.meta, self.norm)
        self.model, self.info = fit(self.tok, Dtr, Dte, self.device, max_epochs=max_epochs)
        self.info["minutes"] = (time.time() - t0) / 60
        self.info["vocab"] = len(self.tok.V)
        self.info["mmse_decile_edges_raw"] = (self.tok.edges[0] * self.norm["sd"][0] + self.norm["mu"][0]).round(2).tolist()
        return self

    def sim(self, start, n_mc, rng, norm):
        D, ii, vv = start["D"], start["ii"], start["vv"]
        pre = [self.tok.person(D, i, upto=v) for i, v in zip(ii, vv)]
        pre = [p[:9] + p[-(MAX_LEN - 709):] if len(p) > MAX_LEN - 700 else p for p in pre]   # 太长：留静态块 + 最近的部分
        prefixes = [p for p in pre for _ in range(n_mc)]
        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng.integers(2 ** 31)))
        gen = generate(self.model, self.tok, prefixes, g, self.device)
        base = np.stack([SIM_stage(D, ii, vv, self.norm, f) for f in ("mmse", "cogn")], 1)
        ad0 = D["X"][ii, vv, I_AD] > 0.5
        sim = decode(self.tok, gen, np.repeat(base, n_mc, 0), np.repeat(ad0, n_mc), rng)
        n = len(ii)
        return {k: v.reshape(n, n_mc, *v.shape[1:]) for k, v in sim.items()}


def SIM_stage(D, ii, vv, norm, fam):
    j = 0 if fam == "mmse" else 1
    x = D["X"][ii, vv]
    raw = x[:, j] * norm["sd"][j] + norm["mu"][j]
    return np.where(x[:, I_MC + j] < 0.5, SIM.stage_of(raw, fam), -1)
