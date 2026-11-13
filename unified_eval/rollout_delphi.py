"""rollout_delphi.py -- Delphi 家族的轨迹生成 + 扰动适配器（fig3 / fig4）。

生成器用 models_code/radc_delphi/engine.py 的 Engine.simulate（figure 2 已验证：死亡终止、各分词自己的
不重复规则、禁采尸检 token、按经验访视大小一次生成一整次访视）。这里只加三件事：
  1. pos_embedding 模型按训练方式右对齐（和 score_delphi 同一个 pos_align_length），否则生成时位置下标
     落在训练没用过的格子上；
  2. 统一的读出：从生成的轨迹里读 4 个终点的首次发生时间、逐年的 MMSE / 认知箱；
  3. 按临床含义的扰动（perturbations.py 定义），翻译成对 token 流的修改。

所有年龄以天计（和 .bin 一样）。token 一律用 MODEL 空间（= labels.csv 行号 = 盘上 id + 1）。
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA_DIR, label_names  # noqa: E402
from score_delphi import pos_align_length  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "models_code"))
from radc_delphi.batching import get_p2i, patient_stream  # noqa: E402
from radc_delphi.engine import PAD_AGE, load as engine_load  # noqa: E402

DPY = 365.25
SHIFT_5Y = 1826            # 年龄 +5 岁：整天数（盘上年龄是整数天）
MMSE_BINS = ("MMSE_impaired", "MMSE_borderline", "MMSE_normal")
COGN_BINS = ("COGN_low", "COGN_mid", "COGN_high")


def mmse_bin(v):
    return 0 if v < 24 else (1 if v < 27 else 2)


def make_engine(ckpt, dataset, device):
    names = label_names(dataset)
    meta = json.load(open(os.path.join(DATA_DIR, dataset, "meta.json")))
    rep = set(int(t) for t in (meta.get("repeatable_tokens_postshift") or []))
    bad = [t for t in ("AD_DX", "Death") if names.index(t) in rep]
    if bad:
        # engine_load 里 vocab.py 的断言会挡住，但断言没有消息；这里先判定，把原因写清楚
        raise RuntimeError(f"{dataset}：{' / '.join(bad)} 在这份分词里可重复（每次访视重发已患疾病），figure-2 生成器的不变量 "
                           f"'{bad[0]} not in REPEATABLE_TOKENS'（vocab.py）不成立 -> 生成器拒绝；生成的流里"
                           "'首次 AD 诊断'没有定义")
    eng = engine_load(ckpt, data_dir=os.path.join(DATA_DIR, dataset), device=device)
    if eng.args.get("visit_heads"):
        # Engine.simulate 自己也会拒绝（lm_head 不再是速率），这里提前判定，免得跑到一半才炸
        raise RuntimeError("visit_heads=True：lm_head 的 logit 不是速率，figure2 的生成器不支持")
    eng.pos_align = None
    if eng.args.get("pos_embedding"):
        cfg = eng.ckpt.get("config", {})
        lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
        L, pct = pos_align_length(dataset, cfg, eng.block_size, lr)
        orig = eng._logits

        def _logits(idx, age, mask_ties=True, _orig=orig, _L=L):
            # 生成时序列不补 padding、从位置 0 开始；训练时是右对齐到 batch 长度（中位数 _L）。
            # 短于 _L 就在左边补 padding（token 0、年龄 PAD_AGE，注意力里被遮掉），让最后一个 token 落在 _L-1。
            n = idx.shape[1]
            if n < _L:
                p = _L - n
                idx = torch.cat([torch.zeros(idx.shape[0], p, dtype=idx.dtype, device=idx.device), idx], 1)
                age = torch.cat([torch.full((age.shape[0], p), PAD_AGE, dtype=age.dtype, device=age.device), age], 1)
            return _orig(idx, age, mask_ties)
        eng._logits = _logits
        eng.pos_align = (L, pct)
    eng.tok = {n: i for i, n in enumerate(names)}
    eng.mmse_per_visit = eng.tok["MMSE_normal"] in rep
    eng.cogn_per_visit = eng.tok["COGN_high"] in rep
    eng.stroke_per_visit = eng.tok["STROKE"] in rep      # 已患疾病每次访视都重发（snapshot）-> STROKE 可重复
    return eng


def person_streams(dataset, split="val"):
    arr = np.fromfile(os.path.join(DATA_DIR, dataset, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    p2i = get_p2i(arr)
    out = {}
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        ages, toks = patient_stream(arr, s, n)
        out[int(arr[s, 0])] = (ages, toks)
    return out


# ----------------------------------------------------------------------------- 扰动（token 空间）

def _insert(ages, toks, a, tok):
    """在年龄 a 处插入一个 token（放在同龄 token 之后，稳定排序保持原有顺序）。"""
    ages = np.append(ages, float(a))
    toks = np.append(toks, int(tok))
    o = np.argsort(ages, kind="stable")
    return ages[o], toks[o]


def _set_bin(eng, ages, toks, t, fam_bins, b, per_visit):
    """把年龄 t 那次访视的某个量表设成第 b 箱，按这份分词的发射规则改 token 流。
    每次都发的分词：替换 t 处的 token；去重分词：t 处插入"变化"token（资格已保证之前都在最高箱、从没发过别的箱）。"""
    ids = [eng.tok[n] for n in fam_bins]
    at = (ages == t) & np.isin(toks, ids)
    target = ids[b]
    if at.any():
        toks = toks.copy()
        toks[at] = target
        return ages, toks
    if per_visit:
        raise ValueError("每次都发的分词在站点上没有这个量表的 token —— 资格判断和数据对不上")
    if b == len(ids) - 1:                       # 去重分词：还在最高箱 = 不发 token
        return ages, toks
    return _insert(ages, toks, t, target)


def apply(eng, pert, ages, toks, tL, t_first, visits=None):
    """返回 (ages, toks, landmark)。ages/toks 是**完整**流（含站点之后），截断由调用方按 landmark 做。
    visits：这个人全部访视的年龄（traj_truth 的 visits）。只有 stroke_early 在"每次访视都重发 STROKE"的分词里要用。"""
    T = eng.tok
    a, k = ages.copy(), toks.copy()
    if pert == "orig":
        return a, k, tL
    if pert == "apoe":
        k[k == T["APOE_e4_0"]] = T["APOE_e4_1"]
        return a, k, tL
    if pert == "male":
        k[k == T["Female"]] = T["Male"]
        return a, k, tL
    if pert == "educ":
        k[np.isin(k, [T["EDU_16to19"], T["EDU_ge20"]])] = T["EDU_lt12"]
        return a, k, tL
    if pert == "age5":
        return a + SHIFT_5Y, k, tL + SHIFT_5Y
    if pert == "stroke":
        a, k = _insert(a, k, tL, T["STROKE"])
        return a, k, tL
    if pert == "stroke_early":
        # "第一次访视就有卒中"，按这份分词自己的发射规则翻译：STROKE 只在首次出现时发 -> 第一次访视插一个；
        # 每次访视都重发已患疾病（snapshot）-> 第一次访视到站点之间的每一次访视都插一个。
        # 2026-10-12 修正：以前一律只插在第一次访视，snapshot 里第 2 次到站点那次访视的快照因此等于在说"没有卒中"。
        if not eng.stroke_per_visit:
            a, k = _insert(a, k, t_first, T["STROKE"])
            return a, k, tL
        if visits is None:
            raise ValueError("这份分词每次访视都重发 STROKE：stroke_early 需要访视年龄（visits）")
        for v in visits:
            if t_first <= v <= tL:
                a, k = _insert(a, k, v, T["STROKE"])
        return a, k, tL
    if pert.startswith("mmse"):
        v = int(pert[4:])
        a, k = _set_bin(eng, a, k, tL, MMSE_BINS, mmse_bin(v), eng.mmse_per_visit)
        return a, k, tL
    if pert == "cogn":
        a, k = _set_bin(eng, a, k, tL, COGN_BINS, 1, eng.cogn_per_visit)   # -0.5 落在中间箱
        return a, k, tL
    raise KeyError(pert)


# ----------------------------------------------------------------------------- 读出

def summarize(eng, sim_t, sim_a, t, years=5):
    """生成轨迹 (n_mc, T)（含前缀） -> 每条轨迹的首次发生时间（距 t 的年数，没有 = inf）和逐年状态。"""
    T = eng.tok
    real = sim_a > PAD_AGE + 1
    after = real & (sim_a > t)

    def first(tok_id):
        hit = after & (sim_t == tok_id)
        return (np.where(hit, sim_a, np.inf).min(1) - t) / DPY

    out = {"death": first(T["Death"]), "ad": first(T["AD_DX"]),
           "mmse": first(T["MMSE_impaired"]), "cogn": first(T["COGN_low"])}
    for fam, bins in (("mmse_state", MMSE_BINS), ("cogn_state", COGN_BINS)):
        ids = np.array([T[n] for n in bins])
        is_f = real & np.isin(sim_t, ids)
        st = np.full((sim_t.shape[0], years + 1), np.nan)
        for y in range(years + 1):
            cut = t + y * DPY
            m = is_f & (sim_a <= cut)
            last = np.where(m, np.arange(sim_t.shape[1])[None, :], -1).max(1)
            ok = last >= 0
            tok_last = sim_t[np.arange(sim_t.shape[0]), np.clip(last, 0, None)]
            st[ok, y] = np.searchsorted(ids, tok_last[ok]) if np.all(np.diff(ids) > 0) else \
                [list(ids).index(x) for x in tok_last[ok]]
        out[fam] = st
    # 逐年"这一年里有没有 token"（不算 No-event / Death / 生成器禁采的尸检 token）：第 k 年 = (t+(k-1)年, t+k年]。
    # 同一个函数也读观测流（n_mc = 1），两边口径一致。另记 token 年龄之间的间隔（天），看访视节奏。
    skip = np.array(sorted(set(eng._no_visit_t.tolist()) | set(eng._blocked_t.tolist())))
    vis = real & (sim_a > t) & (sim_a <= t + years * DPY) & ~np.isin(sim_t, skip)
    out["tok_year"] = np.stack([(vis & (sim_a > t + (k - 1) * DPY) & (sim_a <= t + k * DPY)).any(1)
                                for k in range(1, years + 1)], 1)
    gaps = []
    for r in range(sim_t.shape[0]):
        a = np.unique(sim_a[r][vis[r]])
        if len(a):
            gaps.append(np.diff(np.concatenate([[t], a])))
    out["gaps"] = np.concatenate(gaps) if gaps else np.zeros(0)
    return out


def simulate(eng, ages, toks, t, n_mc, seed, years=5, block=()):
    """block：额外禁采的 token（名字），只在这一次生成里生效（卒中反号的机制检验用）。"""
    pre = ages <= t
    old = eng._blocked_t
    if block:
        eng._blocked_t = torch.cat([old, torch.tensor([eng.tok[b] for b in block], dtype=old.dtype, device=old.device)])
    try:
        st, sa = eng.simulate(toks[pre], ages[pre], n_mc=n_mc, until_age_years=(t + years * DPY + 1) / DPY,
                              max_new_tokens=400, seed=seed)
    finally:
        eng._blocked_t = old
    return summarize(eng, st, sa, t, years)


def observed(eng, ages, toks, t, years=5):
    """观测流当成"一条轨迹"，用和生成轨迹完全相同的读出（summarize）。"""
    return summarize(eng, toks[None, :].astype(np.int64), ages[None, :].astype(float), t, years)
