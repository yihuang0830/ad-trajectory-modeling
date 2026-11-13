"""nacc_delphi.py -- Delphi 也做一遍 NACC 预训练（2026-10-22 加）：Delphi 在 ROSMAP 上差，是不是只因为 ROSMAP 太小。

    python nacc_delphi.py tokens            # NACC -> delphi/data/nacc/（交付版 dedup 的分词规则）、delphi/data/rosmap_naccview/、训练配置
    python nacc_delphi.py finetune_init     # NACC 检查点 -> ROSMAP 微调的起点（delphi/Delphi-ROSMAP-nojitter-nacc[-sNN]/ckpt.pt）
    python nacc_delphi.py zeroshot          # NACC 检查点 -> 不微调、直接在 ROSMAP 上读的副本（delphi/Delphi-NACC-zs[-sNN]/）
    python nacc_delphi.py nacc_test         # NACC 自己的 test 段：Delphi 和其它 NACC 预训练的模型同一批站点、同一套标签
                                            #   -> $ZOO_RUN_DIR/packages/_nacc/delphi_nacc_test.json

分词：词表、切点、365 天网格、两级去重、事件发射的规则逐条照 tokenization/build.py（labels.csv 直接复制交付版的，
token id 一个不变）。NACC 的原始量用 nacc_data.load_nacc 清洗过的同一批列（和 nacc_data.py 的 NACC 数据同源）：
  背景    性别、教育、种族、西语裔、APOE（剂量 + ε2）、吸烟；子研究（ROS / MAP / LATC）和饮酒 NACC 没有，不发射
  基线    高血压、糖尿病、心衰、冠心病（第一次访视有过 = *_PREVALENT）；跛行 NACC 没有
  事件    卒中（CBSTROKE）、抑郁（DEP2YRS）首次；*_ONSET = 第一次知道的状态是"没有"、后来有了；AD_DX（nacc_data 的定义）
  用药    NACCAHTN / NACCLIPL / NACCDBMD / NACCADMD，开关变化时发射
  连续量  BMI、收缩压、舒张压、MMSE（只有 UDS v1/v2）、复合认知（nacc_data 的 z 分数）；化验、睡眠、炎症 NACC 没有
  死亡    nacc_data 的同一个网格年；死后病理不用（NACC 的神经病理只覆盖一小部分人）
人和划分与 nacc_data.py 逐人相同（同一个排列）：train.bin = NACC train，val.bin = NACC test（早停用，和其它模型的预训练一样）。
Delphi 的 pid = nacc_data 的 pid + 1。

零样本：ROSMAP 的 token 去掉 NACC 没有的那些（子研究、饮酒、跛行、化验、睡眠、炎症、病理）= rosmap_naccview，和 pretrain.nacc_view 同一个意思。
配方：delphi-dedup-nojitter（APOE 诊断用的那个，5 个 seed），结构 6 层 × 96 宽不变；NACC 上 40,000 步（ROSMAP 是 12,000 步）。
微调 = 同一个配方从 NACC 的权重开始（优化器重新开始、学习率照原配方），和对照只差初始化。
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import AD_BASE, DATA_DIR, DELPHI_DIR, GRID, RUN_DIR  # noqa: E402

sys.path.insert(0, os.path.join(AD_BASE, "tokenization"))
SEEDS = (42, 43, 44, 45, 46)
NACC_DS, VIEW_DS = "nacc", "rosmap_naccview"
PRE_DIR, FT_DIR, ZS_DIR = "Delphi-NACC-nojitter", "Delphi-ROSMAP-nojitter-nacc", "Delphi-NACC-zs"
# NACC 没有的 token 家族（build.py 的前缀）：零样本时从 ROSMAP 的记录里拿掉
NACC_ABSENT = ("STUDY_", "ALCOHOL_", "CLAUD_", "GLU_", "HBA1C_", "HDL_", "LDL_", "GFR_", "PSQI_", "APNEA_", "CRP_", "IL6_",
               "TNFA_", "BRAAK_", "CERAD_", "GPATH_", "AMYL_", "TANG_", "TDP_", "LEWY_", "ARTSCL_", "CAA_", "CVDA_",
               "MICROINF_", "INFARCT_")
CONT_SRC = {"BMI": "bmi", "SBP": "sbp", "DBP": "dbp", "MMSE": "mmse", "COGN": "cogn"}     # 前缀 -> load_nacc 的列
PREV = {"HTN": "hypertension_cum", "DM": "dm_cum", "CHF": "chf_cum", "HEART": "heart_cum"}
MEDS = {"ANTIHYP": "antihyp_rx", "STATIN": "statin_rx", "DIABRX": "diabetes_rx", "ADRX": "ad_rx"}


def dname(base, s):
    return base if s == 42 else f"{base}-s{s}"


def cut(edges, labs, v):
    """build.make_binner 的 'cut' 分箱。"""
    if v is None or pd.isna(v):
        return None
    return labs[min(int(np.searchsorted(edges, v, side="right")) - 1, len(labs) - 1)] if v >= edges[0] else labs[0]


# ============================================================================== NACC -> token


def person_rows(g, k, ab, ad_k, death_k, TOK, CUT):
    """一个人 -> [(年龄天, pre-shift token)]，发射顺序和 build.py 相同（背景、卒中 / 抑郁、ONSET、用药、连续量、AD、死亡），
    然后按年龄稳定排序、每个 token 每人只留第一次。"""
    import nacc_data as ND
    G = lambda kk: int(round(ab * 365.25)) + int(kk) * GRID     # noqa: E731
    ev = []
    add = lambda name, kk: ev.append((G(kk), TOK[name]))      # noqa: E731
    r0 = g.iloc[0]
    add("MALE" if r0.SEX == 1 else "FEMALE", 0)
    educ = r0.EDUC if 0 <= r0.EDUC <= 36 else None
    lab = cut(*CUT["EDU"], educ)
    if lab:
        add(f"EDU_{lab}", 0)
    rc = r0.RACE
    add("RACE_white" if rc == 1 else "RACE_black" if rc == 2 else "RACE_other" if rc in (3, 4, 5, 50) else "RACE_unk", 0)
    add({1: "HISP_yes", 0: "HISP_no"}.get(r0.HISPANIC, "HISP_unk"), 0)
    ap = r0.NACCAPOE                                   # 1 e3e3, 2 e3e4, 3 e3e2, 4 e4e4, 5 e4e2, 6 e2e2, 9 缺失
    add({1: "APOE_e4_0", 3: "APOE_e4_0", 6: "APOE_e4_0", 2: "APOE_e4_1", 5: "APOE_e4_1", 4: "APOE_e4_2"}.get(ap, "APOE_unk"), 0)
    if ap in (3, 5, 6):
        add("APOE_e2carrier", 0)
    cur, ever = r0.TOBAC30, r0.TOBAC100                # nacc_data.static_row 的同一个规则
    if cur == 1:
        add("SMOKING_current", 0)
    elif ever == 1:
        add("SMOKING_former", 0)
    elif ever == 0:
        add("SMOKING_never", 0)
    # 既往史：每次访视 有过 = 1 / 都是 0 = 0 / 不知道 = NaN（nacc_data.build 的判定）
    hist = {}
    for pre, c in PREV.items():
        vals = g[list(ND.HIST[c])].to_numpy()
        pos = np.isin(vals, (1, 2)).any(1)
        neg = (vals == 0).all(1)
        hist[pre] = np.where(pos, 1.0, np.where(neg, 0.0, np.nan))
        if hist[pre][0] == 1:
            add(f"{pre}_PREVALENT", 0)
    for col, name in (("CBSTROKE", "STROKE"), ("DEP2YRS", "DEPRESSION")):
        hit = np.where(g[col].isin([1, 2]).to_numpy() if col == "CBSTROKE" else (g[col] == 1).to_numpy())[0]
        if len(hit):
            add(name, k[hit[0]])
    for pre, h in hist.items():
        known = np.where(~np.isnan(h))[0]
        if len(known) and h[known[0]] == 0:
            j = np.where(h == 1)[0]
            if len(j):
                add(f"{pre}_ONSET", k[j[0]])
    for pre, c in MEDS.items():
        prev = 0
        for v, kk in zip(g[ND.RX[c]].to_numpy(), k):
            if v in (0, 1):
                if v != prev:
                    add(f"{pre}_{'ON' if v else 'OFF'}", kk)
                prev = v
    for pre, col in CONT_SRC.items():
        prev = None
        for v, kk in zip(g[col].to_numpy(), k):
            lab = cut(*CUT[pre], v)
            if lab is None:
                continue
            if lab != prev:
                add(f"{pre}_{lab}", kk)
            prev = lab
    if ad_k is not None:
        add("AD_DX", min(ad_k, death_k) if death_k is not None else ad_k)
    if death_k is not None:
        add("DEATH", death_k)
    ev.sort(key=lambda t: t[0])
    seen, out = set(), []
    for a, t in ev:
        if t not in seen:
            seen.add(t)
            out.append((a, t))
    return out


def tokenize_nacc():
    """nacc_data.build 的同一个循环（同样的人、同样的同年访视去重、同样的 AD / 死亡网格年），只是输出 token。"""
    import nacc_data as ND
    import spec
    meta_r = json.load(open(os.path.join(DATA_DIR, "rosmap", "meta.json")))
    TOK = meta_r["tok"]
    CUT = {pre: rule for _, pre, kind, rule in spec.CONT if kind == "cut"}
    CUT["EDU"] = dict((pre, rule) for _, pre, _, rule in spec.BACKGROUND)["EDU"]
    d, _ = ND.load_nacc()
    persons = []
    for _, g in d.groupby("NACCID", sort=False):
        r0 = g.iloc[0]
        birth = pd.Timestamp(year=int(r0.BIRTHYR), month=int(r0.BIRTHMO) if 1 <= r0.BIRTHMO <= 12 else 7, day=15)
        ab = (r0.date - birth).days / 365.25
        if ab < 50:
            continue
        k = np.round((g.date - r0.date).dt.days.to_numpy() / 365.25).astype(int)
        keep = ~pd.Series(k).duplicated().to_numpy()
        g, k = g[keep], k[keep]
        ad_rows = np.where(g.ad_vis.to_numpy())[0]
        ad_k = int(k[ad_rows[0]]) if len(ad_rows) else None
        death_k = None
        if r0.NACCDIED == 1 or (g.NACCDIED == 1).any():
            rr = g.iloc[-1]
            if 1900 < rr.NACCYOD < 2100:
                dd = pd.Timestamp(year=int(rr.NACCYOD), month=int(rr.NACCMOD) if 1 <= rr.NACCMOD <= 12 else 7, day=15)
                death_k = max(int(math.ceil((dd - r0.date).days / 365.25 - 1e-9)), int(k[-1]) + 1)
        ages = np.array([int(round(ab * 365.25)) + int(kk) * GRID for kk in k], np.int64)
        persons.append((ages, person_rows(g, k, ab, ad_k, death_k, TOK, CUT)))
    idx = np.random.default_rng(ND.SEED).permutation(len(persons))
    n_te = len(persons) // 10
    split = {"val": sorted(idx[:n_te].tolist()), "train": sorted(idx[n_te:].tolist())}     # val.bin = nacc_data 的 test 段
    return persons, split, TOK


def write_bin(path, persons, ids):
    rows = [(i + 1, a, t) for i in ids for a, t in persons[i][1]]
    arr = np.array(rows, np.uint32)
    arr.tofile(path)
    return arr


def tokens():
    persons, split, TOK = tokenize_nacc()
    # 和 nacc_data.py 那份 NACC 数据逐人对照：同一批 test 人、每个人同样的访视网格
    Z = np.load(os.path.join(RUN_DIR, "nacc", "nacc_data.npz"))
    assert len(persons) == len(Z["train_pid"]) + len(Z["test_pid"]), "人数和 nacc_data.npz 不同"
    assert sorted(Z["test_pid"].tolist()) == split["val"], "test 段和 nacc_data.npz 不是同一批人"
    for j, p in enumerate(Z["test_pid"]):
        assert np.array_equal(persons[p][0], Z["test_T"][j][Z["test_M"][j].astype(bool)]), f"pid {p} 的访视网格不同"
    out = os.path.join(DATA_DIR, NACC_DS)
    os.makedirs(out, exist_ok=True)
    info = {}
    for s, ids in split.items():
        arr = write_bin(os.path.join(out, f"{s}.bin"), persons, ids)
        n = np.unique(arr[:, 0], return_counts=True)[1]
        info[s] = {"persons": int(len(n)), "rows": int(len(arr)), "tokens_median": float(np.median(n)),
                   "tokens_p99": float(np.quantile(n, 0.99)), "tokens_max": int(n.max())}
    shutil.copy(os.path.join(DATA_DIR, "rosmap", "labels.csv"), os.path.join(out, "labels.csv"))
    inv = {v: kk for kk, v in TOK.items()}
    allr = np.concatenate([np.fromfile(os.path.join(out, f"{s}.bin"), np.uint32).reshape(-1, 3) for s in split])
    t, c = np.unique(allr[:, 2], return_counts=True)
    info["token_counts"] = {inv[int(a)]: int(b) for a, b in zip(t, c)}
    meta = dict(json.load(open(os.path.join(DATA_DIR, "rosmap", "meta.json"))),
                source="NACC UDS (models_code/nacc_delphi.py)", split_counts=info,
                split_note="val.bin = nacc_data.npz 的 test 段（同一批人）；pid = nacc_data 的 pid + 1")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    # 零样本读出用的 ROSMAP：去掉 NACC 没有的 token
    view = os.path.join(DATA_DIR, VIEW_DS)
    os.makedirs(view, exist_ok=True)
    drop = [v for kk, v in TOK.items() if kk.startswith(NACC_ABSENT)]
    for s in ("train", "val"):
        a = np.fromfile(os.path.join(DATA_DIR, "rosmap", f"{s}.bin"), np.uint32).reshape(-1, 3)
        a[~np.isin(a[:, 2], drop)].tofile(os.path.join(view, f"{s}.bin"))
    shutil.copy(os.path.join(DATA_DIR, "rosmap", "labels.csv"), os.path.join(view, "labels.csv"))
    json.dump(dict(json.load(open(os.path.join(DATA_DIR, "rosmap", "meta.json"))), view_dropped_preshift=drop,
                   source="rosmap 去掉 NACC 没有的 token（models_code/nacc_delphi.py）"),
              open(os.path.join(view, "meta.json"), "w"), indent=1)

    # 训练配置：交付版 train_delphi_rosmap.py 的配方，只改数据集、步数（命令行再给 --lifestyle_token_max=2 = nojitter）
    src = open(os.path.join(DELPHI_DIR, "config", "train_delphi_rosmap.py")).read()
    cfg = (src.replace("out_dir = 'Delphi-ROSMAP'", f"out_dir = '{PRE_DIR}'")
              .replace("dataset = 'rosmap'", f"dataset = '{NACC_DS}'")
              .replace("max_iters = 12000", "max_iters = 40000").replace("lr_decay_iters = 12000", "lr_decay_iters = 40000")
              .replace("eval_interval = 250", "eval_interval = 500"))
    for a, b in ((f"'{PRE_DIR}'", 1), (f"'{NACC_DS}'", 1), ("40000", 2)):
        assert cfg.count(a) >= b, f"配置替换没生效：{a}"
    cfg = ("# NACC 预训练（models_code/nacc_delphi.py 生成）：train_delphi_rosmap.py 的配方，"
           "只改 dataset / out_dir / 步数（NACC 的人是 ROSMAP 的 12 倍）\n" + cfg)
    open(os.path.join(DELPHI_DIR, "config", "train_delphi_nacc.py"), "w").write(cfg)
    print(json.dumps({s: info[s] for s in split}, indent=1))
    print("tokens:", json.dumps(info["token_counts"]))
    os.makedirs(os.path.join(RUN_DIR, "packages", "_nacc"), exist_ok=True)
    json.dump({s: info[s] for s in split} | {"token_counts": info["token_counts"]},
              open(os.path.join(RUN_DIR, "packages", "_nacc", "delphi_tokens.json"), "w"), indent=1)


# ============================================================================== 微调起点 / 零样本副本


def _load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def finetune_init():
    """NACC 的 best 检查点 -> 微调目录里的 ckpt.pt：权重来自 NACC，iter_num = 0、best_val_loss = 1e9、优化器是新的。
    然后 train.py 用 init_from=resume 从这里开始，配方和对照（delphi-dedup-nojitter）一字不差。"""
    import torch
    sys.path.insert(0, DELPHI_DIR)
    from model import Delphi, DelphiConfig
    for s in SEEDS:
        src = os.path.join(DELPHI_DIR, dname(PRE_DIR, s), "ckpt.pt")
        dst_dir = os.path.join(DELPHI_DIR, dname(FT_DIR, s))
        if not os.path.exists(src):
            print(f"!! 缺 {src}")
            continue
        if os.path.exists(os.path.join(dst_dir, "ckpt.pt")):
            print(f"  {dst_dir} 已经有 ckpt.pt，不覆盖")
            continue
        ck = _load(src)
        cfg = ck["config"]
        m = Delphi(DelphiConfig(**ck["model_args"]))
        m.load_state_dict(ck["model"])
        try:                                           # 和 train.py 在 GPU 上建的一样（fused AdamW）；CPU 节点上建不了就退回普通 AdamW
            opt = m.configure_optimizers(cfg["weight_decay"], cfg["learning_rate"], (cfg["beta1"], cfg["beta2"]), "cuda")
        except Exception:
            opt = m.configure_optimizers(cfg["weight_decay"], cfg["learning_rate"], (cfg["beta1"], cfg["beta2"]), "cpu")
        os.makedirs(dst_dir, exist_ok=True)
        torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(), "model_args": ck["model_args"], "iter_num": 0,
                    "best_val_loss": 1e9, "config": cfg, "init_from_nacc": {"ckpt": src, "iter_num": ck["iter_num"],
                                                                             "nacc_val_loss": float(ck["best_val_loss"])}},
                   os.path.join(dst_dir, "ckpt.pt"))
        print(f"  {src} (iter {ck['iter_num']}, NACC val {float(ck['best_val_loss']):.4f}) -> {dst_dir}")


def zeroshot():
    """NACC 的检查点原样拷一份，config 的 dataset 记成 rosmap_naccview（score_delphi 按它核对读哪份 val.bin）。"""
    import torch
    for s in SEEDS:
        src = os.path.join(DELPHI_DIR, dname(PRE_DIR, s), "ckpt.pt")
        if not os.path.exists(src):
            print(f"!! 缺 {src}")
            continue
        ck = _load(src)
        ck["config"] = dict(ck["config"], dataset=VIEW_DS, trained_on=NACC_DS)
        d = os.path.join(DELPHI_DIR, dname(ZS_DIR, s))
        os.makedirs(d, exist_ok=True)
        torch.save(ck, os.path.join(d, "ckpt.pt"))
        print(f"  {src} -> {d}")


# ============================================================================== NACC test：同一批站点上的 AUC


def nacc_test():
    import torch
    import bl_data as B
    import pretrain as P
    from sklearn.metrics import roc_auc_score
    from score_delphi import load_model, pseudo_records, query_logits
    from score_delphi_trans import READ_TOKENS
    from common import DAYS_PER_YEAR, label_names, post_id
    from transitions import TRANS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Dn = P.load_nacc()["test"]
    M, T, Y = Dn["M"].astype(bool), Dn["T"], Dn["Y"].reshape(*Dn["M"].shape, B.NQ, B.NH)
    Yv = Y[M]                                                   # (站点, 6, 10)
    lm = pd.DataFrame({"pid": np.repeat(Dn["pid"] + 1, M.sum(1)), "t": T[M]})

    def aucs(R):
        out = {}
        for qi, q in enumerate(B.Q):
            for H in (1, 2, 3, 5, 10):
                y, r = Yv[:, qi, H - 1], R[:, qi, H - 1]
                k = y >= 0
                out[f"{q}_{H}y"] = float(roc_auc_score(y[k], r[k])) if len(np.unique(y[k])) == 2 else None
        p1 = [out[f"{q}_2y"] for q in ("ad", "death", "mmse_lt24", "cogn_ltm1")]
        p3 = [out[f"{q}_5y"] for q in B.Q]
        return {"part1_2y_4endpoints": float(np.mean(p1)), "part3_5y_6questions": float(np.mean(p3)), "cells": out}

    res = {"n_test_persons": int(len(M)), "n_landmarks": int(M.sum()),
           "n_labels": {f"{q}_{H}y": {"pos": int((Yv[:, qi, H - 1] == 1).sum()), "neg": int((Yv[:, qi, H - 1] == 0).sum())}
                        for qi, q in enumerate(B.Q) for H in (2, 5)}, "models": {}}

    def report(name, Rs, extra=None):
        ens = aucs(np.mean(Rs, 0))
        per = [aucs(R) for R in Rs]
        res["models"][name] = {"n_seeds": len(Rs), "ensemble": ens,
                               "per_seed_part1": [p["part1_2y_4endpoints"] for p in per],
                               "per_seed_part3": [p["part3_5y_6questions"] for p in per], **(extra or {})}
        print(f"{name:>22s}  seeds {len(Rs)}  part1 2y {ens['part1_2y_4endpoints']:.4f}  part3 5y {ens['part3_5y_6questions']:.4f}",
              flush=True)

    # Delphi：NACC 的 val.bin（= 这批 test 人），截到站点的记录 + 查询 token，和 score_delphi_trans 同一个读法
    arr = np.fromfile(os.path.join(DATA_DIR, NACC_DS, "val.bin"), np.uint32).reshape(-1, 3)
    post = post_id(label_names(NACC_DS))
    tok_post = [post[n] for n in READ_TOKENS]
    data, hist_len = pseudo_records(arr, lm)
    Rs, info = [], []
    for s in SEEDS:
        ck = os.path.join(DELPHI_DIR, dname(PRE_DIR, s), "ckpt.pt")
        if not os.path.exists(ck):
            continue
        model, c = load_model(ck, dev)
        cfg = c["config"]
        lg = query_logits(model, data, hist_len, lm.t.to_numpy() + 1, c["model_args"]["block_size"], tok_post, dev,
                          lifestyle_range=(cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32)))
        R = np.zeros((len(lm), B.NQ, B.NH))
        for qi, (q, (_, _, toks)) in enumerate(TRANS.items()):
            lse = np.logaddexp.reduce(np.stack([lg[:, READ_TOKENS.index(n)] for n in toks], 1), axis=1)
            for hi, H in enumerate(B.HS):
                R[:, qi, hi] = -np.expm1(-np.exp(lse) * H * DAYS_PER_YEAR)
        Rs.append(R)
        info.append({"seed": s, "iter_num": int(c["iter_num"]), "nacc_val_loss": float(c["best_val_loss"])})
    if Rs:
        report("delphi-nojitter", Rs, {"ckpts": info})

    # 其它模型：pretrain.py 的 NACC 检查点（5 个 seed），同一批人
    for fam, key, name in (("seq", "transformer", "transformer"), ("seq", "grud", "grud"), ("seq", "tlstm", "tlstm")):
        Te = P.tensors(fam, Dn, "cpu")
        Rs = []
        for s in range(5):
            try:
                m, _ = P.load_pretrained(fam, key, s, "cpu")
            except FileNotFoundError:
                continue
            Rs.append(P.risk_of(fam, m, Te)[M])
        if Rs:
            report(name, Rs)
    out = os.path.join(RUN_DIR, "packages", "_nacc", "delphi_nacc_test.json")
    json.dump(res, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tokens", "finetune_init", "zeroshot", "nacc_test"])
    a = ap.parse_args()
    {"tokens": tokens, "finetune_init": finetune_init, "zeroshot": zeroshot, "nacc_test": nacc_test}[a.what]()
