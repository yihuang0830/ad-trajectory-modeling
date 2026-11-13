"""rollout_trans.py -- 旧 figure 2 的轨迹评估，搬进统一评估：能生成轨迹的 Delphi 模型从**第一次访视**起往后生成 10 年，
和观测轨迹比"状态怎么走"。

    python rollout_trans.py --name delphi-dedup --workers 16            # 生成 + 汇总（汇总在最后自动跑）
    python rollout_trans.py --name delphi-dedup --aggregate             # 只汇总（读已有的人级结果）
    python rollout_trans.py --name delphi-dedup --limit 20 --n_mc 20    # 冒烟：结果进人级目录下的 smoke 子目录，不进打包盒
    python rollout_trans.py --write_na                                  # 5 个不能生成轨迹的 Delphi 写 applicable=false

为什么第三部分（evaluate_transitions，站点上读一次分数排出谁会越线）之外还要这一份：
  生成模型的卖点是能**走出一整条轨迹**。旧 figure 2 问的是轨迹本身像不像 —— 下一个不同状态的转移矩阵、
  逐年各状态的占比、逐年多分类 Brier 对"状态永远不变"的复读机，以及从轨迹数出来的首次越线风险和直接读出的
  AUC 并排（同一批站点：第一次访视）。这些量只有生成模型答得出，所以只对 11 个能生成的 Delphi 跑；
  snapshot（AD_DX 每次访视重发，"首次 AD"在生成流里没有定义）、visitheads（logit 不是速率）
  写 applicable=false。

起步 = 查询 token（必须）：前缀 = 年龄 <= t0 的全部 token，再在 t0+1 天接一个 No-event，从那里开始生成。
  以前直接从访视末尾的 token 往下生成，第一步的等待时间读的是"访视里最后一个 token 之后"的位置 ——
  训练里那个位置的下一个 token 多半是同一次访视的兄弟 token（mask_ties），卒中扰动的反号就是这么来的
  （rollout_extra.query_start 是它的机制检验）。和直接读出（score_delphi）同一个位置，两边才可比。

产出：
  $ZOO_RUN_DIR/rollout_trans/<name>/persons.npz       人级（只留在 RIS）：逐年占比计数、首次越线累计计数、转移计数
  $ZOO_RUN_DIR/rollout_trans/<name>/run_info.json     运行信息（聚合）
  $ZOO_RUN_DIR/packages/<name>/trajectory.json        聚合数（可以拉回本地）；图在本地按它画（figures.py 的 fig4 / fig5）
时间：年龄以天计；第 k 年的切点 = t0 + k*365（数据网格，和真值同一把尺子）；生成器内部用 365.25 天/年。
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DELPHI_DIR, GRID, RUN_DIR  # noqa: E402
from registry import DELPHI  # noqa: E402
from transitions import STAGE_BINS, TRANS, TRANS_HORIZONS, label_col  # noqa: E402

YEARS = 10
DPY = 365.25
N_MC = 100
MAX_NEW_TOKENS = 400
SEED_OFFSET = 2           # crn_seed(pid) + 2：扰动用 +0、fig3 用 +1，这里另开一条随机数流，不和它们共用随机数
# 对称删失：死在最后一次访视之后超过 1.5 年的人当失访（和 rollout_metrics.DEATH_GAP 同一条规则，这里按整天写成
# death <= last + 548 才算）。rollout_metrics 是 death - last > 547.875 天就删失，两种写法只在恰好差 548 天时不同；
# val 里没有这样的人（2026-10-14 查过：没有恰好 548 天的死亡）
SYM_GAP_DAYS = int(round(1.5 * DPY))
Q = list(TRANS)
FAMS = ("mmse", "cogn")
# 状态编号：0..2 = 分期（最好 -> 最差）、3 = 死亡、4 = 未知。分期 = 2 - 分箱（transitions.STAGE_BINS 的箱序 0 最差..2 最好）
STATES = {"mmse": ("Normal", "Borderline", "Impaired", "Dead"), "cogn": ("High", "Mid", "Low", "Dead")}
FAM_TITLE = {"mmse": "MMSE stage", "cogn": "Global cognition stage"}
FAM_SHORT = {"mmse": "MMSE", "cogn": "Cognition"}
DEAD, UNK = 3, 4
N_BOOT = 200
BOOT_SEED = 20261014
MIN_ROWS_R = 3            # 转移矩阵的 r 至少要这么多个有定义的非对角格
MATRIX_YEAR = 5           # fig4 的转移矩阵：第一次访视的分期 -> 第 5 年的状态
MIN_ROW_MATRIX = 20       # 矩阵一行（同一基线分期、第 5 年状态已知）少于这么多人就不报这一行

GENERATIVE = [m[0] for m in DELPHI if m[3] != "snapshot" and "visitheads" not in m[4]]
assert len(GENERATIVE) == 14, GENERATIVE          # 2026-10-16 加了 delphi-dedup-nojitter；2026-10-22 加了两个 NACC 预训练的
NA_REASON = {
    "snapshot": "not generative here: the snapshot tokenization re-emits AD_DX (and every prevalent disease) at every "
                "visit, so 'first AD diagnosis' is undefined in a generated stream and the figure-2 generator refuses "
                "it (vocab invariant 'AD_DX not in REPEATABLE_TOKENS')",
    "visitheads": "not generative here: visit_heads=True, so lm_head logits are not rates (waiting time lives in "
                  "time_head, visit size in size_head); the figure-2 generator does not support it",
}


def _out_paths(name, limit, n_mc, split="val"):
    """完整运行写进打包盒；冒烟（--limit 或非默认 n_mc）和 val 以外的段（在 test 段上调参用）
    只写人级目录下的 smoke 子目录，不碰打包盒。"""
    sfx = "" if (not limit and n_mc == N_MC and split == "val") else f"_lim{limit}_mc{n_mc}"
    if split != "val":
        sfx = f"_{split}{sfx}"
    rdir = os.path.join(RUN_DIR, "rollout_trans", name)
    pkg = os.path.join(RUN_DIR, "packages", name) if not sfx else os.path.join(rdir, f"smoke{sfx}")
    return rdir, sfx, pkg


# ----------------------------------------------------------------------------- 读出（一个人的 n_mc 条轨迹）

def token_ids(tok):
    """名字 -> 读出要用的 id 表。code[fam][模型 id] = 分期编号（不是这一族 = -1）。"""
    V = max(tok.values()) + 1
    code = {}
    for fam in FAMS:
        c = np.full(V, -1, np.int64)
        for b, n in enumerate(STAGE_BINS[fam]):
            c[tok[n]] = 2 - b
        code[fam] = c
    return {"code": code, "death": int(tok["Death"]),
            "target": {q: np.array([tok[n] for n in TRANS[q][2]]) for q in Q if q != "death"}}


def readout(st, sa, t0, ids, pad_age, years=YEARS):
    """(n_mc, T) 轨迹（含前缀）-> 计数。每条轨迹：
      occ[fam, k, s]  第 k 年（切点 t0+k*365）处在状态 s 的轨迹数：切点前死了 = 死亡；否则沿用切点前最后一个发出的
                      分期 token（前缀也算）；一个都没有 = 未知
      cum[q, k-1]     第 k 年切点之前首次越线的轨迹数：t0 之后第一次发出 q 的任一目标 token，且严格早于这条轨迹的死亡
                      （死亡题 = 死亡本身）
      dfirst[q, k-1]  第 k 年切点之前死了、还没越线的轨迹数（死亡题 = 0）；条件读法 cum / (n − dfirst) 用（rollout_landmarks --cond）
      trans[fam]      (t0, t0+10 年] 里"下一个不同状态"的转移次数（4x4，对角为 0）：从基线分期出发，按时间顺序走
                      发出的分期 token 和死亡，连续重复合并。基线 = 前缀里最后一个分期 token；没有就不数
    """
    n, T = st.shape
    real = sa > pad_age + 1
    dage = np.where(real & (st == ids["death"]), sa, np.inf).min(1)
    cuts = t0 + GRID * np.arange(years + 1)
    rows, pos = np.arange(n), np.arange(T)
    occ = np.zeros((2, years + 1, 5), np.int32)
    trans = np.zeros((2, 4, 4), np.int32)
    base = np.full(2, -1, np.int64)
    end = cuts[-1]
    for j, fam in enumerate(FAMS):
        c = np.where(real, ids["code"][fam][st], -1)
        is_f = c >= 0
        for k, cut in enumerate(cuts):
            last = np.where(is_f & (sa <= cut), pos, -1).max(1)
            s = np.where(last >= 0, c[rows, np.maximum(last, 0)], UNK)
            s[dage <= cut] = DEAD
            occ[j, k] = np.bincount(s, minlength=5)
        lb = np.where(is_f & (sa <= t0), pos, -1).max(1)
        bs = np.where(lb >= 0, c[rows, np.maximum(lb, 0)], -1)
        assert (bs == bs[0]).all(), "前缀在各条轨迹里不一样"
        base[j] = bs[0]
        if base[j] < 0:
            continue
        win = is_f & (sa > t0) & (sa <= end)
        dead_in = dage <= end
        for r in range(n):
            prev = int(base[j])
            for x in c[r, win[r]].tolist() + ([DEAD] if dead_in[r] else []):
                if x != prev:
                    trans[j, prev, x] += 1
                    prev = x
    cum = np.zeros((len(Q), years), np.int32)
    dfirst = np.zeros((len(Q), years), np.int32)
    for qi, q in enumerate(Q):
        if q == "death":
            first = dage
        else:
            # 严格早于死亡：死亡之后的位置生成器已经抹成 padding，这里再按年龄卡一次（同龄的 token 不算"越线后才死"）
            hit = real & np.isin(st, ids["target"][q]) & (sa > t0) & (sa < dage[:, None])
            first = np.where(hit, sa, np.inf).min(1)
        cum[qi] = (first[:, None] <= cuts[None, 1:]).sum(0)
        if q != "death":
            dfirst[qi] = ((dage[:, None] <= cuts[None, 1:]) & (first[:, None] > cuts[None, 1:])).sum(0)
    return {"occ": occ, "cum": cum, "trans": trans, "base": base, "dfirst": dfirst}


# ----------------------------------------------------------------------------- 观测一侧（同一套定义）

def _walk(seq, M):
    prev = seq[0]
    for x in seq[1:]:
        if x != prev:
            M[prev, x] += 1
            prev = x


def observed(d, years=YEARS):
    """traj_truth 的一个人 -> 和 readout 同口径的观测量。
    第 k 年的状态：切点前（含）最后一次测到的分期；只有还在随访（last >= 切点）或按对称删失切点前已死才算"已知"。
    转移：(t0, min(t0+10 年, last)] 里测到的分期，按对称删失 10 年内死了再接一个死亡。"""
    t0, death, last, vis = int(d["t0"]), d["death"], int(d["last"]), d["visits"]
    assert int(vis[0][0]) == t0, "traj_truth 的第一次访视不是 t0"
    dead_ok = death is not None and death <= last + SYM_GAP_DAYS
    cuts = t0 + GRID * np.arange(years + 1)
    occ = np.full((2, years + 1), UNK, np.int64)
    trans = np.zeros((2, 4, 4), np.int64)
    base = np.full(2, -1, np.int64)
    for j in range(2):
        meas = [(int(v[0]), 2 - int(v[1 + j])) for v in vis if v[1 + j] is not None]
        base[j] = meas[0][1] if meas and meas[0][0] == t0 else -1
        for k, cut in enumerate(cuts):
            if dead_ok and death <= cut:
                occ[j, k] = DEAD
            elif last >= cut:
                past = [s for a, s in meas if a <= cut]
                occ[j, k] = past[-1] if past else UNK
        if base[j] >= 0:
            seq = [int(base[j])] + [s for a, s in meas if t0 < a <= min(cuts[-1], last)]
            if dead_ok and death <= cuts[-1]:
                seq.append(DEAD)
            _walk(seq, trans[j])
    return {"occ": occ, "trans": trans, "base": base}


def at_risk_flags(d):
    """第一次访视时是否在风险中。threshold 题：之前（= 第一次访视）没测到过目标区间（没测也算 —— 真值标签对没测的人
    本来就是 -1，评估时两条一起卡）；AD：没有或在 t0 之后；死亡：都在。"""
    t0, v0 = int(d["t0"]), d["visits"][0]
    out = []
    for q in Q:
        fam = TRANS[q][1]
        if fam == "event":
            out.append(True if q == "death" else (d["ad"] is None or d["ad"] > t0))
        else:
            b = v0[1] if fam == "mmse" else v0[2]
            out.append(b is None or STAGE_BINS[fam][int(b)] not in TRANS[q][2])
    return out


# ----------------------------------------------------------------------------- 生成（多进程，每个进程只加载一次模型）

_W = {}


def _init_worker(ckpt, dataset):
    import torch
    torch.set_num_threads(1)
    import rollout_delphi as RD
    eng = RD.make_engine(ckpt, dataset, "cpu")
    _W.update(eng=eng, pad=RD.PAD_AGE, streams=RD.person_streams(dataset), ids=token_ids(eng.tok))


def _engine_info(_=None):
    e = _W["eng"]
    return {"pos_align": e.pos_align, "visit_sizes": getattr(e, "visit_sizes_path", None), "block_size": int(e.block_size),
            "mmse_per_visit": bool(e.mmse_per_visit), "cogn_per_visit": bool(e.cogn_per_visit)}


def _run_one(job):
    pid, t0, n_mc, seed = job
    ages, toks = _W["streams"][pid]
    return pid, simulate_from(ages, toks, t0, n_mc, seed)


def simulate_from(ages, toks, t0, n_mc, seed):
    """流里到 t0 为止的部分 + 查询 token 起步，生成 YEARS 年，读出计数（_run_one 和 rollout_landmarks 的扰动共用）。"""
    eng = _W["eng"]
    m = ages <= t0
    # 查询 token 起步：和直接读出同一个位置（站点后 1 天的 No-event），见文件头
    a2 = np.append(ages[m], t0 + 1.0)
    k2 = np.append(toks[m], int(eng.res.NO_EVENT))
    t1 = time.time()
    st, sa = eng.simulate(k2, a2, n_mc=n_mc, until_age_years=(t0 + 1 + YEARS * DPY + 1) / DPY,
                          max_new_tokens=MAX_NEW_TOKENS, seed=seed)
    r = readout(st, sa, t0, _W["ids"], _W["pad"])
    lp = min(len(k2), int(eng.block_size))                   # 生成器只保留最后 block_size 个前缀 token
    r["n_gen"] = int((sa[:, lp:] > _W["pad"] + 1).sum())     # 生成的 token 数（n_mc 条加总），看生成节奏用
    r["sec"] = time.time() - t1
    return r


def simulate_all(args, name, ckpt, dataset):
    import multiprocessing as mp
    from run_rollout_delphi import crn_seed
    rdir, sfx, _ = _out_paths(name, args.limit, args.n_mc)
    os.makedirs(rdir, exist_ok=True)
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    pids = sorted(traj, key=int)
    if args.limit:
        pids = pids[: args.limit]
    meta = {"ckpt": ckpt, "dataset": dataset, "n_mc": args.n_mc, "years": YEARS, "seed": f"crn_seed+{SEED_OFFSET}",
            "start": "query", "max_new_tokens": MAX_NEW_TOKENS}
    part = os.path.join(rdir, f"persons{sfx}.partial.pkl")
    done = {}
    if os.path.exists(part):
        with open(part, "rb") as f:
            m0, d0 = pickle.load(f)
        if m0 == meta:
            done = {p: v for p, v in d0.items() if str(p) in set(pids)}
            print(f"  续跑：已有 {len(done)} 人", flush=True)
    jobs = [(int(p), int(traj[p]["t0"]), args.n_mc, crn_seed(p) + SEED_OFFSET) for p in pids if int(p) not in done]
    t0w = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(ckpt, dataset)) as pool:
        einfo = pool.apply(_engine_info)
        print(f"  engine {json.dumps(einfo, default=str)}  workers {args.workers}  todo {len(jobs)}/{len(pids)}", flush=True)
        for i, (pid, r) in enumerate(pool.imap_unordered(_run_one, jobs, chunksize=1)):
            done[pid] = r
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                with open(part + ".tmp", "wb") as f:
                    pickle.dump((meta, done), f)
                os.replace(part + ".tmp", part)
                el = time.time() - t0w
                print(f"  {len(done)}/{len(pids)}  {el / 60:.1f} min  ~{el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min left",
                      flush=True)
    P = [int(p) for p in pids]
    assert set(P) == set(done), "有人没跑完"
    np.savez_compressed(os.path.join(rdir, f"persons{sfx}.npz"), pid=np.array(P), t0=np.array([traj[str(p)]["t0"] for p in P]),
                        n_mc=args.n_mc, occ=np.stack([done[p]["occ"] for p in P]),
                        cum=np.stack([done[p]["cum"] for p in P]), trans=np.stack([done[p]["trans"] for p in P]),
                        base=np.stack([done[p]["base"] for p in P]), n_gen=np.array([done[p]["n_gen"] for p in P]),
                        sec=np.array([done[p]["sec"] for p in P]),
                        at_risk=np.array([at_risk_flags(traj[str(p)]) for p in P], bool))
    sec = np.array([done[p]["sec"] for p in P])
    info = dict(meta, name=name, n_persons=len(P), workers=args.workers, engine=einfo,
                wall_minutes=(time.time() - t0w) / 60, cpu_minutes=float(sec.sum() / 60),
                sec_per_person_mean=float(sec.mean()), sec_per_person_max=float(sec.max()),
                gen_tokens_per_trajectory=float(np.mean([done[p]["n_gen"] for p in P]) / args.n_mc))
    json.dump(info, open(os.path.join(rdir, f"run_info{sfx}.json"), "w"), indent=1, default=str)
    os.remove(part)
    print("simulated", json.dumps(info, default=str), flush=True)


# ----------------------------------------------------------------------------- 汇总

def _f(x):
    """JSON 友好：NaN / inf -> null，numpy 标量 -> Python。"""
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, np.ndarray):
        return _f(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    return x


def _rownorm(M):
    s = M.sum(1, keepdims=True)
    return np.where(s > 0, M / np.where(s > 0, s, 1), np.nan)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def transition_block(obs_c, pred_c, min_show):
    """观测 vs 预测的转移矩阵（行归一化）。死亡是吸收态（那一行空），r / MAE 在 3 个活着的行的 9 个非对角格上算。
    观测一行少于 min_show 次转移就不报这一行（和 evaluate_transitions 的 MIN_SHOW 同一条隐私规则）。"""
    O, P = _rownorm(obs_c.astype(float)), _rownorm(pred_c)
    small = obs_c.sum(1) < min_show
    O[small] = np.nan
    oc = obs_c.astype(float)
    oc[small] = np.nan
    cells = [(i, j) for i in range(3) for j in range(4) if i != j]
    o = np.array([O[c] for c in cells])
    p = np.array([P[c] for c in cells])
    ok = np.isfinite(o) & np.isfinite(p)
    r = float(np.corrcoef(o[ok], p[ok])[0, 1]) if ok.sum() >= MIN_ROWS_R and o[ok].std() > 0 and p[ok].std() > 0 else np.nan
    mae = float(np.abs(o[ok] - p[ok]).mean()) if ok.any() else np.nan
    return {"observed_counts": oc, "predicted_counts": pred_c, "observed_rownorm": O, "predicted_rownorm": P,
            "pearson_r": r, "mae": mae, "n_obs_transitions": int(obs_c.sum()), "n_pred_transitions": float(pred_c.sum()),
            "n_cells": int(ok.sum())}


def occupancy_block(occ_obs, occ_pred, n_mc, min_show, k0=1):
    """第 k0..10 年，只看这一年观测状态已知的人：观测的 4 状态占比 vs 同一批人的平均预测概率（预测在 4 个已知状态上重新归一化）。"""
    out = {"years": list(range(k0, YEARS + 1)), "observed": [], "predicted": [], "unknown_sim": [], "n_known": [],
           "n_dropped_all_unknown": []}
    for k in range(k0, YEARS + 1):
        so, pk = occ_obs[:, k], occ_pred[:, k, :4].astype(float)
        mass = pk.sum(1)
        inc = (so != UNK) & (mass > 0)
        n = int(inc.sum())
        out["n_known"].append(n)
        out["n_dropped_all_unknown"].append(int(((so != UNK) & (mass == 0)).sum()))
        if n < min_show:
            out["observed"].append([np.nan] * 4)
            out["predicted"].append([np.nan] * 4)
            out["unknown_sim"].append(np.nan)
            continue
        out["observed"].append([float(np.mean(so[inc] == s)) for s in range(4)])
        out["predicted"].append((pk[inc] / mass[inc, None]).mean(0))
        out["unknown_sim"].append(float((occ_pred[inc, k, UNK] / n_mc).mean()))
    return out


def brier_block(occ_obs, occ_pred, base, min_show):
    """逐年多分类 Brier（4 状态），只看观测已知、基线测过的人年。复读机 = 基线分期上的点质量（对 0、错 2）。
    离开基线的 AUC：标签 = 观测状态 != 基线，分数 = 1 - P(基线状态)。mean_diff 的 CI：按人 bootstrap。"""
    n = len(base)
    Y = YEARS
    inc = np.zeros((n, Y), bool)
    bm = np.zeros((n, Y))
    bc = np.zeros((n, Y))
    out = {"years": list(range(1, Y + 1)), "model": [], "carry_forward": [], "n": [], "auc_left_baseline": [],
           "n_left_baseline": []}
    for k in range(1, Y + 1):
        so, pk = occ_obs[:, k], occ_pred[:, k, :4].astype(float)
        mass = pk.sum(1)
        m = (so != UNK) & (mass > 0) & (base >= 0)
        p = pk / np.where(mass > 0, mass, 1)[:, None]
        y = np.eye(4)[np.where(so == UNK, 0, so)]
        inc[:, k - 1] = m
        bm[:, k - 1] = ((p - y) ** 2).sum(1)
        bc[:, k - 1] = np.where(so == base, 0.0, 2.0)
        nk = int(m.sum())
        out["n"].append(nk)
        if nk < min_show:
            for key in ("model", "carry_forward", "auc_left_baseline", "n_left_baseline"):
                out[key].append(np.nan)
            continue
        out["model"].append(float(bm[m, k - 1].mean()))
        out["carry_forward"].append(float(bc[m, k - 1].mean()))
        left = (so[m] != base[m]).astype(int)
        out["n_left_baseline"].append(int(left.sum()))
        out["auc_left_baseline"].append(_auc(left, 1 - p[m, base[m]]))
    W = inc & (np.array(out["n"]) >= min_show)[None, :]
    D = np.where(W, bm - bc, 0.0)

    def per_year_mean(cnt):
        num, den = cnt @ D, cnt @ W.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        return np.nanmean(per) if np.isfinite(per).any() else np.nan

    out["mean_diff"] = float(per_year_mean(np.ones(n)))
    rng = np.random.default_rng(BOOT_SEED)
    bs = [per_year_mean(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(N_BOOT)]
    out["mean_diff_ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    out["n_boot"], out["boot_seed"] = N_BOOT, BOOT_SEED
    return out


def matrix_block(occ_obs, occ_pred, base, t0, last, death, year=MATRIX_YEAR, min_row=MIN_ROW_MATRIX):
    """第一次访视的分期（行）-> 第 year 年的状态（列：3 个分期 + 死亡），行归一化。
    实际一侧：死亡比例用 Kaplan–Meier（对称删失：最后一次访视之后 1.5 年以上才死的人删失在最后一次访视），
    活着的那部分按"第 year 年还活着、还在随访"的人的分期分配。直接数"状态已知"的人会高估死亡：死了总算已知，
    活着的人会失访。模拟一侧：这一行的所有人的平均概率（在 4 个已知状态上归一化）。
    一行少于 min_row 人、或活着且已知的少于 10 人，就不报这一行；人数只给四舍五入到 10 的近似值。"""
    obs, pred, n_approx = [], [], []
    grid = np.array([GRID * year], float)
    for b in range(3):
        m = base == b
        so = occ_obs[m, year]
        alive_known = (so >= 0) & (so <= 2)
        if m.sum() < min_row or alive_known.sum() < 10:
            obs.append(None)
            pred.append(None)
            n_approx.append(None)
            continue
        ev = np.isfinite(death[m])
        p_dead = float(_aj(t0[m], np.where(ev, death[m], last[m]), ev.astype(int), last[m], grid, "km")[0])
        living = np.array([np.mean(so[alive_known] == s_) for s_ in range(3)])
        obs.append(list(living * (1.0 - p_dead)) + [p_dead])
        pk = occ_pred[m, year, :4].astype(float)
        mass = pk.sum(1)
        ok = mass > 0
        pred.append((pk[ok] / mass[ok, None]).mean(0))
        n_approx.append(int(round(int(m.sum()), -1)))
    return {"year": year, "observed": obs, "predicted": pred, "n_approx": n_approx,
            "observed_method": "death share = 1 - Kaplan-Meier at the cut (symmetric censoring); the living share is "
                               "split by the stages of people alive and still followed at the cut"}


def _aj(start, ev_t, ev_k, cens_t, grid, kind):
    """Aalen–Johansen 累计发生率（天，从 start 起算），在 grid 上取值。ev_k：1 = 目标事件，2 = 竞争事件，0 = 删失于 cens_t。
    kind = 'cif'（有竞争事件）或 'km'（1 − KM，没有竞争事件）。"""
    t = np.where(ev_k > 0, ev_t, cens_t) - start
    order = np.argsort(t, kind="stable")
    t, k = t[order], ev_k[order]
    n = len(t)
    S, F, out, j = 1.0, 0.0, [], 0
    times = np.unique(t[k > 0])
    at_risk = n - np.searchsorted(t, times, side="left")
    d1 = np.array([np.sum((t == x) & (k == 1)) for x in times])
    d2 = np.array([np.sum((t == x) & (k == 2)) for x in times])
    for g in grid:
        while j < len(times) and times[j] <= g:
            r = at_risk[j]
            if r > 0:
                F += S * d1[j] / r
                S *= 1.0 - (d1[j] + d2[j]) / r
            j += 1
        out.append(F if kind == "cif" else 1.0 - S)
    return np.array(out)


def incidence_block(traj, pids, at_risk, cum, n_mc):
    """AD、死亡从第一次访视起 0..10 年的累计发生率：观测（Aalen–Johansen，和第 6 节同一条对称删失：最后一次访视之后
    超过 1.5 年才死的人删失在最后一次访视；AD 以死亡为竞争事件）vs 模拟（同一批人的轨迹里首次事件的比例，
    AD 必须早于这条轨迹的死亡）。观测一侧另报按人 bootstrap 的 95% 区间。"""
    grid = GRID * np.arange(YEARS + 1)
    out = {"years": list(range(YEARS + 1))}
    rng = np.random.default_rng(BOOT_SEED)
    for q in ("ad", "death"):
        qi = Q.index(q)
        m = at_risk[:, qi]
        rows = [traj[str(int(p))] for p in pids[m]]
        t0 = np.array([r["t0"] for r in rows], float)
        last = np.array([r["last"] for r in rows], float)
        dth = np.array([np.nan if r["death"] is None else r["death"] for r in rows], float)
        dth = np.where(np.isfinite(dth) & (dth <= last + SYM_GAP_DAYS), dth, np.nan)
        if q == "ad":
            ad = np.array([np.nan if r["ad"] is None else r["ad"] for r in rows], float)
            first_ad = np.isfinite(ad) & ~(np.isfinite(dth) & (dth < ad))
            ev_t = np.where(first_ad, ad, dth)
            ev_k = np.where(first_ad, 1, np.where(np.isfinite(dth), 2, 0))
            kind = "cif"
        else:
            ev_t, ev_k, kind = dth, np.where(np.isfinite(dth), 1, 0), "km"
        ev_t = np.where(ev_k > 0, ev_t, last)
        obs = _aj(t0, ev_t, ev_k, last, grid, kind)
        bs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(t0), len(t0))
            bs.append(_aj(t0[i], ev_t[i], ev_k[i], last[i], grid, kind))
        bs = np.array(bs)
        sim = np.concatenate([[0.0], (cum[m, qi, :] / n_mc).mean(0)])
        out[q] = {"n": int(m.sum()), "observed": obs, "observed_lo": np.percentile(bs, 2.5, 0),
                  "observed_hi": np.percentile(bs, 97.5, 0), "simulated": sim}
    return out


def auc_block(name, pids, t0s, cum, at_risk, n_mc, min_show, split="val", direct_tag="s42"):
    """第一次访视这个站点上：轨迹数出来的首次越线比例 vs 直接读出（scores_trans/<name>__<direct_tag>.csv 的
    rank_<q>；没有 rank 列就用 risk_<q>_<H>y），同一批站点。只有 val 有直接读出的打分表。"""
    # split811 的 test 段在两分真值表的 train 里
    tt = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv" if split == "val" else "truth_trans_train.csv"))
    sp = os.path.join(RUN_DIR, "scores_trans", f"{name}__{direct_tag}.csv")
    sc = pd.read_csv(sp) if (split == "val" and os.path.exists(sp)) else None
    first = pd.DataFrame({"pid": pids.astype(int), "t": t0s.astype(int)})
    m = first.merge(tt, on=["pid", "t"], how="left", validate="one_to_one")
    assert m["visit_idx"].eq(0).all(), "第一次访视的站点在 truth_trans_val 里找不到（或不是第一个站点）"
    if sc is not None:
        cols = [c for c in sc.columns if c.startswith("rank_") or c.startswith("risk_")]
        m = m.merge(sc[["pid", "t"] + cols], on=["pid", "t"], how="left", validate="one_to_one")
    out, macro = {}, {}
    for qi, q in enumerate(Q):
        out[q] = {}
        for H in TRANS_HORIZONS:
            dcol = f"rank_{q}" if f"rank_{q}" in m.columns else f"risk_{q}_{H}y"
            direct = m[dcol].to_numpy(float) if dcol in m.columns else np.full(len(m), np.nan)
            y = m[label_col(q, H)].to_numpy()
            ok = (y >= 0) & at_risk[:, qi]
            if np.isfinite(direct).any():
                ok &= np.isfinite(direct)                      # 两边同一批站点
            yy = y[ok].astype(int)
            r = {"n": int(ok.sum()), "n_pos": int(yy.sum())}
            if r["n"] < min_show:
                r.update(rollout=np.nan, direct=np.nan)
            else:
                r["rollout"] = _auc(yy, cum[ok, qi, H - 1] / n_mc)
                r["direct"] = _auc(yy, direct[ok]) if np.isfinite(direct).any() else np.nan
            out[q][str(H)] = r
    for H in TRANS_HORIZONS:
        both = [(out[q][str(H)]["rollout"], out[q][str(H)]["direct"]) for q in Q
                if np.isfinite(out[q][str(H)]["rollout"]) and (sc is None or np.isfinite(out[q][str(H)]["direct"]))]
        macro[str(H)] = {"rollout": float(np.mean([b[0] for b in both])) if both else np.nan,
                         "direct": float(np.mean([b[1] for b in both])) if both else np.nan, "n_questions": len(both)}
    return out, macro, (sp if sc is not None else None)


def aggregate(name, dataset, ckpt, limit, n_mc_arg, split="val", direct_tag="s42", notes=None):
    """notes：给不是 Delphi 的生成器（models_code/run.py 的模型）换掉下面关于查询 token、分词的说明。"""
    from evaluate_transitions import MIN_SHOW
    rdir, sfx, pkg = _out_paths(name, limit, n_mc_arg, split)
    d = np.load(os.path.join(rdir, f"persons{sfx}.npz"))
    info = json.load(open(os.path.join(rdir, f"run_info{sfx}.json")))
    traj = json.load(open(os.path.join(RUN_DIR, f"traj_truth_{split}.json")))
    n_mc = int(d["n_mc"])
    pids, t0s = d["pid"], d["t0"]
    occ_p, cum, tr_p, base_p, at_risk = d["occ"], d["cum"], d["trans"], d["base"], d["at_risk"]
    obs = [observed(traj[str(int(p))]) for p in pids]
    occ_o = np.stack([o["occ"] for o in obs])              # (n, 2, 11)
    tr_o = np.stack([o["trans"] for o in obs])             # (n, 2, 4, 4)
    base_o = np.stack([o["base"] for o in obs])            # (n, 2)

    # 不变量（每次都查）：前缀里的基线 = 观测基线；每年的占比计数加起来 = n_mc；累计计数不降；对角为 0；死亡行为空
    idx = np.arange(len(pids))
    checks = {"occupancy_sums_to_n_mc": bool((occ_p.sum(-1) == n_mc).all()),
              "cum_nondecreasing": bool((np.diff(cum, axis=-1) >= 0).all() and (cum <= n_mc).all()),
              "zero_diagonals": bool((tr_p[:, :, np.arange(4), np.arange(4)] == 0).all()
                                     and (tr_o[:, :, np.arange(4), np.arange(4)] == 0).all()),
              "dead_row_empty": bool((tr_p[:, :, DEAD, :] == 0).all() and (tr_o[:, :, DEAD, :] == 0).all()),
              "baseline_mismatch": {fam: int((base_p[:, j] != base_o[:, j]).sum()) for j, fam in enumerate(FAMS)},
              "year0_state_mismatch": {fam: int((occ_p[idx, j, 0, occ_o[:, j, 0]] != n_mc).sum())
                                       for j, fam in enumerate(FAMS)}}
    print("checks", json.dumps(checks), flush=True)

    res = {"name": name, "applicable": True, "ckpt": ckpt, "dataset": dataset, "n_persons": int(len(pids)), "n_mc": n_mc,
           "horizon_years": YEARS, "start": "query", "minutes": info.get("wall_minutes"),
           "cpu_minutes": info.get("cpu_minutes"), "seed": info.get("seed"),
           "gen_tokens_per_trajectory": info.get("gen_tokens_per_trajectory"), "engine": info.get("engine"),
           "checks": checks, "min_show": MIN_SHOW}
    res["transition_matrix"] = {}
    for j, fam in enumerate(FAMS):
        b = transition_block(tr_o[:, j].sum(0), tr_p[:, j].sum(0) / n_mc, MIN_SHOW)
        b["states"] = list(STATES[fam])
        b["n_persons_with_baseline"] = int((base_o[:, j] >= 0).sum())
        res["transition_matrix"][fam] = b
    res["occupancy"] = {}
    for j, fam in enumerate(FAMS):
        b = occupancy_block(occ_o[:, j], occ_p[:, j], n_mc, MIN_SHOW)
        b["states"] = list(STATES[fam])
        res["occupancy"][fam] = b
    # fig4 / fig5 用的三块（本地按 JSON 画）：第 5 年的转移矩阵、0..10 年的 MMSE 分期占比（含第 0 年）、AD / 死亡的累计发生率
    rows = [traj[str(int(p))] for p in pids]
    t0a = np.array([r["t0"] for r in rows], float)
    lasta = np.array([r["last"] for r in rows], float)
    dtha = np.array([np.nan if r["death"] is None else r["death"] for r in rows], float)
    dtha = np.where(np.isfinite(dtha) & (dtha <= lasta + SYM_GAP_DAYS), dtha, np.nan)
    res["matrix_5y"] = {fam: {**matrix_block(occ_o[:, j], occ_p[:, j], base_o[:, j], t0a, lasta, dtha),
                              "states": list(STATES[fam])} for j, fam in enumerate(FAMS)}
    res["stage_over_time"] = {fam: {**occupancy_block(occ_o[:, j], occ_p[:, j], n_mc, MIN_SHOW, k0=0),
                                    "states": list(STATES[fam])} for j, fam in enumerate(FAMS)}
    res["incidence"] = incidence_block(traj, pids, at_risk, cum, n_mc)
    res["brier"] = brier_block(occ_o[:, 0], occ_p[:, 0], base_o[:, 0], MIN_SHOW)
    res["brier"]["states"] = list(STATES["mmse"])
    res["brier_cogn"] = brier_block(occ_o[:, 1], occ_p[:, 1], base_o[:, 1], MIN_SHOW)
    res["brier_cogn"]["states"] = list(STATES["cogn"])
    res["auc_first_visit"], res["auc_first_visit_macro"], res["direct_scores"] = auc_block(
        name, pids, t0s, cum, at_risk, n_mc, MIN_SHOW, split, direct_tag)
    per_visit = (info.get("engine") or {}).get("mmse_per_visit")
    res["notes"] = [
        "10-year window from the first visit (t0); year-k cut = t0 + k*365 days (the data grid, same as the truth tables).",
        "Query-token start: prefix = every token at age <= t0, plus a No-event query token at t0 + 1 day; generation "
        "starts there (same position as the direct readout; removes the end-of-visit start artifact behind the stroke sign flip).",
        f"{n_mc} trajectories per person, seed = crn_seed(pid) + {SEED_OFFSET}, max_new_tokens = {MAX_NEW_TOKENS}; "
        "a trajectory stops at Death.",
        "Predicted state at year k: Dead if Death <= cut, else carry-forward of the last emitted stage token (prefix "
        "included), Unknown if none. B/C renormalise the prediction over the 4 known states; the raw Unknown mass is reported.",
        f"Observed state at year k: carry-forward of the last measured stage at or before the cut; known only if still "
        f"followed (last visit >= cut) or dead by the cut under symmetric censoring (a death counts only if it is within "
        f"{SYM_GAP_DAYS} days of the last visit; later deaths are treated as lost to follow-up after the last visit).",
        "Known-at-year-k conditions on follow-up: living persons lost to follow-up leave the denominator while deaths "
        "close to the last visit stay in, so the observed Dead share among the known overstates cohort mortality; the "
        "predicted side is the model's unconditional distribution for the same persons (it cannot know who will be lost).",
        "Transitions = next DISTINCT state (consecutive repeats collapsed) starting from the baseline stage. Predicted over "
        "(t0, t0+10y]; observed over (t0, min(t0+10y, last visit)] plus Dead under symmetric censoring. Predicted counts "
        "= expected counts (sum over persons of the per-person mean over trajectories). Dead is absorbing (empty row); "
        "r and MAE use the 9 off-diagonal cells of the 3 living rows. Observed follow-up stops at the last visit while "
        "trajectories always run 10 years; row normalisation removes the exposure difference only approximately.",
        "Brier: multiclass over the 4 states among known person-years with a measured baseline; carry-forward = point "
        f"mass on the baseline stage (0 if right, 2 if wrong). mean_diff = mean over years 1..10 of (Brier_model - "
        f"Brier_cf), negative = the model beats carry-forward; CI = person bootstrap (B={N_BOOT}, seed {BOOT_SEED}). "
        "brier = MMSE (as specified), brier_cogn = the same for global cognition.",
        "AUC from the first visit: rollout risk = fraction of trajectories with a first passage by t0 + H*365 (any target "
        "token after t0, strictly before that trajectory's Death; the death question counts Death itself); labels = "
        "truth_trans_val at the first-visit landmark (y >= 0 and at risk at t0); direct = rank_<q> of "
        "scores_trans/<name>__s42.csv on the same landmarks.",
        f"Privacy: observed-side denominators below {MIN_SHOW} are not reported (null), as in evaluate_transitions.",
    ]
    if notes is not None:
        res["notes"] = notes + res["notes"][4:]       # 换掉前 4 条（Delphi 的起步、生成、读出）；观测一侧和打分的说明照旧
    elif per_visit is False:
        res["notes"].append("Dedup tokenization: every token is emitted at most once per person, so a generated trajectory "
                            "cannot return to a stage it has already emitted (e.g. Normal -> Borderline -> Normal is not "
                            "expressible); recoveries in the observed matrix are structurally out of reach for this model.")
    os.makedirs(pkg, exist_ok=True)
    res = _f(res)
    with open(os.path.join(pkg, "trajectory.json"), "w") as f:
        json.dump(res, f, indent=1)
    tm = res["transition_matrix"]
    print(f"aggregated -> {pkg}\n  r/MAE mmse {tm['mmse']['pearson_r']}/{tm['mmse']['mae']}  cogn {tm['cogn']['pearson_r']}/"
          f"{tm['cogn']['mae']}  brier diff {res['brier']['mean_diff']} {res['brier']['mean_diff_ci']}  "
          f"auc macro {json.dumps(res['auc_first_visit_macro'])}", flush=True)
    return res


def write_na():
    """不能生成轨迹的 5 个 Delphi 打包盒：写一句为什么，免得读者以为漏跑了。"""
    todo = []
    for m in DELPHI:
        if m[0] not in GENERATIVE:
            todo.append((m[0], NA_REASON["visitheads"] if "visitheads" in m[4] else NA_REASON["snapshot"]))
    for name, reason in todo:
        pk = os.path.join(RUN_DIR, "packages", name)
        os.makedirs(pk, exist_ok=True)
        with open(os.path.join(pk, "trajectory.json"), "w") as f:
            json.dump({"name": name, "applicable": False, "reason": reason}, f, indent=1)
        print("not applicable:", name, flush=True)
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="registry.DELPHI 里能生成轨迹的模型名")
    ap.add_argument("--n_mc", type=int, default=N_MC)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 人（冒烟）")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--aggregate", action="store_true", help="只汇总")
    ap.add_argument("--write_na", action="store_true", help="给不能生成轨迹的 7 个打包盒写 applicable=false")
    args = ap.parse_args()
    if args.write_na:
        write_na()
        if not args.name:
            return
    reg = {m[0]: m for m in DELPHI}
    if args.name not in GENERATIVE:
        raise SystemExit(f"{args.name} 不能生成轨迹（或不在注册表里）；能生成的：{', '.join(GENERATIVE)}")
    dirs, dataset = reg[args.name][1], reg[args.name][2]
    ckpt = os.path.join(DELPHI_DIR, dirs[0], "ckpt.pt")
    if not args.aggregate:
        simulate_all(args, args.name, ckpt, dataset)
    aggregate(args.name, dataset, ckpt, args.limit, args.n_mc)


if __name__ == "__main__":
    main()
