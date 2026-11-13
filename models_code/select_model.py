"""select_model.py -- 选模型：transformer 在每一项评估上是"最好"、"并列最好"还是"落后"（RIS 上跑，读人级产物）。

    sbatch -p general-short --time=00:30:00 ris/py.sbatch select_model.py
    -> $ZOO_RUN_DIR/packages/_select/select.json（只有聚合数，可以拉回本地 -> reports/select_model.json）

规则（用户 2026-11-04 定、2026-11-08 确认用来选模型；这里的写法在跑之前定下，之后不改）：
  和同一份 val 数据上评估过的全部可用模型比（models.py 里能生成未来的全部 49 个 + 表 1 的 4 个 Delphi = 53 个；
  不可用的 36 个不比；delphi-dedup-sim 是 delphi-dedup 同一个 ckpt 的另一种读法，不单算）。
  每一项评估：按点估计排序；transformer 排第一 = 最好；否则和排在它前面的每个模型做按人配对 bootstrap，
  差值（transformer − 对手，方向统一成"正 = transformer 更好"）用 max-T 同时 95% 区间（对排在前面的这几个对手做多重比较校正）：
  有任何一个区间整个在 0 以下 = 落后，否则 = 并列最好。另报 Bonferroni 的百分位区间作敏感性，不改判定。
  每一项都是最好或并列最好，才算整体最好。各项之间不再校正（这样更不容易判成并列）。
  seed 方差：几乎所有模型都只有一个 seed，算不进去（transformer 重训一次，扰动 62 格的打分也会变）。

7 项评估 = 总览图 OVERVIEW.png 的 7 个面板（compare.overview_values）：
  1 auc2    第一部分 4 个终点 2 年 AUC 的平均（高好）                       metrics.json macro_auc["2y"]
  2 auc5    第三部分 6 道题 5 年 AUC 的平均（高好；没答全 6 道题的不比）     transitions.json macro["5y"]["auc"]
  3 calib   第一部分 4 个终点 2 年 |log(O/E)| 的平均（低好）                metrics.json endpoints[*_2y].o_e
  4 timing  AD、MMSE<24、死亡：时间预测的平均绝对误差 − 常数猜测的，三题平均（低好）  timing.json
  5 inc10   10 年 AD、死亡累计发生率：|模拟 − 实际| 两项平均（低好）          trajectory.json incidence
  6 gap5    5 年 MMSE 转移矩阵：模拟和实际平均每格差（低好）                 trajectory.json matrix_5y（report_sections.matrix_gap）
  7 pert    第三部分扰动 62 格按 5 年 log OR 打分，对的比例（高好）          perturb_v2.json（report_sections.pv2_logor_scores）
每项先用和打包盒 JSON 同一个定义重算点估计，和 JSON 对不上就停（断言），再做 bootstrap。

bootstrap：val 里的人重抽 B = 2000 次，一次抽样所有模型、所有评估共用（配对）；站点 / 轨迹 / 扰动记录跟着人走。
观测一侧（转移矩阵的实际比例、10 年实际发生率）也在同一次抽样里重算。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "unified_eval"))
import models as MZ  # noqa: E402
from common import ENDPOINTS, GRID, RUN_DIR  # noqa: E402
from registry import NOT_USABLE  # noqa: E402
from transitions import TRANS, label_col, risk_col  # noqa: E402

FOCUS = "transformer"
SECONDARY = ("transformer-gru-fill",)          # 只报，不参与判定
DELPHI = ("delphi-dedup", "delphi-dedup-nojitter", "delphi-nodedup", "delphi-nodedup-staticonly-split811")
POOL = [b[0] for b in MZ.BASELINES if b[0] not in NOT_USABLE] + list(DELPHI)
B = 2000
SEED = 20261108
TIMING_Q = ("ad", "mmse_lt24", "death")         # figures.PV_EP
OUT = os.path.join(RUN_DIR, "packages", "_select")
EVALS = {  # 名字 -> (方向：+1 高好 / −1 低好, 说明)
    "auc2": (+1, "Part 1: mean AUC of the 4 endpoints at 2 years"),
    "auc5": (+1, "Part 3: mean AUC of the 6 stage-crossing questions at 5 years"),
    "calib": (-1, "Part 1: mean |log(O/E)| of the 4 endpoints at 2 years"),
    "timing": (-1, "Timing: MAE minus constant-guess MAE (years), mean of AD, MMSE<24, death"),
    "inc10": (-1, "10-year cumulative incidence: mean |simulated - observed| of AD and death"),
    "gap5": (-1, "5-year MMSE transition matrix: mean absolute cell difference"),
    "pert": (+1, "Part 3 perturbation: share of the 62 pre-specified cells moving the right way (5-year log OR)"),
}


def _j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pkg(name, f):
    return _j(os.path.join(RUN_DIR, "packages", name, f))


def wauc(s, y, W):
    """加权 Mann–Whitney AUC（并列算一半），W (k, n) 每行一组权重 -> (k,)。权重 = 重抽样次数时等于重抽样后的 AUC。"""
    o = np.argsort(s, kind="mergesort")
    s, y, Wo = s[o], y[o].astype(float), W[:, o]
    st = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    pos = np.add.reduceat(Wo * y, st, axis=1)
    neg = np.add.reduceat(Wo * (1 - y), st, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def check(tag, mine, ref, tol):
    if ref is None or mine is None or not np.isfinite(mine) or abs(mine - ref) > tol:
        raise AssertionError(f"{tag}: recomputed {mine} vs package {ref} (tol {tol})")


# ============================================================================ 1–3：打分表上的 AUC 和校准
def part1(names, WP, pos):
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    Wl = WP[:, [pos[int(p)] for p in truth.pid]]                       # (B, 站点)
    pt, bt = {"auc2": {}, "calib": {}}, {"auc2": {}, "calib": {}}
    for m in names:
        M = pkg(m, "metrics.json")
        sc = truth[["pid", "t"]].merge(pd.read_csv(os.path.join(RUN_DIR, "scores", f"{m}__{M['main_seed']}.csv")),
                                       on=["pid", "t"], how="left", validate="one_to_one")
        a_pt, a_bt, c_pt, c_bt = [], [], [], []
        for e in ENDPOINTS:
            y = truth[f"y_{e}_2y"].to_numpy()
            k = y >= 0
            p = sc[f"risk_{e}_2y"].to_numpy(float)
            rc = next((c for c in (f"rank_{e}", f"logit_{e}") if c in sc.columns), None)
            r = sc[rc].to_numpy(float) if rc else p
            ref = M["endpoints"][f"{e}_2y"]
            a = wauc(r[k], y[k], np.ones((1, k.sum())))[0]
            check(f"{m} auc {e}", a, ref["auc"], 1e-9)
            a_pt.append(a)
            a_bt.append(wauc(r[k], y[k], Wl[:, k]))
            oe = y[k].sum() / p[k].sum()
            check(f"{m} o_e {e}", oe, ref["o_e"], 1e-9)
            c_pt.append(abs(np.log(oe)))
            c_bt.append(np.abs(np.log((Wl[:, k] @ y[k]) / (Wl[:, k] @ p[k]))))
        check(f"{m} macro auc2", float(np.mean(a_pt)), M["macro_auc"]["2y"], 1e-9)
        pt["auc2"][m], bt["auc2"][m] = float(np.mean(a_pt)), np.mean(a_bt, 0)
        pt["calib"][m], bt["calib"][m] = float(np.mean(c_pt)), np.mean(c_bt, 0)
    return pt, bt


def part3(names, WP, pos):
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv"))
    Wl = WP[:, [pos[int(p)] for p in truth.pid]]
    pt, bt, skip = {}, {}, {}
    for m in names:
        T = pkg(m, "transitions.json")
        ref = ((T.get("macro") or {}).get("5y") or {}).get("auc")
        if not T.get("answers_all") or ref is None:
            skip[m] = "does not answer all 6 questions at 5 years"
            continue
        sc = truth[["pid", "t"]].merge(pd.read_csv(os.path.join(RUN_DIR, "scores_trans", f"{m}__{T['main_seed']}.csv")),
                                       on=["pid", "t"], how="left", validate="one_to_one")
        a_pt, a_bt = [], []
        for q in TRANS:
            y = truth[label_col(q, 5)].to_numpy()
            k = y >= 0
            r = sc[f"rank_{q}"].to_numpy(float) if f"rank_{q}" in sc.columns else sc[risk_col(q, 5)].to_numpy(float)
            a_pt.append(wauc(r[k], y[k], np.ones((1, k.sum())))[0])
            a_bt.append(wauc(r[k], y[k], Wl[:, k]))
        check(f"{m} macro auc5", float(np.mean(a_pt)), ref, 1e-9)
        pt[m], bt[m] = float(np.mean(a_pt)), np.mean(a_bt, 0)
    return pt, bt, skip


# ============================================================================ 4：时间预测（timing.py 的同一套函数）
def timing(names, WP, pos):
    import timing as TM
    from rollout_trans import at_risk_flags
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    pids = np.array(sorted(traj, key=int), dtype=np.int64)
    t0s = np.array([int(traj[str(p)]["t0"]) for p in pids], dtype=np.int64)
    risk = np.array([at_risk_flags(traj[str(p)]) for p in pids], bool)
    v0 = [traj[str(p)]["visits"][0] for p in pids]
    Q = list(TRANS)
    for qi, q in enumerate(Q):                                          # 和 timing.run 一样：第一次访视没测这一族的不算
        fam = TRANS[q][1]
        if fam != "event":
            risk[:, qi] &= np.array([(v[1] if fam == "mmse" else v[2]) is not None for v in v0])
    obs = {q: np.array([TM.observed_time(traj[str(p)], q) for p in pids]) for q in TIMING_Q}
    Wp = WP[:, [pos[int(p)] for p in pids]]
    pt, bt, skip = {}, {}, {}
    for m in names:
        J = pkg(m, "timing.json")
        if not J or not J.get("applicable") or any("mae" not in J["questions"].get(q, {}) for q in TIMING_Q):
            skip[m] = "no timing for all of AD, MMSE<24, death"
            continue
        C, _, _ = TM.curves(m, pids, t0s)
        g_pt, g_bt = [], []
        for q in TIMING_Q:
            qi = Q.index(q)
            o = obs[q]
            p = TM.predicted_time(np.nan_to_num(C[:, qi]))
            p[~np.isfinite(C[:, qi]).all(1)] = np.nan
            ok = risk[:, qi] & np.isfinite(o) & np.isfinite(p)
            oo, pp, w = o[ok], p[ok], Wp[:, ok]
            c = float(np.median(oo))
            mae, cmae = float(np.abs(pp - oo).mean()), float(np.abs(c - oo).mean())
            check(f"{m} timing mae {q}", mae, J["questions"][q]["mae"], 6e-4)
            check(f"{m} timing const {q}", cmae, J["questions"][q]["constant_mae"], 6e-4)
            g_pt.append(mae - cmae)
            g_bt.append((w @ np.abs(pp - oo) - w @ np.abs(c - oo)) / w.sum(1))
        pt[m], bt[m] = float(np.mean(g_pt)), np.mean(g_bt, 0)
    return pt, bt, skip


# ============================================================================ 5–6：10 年轨迹（rollout_trans 的同一套定义）
def trajectories(names, WP, pos):
    import rollout_trans as RT
    from report_sections import matrix_gap
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    have = [m for m in names if os.path.exists(os.path.join(RUN_DIR, "rollout_trans", m, "persons.npz"))]
    skip = {m: "no person-level trajectories (rollout_trans/<name>/persons.npz)" for m in names if m not in have}
    ref = np.load(os.path.join(RUN_DIR, "rollout_trans", FOCUS, "persons.npz"))
    pids, at_risk = ref["pid"], ref["at_risk"]
    n = len(pids)
    rows = [traj[str(int(p))] for p in pids]
    ob = [RT.observed(r) for r in rows]
    occ_o = np.stack([o["occ"] for o in ob])[:, 0]                     # MMSE 这一族 (n, 11)
    base = np.stack([o["base"] for o in ob])[:, 0]
    t0 = np.array([r["t0"] for r in rows], float)
    last = np.array([r["last"] for r in rows], float)
    dth = np.array([np.nan if r["death"] is None else r["death"] for r in rows], float)
    dth = np.where(np.isfinite(dth) & (dth <= last + RT.SYM_GAP_DAYS), dth, np.nan)
    Y5 = RT.MATRIX_YEAR
    so = occ_o[:, Y5]
    alive = (so >= 0) & (so <= 2)
    ev = np.isfinite(dth)
    rows_ok = [b for b in range(3) if (base == b).sum() >= RT.MIN_ROW_MATRIX and (alive & (base == b)).sum() >= 10]
    # 10 年发生率的观测一侧（incidence_block 的同一条口径）
    qa, qd = RT.Q.index("ad"), RT.Q.index("death")
    ad = np.array([np.nan if r["ad"] is None else r["ad"] for r in rows], float)
    first_ad = np.isfinite(ad) & ~(ev & (dth < ad))
    inc = {"ad": (at_risk[:, qa], np.where(first_ad, ad, np.where(ev, dth, last)), np.where(first_ad, 1, np.where(ev, 2, 0)), "cif"),
           "death": (at_risk[:, qd], np.where(ev, dth, last), ev.astype(int), "km")}
    g10, g5 = np.array([GRID * 10.0]), np.array([GRID * float(Y5)])
    Wn = WP[:, [pos[int(p)] for p in pids]].astype(np.int64)            # (B, n) 重抽样次数

    def observed_side(cnt):
        ii = np.repeat(np.arange(n), cnt)
        mat = []
        for b in rows_ok:
            mi = ii[base[ii] == b]
            p_dead = float(RT._aj(t0[mi], np.where(ev[mi], dth[mi], last[mi]), ev[mi].astype(int), last[mi], g5, "km")[0])
            ak = mi[alive[mi]]
            living = np.array([np.mean(so[ak] == s) for s in range(3)])
            mat.append(np.r_[living * (1 - p_dead), p_dead])
        o_inc = {}
        for q, (m_, et, ek, kind) in inc.items():
            mi = ii[m_[ii]]
            o_inc[q] = float(RT._aj(t0[mi], et[mi], ek[mi], last[mi], g10, kind)[0])
        return np.array(mat), o_inc

    om_pt, oi_pt = observed_side(np.ones(n, np.int64))
    obs_b = [observed_side(Wn[b]) for b in range(B)]
    OMb = np.stack([x[0] for x in obs_b])                               # (B, 行, 4)
    OIb = {q: np.array([x[1][q] for x in obs_b]) for q in inc}
    pt, bt = {"gap5": {}, "inc10": {}}, {"gap5": {}, "inc10": {}}
    for m in have:
        d = np.load(os.path.join(RUN_DIR, "rollout_trans", m, "persons.npz"))
        assert (d["pid"] == pids).all() and (d["at_risk"] == at_risk).all(), m
        J = pkg(m, "trajectory.json") or {}
        nmc = float(d["n_mc"])
        pk = d["occ"][:, 0, Y5, :4].astype(float)
        mass = pk.sum(1)
        okp = mass > 0
        P = np.where(okp[:, None], pk / np.where(okp, mass, 1)[:, None], 0.0)
        pred_pt, pred_bt = [], []
        for b in rows_ok:
            k = (base == b) & okp
            pred_pt.append(P[k].mean(0))
            pred_bt.append((Wn[:, k] @ P[k]) / Wn[:, k].sum(1)[:, None])
        gap_pt = float(np.abs(om_pt - np.array(pred_pt)).mean())
        Mj = (J.get("matrix_5y") or {}).get("mmse")
        check(f"{m} gap5", gap_pt, matrix_gap(Mj)[0] if Mj else None, 5e-6)
        pt["gap5"][m] = gap_pt
        bt["gap5"][m] = np.abs(OMb - np.stack(pred_bt, 1)).mean((1, 2))
        i_pt, i_bt = [], []
        for q, qi in (("ad", qa), ("death", qd)):
            k = inc[q][0]
            s = d["cum"][k, qi, YEARS_IDX] / nmc
            sim_pt = float(s.mean())
            check(f"{m} sim10 {q}", sim_pt, J["incidence"][q]["simulated"][-1], 2e-6)
            check(f"{m} obs10 {q}", oi_pt[q], J["incidence"][q]["observed"][-1], 2e-6)
            i_pt.append(abs(sim_pt - oi_pt[q]))
            i_bt.append(np.abs((Wn[:, k] @ s) / Wn[:, k].sum(1) - OIb[q]))
        pt["inc10"][m], bt["inc10"][m] = float(np.mean(i_pt)), np.mean(i_bt, 0)
    return pt, bt, skip


YEARS_IDX = 9                                    # cum[:, q, 9] = 10 年


# ============================================================================ 7：扰动（perturb_v2 的同一套配对和资格）
def perturbation(names, WP, pos):
    import perturb_v2 as PV
    from report_sections import PV2_LOGOR_EPS, PV2_LOGOR_H, pv2_logor_scores
    H = int(PV2_LOGOR_H)
    C, _ = PV.cohort_table()
    pt, bt, skip = {}, {}, {}
    for m in names:
        p = os.path.join(PV.out_base(None), m, "direct.npz")
        J = pkg(m, "perturb_v2.json")
        if not os.path.exists(p) or not J or (J.get("direct") or {}).get("status") != "ok":
            skip[m] = "no direct perturbation readout"
            continue
        Z = np.load(p)
        F = np.clip(Z["F"].astype(np.float64), 1e-9, 1 - 1e-9)
        lor = np.log(F[..., H - 1]) - np.log1p(-F[..., H - 1])         # (seed, 记录, 6)
        idx = PV._index(Z["pid"], Z["arm"])
        pid_rec = Z["pid"]
        vals_pt, vals_bt = [], []
        for arm in PV.ARMS[1:]:
            if arm not in idx:
                continue
            for qi, q in enumerate(PV.EP):
                if q not in PV.prespec(arm) or q in PV.reached(arm):
                    continue
                ia, io = PV._pairs(C, idx, arm, q)
                if len(ia) < PV.MIN_N:
                    continue
                dl = (lor[:, ia, qi] - lor[:, io, qi]).mean(0)          # 每人的 seed 平均
                w = WP[:, [pos[int(x)] for x in pid_rec[ia]]]
                vals_pt.append(dl.mean())
                vals_bt.append((w @ dl) / w.sum(1))
        v, vb = np.array(vals_pt), np.array(vals_bt)
        ref, _ = pv2_logor_scores(J["direct"])
        mine = {"right": int((v > PV2_LOGOR_EPS).sum()), "wrong": int((v < -PV2_LOGOR_EPS).sum()), "n": len(v)}
        if (mine["right"], mine["wrong"], mine["n"]) != (ref["right"], ref["wrong"], ref["n_scored"]):
            raise AssertionError(f"{m} pert: recomputed {mine} vs package {ref}")
        pt[m] = mine["right"] / mine["n"]
        bt[m] = (vb > PV2_LOGOR_EPS).sum(0) / mine["n"]
    return pt, bt, skip


# ============================================================================ 判定
def standing(focus, better, pt, bt):
    s = {m: better * v for m, v in pt.items()}                          # 统一成高好
    order = sorted(s, key=lambda m: -s[m])
    above = [m for m in order if s[m] > s[focus] + 1e-12]
    out = {"focus": focus, "value": pt[focus], "rank": len(above) + 1, "n_models": len(s), "best": order[0],
           "best_value": pt[order[0]], "n_above": len(above)}
    if not above:
        out["verdict"] = "best"
        return out
    D = np.array([better * (bt[focus] - bt[m]) for m in above])        # (k, B)；正 = focus 更好
    d0 = np.array([s[focus] - s[m] for m in above])
    se = D.std(1, ddof=1)
    ok = se > 0
    c = float(np.quantile(np.max(np.abs((D[ok] - d0[ok, None]) / se[ok, None]), 0), 0.95)) if ok.any() else 0.0
    lo, hi = d0 - c * se, d0 + c * se
    a = 0.05 / len(above)
    blo, bhi = np.quantile(D, a / 2, 1), np.quantile(D, 1 - a / 2, 1)
    out.update(c_maxT=c, verdict="behind" if (hi < 0).any() else "tied_best",
               bonferroni_verdict="behind" if (bhi < 0).any() else "tied_best",
               vs_above=[{"model": m, "diff": float(d0[i]), "maxT_ci": [float(lo[i]), float(hi[i])],
                          "bonf_ci": [float(blo[i]), float(bhi[i])], "significant": bool(hi[i] < 0)}
                         for i, m in enumerate(above)])
    out["n_significantly_better"] = int((hi < 0).sum())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    persons = np.unique(truth.pid.to_numpy())
    pos = {int(p): i for i, p in enumerate(persons)}
    rng = np.random.default_rng(SEED)
    WP = np.stack([np.bincount(rng.integers(0, len(persons), len(persons)), minlength=len(persons))
                   for _ in range(B)]).astype(np.float64)             # (B, 人)，所有评估共用
    print(f"pool {len(POOL)} models, {len(persons)} persons, B {B}", flush=True)
    PT, BT, SKIP = {}, {}, {}
    p1, b1 = part1(POOL, WP, pos)
    PT.update(p1); BT.update(b1)
    print("part 1 done", flush=True)
    PT["auc5"], BT["auc5"], SKIP["auc5"] = part3(POOL, WP, pos)
    print("part 3 done", flush=True)
    PT["timing"], BT["timing"], SKIP["timing"] = timing(POOL, WP, pos)
    print("timing done", flush=True)
    t_pt, t_bt, t_skip = trajectories(POOL, WP, pos)
    PT.update(t_pt); BT.update(t_bt)
    SKIP["gap5"] = SKIP["inc10"] = t_skip
    print("trajectories done", flush=True)
    PT["pert"], BT["pert"], SKIP["pert"] = perturbation(POOL, WP, pos)
    print("perturbation done", flush=True)
    res = {"focus": FOCUS, "secondary": list(SECONDARY), "pool": POOL, "n_persons": int(len(persons)), "B": B, "seed": SEED,
           "rule": "best or tied-best (max-T simultaneous 95% CI vs every model ranked above includes 0) in every evaluation",
           "evaluations": {}}
    for k, (better, desc) in EVALS.items():
        r = {"description": desc, "higher_is_better": better > 0, "skipped": SKIP.get(k, {}),
             "ranking": [[m, PT[k][m]] for m in sorted(PT[k], key=lambda m: -better * PT[k][m])]}
        r["standing"] = {f: standing(f, better, PT[k], BT[k]) for f in (FOCUS,) + SECONDARY if f in PT[k]}
        res["evaluations"][k] = r
        st = r["standing"][FOCUS]
        print(f"{k}: transformer {st['value']:.4f} rank {st['rank']}/{st['n_models']}, best {st['best']} "
              f"{st['best_value']:.4f} -> {st['verdict']}", flush=True)
    v = [res["evaluations"][k]["standing"][FOCUS]["verdict"] for k in EVALS]
    res["overall"] = {"verdicts": dict(zip(EVALS, v)), "overall_best": all(x in ("best", "tied_best") for x in v)}
    json.dump(res, open(os.path.join(OUT, "select.json"), "w"), indent=1)
    print("overall best:", res["overall"], flush=True)


if __name__ == "__main__":
    main()
