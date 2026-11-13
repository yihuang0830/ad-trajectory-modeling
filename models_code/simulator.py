"""simulator.py -- 模型（Delphi 以外）的"生成未来"：按年往后走，每人 n_mc 条轨迹，交给 rollout_trans 的同一套汇总
（trajectory.json，fig4 / fig5 的数），和 Delphi 的轨迹逐项可比。

两种起点：
  第一次访视（每人 100 条，10 年）-> persons.npz -> rollout_trans.aggregate -> trajectory.json（第三部分第 12 节）
  每一个站点（val 全部站点）-> 从轨迹数出 1..10 年内首次越线 / AD / 死亡的比例 = 这个模型对未来的回答，
    写成和直接预测同一种打分表（第一部分、第三部分都用它打分）。生成模型除了模拟没有别的办法回答 H > 1 年的题。

逐访视的自回归模拟（ar-lr、ar-lgbm、minimalrnn 用）同一个步进规则：
  1. 当前访视 -> 模型给第 1 年的死亡、AD 风险；各抽一次，都发生时各抽一个年内时刻，AD 早于死亡才算。
  2. 活过这一年：从模型给的下一次访视分布抽一整次访视（连续量高斯 + 是否缺失；状态伯努利 + 是否缺失；卒中 / 抑郁新出现），
     累计类状态只增不减，MMSE 舍入到整数分、截在 0–30，时间特征走一年。
  3. 模型自己的历史表示（手工历史特征 / RNN 隐状态）用这次模拟的访视更新。
多状态 Markov 和联合模型不走访视向量，自己出同样格式的分期轨迹（generators.py）。
读出和 rollout_trans.readout 同一个口径。
"""
import json
import os

import numpy as np

import bl_data as B
from bl_data import I_AD, I_EVER, I_MC, I_MS, I_TF, I_XS, N_CONT, N_CUM, N_STATE
import rollout_trans as RT
from common import RUN_DIR
from transitions import TRANS

YEARS = RT.YEARS
AGE_STEP_Z = 365 / 365.25 / 10                # age_z = (年龄 − 80) / 10，网格每年 365 天
STAGE_CUTS = {"mmse": (24.0, 27.0), "cogn": (-1.0, 0.0)}


def stage_of(raw, fam):
    """原始值 -> 分期 0 = 最好 .. 2 = 最差。先舍到 4 位小数（float32 还原的 27 是 26.9999998，不舍入会落进 Borderline）。"""
    return 2 - np.digitize(np.round(raw, 4), STAGE_CUTS[fam])


class L2C:
    """data.l2c_features 的逐步版本（input_data.l2c_features 的逐步版）：每个连续量存计数和一次回归需要的几个和。
    特征顺序（每个变量 8 个）：最近值、是否测过、距最近一次测量的年数（封顶 10）、最近一次变化、最小、最大、OLS 斜率（截在 ±3）、log(1+次数)。"""

    def __init__(self, n):
        z = lambda: np.zeros((n, N_CONT))      # noqa: E731
        self.n, self.sy, self.sv, self.syy, self.syv = z(), z(), z(), z(), z()
        self.last, self.prev, self.ly = z(), z(), z()
        self.vmin, self.vmax = np.full((n, N_CONT), np.inf), np.full((n, N_CONT), -np.inf)

    def update(self, r, val, o, year):
        v = np.where(o, val, 0.0)
        y = np.broadcast_to(np.asarray(year, float).reshape(-1, 1), v.shape)
        new = [(self.prev, np.where(o, self.last[r], self.prev[r])), (self.last, np.where(o, v, self.last[r])),
               (self.ly, np.where(o, y, self.ly[r])), (self.vmin, np.where(o, np.minimum(self.vmin[r], v), self.vmin[r])),
               (self.vmax, np.where(o, np.maximum(self.vmax[r], v), self.vmax[r]))]
        for a, b in new:
            a[r] = b
        self.n[r] += o
        self.sy[r] += o * y
        self.sv[r] += v
        self.syy[r] += o * y * y
        self.syv[r] += v * y

    def features(self, year):
        n = self.n
        has = n > 0
        y = np.asarray(year, float).reshape(-1, 1)
        den = n * self.syy - self.sy ** 2
        ok = (n >= 2) & (den > 1e-9)
        slope = np.where(ok, (n * self.syv - self.sy * self.sv) / np.where(ok, den, 1.0), 0.0)
        f = np.stack([np.where(has, self.last, 0.0), has.astype(float),
                      np.where(has, np.minimum(y - self.ly, 10.0), 10.0),
                      np.where(n >= 2, self.last - self.prev, 0.0),
                      np.where(has, self.vmin, 0.0), np.where(has, self.vmax, 0.0),
                      np.clip(slope, -3, 3), np.log1p(n)], -1)
        return f.reshape(len(n), -1).astype(np.float32)

    def take(self, idx):
        o = L2C(0)
        for k, v in self.__dict__.items():
            setattr(o, k, v[idx].copy())
        return o


def l2c_replay(X, M, upto):
    """站点起点的手工历史特征状态：把每个站点之前（含）的访视按顺序喂一遍。X (n, V, 54)，upto (n,) 每行回放到第几次访视。"""
    n = len(X)
    st = L2C(n)
    for v in range(int(upto.max()) + 1):
        rows = np.where(upto >= v)[0]
        yr = np.round(X[rows, v, I_TF + 1] * 10.0)
        st.update(rows, X[rows, v, :N_CONT].astype(float), X[rows, v, I_MC:I_XS] < 0.5, yr)
    return st


# ----------------------------------------------------------------------------- 逐访视的自回归模拟


def simulate_visits(stepper, x0, s0, y0, n_mc, rng, norm, years=YEARS, full_draws=False):
    """stepper：init(n_mc) -> 内部状态；step(state, x, S, rows) -> (p_death, p_ad, 下一次访视的分布)；
    update(state, rows, x_new, year)。x0 (n, 54) 起点访视，s0 (n, 25)，y0 (n,) 起点的随访年数。
    full_draws（2026-11-11，默认关，关着时逐位和原来一样）：每一步给全部 N 条轨迹都抽随机数、再按还活着的行取用。
    这样同一批起点、同一个种子的两次调用里，第 j 条轨迹每一步用的随机数一样（不管别的轨迹死没死）——
    扰动的各个臂这样调用就是"共用随机数"（CRN），臂之间的差只来自输入的改动。"""
    def U(rows_, *tail):
        return rng.random((N, *tail))[rows_] if full_draws else rng.random((len(rows_), *tail))

    def Z_(rows_, *tail):
        return rng.standard_normal((N, *tail))[rows_] if full_draws else rng.standard_normal((len(rows_), *tail))
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

    def record(k, rows, xv):
        for f, j in (("mmse", j_mmse), ("cogn", j_cogn)):
            ok = xv[:, I_MC + j] < 0.5
            raw = xv[:, j] * sd[j] + mu[j]
            stage[f][rows[ok], k] = stage_of(raw[ok], f)

    record(0, np.arange(N), x)
    n_visits = np.zeros(N, np.int64)
    for k in range(1, years + 1):
        rows = np.where(alive)[0]
        if not len(rows):
            break
        p_d, p_a, nd = stepper.step(st, x, S, rows)
        dies = U(rows) < p_d
        td = np.where(dies, k - 1 + U(rows), np.inf)
        can_ad = t_ad[rows] == np.inf
        gets = can_ad & (U(rows) < p_a)
        ta = np.where(gets, k - 1 + U(rows), np.inf)
        gets &= ta < td
        t_ad[rows[gets]] = ta[gets]
        t_death[rows] = td
        alive[rows[dies]] = False
        surv = ~dies
        r = rows[surv]
        if not len(r):
            break
        nd = {key: v[surv] for key, v in nd.items()}
        m_c = U(r, N_CONT) < nd["miss_c"]
        v_c = nd["mu"] + nd["sd"] * Z_(r, N_CONT)
        raw = np.clip(np.round(v_c[:, j_mmse] * sd[j_mmse] + mu[j_mmse]), 0.0, 30.0)
        v_c[:, j_mmse] = (raw - mu[j_mmse]) / sd[j_mmse]
        v_c = np.clip(v_c, -6.0, 6.0)
        m_s = U(r, N_STATE) < nd["miss_s"]
        v_s = (U(r, N_STATE) < nd["xs"]).astype(np.float32)
        v_s[:, :N_CUM] = np.maximum(v_s[:, :N_CUM], cum_known[r])
        cum_known[r] = np.where(m_s[:, :N_CUM], cum_known[r], v_s[:, :N_CUM])
        ever = np.maximum(x[r, I_EVER:I_AD], (U(r, I_AD - I_EVER) < nd["ever"]))
        xn = x[r].copy()
        xn[:, :N_CONT] = np.where(m_c, 0.0, v_c)
        xn[:, I_MC:I_XS] = m_c
        xn[:, I_XS:I_MS] = np.where(m_s, 0.0, v_s)
        xn[:, I_MS:I_EVER] = m_s
        xn[:, I_EVER:I_AD] = ever
        xn[:, I_AD] = np.maximum(xn[:, I_AD], (t_ad[r] <= k).astype(np.float32))
        xn[:, I_TF] += AGE_STEP_Z
        xn[:, I_TF + 1] = (yy[r] + k) / 10.0
        xn[:, I_TF + 2] = 1.0
        x[r] = xn
        stepper.update(st, r, xn, yy[r] + k)
        record(k, r, xn)
        n_visits[r] += 1
    rs = lambda a: a.reshape(n, n_mc, *a.shape[1:])            # noqa: E731
    return {"mmse": rs(stage["mmse"]), "cogn": rs(stage["cogn"]), "t_death": rs(t_death), "t_ad": rs(t_ad),
            "n_visits": rs(n_visits)}


# ----------------------------------------------------------------------------- 读出


def first_events(sim):
    """每条轨迹每道题的首次发生年（相对起点；inf = 10 年内没有）。threshold 题只在模拟的测量上算。"""
    out = {}
    for q in B.Q:
        fam = TRANS[q][1]
        if fam == "event":
            continue
        tgt = (1, 2) if q in ("mmse_lt27", "cogn_lt0") else (2,)
        hit = np.isin(sim[fam][..., 1:], tgt)
        out[q] = np.where(hit.any(-1), hit.argmax(-1) + 1, np.inf)
    out["ad"] = np.where(sim["t_ad"] > 0, sim["t_ad"], np.inf)
    out["death"] = sim["t_death"]
    return out


# 条件风险读法（run.py --cond，2026-10-29 加，量化用）：主口径的标签把"窗口内先死、没越线"的站点排除，
# 对应的风险是"没先死的条件下"越线的概率；默认的 k / n 是累计发生率（先死的轨迹算阴性），是竞争风险口径的答案。
COND = False


def risks(sim, smooth=False):
    """(n, n_mc, ...) 的轨迹 -> (n, 6, 10)：H 年内首次发生的比例。smooth：(k + 0.5) / (n + 1)（扰动打分要求不碰 0 / 1）。
    COND：死亡以外的题，分母换成 H 年内没有先死的轨迹数（越线一定早于死亡，所以"先死" = 死在 H 年内且没越线）。"""
    fe = first_events(sim)
    Hs = np.array(B.HS, float)
    k = np.stack([(fe[q][..., None] <= Hs).sum(1) for q in B.Q], 1)
    n_mc = sim["t_death"].shape[1]
    if COND and not smooth:
        dead = fe["death"][..., None] <= Hs
        n = np.stack([n_mc - (dead & (fe[q][..., None] > Hs)).sum(1) if q != "death" else
                      np.full((dead.shape[0], len(Hs)), n_mc) for q in B.Q], 1)
        return np.where(n > 0, k / np.maximum(n, 1), 0.0)
    return (k + 0.5) / (n_mc + 1) if smooth else k / n_mc


def readout(sim, i):
    """第 i 个人的 n_mc 条轨迹 -> occ (2, 11, 5)、cum (6, 10)、trans (2, 4, 4)、base (2)。"""
    td, ta = sim["t_death"][i], sim["t_ad"][i]
    n_mc = len(td)
    occ = np.zeros((2, YEARS + 1, 5), np.int32)
    trans = np.zeros((2, 4, 4), np.int32)
    base = np.full(2, -1, np.int64)
    for j, fam in enumerate(RT.FAMS):
        st = sim[fam][i]
        b = st[:, 0]
        assert (b == b[0]).all()
        base[j] = b[0]
        for k in range(YEARS + 1):
            past = np.where(st[:, :k + 1] >= 0, np.arange(k + 1), -1).max(1)
            s = np.where(past >= 0, st[np.arange(n_mc), np.maximum(past, 0)], RT.UNK)
            s[td <= k] = RT.DEAD
            occ[j, k] = np.bincount(s, minlength=5)
        if base[j] >= 0:
            for r in range(n_mc):
                prev = int(base[j])
                for x in [int(v) for v in st[r, 1:] if v >= 0] + ([RT.DEAD] if td[r] <= YEARS else []):
                    if x != prev:
                        trans[j, prev, x] += 1
                        prev = x
    fe = first_events({k: v[i:i + 1] for k, v in sim.items()})
    cum = np.stack([(fe[q][0][:, None] <= np.arange(1, YEARS + 1)[None, :]).sum(0) for q in B.Q]).astype(np.int32)
    return {"occ": occ, "cum": cum, "trans": trans, "base": base}


# ----------------------------------------------------------------------------- 起点


def norm_info(meta):
    names = meta["features"][:N_CONT]
    return {"names": names, "mu": np.array([meta["norm"]["mu"][c] for c in names]),
            "sd": np.array([meta["norm"]["sd"][c] for c in names])}


def first_visit_start(Z, split="val"):
    """traj_truth 里的人（按 pid 排序），起点 = 第一次访视。"""
    traj = json.load(open(os.path.join(RUN_DIR, f"traj_truth_{split}.json")))
    D = B.split(Z, split)
    idx = np.array([i for i, p in enumerate(D["pid"]) if str(int(p)) in traj])
    assert len(idx) == len(traj), "traj_truth 里有人不在数据里"
    idx = idx[np.argsort(D["pid"][idx])]
    t0 = D["T"][idx, 0]
    bad = [p for p, t in zip(D["pid"][idx], t0) if int(traj[str(int(p))]["t0"]) != int(t)]
    assert not bad, f"{len(bad)} 人的第一次访视和 traj_truth 对不上"
    return {"D": D, "ii": idx, "vv": np.zeros(len(idx), np.int64), "pid": D["pid"][idx], "t": t0, "traj": traj}


def landmark_start(Z, split="val"):
    D = B.split(Z, split)
    ii, vv = np.where(D["M"])
    return {"D": D, "ii": ii, "vv": vv, "pid": D["pid"][ii], "t": D["T"][ii, vv]}


def write_persons(name, sim, start, n_mc, info, split="val", path_n_mc=None):
    """第一次访视起点的轨迹 -> persons.npz + run_info.json（rollout_trans.aggregate 读）。
    path_n_mc：决定写到哪个目录用的条数（rollout_trans 把"不是 100 条"当冒烟；完整运行传 RT.N_MC，实际条数照样记在 npz 里）。"""
    R = [readout(sim, i) for i in range(len(start["pid"]))]
    rdir, sfx, _ = RT._out_paths(name, 0, path_n_mc or n_mc, split)
    os.makedirs(rdir, exist_ok=True)
    traj = start["traj"]
    nv = sim["n_visits"].sum(1)
    np.savez_compressed(os.path.join(rdir, f"persons{sfx}.npz"), pid=start["pid"].astype(int), t0=start["t"], n_mc=n_mc,
                        occ=np.stack([r["occ"] for r in R]), cum=np.stack([r["cum"] for r in R]),
                        trans=np.stack([r["trans"] for r in R]), base=np.stack([r["base"] for r in R]),
                        n_gen=nv, sec=np.full(len(R), info.get("wall_minutes", 0) * 60 / max(len(R), 1)),
                        at_risk=np.array([RT.at_risk_flags(traj[str(int(p))]) for p in start["pid"]], bool))
    info = dict(info, n_persons=len(R), n_mc=n_mc, years=YEARS, start="first visit",
                gen_tokens_per_trajectory=float(nv.mean() / n_mc))
    json.dump(info, open(os.path.join(rdir, f"run_info{sfx}.json"), "w"), indent=1, default=str)
    return rdir
