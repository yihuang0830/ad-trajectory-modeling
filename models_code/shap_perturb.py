"""shap_perturb.py -- transformer（representative/transformer）的 SHAP + 扰动：SHAP 找出模型拿哪些变量放大 AD 速率，
再对这几个变量做扰动，看这个"放大"在 rollout 里是不是真的变成更多、更早的 AD，走的又是哪条路。

    python shap_perturb.py train                      # 同一份配方重训（run.py 没存模型）-> _runs/shap/transformer/model.pt；和打包盒对照
    python shap_perturb.py shap    [--perms 64]       # 人级 phi -> _runs/shap/transformer/phi_landmark.npz（只留 RIS）
    python shap_perturb.py perturb [--top 6] [--n_mc 128]   # 人级扰动读出 -> _runs/shap/transformer/perturb.npz（只留 RIS）
    python shap_perturb.py aggregate                  # -> _runs/packages/transformer/{shap.json, shap_perturb.json}（只有聚合数字）
    python shap_perturb.py plot                       # 本地：representative/transformer/ 的 fig9、fig10
  --limit N：只用队列前 N 人（冒烟），写到 _runs/shap/transformer/_limitN/，不覆盖正式结果。

谁、站在哪：和 shap_vars、perturb_v2 同一批人（>= 4 次访视，站点 = 第 4 次访视；perturb_v2.cohort_table），
每道题只看站点上还处在风险中的人。

1 SHAP（shap_vars 的 players / 拿掉的写法 / 置换抽样 + 反序，原样复用）
  读数 = 模型在站点上的下一步输出（确定的，不抽样）：
    ad、death     下一年内的概率 p，取 log(−log(1−p)) = log 累积风险率：exp(φ) = "有这个变量 ÷ 没有"时 AD（死亡）速率的倍数，
                  和 shap_delphi、shap_vars 同一种读法；
    mmse_next、cogn_next  明年那次访视 MMSE / 认知的预测均值（原始单位：MMSE 分、认知 z 分）——机制用：
                  变量是直接推高 AD 速率，还是先让模型预期的认知变差。
  年龄不当 player，另报"整段病史 +5 岁"（perturb_features.apply('age5')），同一把尺子。

2 扰动（SHAP 对 AD 的 mean|φ| 排前 --top 名的变量；--vars 可以直接指定）
  每个变量三种改法，全部从改过的站点往后模拟（每人 --n_mc 条，10 年），每个臂的随机数从同一个种子开始（只在第一年逐条对齐，见 rollout）：
    ko      拿掉这个变量（和 SHAP 同一种拿掉）：留一法的效应，和 Shapley 值对照
    dose    把站点上的值设成几档临床水平（DOSE 表；状态 / 曾经有过 = 0 / 1；APOE e4 0 / 1 / 2 个；教育 10 / 16 / 20 年）：
            看的是值还是"测过"，量效关系是不是单调
    direct  量效两头的那两档，各跑一遍"只改风险"的 rollout：风险读改过的历史，下一次访视按原样历史的分布抽
            （两个缓冲区，抽出的访视两边都接上）。总效应 − 直接效应 = 经由模拟出来的病程（认知、共病往后怎么走）的那部分。
  读出：站点上下一年 AD / 死亡的 log 速率（确定的）+ 模拟的 6 道题 × 1..10 年累计发生率（死亡是竞争风险，k / n_mc）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import RUN_DIR  # noqa: E402

MODEL = "transformer"
PKG = os.path.join(RUN_DIR, "packages", MODEL)
LOCAL_PKG = os.path.join(HERE, "..", "representative", MODEL)
SEED_ROLL = 20261029
N_BOOT = 1000
QS_SHAP = ("ad", "death", "mmse_next", "cogn_next")
QS_SIM = ("ad", "death", "mmse_lt24", "cogn_ltm1")       # 扰动汇总里报的模拟题
VL = 3                                                    # 站点 = 第 4 次访视（cohort_table 核对过）

# 量效的几档（原始单位，从低到高）。CRP / IL6 / TNFA 没有临床切点：用队列站点值的 1/6、1/2、5/6 分位（运行时算，写进 JSON）
DOSE_CONT = {"MMSE": (21, 25, 29), "COGN": (-1.5, -0.5, 0.5), "SBP": (110, 130, 160), "DBP": (65, 85, 95),
             "BMI": (20, 25, 32), "GLU": (90, 113, 150), "HBA1C": (5.3, 6.1, 7.5), "HDL": (35, 50, 70),
             "LDL": (80, 130, 180), "GFR": (45, 75, 100), "PSQI": (3, 8, 13), "APNEA": (0, 1),
             "CRP": "q", "IL6": "q", "TNFA": "q"}
DOSE_STATIC = {"APOE": (0, 1, 2), "EDU": (10, 16, 20), "SEX": (0, 1)}


def out_dir(limit):
    d = os.path.join(RUN_DIR, "shap", MODEL, f"_limit{limit}" if limit else "")
    os.makedirs(d, exist_ok=True)
    return d


def device():
    import torch
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0))))
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(dev):
    import torch
    import rollout_versions as RV
    import sequence as SQ
    m = RV.SeqARModel(MODEL, dev)
    # transformer 的编码器不读 x_mean（只有 GRU-D 用），给 0 就行
    m.net, _ = RV.build_seqsim(MODEL, np.zeros(SQ.N_OBS, np.float32), len(m.cols))
    ck = torch.load(os.path.join(out_dir(0), "model.pt"), map_location=dev)
    m.net.load_state_dict(ck["state"])
    m.net.to(dev).eval()
    m.info = ck["info"]
    return m


def cohort(Z, meta, limit):
    """-> 队列表 C（perturb_v2.cohort_table）、每人到站点为止的 (X, M, S, L)、两种 norm。"""
    import bl_data as B
    import perturb_v2 as PV
    import shap_vars as SL
    import simulator as SIM
    assert [c for c, *_ in SL.CONT] == meta["features"][:B.N_CONT], "连续量的顺序和 shap_vars.CONT 不一样"
    assert meta["static"][:20] == ["male", "educ_z", "educ_na", "race_white", "race_black", "race_other", "race_na",
                                   "hisp_yes", "hisp_na", "apoe_e4x1", "apoe_e4x2", "apoe_na", "apoe_e2carrier",
                                   "smoke_former", "smoke_current", "smoke_na", "alcohol_log1p", "alcohol_na",
                                   "study_MAP", "study_LATC"], "背景特征的顺序变了"
    C, _ = PV.cohort_table(limit)
    D = B.split(Z, "val")
    ii = C.lm_i.to_numpy()
    assert (C.vL == VL).all() and (D["pid"][ii] == C.index.to_numpy()).all()
    P = {"X": D["X"][ii, :VL + 1].copy(), "M": D["M"][ii, :VL + 1].copy(), "S": D["S"][ii].copy(),
         "L": D["L"][ii, :VL + 1].copy()}
    assert P["M"].all()
    return C, P, meta["norm"], SIM.norm_info(meta)


def step_out(m, X, M, S, norm_sim):
    """一批历史（都到站点 VL 为止）-> (n, 4)：ad、death 的 log 累积风险率，明年 MMSE、认知的预测均值（原始单位）。"""
    assert norm_sim["names"][:2] == ["cts_estmmse30", "cogn_global"]
    lc = lambda p: np.log(-np.log1p(-np.clip(p, 1e-9, 1 - 1e-9)))   # noqa: E731
    n = len(X)
    out = []
    for a in range(0, n, 4096):
        p_d, p_a, nd = m.outputs(X[a:a + 4096], M[a:a + 4096], S[a:a + 4096], None, np.full(min(4096, n - a), VL))
        mu = nd["mu"][:, :2] * norm_sim["sd"][:2] + norm_sim["mu"][:2]
        out.append(np.column_stack([lc(p_a), lc(p_d), mu]))
    return np.concatenate(out)


class DirectStepper:
    """"只改风险"的 rollout：风险读改过的历史，下一次访视按原样历史的分布抽；抽出的访视两边都接上。"""

    def __init__(self, model, start_p, start_o, n_mc):
        import rollout_versions as RV
        self.p = RV.SeqStepper(model, start_p, n_mc, model.chunk)
        self.o = RV.SeqStepper(model, start_o, n_mc, model.chunk)

    def init(self, n_mc):
        return {"p": self.p.init(n_mc), "o": self.o.init(n_mc)}

    def step(self, st, x, S, rows):
        p_d, p_a, _ = self.p.step(st["p"], x, S, rows)
        _, _, nd = self.o.step(st["o"], x, S, rows)
        return p_d, p_a, nd

    def update(self, st, rows, xn, year):
        self.p.update(st["p"], rows, xn, year)
        self.o.update(st["o"], rows, xn, year)


def rollout(m, P, n_mc, norm_sim, P_orig=None):
    """从站点往后模拟 10 年 -> (n, 6, 10) 累计发生率（k / n_mc）。P_orig 给了 = 只改风险（DirectStepper）。
    随机数每次都从 SEED_ROLL 起。只有第一年逐条对齐：simulate_visits 按活着的轨迹数抽随机数，
    任何一条轨迹的死亡在两臂之间不同之后，所有轨迹后面的随机数都错开（2026-10-31 查出）；差值里含模拟噪声。"""
    import bl_data as B
    import rollout_versions as RV
    import simulator as SIM
    n = len(P["X"])
    ii, vv = np.arange(n), np.full(n, VL)
    start = {"D": P, "ii": ii, "vv": vv}
    y0 = np.round(P["X"][:, VL, B.I_TF + 1] * 10.0)
    rng = np.random.default_rng(SEED_ROLL)
    if P_orig is None:
        stp, x0, s0 = RV.SeqStepper(m, start, n_mc, m.chunk), P["X"][:, VL], P["S"]
    else:
        stp = DirectStepper(m, start, {"D": P_orig, "ii": ii, "vv": vv}, n_mc)
        x0, s0 = P_orig["X"][:, VL], P_orig["S"]
    return SIM.risks(SIM.simulate_visits(stp, x0, s0, y0, n_mc, rng, norm_sim))


# ================================================================================ train


def cmd_train(a):
    import torch
    import bl_data as B
    import rollout_versions as RV
    dev = device()
    Z, meta = B.load()
    m = RV.SeqARModel(MODEL, dev).fit(B.split(Z, "train"), B.split(Z, "test"))   # 和 run.py 同一个调用（seed 0）
    torch.save({"state": m.net.state_dict(), "info": m.info}, os.path.join(out_dir(0), "model.pt"))
    ref = json.load(open(os.path.join(PKG, "baseline.json")))["log"][0]
    # 对照：队列站点上原样的模拟风险 vs 打包盒的 val 打分表（只差训练的随机性和抽样）
    C, P, _, norm_sim = cohort(Z, meta, 0)
    R = rollout(m, P, 100, norm_sim)
    sc = pd.read_csv(os.path.join(RUN_DIR, "scores_trans", f"{MODEL}__s0.csv")).set_index(["pid", "t"])
    sc = sc.loc[list(zip(C.index, C.tL))]
    chk = {}
    for q in ("ad", "death", "mmse_lt24", "cogn_ltm1"):
        for H in (2, 5, 10):
            x, y = R[:, B.Q.index(q), H - 1], sc[f"risk_{q}_{H}y"].to_numpy()
            chk[f"{q}_{H}y"] = {"pearson": round(float(np.corrcoef(x, y)[0, 1]), 4),
                                "mean_new": round(float(x.mean()), 4), "mean_package": round(float(y.mean()), 4)}
    info = {"device": dev, "train": m.info, "package_train": ref, "vs_package_scores": chk, "n_cohort": int(len(C))}
    json.dump(info, open(os.path.join(out_dir(0), "train.json"), "w"), indent=1, default=float)
    print(json.dumps(info, indent=1, default=float), flush=True)


# ================================================================================ SHAP


def cmd_shap(a):
    import bl_data as B
    import perturb_features as PF
    import shap_vars as SL
    Z, meta = B.load()
    C, P, norm, norm_sim = cohort(Z, meta, a.limit)
    m = load_model(device())
    PL = SL.players()
    f = lambda Xb, Mb, Sb, Lb: step_out(m, Xb, Mb, Sb, norm_sim)   # noqa: E731
    t0 = time.time()
    rec = {k: [] for k in ("phi", "base", "full", "eff", "age5", "raw_last")}
    for k, pid in enumerate(C.index):
        X, M, S, L = P["X"][k], P["M"][k], P["S"][k], P["L"][k]
        rng = np.random.default_rng(int(pid) % 2_147_483_647)
        phi, base, full, eff = SL.explain_person(f, PL, X, M, S, L, a.perms, rng)
        Xa, Sa, La = PF.apply("age5", X, M, S, L, VL, norm)
        age5 = f(Xa[None], M[None], Sa[None], None)[0] - f(X[None], M[None], S[None], None)[0]
        last = []
        for c, (col, *_r) in enumerate(SL.CONT):
            obs = np.where(X[:, B.N_CONT + c] < 0.5)[0]
            last.append(float(X[obs[-1], c] * norm["sd"][col] + norm["mu"][col]) if len(obs) else np.nan)
        for key, v in zip(rec, (phi, base, full, eff, age5, last)):
            rec[key].append(v)
        if (k + 1) % 50 == 0:
            print(f"  shap {k + 1}/{len(C)}  {time.time() - t0:.0f}s  max eff err {max(rec['eff']):.1e}", flush=True)
    d = out_dir(a.limit)
    risk = np.stack([C[f"risk__{q}"].to_numpy(bool) if f"risk__{q}" in C else np.ones(len(C), bool) for q in QS_SHAP], 1)
    np.savez_compressed(os.path.join(d, "phi_landmark.npz"), pid=C.index.to_numpy(), phi=np.stack(rec["phi"]),
                        base=np.stack(rec["base"]), full=np.stack(rec["full"]), age5=np.stack(rec["age5"]),
                        raw_last=np.array(rec["raw_last"]), eff=np.array(rec["eff"]), risk=risk,
                        players=np.array([p[0] for p in PL]), questions=np.array(QS_SHAP))
    info = {"model": MODEL, "n_persons": len(C), "perms": a.perms, "antithetic": True, "players": [p[0] for p in PL],
            "questions": list(QS_SHAP), "seeds": [0], "max_efficiency_err": float(max(rec["eff"])),
            "minutes": (time.time() - t0) / 60}
    json.dump(info, open(os.path.join(d, "run_landmark.json"), "w"), indent=1)
    print(json.dumps(info, indent=1), flush=True)


# ================================================================================ 扰动


def top_vars(d, k):
    """SHAP 对 AD（站点上处在风险中的人）的 mean|φ| 排前 k 的变量。"""
    Zs = np.load(os.path.join(d, "phi_landmark.npz"))
    m = Zs["risk"][:, 0]
    v = np.abs(Zs["phi"][m, :, 0]).mean(0)
    return [str(Zs["players"][j]) for j in np.argsort(-v)[:k]]


def dose_levels(var, P, norm):
    import shap_vars as SL
    from bl_data import I_MC
    if var in DOSE_STATIC:
        return list(DOSE_STATIC[var])
    fams = [fam for _, fam, *_r in SL.CONT]
    if var in fams:
        lv = DOSE_CONT[var]
        if lv == "q":
            c = fams.index(var)
            col = SL.CONT[c][0]
            x = P["X"][:, VL]
            ok = x[:, I_MC + c] < 0.5
            lv = tuple(round(float(q), 3) for q in np.quantile(x[ok, c] * norm["sd"][col] + norm["mu"][col], [1 / 6, 0.5, 5 / 6]))
        return list(lv)
    if var in [a for _, a in SL.STATE] + [a for _, a in SL.EVER]:
        return [0, 1]
    return []                                     # 种族、西语裔、吸烟、饮酒、基线即患病：只做拿掉


def apply_dose(var, level, X, S, norm):
    """一个人（到站点为止）-> 改过的 (X, S) 副本。连续量 / 用药：只改站点那次访视（记成测过）；
    累计的病（*_cum）、曾经有过：1 = 站点时有，0 = 到站点为止从没有过（之前访视的 1 一起清掉，历史才不自相矛盾）。"""
    import shap_vars as SL
    from bl_data import I_EVER, I_MC, I_MS, I_XS, N_CUM
    X, S = X.copy(), S.copy()
    fams = [fam for _, fam, *_r in SL.CONT]
    states = [a for _, a in SL.STATE]
    evers = [a for _, a in SL.EVER]
    if var in fams:
        c = fams.index(var)
        col = SL.CONT[c][0]
        X[VL, c] = (level - norm["mu"][col]) / norm["sd"][col]
        X[VL, I_MC + c] = 0.0
    elif var in states:
        k = states.index(var)
        if level == 0 and k < N_CUM:
            X[:, I_XS + k] = 0.0
        X[VL, I_XS + k], X[VL, I_MS + k] = float(level), 0.0
    elif var in evers:
        ix = I_EVER + evers.index(var)
        if level == 0:
            X[:, ix] = 0.0
        X[VL, ix] = float(level)
    elif var == "APOE":
        S[9], S[10], S[11] = float(level == 1), float(level == 2), 0.0
    elif var == "EDU":
        S[1], S[2] = (level - 16.0) / 4.0, 0.0
    elif var == "SEX":
        S[0] = float(level)
    else:
        raise KeyError(var)
    return X, S


def cmd_perturb(a):
    import bl_data as B
    import shap_vars as SL
    Z, meta = B.load()
    C, P, norm, norm_sim = cohort(Z, meta, a.limit)
    m = load_model(device())
    d = out_dir(a.limit)
    PL = SL.players()
    names = [p[0] for p in PL]
    vars_ = a.vars.split(",") if a.vars else top_vars(d, a.top)
    n = len(C)
    arms = {"orig": (P, None)}
    levels = {}

    def arm_of(change):
        Xs, Ss = zip(*[change(P["X"][k], P["S"][k], P["L"][k]) for k in range(n)])
        return {**P, "X": np.stack(Xs), "S": np.stack(Ss)}, None

    for v in vars_:
        keep = np.ones((1, len(PL)), bool)
        keep[0, names.index(v)] = False
        arms[f"ko:{v}"] = arm_of(lambda X, S, L: [r[0] for r in SL.remove(PL, keep, X, S, L)[:2]])
        levels[v] = dose_levels(v, P, norm)
        for lv in levels[v]:
            arms[f"dose:{v}:{lv}"] = arm_of(lambda X, S, L: apply_dose(v, lv, X, S, norm))
        pair = [f"dose:{v}:{levels[v][0]}", f"dose:{v}:{levels[v][-1]}"] if levels[v] else ["orig", f"ko:{v}"]
        for arm in pair:
            if arm != "orig":
                arms[f"direct|{arm}"] = (arms[arm][0], P)
    print(f"perturb: {n} persons, vars {vars_}, {len(arms)} arms, n_mc {a.n_mc}", flush=True)
    t0 = time.time()
    HZ = np.zeros((len(arms), n, 4))
    F = np.zeros((len(arms), n, B.NQ, B.NH), np.float32)
    for j, (arm, (Pa, Po)) in enumerate(arms.items()):
        HZ[j] = step_out(m, Pa["X"], Pa["M"], Pa["S"], norm_sim)
        F[j] = rollout(m, Pa, a.n_mc, norm_sim, Po)
        print(f"  {arm:28s} AD 5y {F[j, :, B.Q.index('ad'), 4].mean():.3f}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(d, "perturb.npz"), pid=C.index.to_numpy(), arms=np.array(list(arms)), HZ=HZ, F=F,
                        risk=np.stack([C[f"risk__{q}"].to_numpy(bool) for q in B.Q], 1), questions=np.array(B.Q))
    json.dump({"vars": vars_, "levels": levels, "n_mc": a.n_mc, "n_persons": n, "seed": SEED_ROLL,
               "minutes": (time.time() - t0) / 60}, open(os.path.join(d, "perturb_run.json"), "w"), indent=1)


# ================================================================================ 聚合


def _ci(x, rng):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [round(float(x.mean()), 5), round(float(np.quantile(b, 0.025)), 5), round(float(np.quantile(b, 0.975)), 5)]


def agg_perturb(d, phiZ):
    Zp = np.load(os.path.join(d, "perturb.npz"))
    run = json.load(open(os.path.join(d, "perturb_run.json")))
    arms = [str(x) for x in Zp["arms"]]
    Q = [str(x) for x in Zp["questions"]]
    HZ, F, risk = Zp["HZ"], Zp["F"], Zp["risk"]
    ia = {a: j for j, a in enumerate(arms)}
    o = ia["orig"]
    rng = np.random.default_rng(0)
    rad = risk[:, Q.index("ad")]
    rmtl = lambda Fq: Fq.sum(-1) - Fq[..., -1] / 2           # noqa: E731  10 年内平均"已发生"的年数（梯形；F(0) = 0）

    def arm_row(arm):
        j = ia[arm]
        r = {"arm": arm}
        for k, q in enumerate(("ad", "death")):
            m = risk[:, Q.index(q)]
            r[f"dloghaz_{q}"] = _ci(HZ[j, m, k] - HZ[o, m, k], rng)
        for q in QS_SIM:
            m = risk[:, Q.index(q)]
            qi = Q.index(q)
            r[q] = {"F5": round(float(F[j, m, qi, 4].mean()), 4), "F10": round(float(F[j, m, qi, 9].mean()), 4),
                    "dF5": _ci(F[j, m, qi, 4] - F[o, m, qi, 4], rng), "dF10": _ci(F[j, m, qi, 9] - F[o, m, qi, 9], rng),
                    "drmtl10": _ci(rmtl(F[j, m, qi]) - rmtl(F[o, m, qi]), rng), "n": int(m.sum())}
        return r

    # SHAP 和留一法对照：φ_j（有这个变量带来的 log 速率）vs f(原样) − f(拿掉它)
    P = [str(x) for x in phiZ["players"]]
    pid_s = phiZ["pid"]
    assert (pid_s == Zp["pid"]).all(), "SHAP 和扰动不是同一批人 / 顺序"
    out = {"settings": {"n_mc": run["n_mc"], "seed": run["seed"], "n_persons": run["n_persons"],
                        "readouts": "dloghaz = change in the log 1-year rate at the landmark (exp = rate ratio); "
                                    "dF5 / dF10 = change in simulated cumulative incidence at 5 / 10 years; "
                                    "drmtl10 = change in years lived with the event within 10 years (more = earlier). "
                                    "[mean, 2.5%, 97.5%] over persons at risk (bootstrap)",
                        "arms": "ko = variable removed (same as SHAP); dose = landmark value set to a level; "
                                "direct|<arm> = rate reads the changed history, next visits drawn as if unchanged"},
           "orig": arm_row("orig"), "variables": []}
    for v in run["vars"]:
        j = P.index(v)
        phi = phiZ["phi"][rad, j, 0]
        loo = HZ[o, rad, 0] - HZ[ia[f"ko:{v}"], rad, 0]
        rk = lambda x: np.argsort(np.argsort(x))              # noqa: E731
        V = {"variable": v, "shap_ad": {"mean_phi": round(float(phi.mean()), 5), "mean_abs_phi": round(float(np.abs(phi).mean()), 5)},
             "leave_one_out_ad": {"mean": round(float(loo.mean()), 5),
                                  "spearman_vs_phi": round(float(np.corrcoef(rk(phi), rk(loo))[0, 1]), 4)},
             "ko": arm_row(f"ko:{v}"), "levels": run["levels"][v],
             "dose": [dict(arm_row(f"dose:{v}:{lv}"), level=lv) for lv in run["levels"][v]]}
        lv = run["levels"][v]
        lo, hi = (f"dose:{v}:{lv[0]}", f"dose:{v}:{lv[-1]}") if lv else ("orig", f"ko:{v}")
        med = {"contrast": [lo, hi]}
        qi = Q.index("ad")
        for H in (5, 10):
            tot = F[ia[hi], rad, qi, H - 1] - F[ia[lo], rad, qi, H - 1]
            dlo = F[ia[f"direct|{lo}"], rad, qi, H - 1] if lo != "orig" else F[o, rad, qi, H - 1]
            dir_ = F[ia[f"direct|{hi}"], rad, qi, H - 1] - dlo
            t, dr = _ci(tot, rng), _ci(dir_, rng)
            med[f"ad_{H}y"] = {"total": t, "direct": dr, "via_course": _ci(tot - dir_, rng),
                               "share_via_course": round(1 - dr[0] / t[0], 3) if abs(t[0]) > 1e-3 else None}
        V["mediation"] = med
        out["variables"].append(V)
    return out


def cmd_aggregate(a):
    import shap_vars as SL
    d = out_dir(a.limit)
    SL.SHAP_DIR = os.path.dirname(d.rstrip("/")) if a.limit else os.path.join(RUN_DIR, "shap")
    J = SL.aggregate_model(os.path.basename(d.rstrip("/")) if a.limit else MODEL)
    J["name"] = MODEL
    J["settings"]["readout"] = ("next-step output at the 4th visit: ad / death = log(-log(1-p)) of the 1-year probability "
                                "(exp(phi) = rate ratio with vs without the variable); mmse_next / cogn_next = predicted mean of "
                                "next year's MMSE (points) / cognition (z)")
    J["settings"]["units"] = "ad, death: log rate ratio; mmse_next: MMSE points; cogn_next: cognition z units"
    pk = PKG if not a.limit else os.path.join(d, "_pkg")
    os.makedirs(pk, exist_ok=True)
    json.dump(J, open(os.path.join(pk, "shap.json"), "w"), indent=1)
    if os.path.exists(os.path.join(d, "perturb.npz")):
        phiZ = np.load(os.path.join(d, "phi_landmark.npz"))
        Pj = agg_perturb(d, phiZ)
        Pj["shap_rank_ad"] = [x["variable"] for x in J["questions"]["ad"]["variables"]]
        json.dump(Pj, open(os.path.join(pk, "shap_perturb.json"), "w"), indent=1)
    print(f"-> {pk}", flush=True)


# ================================================================================ 图（本地）


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import figures as FG
    FG._style()
    J = json.load(open(os.path.join(LOCAL_PKG, "shap.json")))
    FG.shap_summary(J, os.path.join(LOCAL_PKG, "fig9_shap.png"),
                    headline="What the transformer looks at to set its next-year AD rate",
                    groups=[("MMSE", ("impaired", "borderline", "normal")), ("COGN", ("low", "mid", "high")),
                            ("SBP", ("high", "elevated", "normal"))])
    p = os.path.join(LOCAL_PKG, "shap_perturb.json")
    if os.path.exists(p):
        fig_perturb(json.load(open(p)), os.path.join(LOCAL_PKG, "fig10_shap_perturb.png"))


VAR_LABEL = {"COGN": "Cognition (z)", "MMSE": "MMSE", "BMI": "BMI", "SMOKING": "Smoking", "ADRX": "AD drug",
             "ALCOHOL": "Alcohol", "SBP": "Systolic BP", "DBP": "Diastolic BP", "GLU": "Glucose", "HDL": "HDL",
             "HBA1C": "HbA1c", "APNEA": "Sleep-apnea risk", "APOE": "APOE ε4 copies", "EDU": "Education (years)",
             "SEX": "Male", "STATIN": "Statin", "HEART": "Heart disease", "HTN": "Hypertension", "DM": "Diabetes"}
MIN_VALUE_EFFECT = 0.01                 # 图 B 只画量效两头差 >= 1 个百分点（10 年 AD）的变量


def fig_perturb(Pj, path):
    """A：每个变量一行，10 年 AD 模拟风险——站点值设成各档（实心，越深值越高）、拿掉这个变量（空心圈，就是 SHAP 的"没有"）。
    点挤在一起 = 值不重要；空心圈离得远 = 模型在意的是"测没测过"。
    B：有量效的变量，两头那两档的差拆成"直接推高速率"和"经由模拟出来的病程"。"""
    import matplotlib.pyplot as plt
    import figures as FG
    FG._style()
    rank = {v: k + 1 for k, v in enumerate(Pj["shap_rank_ad"])}
    V = sorted(Pj["variables"], key=lambda x: rank[x["variable"]])
    base = 100 * Pj["orig"]["ad"]["F10"]
    n_risk = Pj["orig"]["ad"]["n"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 0.42 * len(V) + 2.2), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.5})
    y = np.arange(len(V))[::-1]
    for yi, v in zip(y, V):
        xs = [100 * d["ad"]["F10"] for d in v["dose"]]
        if xs:
            a.plot([min(xs), max(xs)], [yi, yi], color=FG.FAINT, lw=2, zorder=1)
            cols = FG.SHAP_BIN_COL if len(xs) == 3 else (FG.SHAP_BIN_COL[0], FG.SHAP_BIN_COL[2])
            for x, c in zip(xs, cols):
                a.scatter(x, yi, s=70, color=c, edgecolor=FG.SOFT, lw=0.6, zorder=3)
        a.scatter(100 * v["ko"]["ad"]["F10"], yi, s=70, facecolor="white", edgecolor=FG.INK, lw=1.4, zorder=2)
    a.axvline(base, color=FG.SOFT, lw=1, ls="--")
    a.text(base, len(V) - 0.35, f" as recorded {base:.0f}%", color=FG.SOFT, fontsize=9.5, va="bottom")
    a.set_yticks(y, [f"#{rank[v['variable']]}  {VAR_LABEL.get(v['variable'], v['variable'])}"
                     + (f"  ({' / '.join(f'{x:g}' for x in v['levels'])})" if v["levels"] else "") for v in V])
    a.set_ylim(-0.7, len(V) + 0.2)
    a.set_xlim(0, None)
    a.set_xlabel(f"simulated 10-year AD risk (%), {n_risk} people at risk")
    a.set_title("A  Set the value, or remove the variable", fontsize=12)
    a.scatter([], [], s=60, color=FG.SHAP_BIN_COL[1], edgecolor=FG.SOFT, label="value set at the 4th visit (darker = higher value)")
    a.scatter([], [], s=60, facecolor="white", edgecolor=FG.INK, lw=1.4, label="variable removed (what SHAP compares with)")
    a.legend(loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=9.5, handletextpad=0.3)
    # B
    rows = []
    for v in V:
        if not v["levels"]:
            continue
        m = v["mediation"]["ad_10y"]
        t, d, c = m["total"][0], m["direct"][0], m["via_course"][0]
        if abs(t) < MIN_VALUE_EFFECT:
            continue
        lo, hi = v["levels"][0], v["levels"][-1]
        worse, better = (hi, lo) if t > 0 else (lo, hi)
        s = 1 if t > 0 else -1
        rows.append((VAR_LABEL.get(v["variable"], v["variable"]) + f": {worse:g} vs {better:g}", 100 * s * d, 100 * s * c, 100 * s * t))
    yb = np.arange(len(rows))[::-1]
    for yi, (lab, d, c, t) in zip(yb, rows):
        b.barh(yi, d, height=0.6, color=FG.SHAP_BIN_COL[0], edgecolor="white", lw=2)
        b.barh(yi, c, left=d, height=0.6, color=FG.SHAP_BIN_COL[2], edgecolor="white", lw=2)
        b.text(max(t, d, d + c) + 0.8, yi, f"+{t:.0f} pts", va="center", fontsize=9.5, color=FG.INK)
    b.set_yticks(yb, [r[0] for r in rows])
    b.axvline(0, color=FG.SOFT, lw=1)
    b.set_xlim(min(0, min([r[1] for r in rows] + [r[1] + r[2] for r in rows])) - 1, max(r[3] for r in rows) * 1.25 + 2)
    b.set_xlabel("extra 10-year AD risk, worse vs better value (points)")
    b.set_title("B  Through which path?", fontsize=12)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=FG.SHAP_BIN_COL[0], label="direct: the changed value raises the AD rate"),
                      Patch(color=FG.SHAP_BIN_COL[2], label="via the simulated course (later visits)")],
             loc="upper left", bbox_to_anchor=(0, -0.13), fontsize=9.5)
    fig.suptitle("Do the variables SHAP flags move AD in the transformer's simulated futures?", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.07,
             f"One retrained transformer (same recipe as the package); {n_risk} people at risk of AD at the 4th visit, "
             f"{Pj['settings']['n_mc']} simulated futures each, every arm starts from the same random seed. #k = rank by SHAP for AD.\n"
             "A: continuous values and drugs are set at the 4th visit only; a disease set to 0 is removed from all earlier visits. "
             "B: direct = the AD rate reads the changed value but later visits are drawn as if it had not changed.",
             fontsize=9, color=FG.SOFT, ha="left", va="top")
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    p = sub.add_parser("shap")
    p.add_argument("--perms", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("perturb")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--vars", default="")
    p.add_argument("--n_mc", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("aggregate")
    p.add_argument("--limit", type=int, default=0)
    sub.add_parser("plot")
    a = ap.parse_args()
    {"train": cmd_train, "shap": cmd_shap, "perturb": cmd_perturb, "aggregate": cmd_aggregate, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
