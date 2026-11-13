"""perturb_v2.py -- 第三部分的扰动 v2：SHAP 引导的反事实扰动（含组合），问"AD 发病、MMSE / 认知恶化、分期转换的
时间是不是**一起**提前"（10 年内）。所有模型共用这一份扰动定义、资格和读出。

    python perturb_v2.py direct    --name delphi-dedup [--limit 15] [--cpus 8]
    python perturb_v2.py direct    --all [--only a,b]          # 注册表里的全部 Delphi（每个 seed）
    python perturb_v2.py rollout   --name delphi-dedup [--arms orig,mmse_b,...] [--n_mc 64] [--limit 8] [--cpus 16]
    python perturb_v2.py aggregate [--only a,b] [--tag ...]

人级产物只留在 RIS：$ZOO_RUN_DIR/perturb_v2/<模型>/{direct.npz, direct_info.json, rollout.npz, rollout_info.json}
（--limit 的调试运行自动写到 perturb_v2/_limit<N>/，不覆盖正式结果）。
聚合（可以拷回本地）：$ZOO_RUN_DIR/packages/<模型>/perturb_v2.json、packages/PERTURB_V2_SUMMARY.json；图在本地按它画（figures.py 的 fig6）

为什么要 v2（fig4 只看单个扰动、2 年风险、4 个终点）：
  * 问的是"同步"：扰动让 AD 提前时，更早的里程碑（MMSE<27 / <24、认知 <0 / <−1）是不是也一起提前、先后顺序对不对。
    所以读 10 年内的整条风险曲线，用 RMST（10 年内平均无事件时间，年）当效应量，第三部分 6 道题一起看。
  * 组合：SHAP 显示高风险来自几个因素同时出现；单个扰动的效应可能被模型饱和或互相抵消，组合才是临床上真会出现的人。
  * 新发疾病（卒中、高血压、糖尿病、心脏病、抑郁）按**访视内的排序**插入，而不是 RD._insert 的"放在同龄 token 之后"；
    rollout 一律（包括原样臂）从站点后 1 天的 No-event 查询 token 出发。fig4 记录过的卒中反号就是"访视以卒中 token
    结尾、生成的第一步从它读"造成的（rollout_extra.py --query_start 的机制检验）。

两种读出：
  DIRECT  站点上的逐年风险 F_q(H)，H = 1..10。Delphi：站点后 1 天的查询 token，rate_q = Σ exp(logit)（q 的目标 token
          求和：指数竞赛里"任一个先发生"的速率可加），F = 1 − exp(−rate·H·365.25)——和 fig4 / score_delphi_trans
          同一条路径（同一个 pseudo_records / query_logits）。统一输入的模型由 models_code/perturb.py 读自己的风险，交给 aggregate_model。
          RMST_q(10) = S = 1 − F 在 0..10 年网格上的梯形积分（S(0) = 1；Delphi 另和闭式 (1−e^{−λT})/λ 对照）。
          效应 = 扰动 − 原样（同一人、同一 seed），先每个 seed 算、再按 seed 平均；CI = 对"每人的 seed 平均"做人级 bootstrap。
  ROLLOUT 生成模型（默认 delphi-dedup / nodedup / fullvisit，seed 42）：10 年轨迹，每人所有臂用同一串随机数
          crn_seed(pid)+3。每人 × 臂：各题逐年累计首次通过数（站点后第一个目标 token；轨迹在死亡处停，死亡是竞争风险）、
          逐年 MMSE / 认知分期占据（含前缀的前向填充）。汇总：合并累计发生率、ΔRMTL（RMTL = 10 − 无事件曲线的 RMST）、
          MMSE 恶化趋势（比站点时自己的分期更差的比例；死后按最后一个分期前向填充，避免存活选择）。
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
from common import DAYS_PER_YEAR, DELPHI_DIR, RUN_DIR, TRUTH_DATASET, label_names, load_bin, post_id, pre_id  # noqa: E402
from registry import DELPHI, seeds_of  # noqa: E402
from transitions import STAGE_BINS, TRANS  # noqa: E402

import torch  # noqa: E402  （numpy / pandas 必须先于 torch：RIS 上 libstdc++ 的加载顺序）

import perturb_features as PF  # noqa: E402
import rollout_delphi as RD  # noqa: E402
import score_delphi as SD  # noqa: E402
from run_rollout_delphi import _NameOnly, crn_seed  # noqa: E402
from score_delphi_trans import READ_TOKENS  # noqa: E402

# ================================================================================ 配置（唯一需要改的地方）
# 最终的扰动清单会按并行的 SHAP 结果再调；改这里就够了，其余代码都从这几张表推出来。
HORIZON = 10                        # 年
YEARS = tuple(range(1, HORIZON + 1))
DPY = DAYS_PER_YEAR                 # 引擎 / 模型时间用 365.25 天的年
EPS = 0.05                          # |Δ| < 0.05 年（约 18 天）= 没反应
MIN_N = 10                          # 一格（扰动 × 终点）至少这么多人才算"有定义"
N_BOOT = 500
BOOT_SEED = 20261014                # 每次 bootstrap 用自己的 rng（同一个种子）：结果和调用顺序无关
LOGOR_H = (2, 5, 10)
CRN_OFFSET = 3                      # rollout 随机数 crn_seed(pid)+3（fig4 用 +0、fig3 用 +1，避免和它们撞同一串）
COGN_LOW_VALUE = -1.5               # 特征空间：cogn_l 把站点的 cogn_global 设成 -1.5（Low 箱 < -1；-0.5 是 Mid，同 fig4）

# 原子扰动（都在站点 tL = 第 4 次访视）：名字 -> (短标签, 资格)。
#   资格 = perturb_cohort_val.csv 已有的 elig_* 列；"new" = 新发疾病：token 流里站点之前没有这个病
#   （<X>_PREVALENT / <X>_ONSET，或 STROKE / DEPRESSION），且统一输入的站点特征测了、为 0。
ATOMS = {
    "mmse_b": ("MMSE N->B (25)", "elig_mmse"),        # = fig4 的 mmse25
    "mmse_i": ("MMSE N->I (23)", "elig_mmse"),        # = fig4 的 mmse23
    "cogn_m": ("cogn H->M (-0.5)", "elig_cogn"),      # = fig4 的 cogn
    "cogn_l": ("cogn H->L (-1.5)", "elig_cogn"),
    "age5": ("age +5y", "elig_age"),                  # = fig4 的 age5；组合里永远最后做（它平移全部年龄和站点）
    "apoe": ("APOE e4 0->1", "elig_apoe"),            # = fig4 的 apoe
    "stroke": ("new stroke", "elig_stroke"),          # 临床含义同 fig4 的 stroke，但按排序插入
    "htn": ("new HTN", "new"),
    "dm": ("new diabetes", "new"),
    "heart": ("new heart dis.", "new"),
    "depression": ("new depression", "new"),
}
# 组合：资格 = 各原子资格的 AND
COMBOS = {
    "cog2": ("mmse_b", "cogn_m"),
    "cog2_age": ("mmse_b", "cogn_m", "age5"),
    "cog2_severe": ("mmse_i", "cogn_l"),
    "comorb3": ("htn", "dm", "heart"),
    "comorb3_stroke": ("htn", "dm", "heart", "stroke"),
    "cog2_comorb3": ("mmse_b", "cogn_m", "htn", "dm", "heart"),
    "apoe_cog": ("apoe", "mmse_b"),
}
# 组合里原子的施加顺序（固定，结果才可复现）：量表先改（去重分词里是在访视末尾追加），再按排序插入新发疾病，
# 背景（APOE）其次，年龄平移最后（它连站点一起平移）。
ATOM_ORDER = ("mmse_b", "mmse_i", "cogn_m", "cogn_l", "stroke", "htn", "dm", "heart", "depression", "apoe", "age5")
# 新发疾病：原子 -> (插入的 token, 站点之前出现过就不合格的 token, 统一输入的特征名)。特征缺失掩码 = "<特征>_missing"（若有）
NEW_ONSET = {
    "stroke": ("STROKE", ("STROKE",), "r_stroke_ever"),
    "htn": ("HTN_ONSET", ("HTN_PREVALENT", "HTN_ONSET"), "hypertension_cum"),
    "dm": ("DM_ONSET", ("DM_PREVALENT", "DM_ONSET"), "dm_cum"),
    "heart": ("HEART_ONSET", ("HEART_PREVALENT", "HEART_ONSET"), "heart_cum"),
    "depression": ("DEPRESSION", ("DEPRESSION",), "r_depres_ever"),
}
EP = tuple(TRANS)                   # mmse_lt27, mmse_lt24, cogn_lt0, cogn_ltm1, ad, death
COG_EP = ("mmse_lt27", "mmse_lt24", "cogn_lt0", "cogn_ltm1")
EP_SHORT = {"mmse_lt27": "MMSE<27", "mmse_lt24": "MMSE<24", "cogn_lt0": "cog<0", "cogn_ltm1": "cog<-1",
            "ad": "AD", "death": "death"}
# 扰动自己就让这个人越过了这条线 -> 这一臂上这道题没有定义（"首次越过"已经在站点发生）
REACHED = {"mmse_b": ("mmse_lt27",), "mmse_i": ("mmse_lt27", "mmse_lt24"),
           "cogn_m": ("cogn_lt0",), "cogn_l": ("cogn_lt0", "cogn_ltm1")}
# 预设方向（看结果之前写死；全部是"提前"）。组合 = 各原子预设的并集，再去掉 REACHED。
_DECLINE = ("ad", "death") + COG_EP
PRESPEC_ATOM = {
    "mmse_b": _DECLINE, "mmse_i": _DECLINE, "cogn_m": _DECLINE, "cogn_l": _DECLINE,   # 认知下降 -> AD、死亡、其余认知线
    "age5": EP,                                                                     # 老 5 岁 -> 6 道全部
    "stroke": ("death",) + COG_EP,                                                  # 卒中 -> 死亡 + 认知线（AD 不预设）
    "htn": ("death",), "dm": ("death",), "heart": ("death",),                       # 心血管代谢 -> 只预设死亡
    "apoe": ("ad",) + COG_EP,                                                       # APOE e4 -> AD + 认知线
    "depression": (),                                                               # 不预设、不计分
}
# rollout（能生成轨迹的模型；seed 42 的 ckpt）
ROLLOUT_MODELS = ("delphi-dedup", "delphi-nodedup", "delphi-fullvisit")
ROLLOUT_ARMS = ("orig", "mmse_b", "cogn_m", "age5", "stroke", "comorb3", "cog2", "cog2_age", "cog2_severe",
                "cog2_comorb3")
ROLLOUT_N_MC = 64
MAX_NEW_TOKENS = 400
CURVE_ARMS = ("cog2", "cog2_age", "cog2_severe")      # 图 C：MMSE 恶化曲线
# fig4 里定义完全相同的臂：Delphi 直接读出的 logit 必须和 rollout/<模型>/perturb_persons.csv 逐人相同（< 1e-4）
FIG4_SAME = {"orig": "orig", "mmse_b": "mmse25", "mmse_i": "mmse23", "cogn_m": "cogn", "age5": "age5", "apoe": "apoe"}
FIG4_DIFF = {"stroke": "stroke"}                      # 同一个临床含义、插入位置不同（排序 vs 访视末尾）：只报差多少
FIG4_TOK = {"ad": "AD_DX", "death": "Death", "mmse": "MMSE_impaired", "cogn": "COGN_low"}
# ================================================================================

ARMS = ("orig",) + tuple(ATOMS) + tuple(COMBOS)
FAMS = ("mmse", "cogn")
STATES = {"mmse": ("Normal", "Borderline", "Impaired", "Dead", "Unknown"),
          "cogn": ("High", "Mid", "Low", "Dead", "Unknown")}
assert set(ATOM_ORDER) == set(ATOMS) and set(PRESPEC_ATOM) == set(ATOMS) and set(NEW_ONSET) <= set(ATOMS)
assert all(a in ATOMS for c in COMBOS.values() for a in c)
assert set(ROLLOUT_ARMS) <= set(ARMS) and set(CURVE_ARMS) <= set(ARMS)


def atoms_of(arm):
    if arm == "orig":
        return ()
    return (arm,) if arm in ATOMS else tuple(COMBOS[arm])


def ordered(arm):
    return sorted(atoms_of(arm), key=ATOM_ORDER.index)


def reached(arm):
    return set(q for a in atoms_of(arm) for q in REACHED.get(a, ()))


def prespec(arm):
    return set(q for a in atoms_of(arm) for q in PRESPEC_ATOM[a]) - reached(arm)


def arm_label(arm):
    if arm == "orig":
        return "original"
    if arm in ATOMS:
        return f"{arm}: {ATOMS[arm][0]}"
    return f"{arm} ({'+'.join(COMBOS[arm])})"


def out_base(tag):
    return os.path.join(RUN_DIR, "perturb_v2", f"_{tag}" if tag else "")


def pkg_dir(name, tag):
    return os.path.join(out_base(tag), "_pkg", name) if tag else os.path.join(RUN_DIR, "packages", name)


def _j(x):
    """numpy 标量 / NaN -> JSON 能写的值。"""
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_j(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


# ================================================================================ 队列：资格 + 处在风险中（所有模型共用）

def load_landmark_data():
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    D = {k: Z[f"val_{k}"] for k in ("X", "M", "S", "L", "T", "pid")}
    feats = meta["features"]
    # 特征下标核对（perturb_features 里写死的那几个 + 新发疾病用到的），对不上就停
    assert feats.index("cts_estmmse30") == PF.I_MMSE and feats.index("cogn_global") == PF.I_COGN
    assert feats.index("r_stroke_ever") == PF.I_STROKE and feats.index("age_z") == PF.I_AGE
    for at, (_, _, f) in NEW_ONSET.items():
        assert f in feats, f"{at}: 统一输入没有特征 {f}"
    return D, meta


def cohort_table(limit=0):
    """扰动队列（>= 4 次访视、站点 = 第 4 次访视）× 每个臂的资格、每道题在站点上是否处在风险中、统一输入里的站点下标。
    资格只看数据（snapshot 的 token 流 + 统一输入的特征），不看模型 -> 所有模型比的是同一批人。"""
    c = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"))
    tt = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv"),
                     usecols=["pid", "t", "mmse_stage", "cogn_stage", "mmse_worst_so_far", "cogn_worst_so_far"])
    c = c.merge(tt, left_on=["pid", "tL"], right_on=["pid", "t"], how="left", validate="one_to_one")
    assert c.t.notna().all(), "有站点不在 truth_trans_val.csv 里"
    # 处在风险中：和 transition_truth 的"不计入"同一条规则（测过这一族、且到站点为止从没进过目标区间；AD 站点时还没诊断）
    risk = {"mmse_lt27": c.mmse_worst_so_far == 2, "mmse_lt24": c.mmse_worst_so_far >= 1,
            "cogn_lt0": c.cogn_worst_so_far == 2, "cogn_ltm1": c.cogn_worst_so_far >= 1,
            "ad": c.risk_ad.astype(bool), "death": pd.Series(True, index=c.index)}
    for q in EP:
        c[f"risk__{q}"] = risk[q].to_numpy(bool)
    # 新发疾病的资格（一）：token 流里站点之前（含站点那次访视）没有这个病。用 snapshot（每次访视重发已患疾病）；
    # 各分词里这些 token 的出现完全一致（val 里每个疾病 token 的人数六份数据逐个相同），运行时再逐模型核对。
    names = label_names(TRUTH_DATASET)
    pre = pre_id(names)
    s = load_bin(TRUTH_DATASET, "val")
    tl = dict(zip(c.pid.astype(int), c.tL.astype(int)))
    D, meta = load_landmark_data()
    feats = meta["features"]
    idx = {int(p): i for i, p in enumerate(D["pid"])}
    vL = np.array([int(np.where(D["T"][idx[int(p)]] == int(t))[0][0]) for p, t in zip(c.pid, c.tL)])
    assert (vL == 3).all(), "站点必须是统一输入里的第 4 次访视"
    c["vL"] = vL
    c["lm_i"] = [idx[int(p)] for p in c.pid]
    diag = {}
    for at, (tok, prior, f) in NEW_ONSET.items():
        m = np.isin(s[:, 2], [pre[n] for n in prior])
        first = pd.DataFrame(s[m][:, :2], columns=["pid", "age"]).groupby("pid").age.min().to_dict()
        tok_free = np.array([first.get(int(p), np.inf) > tl[int(p)] for p in c.pid])
        fi = feats.index(f)
        mk = feats.index(f"{f}_missing") if f"{f}_missing" in feats else None
        x = D["X"][c.lm_i.to_numpy(), vL]
        feat_ok = (x[:, fi] < 0.5) & ((x[:, mk] < 0.5) if mk is not None else True)
        if ATOMS[at][1] == "new":
            c[f"atom__{at}"] = tok_free & feat_ok
            diag[at] = {"token_free": int(tok_free.sum()), "feature_0": int(feat_ok.sum()), "eligible": int((tok_free & feat_ok).sum())}
        else:
            # 已有资格列（卒中 = elig_stroke）：照用；另记它和 token / 特征两条规则对不上的人数
            e = c[ATOMS[at][1]].astype(bool).to_numpy()
            c[f"atom__{at}"] = e
            diag[at] = {"eligible": int(e.sum()), "elig_but_token_present": int((e & ~tok_free).sum()),
                        "elig_but_feature_not_0": int((e & ~feat_ok).sum())}
    for at, (_, col) in ATOMS.items():
        if col != "new" and f"atom__{at}" not in c:
            c[f"atom__{at}"] = c[col].astype(bool).to_numpy()
    for arm in ARMS:
        e = np.ones(len(c), bool)
        for a in atoms_of(arm):
            e &= c[f"atom__{a}"].to_numpy(bool)
        c[f"elig__{arm}"] = e
    c = c.sort_values("pid").set_index("pid")
    if limit:
        c = c.iloc[:limit]
    diag["n_eligible"] = {a: int(c[f"elig__{a}"].sum()) for a in ARMS}
    diag["n_at_risk"] = {q: int(c[f"risk__{q}"].sum()) for q in EP}
    return c, diag


# ================================================================================ 扰动 -> token 流（Delphi）

def insert_rank(tr, ages, toks, t, name):
    """新发疾病 token 插到 t 那次访视里"它该在的位置"。数据里同龄 token 按模型 id 升序（六份分词都满足），
    唯一例外是 AD_DX：它排在量表之后、整次访视的末尾（val 里几乎都是最后一个，其余的后面只跟 Death / 病理）。
    所以排序键 = id，AD_DX 记成 Death 前一位。RD._insert 放在同龄 token 之后（访视末尾）：rollout 第一步从最后一个
    token 读，那正是卒中反号的来源；多层模型里"谁是访视最后一个 token"（只有它在 mask_ties 下看得到整次访视）也会变。"""
    T = tr.tok
    tok = T[name]
    key = toks.astype(np.float64)
    key[toks == T["AD_DX"]] = T["Death"] - 0.5
    same = np.flatnonzero(ages == t)
    if len(same):
        gt = same[key[same] > tok]
        pos = int(gt[0]) if len(gt) else int(same[-1]) + 1
    else:
        pos = int(np.searchsorted(ages, t, side="right"))
    return np.insert(ages, pos, float(t)), np.insert(toks, pos, int(tok))


def translate_tokens(tr, arm, ages, toks, tL, t_first):
    """臂 -> (ages, toks, landmark)。完整流（含站点之后），截断由调用方按 landmark 做。
    与 fig4 相同定义的原子直接调 RD.apply（逐位相同，回归核对靠这个）。新发疾病的人在这个模型自己的流里若已有
    这个病（资格表和这份分词对不上）-> 返回 None，调用方跳过并计数。"""
    a, k, tl = ages.copy(), toks.copy(), int(tL)
    for at in ordered(arm):
        if at in ("mmse_b", "mmse_i", "cogn_m", "apoe", "age5"):
            pert = {"mmse_b": "mmse25", "mmse_i": "mmse23", "cogn_m": "cogn"}.get(at, at)
            a, k, tl = RD.apply(tr, pert, a, k, tl, t_first)
        elif at == "cogn_l":
            a, k = RD._set_bin(tr, a, k, tl, RD.COGN_BINS, 0, tr.cogn_per_visit)
        elif at in NEW_ONSET:
            tok, prior, _ = NEW_ONSET[at]
            if np.isin(k[a <= tl], [tr.tok[n] for n in prior]).any():
                return None, None, None
            a, k = insert_rank(tr, a, k, tl, tok)
        else:
            raise KeyError(at)
    return a, k, tl


def stage_of(tr, toks, fam):
    """流里最后一个分箱 token 的箱号（0 最差 .. 2 最好）；没有 = -1。"""
    ids = [tr.tok[n] for n in STAGE_BINS[fam]]
    m = np.isin(toks, ids)
    return ids.index(int(toks[m][-1])) if m.any() else -1


# ================================================================================ 扰动 -> 特征（统一输入的模型，统一输入的特征空间）

def translate_features(arm, X, M, S, Lf, vL, norm, feats):
    """同一组临床含义在统一输入（models_code/input_data.py）的特征空间里的翻译（统一输入的模型用；改法同 perturb_features.apply）。"""
    for at in ordered(arm):
        if at in ("mmse_b", "mmse_i", "cogn_m", "apoe", "age5"):
            pert = {"mmse_b": "mmse25", "mmse_i": "mmse23", "cogn_m": "cogn"}.get(at, at)
            X, S, Lf = PF.apply(pert, X, M, S, Lf, vL, norm)
        elif at == "cogn_l":
            X = X.copy()
            X[vL, PF.I_COGN] = (COGN_LOW_VALUE - norm["mu"]["cogn_global"]) / norm["sd"]["cogn_global"]
            X[vL, PF.N_CONT + PF.I_COGN] = 0.0
            Lf = PF.l2c_from_x(X, M)
        elif at in NEW_ONSET:
            # 累计 / "曾经有过"：站点起每次访视都是 1（读的是站点；因果模型看不到之后的访视，设不设都一样）
            f = NEW_ONSET[at][2]
            X = X.copy()
            fi = feats.index(f)
            X[vL:, fi] = np.where(M[vL:], 1.0, X[vL:, fi])
            if f"{f}_missing" in feats:
                mi = feats.index(f"{f}_missing")
                X[vL:, mi] = np.where(M[vL:], 0.0, X[vL:, mi])
        else:
            raise KeyError(at)
    return X, S, Lf


# ================================================================================ DIRECT 读出

def delphi_F(lg):
    """lg: (n, len(READ_TOKENS)) 查询位置的 logit（每天的 log 速率）-> F (n, 6, 10)、log Σλ (n, 6)。
    和 score_delphi_trans 同一个公式：rank = logsumexp，risk = −expm1(−Σλ·H·365.25)。"""
    lse = np.stack([np.logaddexp.reduce(lg[:, [READ_TOKENS.index(t) for t in TRANS[q][2]]], axis=1) for q in EP], 1)
    H = np.array(YEARS, np.float64)
    F = -np.expm1(-np.exp(lse)[:, :, None] * H[None, None, :] * DPY)
    return F, lse


def rmst_from_F(F):
    """F (..., 10) 逐年累计风险 -> RMST（年）：S = 1 − F 在 0..10 年网格上的梯形积分，S(0) = 1。"""
    S = np.concatenate([np.ones(F.shape[:-1] + (1,)), 1.0 - F], -1)
    return 0.5 * (S[..., :-1] + S[..., 1:]).sum(-1)


def rmst_closed(lse):
    """Delphi 常数速率下的精确 RMST：∫0^T e^{−λt} dt = (1 − e^{−λT}) / λ（λ 每年）。"""
    lam = np.exp(lse) * DPY
    return np.where(lam > 1e-12, -np.expm1(-lam * HORIZON) / np.maximum(lam, 1e-300), HORIZON)


def direct_delphi(name, C, out_dir, cpus):
    reg = {m[0]: m for m in DELPHI}
    _, dirs, dataset, tokz, sw, _ = reg[name]
    info = {"name": name, "family": "delphi", "dataset": dataset, "tokenization": tokz, "switches": sw,
            "n_persons": len(C), "host": os.uname().nodename}
    if "visitheads" in sw:
        # lm_head 的 logit 在 visit_heads 下不是速率：F、RMST 都没有定义（AUC 只看排序，这里要的是时间）
        info["status"] = "undefined: visit_heads -- lm_head logits are not rates, so F(H) / RMST are undefined"
        json.dump(info, open(os.path.join(out_dir, "direct_info.json"), "w"), indent=1)
        print("!!", info["status"], flush=True)
        return info
    torch.set_num_threads(max(1, cpus))
    t0 = time.time()
    tr = _NameOnly(dataset)                      # 扰动翻译只需要词表和发射规则（snapshot 也能用）
    streams = RD.person_streams(dataset)
    rows, pseudo, lands, skipped = [], [], [], {}
    for pid, r in C.iterrows():
        ages, toks = streams[int(pid)]
        for arm in ARMS:
            if not r[f"elig__{arm}"]:
                continue
            a, k, tl = translate_tokens(tr, arm, ages, toks, int(r.tL), int(r.t_first))
            if a is None:
                skipped[arm] = skipped.get(arm, 0) + 1
                continue
            j = len(rows)
            m = a <= tl
            pseudo.append(np.stack([np.full(int(m.sum()), j), a[m].astype(np.int64), k[m] - 1], 1))
            lands.append((j, int(tl)))
            rows.append((int(pid), arm, int(tl)))
    info["skipped_prior_token_in_own_stream"] = skipped
    arr = np.concatenate(pseudo).astype(np.int64)
    land = pd.DataFrame(lands, columns=["pid", "t"])
    data, hist_len = SD.pseudo_records(arr, land)
    q_age = land.t.to_numpy() + 1
    post = post_id(label_names(dataset))
    tok_post = [post[n] for n in READ_TOKENS]
    seeds, Fs, LGs = [], [], []
    for d, s in zip(dirs, seeds_of(dirs)):
        ck = os.path.join(DELPHI_DIR, d, "ckpt.pt")
        if not os.path.exists(ck):
            print(f"!! 缺 ckpt {ck}，跳过", flush=True)
            continue
        model, ckd = SD.load_model(ck, "cpu")
        cfg = ckd.get("config", {})
        lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
        bsz = ckd["model_args"]["block_size"]
        pos_len = SD.pos_align_length(dataset, cfg, bsz, lr)[0] if ckd["model_args"].get("pos_embedding") else None
        lg = SD.query_logits(model, data, hist_len, q_age, bsz, tok_post, "cpu", lifestyle_range=lr, pos_len=pos_len)
        F, _ = delphi_F(lg.astype(np.float64))
        seeds.append(s)
        Fs.append(F)
        LGs.append(lg.astype(np.float64))
        print(f"  {name} s{s}: {len(rows)} 条伪记录  pos_len={pos_len}  {time.time() - t0:.0f}s", flush=True)
    pid_a = np.array([r[0] for r in rows], np.int64)
    arm_a = np.array([r[1] for r in rows])
    np.savez_compressed(os.path.join(out_dir, "direct.npz"), pid=pid_a, arm=arm_a,
                        tland=np.array([r[2] for r in rows], np.int64), seeds=np.array(seeds),
                        F=np.stack(Fs), logit=np.stack(LGs), read_tokens=np.array(READ_TOKENS))
    info["seeds"] = seeds
    info["n_records"] = len(rows)
    info["regression_fig4"] = regression_fig4(name, pid_a, arm_a, seeds, LGs)
    info["regression_scores_trans"] = regression_scores_trans(name, C, pid_a, arm_a, seeds, LGs)
    info["minutes"] = (time.time() - t0) / 60
    info["status"] = "ok"
    json.dump(_j(info), open(os.path.join(out_dir, "direct_info.json"), "w"), indent=1)
    rg = info["regression_fig4"]
    print(f"  {name}: fig4 回归 {rg.get('pass')} (max {rg.get('max_abs_same_arms')})；"
          f"卒中排序插入 vs 末尾 max {rg.get('stroke_max_abs')}；scores_trans {info['regression_scores_trans'].get('pass')}"
          f"  {info['minutes']:.1f} min", flush=True)
    bad = [k for k in ("regression_fig4", "regression_scores_trans") if info[k].get("pass") is False]
    if bad:
        raise SystemExit(f"{name}: 回归核对没过：{bad}（结果已写出，见 direct_info.json）")
    return info


def regression_fig4(name, pid_a, arm_a, seeds, LGs):
    """和 fig4 定义相同的臂（orig / mmse25 / mmse23 / cogn / age5 / apoe），logit 必须和 perturb_persons.csv 逐人相同。"""
    p = os.path.join(RUN_DIR, "rollout", name, "perturb_persons.csv")
    if not os.path.exists(p):
        return {"pass": None, "note": "没有 fig4 的 perturb_persons.csv"}
    old = pd.read_csv(p)
    df = pd.DataFrame({"pid": pid_a, "arm": arm_a, "j": np.arange(len(pid_a))})
    cells, same_max, stroke = {}, 0.0, []
    n_cmp = 0
    for s, lg in zip(seeds, LGs):
        for arm, pert in {**FIG4_SAME, **FIG4_DIFF}.items():
            a = df[df.arm == arm].merge(old[old.pert == pert], on="pid", how="inner")
            cols = [(e, tok) for e, tok in FIG4_TOK.items() if f"dir_s{s}_{e}" in a]
            if not len(a) or not cols:
                continue
            d = np.concatenate([np.abs(lg[a.j.to_numpy(), READ_TOKENS.index(tok)] - a[f"dir_s{s}_{e}"].to_numpy())
                                for e, tok in cols])
            cells[f"s{s}:{arm}"] = {"n": int(len(a)), "max_abs": float(d.max()), "mean_abs": float(d.mean())}
            if arm in FIG4_SAME:
                same_max = max(same_max, float(d.max()))
                n_cmp += len(d)
            else:
                stroke.append(d)
    st = np.concatenate(stroke) if stroke else np.zeros(0)
    return {"pass": bool(n_cmp > 0 and same_max < 1e-4), "max_abs_same_arms": same_max, "n_compared": n_cmp,
            "stroke_max_abs": float(st.max()) if len(st) else None,
            "stroke_mean_abs": float(st.mean()) if len(st) else None, "cells": cells}


def regression_scores_trans(name, C, pid_a, arm_a, seeds, LGs):
    """原样臂 = 第三部分打分表在站点 (pid, tL) 上的 6 个 logit（同一个查询位置）。"""
    out, worst = {}, 0.0
    for s, lg in zip(seeds, LGs):
        p = os.path.join(RUN_DIR, "scores_trans", f"{name}__s{s}.csv")
        if not os.path.exists(p):
            continue
        sc = pd.read_csv(p)
        o = pd.DataFrame({"pid": pid_a, "arm": arm_a, "j": np.arange(len(pid_a))})
        o = o[o.arm == "orig"].assign(t=lambda d: C.loc[d.pid, "tL"].to_numpy().astype(int))
        m = o.merge(sc, on=["pid", "t"], how="inner")
        d = max(float(np.abs(lg[m.j.to_numpy(), i] - m[f"logit_{t}"].to_numpy()).max()) for i, t in enumerate(READ_TOKENS))
        out[f"s{s}"] = {"n": int(len(m)), "max_abs": d}
        worst = max(worst, d)
    return {"pass": bool(out) and worst < 1e-4 if out else None, "max_abs": worst, "cells": out}


# ================================================================================ ROLLOUT

_W = {}


def _init_worker(dataset, ck):
    """每个 worker 只加载一次模型和 token 流；单线程（并行在人这一层）。"""
    torch.set_num_threads(1)
    _W["eng"] = RD.make_engine(ck, dataset, "cpu")
    _W["streams"] = RD.person_streams(dataset)


def rollout_one(eng, a, k, tl, n_mc, seed, s0_obs):
    """一个臂：前缀 = 站点为止的流 + 站点后 1 天的 No-event 查询 token，生成 10 年。"""
    m = a <= tl
    t0 = tl + 1.0
    a2, k2 = np.append(a[m], t0), np.append(k[m], int(eng.res.NO_EVENT))
    st, sa = eng.simulate(k2, a2, n_mc=n_mc, until_age_years=(t0 + HORIZON * DPY + 1) / DPY,
                          max_new_tokens=MAX_NEW_TOKENS, seed=seed)
    P = min(len(k2), eng.block_size)              # simulate 返回的前缀列数（前缀超过 block_size 时只留最后 block_size 个）
    s0_own = [stage_of(eng, k[m], f) for f in FAMS]
    return readout(eng, st, sa, P, t0, s0_own, s0_obs), (st, sa)


def readout(tr, st, sa, P, t0, s0_own, s0_obs):
    """生成轨迹 (n_mc, W)（含前缀）-> 逐年累计首次通过数、分期占据、比站点分期更差的条数。"""
    n, W = st.shape
    real = sa > RD.PAD_AGE + 1
    col = np.arange(W)[None, :]
    gen = real & (col >= P)
    first = np.empty((len(EP), n))
    for qi, q in enumerate(EP):
        ids = [tr.tok[x] for x in TRANS[q][2]]
        # 首次通过 = 生成部分里第一个目标 token；轨迹在 Death 处停，所以天然在这条轨迹的死亡之前
        first[qi] = (np.where(gen & np.isin(st, ids), sa, np.inf).min(1) - t0) / DPY
    cum = np.stack([(first <= y).sum(1) for y in YEARS], 1).astype(np.int32)          # (6, 10)
    death = first[EP.index("death")]
    occ = np.zeros((len(FAMS), 5, HORIZON + 1), np.int32)
    ff = np.zeros((len(FAMS), 4, HORIZON + 1), np.int32)     # 死后前向填充的分期（不单列死亡）：最好/中/最差/未知
    w_own = np.zeros((len(FAMS), HORIZON + 1), np.int32)
    w_obs = np.zeros((len(FAMS), HORIZON + 1), np.int32)
    for fi, fam in enumerate(FAMS):
        ids = [tr.tok[x] for x in STAGE_BINS[fam]]
        lvl = np.full(st.shape, -1)
        for b, i in enumerate(ids):
            lvl[st == i] = b
        isf = gen & (lvl >= 0)
        for y in range(HORIZON + 1):
            cut = t0 + y * DPY
            last = np.where(isf & (sa <= cut), col, -1).max(1)
            # 前向填充：生成部分里还没有这一族的 token -> 用前缀（这一臂自己的流）在站点的分期；死后没有 token，
            # 自然停在死前最后一个分期（恶化趋势要的就是这个，免得存活选择）
            cur = np.where(last >= 0, lvl[np.arange(n), np.maximum(last, 0)], s0_own[fi])
            dead = death <= y
            state = np.where(dead, 3, np.where(cur < 0, 4, 2 - cur))       # 0 最好 .. 2 最差、3 死亡、4 未知
            occ[fi, :, y] = np.bincount(state, minlength=5)
            ff[fi, :, y] = np.bincount(np.where(cur < 0, 3, 2 - cur), minlength=4)
            if s0_own[fi] >= 0:
                w_own[fi, y] = int(((cur >= 0) & (cur < s0_own[fi])).sum())
            if s0_obs[fi] >= 0:
                w_obs[fi, y] = int(((cur >= 0) & (cur < s0_obs[fi])).sum())
    assert (np.diff(cum, axis=1) >= 0).all(), "累计首次通过数不单调"
    assert (occ.sum(1) == n).all() and (ff.sum(1) == n).all(), "分期占据加起来不等于轨迹数"
    return {"cum": cum, "occ": occ, "ff": ff, "w_own": w_own, "w_obs": w_obs, "s0_own": np.array(s0_own, np.int8),
            "s0_obs": np.array(s0_obs, np.int8), "gen_cols": int(W - P)}


def _roll_person(job):
    pid, tL, t_first, arms, n_mc, check = job
    eng = _W["eng"]
    ages, toks = _W["streams"][pid]
    s0_obs = [stage_of(eng, toks[ages <= tL], f) for f in FAMS]
    seed = crn_seed(pid) + CRN_OFFSET                  # 共同随机数：这个人的所有臂同一串
    out, crn_ok, skipped = [], None, []
    for arm in arms:
        a, k, tl = translate_tokens(eng, arm, ages, toks, tL, t_first)
        if a is None:
            skipped.append(arm)
            continue
        r, (st, sa) = rollout_one(eng, a, k, tl, n_mc, seed, s0_obs)
        if check and arm == "orig":
            r2, (st2, sa2) = rollout_one(eng, a, k, tl, n_mc, seed, s0_obs)
            crn_ok = bool(np.array_equal(st, st2) and np.array_equal(sa, sa2) and np.array_equal(r["cum"], r2["cum"]))
        out.append((arm, r))
    return pid, out, crn_ok, skipped


def run_rollout(name, C, out_dir, cpus, arms, n_mc, check_crn):
    import concurrent.futures as cf
    import multiprocessing as mp
    reg = {m[0]: m for m in DELPHI}
    _, dirs, dataset, tokz, sw, _ = reg[name]
    ck = os.path.join(DELPHI_DIR, dirs[0], "ckpt.pt")
    info = {"name": name, "dataset": dataset, "tokenization": tokz, "ckpt": ck, "seed_ckpt": seeds_of(dirs)[0],
            "n_mc": n_mc, "arms": list(arms), "horizon_years": HORIZON, "crn": f"crn_seed(pid)+{CRN_OFFSET}",
            "start": "query token (No event) at tL+1 for every arm", "cpus": cpus, "host": os.uname().nodename}
    try:
        eng = RD.make_engine(ck, dataset, "cpu")          # snapshot / visit_heads：生成器在这里明确拒绝
    except Exception as ex:
        info["status"] = f"undefined: {type(ex).__name__}: {ex}"
        json.dump(info, open(os.path.join(out_dir, "rollout_info.json"), "w"), indent=1)
        print("!!", info["status"], flush=True)
        return info
    info["pos_align"], info["visit_sizes"] = eng.pos_align, eng.visit_sizes_path
    del eng
    jobs = []
    for pid, r in C.iterrows():
        al = tuple(a for a in arms if r[f"elig__{a}"])
        jobs.append((int(pid), int(r.tL), int(r.t_first), al, n_mc, bool(check_crn and len(jobs) < 3)))
    t0 = time.time()
    res = []
    if cpus <= 1:
        _init_worker(dataset, ck)
        for i, jb in enumerate(jobs):
            res.append(_roll_person(jb))
    else:
        with cf.ProcessPoolExecutor(max_workers=cpus, mp_context=mp.get_context("spawn"),
                                    initializer=_init_worker, initargs=(dataset, ck)) as ex:
            futs = [ex.submit(_roll_person, jb) for jb in jobs]
            for i, f in enumerate(cf.as_completed(futs)):
                res.append(f.result())
                if i % 25 == 0:
                    print(f"  rollout {name}: {i + 1}/{len(jobs)} 人  {time.time() - t0:.0f}s", flush=True)
    res.sort(key=lambda x: x[0])
    flat = [(pid, arm, r) for pid, out, _, _ in res for arm, r in out]
    crn = [c for _, _, c, _ in res if c is not None]
    skipped = {}
    for _, _, _, sk in res:
        for a in sk:
            skipped[a] = skipped.get(a, 0) + 1
    np.savez_compressed(os.path.join(out_dir, "rollout.npz"),
                        pid=np.array([f[0] for f in flat], np.int64), arm=np.array([f[1] for f in flat]),
                        cum=np.stack([f[2]["cum"] for f in flat]), occ=np.stack([f[2]["occ"] for f in flat]),
                        ff=np.stack([f[2]["ff"] for f in flat]),
                        w_own=np.stack([f[2]["w_own"] for f in flat]), w_obs=np.stack([f[2]["w_obs"] for f in flat]),
                        s0_own=np.stack([f[2]["s0_own"] for f in flat]), s0_obs=np.stack([f[2]["s0_obs"] for f in flat]),
                        gen_cols=np.array([f[2]["gen_cols"] for f in flat]), n_mc=np.array(n_mc))
    gc = np.array([f[2]["gen_cols"] for f in flat])
    info.update({"status": "ok", "n_persons": len(jobs), "n_person_arms": len(flat), "minutes": (time.time() - t0) / 60,
                 "crn_reproducible": {"n_checked": len(crn), "all_identical": bool(all(crn)) if crn else None},
                 "skipped_prior_token_in_own_stream": skipped,
                 # 一次生成最多 MAX_NEW_TOKENS 步；生成列数 >= 它才可能是"步数用完、没走到 10 年"（保守的截断标记）
                 "possibly_truncated_person_arms": int((gc >= MAX_NEW_TOKENS).sum()), "max_gen_cols": int(gc.max())})
    json.dump(_j(info), open(os.path.join(out_dir, "rollout_info.json"), "w"), indent=1)
    print(f"  rollout {name}: {len(flat)} 人×臂  {info['minutes']:.1f} min  CRN {info['crn_reproducible']}  "
          f"可能截断 {info['possibly_truncated_person_arms']}", flush=True)
    if crn and not all(crn):
        raise SystemExit("共同随机数不可复现：同一个人同一个臂跑两次结果不同")
    return info


# ================================================================================ 汇总

def boot(x, seed=BOOT_SEED):
    """人级 bootstrap：均值和 95% 区间（B = N_BOOT，自己的 rng）。"""
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bm = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def score(d):
    if d is None or not np.isfinite(d):
        return None
    return "right" if d < -EPS else ("wrong" if d > EPS else "no_response")


def _pairs(C, idx, arm, q):
    """(扰动臂行号, 原样臂行号)：有资格、站点上处在风险中、两臂都有结果的人。"""
    ia, io = [], []
    for p in C.index[C[f"elig__{arm}"].to_numpy(bool) & C[f"risk__{q}"].to_numpy(bool)]:
        if p in idx[arm] and p in idx["orig"]:
            ia.append(idx[arm][p])
            io.append(idx["orig"][p])
    return np.array(ia, int), np.array(io, int)


def _index(pid, arm):
    idx = {}
    for j, (p, a) in enumerate(zip(pid.tolist(), arm.tolist())):
        idx.setdefault(a, {})[int(p)] = j
    return idx


def agg_direct(C, Z):
    F = Z["F"]                                         # (n_seed, n, 6, 10)
    idx = _index(Z["pid"], Z["arm"])
    rm = rmst_from_F(F)                                # (n_seed, n, 6)
    Fc = np.clip(F, 1e-9, 1 - 1e-9)
    lor = np.log(Fc) - np.log1p(-Fc)
    cf = None
    if "logit" in Z.files:
        lse = np.stack([delphi_F(lg)[1] for lg in Z["logit"]])
        cf = rmst_closed(lse)
    eff, cf_gap = {}, 0.0
    for arm in ARMS[1:]:
        if arm not in idx:
            continue
        e = {}
        for qi, q in enumerate(EP):
            ia, io = _pairs(C, idx, arm, q)
            cell = {"n": int(len(ia)), "prespec": "earlier" if q in prespec(arm) else None}
            if q in reached(arm):
                cell.update(defined=False, reason="reached by the perturbation itself")
                e[q] = cell
                continue
            if len(ia) == 0:
                cell.update(defined=False, reason="no eligible at-risk persons")
                e[q] = cell
                continue
            d = (rm[:, ia, qi] - rm[:, io, qi]).mean(0)         # 每人的 seed 平均
            m, lo, hi = boot(d)
            cell.update(defined=bool(len(ia) >= MIN_N), d_rmst=m, ci=[lo, hi],
                        rmst_orig=float(rm[:, io, qi].mean()), rmst_pert=float(rm[:, ia, qi].mean()),
                        d_logor={}, d_logor_ci={})
            if len(ia) < MIN_N:
                cell["reason"] = f"n < {MIN_N}"
            # fig6 面板 B（本地画）：两臂逐年的平均风险 F(1..10)（先对 seed、再对人平均）
            cell["cif_orig"] = F[:, io, qi, :].mean(axis=(0, 1))
            cell["cif_pert"] = F[:, ia, qi, :].mean(axis=(0, 1))
            for H in LOGOR_H:
                dl = (lor[:, ia, qi, H - 1] - lor[:, io, qi, H - 1]).mean(0)
                mm, l2, h2 = boot(dl)
                cell["d_logor"][str(H)] = mm
                cell["d_logor_ci"][str(H)] = [l2, h2]
            cell["earlier"] = bool(cell["defined"] and m < -EPS)
            cell["ci_excludes_0"] = bool(lo > 0 or hi < 0)
            cell["score"] = score(m) if (cell["prespec"] and cell["defined"]) else None
            if cf is not None:
                dcf = float((cf[:, ia, qi] - cf[:, io, qi]).mean(0).mean())
                cell["d_rmst_closed_form"] = dcf
                cf_gap = max(cf_gap, abs(dcf - m))
            e[q] = cell
        eff[arm] = e
    extra = {}
    if cf is not None:
        extra["closed_form_check"] = {"max_abs_rmst_trapezoid_minus_exact": float(np.abs(rm - cf).max()),
                                      "max_abs_cell_delta_diff": cf_gap,
                                      "note": "trapezoid on the yearly grid vs exact (1-exp(-lam*10))/lam; trapezoid is primary"}
    return eff, extra


def agg_rollout(C, Z):
    n_mc = int(Z["n_mc"])
    idx = _index(Z["pid"], Z["arm"])
    cum = Z["cum"].astype(np.float64) / n_mc           # (R, 6, 10) 每人的累计发生率
    S = np.concatenate([np.ones(cum.shape[:2] + (1,)), 1.0 - cum], -1)
    rmtl = HORIZON - 0.5 * (S[..., :-1] + S[..., 1:]).sum(-1)     # (R, 6)
    eff, curves = {}, {}
    for arm in ARMS[1:]:
        if arm not in idx:
            continue
        e = {}
        for qi, q in enumerate(EP):
            ia, io = _pairs(C, idx, arm, q)
            cell = {"n": int(len(ia)), "prespec": "earlier" if q in prespec(arm) else None}
            if q in reached(arm):
                cell.update(defined=False, reason="reached by the perturbation itself")
                e[q] = cell
                continue
            if len(ia) == 0:
                cell.update(defined=False, reason="no eligible at-risk persons")
                e[q] = cell
                continue
            d = rmtl[ia, qi] - rmtl[io, qi]
            m, lo, hi = boot(d)
            excl = bool(lo > 0 or hi < 0)
            cell.update(defined=bool(len(ia) >= MIN_N), d_rmtl=m, ci_rmtl=[lo, hi],
                        d_rmst=-m, ci=[-hi, -lo], ci_excludes_0=excl,
                        cif_orig=Z["cum"][io, qi].sum(0) / (len(io) * n_mc),
                        cif_pert=Z["cum"][ia, qi].sum(0) / (len(ia) * n_mc))
            if len(ia) < MIN_N:
                cell["reason"] = f"n < {MIN_N}"
            cell["earlier"] = bool(cell["defined"] and -m < -EPS and excl)
            cell["score"] = score(-m) if (cell["prespec"] and cell["defined"]) else None
            e[q] = cell
        eff[arm] = e
        # 恶化趋势：比站点时自己的分期更差的合并比例（逐年 0..10），原样 vs 扰动，同一批人
        cv = {}
        for fi, fam in enumerate(FAMS):
            ia, io = [], []
            for p in C.index[C[f"elig__{arm}"].to_numpy(bool)]:
                if p in idx[arm] and p in idx["orig"] and Z["s0_own"][idx[arm][p], fi] >= 0 \
                        and Z["s0_own"][idx["orig"][p], fi] >= 0:
                    ia.append(idx[arm][p])
                    io.append(idx["orig"][p])
            if not ia:
                continue
            ia, io = np.array(ia), np.array(io)
            den = len(ia) * n_mc
            occ_o = Z["occ"][io, fi].sum(0) / den
            occ_p = Z["occ"][ia, fi].sum(0) / den
            cv[fam] = {"n": int(len(ia)), "worse_than_own_tL_orig": Z["w_own"][io, fi].sum(0) / den,
                       "worse_than_own_tL_pert": Z["w_own"][ia, fi].sum(0) / den,
                       # 扰动把这一族直接设到最差箱（cog2_severe 的 MMSE）-> "比自己更差"恒为 0，这条曲线不提供信息
                       "pert_at_floor": bool((Z["s0_own"][ia, fi] == 0).all()),
                       "worse_than_observed_tL_orig": Z["w_obs"][io, fi].sum(0) / den,
                       "worse_than_observed_tL_pert": Z["w_obs"][ia, fi].sum(0) / den,
                       # 共同尺子：最差箱（MMSE Impaired / 认知 Low）的比例，死后按最后一个分期前向填充
                       "p_worst_ff_orig": Z["ff"][io, fi, 2].sum(0) / den,
                       "p_worst_ff_pert": Z["ff"][ia, fi, 2].sum(0) / den,
                       "occupancy_states": list(STATES[fam]), "occupancy_orig": occ_o, "occupancy_pert": occ_p}
        curves[arm] = cv
    return eff, curves


def synchrony(eff):
    """每个扰动：有定义的终点数、提前的终点数、"同步提前"（AD 提前且每道有定义的认知题都提前）、先后顺序。
    跨扰动：ΔAD 和每道认知题 Δ 的 Spearman。"""
    from scipy.stats import spearmanr
    syn = {}
    for arm, e in eff.items():
        dfn = [q for q in EP if e[q].get("defined")]
        ear = [q for q in dfn if e[q].get("earlier")]
        cog = [q for q in COG_EP if q in dfn]
        ad_def = "ad" in dfn
        order = {}
        for a1, a2 in (("mmse_lt27", "mmse_lt24"), ("cogn_lt0", "cogn_ltm1")):
            if a1 in dfn and a2 in dfn:
                d1, d2 = e[a1]["d_rmst"], e[a2]["d_rmst"]
                # 更早的里程碑（<27）的提前量至少和更晚的（<24）一样大（Δ 更负或相等）
                order[f"{a1}_vs_{a2}"] = {"d_first": d1, "d_second": d2, "ok": bool(d1 <= d2 + 1e-9)}
        syn[arm] = {"n_defined": len(dfn), "n_earlier": len(ear), "defined": dfn, "earlier": ear,
                    "ad_defined": ad_def, "ad_earlier": bool("ad" in ear) if ad_def else None,
                    "cog_defined": cog, "cog_earlier": [q for q in cog if q in ear],
                    # 至少要有一道认知题有定义（cog2_severe 把认知题都直接越过了，不能只凭 AD 算同步；2026-10-31 改）
                    "synchronous_advance": bool(ad_def and cog and "ad" in ear and all(q in ear for q in cog)),
                    "ordering": order}
    sp = {}
    for q in COG_EP + ("death",):
        xs, ys = [], []
        for arm, e in eff.items():
            if e["ad"].get("defined") and e[q].get("defined"):
                xs.append(e["ad"]["d_rmst"])
                ys.append(e[q]["d_rmst"])
        if len(xs) >= 3 and np.ptp(xs) > 0 and np.ptp(ys) > 0:
            r, p = spearmanr(xs, ys)
            sp[q] = {"rho": float(r), "p": float(p), "n_perturbations": len(xs)}
        else:
            sp[q] = {"rho": None, "p": None, "n_perturbations": len(xs)}
    tally = {"right": 0, "no_response": 0, "wrong": 0}
    for e in eff.values():
        for c in e.values():
            if c.get("score"):
                tally[c["score"]] += 1
    tally["n_scored"] = sum(tally.values())
    return syn, sp, tally


def model_meta(name):
    reg = {m[0]: m for m in DELPHI}
    _, _, dataset, tokz, sw, _ = reg[name]
    meta = {"family": "delphi", "dataset": dataset, "tokenization": tokz, "switches": sw,
            "ranked": tokz != "snapshot" and "visitheads" not in sw}
    if tokz == "snapshot":
        meta["unranked_reason"] = ("snapshot: every visit re-emits the full state, the between-visit query position is "
                                   "never supervised (see registry.REFERENCES) -- computed, not ranked")
    if "visitheads" in sw:
        meta["unranked_reason"] = "visit_heads: lm_head logits are not rates -- direct readout undefined"
    return meta


def aggregate_model(name, C, tag):
    d = os.path.join(out_base(tag), name)
    J = {"name": name, **model_meta(name), "generated": time.strftime("%Y-%m-%d %H:%M"),
         "config": {"atoms": {a: {"label": v[0], "eligibility": v[1]} for a, v in ATOMS.items()},
                    "combos": COMBOS, "prespec": {a: sorted(prespec(a)) for a in ARMS[1:]},
                    "reached": {a: sorted(reached(a)) for a in ARMS[1:] if reached(a)},
                    "eps_years": EPS, "min_n": MIN_N, "n_boot": N_BOOT, "horizon_years": HORIZON,
                    "endpoints": {q: TRANS[q][0] for q in EP},
                    "n_eligible": {a: int(C[f"elig__{a}"].sum()) for a in ARMS},
                    "n_at_risk": {q: int(C[f"risk__{q}"].sum()) for q in EP}}}
    di = os.path.join(d, "direct_info.json")
    if os.path.exists(os.path.join(d, "direct.npz")):
        info = json.load(open(di))
        eff, extra = agg_direct(C, np.load(os.path.join(d, "direct.npz")))
        syn, sp, tally = synchrony(eff)
        J["direct"] = {"status": "ok", "seeds": info.get("seeds"), "effects": eff, "synchrony": syn,
                       "spearman_ad_vs": sp, "prespec_score": tally,
                       "regression_fig4": {k: v for k, v in info.get("regression_fig4", {}).items() if k != "cells"},
                       "regression_scores_trans": {k: v for k, v in info.get("regression_scores_trans", {}).items()
                                                   if k != "cells"},
                       "stroke_feature_conflict": info.get("stroke_feature_conflict"), **extra}
    elif os.path.exists(di):
        J["direct"] = {"status": json.load(open(di)).get("status", "missing")}
    else:
        J["direct"] = {"status": "not run"}
    ri = os.path.join(d, "rollout_info.json")
    if os.path.exists(os.path.join(d, "rollout.npz")):
        info = json.load(open(ri))
        eff, curves = agg_rollout(C, np.load(os.path.join(d, "rollout.npz")))
        syn, sp, tally = synchrony(eff)
        J["rollout"] = {"status": "ok", "n_mc": info.get("n_mc"), "seed_ckpt": info.get("seed_ckpt"),
                        "arms": info.get("arms"), "start": info.get("start"), "crn": info.get("crn"),
                        "effects": eff, "deterioration": curves, "synchrony": syn, "spearman_ad_vs": sp,
                        "prespec_score": tally, "crn_reproducible": info.get("crn_reproducible"),
                        "possibly_truncated_person_arms": info.get("possibly_truncated_person_arms")}
    elif os.path.exists(ri):
        J["rollout"] = {"status": json.load(open(ri)).get("status", "missing")}
    else:
        J["rollout"] = {"status": "not run"}
    J = _j(J)
    pk = pkg_dir(name, tag)
    os.makedirs(pk, exist_ok=True)
    json.dump(J, open(os.path.join(pk, "perturb_v2.json"), "w"), indent=1)
    return J


def summary_entry(J):
    out = {k: J.get(k) for k in ("family", "tokenization", "ranked", "unranked_reason") if k in J}
    for ro in ("direct", "rollout"):
        R = J.get(ro, {})
        if R.get("status") != "ok":
            out[ro] = {"status": R.get("status")}
            continue
        md = {}
        for arm, e in R["effects"].items():
            cog = {q: e[q]["d_rmst"] for q in COG_EP if e[q].get("defined")}
            md[arm] = {"ad": e["ad"].get("d_rmst") if e["ad"].get("defined") else None,
                       "death": e["death"].get("d_rmst") if e["death"].get("defined") else None,
                       "cognitive_mean": float(np.mean(list(cog.values()))) if cog else None, "cognitive": cog}
        out[ro] = {"status": "ok", "prespec": R["prespec_score"],
                   "synchronous_advance": [a for a, s in R["synchrony"].items() if s["synchronous_advance"]],
                   "n_earlier_of_defined": {a: f"{s['n_earlier']}/{s['n_defined']}" for a, s in R["synchrony"].items()},
                   "spearman_ad_vs": {q: v["rho"] for q, v in R["spearman_ad_vs"].items()},
                   "mean_d_rmst_years": md}
    return out


def aggregate(only, tag):
    C, diag = cohort_table()
    names = [m[0] for m in DELPHI]
    if only:
        names = [n for n in names if n in only]
    summ = {"generated": time.strftime("%Y-%m-%d %H:%M"), "definition": {
        "effect": "Delta RMST over 10 years (perturbed - original, years; negative = transition EARLIER)",
        "direct": "per-seed effect, averaged over seeds; 95% CI = person bootstrap of each person's seed mean",
        "rollout": "seed-42 ckpt, query-token start at tL+1, common random numbers; Delta RMST = -Delta RMTL; "
                   "'earlier' also needs the bootstrap CI to exclude 0",
        "synchronous_advance": "AD earlier AND every defined cognitive endpoint earlier",
        "prespec_score": f"right = moved earlier by > {EPS} y; no_response = |Delta| < {EPS} y; wrong = later by > {EPS} y"},
        "cohort": diag, "models": {}}
    for n in names:
        J = aggregate_model(n, C, tag)
        summ["models"][n] = summary_entry(J)
        print(f"  {n}: direct {J['direct'].get('status')}  rollout {J['rollout'].get('status')}", flush=True)
    p = os.path.join(pkg_dir("", tag).rstrip("/"), "PERTURB_V2_SUMMARY.json")
    json.dump(_j(summ), open(p, "w"), indent=1)
    print("->", p, flush=True)


# ================================================================================ 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["direct", "rollout", "aggregate"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--all", action="store_true", help="direct：注册表里的全部 Delphi")
    ap.add_argument("--only", default=None, help="逗号分隔的模型名")
    ap.add_argument("--limit", type=int, default=0, help="只跑队列里前 N 人（调试；结果写到 perturb_v2/_limit<N>/）")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--arms", default=",".join(ROLLOUT_ARMS))
    ap.add_argument("--n_mc", type=int, default=ROLLOUT_N_MC)
    ap.add_argument("--check_crn", type=int, default=1, help="前 3 人的原样臂各跑两次，核对共同随机数可复现")
    args = ap.parse_args()
    tag = args.tag if args.tag is not None else (f"limit{args.limit}" if args.limit else "")
    only = set(args.only.split(",")) if args.only else None
    if args.mode == "aggregate":
        aggregate(only, tag)
        return
    C, diag = cohort_table(args.limit)
    print(f"队列 {len(C)} 人；资格 {diag['n_eligible']}", flush=True)
    os.makedirs(out_base(tag), exist_ok=True)
    # 人级，只留在 RIS。direct 和几个 rollout 作业同时写同一份：先写临时文件再原子替换（内容相同，谁最后都一样）
    tmp = os.path.join(out_base(tag), f".cohort_v2.{os.getpid()}.csv")
    C.to_csv(tmp)
    os.replace(tmp, os.path.join(out_base(tag), "cohort_v2.csv"))
    tmp = os.path.join(out_base(tag), f".cohort_v2_diag.{os.getpid()}.json")
    json.dump(diag, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(out_base(tag), "cohort_v2_diag.json"))
    failed = {}
    if args.mode == "direct":
        names = [m[0] for m in DELPHI] if args.all else [args.name]
        if only:
            names = [n for n in names if n in only]
        for n in names:
            od = os.path.join(out_base(tag), n)
            os.makedirs(od, exist_ok=True)
            print(f"== direct {n}", flush=True)
            # 一个模型失败（回归没过、ckpt 坏了）不拖累其余模型；最后统一以非零退出
            try:
                direct_delphi(n, C, od, args.cpus)
            except (Exception, SystemExit) as ex:
                failed[n] = f"{type(ex).__name__}: {ex}"
                print(f"!! {n} 失败：{failed[n]}", flush=True)
    else:
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
        assert arms[0] == "orig" and set(arms) <= set(ARMS), f"臂必须以 orig 开头、在 {ARMS} 里"
        names = [args.name] if args.name else list(ROLLOUT_MODELS)
        for n in names:
            od = os.path.join(out_base(tag), n)
            os.makedirs(od, exist_ok=True)
            print(f"== rollout {n}  arms {arms}  n_mc {args.n_mc}  cpus {args.cpus}", flush=True)
            try:
                run_rollout(n, C, od, args.cpus, arms, args.n_mc, args.check_crn)
            except (Exception, SystemExit) as ex:
                failed[n] = f"{type(ex).__name__}: {ex}"
                print(f"!! {n} 失败：{failed[n]}", flush=True)
    if failed:
        raise SystemExit(f"失败的模型：{failed}")


if __name__ == "__main__":
    main()
