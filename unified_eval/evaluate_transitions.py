"""evaluate_transitions.py -- 第三部分（疾病分期转换）的统一打分 + 统一出图。**所有模型都用这一份代码**。

    python evaluate_transitions.py --name delphi-nodedup \
        --scores s42=$RUNS/scores_trans/delphi-nodedup__s42.csv s43=... s44=... \
        --outdir <package dir> [--calibration_undefined]

输入：truth_trans_val.csv / truth_trans_train.csv（transition_truth.py；基线只从训练段学）、traj_truth_val.json（死亡 / AD 年龄，
竞争风险敏感性用）、newmodel_data.npz（连续值快照基线用；没有就跳过），每个 seed 一张风险表（pid, t, risk_<题>_<H>y，可选 rank_<题>）。
第一个 --scores 是主 seed。某道题的风险列**不在表里** = 这个模型不答这道题（记"未答"，不进平均）；在表里就必须覆盖全部站点。

指标（EVAL_PROTOCOL.md 第 11 节）：
  1. 判别力随时间窗（2 / 5 / 10 年；图上画 1 / 2 / 3 / 5 / 10）：AUC，按人 bootstrap（和第一部分同一套重抽样）。
     主口径只数答案已知的站点（complete case）：长时间窗里，窗口内没越线就死了的人被排除，留下的阴性更年轻、更健康。
     敏感性 auc_cr：这些站点算阴性（竞争风险口径，和第一部分 8 节"AD：先死亡算阴性"同一个意思）。
  2. 现状之外还剩多少：
     - 同一当前状态内的 AUC：只在站点时 MMSE 分期和认知分期都相同的站点之间比（分层合并：只数同一层里的阳性-阴性对）。
       只会读出当前**分期**的预测在这里恰好 0.5。再固定 5 岁年龄段，剩下的是分期内更细的当前值、背景（APOE、教育、性别、
       段内年龄）和病史 —— 不全是"动态"，所以要和尺子比：只看人口学的 Delphi、快照基线（分期 / 连续值）。
     - 转变 vs 平稳（图上用的口径）：特异度固定在 80%（只把 20% 的平稳站点判为"会转变"）时，抓到的转变比例
       sens_at_spec80。随机排序时它是 0.20，和题目、发生率都无关，所以 6 道题能放在一张图上比。阈值按排序定，不依赖校准
       （旧 figure 2 的教训：众数 / 0.5 阈值量的是校准，不是信息）。
       另存一个按发生率定阈值的版本（recall_at_prev / spec_at_prev：判 k = 真实转变数个为"会转变"），只进 JSON。
     - Brier skill：相对训练段总发生率（气候态）和相对分期基线；只在风险是概率时才有意义，只作参考、不排序。
  3. 按当前分期分格的表（行 = 站点时的当前分期，列 = H 年内越过的线；观察到的比例 vs 模型的平均风险、每格的 AUC），
     只进 JSON（heatmaps）。小格子（发生或没发生的人少于 10 个）整格不报。
三条不是模型的尺子（同一套指标）：分期基线、快照基线（分期）、快照基线（连续值）。

输出：transitions.json（只有聚合数）。图在本地按 JSON 画（figures.py 的 fig3_transitions.png）。
"""
import argparse
import json
import os

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from common import GRID, RUN_DIR  # noqa: E402
from evaluate import boot_persons  # noqa: E402
from transitions import (COGNITIVE, HEATMAP_COLS, HEATMAP_ROWS, STAGE_BINS, STAGE_NAMES, TABLE_H, TRANS,  # noqa: E402
                         TRANS_HORIZONS, label_col, risk_col)

N_BOOT = 1000
# 热图：一个格子要报 AUC，至少要有这么多阳性和阴性（少于这个只报发生率）。30 = 旧 figure 2 的报告门槛（MIN_POS=30）
MIN_CELL = 30
# 热图小格子：发生的人或没发生的人少于这个数（按人，不按站点），整格只写 small（不报比例、计数、AUC），见 heatmaps()
MIN_SHOW = 10
MIN_STRATUM = 20            # 分期基线：训练段里一个当前状态少于这么多已知标签，就退回这道题的总发生率
SPEC_REF = 0.80             # "转变 vs 平稳"：特异度固定在这里，报抓到的转变比例（随机排序时 = 1 − SPEC_REF）


# ----------------------------------------------------------------------------- 基本统计

AGE_BAND = 5                # 年；"同一当前状态 + 同一年龄段"里的 5 岁一段


def stratum_key(df):
    """当前状态 = (MMSE 分期, 认知分期)，没测过记 -1。"""
    return df["mmse_stage"].astype(int).to_numpy() * 10 + df["cogn_stage"].astype(int).to_numpy()


def stratum_age_key(df):
    """当前状态 × 5 岁年龄段：连年龄也一样的人之间比。只看人口学的模型在"同一当前状态"里还能靠年龄排序，
    在这一层里就只剩性别、教育、APOE 这些背景 —— 剩下的区分度才更接近"动态"（病史、变化的轨迹）。"""
    band = np.floor(df["age"].to_numpy() / AGE_BAND).astype(int)
    return stratum_key(df) * 1000 + band


def auc_within(y, s, strata):
    """分层合并 AUC：只比同一层里的阳性-阴性对，按对数加权 = Σ_k U_k / Σ_k n⁺_k n⁻_k（U = 层内 Mann-Whitney 统计量，
    并列取平均秩）。等价于各层 AUC 按 n⁺n⁻ 加权平均；只有一类的层不贡献对。向量化（bootstrap 里要算几万次）。"""
    df = pd.DataFrame({"y": y.astype(float), "s": s, "k": strata})
    r = df.groupby("k", sort=False)["s"].rank(method="average")
    g = pd.DataFrame({"k": df["k"], "y": df["y"], "ry": r * df["y"]}).groupby("k", sort=False)
    npos, n, sry = g["y"].sum(), g["y"].size(), g["ry"].sum()
    nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    den = float((npos[ok] * nneg[ok]).sum())
    if den == 0:
        return float("nan"), 0
    U = (sry[ok] - npos[ok] * (npos[ok] + 1) / 2).sum()
    return float(U / den), int(den)


def capture_at_prevalence(y, s):
    """把分数最高的 k 个判为"会转变"，k = 阳性个数。并列在阈值上时按均匀随机打破（取期望）。
    返回 (召回 = 抓到的转变 / 全部转变, 特异度 = 判为平稳的平稳站点 / 全部平稳站点)。"""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.sort(s)[::-1]
    thr = order[n_pos - 1]                      # 第 k 大的分数
    above = s > thr
    tie = s == thr
    k_left = n_pos - int(above.sum())           # 还要从并列里挑几个
    tp = float(y[above].sum()) + k_left * (float(y[tie].sum()) / int(tie.sum()))
    fp = n_pos - tp
    return tp / n_pos, 1.0 - fp / n_neg


def sens_at_spec(y, s, spec=SPEC_REF):
    """特异度固定在 spec 时抓到的转变比例：阈值取在平稳站点分数的 spec 分位上（只有 1 − spec 的平稳站点被判为"会转变"）。
    并列在阈值上时按均匀随机打破（取期望），所以全并列的分数（例如只看人口学的模型）得到的正好是随机水平。"""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    k = (1.0 - spec) * len(neg)                         # 允许判错的平稳站点个数（可以是小数）
    neg_desc = np.sort(neg)[::-1]
    thr = neg_desc[max(int(np.ceil(k)) - 1, 0)]
    n_above, n_tie = int((neg > thr).sum()), int((neg == thr).sum())
    frac = min(max((k - n_above) / n_tie, 0.0), 1.0)    # 并列在阈值上的站点被判为"会转变"的概率
    return float((pos > thr).mean() + frac * (pos == thr).mean())


def stage_baseline(train, val):
    """当前分期基线：训练段里同一当前状态（MMSE 分期 × 认知分期）的发生率；层太小退回总发生率。
    只用训练段，所以它和验证集独立。返回 {(q, H): val 上每个站点的基线风险}。"""
    ks_tr, ks_va = stratum_key(train), stratum_key(val)
    out = {}
    for q in TRANS:
        for H in TRANS_HORIZONS:
            y = train[label_col(q, H)].to_numpy()
            ok = y >= 0
            overall = float(y[ok].mean()) if ok.any() else float("nan")
            rate = {}
            for k in np.unique(ks_tr[ok]):
                m = ok & (ks_tr == k)
                if m.sum() >= MIN_STRATUM:
                    rate[k] = float(y[m].mean())
            out[(q, H)] = np.array([rate.get(k, overall) for k in ks_va])
            out[("prev", q, H)] = overall                  # 训练段的总发生率：Brier skill 的参照（"气候态"预报）
    return out


def metrics(y, p, rk, strata, base_p, calib_ok=True, strata_age=None, prev_ref=None):
    """一道题 × 一个时间窗。y ∈ {0,1}（已去掉不计入的），p = 风险，rk = 排序分数，base_p = 分期基线，
    prev_ref = 训练段的总发生率（Brier skill 的参照）。"""
    n_pos = int(y.sum())
    r = {"n": int(len(y)), "n_pos": n_pos, "prevalence": float(y.mean())}
    if n_pos == 0 or n_pos == len(y):
        return r
    r["auc"] = float(roc_auc_score(y, rk))
    r["auc_within_state"], r["within_state_pairs"] = auc_within(y, rk, strata)
    if strata_age is not None:
        r["auc_within_state_age"], r["within_state_age_pairs"] = auc_within(y, rk, strata_age)
    r["auprc"] = float(average_precision_score(y, rk))
    r["auprc_lift"] = r["auprc"] / r["prevalence"]
    rec, spec = capture_at_prevalence(y, rk)
    r["recall_at_prev"], r["spec_at_prev"] = rec, spec
    # 同一个阈值规则（判 k = 真实转变数 个为"会转变"）下随机排序的期望：召回 = 发生率，特异度 = 1 − 发生率。
    # 这才是这一对数字的"没有信息"水平；"把所有人都判为平稳"是 0 / 1，但那是另一种决策规则（k = 0）。
    r["recall_chance"], r["spec_chance"] = r["prevalence"], 1.0 - r["prevalence"]
    r["balanced_acc_at_prev"] = (rec + spec) / 2
    r["sens_at_spec80"] = sens_at_spec(y, rk, SPEC_REF)
    if calib_ok:
        brier = float(np.mean((p - y) ** 2))
        b_st = float(np.mean((base_p - y) ** 2))
        r.update({"mean_pred": float(p.mean()), "o_e": float(y.sum() / max(p.sum(), 1e-12)), "brier": brier,
                  "bss_vs_stage": 1 - brier / b_st if b_st > 0 else float("nan")})
        if prev_ref is not None and np.isfinite(prev_ref):
            # 参照 = 对每个人都报训练段的总发生率（"气候态"）。以前用"风险恒 0"当参照，那个分数主要量的是发生率本身
            b_pr = float(np.mean((prev_ref - y) ** 2))
            r["bss_vs_prevalence"] = 1 - brier / b_pr if b_pr > 0 else float("nan")
    return r


def baseline_metrics(y, base_p, strata, strata_age=None):
    """基线自己的成绩（同一套指标；分期基线在同一当前状态内的 AUC 按构造是 0.5）。"""
    r = metrics(y, base_p, base_p, strata, base_p, strata_age=strata_age)
    for k in ("bss_vs_stage",):
        r.pop(k, None)
    return r


def snapshot_baseline(train, val):
    """"当前快照"基线：训练段上的逻辑回归，输入只有站点时的当前 MMSE 分期、认知分期（one-hot，含'没测过'）、
    年龄、年龄²、性别。一个只会读出"现在是什么状态、多大年纪"的模型能拿到的分数 —— 模型要比它好，才说明用上了
    病史和变化。和第一部分"只看当前访视的 Delphi"是同一个意思，但不是 transformer，结果不依赖训练的随机性。"""
    from sklearn.linear_model import LogisticRegression

    def X(df):
        cols = []
        for fam in ("mmse", "cogn"):
            st = df[f"{fam}_stage"].to_numpy()
            cols += [(st == k).astype(float) for k in (-1, 0, 1)]          # 2（最好）当参照
        a = (df["age"].to_numpy() - 80.0) / 10.0
        cols += [a, a ** 2, df["male"].to_numpy().astype(float)]
        return np.stack(cols, 1)

    Xtr, Xva = X(train), X(val)
    out = {}
    for q in TRANS:
        for H in TRANS_HORIZONS:
            y = train[label_col(q, H)].to_numpy()
            ok = y >= 0
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[ok], y[ok])
            out[(q, H)] = lr.predict_proba(Xva)[:, 1]
    return out


# ----------------------------------------------------------------------------- 主计算

def cr_labels(truth, traj):
    """竞争风险敏感性的标签：主口径的 y；另外，站点时处在风险中、窗口内没观察到越线就死了的站点 = 0（不再排除）。
    traj = traj_truth_val.json（死亡、AD 诊断年龄）。死亡题本身不变。"""
    pid = truth["pid"].to_numpy()
    t = truth["t"].to_numpy()
    death = np.array([np.inf if traj[str(p)]["death"] is None else traj[str(p)]["death"] for p in pid], float)
    ad = np.array([np.inf if traj[str(p)]["ad"] is None else traj[str(p)]["ad"] for p in pid], float)
    out = {}
    for q, (_, fam, toks) in TRANS.items():
        if q == "death":
            continue
        if fam == "event":                                   # AD：站点时还没诊断
            at_risk = ad > t
        else:                                                # 首次越线：到站点为止测过、且从没进入目标区间
            tgt = [STAGE_BINS[fam].index(n) for n in toks]
            w = truth[f"{fam}_worst_so_far"].to_numpy()
            at_risk = (w >= 0) & ~np.isin(w, tgt)
        for H in TRANS_HORIZONS:
            y = truth[label_col(q, H)].to_numpy().copy()
            died = (y < 0) & at_risk & (death <= t + H * GRID)
            y[died] = 0
            out[(q, H)] = y
    return out


def snapshot_cont_baseline(train, val):
    """"当前快照（连续值）"基线：和 snapshot_baseline 同一个意思，但当前 MMSE、全局认知用**连续值**（统一输入 models_code/input_data.py 的特征，
    只用站点那一次访视、不看历史），外加年龄、性别。分期基线 / 快照基线只知道粗分期；用连续值输入的模型看得到分期里面的连续值，
    "同一当前状态内"的区分度有一部分就是这个 —— 用这条尺子才分得清"现状里的细节"和"病史 / 变化"。
    训练 = split811 的训练段（newmodel_data.npz 的 train）；没有这个文件就跳过（返回 None）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    path = os.path.join(RUN_DIR, "newmodel_data.npz")
    if not os.path.exists(path):
        return None
    Z = np.load(path)

    def feats(split):
        X, M, T, S, P = (Z[f"{split}_{k}"] for k in ("X", "M", "T", "S", "pid"))
        ii, vv = np.nonzero(M)
        x = X[ii, vv]
        # models_code/input_data.py 的列：0 MMSE、1 cogn_global（标准化），15 / 16 它们的缺失标志，51 (年龄-80)/10；S[0] 男
        mm, cg, mna, cna, age = x[:, 0], x[:, 1], x[:, 15], x[:, 16], x[:, 51]
        F = np.column_stack([mm, mm ** 2, mna, cg, cg ** 2, cna, age, age ** 2, S[ii, 0]])
        return pd.DataFrame({"pid": P[ii], "t": T[ii, vv]}), F

    ktr, Ftr = feats("train")
    kva, Fva = feats("val")
    tr = ktr.reset_index().merge(train, on=["pid", "t"], how="inner", validate="one_to_one")
    assert len(tr) == len(ktr), "训练段的站点不全在第三部分的训练真值里"
    va = val[["pid", "t"]].merge(kva.reset_index(), on=["pid", "t"], how="left", validate="one_to_one")
    assert va["index"].notna().all(), "验证集有站点不在统一输入的特征里"
    sc = StandardScaler().fit(Ftr[tr["index"].to_numpy()])
    Xtr, Xva = sc.transform(Ftr[tr["index"].to_numpy()]), sc.transform(Fva[va["index"].astype(int).to_numpy()])
    out = {}
    for q in TRANS:
        for H in TRANS_HORIZONS:
            y = tr[label_col(q, H)].to_numpy()
            ok = y >= 0
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[ok], y[ok])
            out[(q, H)] = lr.predict_proba(Xva)[:, 1]
    return out


def load_scores(truth, path):
    """打分表 -> (和真值逐站点对齐的表, 每道题答了哪些时间窗)。列在 = 答了；答了的列必须每个站点都有分。
    （对不上真值的站点、整列是空的列，都会在这里停下，不会悄悄变成"没答"。）"""
    sc0 = pd.read_csv(path)
    sc = truth[["pid", "t"]].merge(sc0, on=["pid", "t"], how="left", validate="one_to_one")
    answered = {}
    for q in TRANS:
        answered[q] = [H for H in TRANS_HORIZONS if risk_col(q, H) in sc0.columns]
        for H in answered[q]:
            miss = sc[risk_col(q, H)].isna()
            assert not miss.any(), f"{path}: {q} {H}y 有 {int(miss.sum())} 个站点没有分（答了的列必须覆盖全部站点）"
    assert any(answered.values()), f"{path}: 一道题都没答"
    return sc, answered


def per_seed(truth, sc, answered, base, calib_ok, boot_idx=None, cr=None):
    strata = stratum_key(truth)
    strata_age = stratum_age_key(truth)
    pid = truth["pid"].to_numpy()
    out = {}
    for q in TRANS:
        for H in TRANS_HORIZONS:
            key = f"{q}_{H}y"
            y_all = truth[label_col(q, H)].to_numpy()
            m = y_all >= 0
            # 人数（不只是站点数）：10 年的阳性多是同一批人在更多站点上重复计数
            cnt = {"n": int(m.sum()), "n_pos": int(y_all[m].sum()), "n_persons": int(len(np.unique(pid[m]))),
                   "n_persons_pos": int(len(np.unique(pid[m & (y_all == 1)])))}
            if H not in answered[q]:
                out[key] = {"answered": False, **cnt}
                continue
            p_all = sc[risk_col(q, H)].to_numpy(dtype=float)
            rc = f"rank_{q}"
            rk_all = sc[rc].to_numpy(dtype=float) if rc in sc.columns else p_all
            r = metrics(y_all[m].astype(int), p_all[m], rk_all[m], strata[m], base[(q, H)][m], calib_ok,
                        strata_age=strata_age[m], prev_ref=base.get(("prev", q, H)))
            r.update(cnt)
            r["answered"] = True
            if cr is not None and (q, H) in cr:
                # 竞争风险敏感性：窗口内没越线就先死了的站点算阴性（主口径把它们排除）。长时间窗里主口径只比
                # "活到窗口末、还在随访"的阴性，这些人更年轻、更健康，和年龄 / 死亡相关的分数会被抬高
                yc = cr[(q, H)]
                mc = yc >= 0
                if 0 < yc[mc].sum() < mc.sum():
                    r["auc_cr"] = float(roc_auc_score(yc[mc], rk_all[mc]))
                    r["n_cr"] = int(mc.sum())
            if boot_idx is not None and H in TABLE_H and "auc" in r:
                pos = np.full(len(y_all), -1)
                pos[m] = np.arange(m.sum())
                yy, rr, ss, sa = y_all[m].astype(int), rk_all[m], strata[m], strata_age[m]
                ba, bw, bwa = [], [], []
                for ii in boot_idx:
                    jj = pos[ii]
                    jj = jj[jj >= 0]
                    yb = yy[jj]
                    if 0 < yb.sum() < len(yb):
                        ba.append(roc_auc_score(yb, rr[jj]))
                        bw.append(auc_within(yb, rr[jj], ss[jj])[0])
                        bwa.append(auc_within(yb, rr[jj], sa[jj])[0])
                    else:
                        ba.append(np.nan)
                        bw.append(np.nan)
                        bwa.append(np.nan)
                for f, b in (("auc", ba), ("auc_within_state", bw), ("auc_within_state_age", bwa)):
                    r[f + "_ci"] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
                r["_boot_auc"], r["_boot_within"], r["_boot_within_age"] = ba, bw, bwa
            out[key] = r
    return out


def macro(res, H, field="auc"):
    vals = [res[f"{q}_{H}y"].get(field) for q in TRANS]
    if any(v is None for v in vals) or not all(res[f"{q}_{H}y"].get("answered") for q in TRANS):
        return None
    return float(np.mean(vals))


def heatmaps(truth, sc, answered, H, calib_ok=True):
    """当前分期（行）× 越过的线（列）。只在"这一行的人可能处在风险中"的格子上有数。
    calib_ok=False（visit_heads：logit 不是速率，风险饱和成 1）：不报"预测的平均风险"，只报 AUC。

    小格子保护（聚合产物要离开 RIS）：按**人**数，不按站点数（一个人有多个站点）。一个格子里发生的人或没发生的人
    少于 MIN_SHOW 个，整格只写 small（不报比例、不报计数、不报 AUC）—— 例如"10 年、当前 Impaired、AD：N 个站点里 N−1 个阳性"
    等于说出了一个人的结局。报出来的格子也只给四舍五入到 10 的站点数（n_approx），没有精确计数，不能靠行列相减还原小格子。"""
    pid = truth["pid"].to_numpy()
    out = {}
    for fam, cols in HEATMAP_COLS.items():
        st = truth[f"{fam}_stage"].to_numpy()
        cells = []
        for row in HEATMAP_ROWS[fam]:
            line = []
            for q in cols:
                y_all = truth[label_col(q, H)].to_numpy()
                m = (st == row) & (y_all >= 0)
                c = {"q": q, "row": int(row)}
                if not m.any():
                    c["empty"] = True                     # 这一行没有人处在风险中（例如当前已 <27 的人不问"首次 <27"）
                    line.append(c)
                    continue
                n_pp = len(np.unique(pid[m & (y_all == 1)]))
                n_pn = len(np.unique(pid[m & (y_all == 0)]))
                if min(n_pp, n_pn) < MIN_SHOW:
                    c["small"] = True
                    line.append(c)
                    continue
                c["n_approx"] = int(round(int(m.sum()), -1))
                c["obs"] = float(y_all[m].mean())
                if H in answered[q]:
                    p = sc[risk_col(q, H)].to_numpy(dtype=float)[m]
                    rc = f"rank_{q}"
                    rk = sc[rc].to_numpy(dtype=float)[m] if rc in sc.columns else p
                    if calib_ok:
                        c["pred"] = float(p.mean())
                    yy = y_all[m].astype(int)
                    if yy.sum() >= MIN_CELL and len(yy) - yy.sum() >= MIN_CELL:
                        c["auc"] = float(roc_auc_score(yy, rk))
                line.append(c)
            cells.append(line)
        out[fam] = {"rows": [STAGE_NAMES[fam][r] for r in HEATMAP_ROWS[fam]], "cols": list(cols), "cells": cells}
    return out


def evaluate(name, truth, train, scores, calib_ok=True, traj=None):
    base = stage_baseline(train, truth)
    boot_idx = boot_persons(truth.pid.to_numpy(), N_BOOT)
    cr = cr_labels(truth, traj) if traj is not None else None
    seeds, ans = {}, None
    for tag, path in scores:
        sc, answered = load_scores(truth, path)
        if ans is None:
            ans, sc_main = answered, sc
        assert answered == ans, f"{path}: 各 seed 答的题不一样"
        seeds[tag] = per_seed(truth, sc, answered, base, calib_ok, boot_idx if tag == scores[0][0] else None, cr)
    main = scores[0][0]
    res = seeds[main]
    summ = {"name": name, "main_seed": main, "seeds": [t for t, _ in scores], "calibration_defined": calib_ok,
            "n_persons": int(truth.pid.nunique()), "n_landmarks": int(len(truth)),
            "answered": {q: ans[q] for q in TRANS},
            "answers_all": all(ans[q] == list(TRANS_HORIZONS) for q in TRANS),
            "questions": {q: TRANS[q][0] for q in TRANS}, "horizons": list(TRANS_HORIZONS), "table_horizons": list(TABLE_H)}
    # 宏平均（6 道题都答了才有）+ 按人 bootstrap 的 CI（同一批重抽样上 6 道题取平均）
    summ["macro"] = {}
    for H in TRANS_HORIZONS:
        d = {f: macro(res, H, f) for f in ("auc", "auc_within_state", "auc_within_state_age", "recall_at_prev",
                                          "spec_at_prev", "balanced_acc_at_prev", "auprc_lift", "sens_at_spec80")}
        if H in TABLE_H and d["auc"] is not None:
            for f, bk in (("auc", "_boot_auc"), ("auc_within_state", "_boot_within"),
                          ("auc_within_state_age", "_boot_within_age")):
                B = np.array([res[f"{q}_{H}y"][bk] for q in TRANS], float).mean(0)
                d[f + "_ci"] = [float(np.nanpercentile(B, 2.5)), float(np.nanpercentile(B, 97.5))]
        # 4 道认知分期题单独再平均一次（AD、死亡有第一部分管着，这里看"分期"本身）
        cg = [res[f"{q}_{H}y"].get("auc") for q in COGNITIVE]
        d["auc_cognitive4"] = float(np.mean(cg)) if all(v is not None for v in cg) else None
        # 竞争风险敏感性的 6 题平均（死亡题本身没有竞争风险，用主口径）
        crv = [res[f"{q}_{H}y"].get("auc" if q == "death" else "auc_cr") for q in TRANS]
        d["auc_cr"] = float(np.mean(crv)) if (d["auc"] is not None and all(v is not None for v in crv)) else None
        summ["macro"][f"{H}y"] = d
    for k in res.values():
        for bk in ("_boot_auc", "_boot_within", "_boot_within_age"):
            k.pop(bk, None)
    summ["results"] = res
    # 三条非模型的尺子，同一套指标、只用训练段：
    #   分期基线         = 训练段里同一当前状态（MMSE 分期 × 认知分期）的发生率
    #   快照基线         = 逻辑回归：当前两个**分期** + 年龄 + 性别
    #   快照基线（连续值）= 逻辑回归：当前访视的 MMSE、全局认知**连续值** + 年龄 + 性别（统一输入的特征，只看这一次访视）
    strata, strata_age = stratum_key(truth), stratum_age_key(truth)
    rulers = [("stage_baseline", base), ("snapshot_baseline", snapshot_baseline(train, truth))]
    cont = snapshot_cont_baseline(train, truth)
    if cont is not None:
        rulers.append(("snapshot_cont_baseline", cont))
    for tag, B in rulers:
        bl = {}
        for q in TRANS:
            for H in TRANS_HORIZONS:
                y_all = truth[label_col(q, H)].to_numpy()
                m = y_all >= 0
                bl[f"{q}_{H}y"] = baseline_metrics(y_all[m].astype(int), B[(q, H)][m], strata[m], strata_age[m])
                if cr is not None and (q, H) in cr:
                    yc = cr[(q, H)]
                    mc = yc >= 0
                    bl[f"{q}_{H}y"]["auc_cr"] = float(roc_auc_score(yc[mc], B[(q, H)][mc]))
        summ[tag] = bl
        summ[tag + "_macro"] = {f"{H}y": {f: float(np.mean([bl[f"{q}_{H}y"][f] for q in TRANS]))
                                          for f in ("auc", "auc_within_state", "auc_within_state_age", "recall_at_prev",
                                                    "sens_at_spec80")}
                                for H in TRANS_HORIZONS}
        if cr is not None:
            for H in TRANS_HORIZONS:
                summ[tag + "_macro"][f"{H}y"]["auc_cr"] = float(np.mean(
                    [bl[f"{q}_{H}y"]["auc" if q == "death" else "auc_cr"] for q in TRANS]))
    bl = summ["stage_baseline"]
    # 各 seed
    single = {t: v for t, v in seeds.items() if t != "ens"}
    if len(single) > 1 and summ["answers_all"]:
        summ["across_seeds"] = {f"macro_{H}y_{f}": {"values": [macro(single[t], H, f) for t in single]}
                                for H in TABLE_H for f in ("auc", "auc_within_state", "auc_within_state_age", "recall_at_prev",
                                                           "sens_at_spec80")}
        for v in summ["across_seeds"].values():
            v["mean"], v["sd"] = float(np.mean(v["values"])), float(np.std(v["values"], ddof=1))
    summ["heatmaps"] = {f"{H}y": heatmaps(truth, sc_main, ans, H, calib_ok) for H in TABLE_H}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="tag=path，第一个是主 seed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--calibration_undefined", action="store_true",
                    help="读法不是概率（visit_heads）：不报 O/E、Brier")
    args = ap.parse_args()
    truth = pd.read_csv(os.path.join(RUN_DIR, f"truth_trans_{args.split}.csv"))
    train = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_train.csv"))
    tp = os.path.join(RUN_DIR, f"traj_truth_{args.split}.json")
    traj = json.load(open(tp)) if os.path.exists(tp) else None          # 竞争风险敏感性要死亡 / AD 年龄
    scores = [tuple(x.split("=", 1)) for x in args.scores]
    s = evaluate(args.name, truth, train, scores, calib_ok=not args.calibration_undefined, traj=traj)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "transitions.json"), "w") as f:
        json.dump(s, f, indent=1)
    mac = s["macro"]
    print(f"{args.name}: " + "  ".join(
        f"{H}y AUC {mac[f'{H}y']['auc']:.3f} within {mac[f'{H}y']['auc_within_state']:.3f}/{mac[f'{H}y']['auc_within_state_age']:.3f}"
        for H in TABLE_H if mac[f"{H}y"]["auc"] is not None) + ("" if s["answers_all"] else "  (not all questions answered)"))


if __name__ == "__main__":
    main()
