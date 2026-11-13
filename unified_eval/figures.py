"""figures.py -- 统一评估第三部分的全部图，只读仓库里的聚合 JSON，在本地画（不需要人级数据）。

    python figures.py            # 每个打包盒的 fig3–fig6 + reports/SHAP_SUMMARY.png（leaderboard.png 由 leaderboard.py 调这里画）

设计规则（用户 2026-10-15 的要求）：每张图一眼能看懂；只放回答问题必需的面板；一个面板一个信息；
参照线最多一两条，直接标在图上；标题里写结论（关键数字），方法放脚注。细节（各种尺子、诊断、审计）进 README 的表和 JSON。
全套图用同一套颜色和用词：实际（"observed"）= 黑，模型 / 模拟 = 蓝；AD = 紫，MMSE<24 = 橙，死亡 = 深灰；
MMSE 分期越差橙色越深（Impaired 就是 MMSE<24）；"cumulative risk"、"AD dementia"、减号用 −。

每个模型的图（fig1 判别、fig2 校准由 evaluate.py / replot.py 画，不在这里）：
  fig3_transitions.png       分期转换：A 2 / 5 / 10 年的 AUC；B 5 年内抓到的转变者（特异度 80%）            所有模型
  fig4_transition_matrix.png 5 年转移矩阵：第一次访视的 MMSE 分期 -> 5 年后的状态，实际 vs 模拟              能生成轨迹的模型
  fig5_trajectories.png      10 年轨迹：A AD、死亡的累计风险；B 活着的人里 MMSE 分期的占比，实际 vs 模拟      能生成轨迹的模型
  fig6_perturbation.png      扰动：A 7 个改动对 AD、MMSE<24、死亡的 5 年优势比（柱子从 1 往外长）；B 认知一起下降之后的 10 年进程  有直接读出的模型
  fig8_timing.png            时间：10 年内真的发生的人，预测时间 − 实际时间，按实际隔了多久分段（timing.py）            有 10 年风险曲线的模型
reports/：leaderboard.png（leaderboard.py 调 leaderboard_figure）、SHAP_SUMMARY.png（这里）。
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from registry import DELPHI_SIM, REPORT_DIR, pkg_dir, zoo_pkgs  # noqa: E402
from report_sections import matrix_gap, matrix_gap_floor  # noqa: E402  （fig4 标题、README、表 5 用同一个定义）

# ----------------------------------------------------------------------------- 样式（全套图共用）

INK, SOFT, FAINT = "#222222", "#6b6b6b", "#d9d9d9"
OBS = "#222222"                                   # 实际
MODEL = "#2166ac"                                 # 模型 / 模拟
H_COL = {2: "#6baed6", 5: "#2171b5", 10: "#08306b"}
STAGE_COL = ("#feedde", "#fdae6b", "#d95f02")     # Normal / Borderline / Impaired：越差越深；Impaired = MMSE<24，和 EP_COL 的橙色一致
EP_COL = {"ad": "#7b3294", "mmse_lt24": "#d95f02", "death": "#525252"}
EXPECTED, OPPOSITE, NEUTRAL = MODEL, "#d73027", "#9e9e9e"       # 图 6 A 全是模型的输出，方向对 = 模型蓝
REF = "#7f7f7f"                                   # 不用模型的参照（逻辑回归）
MINUS = "−"

TRANS_ORDER = ("mmse_lt27", "mmse_lt24", "cogn_lt0", "cogn_ltm1", "ad", "death")
TRANS_LABEL = {"mmse_lt27": "MMSE < 27", "mmse_lt24": "MMSE < 24", "cogn_lt0": "Cognition < 0",
               "cogn_ltm1": f"Cognition < {MINUS}1", "ad": "AD dementia", "death": "Death"}
STAGE_LABEL = ("Normal\n(MMSE ≥27)", "Borderline\n(24–26)", "Impaired\n(<24)")
STATE_NAME = ("Normal", "Borderline", "Impaired", "Dead")
PV_ARMS = ("mmse_b", "cogn_m", "cog2", "age5", "stroke", "comorb3", "apoe")
PV_LABEL = {"mmse_b": "MMSE ↓ (normal → borderline)", "cogn_m": "Cognition ↓ (high → mid)",
            "cog2": "Both ↓ (MMSE and cognition)", "age5": "Age + 5 years", "stroke": "New stroke",
            "comorb3": "Hypertension + diabetes\n+ heart disease", "apoe": "APOE ε4: 0 → 1"}
PV_EP = ("ad", "mmse_lt24", "death")
PV_EP_LABEL = {"ad": "AD dementia", "mmse_lt24": "MMSE < 24", "death": "Death"}
NO_RESPONSE = 0.05                    # |log OR| 小于这个 = 没反应（优势比 0.95–1.05；和第二部分、扰动 v2 同一个门槛）
SHAP_MIN_CARRIERS = 20                # SHAP 图 A：至少这么多人有这条记录才列
SHAP_BIN_COL = ("#c6dbef", "#6baed6", "#08519c")   # SHAP 图 B：模型输出用蓝，值越差越深（不借用 MMSE 分期的橙色）


def _style():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.edgecolor": SOFT, "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": INK,
                         "xtick.labelsize": 10, "ytick.labelsize": 10.5, "legend.fontsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})


def _j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def _pct(ax):
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _spread(ys, gap):
    """把一组标签的纵坐标推开，相邻至少差 gap（保持原来的上下顺序）。"""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# ----------------------------------------------------------------------------- fig3：分期转换

def _answered(res, q, H, key="auc"):
    r = res.get(f"{q}_{H}y", {})
    return r.get(key) if r.get("answered") else None


def fig3_transitions(t, path, note=""):
    """一个面板：6 道越线题在 2 / 5 / 10 年的 AUC，每道题一组横向柱子（浅 → 深 = 2 → 5 → 10 年），柱子从瞎猜（0.5）往右长，
    柱尾写数值；5 年那根带 95% CI。最底下一组是平均。只和瞎猜比（逻辑回归之类的参照留在表 4）。
    只回答一部分题的模型不画平均（和回答 6 道题的模型的平均不可比）。"""
    res = t["results"]
    qs = [q for q in TRANS_ORDER if any(_answered(res, q, H) is not None for H in (2, 5, 10))]
    full = len(qs) == len(TRANS_ORDER)
    rows = TRANS_ORDER + (("_mean",) if full else ())
    y = np.arange(len(rows))[::-1].astype(float)
    if full:
        y[-1] -= 0.35                                 # 平均那一组和上面隔开一点
    fig, a = plt.subplots(figsize=(7.6, 6.4))
    na = dict(va="center", fontsize=9.5, color=SOFT, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    present, lows, highs = set(), [], []
    BH = 0.26                                         # 每根柱子的高度（一组三根）
    for i, q in enumerate(rows):
        hit = False
        for H, dy in ((2, BH), (5, 0.0), (10, -BH)):
            if q == "_mean":
                vals = [_answered(res, q2, H) for q2 in qs]
                v = float(np.mean(vals)) if vals and all(x is not None for x in vals) else None
                ci = t["macro"][f"{H}y"].get("auc_ci") if H == 5 else None
            else:
                v = _answered(res, q, H)
                ci = _answered(res, q, H, "auc_ci") if H == 5 else None
            if v is None:
                continue
            yy = y[i] + dy
            a.barh(yy, v - 0.5, left=0.5, height=BH * 0.92, color=H_COL[H], zorder=2)
            end = max(v, 0.5)                         # 低于瞎猜的柱子往左长，数字仍写在 0.5 线右边
            if ci:
                a.plot(ci, [yy, yy], "-", color=INK, lw=1.1, zorder=3)
                end = max(v, ci[1])
                lows.append(ci[0])
            lows.append(v)
            highs.append(end)
            a.text(end + 0.006, yy, f"{v:.2f}", va="center", fontsize=8.5, color=INK,
                   fontweight="bold" if q == "_mean" else "normal", zorder=4)
            present.add(H)
            hit = True
        if not hit:
            a.text(0.52, y[i], "not answered by this model", **na)
    if full:
        a.axhline((y[-2] + y[-1]) / 2, color=FAINT, lw=1)
    a.axvline(0.5, color=SOFT, lw=1, zorder=1)
    a.text(0.503, y[-1] - 0.7, "chance", fontsize=9, color=SOFT, va="center")
    a.set_xlim(min([0.45] + [x - 0.02 for x in lows]), max(1.0, max(highs + [0.9]) + 0.06))
    a.set_ylim(y[-1] - 0.95, y[0] + 0.55)

    def lab(q):
        r = res.get(f"{q}_5y") or {}
        n = r.get("n_persons_pos") if r.get("answered") else None
        return TRANS_LABEL[q] + (f"\nn = {n}" if n else "")
    a.set_yticks(y, [lab(q) for q in TRANS_ORDER] + (["Average"] if full else []))
    if full:
        a.get_yticklabels()[-1].set_fontweight("bold")
    a.tick_params(axis="y", length=0)
    a.set_xlabel("AUC (bars start at 0.5 = chance; 1 = perfect ranking)")
    a.legend(handles=[Patch(facecolor=H_COL[H], label=f"within {H} years") for H in (2, 5, 10) if H in present],
             loc="upper left", bbox_to_anchor=(-0.02, -0.1), ncol=3, handletextpad=0.4, columnspacing=1.2)
    mac = t["macro"]
    if full:
        head = (f"{t['name']}: who crosses each stage line first?\naverage AUC {mac['5y']['auc']:.3f} within 5 years "
                f"({mac['2y']['auc']:.3f} within 2, {mac['10y']['auc']:.3f} within 10)")
    else:
        head = f"{t['name']}: who crosses each stage line first?\nanswers {len(qs)} of the 6 questions (no average)"
    fig.suptitle(head, x=0.01, ha="left", fontsize=12, y=1.0)
    cr = (mac.get("10y") or {}).get("auc_cr")
    foot = (f"Validation set, {t['n_persons']} people; every visit is a starting point; each row = the first crossing of that "
            "line;\nn = people who crossed it within 5 years; black line = 95% CI (5 years).")
    if full and cr is not None:
        foot += (f"\nEach horizon counts only starting points whose answer is known; counting people who died first as "
                 f"non-crossers\nlowers the 10-year average AUC from {mac['10y']['auc']:.2f} to {cr:.2f}.")
    if note:
        foot += "\n" + note
    fig.text(0.01, -0.1, foot, fontsize=9, color=SOFT, va="top")
    _save(fig, path)


# ----------------------------------------------------------------------------- fig4：转移矩阵

MATRIX_FAM = {   # 分期的名字（行）和第 5 年的状态（列）
    "mmse": ("MMSE", ("Normal\n(MMSE ≥27)", "Borderline\n(24–26)", "Impaired\n(<24)"), ("Normal", "Borderline", "Impaired", "Dead")),
    "cogn": ("Global cognition", ("High\n(≥0)", f"Mid\n({MINUS}1 to 0)", f"Low\n(<{MINUS}1)"), ("High", "Mid", "Low", "Dead")),
}


def _matrix_ax(ax, M, title, rows, cols, ylabel=None, n_approx=None, show_y=True, cmap="Blues"):
    """一个 3 × 4 的矩阵。实际 = 灰，模拟 = 蓝（全套图的颜色规则）；格子里的字按底色深浅选黑或白。"""
    A = np.full((3, 4), np.nan)
    for i, row in enumerate(M):
        if row is not None:
            A[i] = row
    cm = matplotlib.colormaps[cmap]
    ax.imshow(np.nan_to_num(A, nan=0.0), cmap=cm, vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(4):
            if np.isnan(A[i, j]):
                ax.text(j, i, "too few\npeople", ha="center", va="center", fontsize=9, color=SOFT)
                continue
            v = A[i, j]
            r, g, b_, _ = cm(v)
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b_ < 0.45
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=12, color="white" if dark else INK)
    ax.set_xticks(range(4), cols)
    labels = [s if not n_approx or n_approx[i] is None else f"{s}\nn≈{n_approx[i]}" for i, s in enumerate(rows)]
    ax.set_yticks(range(3), labels if show_y else [""] * 3)
    ax.set_xlabel("state 5 years later")
    if show_y and ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def fig4_transition_matrix(tr, path):
    """5 年转移矩阵，每个分期一行：MMSE、全局认知。左 = 实际（灰），右 = 模拟（蓝）。
    标题给每个分期的平均每格差和"最明显的一格"（按抽样误差衡量），脚注给噪声下限。"""
    fams = [f for f in ("mmse", "cogn") if f in (tr.get("matrix_5y") or {})]
    fig, axes = plt.subplots(len(fams), 2, figsize=(9.5, 4.0 * len(fams)), gridspec_kw={"wspace": 0.08, "hspace": 0.55},
                             squeeze=False)
    head = [f"{tr['name']}: 5-year transitions from the first visit"]
    floors = []
    for k, fam in enumerate(fams):
        M = tr["matrix_5y"][fam]
        name, rows, cols = MATRIX_FAM[fam]
        letter = "AB"[k] if len(fams) > 1 else ""
        _matrix_ax(axes[k, 0], M["observed"], f"{letter}  {name}: observed".strip(), rows, cols,
                   ylabel=f"{name} stage at the first visit", n_approx=M.get("n_approx"), show_y=True, cmap="Greys")
        _matrix_ax(axes[k, 1], M["predicted"], "simulated by the model", rows, cols, show_y=False, cmap="Blues")
        gap, main = matrix_gap(M)
        line = f"{name}: average cell gap {100 * gap:.0f} points"
        if main:
            i, j, o, p = main
            line += f"; clearest gap {rows[i].splitlines()[0]} → {cols[j]}, {o:.0%} observed vs {p:.0%} simulated"
        head.append(line)
        fl = matrix_gap_floor(M)
        if fl:
            floors.append(f"{name} about {100 * fl:.0f}")
    fig.suptitle("\n".join(head), x=0.01, ha="left", fontsize=12, y=0.99 if len(fams) > 1 else 1.09)
    fig.text(0.01, -0.03, "Rows add up to 100% (up to rounding); n rounded to 10. Observed: death share from Kaplan–Meier "
             "(allows for loss to follow-up),\nthe living share split by the stage of people still followed at 5 years. "
             f"Simulated: the same people, {tr.get('n_mc', 100)} simulated futures each.\nAverage cell gap: the three rows count equally"
             + (f"; sampling noise alone would give even a perfect simulator a gap of {', '.join(floors)} points." if floors else ".")
             + "\nClearest gap = the largest gap relative to its sampling noise.", fontsize=9, color=SOFT, va="top")
    _save(fig, path)


# ----------------------------------------------------------------------------- fig5：10 年轨迹

def fig5_trajectories(tr, path):
    inc, occ = tr["incidence"], tr["stage_over_time"]["mmse"]
    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.8, 1, 1], wspace=0.12)   # 第 2 列空着，放 A 的线尾标签和 B 的纵轴名
    a = fig.add_subplot(gs[0])
    yrs = np.array(inc["years"])
    ends, labs, cols = [], [], []
    for q in ("ad", "death"):
        d, col = inc[q], EP_COL[q]
        a.fill_between(yrs, d["observed_lo"], d["observed_hi"], color=col, alpha=0.12, lw=0)
        a.plot(yrs, d["observed"], "-", color=col, lw=2.4)
        a.plot(yrs, d["simulated"], "--", color=col, lw=2.4)
        name = "AD" if q == "ad" else "Death"
        ends += [d["observed"][-1], d["simulated"][-1]]
        labs += [f"{name} {d['observed'][-1]:.0%}", f"{name} {d['simulated'][-1]:.0%} (sim.)"]
        cols += [col, col]
    a.set_ylim(0, None)
    gap = 0.06 * a.get_ylim()[1]                      # 标签之间至少隔 6% 的纵轴高度
    for yv, lab, col in zip(_spread(ends, gap), labs, cols):
        a.text(10.3, yv, lab, color=col, va="center", fontsize=9.5, clip_on=False)
    a.legend(handles=[Line2D([], [], color=OBS, lw=2.2, label="observed (band: 95% CI)"),
                      Line2D([], [], color=OBS, lw=2.2, ls="--", label="simulated by the model")],
             loc="upper left", bbox_to_anchor=(-0.02, -0.17), ncol=1)
    a.set_xlim(0, 10)
    a.set_xticks(range(0, 11, 2))
    _pct(a)
    a.set_xlabel("years after the first visit")
    a.set_ylabel("cumulative risk")
    yrs2 = np.array(occ["years"])
    axes = []
    for k, (key, ttl) in enumerate((("observed", "Observed"), ("predicted", "Simulated"))):
        ax = fig.add_subplot(gs[k + 2], sharey=axes[0] if axes else None)
        axes.append(ax)
        rows = occ[key]
        ok = [i for i, r in enumerate(rows) if r is not None and all(v is not None for v in r[:3]) and sum(r[:3]) > 0]
        Y = np.array([np.array(rows[i][:3]) / sum(rows[i][:3]) for i in ok]).T
        ax.stackplot(yrs2[ok], Y, colors=STAGE_COL, labels=("Normal", "Borderline", "Impaired"), edgecolor="white", lw=0.5)
        ax.text(9.6, Y[0][-1] / 2, f"Normal\n{Y[0][-1]:.0%}", ha="right", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("years after the first visit")
        ax.set_title(ttl, fontsize=11, loc="center", fontweight="normal")
        _pct(ax)
        if k == 1:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("share of living people")
    axes[1].legend(handles=[Patch(facecolor=STAGE_COL[k], edgecolor=SOFT, lw=0.6, label=nm)
                            for k, nm in ((2, "Impaired"), (1, "Borderline"), (0, "Normal"))],
                   loc="upper left", bbox_to_anchor=(1.01, 1.0))
    top = axes[0].get_position().y1 + 0.085
    fig.text(a.get_position().x0, top, "A  AD dementia and death", fontsize=12, fontweight="bold")
    fig.text(axes[0].get_position().x0, top, "B  MMSE stage among people still alive", fontsize=12, fontweight="bold")
    ad, de = inc["ad"], inc["death"]
    fig.suptitle(f"{tr['name']}: simulated futures vs observed, 10 years from the first visit\n"
                 f"by year 10: AD {ad['simulated'][-1]:.0%} simulated vs {ad['observed'][-1]:.0%} observed; "
                 f"death {de['simulated'][-1]:.0%} vs {de['observed'][-1]:.0%}",
                 x=0.01, ha="left", fontsize=12, y=1.13)
    nk, ob = occ.get("n_known"), occ["observed"]
    alive = [round(nk[k] * (1 - (ob[k][3] or 0))) if nk and ob[k] is not None else None for k in range(len(ob))]
    n_txt = (f" ({alive[1]} people at year 1, {alive[-1]} at year 10)" if alive and alive[1] and alive[-1] else "")
    fig.text(0.01, -0.2, f"{tr.get('n_persons', 443)} people, {tr.get('n_mc', 100)} simulated futures each. A: observed "
             "allows for loss to follow-up (Aalen–Johansen). B: observed = people alive and still followed that year"
             f"{n_txt};\nsimulated = living simulated people.", fontsize=9, color=SOFT, va="top")
    _save(fig, path)


# ----------------------------------------------------------------------------- fig6：扰动

def _forest_cell(ax, y, c):
    """一格：一根从优势比 1 往外长的横向柱子。有预设方向的格子按得分上色（黑 = 方向对，红 = 方向反，空心 = 没反应（柱子太短，加空心圈），
    和 README / 表 6 的 log OR 得分同一个门槛）；没有预设方向的格子不打分，画成细的灰柱子。黑线 = 95% CI。"""
    lo = c["d_logor"]["5"]
    ci = c.get("d_logor_ci", {}).get("5")
    v = math.exp(lo)
    if c.get("prespec") != "earlier":
        ax.barh(y, v - 1, left=1, height=0.32, color=NEUTRAL, zorder=2)
        if ci:
            ax.plot(np.exp(ci), [y, y], "-", color=NEUTRAL, lw=1.0, zorder=3)
        return
    if abs(lo) < NO_RESPONSE:
        ax.barh(y, v - 1, left=1, height=0.62, color="white", edgecolor=NEUTRAL, lw=1.4, zorder=2)
        ax.plot(v, y, "o", ms=7, mfc="white", mec=NEUTRAL, mew=1.4, zorder=4)   # 柱子几乎没有宽度，加个空心圈看得见
    else:
        ax.barh(y, v - 1, left=1, height=0.62, color=EXPECTED if lo > 0 else OPPOSITE, zorder=2)
    if ci:
        ax.plot(np.exp(ci), [y, y], "-", color=INK, lw=1.2, zorder=3)


def _or_axis(lo_vals):
    """优势比的对数轴：默认 0.4–80，数据超出就放宽（不能悄悄把点裁掉）；刻度从 1-2-5 序列里取。"""
    lo_vals = [v for v in lo_vals if v is not None and np.isfinite(v)]
    xmin = min([0.4] + [math.exp(v) * 0.8 for v in lo_vals])
    xmax = max([80] + [math.exp(v) * 1.25 for v in lo_vals])
    ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if xmin <= t <= xmax]
    return xmin, xmax, ticks


def fig6_perturbation(J, path, future=None):
    """A：7 个改动 × 3 个终点的 5 年优势比（各 seed 平均；future = "rollout" 的模型是改完之后模拟的未来数出来的，"exact" 精确算，其余直接读出）。
    B：MMSE 和认知一起下降之后 AD、MMSE<24 的 10 年累计风险：能生成轨迹的模型用模拟的未来。
    标题先回答用户的问题：认知下降之后 AD 和 MMSE<24 是不是一起变坏（10 年内平均无事件时间少几年）。
    不能模拟的 Delphi 只有 A。Delphi 的 A、B 读法不同（直接读出的 AD 风险接近 0，优势比大；模拟的未来里风险挪得少），脚注写明。"""
    from report_sections import pv2_logor_scores
    D, R = J.get("direct") or {}, J.get("rollout") or {}
    gen = R.get("status") == "ok"
    curve = (R if gen else D).get("effects", {}).get("cog2", {})
    has_b = all((curve.get(q) or {}).get("cif_orig") is not None for q in ("ad", "mmse_lt24")) and \
        (gen or J.get("family") == "landmark")
    c2ad = (D.get("effects", {}).get("cog2") or {}).get("ad") or {}
    responsive = bool(c2ad.get("defined")) and abs(c2ad["d_logor"]["5"]) >= NO_RESPONSE
    fig = plt.figure(figsize=(13 if has_b else 9.5, 4.5))
    gs = fig.add_gridspec(1, 5 if has_b else 3, width_ratios=([1, 1, 1, 0.42, 1.55] if has_b else [1, 1, 1]), wspace=0.1)
    arms = [a for a in PV_ARMS if a in D.get("effects", {})]
    shown = {a: {q: D["effects"][a][q] for q in PV_EP if q in D["effects"][a]} for a in arms}
    vals = []
    for a in arms:
        for q, c in shown[a].items():
            if c.get("defined"):
                vals += [c["d_logor"]["5"]] + list(c.get("d_logor_ci", {}).get("5") or [])
    xmin, xmax, ticks = _or_axis(vals)
    y = np.arange(len(arms))[::-1]
    axes = []
    for k, q in enumerate(PV_EP):
        ax = fig.add_subplot(gs[k], sharey=axes[0] if axes else None)
        axes.append(ax)
        for i, arm in enumerate(arms):
            c = shown[arm].get(q) or {}
            if not c.get("defined"):
                ax.text(1, y[i], "—", ha="center", va="center", color=SOFT)
                continue
            _forest_cell(ax, y[i], c)
        ax.axvline(1, color=SOFT, lw=1, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(PV_EP_LABEL[q], fontsize=11.5, loc="center")
        ax.set_xlabel("odds ratio\n(1 = no change)")
        if k == 0:
            ax.set_yticks(y, [PV_LABEL[a] for a in arms])
            ax.tick_params(axis="y", length=0)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
    axes[0].set_ylim(-0.7, len(arms) - 0.3)
    top = axes[0].get_position().y1 + 0.1
    letter = "A  " if has_b else ""
    left = axes[0].get_position().x0 - (0.17 if has_b else 0.23)          # 和最长的行名左边对齐
    how_a = {"rollout": "Simulated futures", "exact": "Exact multistate risk"}.get(future, "Direct readout")
    fig.text(left, top, f"{letter}{how_a}: what each change does to the 5-year odds", fontsize=12, fontweight="bold")
    sc, _ = pv2_logor_scores({"effects": shown})
    count = f"{'A: ' if has_b else ''}{sc['right']} of {sc['n_scored']} expected directions right"
    n = (curve.get("ad") or {}).get("n", "")
    if has_b:
        b = fig.add_subplot(gs[4])
        yrs = np.arange(0, 11)
        ends = {}
        for q in ("ad", "mmse_lt24"):
            c = curve[q]
            o = np.concatenate([[0], c["cif_orig"]])
            p = np.concatenate([[0], c["cif_pert"]])
            ends[q] = (o[-1], p[-1])
            b.plot(yrs, o, "--", color=EP_COL[q], lw=1.6, alpha=0.45)
            b.plot(yrs, p, "--", color=EP_COL[q], lw=2.6)
        b.set_ylim(0, None)
        ys = _spread([ends["ad"][1], ends["mmse_lt24"][1]], 0.12 * b.get_ylim()[1])
        for (q, yv) in zip(("ad", "mmse_lt24"), ys):
            o, p = ends[q]
            b.text(10.3, yv, f"{PV_EP_LABEL[q]}\n{o:.0%} → {p:.0%}", color=EP_COL[q], va="center", fontsize=10,
                   clip_on=False)
        same = all(abs(ends[q][1] - ends[q][0]) < 0.005 for q in ends)
        if same:
            b.text(9.8, 0.02 * b.get_ylim()[1], "no change: the curves coincide", ha="right", va="bottom",
                   fontsize=9.5, color=SOFT)
        b.legend(handles=[Line2D([], [], color=INK, lw=2.6, ls="--", label="after 'both ↓'"),
                          Line2D([], [], color=INK, lw=1.6, ls="--", alpha=0.45, label="record unchanged")], loc="upper left")
        b.set_xlim(0, 10)
        b.set_xticks(range(0, 11, 2))
        _pct(b)
        b.set_xlabel("years after the change")
        b.set_ylabel("cumulative risk")
        ttl = "B  'Both ↓' in simulated futures" if gen else "B  'Both ↓': the model's risk over 10 years"
        fig.text(b.get_position().x0 - 0.04, top, ttl, fontsize=12, fontweight="bold")
        src = "simulated futures" if gen else "the model's own risk"
        if same:
            head2 = f"after 'both ↓': no change (the model does not react to MMSE or cognition); {count}"
        else:
            m24, ad = -curve["mmse_lt24"]["d_rmst"], -curve["ad"]["d_rmst"]
            head2 = (f"after 'both ↓' ({src}): 10-year time free of MMSE < 24 {m24:.1f} y shorter, of AD {ad:.1f} y shorter "
                     f"(averaged over everyone, mostly from extra cases); {count}")
    elif future == "rollout":
        head2 = f"{count} (5-year odds, counted from simulated futures after the edit)"
    elif future == "exact":
        head2 = f"{count} (5-year odds, computed exactly from the multistate model)"
    else:
        head2 = f"{count} (5-year odds); no time course: this model cannot simulate futures"
    hs = [Patch(facecolor=EXPECTED, label="expected direction"),
          Patch(facecolor=OPPOSITE, label="opposite to expected"),
          Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=NEUTRAL, mew=1.4, label="no response (odds ratio 0.95–1.05)"),
          Patch(facecolor=NEUTRAL, label="no pre-specified direction (not scored; thin bar)")]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(left, -0.2), ncol=4, fontsize=9.5)
    ci_txt = "Bars start at odds ratio 1 (no change); black lines = 95% CI."
    if gen and responsive:
        foot = (f"A and B use different readouts. A: the model's risk read right after the edit (the same readout for every model). "
                f"B: {n} people with normal MMSE and high cognition,\nsimulated futures in which later visits are regenerated. "
                "Delphi's directly read AD risk is near zero, so its odds ratios are large while the simulated risks move less. "
                f"{ci_txt}")
    elif gen:
        foot = (f"A: the model's risk read right after the edit. B: {n} people with normal MMSE and high cognition, simulated futures. "
                f"{ci_txt}\nThis model only sees demographics and age, so the MMSE and cognition edits change nothing in A or B "
                "(as expected).")
    elif has_b:
        foot = (f"A and B: the model's own risk, read right after the edit. B: {n} people with normal MMSE and high cognition "
                f"at the 4th visit. {ci_txt}")
    elif future == "rollout":
        foot = ("Risk = share of the model's simulated futures (from the edited 4th visit) with the event, "
                f"smoothed as (k + 0.5) / (n + 1). {ci_txt}")
    elif future == "exact":
        foot = f"Risk computed exactly from the multistate model at the edited 4th visit. {ci_txt}"
    else:
        foot = f"The model's risk read right after the edit. {ci_txt}"
        if not responsive and "staticonly" in J.get("name", ""):
            foot += "\nThis model only sees demographics and age, so the MMSE and cognition edits change nothing (as expected)."
    if has_b:
        foot += ("\nYears shorter = the area between the two curves over 10 years (time without the event lost, averaged over all "
                 f"{n} people, including those who never cross).")
    foot += "\nThe directions were written down before the results; the sizes of the effects are not checked against data."
    if J.get("ranked") is False and J.get("tokenization") == "snapshot":
        foot += " Not ranked: in the snapshot tokenization the readout position is not supervised in training."
    fig.text(left, -0.26, foot, fontsize=9, color=SOFT, va="top")
    fig.suptitle(f"{J['name']}: edit the record at the 4th visit (one change or a combination)\n{head2}",
                 x=left, ha="left", fontsize=12, y=top + 0.2)
    _save(fig, path)


# ----------------------------------------------------------------------------- 根目录：SHAP

SHAP_NAME = {"COGN": "Cognition", "MMSE": "MMSE", "DBP": "Diastolic BP", "SBP": "Systolic BP", "BMI": "BMI",
             "GLU": "Glucose", "HBA1C": "HbA1c", "ANTIHYP": "Blood-pressure drug", "STATIN": "Statin", "LDL": "LDL",
             "HDL": "HDL", "GFR": "Kidney function", "PSQI": "Sleep quality", "DIABRX": "Diabetes drug", "ADRX": "AD drug",
             "CRP": "CRP", "IL6": "IL-6", "TNFA": "TNF-α", "APNEA": "Sleep-apnoea risk", "APOE": "APOE", "EDU": "Education",
             "RACE": "Race", "HISP": "Hispanic", "SMOKING": "Smoking", "ALCOHOL": "Alcohol", "STUDY": "Study",
             "HTN": "Hypertension", "DM": "Diabetes", "CHF": "Heart failure", "CLAUD": "Claudication", "HEART": "Heart disease"}
SHAP_BIN = {"normal": "normal", "borderline": "borderline", "impaired": "impaired", "high": "high", "mid": "mid",
            "low": "low", "elevated": "elevated", "pre": "pre-diabetic", "diabetic": "diabetic", "ON": "on", "OFF": "off",
            "good": "good", "poor": "poor", "verypoor": "very poor", "optimal": "optimal", "mild": "mildly low",
            "ckd": "CKD", "ONSET": "new diagnosis", "PREVALENT": "at enrolment", "e4_0": "no ε4", "e4_1": "one ε4",
            "e4_2": "two ε4", "e2carrier": "ε2 carrier", "unk": "unknown", "lt12": "<12 years", "12to15": "12–15 years",
            "16to19": "16–19 years", "ge20": "≥20 years", "yes": "yes", "no": "no", "never": "never", "former": "former",
            "current": "current", "none": "none", "light": "light", "heavy": "heavy"}
SHAP_SINGLE = {"Male": "Sex: male", "Female": "Sex: female", "STROKE": "Stroke", "DEPRESSION": "Depression",
               "AD_DX": "AD diagnosis"}


def _tok_label(tok):
    if tok in SHAP_SINGLE:
        return SHAP_SINGLE[tok]
    fam, _, b = tok.partition("_")
    if not b:
        return SHAP_NAME.get(fam, fam.capitalize())
    return f"{SHAP_NAME.get(fam, fam)}: {SHAP_BIN.get(b, b)}"


def _ratio_axis(ax, axis="x"):
    """对数刻度上标 ×1、×2、×4、×8。"""
    loc, fmt = FixedLocator([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]), FuncFormatter(lambda v, _: f"×{v:g}")
    if axis == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(NullLocator())


def _shap_x(fam, g, bn):
    v = fam.get(g, {}).get(f"{g}_{bn}")
    return None if v is None else math.exp(v)


def shap_summary(J, path, S=None, q="ad", top=10, headline=None, groups=None):
    """J = 一个包的 shap.json（Delphi：shap_delphi.py；按变量拆账的：shap_vars.py，格式对齐）。
    A：把 AD 速率推得最高的 10 种记录（按"有这条记录的人"平均，至少 20 人有）。
    B：同一个量的几档放在一起：值越差推得越高 = 看的是值；"正常"也推得高 = 只在意测没测过。
    S（reports/SHAP_SUMMARY.json）给了的话，脚注加一句和另两个 Delphi 的一致性（reports/ 那张图用）。"""
    model = J["name"]
    lm = J.get("family") == "landmark"
    unit = "variable" if lm else "record"
    Q = J["questions"][q]
    n_risk = Q["n_at_risk"]
    toks = sorted([(t["mean_person_phi"], t["token"], t["n_persons"]) for t in Q["tokens"]
                   if t["n_persons"] >= SHAP_MIN_CARRIERS], reverse=True)[:top]
    fam = {f["family"]: {b_["token"]: b_["mean_person_phi"] for b_ in f["bins"]} for f in Q.get("families", [])}
    carriers = {b_["token"]: b_["n_persons"] for f in Q.get("families", []) for b_ in f["bins"]}
    carriers.update({t["token"]: t["n_persons"] for t in Q["tokens"]})
    age = Q.get("age5", {})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.3})
    fig.subplots_adjust(left=0.27, right=0.98)
    y = np.arange(len(toks) + 2)[::-1]
    for i, (phi, tok, n) in enumerate(toks):
        a.barh(y[i], math.exp(phi) - 1, left=1, height=0.62, color=MODEL)
        a.text(math.exp(phi) * 1.04, y[i], f"×{math.exp(phi):.1f}", va="center", fontsize=9.5)
    av = math.exp(age.get("mean", 0))
    a.barh(y[-1], av - 1, left=1, height=0.62, color="white", edgecolor="#525252", hatch="///")
    a.text(av * 1.04, y[-1], f"×{av:.1f}", va="center", fontsize=9.5)
    a.set_yticks(list(y[:len(toks)]) + [y[-1]],
                 [f"{_tok_label(t)}  ({n} of {n_risk})" for _, t, n in toks] + ["For scale: whole history 5 years older"])
    _ratio_axis(a, "x")
    a.set_xlim(0.9, max(10, max([math.exp(t[0]) for t in toks] + [1]) * 1.4))
    a.axvline(1, color=SOFT, lw=1)
    a.set_xlabel(f"model's AD rate, with ÷ without this {unit} (SHAP)")
    a.set_title(f"A  {unit.capitalize()}s that raise the AD rate the most", fontsize=12)
    groups = groups or [("MMSE", ("impaired", "borderline", "normal")), ("COGN", ("low", "mid", "high")),
                        ("DBP", ("high", "elevated", "normal"))]
    x0, xt, xl, hi = 0, [], [], 1.0
    for g, bins in groups:
        for k, bn in enumerate(bins):
            r = _shap_x(fam, g, bn)
            if r is None:
                continue
            hi = max(hi, r)
            nc = carriers.get(f"{g}_{bn}")
            few = nc is not None and nc < SHAP_MIN_CARRIERS
            b.bar(x0 + k, r - 1, bottom=1, width=0.75, color=SHAP_BIN_COL[2 - k], edgecolor=SOFT, lw=0.6,
                  hatch="///" if few else None)
            b.text(x0 + k, r * 1.06 if r >= 1 else r / 1.06, f"×{r:.1f}" if r >= 0.095 else f"×{r:.2f}", ha="center",
                   va="bottom" if r >= 1 else "top", fontsize=9)
            xt.append(x0 + k)
            xl.append(f"{bn} (n={nc})" if few else bn)
        x0 += 4
    top_y = max(14, hi * 2.2)
    for k, (g, _) in enumerate(groups):
        b.text(4 * k + 1, top_y / 1.25, SHAP_NAME[g], ha="center", fontsize=10.5, fontweight="bold")
    _ratio_axis(b, "y")
    lo = min([_shap_x(fam, g, bn) or 1 for g, bs in groups for bn in bs] + [1])
    b.set_ylim(min(0.6, lo / 1.8), top_y)                  # 最低的柱和它下面的数字都要在图里
    b.axhline(1, color=SOFT, lw=1)
    b.set_xticks(xt, xl, fontsize=9.5, rotation=40, ha="right")
    b.set_ylabel(f"model's AD rate, with ÷ without (SHAP)")
    b.set_title("B  Does the value matter, or only that it was measured?", fontsize=12)

    def pair(g, worst, best, name):
        w, bst = _shap_x(fam, g, worst), _shap_x(fam, g, best)
        return None if w is None or bst is None else f"{name} {worst} ×{w:.1f} vs {best} ×{bst:.1f}"
    line2 = "; ".join(x for x in (pair("MMSE", "impaired", "normal", "MMSE"), pair("COGN", "low", "high", "cognition"),
                                  pair("DBP", "high", "normal", "diastolic BP")) if x)
    fig.suptitle(headline or f"What {model} looks at to predict AD (SHAP)\n{line2}", x=0.01, ha="left", fontsize=12.5,
                 fontweight="bold" if headline else "normal", y=1.03 if headline else 1.07)
    if lm:
        ns = len(J.get("checks", {}).get("seeds", [])) or 1
        how = (f"SHAP by variable for {model} ({f'{ns}-seed ensemble' if ns > 1 else 'one trained model'}): "
               "removing a variable = never measured up to the 4th visit; "
               f"{n_risk} people at risk of AD at the 4th visit.\nBars = people grouped by the last value measured "
               "(the Delphi cut points). A lists groups of at least 20 people. B: darker = worse value; hatched = fewer than 20 people.")
    else:
        ns = len(J.get("checkpoints", [])) or 1
        how = (f"SHAP (the upstream Delphi method) for {model}, {'averaged over ' + str(ns) + ' seeds' if ns > 1 else 'seed 42'}; "
               f"{n_risk} people at risk of AD at the 4th visit. A lists records that at least 20 people have.\n"
               "B: darker = worse value; hatched = fewer than 20 people.")
    if S is not None:
        SQ = S["questions"][q]
        rhos = [v["rho"] for k, v in SQ.get("model_agreement", {}).get("pairs", {}).items() if model in k.split("~")]
        others = [f"{m2.replace('delphi-', '')} ×{math.exp(d2['bins']['MMSE_impaired']):.1f}"
                  for f in SQ["consensus_families"] if f["family"] == "MMSE"
                  for m2, d2 in f["per_model"].items() if m2 != model and d2.get("bins", {}).get("MMSE_impaired") is not None]
        if rhos and others:
            how += (f"\nnodedup and fullvisit rank records by total |SHAP| much the same way (Spearman {min(rhos):.2f}–"
                    f"{max(rhos):.2f}), but their per-record effects are smaller (MMSE impaired: {', '.join(others)}).")
    fig.text(0.01, -0.1, how, fontsize=9, color=SOFT, va="top")
    _save(fig, path)


# ----------------------------------------------------------------------------- reports/：排行榜总图

def leaderboard_figure(rows, tby, trajby, unranked, path, roles=None):
    """rows = [(名字, 第一部分 metrics, 第一部分是否只作对照)]，按第一部分的名次。四个面板，每个模型每个面板一两根柱子：
    A 第一部分 2 年 4 终点平均 AUC；B 第三部分 5 年 6 道分期转换题平均 AUC（两条逻辑回归的参照线直接标在图上）；
    C 模拟的 10 年 AD、死亡累计风险 ÷ 实际（浅色带 = 实际的 95% CI）；D 5 年转移矩阵平均每格差（灰带 = 抽样噪声的下限）。
    C、D 只有能生成轨迹的模型。空心柱 = 这个面板里不排名；斜体名字 = 参照模型（roles：只看人口学、只看当前访视）。"""
    roles = roles or {}
    n = len(rows)
    fig, (a, b, c, d) = plt.subplots(1, 4, figsize=(15.5, 0.34 * n + 1.9), sharey=True,
                                     gridspec_kw={"width_ratios": [1, 1, 0.8, 0.62], "wspace": 0.16})
    # 底边距按英寸留（刻度 + 两行横轴标签约 1 英寸）：行少的时候图矮，按比例留会让横轴标签和脚注叠在一起
    fig.subplots_adjust(left=0.19, right=0.99, bottom=min(0.35, 1.05 / (0.34 * n + 1.9)))
    yy = np.arange(n)[::-1]
    ref, inc0 = None, None
    for i, (name, m, ref_only) in enumerate(rows):
        y = yy[i]
        unr = name in unranked
        v, ci = m["macro_auc"]["2y"], m.get("macro_auc_ci_2y")
        hollow = unr or ref_only
        a.barh(y, v - 0.5, left=0.5, height=0.62, color="white" if hollow else MODEL, edgecolor=MODEL, lw=1.3, zorder=2)
        if ci:
            a.plot(ci, [y, y], "-", color=INK, lw=1.2, alpha=0.5 if hollow else 1, zorder=3)
        t = tby.get(name)
        if t is not None and t.get("answers_all"):
            ref = ref or t
            mv = t["macro"]["5y"]
            b.barh(y, mv["auc"] - 0.5, left=0.5, height=0.62, color="white" if unr else MODEL, edgecolor=MODEL, lw=1.3, zorder=2)
            if mv.get("auc_ci"):
                b.plot(mv["auc_ci"], [y, y], "-", color=INK, lw=1.2, alpha=0.5 if unr else 1, zorder=3)
        elif t is not None:
            b.text(0.51, y, "answers 4 of 6 questions", va="center", fontsize=9, color=SOFT)
        tr = trajby.get(name)
        if tr and tr.get("applicable") and tr.get("incidence"):
            inc0 = inc0 or tr["incidence"]
            for q, dy in (("ad", 0.16), ("death", -0.16)):
                o, s_ = tr["incidence"][q]["observed"][-1], tr["incidence"][q]["simulated"][-1]
                if o:
                    c.barh(y + dy, s_ / o - 1, left=1, height=0.3, color=EP_COL[q], zorder=2)
            M = (tr.get("matrix_5y") or {}).get("mmse")
            if M:
                d.barh(y, 100 * matrix_gap(M)[0], height=0.62, color=MODEL, zorder=2)
        else:
            for ax in (c, d):
                ax.text(0.03, y, "cannot simulate", va="center", ha="left", fontsize=9, color=SOFT, transform=ax.get_yaxis_transform(),
                        bbox=dict(facecolor="white", edgecolor="none", pad=1), zorder=5)
    a.set_yticks(yy, [r[0] for r in rows], fontsize=9.5)
    for lab, (name, _, _) in zip(a.get_yticklabels(), rows):
        if name in unranked:
            lab.set_color(SOFT)
        if name in roles:
            lab.set_style("italic")
    for ax in (a, b):
        ax.set_xlim(0.5, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel("AUC (95% CI)\nbars start at 0.5 = chance")
    c.axvline(1, color=SOFT, lw=1, zorder=0)
    if inc0:                                          # 一条带 = AD、死亡两个区间里宽的那个（两者几乎重合；精确值在脚注）
        lo_ = min(inc0[q]["observed_lo"][-1] / inc0[q]["observed"][-1] for q in ("ad", "death"))
        hi_ = max(inc0[q]["observed_hi"][-1] / inc0[q]["observed"][-1] for q in ("ad", "death"))
        c.axvspan(lo_, hi_, color=FAINT, alpha=0.7, lw=0, zorder=0)
        c.text(1.0, n - 0.3, "observed\n95% CI", ha="center", va="bottom", fontsize=8.5, color=SOFT,
               bbox=dict(facecolor="white", edgecolor="none", pad=1))
    floor = None
    for name, _, _ in rows:
        M = ((trajby.get(name) or {}).get("matrix_5y") or {}).get("mmse")
        if M:
            floor = matrix_gap_floor(M)
            break
    if floor:
        d.axvspan(0, 100 * floor, color=FAINT, alpha=0.7, lw=0, zorder=0)
        d.text(100 * floor + 0.3, n - 0.3, "noise\nfloor", ha="left", va="bottom", fontsize=8.5, color=SOFT)
    c.set_xlim(0, 2)
    c.set_xticks([0, 0.5, 1, 1.5, 2])
    c.set_xlabel("simulated ÷ observed\n(bars start at 1 = matches)")
    c.legend(handles=[Patch(facecolor=EP_COL["ad"], label="AD"), Patch(facecolor=EP_COL["death"], label="death")],
             loc="lower right", ncol=1, handletextpad=0.1)
    d.set_xlim(0, 20)
    d.set_xticks([0, 5, 10, 15, 20])
    d.set_xlabel("percentage points\n(0 = matches)")
    a.set_title("A  Next 2 years (Part 1)\n    mean AUC of 4 outcomes", fontsize=12)
    b.set_title("B  Stage transitions (Part 3)\n    mean AUC of 6 lines, 5 years", fontsize=12)
    c.set_title("C  Simulated futures\n    AD and death by year 10", fontsize=12)
    d.set_title("D  MMSE transition matrix\n    average cell gap, 5 years", fontsize=12)
    for ax in (b, c, d):
        ax.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, n + 0.9)
    note = ("Rows ranked by panel A. Hollow bar = not ranked in that panel: grey name = delphi-dedup-sim (rollout readout of "
            "delphi-dedup, not ranked separately).")
    if roles:
        note += ("\nItalic = reference models: " + "; ".join(f"{k} ({v})" for k, v in roles.items()) + ".")
    if inc0:
        ad, de = inc0["ad"], inc0["death"]
        def band(x):
            return f"{x['observed_lo'][-1] / x['observed'][-1]:.2f}–{x['observed_hi'][-1] / x['observed'][-1]:.2f}"
        note += (f"\nC: observed 10-year AD {ad['observed'][-1]:.0%} (95% CI {ad['observed_lo'][-1]:.0%}–{ad['observed_hi'][-1]:.0%}), "
                 f"death {de['observed'][-1]:.0%} ({de['observed_lo'][-1]:.0%}–{de['observed_hi'][-1]:.0%}); ratios inside "
                 f"{band(ad)} (AD) and {band(de)} (death) are within that uncertainty (grey band = the wider of the two).")
    if floor:
        note += (f"\nD: rows (first-visit MMSE stage) weighted equally; even a perfect simulator would score about "
                 f"{100 * floor:.0f} points from sampling noise alone (grey); see Table 5.")
    fig.text(0.01, 0.0, note, fontsize=9, color=SOFT, va="top")
    _save(fig, path)


# ----------------------------------------------------------------------------- fig8：时间

def fig8_timing(J, path):
    """一个面板：真的发生了的人，预测时间 − 实际时间（年）的箱线图，按实际隔了多久分 3 段，每段 AD / MMSE<24 / 死亡三个箱子。
    黑菱形 = 平均绝对误差；空心短横 = 所有人都猜同一个数（实际时间的中位数）的平均绝对误差 —— 菱形比短横低才说明模型分得出早晚。"""
    Q3 = [q for q in PV_EP if "by_bin" in J["questions"].get(q, {})]
    bins = J["bins"]
    W = 0.8 / len(Q3)
    fig, a = plt.subplots(figsize=(8.6, 5.0))
    tops, bots = [], []
    for gi, q in enumerate(Q3):
        r = J["questions"][q]
        for bi, b in enumerate(r["by_bin"]):
            x = bi + (gi - (len(Q3) - 1) / 2) * W
            if "box" not in b:
                a.text(x, 0, f"n = {b['n']}\ntoo few", ha="center", va="center", fontsize=8, color=SOFT)
                continue
            bx = dict(b["box"], fliers=[])
            bp = a.bxp([bx], positions=[x], widths=W * 0.8, showfliers=False, showmeans=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.3), whiskerprops=dict(color=SOFT), capprops=dict(color=SOFT))
            bp["boxes"][0].set(facecolor=EP_COL[q], alpha=0.7, edgecolor="white")
            a.plot([x], [b["mae"]], "D", ms=5.5, color=INK, zorder=5)
            a.plot([x - W * 0.3, x + W * 0.3], [b["constant_mae"]] * 2, "-", color="white", lw=2.4, zorder=6,
                   solid_capstyle="butt", path_effects=[patheffects.withStroke(linewidth=4.4, foreground=INK)])
            a.text(x, bx["whislo"], f"\n{b['n']}", ha="center", va="top", fontsize=8, color=SOFT)
            tops += [bx["whishi"], b["mae"], b["constant_mae"]]
            bots.append(bx["whislo"])
    a.axhline(0, ls=":", color=SOFT, lw=1.2, zorder=1)
    a.set_xticks(range(len(bins)), [f"{lo}–{hi} y" for lo, hi in bins])
    a.set_xlim(-0.55, len(bins) - 0.45)
    lo, hi = (min(bots), max(tops)) if tops else (-5, 5)
    a.set_ylim(lo - 0.12 * (hi - lo), hi + 0.06 * (hi - lo))
    a.set_xlabel("how long after the first visit the event actually happened")
    a.set_ylabel(f"predicted {MINUS} actual time (years)\nabove 0 = predicted too late")
    a.legend(handles=[Patch(facecolor=EP_COL[q], alpha=0.7, label=PV_EP_LABEL[q]) for q in Q3]
             + [Line2D([], [], marker="D", ls="", color=INK, ms=5.5, label="average error (model)"),
                Line2D([], [], color="white", lw=2.4, path_effects=[patheffects.withStroke(linewidth=4.4, foreground=INK)],
                       label="average error if everyone gets\nthe same guess (median time)")],
             loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=3, handletextpad=0.5, columnspacing=1.2)
    short = {"ad": "AD", "mmse_lt24": "MMSE<24", "death": "death"}
    head = "; ".join(f"{short[q]} {J['questions'][q]['mae']:.1f} vs {J['questions'][q]['constant_mae']:.1f}" for q in Q3)
    fig.suptitle(f"{J['name']}: when will it happen? Error in the predicted time, among people it happened to\n"
                 f"average error in years, model vs same guess for everyone: {head}", x=0.01, ha="left", fontsize=12, y=1.02)
    Y = J["years"]
    pred = (f"the first whole year in which the cumulative risk from the first visit "
            f"({'share of simulated futures' if J['source'] == 'rollout' else 'the model risk'}) reaches half its {Y}-year value\n"
            f"(no interpolation: the median time in whole years, given the event happens within {Y} years)")
    miss = sum(J["questions"][q]["n_no_prediction"] for q in Q3)
    foot = (f"Validation set, {J['n_persons']} people, starting at the first visit; only people who had the event within "
            f"{Y} years (number under each box).\nPredicted time = {pred}.\nBox = middle half of the errors, line = median, "
            "whiskers = 1.5× the box height.")
    if miss:
        foot += (f"\n{miss} {'person' if miss == 1 else 'people'} who had the event got no predicted time "
                 f"(the model gave it no chance within {Y} years); left out.")
    fig.text(0.01, -0.16, foot, fontsize=9, color=SOFT, va="top")
    _save(fig, path)


# ----------------------------------------------------------------------------- 入口

def build_package(d):
    """一个打包盒：按有的 JSON 画能画的图，返回画了哪些。"""
    out = []
    t = _j(os.path.join(d, "transitions.json"))
    J = _j(os.path.join(d, "perturb_v2.json"))
    note = ("Not ranked: in the snapshot tokenization the readout position is not supervised in training."
            if J and J.get("ranked") is False and J.get("tokenization") == "snapshot" else "")
    if t:
        fig3_transitions(t, os.path.join(d, "fig3_transitions.png"), note)
        out.append("fig3")
    tr = _j(os.path.join(d, "trajectory.json"))
    if tr and tr.get("applicable") and tr.get("matrix_5y"):
        fig4_transition_matrix(tr, os.path.join(d, "fig4_transition_matrix.png"))
        fig5_trajectories(tr, os.path.join(d, "fig5_trajectories.png"))
        out += ["fig4", "fig5"]
    if J and (J.get("direct") or {}).get("status") == "ok":
        bl = _j(os.path.join(d, "baseline.json")) or {}
        future = "rollout" if os.path.basename(d) == DELPHI_SIM else bl.get("future")
        fig6_perturbation(J, os.path.join(d, "fig6_perturbation.png"), future)
        out.append("fig6")
    Sj = _j(os.path.join(d, "shap.json"))
    if Sj and (Sj.get("questions") or {}).get("ad"):
        shap_summary(Sj, os.path.join(d, "fig7_shap.png"))
        out.append("fig7")
    Tj = _j(os.path.join(d, "timing.json"))
    if Tj and Tj.get("applicable"):
        fig8_timing(Tj, os.path.join(d, "fig8_timing.png"))
        out.append("fig8")
    return out


def main():
    _style()
    for d in zoo_pkgs(ZOO):
        if not os.path.exists(os.path.join(d, "transitions.json")):
            continue
        print(os.path.basename(d), build_package(d))
    S = _j(os.path.join(ZOO, REPORT_DIR, "SHAP_SUMMARY.json"))
    Jd = _j(os.path.join(ZOO, pkg_dir("delphi-dedup"), "shap.json"))
    if S and Jd:
        shap_summary(Jd, os.path.join(ZOO, REPORT_DIR, "SHAP_SUMMARY.png"), S=S,
                     headline="What delphi-dedup looks at to predict AD: MMSE and cognition by their value; lab and "
                              "blood-pressure records just by being there")
        print("SHAP_SUMMARY.png")


if __name__ == "__main__":
    main()
