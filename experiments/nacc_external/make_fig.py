"""make_fig.py -- 本地：从揭盲结果（聚合 JSON）画 3 张图（PROTOCOL.md 第 7 节第 6 步）。

    python make_fig.py                         # results/E_results.json -> fig1_auc.png, fig2_centers.png, fig3_calibration.png
    python make_fig.py results/DRYRUN_H.json   # 演练（H），输出加前缀 dryH_
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROS, NACC, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
ORDER = ["xgb-l2c", "xgb-l2c-sim", "minimalrnn", "lr-cur", "transformer", "fnn-l2c-sim", "delphi-dedup", "ethos", "ar-lr",
         "cox-cur-sim", "delphi-dedup-sim", "lr-demo", "bdsi-sim"]           # ROSMAP val 上 2 年 AD AUC 从高到低


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "E_results.json")
    pre = "dryH_" if "DRYRUN" in src else ""
    tgt = "NACC H (dry run)" if pre else "NACC E"
    R = json.load(open(src))
    names = [n for n in ORDER if n in R["models"]]
    y = np.arange(len(names))[::-1]

    # ---- 图 1：2 年、5 年 AD 的 AUC，ROSMAP val 对 NACC
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(names) + 1.6), sharey=True, facecolor=SURF)
    for ax, key, title in zip(axes, ("ad_2y", "ad_5y"), ("AD dementia within 2 years", "AD dementia within 5 years")):
        style(ax)
        for yi, n in zip(y, names):
            e = R["models"][n][key]
            r = (R["decomposition"][n].get("rosmap_val") or {}).get(key)
            if r is not None:
                ax.plot([r, e["auc"]], [yi, yi], color=GRID, lw=2, zorder=1)
                ax.scatter(r, yi, s=46, color=ROS, edgecolor=SURF, lw=1.5, zorder=3)
            ax.plot(e.get("auc_ci", [e["auc"]] * 2), [yi, yi], color=NACC, lw=2, zorder=2)
            ax.scatter(e["auc"], yi, s=46, color=NACC, edgecolor=SURF, lw=1.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("AUC", color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[1].scatter([], [], s=46, color=ROS, label="ROSMAP val")
    axes[1].scatter([], [], s=46, color=NACC, label=f"{tgt} (95% CI)")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, fontsize=9)   # 图例放顶上，不压数据
    fig.suptitle("Trained on ROSMAP only: AUC on ROSMAP val vs NACC", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(HERE, f"{pre}fig1_auc.png"), dpi=150, facecolor=SURF)

    # ---- 图 2：按中心的随机效应汇总（2 年 AD）：点 = 汇总值，粗线 = 95% CI，细线 = 95% 预测区间
    C = R["centers"]["ad_2y"]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(names) + 1.4), facecolor=SURF)
    style(ax)
    for yi, n in zip(y, names):
        c = C[n]
        ax.plot(c["pi"], [yi, yi], color=MUTED, lw=1, zorder=1)
        ax.plot(c["ci"], [yi, yi], color=NACC, lw=3, solid_capstyle="round", zorder=2)
        ax.scatter(c["pooled"], yi, s=40, color=NACC, edgecolor=SURF, lw=1.5, zorder=3)
        ax.text(1.0, yi, f"I² {c['I2']:.0%}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"AUC, AD within 2 years (pooled over {C[names[0]]['k']} centres)", color=INK)
    ax.set_title("Pooled over centres: thick = 95% CI, thin = 95% range for a new centre",
                 loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"{pre}fig2_centers.png"), dpi=150, facecolor=SURF)

    # ---- 图 3：O/E（2 年 AD），原样 vs 用 H 重新校准后
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(names) + 1.4), facecolor=SURF)
    style(ax)
    for yi, n in zip(y, names):
        a = R["models"][n]["ad_2y"]["o_e"]
        b = R["recalibrated"][n]["ad_2y"]["o_e"]
        ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(a, yi, s=46, color=NACC, edgecolor=SURF, lw=1.5, zorder=3)
        ax.scatter(b, yi, s=46, color=INK, edgecolor=SURF, lw=1.5, zorder=4)   # 不用 ROSMAP 的蓝：这里两点都是 NACC E
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_xscale("log")
    tk = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(tk)
    ax.set_xticklabels([f"{t:g}" for t in tk])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Observed / expected, AD within 2 years (log scale; > 1 = under-predicts)", color=INK)
    ax.scatter([], [], color=NACC, label="as trained on ROSMAP")
    ax.scatter([], [], color=INK, label="recalibrated on NACC H (intercept + slope)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"{pre}fig3_calibration.png"), dpi=150, facecolor=SURF)
    print("wrote", [f"{pre}fig{i}" for i in (1, 2, 3)])


if __name__ == "__main__":
    main()
