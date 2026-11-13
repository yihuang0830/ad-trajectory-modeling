"""make_fig.py -- results/results.json（nacc_scale.py collect 拉回来的聚合结果）-> results/nacc_scale.png 和 results/table.md。

    python make_fig.py [results/results.json]

左：NACC test 上 4 终点 2 年平均 AUC 随训练人数（ROSMAP 大小 -> 12,000 -> 全部）怎么变；灰线是其余模型。
右：每个模型减 lr-cur（同一档训练量，按人配对 bootstrap 的 95% CI），ROSMAP val 和 NACC 两头。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = ("rosmap", "12k", "full")
ORDER = ["lr-cur", "lr-l2c", "dt-l2c", "rf-l2c", "xgb-l2c", "lgbm-l2c",
         "transformer", "grud", "tlstm", "delphi-upstream", "delphi-zoo"]
LABEL = {"lr-cur": "LR（只看当前）", "lr-l2c": "LR（当前 + 历史摘要）", "dt-l2c": "决策树", "rf-l2c": "随机森林",
         "xgb-l2c": "XGBoost", "lgbm-l2c": "LightGBM",
         "transformer": "Transformer", "grud": "GRU-D", "tlstm": "T-LSTM", "delphi-upstream": "Delphi（上游原版）",
         "delphi-zoo": "Delphi（zoo 版）"}
HI = {"lr-cur": "#222222", "xgb-l2c": "#8a3ffc", "dt-l2c": "#2a78d6", "transformer": "#1baf7a", "delphi-upstream": "#eb6834"}
GREY = "#c4c4c4"


def main(path):
    R = json.load(open(path))
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    n = {s: R["nacc"][s]["n_train_persons"] for s in SIZES if s in R["nacc"]}
    xs = [n[s] for s in SIZES if s in n]
    models = [m for m in ORDER if any(m in R["nacc"].get(s, {}).get("models", {}) for s in SIZES)]
    for m in sorted(models, key=lambda m: m in HI):
        pts = [(n[s], R["nacc"][s]["models"][m]["p1"]) for s in SIZES if m in R["nacc"].get(s, {}).get("models", {})]
        x, y = zip(*pts)
        c = HI.get(m, GREY)
        a.plot(x, y, "-o", color=c, lw=2.2 if m in HI else 1, ms=7 if m in HI else 4, zorder=3 if m in HI else 2)
        if m in HI:
            a.annotate(LABEL[m], (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=c, fontsize=10)
    a.set_xscale("log")
    a.set_xticks(xs, [f"{v:,}" for v in xs])
    a.minorticks_off()
    a.set_xlim(xs[0] * 0.8, xs[-1] * 2.6)
    a.set_xlabel("NACC 训练人数（最左 = 和 ROSMAP 训练集一样大）")
    a.set_ylabel("NACC test：4 个终点 2 年 AUC 的平均")
    a.set_title("训练量变大，谁涨得多？（灰线：其余模型）", fontsize=11)
    a.grid(alpha=0.3)

    # 右：减 lr-cur。每个模型三行：ROSMAP val、NACC 同样大小、NACC 全部
    rows = [("ROSMAP val", R["rosmap"]["models"], "#9b9b9b", "s"),
            (f"NACC {n.get('rosmap', 0):,} 人", R["nacc"].get("rosmap", {}).get("models", {}), "#7fb2e8", "o"),
            (f"NACC {n.get('full', 0):,} 人", R["nacc"].get("full", {}).get("models", {}), "#1f4e9c", "D")]
    ms = [m for m in ORDER if m != "lr-cur" and any(m in r[1] for r in rows)]
    for i, m in enumerate(ms):
        for j, (lab, M, c, mk) in enumerate(rows):
            if m not in M or "p1_vs_lr" not in M[m]:
                continue
            d = M[m]["p1_vs_lr"]
            y = i + (j - 1) * 0.25
            b.plot(d["ci"], [y, y], color=c, lw=1.6)
            b.plot(d["diff"], y, mk, color=c, ms=6, label=lab if i == 0 else None)
    b.axvline(0, color="#222222", lw=1)
    b.set_yticks(range(len(ms)), [LABEL[m] for m in ms])
    b.invert_yaxis()
    b.set_xlabel("减 LR（只看当前）的 AUC 差（按人配对 bootstrap 95% CI）", labelpad=2)
    b.set_title("和 LR 比：> 0 才算赢过 LR", fontsize=11)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=9, frameon=False)
    b.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(path), "nacc_scale.png")
    fig.savefig(out, dpi=150)
    print("->", out)

    # 表：每档每个模型 p1 / p3 和减 LR
    L = ["| 模型 | ROSMAP val p1 | NACC ROSMAP 大小 p1 | NACC 12k p1 | NACC 全部 p1 | NACC 全部 p1 − LR [95% CI] | NACC 全部 p3 |",
         "|---|---|---|---|---|---|---|"]
    f = lambda M, m, k="p1": f"{M[m][k]:.3f}" if m in M else "—"  # noqa: E731
    for m in ORDER:
        F = R["nacc"].get("full", {}).get("models", {})
        d = F.get(m, {}).get("p1_vs_lr")
        L.append(f"| {LABEL[m]} | {f(R['rosmap']['models'], m)} | " + " | ".join(f(R["nacc"].get(s, {}).get("models", {}), m) for s in SIZES)
                 + f" | {'参照' if m == 'lr-cur' else (format(d['diff'], '+.3f') + f' [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]' if d else '—')} | {f(F, m, 'p3')} |")
    open(os.path.join(os.path.dirname(path), "table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "results.json"))
