"""make_fig.py -- results/results.json（nacc_scaling_ar.py collect 拉回来的聚合结果）-> 幂律拟合 + 图 + 表。

    python make_fig.py [results/results.json]
    -> results/fits.json、results/scaling.png、results/table.md

拟合：每个模型 test log-loss（10 个格子平均）= a + b·N^(−α)，N = 训练人数，所有抽样的点一起最小二乘。
a = 数据无限多时的极限，α = 下降有多快（数据翻 10 倍，"还能降的那部分"剩 10^(−α)）。
CI：每档训练量在 3 次抽样里有放回地重抽（全部人那一档的 ar 模型只有一个点），重拟合 1000 次，取 2.5% / 97.5% 分位数。
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["ar-lr", "ar-lgbm", "trf-s", "trf-m", "trf-l"]
LABEL = {"ar-lr": "ar-lr", "ar-lgbm": "ar-lgbm",
         "trf-s": "Transformer S", "trf-m": "Transformer M", "trf-l": "Transformer L"}
COL = {"ar-lr": "#222222", "ar-lgbm": "#8a3ffc",
       "trf-s": "#8fd6b4", "trf-m": "#2fae74", "trf-l": "#0b6b43"}
N_BOOT = 1000


def law(n, a, b, al):
    return a + b * np.power(n / 1000.0, -al)


def fit(n, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = curve_fit(law, n, y, p0=(y.min() * 0.95, y.max() - y.min() + 1e-3, 0.5),
                         bounds=([0, 0, 0.01], [np.inf, np.inf, 3]), maxfev=20000)
    return p


def fit_model(rows, key="ll_all", rng=None):
    n = np.array([r["n_train"] for r in rows], float)
    y = np.array([r[key] for r in rows])
    p = fit(n, y)
    sizes = sorted(set(n))
    by = {s: y[n == s] for s in sizes}
    B = []
    for _ in range(N_BOOT):
        nn = np.concatenate([[s] * len(by[s]) for s in sizes])
        yy = np.concatenate([rng.choice(by[s], len(by[s])) for s in sizes])
        try:
            B.append(fit(nn, yy))
        except RuntimeError:
            pass
    B = np.array(B)
    ci = lambda j: [float(np.quantile(B[:, j], 0.025)), float(np.quantile(B[:, j], 0.975))]  # noqa: E731
    nmax = max(sizes)
    return {"a": float(p[0]), "b": float(p[1]), "alpha": float(p[2]), "a_ci": ci(0), "alpha_ci": ci(2),
            "pred_at_max": float(law(nmax, *p)), "pred_at_10x_max": float(law(10 * nmax, *p)),
            "pred_10x_ci": [float(np.quantile([law(10 * nmax, *q) for q in B], 0.025)),
                            float(np.quantile([law(10 * nmax, *q) for q in B], 0.975))],
            "n_points": int(len(y)), "n_boot_ok": int(len(B))}


def main(path):
    R = json.load(open(path))
    out_dir = os.path.dirname(path)
    runs = R["runs"]
    models = [m for m in ORDER if any(r["model"] == m for r in runs)]
    rng = np.random.default_rng(0)
    F = {m: fit_model([r for r in runs if r["model"] == m], rng=rng) for m in models}
    json.dump(F, open(os.path.join(out_dir, "fits.json"), "w"), indent=1)

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 图只画数（线 = 3 次抽样的平均，点 = 每次抽样）；幂律拟合只进表：这里有几条曲线（Transformer M、L）拟合出的极限没有约束
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    ns = sorted({r["n_train"] for r in runs})
    panels = (("ll_all", "NACC test log-loss（10 个格子平均，越低越好）", "损失"),
              ("auc_p1", "AUC：4 个终点 2 年的平均", "2 年 AUC"),
              ("auc_p3", "AUC：6 道题 5 年的平均", "5 年 AUC"))
    for ax, (key, ylab, title) in zip(axes, panels):
        for m in models:
            rs = [r for r in runs if r["model"] == m]
            n = np.array([r["n_train"] for r in rs])
            y = np.array([r[key] for r in rs])
            ax.scatter(n, y, s=12, color=COL[m], alpha=0.45, zorder=3)
            ax.plot(ns, [y[n == s].mean() for s in ns], "-o", color=COL[m], lw=2.2 if m in ("ar-lgbm", "trf-s") else 1.6,
                    ms=4, label=LABEL[m], zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(ns, [f"{v / 1000:.1f}k".replace(".0k", "k") for v in ns])
        ax.minorticks_off()
        ax.set_xlabel("NACC 训练人数")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(top=0.42)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(models), fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out_dir, "scaling.png"), dpi=150)
    print("->", os.path.join(out_dir, "scaling.png"))

    L = ["| 模型 | 参数个数 | " + " | ".join(f"{v:,}" for v in ns) + " | 极限 a [95% CI] | α [95% CI] | 外推到 10 倍人数 [95% CI] |",
         "|---|---|" + "---|" * len(ns) + "---|---|---|"]
    for m in models:
        rs = [r for r in runs if r["model"] == m]
        npar = next((r["info"]["n_params"] for r in rs if "info" in r), None)
        cells = []
        for s in ns:
            v = [r["ll_all"] for r in rs if r["n_train"] == s]
            cells.append(f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else "") if v else "—")
        f = F[m]
        L.append(f"| {LABEL[m]} | {f'{npar:,}' if npar else '—'} | " + " | ".join(cells)
                 + f" | {f['a']:.4f} [{f['a_ci'][0]:.4f}, {f['a_ci'][1]:.4f}] | {f['alpha']:.2f} [{f['alpha_ci'][0]:.2f}, {f['alpha_ci'][1]:.2f}]"
                 + f" | {f['pred_at_10x_max']:.4f} [{f['pred_10x_ci'][0]:.4f}, {f['pred_10x_ci'][1]:.4f}] |")
    L += ["", "| 模型 | " + " | ".join(f"{v:,}" for v in ns) + " |", "|---|" + "---|" * len(ns)]
    for m in models:
        rs = [r for r in runs if r["model"] == m]
        L.append(f"| {LABEL[m]} | " + " | ".join(
            (f"{np.mean([r['auc_p1'] for r in rs if r['n_train'] == s]):.3f}" if any(r["n_train"] == s for r in rs) else "—") for s in ns) + " |")
    open(os.path.join(out_dir, "table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "results.json"))
