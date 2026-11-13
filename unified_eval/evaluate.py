"""evaluate.py -- 统一打分 + 统一出图。**所有模型都用这一份代码**，所以图可以并排比。

    python evaluate.py --name delphi-nodedup \
        --scores s42=$RUNS/scores/delphi-nodedup__s42.csv s43=... s44=... \
        --outdir <package dir>

输入只有两张表：truth（truth.py）和每个 seed 一张风险表（列 pid, t, risk_<终点>_<H>y）。
第一个 --scores 是主 seed：图上的点和 CI 来自它；其余 seed 画成小空心点，表里给均值 ± sd。

输出：metrics.json、fig1_auc_2y.png、fig2_calibration_2y.png。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import ENDPOINTS, HORIZONS, MAIN_H, RUN_DIR  # noqa: E402

N_BOOT = 1000
N_BINS = 10
CAL_LIM = {"ad": 0.5, "death": 0.8, "mmse": 0.6, "cogn": 0.6}   # 校准图坐标上限，所有模型固定
COLOR = {"ad": "#7b3294", "death": "#404040", "mmse": "#0571b0", "cogn": "#1a9641", "macro": "#000000"}


def boot_persons(pid, n_boot=N_BOOT, seed=20261007):
    """按人重抽样的下标。种子固定、只依赖人的集合 -> 所有模型、所有终点抽到同样的人（配对）。"""
    persons, inv = np.unique(pid, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(persons))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[k] for k in rng.integers(0, len(persons), len(persons))])
            for _ in range(n_boot)]


def calib(y, p):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() == 0:
            continue
        n, o = int(m.sum()), float(y[m].mean())
        se = np.sqrt(max(o * (1 - o), 1e-9) / n)
        rows.append({"n": n, "pred": float(p[m].mean()), "obs": o, "lo": max(0.0, o - 1.96 * se),
                     "hi": min(1.0, o + 1.96 * se)})
    ece = sum(r["n"] * abs(r["obs"] - r["pred"]) for r in rows) / len(y)
    return rows, float(ece)


def endpoint_metrics(truth, sc, boot_idx=None):
    out = {}
    for e in ENDPOINTS:
        for H in HORIZONS:
            y = truth[f"y_{e}_{H}y"].to_numpy()
            p = sc[f"risk_{e}_{H}y"].to_numpy()
            # AUC 只看排序。可选列 rank_<终点> 是和风险单调的打分（Delphi 交的是 logit）：
            # 风险 1-exp(-exp(logit)*H) 在 logit 很大时会在浮点上饱和成 1.0，排序信息就丢了。
            rcol = next((c for c in (f"rank_{e}", f"logit_{e}") if c in sc.columns), None)
            rk = sc[rcol].to_numpy() if rcol else p
            m = y >= 0
            yy, pp, rr = y[m], p[m], rk[m]
            r = {"n": int(m.sum()), "n_pos": int(yy.sum()), "auc": float(roc_auc_score(yy, rr)),
                 "brier": float(np.mean((pp - yy) ** 2)), "o_e": float(yy.sum() / max(pp.sum(), 1e-12)),
                 "mean_pred": float(pp.mean()), "obs_rate": float(yy.mean())}
            if H == MAIN_H:
                r["calibration"], r["ece"] = calib(yy, pp)
                if boot_idx is not None:
                    # 站点在 truth 里的行号 -> 只保留可计入的
                    pos = np.full(len(y), -1)
                    pos[m] = np.arange(m.sum())
                    bs = []
                    for ii in boot_idx:
                        jj = pos[ii]
                        jj = jj[jj >= 0]
                        yb = yy[jj]
                        if 0 < yb.sum() < len(yb):
                            bs.append(roc_auc_score(yb, rr[jj]))
                    r["auc_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    r["auc_boot"] = bs
            out[f"{e}_{H}y"] = r
    return out


def macro(ep, H):
    return float(np.mean([ep[f"{e}_{H}y"]["auc"] for e in ENDPOINTS]))


def evaluate(name, truth, scores, boot_idx):
    per_seed = {}
    for tag, path in scores:
        sc = pd.read_csv(path)
        sc = truth[["pid", "t"]].merge(sc, on=["pid", "t"], how="left", validate="one_to_one")
        miss = sc[[f"risk_{e}_{MAIN_H}y" for e in ENDPOINTS]].isna().any(axis=1)
        assert not miss.any(), f"{path}: {int(miss.sum())} 个站点没有打分"
        per_seed[tag] = endpoint_metrics(truth, sc, boot_idx if tag == scores[0][0] else None)
    main = scores[0][0]
    ep = per_seed[main]
    # 综合 AUC 的 CI：同一批 bootstrap 抽样上 4 个终点取平均
    boots = np.array([ep[f"{e}_{MAIN_H}y"]["auc_boot"] for e in ENDPOINTS if len(ep[f"{e}_{MAIN_H}y"]["auc_boot"]) == N_BOOT])
    macro_ci = [float(np.percentile(boots.mean(0), 2.5)), float(np.percentile(boots.mean(0), 97.5))] \
        if len(boots) == len(ENDPOINTS) else None
    for e in ENDPOINTS:
        ep[f"{e}_{MAIN_H}y"].pop("auc_boot", None)
    summary = {"name": name, "main_seed": main, "seeds": [t for t, _ in scores],
               "n_persons": int(truth.pid.nunique()), "n_landmarks": int(len(truth)),
               "macro_auc": {f"{H}y": macro(ep, H) for H in HORIZONS},
               "macro_auc_ci_2y": macro_ci,
               "endpoints": ep}
    single = {t: v for t, v in per_seed.items() if t != "ens"}      # 集成不是一个 seed
    if len(single) > 1:
        summary["across_seeds"] = {
            k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "values": v}
            for k, v in (
                [(f"{e}_{H}y_auc", [single[t][f"{e}_{H}y"]["auc"] for t in single])
                 for e in ENDPOINTS for H in HORIZONS]
                + [(f"macro_{H}y_auc", [macro(single[t], H) for t in single]) for H in HORIZONS])}
    if len(per_seed) > 1:
        summary["per_seed_auc_2y"] = {t: {e: per_seed[t][f"{e}_{MAIN_H}y"]["auc"] for e in ENDPOINTS}
                                      for t in per_seed}
    return summary


# ----------------------------------------------------------------------------- 图

def fig_auc(summary, path):
    """每个终点一根柱子（从 0.5 = 瞎猜往上长），柱顶写数值、带 95% CI；其余 seed 画成柱子里的短横线。
    2026-10-16 起从点图改成柱状图（意思不变）；2026-10-29 起不画 Delphi 参照线（各模型平等，不拿谁当基准）。"""
    labels = [ENDPOINTS[e][0] for e in ENDPOINTS] + ["Mean of 4"]
    keys = list(ENDPOINTS) + ["macro"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ep = summary["endpoints"]
    W = 0.56
    for i, k in enumerate(keys):
        if k == "macro":
            v, ci = summary["macro_auc"][f"{MAIN_H}y"], summary["macro_auc_ci_2y"]
        else:
            r = ep[f"{k}_{MAIN_H}y"]
            v, ci = r["auc"], r.get("auc_ci")
        ax.bar(i, v - 0.5, bottom=0.5, width=W, color=COLOR[k], alpha=0.85, zorder=2)
        top = v
        if ci:
            ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#222222", capsize=4, lw=1.4, zorder=4)
            top = ci[1]
        ax.text(i, top + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, zorder=5)
        # 其余 seed：柱子里的白色短横线
        if "per_seed_auc_2y" in summary:
            for t, d in summary["per_seed_auc_2y"].items():
                if t == summary["main_seed"]:
                    continue
                sv = np.mean(list(d.values())) if k == "macro" else d[k]
                ax.hlines(sv, i - W / 2 + 0.06, i - 0.06, colors="white", lw=1.6, zorder=3)
    ax.axhline(0.5, color="#888888", lw=0.8, zorder=1)
    ax.set_xticks(range(len(keys)), labels, fontsize=9)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel(f"AUC, {MAIN_H}-year risk (bars start at 0.5 = chance)", fontsize=9)
    has_seeds = len(summary.get("per_seed_auc_2y", {})) > 1
    ax.set_title(f"{summary['name']}\n(val: {summary['n_persons']} persons, {summary['n_landmarks']} landmarks;"
                 f" 95% CI by person bootstrap" + ("; white ticks = other seeds)" if has_seeds else ")"),
                 fontsize=9.5)
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_cal(summary, path, note=None):
    ep = summary["endpoints"]
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(11, 3.1))
    for ax, e in zip(axes, ENDPOINTS):
        r = ep[f"{e}_{MAIN_H}y"]
        lim = CAL_LIM[e]
        ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=1)
        for c in r["calibration"]:
            px, py = min(c["pred"], lim), min(c["obs"], lim)
            off = c["pred"] > lim
            ax.errorbar(px, py, yerr=[[py - min(c["lo"], lim)], [min(c["hi"], lim) - py]],
                        fmt=">" if off else "o", color=COLOR[e], ms=4.5, capsize=2, lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(ENDPOINTS[e][0], fontsize=9.5)
        ax.set_xlabel("predicted 2-y risk", fontsize=8.5)
        if e == list(ENDPOINTS)[0]:
            ax.set_ylabel("observed fraction", fontsize=8.5)
        ax.text(0.04 * lim, 0.93 * lim, f"O/E {r['o_e']:.2f}\nECE {r['ece']:.3f}", fontsize=8, va="top")
        ax.tick_params(labelsize=7.5)
    t = f"{summary['name']} -- calibration of {MAIN_H}-year risk (deciles of predicted risk; '>' = off the chart)"
    if note:
        t += f"\n{note}"
    fig.suptitle(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="tag=path，第一个是主 seed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cal_note", default=None)
    args = ap.parse_args()
    truth = pd.read_csv(os.path.join(RUN_DIR, f"truth_{args.split}.csv"))
    scores = [tuple(s.split("=", 1)) for s in args.scores]
    boot_idx = boot_persons(truth.pid.to_numpy())
    summary = evaluate(args.name, truth, scores, boot_idx)
    if args.cal_note:
        summary["calibration_note"] = args.cal_note
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    fig_auc(summary, os.path.join(args.outdir, "fig1_auc_2y.png"))
    fig_cal(summary, os.path.join(args.outdir, "fig2_calibration_2y.png"), args.cal_note)
    ep = summary["endpoints"]
    print(f"{args.name}: macro AUC 2y {summary['macro_auc']['2y']:.3f}  " +
          "  ".join(f"{e} {ep[f'{e}_2y']['auc']:.3f}" for e in ENDPOINTS))


if __name__ == "__main__":
    main()
