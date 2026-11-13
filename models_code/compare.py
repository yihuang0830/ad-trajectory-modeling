"""compare.py -- 全部模型（含 Delphi）在同一套评估上的横向对比，只读聚合 JSON，在本地跑。

    bash ris/fetch.sh && python compare.py
    -> <打包盒>/README.md + fig3 / fig4 / fig5（models_self/、representative/，见 registry.pkg_dir）（figures.build_package，和 Delphi 的打包盒同一份画图代码）
    -> reports/MODELS.md、MODELS.png（预测）、MODELS_traj.png（10 年轨迹）、OVERVIEW.png（7 个维度）、MODELS_calibration.png（校准）、MODELS_timing.png（时间）：图只画 SELECTED 的 9 个；
       OVERVIEW_all.png 是全部 52 个
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)
BZ = os.path.join(ZOO, "models_self")                 # NACC 预训练、轨迹比较的聚合 JSON（_nacc、_traj）；打包盒的位置见 registry.pkg_dir
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ZOO, "unified_eval"))
import models as MZ  # noqa: E402
import figures as FG  # noqa: E402
from registry import (DELPHI_SIM, NOT_USABLE, NOT_USABLE_DIR, REP_DIR, REPORT_DIR, REPRESENTATIVE,  # noqa: E402
                      folder_md, pkg_dir, rel_link, report_md, zoo_files)
from report_sections import matrix_gap, matrix_gap_floor, pv2_logor_scores, sec_timing, traj_summary  # noqa: E402
from report_text import CONCLUSIONS  # noqa: E402  （2026-11-05 以前的结论，现在只放进 models_not_usable/README.md）

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator, PercentFormatter  # noqa: E402

ENDS = ("ad", "death", "mmse", "cogn")
END_ZH = {"ad": "AD", "death": "死亡", "mmse": "MMSE<24", "cogn": "认知<−1"}
# 表里 Delphi 的几行（第一部分 / 第三部分都从各自的打包盒读）
REFS = [("delphi-dedup", "Delphi 最好的分词（dedup）"),
        ("delphi-dedup-nojitter", "Delphi dedup，关掉抖动"), ("delphi-nodedup", "Delphi nodedup"),
        ("delphi-nodedup-staticonly-split811", "Delphi 只看人口学")]     # 本地留的 4 个 Delphi（models_self/；其余在 models_self/archive/）
FUT_ZH = {"rollout": "模拟（rollout）", "exact": "精确（转移矩阵）", "survival": "生存曲线", "direct": "逐题逐年直接给",
          "delphi": "读速率（也能模拟）"}
FUT_EXPLAIN = {
    "rollout": "**能生成未来。** 从站点起逐年往后模拟（每个站点多条轨迹，条数见 MODELS.md 的'rollout 版的模拟条数'），H 年风险 = 轨迹里 H 年内首次发生的比例；"
               "从第一次访视起的 10 年轨迹就是 fig4、fig5。",
    "exact": "**能生成未来。** 多状态 Markov 链：每年的状态分布由一年转移矩阵连乘精确算出，站点上的风险不用抽样"
             "（目标状态设为吸收态）；10 年轨迹是从同一个转移矩阵抽出来的（fig4、fig5）。",
    "survival": "**不能生成轨迹。** 每个站点给一条生存曲线 S(t)，任意 H 年的风险 = 1 − S(H)，对 H 单调。"
                "它回答'H 年内会不会首次越线'，回答不了'5 年后处在哪个分期'（会不会先变差再好转）、"
                "也没法在一条轨迹中途改一个变量看后果——这两件事要靠模拟。",
    "direct": "**不能生成轨迹。** 每道题 × 每个时间窗（1–10 年）单独给一个概率。和生存模型一样只回答'H 年内会不会首次越线'，"
              "而且各时间窗之间不保证一致（可能出现 5 年风险 < 2 年风险）；回答不了'几年后在哪个分期'，也不能中途扰动。",
}
GROUP_COL = {"gen": "#2a78d6", "direct": "#eb6834", "sim": "#eb6834", "zoo": "#1baf7a"}  # 数据可视化色板的前三个（全对比验证通过）
GROUP_MARK = {"gen": "o", "direct": "s", "sim": "s", "zoo": "D"}
# sim = 直接预测基线的 rollout 版（-sim、ar-*）出现在只比 rollout 的图里时的组；和 direct 同色（同一个模型的两种读法），不会同图出现
GROUP_LABEL = {"gen": "Can simulate the future", "direct": "Direct prediction only",
               "sim": "Rollout version of a direct-prediction model", "zoo": "Delphi"}
# 最后选用的模型（2026-11-08 用户定：所有模型互相比较，选原版 transformer；改进版 transformer-gru-fill 只作补充）；表里加粗、图里名字加粗
CHOSEN = ("transformer",)


# 总图（MODELS.png、MODELS_traj.png、MODELS_sim.png、OVERVIEW.png）只画这 9 个代表（用户 2026-10-28 挑的）：
# 临床评分、Cox、逻辑回归 / 神经网络 / 树（同一份 l2c 输入）、序列 transformer、三个生成式（minimalrnn、ethos、Delphi）。
# 全部可用模型都在 MODELS.md 的表里和 OVERVIEW_all.png 上。
SELECTED = ["bdsi", "cox-cur", "lr-l2c", "fnn-l2c", "xgb-l2c", "transformer", "minimalrnn", "ethos", "delphi-dedup"]
# 这 9 个在图里画的读法（rollout 版）的打包盒放在 representative/（registry.REPRESENTATIVE），delphi-dedup-sim 在 delphi/
assert set(REPRESENTATIVE) == {MZ.SIM_OF.get(n, n) for n in SELECTED if n != "delphi-dedup"}
SIMS = {v for v in MZ.SIM_OF.values() if v.endswith("-sim")}        # 2026-10-21 加的 rollout 版（ar-lr、ar-lgbm 原本就在主表里）
NACC = MZ.NACC_NAMES                                                 # 2026-10-22 的 NACC 预训练实验：单独一节（表 5），不进表 1–4
SEQ_AR = {b[0] for b in MZ.BASELINES if b[2] in ("seqar", "ddhar")}  # 2026-10-28 起自回归的序列网络
TRFX = {b[0] for b in MZ.BASELINES if b[2] == "trfx"}               # 2026-11-05 TRFX 实验的组合（models_code/trfx/）
TRFX_TEXT = (
    "transformer 的改进版（`models_code/trfx/`，2026-11-05 的 TRFX 实验）。和 [transformer]({tr}) 一样是自回归的："
    "只学下一步（这一年里死亡、得 AD 的概率 + 明年那次访视的分布），所有答案都从一年一年模拟出来的未来里数。和原版的区别：\n\n"
    "- 明年的 MMSE 是 0–30 分上的 31 类分布（交叉熵），不是截断到 0–30 的高斯：天花板附近不再往下漂；\n"
    "- 不要\"第几次访视\"的编号嵌入，时间只从年龄、随访年数、间隔这几个输入来（晚期站点、10 年模拟不再用到没训练过的编号）；\n"
    "- 两层因果 transformer 后面接一层 GRU（残差），宽度 32，最多训练 300 轮；\n"
    "- 缺失的连续量用模型上一步自己的预测值填（MinimalRNN 的做法），训练和模拟都这样。\n\n"
    "怎么来的：在 test 段上比较了约 70 个变体（每个 3–6 个 seed，按 seed 和原版配对），从中选出这个组合；"
    "事先声明好之后只在 val 上看了一次（5 个 seed），按 seed 和原版配置配对比较第一部分 2 年 AUC、"
    "第三部分和 5 年 MMSE 转移矩阵。"
    "因为是从很多变体里挑出来的，这里的 val 数要按\"挑过一次\"来读。"
    "这里的打包盒是 seed 0，CPU、8 线程训练，和实验在 val 上评估过的模型逐位相同。")
ORIG_OF = {v: k for k, v in MZ.SIM_OF.items()}
# 不可用的（registry.NOT_USABLE，2026-11-05）= 不能生成未来的全部；不进表 1–3 和总图，表 4、表 5 挪到 models_not_usable/README.md
assert set(NOT_USABLE) == {b[0] for b in MZ.BASELINES if b[5] in ("direct", "survival")}
# 每个不可用模型留在评估里的那个能生成未来的版本：rollout 版（-sim、ar-*）；-x5 是同一结构的自回归版；transformer-nacc 是 transformer-sim-nacc
TWIN = {**{o: s for o, s in MZ.SIM_OF.items() if o in NOT_USABLE}, **{f"{k}-x5": k for k in ("transformer", "grud", "tlstm")},
        "transformer-nacc": "transformer-sim-nacc"}


def _j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def part1(m):
    if not m:
        return None
    ep = m["endpoints"]
    oe = [ep[f"{e}_2y"].get("o_e") for e in ENDS]
    return {"auc": m["macro_auc"]["2y"], "ci": m["macro_auc_ci_2y"], "ep": {e: ep[f"{e}_2y"]["auc"] for e in ENDS},
            "oe": float(np.mean([x for x in oe if x is not None])) if all(x is not None for x in oe) else None}


def part3(t):
    if not t:
        return None
    mac = t["macro"]
    g = lambda H, k: (mac.get(f"{H}y") or {}).get(k)          # noqa: E731
    return {"auc": {H: g(H, "auc") for H in (2, 5, 10)}, "ci5": g(5, "auc_ci"),
            "within_age5": g(5, "auc_within_state_age"), "cr10": g(10, "auc_cr"), "all": t.get("answers_all")}


def rows():
    out = []
    for name, fam, impl, kw, cite, fut in MZ.BASELINES:
        d = os.path.join(ZOO, pkg_dir(name))
        p1, p3 = part1(_j(os.path.join(d, "metrics.json"))), part3(_j(os.path.join(d, "transitions.json")))
        if p1 is None:
            continue
        tr = _j(os.path.join(d, "trajectory.json"))
        out.append({"name": name, "link": f"{pkg_dir(name)}/README.md", "label": name, "family": fam, "args": kw,
                    "cite": cite, "future": fut, "group": "gen" if fut in ("rollout", "exact") else "direct",
                    "p1": p1, "p3": p3, "traj": tr if tr and tr.get("applicable") else None,
                    "bl": _j(os.path.join(d, "baseline.json")), "pert": pert_summary(d, d)})
    for name, zh in REFS:
        d = os.path.join(ZOO, pkg_dir(name))
        p1 = part1(_j(os.path.join(d, "metrics.json")))
        t = _j(os.path.join(d, "transitions.json"))
        tr = _j(os.path.join(d, "trajectory.json"))
        out.append({"name": name, "link": f"{pkg_dir(name)}/README.md", "label": zh, "family": "zoo", "cite": zh, "args": {},
                    "future": "delphi", "group": "zoo",
                    "p1": p1, "p3": part3(t), "traj": tr if tr and tr.get("applicable") else None, "bl": None,
                    "pert": pert_summary(d, d)})
    return out


def pert_summary(d1, d2):
    """第二部分 perturb.json（d1）+ 第三部分 perturb_v2.json（d2）-> 表 3 的几个数；没有就是 None。"""
    p, J = _j(os.path.join(d1, "perturb.json")), _j(os.path.join(d2, "perturb_v2.json"))
    out = {}
    if p and p.get("readout_defined", True) and p.get("direction"):
        pr = p["direction"]["prespec"]
        ap = ((p.get("effects") or {}).get("apoe") or {}).get("ad") or {}
        out["v1"] = {"right": pr["right"], "no": pr["no_response"], "wrong": pr["wrong"], "cells": pr["cells"],
                     "dose": (p.get("dose") or {}).get("monotone_frac"), "apoe_ad": ap.get("dlogit")}
    if J and (J.get("direct") or {}).get("status") == "ok":
        dct = J["direct"]
        lo, (n_sync, n_def) = pv2_logor_scores(dct)
        eff = dct["effects"]
        g = lambda a, q: (eff.get(a, {}).get(q) or {}).get("d_rmst") if (eff.get(a, {}).get(q) or {}).get("defined") else None  # noqa: E731
        out["v2"] = {"right": lo["right"], "no": lo["no_response"], "wrong": lo["wrong"], "n": lo["n_scored"],
                     "sync": n_sync, "n_def": n_def, "cog2_ad": g("cog2", "ad"), "cog2_mmse": g("cog2", "mmse_lt24"),
                     "apoe_ad": g("apoe", "ad"), "comorb3_death": g("comorb3", "death")}
    return out or None


def zoo_generators():
    """Delphi 打包盒里能生成轨迹的（表 5 的那些），轨迹对比用。"""
    out = []
    for p in zoo_files(ZOO, "trajectory.json"):
        d, tr = os.path.basename(os.path.dirname(p)), _j(p)
        if tr and tr.get("applicable") and tr.get("matrix_5y"):
            out.append({"name": d, "link": f"{pkg_dir(d)}/README.md", "label": d, "group": "zoo", "traj": tr})
    return out


def rollout_rows(R):
    """9 模型图的行，全部用 rollout（生成未来再数）的读法（用户 2026-10-28：9 模型的图只比 rollout）：
    本来就能生成的（minimalrnn、ethos）用自己；只能直接预测的用它的 rollout 版（-sim、ar-*）；
    delphi-dedup 用 delphi-dedup-sim（打分表来自 rollout_landmarks.py；轨迹本来就是生成的，沿用 delphi-dedup 的；
    扰动来自 rollout_landmarks.py --perturb，和其它生成模型同一种打分（62 格 log OR）；还没跑的时候退回 perturb_v2 的
    rollout 一栏，它只有 ΔRMST 的打分、格子也少，和其它模型的 log OR 打分不能放一根柱子上比，只写数）。"""
    by = {r["name"]: r for r in R}
    out = []
    for name in SELECTED:
        r = by[name]
        if name in MZ.SIM_OF:
            out.append(dict(by[MZ.SIM_OF[name]], group="sim"))
        elif r["group"] == "gen":
            out.append(r)
        else:
            d = os.path.join(ZOO, pkg_dir(DELPHI_SIM))
            pert = pert_summary(d, d)
            pv = (_j(os.path.join(ZOO, pkg_dir(name), "perturb_v2.json")) or {}).get("rollout") or {}
            ps = pv.get("prespec_score") if pv.get("status") == "ok" and not pert else None
            out.append(dict(r, name=DELPHI_SIM, label=DELPHI_SIM, link=f"{pkg_dir(DELPHI_SIM)}/README.md",
                            p1=part1(_j(os.path.join(d, "metrics.json"))), p3=part3(_j(os.path.join(d, "transitions.json"))),
                            pert=pert, pert_note=f"ΔRMST score {ps['right']}/{ps['n_scored']}" if ps else None))
    return out


def nf(x):
    """计数：整数照写，seed 平均出来的小数留一位。"""
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def f3(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pc(x):
    return "—" if x is None else f"{x:.0%}"


# ----------------------------------------------------------------------------- 每个基线的 README


def prediction_sections(p1, p3, figs):
    """README 的第 1 节（第一部分 2 年预测，fig1 / fig2）和第 2 节（第三部分分期转换，fig3）。"""
    L = ["## 1 预测（第一部分，2 年）", "",
         "| 4 终点平均 AUC [95% CI] | " + " | ".join(END_ZH[e] for e in ENDS) + " | 平均 O/E |",
         "|---|---|---|---|---|---|",
         f"| **{f3(p1['auc'])}** [{f3(p1['ci'][0])}, {f3(p1['ci'][1])}] | "
         + " | ".join(f3(p1["ep"][e]) for e in ENDS) + f" | {'—' if p1['oe'] is None else f'{p1['oe']:.2f}'} |", "",
         "![fig1](fig1_auc_2y.png)", "", "![fig2](fig2_calibration_2y.png)", ""]
    if p3:
        L += ["## 2 分期转换（第三部分）", "",
              "| 6 题平均 AUC 2 / 5 / 10 年 | 同一当前状态 + 年龄段内（5 年） | 竞争风险敏感性（10 年） |", "|---|---|---|",
              f"| {' / '.join(f3(p3['auc'][H]) for H in (2, 5, 10))} | {f3(p3['within_age5'])} | {f3(p3['cr10'])} |", ""]
        if "fig3" in figs:
            L += ["![fig3](fig3_transitions.png)", ""]
    return L


def perturb_section(pt, figs, num, how):
    """README 的扰动一节（表 3 同一组数）：第二部分 perturb.json、第三部分 perturb_v2.json；都没有就不写。"""
    if not pt:
        return []
    a, b = pt.get("v1"), pt.get("v2")
    y = lambda x: "—" if x is None else f"{x:+.2f}"           # noqa: E731
    c1 = f"{nf(a['right'])} / {nf(a['no'])} / {nf(a['wrong'])}" if a else "—"
    c2 = y(a["apoe_ad"]) if a else "—"
    c3 = f"{b['right']} / {b['no']} / {b['wrong']}" if b else "—"
    c4 = f"{b['sync']} / {b['n_def']}" if b else "—"
    c5 = f"{y(b['cog2_ad'])} / {y(b['cog2_mmse'])}" if b else "—"
    L = [f"## {num} 扰动", "", how, "",
         "| 第二部分：预设方向 对 / 无响应 / 错（/ 20） | APOE→AD 2 年 log OR | 第三部分：5 年 log OR 对 / 无响应 / 错（/ 62） | "
         "AD 和认知风险一起升高 | MMSE + 认知一起下降：10 年内无 AD / 无 MMSE<24 时间的变化（年） |", "|---|---|---|---|---|",
         f"| {c1} | {c2} | {c3} | {c4} | {c5} |", ""]
    if "fig6" in figs:
        L += ["![fig6](fig6_perturbation.png)", ""]
    return L


def delphi_sim_readme(r):
    """delphi-dedup-sim 的 README（rollout_rows 里 Delphi 那一行）：只有打分表是 rollout 出来的，轨迹、扰动沿用 delphi-dedup。"""
    d = os.path.join(ZOO, pkg_dir(DELPHI_SIM))
    figs = FG.build_package(d)
    base = "delphi-dedup"
    L = [f"# {DELPHI_SIM}", "",
         f"[{base}]({rel_link(DELPHI_SIM, base)}) 的 **rollout 读法**（`unified_eval/rollout_landmarks.py`，2026-10-28 加）：同一个 ckpt（seed 42），"
         "不把查询位置的速率换算成风险（1 − exp(−速率 × H)），而是从每个站点的查询 token 起往后生成 100 条 10 年的未来，"
         "H 年内首次越线的条数 ÷ 100 = 风险。9 模型图（[representative/](" + rel_link(DELPHI_SIM, "representative/README.md")
         + ")）一律用这种读法，这个包就是图里 Delphi 那一行。", "",
         f"10 年轨迹本来就是生成的，见 [{base}]({rel_link(DELPHI_SIM, base)}) 的 fig4、fig5；这里只有用 rollout 读法重算的部分。", ""]
    L += prediction_sections(r["p1"], r["p3"], figs)
    if (r.get("pert") or {}).get("v2"):
        L += perturb_section(r["pert"], figs, num="3", how=(
            "第三部分的扰动（`rollout_landmarks.py --perturb`）：和其它生成模型同一种做法、同一种打分。同一批人、同一组改动，"
            "改法是 Delphi 的 token 改法（`perturb_v2.translate_tokens`）；每条改过的记录从站点往后生成 100 条，"
            "风险 = (事件条数 + 0.5) / 101。第二部分（20 格）没有用这种读法重跑。"))
    else:
        L += [f"扰动只有 delphi-dedup 第 5 节里 rollout 一栏按 ΔRMST 的打分{'（' + r['pert_note'] + '）' if r.get('pert_note') else ''}，"
              "没有其它模型那种 62 格 log OR 的打分。", ""]
    if "fig8" in figs:
        L += sec_timing(d, num="4" if (r.get("pert") or {}).get("v2") else "3")
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("\n".join(L))


def representative_readme(Ra):
    """representative/README.md：9 模型图的索引（哪 9 个、打包盒在哪、图在哪），数字和 OVERVIEW.png 同源。"""
    def link(r):
        return f"[{r['name']}]({os.path.relpath(r['link'], REP_DIR)})"
    L = ["# 9 个代表模型", "",
         "9 模型图（OVERVIEW.png、MODELS*.png）里的模型。图里一律用 rollout 的读法（每个站点往后模拟多条未来，数事件当风险；多数 100 条，ETHOS 50 条）："
         "只能直接预测的模型画它的 rollout 版（-sim / ar-*）。9 个打包盒都在这里，"
         f"Delphi 那一个是它的 rollout 版 {DELPHI_SIM}。"
         "每个模型的代码在哪：[models_code/README.md](../models_code/README.md) 的第一节。"
         "原版（直接预测，2026-11-05 起不参与评估）的打包盒在 [models_not_usable/](../models_not_usable/README.md)，"
         "其余所有模型（包括别的 Delphi）在 [models_self/](../models_self/README.md)，全部可用模型的表在 [MODELS.md](../reports/MODELS.md)。", "",
         "## 图", "",
         "| 图 | 内容 |", "|---|---|",
         "| [OVERVIEW.png](../reports/OVERVIEW.png) | 总图：7 个面板（2 年 AUC、5 年 AUC、校准、时间误差、10 年 AD / 死亡、5 年转移矩阵、扰动方向） |",
         "| [MODELS.png](../reports/MODELS.png) | 2 年、5 年 AUC |",
         "| [MODELS_calibration.png](../reports/MODELS_calibration.png) | 2 年校准：O/E |",
         "| [MODELS_traj.png](../reports/MODELS_traj.png) | 5 年转移矩阵、10 年 AD / 死亡发生率 |",
         "| [MODELS_sim.png](../reports/MODELS_sim.png) | 5 个直接预测模型改成 rollout 前后 |",
         "| [MODELS_timing.png](../reports/MODELS_timing.png) | 时间：AD / MMSE<24 / 死亡，模型平均错几年，虚线 = 所有人猜同一个数 |", "",
         "## 模型", "",
         "| 模型 | 原版 | 2 年 4 终点 AUC | 5 年 6 题 AUC | 5 年转移矩阵每格差（点） | 扰动 对 / 无 / 错（/ 62） |",
         "|---|---|---|---|---|---|"]
    for r in sorted(Ra, key=lambda r: -r["p1"]["auc"]):
        o = ORIG_OF.get(r["name"], "delphi-dedup" if r["name"] == DELPHI_SIM else None)
        orig = f"[{o}]({os.path.relpath(pkg_dir(o), REP_DIR)}/README.md)" if o else "—（本身就能生成未来）"
        v2 = (r.get("pert") or {}).get("v2")
        g = traj_summary(r["traj"])["matrix_gap"] if r.get("traj") else None
        p3 = r["p3"]
        L.append(f"| {link(r)} | {orig} | {f3(r['p1']['auc'])} | {f3(p3['auc'][5]) if p3 else '—'} | "
                 f"{'—' if g is None else f'{100 * g:.1f}'} | "
                 f"{f'{v2['right']} / {v2['no']} / {v2['wrong']}' if v2 else (r.get('pert_note') or '—')} |")
    L += ["", "数字同 OVERVIEW.png（val）。delphi-dedup-sim 的转移矩阵是 delphi-dedup 的（轨迹本来就是生成的）。", "",
          "## 时间：6 道题", "", "MODELS_timing.png 只画 AD、MMSE<24、死亡；6 道题的数如下（和图同源，各模型的 timing.json）。", ""]
    L += table_timing_md(Ra) + [""]
    with open(os.path.join(ZOO, REP_DIR, "README.md"), "w") as f:
        f.write("\n".join(L))


def package_readme(r):
    d = os.path.join(ZOO, pkg_dir(r["name"]))
    figs = FG.build_package(d)
    p1, p3, bl = r["p1"], r["p3"], r["bl"] or {}
    who = "NACC 预训练实验（代码在 `models_code/pretrain.py`）" if r["name"] in NACC else "代码在 `models_code/`"
    L = [f"# {r['name']}", "", f"**{r['cite']}**。家族：{r['family']}。{who}，"
         "输入是除 Delphi 外所有模型共用的同一份数据（`models_code/input_data.py` 产出的 `newmodel_data_trans.npz`：每次访视 54 维 + 静态背景 25 维，l2c 再加手工历史特征 120 维），"
         "split811：train 训练、test 早停（有超参搜索的模型也在它上面挑超参）、val 只报一次。", ""]
    if r["name"] in CHOSEN:
        L += [chosen_text(), ""]
    if r["name"] in NOT_USABLE:
        tw = TWIN.get(r["name"])
        L += [f"**不可用，2026-11-05 起不参与评估**（[models_not_usable/]({rel_link(r['name'], NOT_USABLE_DIR + '/README.md')})）："
              "不能生成未来，只回答'H 年内会不会'，训练目标就是评估题本身，和要从模拟出来的未来里数答案的模型不在一条起跑线上。"
              + (f"留在评估里的是 [{tw}]({rel_link(r['name'], tw)})。" if tw else "这个没有留在评估里的版本。"), ""]
    L += ["## 怎么预测未来", "", FUT_EXPLAIN[r["future"]], ""]
    if r["name"] in NACC:
        c = MZ.NACC_CONTROL.get(r["name"])
        L += [f"这是 NACC 预训练实验的一个变体（`pretrain.py`，表 5，在 [models_not_usable/]({rel_link(r['name'], NOT_USABLE_DIR + '/README.md')})）："
              + {"scratch": "只在 ROSMAP 上训练，是同一结构的对照。", "nacc": "先在 NACC 上从零训练，再在 ROSMAP 的 train 上微调。",
                 "zeroshot": "NACC 上训练好的权重直接在 ROSMAP 上读出，不微调（NACC 没有的输入改成'没测'）。"}.get(
                  r["args"].get("init", "nacc"), "先在 NACC 上训练，再在 ROSMAP 上微调。")
              + (f"对照：[{c}]({rel_link(r['name'], c)})。" if c else ""), ""]
    if r["name"] in SEQ_AR:
        L += ["自回归（`rollout_versions.py`，2026-10-28 起）：读到当前为止的整条病历，只学下一步——这一年里死亡、得 AD 的概率 + "
              "明年那次访视的完整分布（每个连续量一个高斯和是否缺失，每个状态和是否缺失，卒中 / 抑郁新出现），"
              "训练目标就是这一步的似然，没有任何'H 年内会不会'的题目头。抽出的访视接到序列后面再过一遍网络，一年一年往后滚；"
              "各题各时间窗的风险、轨迹、扰动都从模拟里数出来。", ""]
    if r["name"] in TRFX:
        L += [TRFX_TEXT.format(tr=rel_link(r["name"], "transformer")), ""]
    if r["name"] in SIMS:
        o = ORIG_OF[r["name"]]
        L += [f"这是 [{o}]({rel_link(r['name'], o)}) 的 rollout 版（`rollout_versions.py`）：不让它直接回答\"H 年内会不会发生\"，"
              "改学\"明年会怎样\"（这一年里死亡、得 AD 的概率 + 明年那次访视的分布），再一年一年往后滚；"
              "风险部分和原模型同一类、同一组超参，随访指标用同族的学习器。", ""]
    L += prediction_sections(p1, p3, figs)
    if r["traj"]:
        s = traj_summary(r["traj"])
        M = r["traj"]["matrix_5y"]["mmse"]
        L += ["## 3 转移矩阵和 10 年轨迹", "",
              "| 5 年 MMSE 转移矩阵：平均每格相差（百分点；完美模拟器的噪声下限） | 10 年 AD 实际 / 模拟 | 10 年死亡 实际 / 模拟 |",
              "|---|---|---|",
              f"| {100 * s['matrix_gap']:.0f}（{100 * matrix_gap_floor(M):.0f}） | {pc(s['ad_obs'])} / {pc(s['ad_sim'])} | "
              f"{pc(s['death_obs'])} / {pc(s['death_sim'])} |", ""]
        if "fig4" in figs:
            L += ["![fig4](fig4_transition_matrix.png)", "", "![fig5](fig5_trajectories.png)", ""]
    L += perturb_section(r.get("pert"), figs, num="4", how=(
        "和其它模型同一批人、同一组改动（改法照统一输入的特征空间，`perturb_features.apply`、`perturb_v2.translate_features`）。"
        + (("能生成未来的模型：风险 = 从改过的站点往后模拟的轨迹里事件的比例（(k + 0.5) / (n + 1)，条数见 MODELS.md）。" if r["future"] == "rollout" else
           "多状态 Markov：改过的站点上的风险精确算，不抽样。" if r["future"] == "exact" else ""))))
    if "fig8" in figs:
        L += sec_timing(d, num="5")
    L += ["## 训练", "", f"- 选中的设置：`{json.dumps(bl.get('chosen'), ensure_ascii=False)}`",
          f"- test 段（早停 / 选择用）6 题 × 2/5/10 年平均 AUC：{f3(bl.get('test_macro_auc_2_5_10'))}；"
          f"总用时 {bl.get('minutes_total', float('nan')):.1f} 分钟", ""]
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------- 图


def fig_prediction(R, path):
    R = sorted([r for r in R if r["p1"]], key=lambda r: r["p1"]["auc"])
    n = len(R)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.27 * n + 1.8), sharey=True)
    y = np.arange(n)
    for ax, key, title in ((axes[0], "p1", "A  Four endpoints, 2-year risk (Part 1)"),
                           (axes[1], "p3", "B  Six stage-crossing questions, 5-year (Part 3)")):
        for i, r in enumerate(R):
            if key == "p1":
                v, ci = r["p1"]["auc"], r["p1"]["ci"]
            else:
                if not r["p3"] or r["p3"]["auc"][5] is None or r["p3"]["all"] is False:
                    continue
                v, ci = r["p3"]["auc"][5], r["p3"]["ci5"]
            ax.barh(i, v - 0.5, left=0.5, height=0.7, color=GROUP_COL[r["group"]], zorder=2)
            if ci:
                ax.plot(ci, [i, i], color=FG.INK, lw=0.9, zorder=3)
        ax.set_title(title)
        ax.set_xlim(auc_left([r["p1"]["auc"] for r in R]), 0.95)   # 有低于 0.5 的（caide）才往左放宽
        ax.axvline(0.5, color=FG.INK, lw=1, zorder=3)
        ax.set_xlabel("Mean AUC\n(bars start at 0.5 = chance; black line = 95% CI)")
        ax.grid(axis="x", color=FG.FAINT, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["name"] for r in R])
    for t, r in zip(axes[0].get_yticklabels(), R):
        if r["name"] in CHOSEN:
            t.set_fontweight("bold")
    axes[0].set_ylim(-0.7, n - 0.3)
    h = [Patch(facecolor=GROUP_COL[g], label=GROUP_LABEL[g]) for g in GROUP_COL if any(r["group"] == g for r in R)]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01))
    best = max(R, key=lambda r: r["p1"]["auc"])
    tr = next((r for r in R if r["name"] == "transformer"), None)
    dl = next((r for r in R if r["name"] in ("delphi-dedup", DELPHI_SIM)), None)
    sup = "; ".join([f"Highest {best['name']} {best['p1']['auc']:.3f}"]
                    + ([f"transformer {tr['p1']['auc']:.3f}"] if tr and tr is not best else [])
                    + ([f"Delphi {dl['p1']['auc']:.3f}"] if dl and dl is not best else []))
    fig.suptitle(sup, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    FG._save(fig, path)


def fig_calibration(R, path):
    """选中模型的 2 年风险校准（fig2 的横向对比）：O/E（实际 ÷ 预测，柱子从 1 往外长，对数轴），一列一个终点。
    行序 = 校准从好到差（4 个终点 |log O/E| 的平均，最准的在最上面）。ECE 不画（和 O/E 大体同向；数在各模型的 metrics.json / README）。"""
    R = [r for r in R if r["p1"]]
    M = [_j(os.path.join(ZOO, pkg_dir(r["name"]), "metrics.json"))["endpoints"] for r in R]
    off = [np.mean([abs(math.log(m[f"{e}_2y"]["o_e"])) for e in ENDS]) for m in M]
    order = np.argsort(off)
    R, M = [R[k] for k in order], [M[k] for k in order]
    n = len(R)
    lab = {"ad": "AD dementia", "death": "Death", "mmse": "MMSE < 24", "cogn": "Cognition < −1"}
    fig, axes = plt.subplots(1, len(ENDS), figsize=(13.5, 0.32 * n + 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(n)
    OE_MAX = 14
    oe_hi = min(OE_MAX, max(2.5, 1.3 * max(m[f"{e}_2y"]["o_e"] for m in M for e in ENDS)))
    for a, e in zip(axes, ENDS):
        for i, (r, m) in enumerate(zip(R, M)):
            oe = m[f"{e}_2y"]["o_e"]
            a.barh(i, min(oe, OE_MAX) - 1, left=1, height=0.7, color=GROUP_COL[r["group"]], zorder=2)
            if oe > 6:                                        # 长柱：数写在柱子里，不越界
                a.text(min(oe, OE_MAX) * 0.94, i, f"{oe:.2f}", va="center", ha="right", fontsize=8, color="white",
                       fontweight="bold", zorder=4)
            else:                                             # 其余写在柱尾（小于 1 的写在参照线右边）
                a.text(max(oe, 1) * 1.05, i, f"{oe:.2f}", va="center", ha="left", fontsize=8, color=FG.INK)
        a.set_xscale("log")
        a.set_xlim(0.5, oe_hi)
        a.set_xticks([t for t in (0.5, 1, 2, 4, 8) if t <= oe_hi])
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        a.xaxis.set_minor_locator(NullLocator())
        a.axvline(1, color=FG.INK, lw=1, zorder=3)
        a.set_title(lab[e], fontsize=11)
        a.set_xlabel("observed ÷ predicted", fontsize=9)
        a.grid(axis="x", color=FG.FAINT, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(axis="x", labelsize=8.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["name"] for r in R], fontsize=9.5)
    for t, r in zip(axes[0].get_yticklabels(), R):
        if r["name"] in CHOSEN:
            t.set_fontweight("bold")
    axes[0].set_ylim(n - 0.4, -0.6)
    fig.suptitle("2-year calibration: observed ÷ predicted (1 = calibrated, > 1 = predicts too low)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.0)
    h = [Patch(facecolor=GROUP_COL[g], label=GROUP_LABEL[g]) for g in GROUP_COL if any(r["group"] == g for r in R)]
    fig.legend(handles=h, loc="upper left", ncol=len(h), bbox_to_anchor=(0.01, 0.0), fontsize=9.5)
    FG._save(fig, path)


def auc_left(vals):
    """AUC 柱子从 0.5 起；有低于 0.5 的就把轴往左放宽到 0.4，柱子往左长。"""
    return 0.4 if min(v for v in vals if v is not None) < 0.5 else 0.5


def fig_traj(G, path):
    G = [g for g in G if g["traj"] and g["traj"].get("matrix_5y")]
    for g in G:
        g["s"] = traj_summary(g["traj"])
    G = sorted(G, key=lambda g: -g["s"]["matrix_gap"])
    n = len(G)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.32 * n + 1.9), sharey=True)
    y = np.arange(n)
    floor = matrix_gap_floor(G[0]["traj"]["matrix_5y"]["mmse"])
    ax = axes[0]
    ax.axvspan(0, 100 * floor, color=FG.FAINT, alpha=0.6, lw=0, zorder=0)
    ax.text(100 * floor / 2, -0.9, "noise floor", ha="center", va="top", fontsize=9, color=FG.SOFT)
    for i, g in enumerate(G):
        ax.barh(i, 100 * g["s"]["matrix_gap"], height=0.7, color=GROUP_COL[g["group"]], zorder=2)
    ax.set_title("A  5-year MMSE transition matrix error")
    ax.set_xlabel("Mean |simulated − observed|, points per cell\n(lower = more realistic)")
    ax.set_xlim(left=0)
    ax.set_ylim(-1.6, n - 0.4)
    ax = axes[1]
    ax.axvline(1, color=FG.INK, lw=1, zorder=3)
    for i, g in enumerate(G):
        s = g["s"]
        for q, dy in (("ad", 0.18), ("death", -0.18)):
            if s[f"{q}_obs"]:
                ax.barh(i + dy, s[f"{q}_sim"] / s[f"{q}_obs"] - 1, left=1, height=0.36, color=FG.EP_COL[q], zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 4)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title("B  10-year incidence, simulated ÷ observed")
    ax.set_xlabel("Ratio (bars start at 1 = right;\nlog scale)")
    ax.legend(handles=[Patch(facecolor=FG.EP_COL["ad"], label="AD dementia"), Patch(facecolor=FG.EP_COL["death"], label="Death")],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for a in axes:
        a.grid(axis="x", color=FG.FAINT, lw=0.8)
        a.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([g["name"] for g in G])
    for t, g in zip(axes[0].get_yticklabels(), G):
        if g["name"] in CHOSEN:
            t.set_fontweight("bold")
    lab = dict(GROUP_LABEL, direct="Rollout version of a direct-prediction model")
    h = [Patch(facecolor=GROUP_COL[k], label=lab[k]) for k in GROUP_COL if any(g["group"] == k for g in G)]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01))
    best = min(G, key=lambda g: g["s"]["matrix_gap"])
    fig.suptitle(f"Most realistic 10-year simulator: {best['name']} ({100 * best['s']['matrix_gap']:.0f} points per cell; "
                 f"noise floor {100 * floor:.0f})", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    FG._save(fig, path)


def overview_values(r):
    """一行（一个方法）在总览图 7 个面板上的数。不能生成未来的方法，模拟那两栏用它的 rollout 版（-sim、ar-*），proxy 记下来。"""
    d = os.path.join(ZOO, pkg_dir(r["name"]))
    tm = _j(os.path.join(d, "timing.json"))
    gain = None
    if tm and tm.get("applicable"):
        qs = [tm["questions"][q] for q in FG.PV_EP if "mae" in tm["questions"].get(q, {})]
        gain = float(np.mean([q["mae"] - q["constant_mae"] for q in qs])) if len(qs) == len(FG.PV_EP) else None
    tr, proxy = r["traj"], None
    if tr is None and r["name"] in MZ.SIM_OF:
        proxy = MZ.SIM_OF[r["name"]]
        t2 = _j(os.path.join(ZOO, pkg_dir(proxy), "trajectory.json"))
        tr = t2 if t2 and t2.get("applicable") else None
    s = traj_summary(tr) if tr else None
    v2 = (r["pert"] or {}).get("v2")
    p3 = r["p3"] if r["p3"] and r["p3"]["auc"][5] is not None and r["p3"]["all"] is not False else None
    return {"auc2": (r["p1"]["auc"], r["p1"]["ci"]), "auc5": (p3["auc"][5], p3["ci5"]) if p3 else None,
            "oe": r["p1"]["oe"], "timing": gain,
            "ad": s["ad_sim"] / s["ad_obs"] if s and s["ad_obs"] else None,
            "death": s["death_sim"] / s["death_obs"] if s and s["death_obs"] else None,
            "gap": 100 * s["matrix_gap"] if s and s["matrix_gap"] is not None else None,
            "floor": 100 * matrix_gap_floor(tr["matrix_5y"]["mmse"]) if s and s["matrix_gap"] is not None else None,
            "pert": v2["right"] / v2["n"] if v2 and v2.get("n") else None,
            "proxy": proxy if tr else None}


def fig_overview(R, path):
    """一张总览：每个方法一行（按 2 年 AUC 排），7 个横向柱状面板，柱子从参照往外长，颜色 = 三组。
    A 2 年 AUC  B 5 年 AUC  C 校准 O/E  D 时间（比常数少错几年）  E 10 年 AD / 死亡 模拟 ÷ 实际  F 5 年转移矩阵每格差  G 扰动方向对的比例。
    不能生成未来的方法，E、F 画它的 rollout 版（浅色柱）。"""
    R = sorted([r for r in R if r["p1"]], key=lambda r: r["p1"]["auc"])
    V = [overview_values(r) for r in R]
    n = len(R)
    y = np.arange(n)
    H = 0.72
    P = [("auc2", "A  Predict: 2-year AUC\n(4 outcomes)", "AUC; bars start at 0.5 = chance"),
         ("auc5", "B  Predict: 5-year AUC\n(6 stage-crossing questions)", "AUC; bars start at 0.5 = chance"),
         ("oe", "C  Calibration: observed ÷\npredicted, 2 years", "ratio, log scale (1 = calibrated)"),
         ("timing", "D  Timing: years of error\nvs the same guess for all", "model − same guess (years;\n< 0 = better)"),
         ("sim", "E  Simulate: 10-year AD and\ndeath, simulated ÷ observed", "ratio, log scale (1 = matches)"),
         ("gap", "F  Simulate: 5-year MMSE\ntransition matrix error", "points per cell (0 = matches)"),
         ("pert", "G  Perturb: responses in the\nexpected direction", "share of 62 scored cells")]
    FH = (0.24 if n > 20 else 0.42) * n + 2.6           # 行少（代表模型）时每行高一些
    fig, axes = plt.subplots(1, len(P), figsize=(2.75 * len(P) + 2.6, FH), sharey=True,
                             gridspec_kw={"wspace": 0.16})
    na = dict(va="center", fontsize=7.5, color=FG.SOFT)
    for ax, (key, title, xl) in zip(axes, P):
        for i, (r, v) in enumerate(zip(R, V)):
            c = GROUP_COL[r["group"]]
            if key in ("auc2", "auc5"):
                if v[key] is None:
                    ax.text(0.505, i, "answers only some questions", **na)
                    continue
                a, ci = v[key]
                ax.barh(i, a - 0.5, left=0.5, height=H, color=c)
                if ci:
                    ax.plot(ci, [i, i], color=FG.INK, lw=0.8)
            elif key == "oe":
                x = min(v["oe"], 8.0)
                ax.barh(i, x - 1, left=1, height=H, color=c)
                if v["oe"] > 8:
                    ax.text(8.3, i, f"{v['oe']:.0f}", **na)
            elif key == "timing":
                if v["timing"] is None:
                    ax.text(0.02, i, "no 10-year curve", **na)
                    continue
                ax.barh(i, v["timing"], height=H, color=c)
            elif key == "sim":
                if v["ad"] is None:
                    ax.text(1.03, i, "cannot simulate", **na)
                    continue
                al = 0.45 if v["proxy"] else 1.0
                ax.barh(i + H / 4, v["ad"] - 1, left=1, height=H / 2, color=FG.EP_COL["ad"], alpha=al)
                ax.barh(i - H / 4, v["death"] - 1, left=1, height=H / 2, color=FG.EP_COL["death"], alpha=al)
            elif key == "gap":
                if v["gap"] is None:
                    ax.text(0.3, i, "cannot simulate", **na)
                    continue
                ax.barh(i, v["gap"], height=H, color=c, alpha=0.45 if v["proxy"] else 1.0)
            else:
                if v["pert"] is None:
                    ax.text(0.01, i, r.get("pert_note") or "not run", **na)
                    continue
                ax.barh(i, v["pert"], height=H, color=c)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(xl, fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="x", color=FG.FAINT, lw=0.8)
        ax.set_axisbelow(True)
    ax = dict(zip([p[0] for p in P], axes))
    for k in ("auc2", "auc5"):
        ax[k].set_xlim(auc_left([v[k][0] for v in V if v[k]]), 0.95)
        ax[k].axvline(0.5, color=FG.INK, lw=1, zorder=3)
    for k in ("oe", "sim"):
        ax[k].set_xscale("log")
        ax[k].axvline(1, color=FG.INK, lw=1)
        ax[k].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax[k].xaxis.set_minor_locator(NullLocator())
    ax["oe"].set_xlim(0.5, 12)
    ax["oe"].set_xticks([0.5, 1, 2, 4, 8])
    ax["sim"].set_xlim(0.25, 4)
    ax["sim"].set_xticks([0.25, 0.5, 1, 2, 4])
    ax["timing"].axvline(0, color=FG.INK, lw=1)
    ax["timing"].set_xlim(-0.8, 0.8)
    floor = next((v["floor"] for v in V if v["floor"]), None)
    if floor:
        ax["gap"].axvspan(0, floor, color=FG.FAINT, alpha=0.7, lw=0, zorder=0)
    ax["gap"].set_xlim(0, None)
    ax["pert"].set_xlim(0, 1)
    ax["pert"].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["name"] for r in R], fontsize=9)
    for t, r in zip(axes[0].get_yticklabels(), R):
        if r["name"] in CHOSEN:
            t.set_fontweight("bold")
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.7, n + 0.2)
    h = [Patch(facecolor=GROUP_COL[g], label=GROUP_LABEL[g]) for g in GROUP_COL if any(r["group"] == g for r in R)]
    h += [Patch(facecolor=FG.EP_COL["ad"], label="E: AD dementia"), Patch(facecolor=FG.EP_COL["death"], label="E: death")]
    if any(v["proxy"] for v in V):
        h.append(Patch(facecolor=FG.SOFT, alpha=0.45, label="E, F: lighter = its rollout version (-sim / ar-*)"))
    fig.legend(handles=h, loc="lower left", ncol=3, bbox_to_anchor=(0.01, -0.005), fontsize=9.5)
    top = lambda k, sign=1: max((v for v in zip(R, V) if v[1][k] is not None),   # noqa: E731
                                key=lambda rv: sign * (rv[1][k][0] if isinstance(rv[1][k], tuple) else rv[1][k]))
    b2, bt, bg = top("auc2"), top("timing", -1), top("gap", -1)
    fig.suptitle(f"{'All ' if n > 20 else ''}{n} {'' if n > 20 else 'representative '}models: best 2-year AUC {b2[0]['name']} {b2[1]['auc2'][0]:.3f}; "
                 f"best timing {bt[0]['name']} ({bt[1]['timing']:+.2f} y vs the same guess); "
                 f"most realistic simulator {(bg[1]['proxy'] or bg[0]['name'])} ({bg[1]['gap']:.0f} points per cell)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=0.995)
    ar = any(r["name"] == DELPHI_SIM for r in R)
    nobar = any(r["name"] == DELPHI_SIM and not (r.get("pert") or {}).get("v2") for r in R)
    fig.text(0.01, -0.035, "Validation set. Rows sorted by panel A; bold = zoo models (Delphi). "
             + ("Every model is read the same way: risks counted from 100 simulated futures per starting point (rollout). "
                if ar else "")
             + "Black lines in A and B = 95% CI. C: mean over 4 outcomes"
             + ("" if ar else "; Delphi rates are not probabilities") + ", values above 8 are "
             "written out. D: mean over AD, MMSE<24 and death (fig8 of each model).\nF: grey band = what a perfect simulator would "
             "score from sampling noise alone. G: 5-year odds ratios of 7 changes × 3 outcomes with a pre-specified direction "
             "(perturb_v2)" + ("; Delphi's rollout perturbation run was scored on ΔRMST only, so it gets no bar" if nobar
                               else "" if ar else ". NACC-pretraining variants are in MODELS_nacc.png") + ".",
             fontsize=9, color=FG.SOFT, va="top")
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 1.1 / FH, bottom=1.0 / FH)
    FG._save(fig, path)


# ----------------------------------------------------------------------------- MODELS.md


def timing_of(r):
    tm = _j(os.path.join(ZOO, pkg_dir(r["name"]), "timing.json"))
    return tm if tm and tm.get("applicable") else None


def fig_timing(R, path):
    """9 模型的时间对比（每个模型自己的 fig8 放在一起），AD / MMSE<24 / 死亡各一个面板：
    柱 = 模型的平均绝对误差（年，10 年内真的发生的人），虚线 = 所有人都猜同一个数（实际时间的中位数）的平均绝对误差。
    柱比虚线短才说明模型分得出谁早谁晚。95% CI、秩相关、按实际时间分段的误差在 table_timing_md 和各模型的 fig8。"""
    R = sorted([r for r in R if r["p1"] and timing_of(r)], key=lambda r: r["p1"]["auc"])
    T = [timing_of(r) for r in R]
    n = len(R)
    Q3 = FG.PV_EP
    fig, axes = plt.subplots(1, len(Q3), figsize=(12, 0.34 * n + 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    hi = 1.15 * max(t["questions"][q]["mae"] for t in T for q in Q3)
    for a, q in zip(axes, Q3):
        for i, (r, t) in enumerate(zip(R, T)):
            v = t["questions"][q]["mae"]
            a.barh(i, v, height=0.7, color=GROUP_COL[r["group"]], zorder=2)
            a.text(v - 0.06, i, f"{v:.1f}", va="center", ha="right", fontsize=8.5, color="white", fontweight="bold",
                   zorder=4)
        c = T[-1]["questions"][q]["constant_mae"]
        a.axvline(c, color=FG.INK, lw=1.2, ls="--", zorder=3)
        a.set_title(FG.PV_EP_LABEL[q], fontsize=11)
        a.set_xlim(0, hi)
        a.set_xlabel("average error (years)", fontsize=9)
        a.grid(axis="x", color=FG.FAINT, lw=0.8)
        a.set_axisbelow(True)
    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels([r["name"] for r in R], fontsize=9.5)
    for tl, r in zip(axes[0].get_yticklabels(), R):
        if r["name"] in CHOSEN:
            tl.set_fontweight("bold")
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.6, n - 0.4)
    fig.suptitle("When will it happen? Average error in years (dashed line = same guess for everyone)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.0)
    h = [Patch(facecolor=GROUP_COL[g], label=GROUP_LABEL[g]) for g in GROUP_COL if any(r["group"] == g for r in R)]
    fig.legend(handles=h, loc="upper left", ncol=len(h), bbox_to_anchor=(0.01, 0.0), fontsize=9.5)
    FG._save(fig, path)


def table_timing_md(R):
    """6 道题的时间误差（模型 − 同一个猜测），和 fig_timing 同源；图只画 3 道。"""
    R = sorted([r for r in R if r["p1"] and timing_of(r)], key=lambda r: -r["p1"]["auc"])
    T = [timing_of(r) for r in R]
    t0 = T[0]["questions"]
    L = ["| 模型 | " + " | ".join(f"{FG.TRANS_LABEL[q]}（{t0[q]['n']} 人）" for q in FG.TRANS_ORDER) + " |",
         "|---|" + "---|" * len(FG.TRANS_ORDER)]
    for r, t in zip(R, T):
        c = []
        for q in FG.TRANS_ORDER:
            v = t["questions"][q]
            lo, hi = v["mae_minus_constant_ci"]
            s = f"{v['mae'] - v['constant_mae']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            c.append(f"**{s}**" if hi < 0 else s)
        L.append(f"| {r['name']} | " + " | ".join(c) + " |")
    L += ["", "模型的平均绝对误差 − 所有人都猜同一个数（实际时间的中位数）的平均绝对误差，年；负 = 比同一个猜测准；"
          "加粗 = 95% CI 整个在 0 以下。人数 = 10 年内真的发生的人（个别模型少几个没有预测时间的人）。"]
    return L


def name_md(r):
    """表里的模型名：Delphi 用它的中文标签；选定的模型（CHOSEN）加粗。"""
    s = f"[{r['label'] if r['group'] == 'zoo' else r['name']}]({r['link']})"
    return f"**{s}**" if r["name"] in CHOSEN else s


# ----------------------------------------------------------------------------- 选模型（select_model.py 的结果）和结论（2026-11-08）

SELECT_JSON = os.path.join(ZOO, REPORT_DIR, "select_model.json")
EVAL_ZH = {"auc2": "预测：4 终点 2 年 AUC", "auc5": "分期转换：6 题 5 年 AUC", "calib": "校准：2 年 log(O/E) 绝对值的平均",
           "timing": "时间：平均绝对误差 − 常数猜测（年）", "inc10": "10 年 AD / 死亡发生率：模拟和实际差的绝对值",
           "gap5": "5 年 MMSE 转移矩阵每格差", "pert": "扰动 62 格：方向对的比例"}
VERDICT_ZH = {"best": "最好", "tied_best": "并列最好", "behind": "落后"}


def _sel():
    return _j(SELECT_JSON)


def chosen_text():
    S = _sel()
    s = "**选定的模型**：所有模型在同一套评估上互相比较，最后选用 transformer（2026-11-08）。"
    if S:
        v = list(S["overall"]["verdicts"].values())
        s += (f"按选模型的规则（每一项都是最好或并列最好），7 项评估里 {v.count('best')} 项最好、{v.count('tied_best')} 项并列最好、"
              f"{v.count('behind')} 项落后，见 [MODELS.md]({rel_link('transformer', REPORT_DIR + '/MODELS.md')}) 的'选模型'一节。")
    return s


def selection_md():
    """MODELS.md 的'选模型'一节：7 项评估上 transformer 的位置（reports/select_model.json，RIS 上 select_model.py 算的）。"""
    S = _sel()
    if not S:
        return ["## 选模型", "", "（还没有 reports/select_model.json：在 RIS 上跑 `models_code/select_model.py`。）", ""]
    f3_ = lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"        # noqa: E731
    L = ["## 选模型：transformer 在 7 项评估上的位置", "",
         f"规则（用户 2026-11-04 定、2026-11-08 确认）：和同一份 val 数据上的全部 {len(S['pool'])} 个可用模型比"
         "（能生成未来的 49 个 + 表 1 的 4 个 Delphi），7 项评估 = 总览图的 7 个面板。每一项按点估计排序；transformer 排第一 = 最好；"
         "否则和排在它前面的每个模型做按人配对 bootstrap（val 的人重抽 "
         f"{S['B']} 次，所有模型、所有评估共用同一组重抽样），差值用 max-T 同时 95% 区间（对排在前面的这些对手做多重比较校正）："
         "有一个区间整个在 0 的不利一侧 = 落后，否则 = 并列最好。每一项都是最好或并列最好，才算整体最好。"
         "规则和代码（`models_code/select_model.py`）在跑之前写定；每一项的点估计先和各打包盒的 JSON 逐个核对，对不上就停。", "",
         "| 评估 | transformer（名次） | 点估计最好的 | 离判'落后'最近的对手：差 [max-T 同时 95% 区间] | 判定 | Bonferroni（敏感性） |",
         "|---|---|---|---|---|---|"]
    for k, r in S["evaluations"].items():
        s = r["standing"]["transformer"]
        if s["verdict"] == "best":
            near, bon = "—", "—"
        else:
            w = min(s["vs_above"], key=lambda x: x["maxT_ci"][1])
            near = f"{w['model']}：{w['diff']:+.4f} [{w['maxT_ci'][0]:+.4f}, {w['maxT_ci'][1]:+.4f}]"
            bon = VERDICT_ZH[s["bonferroni_verdict"]]
        L.append(f"| {EVAL_ZH[k]} | {f3_(s['value'])}（{s['rank']} / {s['n_models']}） | {s['best']} {f3_(s['best_value'])} | {near} | "
                 f"**{VERDICT_ZH[s['verdict']]}** | {bon} |")
    v = list(S["overall"]["verdicts"].values())
    L += ["", f"**整体：{'最好' if S['overall']['overall_best'] else '不是最好'}**（{v.count('best')} 项最好、"
          f"{v.count('tied_best')} 项并列最好、{v.count('behind')} 项落后）。"
          "差值的方向统一成'正 = transformer 更好'；越低越好的指标（校准、时间、发生率、转移矩阵）已经换过号。", "",
          "读这张表要知道的："]
    a = S["evaluations"]["auc2"]["standing"]["transformer"]
    if a["verdict"] != "best":
        w = min(a["vs_above"], key=lambda x: x["maxT_ci"][1])
        L.append(f"- **主评估（2 年 AUC）在边界上。** transformer 排第 {a['rank']}，比 {w['model']} 低 {-w['diff']:.4f}，"
                 f"max-T 区间上界 {w['maxT_ci'][1]:+.4f}，判并列；换成 Bonferroni 的百分位区间，上界 {w['bonf_ci'][1]:+.4f}，"
                 f"会判{VERDICT_ZH[a['bonferroni_verdict']]}。判定按事先定的 max-T。")
    L += ["- **只有一个 seed**：几乎所有模型都只训练了一次，区间里没有 seed 方差（transformer 重训一次，扰动 62 格的打分也会变）。",
          "- **val 不是第一次看**：TRFX 变体、特征消融、基因实验各在 val 上看过一次（都没有换掉主模型）；这张表的点估计也早就在打包盒里。",
          "- **'并列'多数是因为区间宽**：val 上 2 年 AUC 的区间不窄，转移矩阵、校准、时间更宽，排在前面的多个模型都分不开。"]
    sk = {m for r in S["evaluations"].values() for m in r["skipped"]}
    if sk:
        L.append("- 没有参加某一项的：" + "；".join(f"{m}（{', '.join(k for k, r in S['evaluations'].items() if m in r['skipped'])}）"
                                             for m in sorted(sk)) + "，原因见 select_model.json 的 skipped。")
    g = [(k, r["standing"].get("transformer-gru-fill")) for k, r in S["evaluations"].items()]
    if all(x for _, x in g):
        vs = [x["verdict"] for _, x in g]
        L.append("- 补充：在 test 段上挑出的改进版 transformer-gru-fill 用同一规则算，"
                 + (f"7 项都是{VERDICT_ZH[vs[0]]}" if len(set(vs)) == 1 else
                    "、".join(f"{EVAL_ZH[k].split('：')[0]}{VERDICT_ZH[x['verdict']]}" for k, x in g))
                 + f"（2 年 AUC {g[0][1]['value']:.3f}，第 {g[0][1]['rank']}）；它是从约 70 个变体里挑出来的，只作补充。")
    return L + [""]


def conclusions_md(R):
    """MODELS.md 的结论（2026-11-08 起）：只比能生成未来的模型（R = 表 1 的行）。数字全部现算。"""
    by = {r["name"]: r for r in R}
    tr = by["transformer"]
    ndz = [r for r in R if r["group"] != "zoo"]
    top = max(R, key=lambda r: r["p1"]["auc"])
    close = [r for r in R if r["p1"]["auc"] >= top["p1"]["auc"] - 0.01]
    rank = 1 + sum(r["p1"]["auc"] > tr["p1"]["auc"] for r in R)
    dz = [r for r in R if r["group"] == "zoo"]
    oe_g = [r["p1"]["oe"] for r in ndz if r["p1"]["oe"] is not None]
    T = _j(os.path.join(ZOO, pkg_dir("transformer"), "transitions.json")) or {}
    sc = T.get("snapshot_cont_baseline_macro") or {}
    tm = T.get("macro") or {}
    trs = traj_summary(tr["traj"]) if tr.get("traj") else None
    gaps = sorted([(traj_summary(r["traj"])["matrix_gap"], r["name"]) for r in R if r.get("traj") and r["traj"].get("matrix_5y")])
    pv = [(r["pert"]["v2"]["right"], r["name"]) for r in R if (r.get("pert") or {}).get("v2")]
    tv = (tr.get("pert") or {}).get("v2") or {}
    L = ["## 结论（2026-11-08；只比能生成未来的模型，选定 transformer）", "",
         f"- **预测：前面一大群分不开。** 4 终点 2 年 AUC 最高的是 {top['name']} {top['p1']['auc']:.3f}；"
         f"{len(close)} 个模型在它的 0.01 以内，transformer {tr['p1']['auc']:.3f} 排第 {rank}。"
         f"Delphi 低一截：{min(r['p1']['auc'] for r in dz):.3f}–{max(r['p1']['auc'] for r in dz):.3f}（最低的是只看人口学的 staticonly），而且风险不能当概率用"
         f"（平均 O/E {min(r['p1']['oe'] for r in dz):.1f}–{max(r['p1']['oe'] for r in dz):.1f}；其余模型 {min(oe_g):.2f}–{max(oe_g):.2f}，"
         f"transformer {tr['p1']['oe']:.2f}）。",
         "- **分期转换：transformer 最高，但和只读当前分数的逻辑回归差不多。** 6 题平均 AUC 2 / 5 / 10 年，transformer "
         + " / ".join(f"{tm[f'{H}y']['auc']:.3f}" for H in (2, 5, 10))
         + "；评估自带的参照'当前访视的 MMSE、认知连续值 + 年龄、性别的逻辑回归'是 "
         + " / ".join(f"{sc[f'{H}y']['auc']:.3f}" for H in (2, 5, 10))
         + f"。只在当前分期、5 岁年龄段都相同的站点之间比（5 年），transformer {tm['5y']['auc_within_state_age']:.3f}，"
           f"参照 {sc['5y']['auc_within_state_age']:.3f}。病史和变化能多给的信息不多。",
         f"- **10 年轨迹：前几个生成模型分不开。** 5 年 MMSE 转移矩阵平均每格差（百分点）最小的是 {gaps[0][1]} {100 * gaps[0][0]:.1f}，"
         f"transformer {100 * trs['matrix_gap']:.1f}（完美模拟器光抽样误差也有约 x.x）。transformer 的 10 年 AD 模拟 {100 * trs['ad_sim']:.0f}%、"
         f"实际 {100 * trs['ad_obs']:.0f}%，死亡 {100 * trs['death_sim']:.0f}% 对 {100 * trs['death_obs']:.0f}%。",
         f"- **扰动方向：高分端饱和。** 第三部分 62 格（5 年 log OR）：{sum(x >= 60 for x, _ in pv)} 个模型对 60 格以上，"
         f"最高 {max(pv)[0]}（{', '.join(n for x, n in pv if x == max(pv)[0])}），transformer {tv.get('right')} 对 {tv.get('wrong')} 错；"
         "同一个配置重训一次，对的格数也会变。",
         "- **校准：rollout 的风险偏低**（O/E > 1）。一部分是读法（rollout 数的是累计发生率，标签是条件的，还没定换不换），"
         "一部分是模型本身（transformer 的 AD 单步头在训练集上就低估），见 EVAL_PROTOCOL.md 第三部分修订 13、14。",
         "- 2026-11-05 以前写的结论（当时包括 36 个只能直接预测的模型）挪到了 "
         "[models_not_usable/README.md](models_not_usable/README.md) 的最后一节。", ""]
    return L


def table_md(R, G):
    R = sorted([r for r in R if r["p1"]], key=lambda r: -r["p1"]["auc"])
    nb, nz = sum(r["group"] != "zoo" for r in R), sum(r["group"] == "zoo" for r in R)
    L = ["# 全部模型：同一份数据、同一套题，互相比较", "",
         f"{nb} 个模型和 {nz} 个 Delphi 互相比较，最后选用 transformer（表里加粗）。"
         "多数模型照原文的做法实现，超参用原文的默认值或在 test 段上挑；transformer 另外在 test 段上比较过约 70 个变体，"
         f"挑出的组合是 [transformer-gru-fill]({pkg_dir('transformer-gru-fill')}/README.md)，只作补充（它的 val 数要按'挑过一次'来读），主模型是原版。"
         "所有模型用同一张真值表、同一份打分代码"
         "（`unified_eval/evaluate.py`、`evaluate_transitions.py`、`rollout_trans.aggregate`）。"
         "每个模型一个打包盒（只有结果：README + 图 + 聚合 JSON）：9 模型图里的在 [representative/](representative/README.md)，"
         "不可用的 36 个在 [models_not_usable/](models_not_usable/README.md)，"
         "其余（包括 Delphi）在 [models_self/](models_self/README.md)。代码在 [models_code/](models_code/)。", "",
         "- **输入**：除 Delphi（读自己的 token）外，所有模型共用同一份数据（`models_code/input_data.py` -> `input_data_trans.py` 产出的 `newmodel_data_trans.npz`）：每次访视 15 个连续量（不分箱）+ 缺失掩码 + 9 个状态 + "
         "卒中 / 抑郁 + AD 状态 + 时间；25 维静态背景；需要的模型再加 120 维手工历史特征（TADPOLE Frog / L2C）。",
         "- **划分**：split811。train 训练，test 调参、早停、挑版本，val（和所有模型同一批）只报一次。单个 seed（Delphi 的主数字"
         "也是单个 seed）。",
         "- **怎么预测未来**：'模拟' = 从站点起逐年生成轨迹，风险是轨迹里事件的比例；'精确' = 多状态 Markov 的转移矩阵连乘；"
         "'生存曲线' = 一条 S(t)，任意年数都能读；'逐题逐年' = 60 个输出各管一道题一个时间窗。后两类不能生成未来，"
         "H 年风险是直接学出来的，训练目标就是评估题本身：这 36 个 2026-11-05 起不参与评估（用户定），挪到 "
         "[models_not_usable/](models_not_usable/README.md)。下面的表和图都不含它们，能 rollout 的换成它们的 rollout 版（-sim、ar-*）。",
         "",
         ""] + selection_md() + conclusions_md(R) + ["![models](MODELS.png)", "",
         f"图只画 {len(SELECTED)} 个代表（{', '.join(SELECTED)}），而且一律用 rollout 的读法（每个站点往后模拟多条未来，多数 100 条，ETHOS 50 条，"
         "数事件当风险）：只能直接预测的画它的 rollout 版（-sim / ar-*），Delphi 画 delphi-dedup-sim"
         "（`unified_eval/rollout_landmarks.py`，不是从速率换算的直接读出）。下面的表是全部可用的模型"
         "（只能直接预测的那些换成它们的 rollout 版），它们的总览图是 [OVERVIEW_all.png](OVERVIEW_all.png)。", "",
         "同样这几个模型（rollout 版）的 2 年风险校准（每个模型 fig2 的横向对比）：", "", "![calibration](MODELS_calibration.png)", "",
         "## 表 1：预测（第一部分 2 年；第三部分 2 / 5 / 10 年）", "",
         "按第一部分的 4 终点平均 AUC 排序。粗体 = 选定的模型 transformer。CI = 按人 bootstrap 1000 次。"
         "第三部分：6 道'首次越线'题的平均 AUC；'同状态 + 年龄段' = 只在当前 MMSE 分期、认知分期、5 岁年龄段都相同的站点之间比（5 年）；"
         "'竞争风险' = 窗口内先死亡的站点算阴性（10 年）。", "",
         "| # | 模型 | 做法 | 未来怎么来 | 4 终点 2 年 AUC [95% CI] | AD | 死亡 | MMSE<24 | 认知<−1 | 平均 O/E | "
         "6 题 AUC 2 / 5 / 10 年 | 同状态 + 年龄段（5 年） | 竞争风险（10 年） |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(R, 1):
        p1, p3 = r["p1"], r["p3"]
        nm = name_md(r)
        cite = "" if r["group"] == "zoo" else r["cite"]
        t3 = " / ".join(f3(p3["auc"][H]) for H in (2, 5, 10)) if p3 else "—"
        if p3 and p3["all"] is False:
            t3 += "（只答了 4 题）"
        oe = "—" if p1["oe"] is None else f"{p1['oe']:.2f}"
        L.append(f"| {i} | {nm} | {cite} | {FUT_ZH[r['future']]} | {f3(p1['auc'])} [{f3(p1['ci'][0])}, {f3(p1['ci'][1])}] | "
                 + " | ".join(f3(p1["ep"][e]) for e in ENDS)
                 + f" | {oe} | {t3} | {f3(p3['within_age5']) if p3 else '—'} | {f3(p3['cr10']) if p3 else '—'} |")
    L += ["", "O/E = 实际发生数 / 预测风险之和（1 = 总量上校准，> 1 = 低估）。固定权重的评分（BDSI、ANU-ADRI、CAIDE）只在 train 上"
          "做了一次单变量换算（截距 + 非负斜率），排序就是原评分的排序。", ""]
    G = [g for g in G if g["traj"] and g["traj"].get("matrix_5y")]
    for g in G:
        g["s"] = traj_summary(g["traj"])
    G.sort(key=lambda g: g["s"]["matrix_gap"])
    L += ["## 表 2：10 年轨迹（只有能生成未来的模型）", "",
          "每人从第一次访视出发模拟 100 条 10 年的未来（TabPFN 30 条），和实际的后续比（EVAL_PROTOCOL.md 第 12 节，同一个汇总函数）。"
          "按 5 年 MMSE 转移矩阵的平均每格差排序（三行等权，越小越像）；完美的模拟器光是抽样误差也会差约 x.x 个百分点。"
          "'站点 AUC：模拟 / 直接'= 第一次访视这个站点上，6 题 5 年平均 AUC：从这些轨迹数出来的 vs 这个模型的打分表"
          "（除 Delphi 外，生成模型的打分表本身也是模拟出来的，两者只差随机数）。", "",
          "![trajectories](MODELS_traj.png)", "",
          "图只画选中的代表；只能直接预测的，画它的 rollout 版。下面的表是全部能生成未来的模型。", "",
          "| 模型 | 5 年转移矩阵平均每格差（百分点） | 10 年 AD 实际 / 模拟 | 10 年死亡 实际 / 模拟 | 第一次访视 5 年 AUC：模拟 / 直接 |",
          "|---|---|---|---|---|"]
    for g in G:
        s = g["s"]
        mac = (g["traj"].get("auc_first_visit_macro") or {}).get("5") or {}
        nm = name_md(g)
        L.append(f"| {nm} | {100 * s['matrix_gap']:.0f} | {pc(s['ad_obs'])} / {pc(s['ad_sim'])} | "
                 f"{pc(s['death_obs'])} / {pc(s['death_sim'])} | {f3(mac.get('rollout'))} / {f3(mac.get('direct'))} |")
    L += [""]
    P = [r for r in R if r.get("pert")]
    if P:
        P.sort(key=lambda r: (-(r["pert"].get("v2") or {}).get("right", -1), (r["pert"].get("v2") or {}).get("wrong", 99)))
        L += ["## 表 3：扰动（第二部分表 2b、第三部分表 6 的同一套改动和打分代码）", "",
              "除 Delphi 外的模型共用同一份输入，所以扰动在这份输入的特征空间里改（`perturb_features.apply`、`perturb_v2.translate_features`；Delphi 换 / 插 token），"
              "同一批人、同一组改动。能生成未来的模型，'直接读出'就是从改过的站点往后模拟（每条记录 100 条轨迹）。"
              "第二部分：20 个预设方向的格子（2 年风险），对 / 无响应 / 错；剂量反应单调的比例；APOE ε4 0→1 对 AD 的 2 年 log OR。"
              "第三部分：62 个预设格子按 5 年 log OR 打分（不依赖校准）；AD 和每条认知线风险一起升高的扰动数；"
              "MMSE + 认知一起下降（cog2）、APOE、三种共病之后，10 年内平均无事件时间（AD、MMSE<24、死亡）的变化（年，负 = 缩短；按全部人平均，多出来的病例占大头）。", "",
              "| 模型 | 第二部分：对 / 无响应 / 错（/ 20） | 剂量单调 | APOE→AD log OR | 第三部分 log OR：对 / 无响应 / 错（/ 62） | "
              "AD 和认知风险一起升高 | cog2：AD / MMSE<24（年） | APOE：AD（年） | 三种共病：死亡（年） |",
              "|---|---|---|---|---|---|---|---|---|"]
        y = lambda x: "—" if x is None else f"{x:+.2f}"           # noqa: E731
        for r in P:
            a, b = r["pert"].get("v1"), r["pert"].get("v2")
            nm = name_md(r)
            c1 = f"{nf(a['right'])} / {nf(a['no'])} / {nf(a['wrong'])}" if a else "—"
            c2 = "—" if not a or a["dose"] is None else f"{a['dose']:.2f}"
            c4 = f"{b['right']} / {b['no']} / {b['wrong']}" if b else "—"
            c5 = f"{b['sync']} / {b['n_def']}" if b else "—"
            c6 = f"{y(b['cog2_ad'])} / {y(b['cog2_mmse'])}" if b else "—"
            L.append(f"| {nm} | {c1} | {c2} | {y(a['apoe_ad']) if a else '—'} | {c4} | {c5} | {c6} | "
                     f"{y(b['apoe_ad']) if b else '—'} | {y(b['comorb3_death']) if b else '—'} |")
        L += [""]
    return L


def sim_pairs(R):
    by = {r["name"]: r for r in R}
    P = [(by[o], by[s]) for o, s in MZ.SIM_OF.items() if o in by and s in by and by[o]["p1"] and by[s]["p1"]]
    return sorted(P, key=lambda p: -p[0]["p1"]["auc"])


def fig_sim(R, path, refs):
    """每个不能 rollout 的原模型一行：A 2 年 AUC，直接预测（橙）和 rollout 版（蓝）；B rollout 版的 5 年转移矩阵差。
    refs：9 模型图里本来就能生成未来的参照（transformer、delphi-dedup-sim），两个面板上各画一条竖线，数字和 OVERVIEW.png 同源。"""
    P = sim_pairs(R)
    if not P:
        return
    n = len(P)
    ref_style = {"transformer": (FG.INK, "-."), DELPHI_SIM: (GROUP_COL["zoo"], "--")}
    ref_lines = [(r["name"], *ref_style[r["name"]], r["p1"]["auc"],
                  100 * traj_summary(r["traj"])["matrix_gap"] if r.get("traj") else None) for r in refs]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.3 * n + 1.9), sharey=True)
    y = np.arange(n)
    ax = axes[0]
    for i, (o, s) in enumerate(P):
        ax.barh(i - 0.18, o["p1"]["auc"] - 0.5, left=0.5, height=0.36, color=GROUP_COL["direct"], zorder=2)
        ax.barh(i + 0.18, s["p1"]["auc"] - 0.5, left=0.5, height=0.36, color=GROUP_COL["gen"], zorder=2)
    ax.set_xlim(auc_left([x["p1"]["auc"] for pr in P for x in pr] + [a for _, _, _, a, _ in ref_lines]), 0.95)
    ax.axvline(0.5, color=FG.INK, lw=1, zorder=3)
    for _, col, ls, a, _ in ref_lines:
        ax.axvline(a, color=col, lw=1.5, ls=ls, zorder=3)
    ax.set_title("A  Four endpoints, 2-year AUC")
    ax.set_xlabel("Mean AUC (bars start at 0.5 = chance)")
    ax = axes[1]
    gaps = [(i, s["traj"]) for i, (_, s) in enumerate(P) if s.get("traj") and s["traj"].get("matrix_5y")]
    if gaps:
        floor = matrix_gap_floor(gaps[0][1]["matrix_5y"]["mmse"])
        ax.axvspan(0, 100 * floor, color=FG.FAINT, alpha=0.6, lw=0, zorder=0)
        for _, col, ls, _, g in ref_lines:
            if g is not None:
                ax.axvline(g, color=col, lw=1.5, ls=ls, zorder=3)
        for i, tr in gaps:
            ax.barh(i, 100 * traj_summary(tr)["matrix_gap"], height=0.7, color=GROUP_COL["gen"], zorder=2)
    ax.set_title("B  Rollout version: 5-year MMSE transition matrix error")
    ax.set_xlabel("Mean |simulated − observed|, points per cell\n(lower = more realistic; grey = noise floor)")
    ax.set_xlim(left=0)
    ax.set_ylim(n + 0.8, -0.6)                                  # 从上到下按直接预测的 AUC 降序
    for a in axes:
        a.grid(axis="x", color=FG.FAINT, lw=0.8)
        a.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([o["name"] for o, _ in P])
    h = [Patch(facecolor=GROUP_COL["direct"], label="Direct prediction (original)"),
         Patch(facecolor=GROUP_COL["gen"], label="Its rollout version")]
    h += [Line2D([], [], color=col, ls=ls, lw=1.5, label=f"{name} (reference)") for name, col, ls, _, _ in ref_lines]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, -0.01))
    d = np.median([s["p1"]["auc"] - o["p1"]["auc"] for o, s in P])
    fig.suptitle(f"Rollout versions of the {n} direct models: median change in 2-year AUC {d:+.3f}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    FG._save(fig, path)


def sim_findings(P):
    """表 4 上面的几条结论：数量相关的现算（TabPFN 之类跑完了重新生成就会更新）。"""
    import statistics
    ref = _j(os.path.join(ZOO, pkg_dir("delphi-nodedup"), "trajectory.json"))
    g_ref = 100 * traj_summary(ref)["matrix_gap"] if ref else None
    gaps = sorted((100 * traj_summary(s["traj"])["matrix_gap"], s["name"]) for _, s in P if s.get("traj") and s["traj"].get("matrix_5y"))
    better = [n for g, n in gaps if g_ref is not None and g < g_ref]
    dauc = statistics.median(s["p1"]["auc"] - o["p1"]["auc"] for o, s in P)
    mr = _j(os.path.join(ZOO, pkg_dir("minimalrnn"), "trajectory.json"))
    sb = traj_summary(_j(os.path.join(ZOO, pkg_dir("survivalboost-l2c-sim"), "trajectory.json")))
    sbm = _j(os.path.join(ZOO, pkg_dir("survivalboost-l2c-sim"), "metrics.json"))
    g_mr = 100 * traj_summary(mr)["matrix_gap"] if mr else None
    return [f"- **改成 rollout，预测只掉一点**：{len(P)} 对里 2 年 AUC 的中位数变化 {dauc:+.3f}。可能的原因（没有分开验证）：误差一年一年累积、"
            "风险变成了抽样比例；直接预测本来就是为\"H 年内\"这道题训练的。",
            f"- **轨迹普遍不错**：{len(better)} 个 rollout 版的 5 年转移矩阵比最好的 Delphi（nodedup，{g_ref:.1f} 个点）更像真实："
            + "、".join(f"{n}（{g:.1f}）" for g, n in gaps if n in better)
            + f"。整体最好的仍是 MinimalRNN（{g_mr:.1f}；完美的模拟器光抽样误差也有约 x.x）。结构上，"
            "深度生存网络的 rollout 版最好，线性 / Cox 一类居中，最差的是 "
            + "、".join(f"{n}（{g:.1f}）" for g, n in gaps[::-1][:3]) + "。",
            "- **临床评分的 rollout 版 AUC 反而大涨**（BDSI、CAIDE）：随访指标是线性模型按当前访视"
            "（含 MMSE、认知）学出来的，MMSE / 认知那几道题的信息来自它，评分本身只管死亡和 AD 的风险。所以它已经不是那张评分表了，"
            "不能拿来说明评分好。",
            "- **TabPFN 的 rollout 版掉得最多**（2 年 AUC 下降，10 年死亡模拟低于实际），至少一部分是被迫压低设置造成的："
            "每一步要过 54 个 TabPFN，只能每个站点模拟 10 条（风险只有 11 种取值，大量并列，会直接压低 AUC），随访指标也只用 1 个集成成员；"
            "这一行不能当成 TabPFN 做 rollout 的上限。",
            f"- **SurvivalBoost 的 rollout 版不合格**：它的 1 年死亡风险偏低很多，10 年死亡模拟 {pc(sb['death_sim'])}"
            f"（实际 {pc(sb['death_obs'])}）、2 年死亡 O/E {sbm['endpoints']['death_2y']['o_e']:.1f}；"
            "原模型的 2 年死亡 O/E 本来就大于 1，滚 10 步以后放大了。它在同一份代码、同一组随机种子下重跑，结果也会变"
            "（2026-10-29 连跑两次，2 年 AUC 和之前那次都不完全一样），很小的差别对它没有意义。",
            "- **扰动**：深度生存网络的 rollout 版比原版略差（62 格里多错几格；L2C-FNN 几乎不变）；"
            "Cox 的当前访视版和 supermodel 反而更好（错的格子变少，差别接近重训噪声），加历史的 Cox 不变；"
            "评分的 rollout 版因为有了随访指标，开始对认知改动有反应。", ""]


def table_sim_md(R):
    P = sim_pairs(R)
    if not P:
        return []
    L = ["## 表 4：不能 rollout 的模型，各自的 rollout 版（2026-10-21 加）", "",
         "做法见 `models_code/rollout_versions.py` 开头：不让模型直接回答\"H 年内会不会发生\"，改学\"明年会怎样\"——这一年里死亡、"
         "得 AD 的概率（和原模型同一类、同一组超参）+ 明年那次访视的分布（同族的学习器）——再一年一年往后滚。"
         "评估和其它模型完全同一套：打分表也是从每个站点模拟出来的；扰动从改过的站点往后模拟。lr-l2c、lgbm-l2c 的 rollout 版就是 ar-lr、ar-lgbm。"
         "看整条病历的序列网络（gru、lstm、transformer、grud、tlstm、retain、dynamic-deephit）2026-10-28 起本身就是自回归的，"
         "不在这张表里（见表 1、表 2）。", "",
         ] + sim_findings(P) + ["![rollout versions](MODELS_sim.png)", "",
         "图只画选中的代表里只能直接预测的那几个；表是全部。两条竖线是 9 模型图里本来就能生成未来的两个参照"
         "（transformer、delphi-dedup-sim，和 OVERVIEW.png 同一组数）。", "",
         "| 原模型 | rollout 版 | 2 年 AUC 直接 → rollout | 5 年分期 AUC 直接 → rollout | O/E 直接 → rollout | 5 年矩阵差（点） | "
         "10 年 AD 模拟 / 实际 | 10 年死亡 模拟 / 实际 | 扰动 对/无/错（/62）直接 → rollout | cog2：10 年内无 AD 时间缩短（年）直接 → rollout |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    y = lambda x: "—" if x is None else f"{-x:.1f}"            # noqa: E731
    for o, s in P:
        a5 = lambda r: f3((r.get("p3") or {}).get("auc", {}).get(5)) if r.get("p3") else "—"      # noqa: E731
        oe = lambda r: "—" if r["p1"]["oe"] is None else f"{r['p1']['oe']:.2f}"                    # noqa: E731
        pv = lambda r: "—" if not (r.get("pert") or {}).get("v2") else "{right} / {no} / {wrong}".format(**r["pert"]["v2"])  # noqa: E731
        cg = lambda r: "—" if not (r.get("pert") or {}).get("v2") else y(r["pert"]["v2"]["cog2_ad"])   # noqa: E731
        st = traj_summary(s["traj"]) if s.get("traj") else {}
        gap = "—" if not st.get("matrix_gap") else f"{100 * st['matrix_gap']:.0f}"
        L.append(f"| [{o['name']}]({o['link']}) | [{s['name']}]({s['link']}) | {f3(o['p1']['auc'])} → {f3(s['p1']['auc'])} | "
                 f"{a5(o)} → {a5(s)} | {oe(o)} → {oe(s)} | {gap} | {pc(st.get('ad_sim'))} / {pc(st.get('ad_obs'))} | "
                 f"{pc(st.get('death_sim'))} / {pc(st.get('death_obs'))} | {pv(o)} → {pv(s)} | {cg(o)} → {cg(s)} |")
    return L + [""]


# ----------------------------------------------------------------------------- 表 6：轨迹的数值准不准（traj_rmse.py，2026-10-22 加）

TRAJ_ROWS = [("locf", "最近一次的值原样往后（参照）"), ("lme", "lmm-joint 的线性混合模型"), ("jm", "jm 的双变量线性混合模型（联合估计）"),
             ("gpboost", "gpboost-joint 的 GPBoost"), ("merf", "merf-joint 的 MERF"), ("gp", "gp-joint 的分层高斯过程")]


def table_traj_md():
    J = _j(os.path.join(BZ, "_traj", "rmse.json"))
    if not J:
        return []
    H = [str(h) for h in J["offsets_years"]]
    M = J["models"]
    n = {f: " / ".join(str(M["locf"][f][h]["n"]) for h in H) for f in ("mmse", "cogn")}
    L = ["## 表 6：轨迹本身准不准（2026-10-22 加）", "",
         "第一、三部分只看越线和事件。这里直接比数值：站在 val 的每个站点上，只用到站点为止的测量，预测 " + " / ".join(H)
         + " 年后那次访视的 MMSE 和全局认知（那次真的测了才算），报 RMSE（括号里是 MAE）。和打包盒里同一个实现、同一份训练（`traj_rmse.py`）。"
         f"格子数：MMSE {n['mmse']}，认知 {n['cogn']}。", "",
         "| 轨迹模型 | MMSE RMSE " + " / ".join(f"{h} 年" for h in H) + " | 认知 RMSE " + " / ".join(f"{h} 年" for h in H) + " |",
         "|---|---|---|"]
    for k, zh in TRAJ_ROWS:
        if k not in M:
            continue
        r = M[k]
        c1 = " / ".join(f"{r['mmse'][h]['rmse']:.2f}（{r['mmse'][h]['mae']:.2f}）" for h in H)
        c2 = " / ".join(f"{r['cogn'][h]['rmse']:.3f}（{r['cogn'][h]['mae']:.3f}）" for h in H)
        L.append(f"| {zh} | {c1} | {c2} |")
    from report_text import TRAJ_NOTES
    return L + [""] + TRAJ_NOTES


# ----------------------------------------------------------------------------- 表 5：NACC 预训练（2026-10-22 加）

NACC_ARCH = [("transformer", "因果 transformer（基线 transformer 的结构）"),
             ("grud", "GRU-D（基线 grud 的结构）"), ("tlstm", "T-LSTM（基线 tlstm 的结构）")]
NACC_INIT = [("", "只在 ROSMAP 上训练（对照）"), ("-nacc", "NACC 预训练 → ROSMAP 微调"), ("-nacc-zs", "NACC 预训练，不微调")]
NACC_COL = {"": "#8c8c8c", "-nacc": GROUP_COL["gen"], "-nacc-zs": GROUP_COL["direct"]}


def nacc_name(arch, sfx):
    return f"{arch}-x5" if sfx == "" else f"{arch}{sfx}"


def nacc_pairs():
    J = _j(os.path.join(BZ, "_nacc", "compare.json")) or {}
    return J, {(p["a"], p["b"]): p for p in J.get("pairs", [])}


def fig_nacc(R, path):
    """每个结构三种起点（第一部分 2 年 4 终点平均 AUC）。"""
    by = {r["name"]: r for r in R}
    rows_ = [a for a, _ in NACC_ARCH if any(nacc_name(a, s) in by for s, _ in NACC_INIT)]
    if not rows_:
        return
    n = len(rows_)
    fig, ax = plt.subplots(figsize=(7.5, max(0.8 * n + 2.0, 4.2)))
    for i, a in enumerate(rows_):
        for j, (sfx, _) in enumerate(NACC_INIT):
            r = by.get(nacc_name(a, sfx))
            if not r or not r["p1"]:
                continue
            v, ci = r["p1"]["auc"], r["p1"]["ci"]
            y = i + (j - 1) * 0.26
            ax.barh(y, v - 0.5, left=0.5, height=0.25, color=NACC_COL[sfx], zorder=2)
            ax.plot(ci, [y, y], color=FG.INK, lw=0.9, zorder=3)
            ax.text(ci[1] + 0.004, y, f"{v:.3f}", va="center", fontsize=8.5, color=FG.INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([dict(NACC_ARCH_EN)[a] for a in rows_])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0.5, 0.97)
    ax.set_xlabel("Four endpoints, mean 2-year AUC\n(bars start at 0.5 = chance; black line = 95% CI)")
    ax.grid(axis="x", color=FG.FAINT, lw=0.8)
    ax.set_axisbelow(True)
    h = [Patch(facecolor=NACC_COL[s_], label=lab) for s_, lab in NACC_INIT_EN]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("NACC pretraining vs training on ROSMAP only", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    FG._save(fig, path)


NACC_ARCH_EN = [("transformer", "Causal transformer\n('transformer')"),
                ("grud", "GRU-D"), ("tlstm", "T-LSTM (time-aware LSTM)")]
NACC_INIT_EN = [("", "ROSMAP only (control)"), ("-nacc", "NACC pretrained, fine-tuned on ROSMAP"),
                ("-nacc-zs", "NACC pretrained, no fine-tuning")]


def table_nacc_md(R):
    by = {r["name"]: r for r in R}
    if not any(n in by for n in NACC):
        return []
    J, P = nacc_pairs()
    ds = _j(os.path.join(BZ, "_nacc", "data_summary.json")) or {}
    pre = _j(os.path.join(BZ, "_nacc", "pretrain.json")) or {}
    from report_text import NACC_TEXT
    L = ["## 表 5：先在 NACC 上预训练，再到 ROSMAP 上微调（2026-10-22 加）", ""] + NACC_TEXT(ds, J, P, by) + [
         "![nacc](MODELS_nacc.png)", "",
         "同一结构、同一份训练代码、同一组 seed（5 个，风险取平均），只差初始化。'配对差' = 减去同一结构只在 ROSMAP 上训练的那一行，"
         "val 的人按人重抽 1000 次（两边同一批人）。'去掉 LATC' = 去掉 Rush 拉丁裔核心队列的人再算一遍配对差（这批人可能也在 NACC 里）。"
         "'NACC test 损失' = 预训练时 NACC 留出的 10% 人上的风险 BCE（5 个 seed 的平均）。", "",
         "| 结构 | 起点 | 4 终点 2 年 AUC [95% CI] | 配对差（2 年） | 6 题 5 年 AUC | 配对差（5 年） | 去掉 LATC 的配对差（2 年 / 5 年） | "
         "平均 O/E | 扰动 v2 对 / 无 / 错（/ 62） | NACC test 损失 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    dd = lambda x: "—" if x is None else f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"   # noqa: E731
    for a, zh in NACC_ARCH:
        for sfx, izh in NACC_INIT:
            nm = nacc_name(a, sfx)
            r = by.get(nm)
            if not r:
                continue
            p1, p3 = r["p1"], r["p3"]
            pr = P.get((nm, nacc_name(a, ""))) if sfx else None
            c1 = dd(pr["all"]["part1_2y_4endpoints"]) if pr else ("（对照）" if not sfx else "—")
            c3 = dd(pr["all"]["part3_5y_6questions"]) if pr else ("（对照）" if not sfx else "—")
            cl = (f"{pr['no_latc']['part1_2y_4endpoints']['diff']:+.3f} / {pr['no_latc']['part3_5y_6questions']['diff']:+.3f}"
                  if pr and pr.get("no_latc") else "—")
            v2 = (r.get("pert") or {}).get("v2")
            pv = f"{v2['right']} / {v2['no']} / {v2['wrong']}" if v2 else "—"
            tag = f"seq-{a}"
            losses = [v.get("best_test_loss") for k, v in pre.items() if k.startswith(tag + "_s")]
            pl = f"{np.mean(losses):.4f}" if (sfx and losses) else ""
            oe = "—" if p1["oe"] is None else f"{p1['oe']:.2f}"
            L.append(f"| {zh if sfx == '' else ''} | [{izh}]({r['link']}) | {f3(p1['auc'])} [{f3(p1['ci'][0])}, {f3(p1['ci'][1])}] | {c1} | "
                     f"{f3(p3['auc'][5]) if p3 else '—'} | {c3} | {cl} | {oe} | {pv} | {pl} |")
    L += [""]
    L += ["扰动（和表 3 同一套改动和打分）：预训练改没改变模型对改动的反应。APOE→AD = ε4 0→1 对 2 年 AD 的 log OR（第二部分）；"
          "后三列 = 第三部分里 10 年内平均无事件时间的变化（年，负 = 缩短）。", "",
          "| 结构 | 起点 | APOE→AD 2 年 log OR | APOE：无 AD 时间（年） | MMSE + 认知一起下降：无 AD 时间（年） | 三种共病：无死亡时间（年） | 第三部分 对 / 无 / 错 |",
          "|---|---|---|---|---|---|---|"]
    yy = lambda x: "—" if x is None else f"{x:+.2f}"                 # noqa: E731
    for a, zh in NACC_ARCH:
        for sfx, izh in NACC_INIT:
            r = by.get(nacc_name(a, sfx))
            if not r or not r.get("pert"):
                continue
            v1, v2 = r["pert"].get("v1") or {}, r["pert"].get("v2") or {}
            L.append(f"| {zh if sfx == '' else ''} | {izh} | {yy(v1.get('apoe_ad'))} | {yy(v2.get('apoe_ad'))} | {yy(v2.get('cog2_ad'))} | "
                     f"{yy(v2.get('comorb3_death'))} | {v2.get('right', '—')} / {v2.get('no', '—')} / {v2.get('wrong', '—')} |")
    L += [""]
    gens = [("transformer-sim", "transformer-sim-nacc"), ("ethos", "ethos-nacc")]
    G = [(by.get(o), by.get(n)) for o, n in gens if by.get(n)]
    if G:
        L += ["能生成未来的两个（单个 seed，和原基线一样）：", "",
              "| 模型 | 2 年 AUC：只在 ROSMAP 上 → NACC 预训练 | 5 年 6 题 AUC | 5 年 MMSE 转移矩阵每格差（点） | 10 年 AD 模拟（实际） | "
              "10 年死亡 模拟（实际） | 扰动 v2 对 / 无 / 错 |", "|---|---|---|---|---|---|---|"]
        for o, n in G:
            so = traj_summary(o["traj"]) if o and o.get("traj") else {}
            sn = traj_summary(n["traj"]) if n.get("traj") else {}
            gp = lambda st: "—" if not st.get("matrix_gap") else f"{100 * st['matrix_gap']:.0f}"     # noqa: E731
            pv = lambda r: "—" if not (r and (r.get("pert") or {}).get("v2")) else "{right} / {no} / {wrong}".format(**r["pert"]["v2"])   # noqa: E731
            a5 = lambda r: f3(r["p3"]["auc"][5]) if r and r.get("p3") else "—"              # noqa: E731
            L.append(f"| [{n['name']}]({n['link']}) | {f3(o['p1']['auc']) if o else '—'} → {f3(n['p1']['auc'])} | {a5(o)} → {a5(n)} | "
                     f"{gp(so)} → {gp(sn)} | {pc(so.get('ad_sim'))} → {pc(sn.get('ad_sim'))}（{pc(sn.get('ad_obs'))}） | "
                     f"{pc(so.get('death_sim'))} → {pc(sn.get('death_sim'))}（{pc(sn.get('death_obs'))}） | {pv(o)} → {pv(n)} |")
        L += [""]
    return L + table_nacc_delphi_md(P)


DELPHI_NACC = [("delphi-dedup-nojitter", "只在 ROSMAP 上训练（对照）", os.path.join(ZOO, pkg_dir("delphi-dedup-nojitter"))),
               ("delphi-dedup-nojitter-nacc", "NACC 预训练 → ROSMAP 微调", os.path.join(BZ, "_nacc", "delphi", "delphi-dedup-nojitter-nacc")),
               ("delphi-dedup-nojitter-nacc-zs", "只在 NACC 上训练，不微调", os.path.join(BZ, "_nacc", "delphi", "delphi-dedup-nojitter-nacc-zs"))]
DELPHI_NACC_REF = {"delphi-dedup-nojitter": "transformer-x5", "delphi-dedup-nojitter-nacc": "transformer-nacc",
                   "delphi-dedup-nojitter-nacc-zs": "transformer-nacc-zs"}
NACC_TEST_ROWS = [("grud", "GRU-D"), ("transformer", "因果 transformer"), ("tlstm", "T-LSTM"), ("delphi-nojitter", "Delphi（nojitter 配方）")]


def table_nacc_delphi_md(P):
    """Delphi 的 NACC 预训练（nacc_delphi.py）：ROSMAP val 上三种起点 + NACC 自己的 test 段上和别的结构比。"""
    T = _j(os.path.join(BZ, "_nacc", "delphi_nacc_test.json"))
    A = _j(os.path.join(BZ, "_nacc", "delphi_apoe.json")) or {}
    if not T or not any(P.get((n, "delphi-dedup-nojitter")) for n, *_ in DELPHI_NACC[1:]):
        return []
    from report_text import NACC_DELPHI_VERDICT
    dd = lambda x: f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"     # noqa: E731
    tag = {"delphi-dedup-nojitter": "ctrl", "delphi-dedup-nojitter-nacc": "ft", "delphi-dedup-nojitter-nacc-zs": "zs"}
    L = ["### Delphi 也做一遍（2026-10-26 加，`models_code/nacc_delphi.py`）", "",
         "NACC 按交付版 dedup 的规则分词（同一个词表、切点、365 天网格；NACC 没有的家族 —— 子研究、饮酒、跛行、化验、睡眠、炎症、病理 —— 不发射），"
         "人和划分与上面的 NACC 数据逐人相同。配方是 delphi-dedup-nojitter（6 层 × 96 宽，5 个 seed）：NACC 上 40,000 步；微调 = 同一个配方"
         "从 NACC 的权重开始（优化器重新开始），和对照只差初始化；'只在 NACC 上训练'读 ROSMAP 时把 NACC 没有的 token 拿掉。"
         "AUC 用 5 个 seed 的风险平均（和上表一样）；配对差同上。O/E = 主 seed（s42）4 个终点 2 年的平均；"
         "APOE = ε4 0→1 对 2 年 AD 的 log OR（第 4 次访视，和 fig4 同一个读出，5 个 seed 的平均）。", "",
         "| 起点 | 4 终点 2 年 AUC | 配对差（2 年） | 6 题 5 年 AUC | 配对差（5 年） | 对同样起点的因果 transformer（2 年） | 平均 O/E | APOE→AD 2 年 log OR |",
         "|---|---|---|---|---|---|---|---|"]
    for n, zh, d in DELPHI_NACC:
        m = _j(os.path.join(d, "metrics.json")) or {}
        e = m.get("endpoints", {})
        oe = [e[f"{k}_2y"]["o_e"] for k in ("ad", "death", "mmse", "cogn") if f"{k}_2y" in e]
        pr = P.get((n, "delphi-dedup-nojitter"))
        rf = P.get((n, DELPHI_NACC_REF[n]))
        base = pr or rf
        a1 = base["all"]["part1_2y_4endpoints"]["a"]
        a3 = base["all"]["part3_5y_6questions"]["a"]
        ap = [A[f"{tag[n]}-s{s}"]["apoe"]["visit4"]["d_logor_2y"] for s in (42, 43, 44, 45, 46) if f"{tag[n]}-s{s}" in A]
        L.append(f"| {zh} | {a1:.3f} | {dd(pr['all']['part1_2y_4endpoints']) if pr else '（对照）'} | {a3:.3f} | "
                 f"{dd(pr['all']['part3_5y_6questions']) if pr else '（对照）'} | "
                 f"{rf['all']['part1_2y_4endpoints']['b']:.3f}，Delphi 差 {dd(rf['all']['part1_2y_4endpoints'])} | "
                 f"{np.mean(oe):.2f} | {np.mean(ap):+.2f}（{min(ap):+.2f} 到 {max(ap):+.2f}） |" if rf else "")
    L += ["", f"NACC 自己的 test 段（{T['n_test_persons']:,} 人、{T['n_landmarks']:,} 个站点；每个模型都只在 NACC 的 N 人上训练，"
          "这一段也是它们预训练时早停用的）：不涉及迁移、训练量是 ROSMAP 的 N 倍。标签和上面 NACC 数据的第三部分标签是同一份；"
          "'4 终点' = AD、死亡、MMSE<24、认知<−1 的 2 年。", "",
          "| 模型（5 seed 集成） | 4 终点 2 年 AUC | AD | 死亡 | MMSE<24 | 认知<−1 | 6 题 5 年 AUC |", "|---|---|---|---|---|---|---|"]
    for k, zh in NACC_TEST_ROWS:
        r = T["models"].get(k)
        if not r:
            continue
        c = r["ensemble"]["cells"]
        L.append(f"| {zh} | {r['ensemble']['part1_2y_4endpoints']:.3f} | {c['ad_2y']:.3f} | {c['death_2y']:.3f} | {c['mmse_lt24_2y']:.3f} | "
                 f"{c['cogn_ltm1_2y']:.3f} | {r['ensemble']['part3_5y_6questions']:.3f} |")
    return L + ["", "**结论**", ""] + NACC_DELPHI_VERDICT + [""] + table_upstream_md(P, T)


UP_COL = {"up": "#eb6834", "zoo": "#2a78d6", "tr": "#1baf7a"}   # dataviz 的类别色 1–3（validate_palette.js 过了）


def fig_upstream(P, T, path):
    """上游原版 Delphi vs zoo 改过的 Delphi vs 因果 transformer：ROSMAP val、NACC test，各 4 个终点 2 年 AUC 和平均。"""
    U = _j(os.path.join(BZ, "_nacc", "delphi_upstream.json"))
    pr = {b: P.get(("delphi-upstream", b)) for b in ("delphi-dedup-nojitter", "transformer-x5")}
    if not U or not all(pr.values()):
        return
    eps = [("ad", "AD"), ("death", "death"), ("mmse", "MMSE<24"), ("cogn", "cognition<−1")]
    nk = {"ad": "ad_2y", "death": "death_2y", "mmse": "mmse_lt24_2y", "cogn": "cogn_ltm1_2y"}
    pq = lambda b, s: [pr[b]["all"]["per_question"][f"p1_{e}"][s] for e, _ in eps]   # noqa: E731
    cl = lambda m: [m["cells"][nk[e]] for e, _ in eps]                              # noqa: E731
    M = T["models"]
    others = ("grud", "tlstm")
    oth = np.array([cl(M[k]["ensemble"]) for k in others if k in M])
    panels = [("ROSMAP val (trained on ROSMAP)", {"up": pq("transformer-x5", "a"), "zoo": pq("delphi-dedup-nojitter", "b"),
                                                  "tr": pq("transformer-x5", "b")}, None),
              ("NACC test (trained on NACC)", {"up": cl(U["nacc"]["ensemble"]), "zoo": cl(M["delphi-nojitter"]["ensemble"]),
                                               "tr": cl(M["transformer"]["ensemble"])}, oth)]
    lab = {"up": "Delphi, upstream code + config", "zoo": "Delphi, zoo version", "tr": "causal transformer"}
    rows_ = ["mean of 4"] + [z for _, z in eps]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, S, band) in zip(axes, panels):
        off = {"up": -0.3, "zoo": -0.1, "tr": 0.1, "oth": 0.3}
        for k, v in S.items():
            y = np.arange(len(rows_)) + off[k]
            ax.barh(y, np.r_[np.mean(v), v] - 0.5, left=0.5, height=0.19, color=UP_COL[k], zorder=2, label=lab[k])
        if band is not None:                       # 其他 5 种结构：灰柱 = 平均，黑线 = 最低到最高
            mid = np.r_[band.mean(1).mean(), band.mean(0)]
            lo, hi = np.r_[band.mean(1).min(), band.min(0)], np.r_[band.mean(1).max(), band.max(0)]
            y = np.arange(len(rows_)) + off["oth"]
            ax.barh(y, mid - 0.5, left=0.5, height=0.19, color="#bdbdbd", zorder=2, label="5 other structures (mean; line = range)")
            for i in range(len(rows_)):
                ax.plot([lo[i], hi[i]], [y[i], y[i]], color=FG.INK, lw=1, zorder=3)
        ax.set_xlim(0.5, 0.97)
        ax.set_title(title)
        ax.set_xlabel("2-year AUC (bars start at 0.5 = chance)")
        ax.grid(axis="x", color=FG.FAINT, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows_)))
    axes[0].set_yticklabels(rows_)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].invert_yaxis()
    h, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=9.5)
    fig.suptitle("Upstream Delphi matches our modified Delphi; both trail the other structures", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    FG._save(fig, path)


def table_upstream_md(P, T):
    """上游原版 Delphi（delphi_upstream.py）：同一份分词，上游的代码和配方，分别在 ROSMAP、NACC 上从零训练，各自的留出集上评估。"""
    U = _j(os.path.join(BZ, "_nacc", "delphi_upstream.json"))
    pr = {b: P.get(("delphi-upstream", b)) for b in ("delphi-dedup-nojitter", "delphi-dedup", "transformer-x5")}
    if not U or not all(pr.values()):
        return []
    from report_text import UPSTREAM_VERDICT
    dd = lambda x: f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"     # noqa: E731
    v = lambda b, k, s: pr[b]["all"][k][s]                                        # noqa: E731
    k1, k3 = "part1_2y_4endpoints", "part3_5y_6questions"
    nu, nm = U["nacc"]["ensemble"], T["models"]
    it = lambda ds: "{:,}–{:,}".format(*(f(U[ds][s]["iter_num"] for s in U[ds] if s.startswith("s")) for f in (min, max)))   # noqa: E731
    L = ["### 上游原版 Delphi，分别在 ROSMAP、NACC 上从零训练（2026-10-27 加，`models_code/delphi_upstream.py`）", "",
         "排除'Delphi 差是因为我们改了它'：github.com/gerstung-lab/Delphi（fb72166）的 model.py / utils.py / train.py 原样拷过去"
         "（md5 和克隆一致），配置用上游的 `config/train_delphi.py`（12 层 × 12 头 × 120 宽，210 万参数，dropout 0，15 万步的学习率日程），"
         "只改数据相关的 dataset、vocab_size、ignore_tokens。分词用已有的两份（ROSMAP 交付版 dedup、NACC 同一个词表）。不微调：各自在自己的数据上训练，"
         "ROSMAP 在 val（统一评估的真值）上评估、NACC 在 test 段（和上表同一批站点、同一套标签）上评估；读法和 zoo 的 Delphi 一样。"
         f"上游只在 val loss 变好时存检查点：选中的是 ROSMAP {it('rosmap')} 步、NACC {it('nacc')} 步，之后 val loss 一路上升到停。"
         f"上游的版本 {len([s for s in U['rosmap'] if s.startswith('s')])} 个 seed、zoo 的 5 个，都是风险按 seed 平均。", "",
         "| 版本 | ROSMAP val 4 终点 2 年 | ROSMAP 6 题 5 年 | NACC test 4 终点 2 年 | NACC 6 题 5 年 |", "|---|---|---|---|---|",
         f"| 上游原版 Delphi | {v('transformer-x5', k1, 'a'):.3f} | {v('transformer-x5', k3, 'a'):.3f} | "
         f"{nu[k1]:.3f} | {nu[k3]:.3f} |",
         f"| zoo 的 Delphi（nojitter 配方） | {v('delphi-dedup-nojitter', k1, 'b'):.3f} | {v('delphi-dedup-nojitter', k3, 'b'):.3f} | "
         f"{nm['delphi-nojitter']['ensemble'][k1]:.3f} | {nm['delphi-nojitter']['ensemble'][k3]:.3f} |",
         f"| zoo 的 Delphi（交付版 dedup，有背景抖动） | {v('delphi-dedup', k1, 'b'):.3f} | {v('delphi-dedup', k3, 'b'):.3f} | — | — |",
         f"| 因果 transformer | {v('transformer-x5', k1, 'b'):.3f} | {v('transformer-x5', k3, 'b'):.3f} | "
         f"{nm['transformer']['ensemble'][k1]:.3f} | {nm['transformer']['ensemble'][k3]:.3f} |", "",
         "ROSMAP val 上按人配对（同上，1000 次）：上游原版 − zoo nojitter：2 年 " + dd(pr["delphi-dedup-nojitter"]["all"][k1])
         + "，5 年 6 题 " + dd(pr["delphi-dedup-nojitter"]["all"][k3]) + "；上游原版 − zoo 交付版：2 年 " + dd(pr["delphi-dedup"]["all"][k1])
         + "；上游原版 − 因果 transformer：2 年 " + dd(pr["transformer-x5"]["all"][k1]) + "，5 年 6 题 " + dd(pr["transformer-x5"]["all"][k3]) + "。"
         f"NACC test 上上游原版每个终点（2 年）：AD {nu['cells']['ad_2y']:.3f}、死亡 {nu['cells']['death_2y']:.3f}、"
         f"MMSE<24 {nu['cells']['mmse_lt24_2y']:.3f}、认知<−1 {nu['cells']['cogn_ltm1_2y']:.3f}。", "",
         "![delphi upstream](MODELS_delphi_upstream.png)", "",
         "**结论**", ""]
    fig_upstream(P, T, os.path.join(ZOO, REPORT_DIR, "MODELS_delphi_upstream.png"))   # 图要 P（配对结果），所以在这里画
    return L + UPSTREAM_VERDICT + [""]


# ----------------------------------------------------------------------------- models_not_usable/README.md（2026-11-05 加）


def moved_md():
    """MODELS.md 里表 4、表 5 原来的位置：指到 models_not_usable/README.md。"""
    return ["## 表 4、表 5：挪到 [models_not_usable/](models_not_usable/README.md)（2026-11-05）", "",
            "表 4（只能直接预测的模型和它们的 rollout 版）和表 5（NACC 预训练：主臂是序列网络的直接作答版 -x5 / -nacc / -nacc-zs，"
            "Delphi 那两小节也拿它们当参照）的主角都是不参与评估的模型，整节挪过去，编号不变。", ""]


def not_usable_readme(R):
    """models_not_usable/README.md：哪 36 个、为什么不参与评估、各自留在评估里的版本；后面接原来 MODELS.md 的表 4、表 5。"""
    by = {r["name"]: r for r in R}
    a5 = lambda r: f3(r["p3"]["auc"][5]) if r and r.get("p3") else "—"       # noqa: E731
    P = sim_pairs(R)
    up = [o["name"] for o, s in P if o["p1"]["auc"] > s["p1"]["auc"]]
    L = ["# 不可用的模型：不参与评估", "",
         f"2026-11-05 起（用户定），只能直接作答的 {len(NOT_USABLE)} 个模型不参与评估"
         "（[MODELS.md](MODELS.md) 的表 1–3、[OVERVIEW_all.png](OVERVIEW_all.png)）。"
         "它们不能生成未来，只回答'H 年内会不会'：逐题逐年直接给一个概率（每道题 × 每个时间窗一个输出），或者给一条生存曲线。"
         "训练目标就是评估题本身；能生成未来的模型要先一年一年模拟出整个未来，再从里面数答案。两种模型不在一条起跑线上：",
         f"同一个模型的两种读法（表 4），{len(P)} 对里有 {len(up)} 对是直接作答的 2 年 AUC 更高"
         + ("（例外：" + "、".join(sorted({o["name"] for o, _ in P} - set(up))) + "）" if len(up) < len(P) else "")
         + "。分高不说明它是更好的模型。", "",
         "打包盒原样留着（README + 图 + 聚合 JSON），RIS 上的 `_runs/packages` 不动。名单在 `unified_eval/registry.py` 的 `NOT_USABLE`"
         "（= `models_code/models.py` 里 future 是 direct / survival 的全部，compare.py 断言两者一致）。", "",
         "| 模型 | 怎么回答 | 2 年 4 终点 AUC | 5 年 6 题 AUC | 留在评估里的版本 | 它的 2 年 / 5 年 AUC |", "|---|---|---|---|---|---|"]
    for n in sorted(NOT_USABLE, key=lambda n: -by[n]["p1"]["auc"]):
        r, tw = by[n], TWIN.get(n)
        t = by.get(tw)
        note = "（同一结构的自回归版，不是同一个模型）" if n.endswith("-x5") else ""
        L.append(f"| [{n}]({r['link']}) | {FUT_ZH[r['future']]} | {f3(r['p1']['auc'])} | {a5(r)} | "
                 + (f"[{tw}]({t['link']}){note} | {f3(t['p1']['auc'])} / {a5(t)} |" if t else "— | — |"))
    L += ["", "AUC 都是 val 上的点估计（各自打包盒的 metrics.json、transitions.json）。", ""]
    L += table_sim_md(R) + table_nacc_md(R)
    L += ["## 2026-11-05 以前写的结论（当时包括这里的 36 个模型）", "",
          "原来在 MODELS.md（当时叫 BASELINES.md）的开头，2026-11-08 换成只比能生成未来的模型的结论后挪到这里，原样保留（只改过'外部模型''文献原样的结构'两处措辞）。", ""]
    L += [x.replace("### 2026-10-22", "#### 2026-10-22").replace("## 结论（", "### 结论（") for x in CONCLUSIONS]
    with open(os.path.join(ZOO, NOT_USABLE_DIR, "README.md"), "w") as f:
        f.write(folder_md("\n".join(L), NOT_USABLE_DIR))


def main():
    FG._style()
    R = rows()
    for r in R:
        if r["group"] != "zoo":
            package_readme(r)
    # 表 1–3 和全部模型的总图：全部可用模型（只能直接预测的换成它们的 rollout 版）+ Delphi；NACC 实验单独一节
    Rm = [r for r in R if r["name"] not in NOT_USABLE and r["name"] not in NACC]
    G = [r for r in Rm if r["group"] != "zoo" and r["traj"]] + zoo_generators()
    assert set(SELECTED) <= {r["name"] for r in R}, sorted(set(SELECTED) - {r["name"] for r in R})
    Ra = rollout_rows(R)                 # 9 模型的图只比 rollout 版
    delphi_sim_readme(next(r for r in Ra if r["name"] == DELPHI_SIM))
    representative_readme(Ra)
    fig_prediction(Ra, os.path.join(ZOO, REPORT_DIR, "MODELS.png"))
    fig_traj(Ra, os.path.join(ZOO, REPORT_DIR, "MODELS_traj.png"))
    fig_sim([r for r in R if r["name"] in SELECTED or ORIG_OF.get(r["name"]) in SELECTED], os.path.join(ZOO, REPORT_DIR, "MODELS_sim.png"),
            [r for r in Ra if r["name"] in ("transformer", DELPHI_SIM)])
    fig_nacc(R, os.path.join(ZOO, REPORT_DIR, "MODELS_nacc.png"))
    fig_overview(Ra, os.path.join(ZOO, REPORT_DIR, "OVERVIEW.png"))
    fig_calibration(Ra, os.path.join(ZOO, REPORT_DIR, "MODELS_calibration.png"))
    fig_timing(Ra, os.path.join(ZOO, REPORT_DIR, "MODELS_timing.png"))
    fig_overview(Rm, os.path.join(ZOO, REPORT_DIR, "OVERVIEW_all.png"))
    from report_text import notes_md
    with open(os.path.join(ZOO, REPORT_DIR, "MODELS.md"), "w") as f:
        f.write(report_md("\n".join(table_md(Rm, G) + moved_md() + table_traj_md() + notes_md(Rm, G))))
    not_usable_readme(R)
    print(f"{sum(r['group'] != 'zoo' for r in Rm)} models + Delphi, {len(G)} generators -> reports/MODELS.md; "
          f"{len(NOT_USABLE)} not usable, {len(sim_pairs(R))} rollout pairs -> {NOT_USABLE_DIR}/README.md")


if __name__ == "__main__":
    main()
