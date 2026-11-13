"""collect_prior.py -- 把统一评估之前算过的指标摘到每个打包盒的 prior_results.md。

    python collect_prior.py --prior <从 RIS 拉回的结果目录（只含聚合指标）> [--legacy_auc ../../delphi/Delphi-ROSMAP/auc]

只摘**模型层面的聚合数字**；含 pid 的逐人文件一律不读（也不该在本地）。
口径各不相同，所以每一行都写清楚"问的是什么"。
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from registry import SELF_DIR  # noqa: E402

# figure 3（Delphi 上游 AUC 口径 + post_entry）：config -> 打包盒
FIG3_PKG = {"dedup": "delphi-dedup", "nodedup": "delphi-nodedup", "fullvisit": "delphi-fullvisit",
            "nodedup-pos-3w": "delphi-nodedup-pos-split811", "fullvisit3w": "delphi-fullvisit-split811",
            "static": "delphi-nodedup-staticonly-split811", "visit1": "delphi-fullvisit-visit1only-split811",
            "snapshot": "delphi-snapshot", "snapshot-pos": "delphi-snapshot-pos",
            "snapshot-static": "delphi-snapshot-staticonly", "snapshot-visit1": "delphi-snapshot-visit1only"}
# 每个 config 用最新的有效版本（v3 起 post_entry=1；v2 及以前含入组前预测点，作废）
FIG3_VERSION_ORDER = ["v8_panelA", "v7_panelB", "v6_sameset", "v6_panelA", "v6_panelB", "v5_readout_all",
                      "v4_main", "v4_baselines", "v3_main", "v3_baselines"]
FIG2_PKG = {"gpubase": "delphi-dedup", "nodedup": "delphi-nodedup", "fullvisit": "delphi-fullvisit",
            "": "delphi-dedup-mps-legacy"}
DH_PKG = {("direct_head_three.json", "dedup"): "delphi-dedup", ("direct_head_three.json", "nodedup"): "delphi-nodedup",
          ("direct_head_three.json", "fullvisit"): "delphi-fullvisit",
          ("direct_head_seeds.json", "pos-s42"): "delphi-nodedup-pos",
          ("direct_head_arch.json", "aux"): "delphi-nodedup-aux", ("direct_head_arch.json", "vh"): "delphi-nodedup-visitheads",
          ("direct_head_test3w.json", "nd3w"): "delphi-nodedup-split811",
          ("direct_head_test3w.json", "pos3w"): "delphi-nodedup-pos-split811"}
WB_PKG = {("window_baselines_three.json", "dedup"): "delphi-dedup", ("window_baselines_three.json", "nodedup"): "delphi-nodedup",
          ("window_baselines_three.json", "fullvisit"): "delphi-fullvisit",
          ("window_baselines_pos.json", "nodedup-pos"): "delphi-nodedup-pos",
          ("window_baselines_arch.json", "aux"): "delphi-nodedup-aux",
          ("window_baselines_arch.json", "vh"): "delphi-nodedup-visitheads"}
UPSTREAM_AUC = {"delphi-dedup": "delphi/Delphi-ROSMAP-gpubase/auc", "delphi-nodedup": "delphi/Delphi-ROSMAP-nodedup/auc",
                "delphi-fullvisit": "delphi/Delphi-ROSMAP-fullvisit/auc"}
NT_PKG = {"dedup": "delphi-dedup", "nodedup": "delphi-nodedup", "fullvisit": "delphi-fullvisit"}


def fig3_rows(prior):
    out = {}
    for cfg, pkg in FIG3_PKG.items():
        for v in FIG3_VERSION_ORDER:
            f = os.path.join(prior, "delphi/results/fig3", v, "fig3_auc_summary.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            d = d[(d.config == cfg) & (d.score == "model") & (d.post_entry == 1)]
            if not len(d):
                continue
            rows = []
            for off, lab in ((0.1, "nogap（≈提前 1 年）"), (365.25, "gap 1 年（≈提前 2 年）")):
                x = d[np.isclose(d.offset, off)]
                per_seed = x.groupby("seed").auc.median()
                grp = x.groupby(["seed", "group"]).auc.median().groupby("group").mean()
                rows.append((f"Delphi 上游 AUC，{lab}，post_entry=1",
                             "每个 token 一题：站在随机预测点上，这个人下一段时间会不会首次出现该 token（上游 evaluate_auc 的病例/对照构造，年龄性别分层）",
                             f"所有 token 中位 {per_seed.mean():.3f}" + (f" ± {per_seed.std(ddof=1):.3f}" if len(per_seed) > 1 else "")
                             + "；" + "，".join(f"{g} {a:.3f}" for g, a in grp.items()),
                             f"delphi/results/fig3/{v}/fig3_auc_summary.csv（{len(per_seed)} 个 seed）"))
            out[pkg] = rows
            break
    return out


def upstream_rows(prior, legacy_auc):
    out = {}
    srcs = dict(UPSTREAM_AUC)
    for pkg, rel in list(srcs.items()) + ([("delphi-dedup-mps-legacy", None)] if legacy_auc else []):
        rows = []
        for gap, lab in (("nogap", "offset 0.1"), ("gap365d", "offset 365.25")):
            f = os.path.join(legacy_auc, f"auc_pooled_{gap}.csv") if rel is None else os.path.join(prior, rel, f"auc_pooled_{gap}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            rows.append((f"evaluate_auc_rosmap.py（上游原样，{lab}）",
                         "同上游 Delphi AUC，但**含入组前预测点**（旧口径，事件偏高、用药/状态偏低）",
                         f"token 中位 {d.auc.median():.3f}，均值 {d.auc.mean():.3f}（{len(d)} 个 token）",
                         f.replace(prior + "/", "") if rel is not None else "delphi/Delphi-ROSMAP/auc/" + os.path.basename(f)))
        if rows:
            out[pkg] = rows
    return out


def fig2_rows(prior):
    out = {}
    for tag, pkg in FIG2_PKG.items():
        f = os.path.join(prior, "figure2_eval/results/figure2", tag, "metrics_allcohort.json")
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        a = m["a"]
        ps = a.get("per_state", {})
        det = "，".join(f"{k} {v['auc']:.3f}" for k, v in ps.items() if k in ("AD diagnosis", "Death", "Impaired"))
        out[pkg] = [("figure 2 panel a（Monte-Carlo rollout）",
                     f"从基线前缀生成 100 条未来轨迹，{a['horizon']} 年内会不会到达某状态，{a['n_states']} 个状态的中位 AUC",
                     f"中位 {float(a['median_auc']):.3f}（{det}）",
                     f"figure2_eval/results/figure2/{tag + '/' if tag else ''}metrics_allcohort.json")]
    return out


def dh_rows(prior):
    out = {}
    for (fn, key), pkg in DH_PKG.items():
        f = os.path.join(prior, "figure2_eval/results", fn)
        if not os.path.exists(f):
            continue
        d = json.load(open(f)).get(key)
        if not d:
            continue
        pa = d["panel_a"]
        med = lambda k: float(np.median([r[k] for r in pa if r.get(k) is not None]))  # noqa: E731
        txt = f"冻结表征 + logistic 中位 {med('auc_emb'):.3f}；同题 rollout {med('auc_rollout'):.3f}；只用原始前缀特征的 LR {med('auc_raw'):.3f}"
        split = "test（split811）" if "test3w" in fn else "val"
        out.setdefault(pkg, []).append((
            "直接判别头探针（figure 2 同题，不做 rollout）",
            f"冻结 encoder，取基线前缀最后一位隐状态接 logistic，预测 5 年内 {len(pa)} 个状态；{split}",
            txt, f"figure2_eval/results/{fn}[{key}]"))
    return out


def wb_rows(prior):
    out = {}
    for (fn, key), pkg in WB_PKG.items():
        f = os.path.join(prior, "figure2_eval/results", fn)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        parts = []
        for fam in ("MMSE", "cogn_global"):
            pr = d[fam]["per_run"]
            if key not in pr:
                continue
            W = pr.get("W 单访视窗口", {}).get("marginal_macro_auc")
            N = pr.get("N 完全不转变", {}).get("marginal_macro_auc")
            parts.append(f"{fam} {pr[key]['marginal_macro_auc']:.3f}（地板 N {N:.3f} / 单访视 LR W {W:.3f}）")
        if parts:
            out.setdefault(pkg, []).append((
                "下一次访视的认知分箱（per-token macro AUC，无 rollout）",
                "给定到本次访视为止的历史，下一次访视 MMSE / cogn_global 落在哪个箱；N = 完全不转变，W = 只看当次访视的 logistic",
                "；".join(parts), f"figure2_eval/results/{fn}[{key}]"))
    return out


def nt_rows(prior):
    out = {}
    f = os.path.join(prior, "figure2_eval/results/next_token_auc_three.json")
    if not os.path.exists(f):
        return out
    meta = json.load(open(f))["meta"]
    for key, pkg in NT_PKG.items():
        if key in meta:
            m = meta[key]
            out[pkg] = [("下一个 token top-1 准确率", "teacher forcing，每个位置预测下一个 token",
                         f"{float(m['top1']):.3f}（多数类基线 {float(m['majority_top1']):.3f}）",
                         "figure2_eval/results/next_token_auc_three.json")]
    return out


def pos_leak_note(d):
    """带位置编码的包：以前的读法（完整记录）漏了多少。数字从这个包自己的 pos_leak.json 读（3 种对齐下的范围），不写死。"""
    txt = ("- ⚠ 带位置编码：以前的评估把**完整记录**喂给模型，右对齐下位置下标和'后面还剩几个 token'挂钩，"
           "会把未来的长度漏给模型；而且以前按 block_size 对齐，最后几个 token 落在训练里从没占用过的位置上。")
    p = os.path.join(d, "pos_leak.json")
    if os.path.exists(p):
        ba = json.load(open(p))["by_align"]
        diff = {e: [ba[k]["endpoints"][e]["leaky"] - ba[k]["endpoints"][e]["honest"] for k in ba]
                for e in ("ad", "death", "mmse", "cogn")}
        other = max(abs(x) for e in ("ad", "mmse", "cogn") for x in diff[e])
        txt += ("统一评估的位置编码泄漏检查（README 里'位置编码会不会漏未来'一节，数字在 pos_leak.json）量了泄漏本身（按训练方式对齐、同一 ckpt 只换排布）：死亡 2 年 AUC 被抬高 "
                f"{min(diff['death']):+.2f} 到 {max(diff['death']):+.2f}，其它终点的变化在 ±{math.ceil(other * 100) / 100:.2f} 以内。")
    return txt + "上表里这个模型的旧数字都受影响，仅作记录。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--legacy_auc", default=os.path.join(ZOO, "..", "delphi", "Delphi-ROSMAP", "auc"))
    args = ap.parse_args()
    by_pkg = {}
    for fn in (fig3_rows(args.prior), upstream_rows(args.prior, args.legacy_auc), fig2_rows(args.prior),
               dh_rows(args.prior), wb_rows(args.prior), nt_rows(args.prior)):
        for pkg, rows in fn.items():
            by_pkg.setdefault(pkg, []).extend(rows)
    for d in sorted(glob.glob(os.path.join(ZOO, SELF_DIR, "delphi-*"))):
        pkg = os.path.basename(d)
        rows = by_pkg.get(pkg, [])
        L = [f"# {pkg}：以前的指标", "",
             "以下是统一评估之前已经算过的指标，口径各不相同，**不能和统一评估的数字直接比**。", ""]
        if rows:
            L += ["| 指标 | 问的是什么 | 数值（seed 42；多 seed 为均值 ± sd） | 来源（RIS，相对 project/） |", "|---|---|---|---|"]
            L += [f"| {a} | {b} | {c} | `{s}` |" for a, b, c, s in rows]
        else:
            L.append("这个模型在统一评估之前没有单独算过指标。")
        L += ["", "注意：", "",
              "- figure 3 的 v2 及更早版本含入组前预测点（事件 AUC 偏高、用药/状态偏低），已作废，这里只取 v3 之后（post_entry=1）的最新版本。"]
        if "-pos" in pkg:
            L.append(pos_leak_note(d))
        if "snapshot" in pkg:
            L.append("- snapshot 分词的 Delphi 读分位置在训练中不受监督（见 README），上表的 Delphi AUC 同样受影响。")
        with open(os.path.join(d, "prior_results.md"), "w") as f:
            f.write("\n".join(L) + "\n")
        print(pkg, len(rows), "rows")


if __name__ == "__main__":
    main()
