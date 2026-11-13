"""vista_nacc_compare.py -- 选定的 transformer（VISTA）先在 NACC 上预训练再到 ROSMAP 上微调，有没有比只在 ROSMAP 上训练好。
RIS 上跑，读人级打分表。

    python vista_nacc_compare.py      # -> $ZOO_RUN_DIR/packages/_vista_nacc/compare.json（聚合，可以拉回本地）

两臂（run.py --variant，各 5 个 seed，同一份数据、同一个训练函数、同一套模拟和打分）：
  naccft  transformer-naccft-s0..s4：seed k 的 NACC 预训练检查点（pretrain.py --what seqar）-> ROSMAP train 微调（pretrain.SeqARNACC）
  base    transformer-base-s0..s4：基因实验里只在 ROSMAP 上训练的同一个模型（原版数据），作对照

读法在看 val 之前定下（2026-11-10），之后不改：
  主指标（两个）：第一部分 4 个终点 2 年 AUC 的平均（p1）；第三部分 6 道题 5 年 AUC 的平均（p3）。和 MODELS.md 表 1 同一种定义。
  人群：val 全部的人（他们的全部站点）。
  主对比：naccft − base。风险 = 5 个 seed 的平均（两臂同一个口径），按人配对 bootstrap 1000 次。
  判定：两个主指标做 Bonferroni（98.75% / 1.25% 分位数 = 每个指标双侧 2.5%）。
        任一个的校正后区间下限 > 0 -> "NACC 预训练提高了"；任一个的上限 < 0 -> "变差了"；否则 -> "没有可检出的差别"，
        并报校正后区间的上限（= 数据能排除的最大提升）。
  次要（不做多重比较校正，只描述）：每道题各自的 AUC 差；第一部分 4 个终点的 O/E 和它们的平均；AD 的 2 / 5 / 10 年 AUC 平均；
        5 对 seed 各自的差（种子方差）；去掉 LATC（Rush 拉丁裔核心队列，可能也在 NACC 里）的人之后的同一组数；
        每个 seed 的 test 段下一步损失（训练记录）、NACC 预训练的损失和轮数。
  扰动和轨迹（第二、三部分的扰动，转移矩阵）不在这里：在本地从拉回来的聚合 JSON 算（experiments/vista_nacc/README.md）。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import bl_data as B
from common import ENDPOINTS, RUN_DIR
from geno_compare import _auc, tables

NAME = "transformer"
ARMS = ("naccft", "base")
SEEDS = range(5)
N_BOOT = 1000
SEED = 20261110
OUT = os.path.join(RUN_DIR, "packages", "_vista_nacc")
S_LATC = 19                                       # 静态背景里 LATC 的那一列（pretrain.S_LATC、nacc_compare 同一个）
ALPHA_BONF = 0.05 / 2                             # 两个主指标


def cells(truth1, truth3):
    """(组, 名字, 第几部分, 站点, 标签, 风险列)。p1 = 第一部分 4 个终点 2 年；p3 = 第三部分 6 道题 5 年；ad = 第三部分 AD 2/5/10 年。"""
    C = [("p1", f"{e}_2y", 0, f"y_{e}_2y", f"risk_{e}_2y") for e in ENDPOINTS]
    C += [("p3", f"{q}_5y", 1, f"y_{q}_5y", f"risk_{q}_5y") for q in B.Q]
    C += [("ad", f"ad_{H}y", 1, f"y_ad_{H}y", f"risk_ad_{H}y") for H in (2, 5, 10)]
    out = []
    for g, nm, part, yc, rc in C:
        t = (truth1, truth3)[part]
        y = t[yc]
        k = y >= 0
        # 名字带上组：第一部分的 ad_2y 和第三部分的 ad_2y 是两个不同的格子（不同的真值表），同名会互相覆盖
        out.append({"group": g, "name": f"{g}:{nm}", "part": part, "idx": t.index[k], "y": y[k].to_numpy(), "col": rc})
    return out


def prepare(C, tabs, pids):
    """每格：参与的人的下标、标签、每臂 × seed 的风险 (seed, 站点)。"""
    pos = {p: i for i, p in enumerate(pids)}
    P = []
    for c in C:
        pp = c["idx"].get_level_values(0)
        k = np.array([p in pos for p in pp])
        idx = c["idx"][k]
        risk = {a: np.stack([T[c["part"]].loc[idx, c["col"]].to_numpy() for T in tabs[a]]) for a in tabs}
        P.append({**c, "pi": np.array([pos[p] for p in pp[k]]), "y": c["y"][k], "risk": risk})
    return P


def metrics(P, arm, w, seed=None):
    """一臂在人权重 w 下：每格 AUC；第一部分每格 O/E；p1 / p3 / ad 的平均和 p1 的平均 O/E。seed=None = 5 个 seed 的平均风险。"""
    res = {}
    for c in P:
        r = c["risk"][arm].mean(0) if seed is None else c["risk"][arm][seed]
        ww = w[c["pi"]]
        res[c["name"]] = _auc(c["y"], r, ww)
        if c["group"] == "p1":
            res[c["name"] + "_oe"] = float((ww * c["y"]).sum() / (ww * r).sum())
    for g in ("p1", "p3", "ad"):
        res[g] = float(np.nanmean([res[c["name"]] for c in P if c["group"] == g]))
    res["p1_oe"] = float(np.mean([res[c["name"] + "_oe"] for c in P if c["group"] == "p1"]))
    return res


def compare(P, n, rng):
    one = np.ones(n)
    full = {a: metrics(P, a, one) for a in ARMS}
    keys = list(full[ARMS[0]])
    boot = {a: {k: [] for k in keys} for a in ARMS}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(n, size=n), minlength=n).astype(float)
        for a in ARMS:
            m = metrics(P, a, w)
            for k in keys:
                boot[a][k].append(m[k])
    per_seed = {a: [metrics(P, a, one, s) for s in SEEDS] for a in ARMS}
    out = {"value": {a: full[a] for a in ARMS}, "diff": {}}
    for k in keys:
        d = np.array(boot["naccft"][k]) - np.array(boot["base"][k])
        ds = [per_seed["naccft"][s][k] - per_seed["base"][s][k] for s in SEEDS]
        out["diff"][k] = {"diff": full["naccft"][k] - full["base"][k],
                          "ci95": [float(np.nanquantile(d, q)) for q in (0.025, 0.975)],
                          "ci_bonf2": [float(np.nanquantile(d, q)) for q in (ALPHA_BONF / 2, 1 - ALPHA_BONF / 2)],
                          "p_le_0": float(np.nanmean(d <= 0)), "per_seed": ds, "per_seed_sd": float(np.std(ds, ddof=1))}
    out["per_seed_value"] = per_seed
    return out


def verdict(r):
    """主判定（只看全部 val、两个主指标的 Bonferroni 区间）。"""
    lo = [r["diff"][k]["ci_bonf2"][0] for k in ("p1", "p3")]
    hi = [r["diff"][k]["ci_bonf2"][1] for k in ("p1", "p3")]
    if max(lo) > 0:
        return "improved"
    if min(hi) < 0:
        return "worse"
    return "no detectable difference"


def training_records():
    """每个 seed 的训练记录（打包盒的 baseline.json）+ NACC 预训练检查点的记录。"""
    out = {}
    for a in ARMS:
        for k in SEEDS:
            p = os.path.join(RUN_DIR, "packages", f"{NAME}-{a}-s{k}", "baseline.json")
            if os.path.exists(p):
                rec = json.load(open(p))
                info = rec["log"][0]
                out[f"{a}-s{k}"] = {"test_bce_risk": rec.get("test_bce"), "test_macro_auc_2_5_10": rec.get("test_macro_auc_2_5_10"),
                                    "best_test_loss": info.get("best_test_loss"), "epochs": info.get("epochs"),
                                    "lr": info.get("lr", (info.get("hp") or {}).get("lr")), "tries": info.get("tries"),
                                    "minutes_total": rec.get("minutes_total")}
    import pretrain as PRT
    for pj in sorted(glob.glob(os.path.join(PRT.CKPT, "seqar-*.json"))):
        out["pretrain:" + os.path.basename(pj)[:-5]] = json.load(open(pj))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    truth1 = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv")).set_index(["pid", "t"])
    truth3 = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv")).set_index(["pid", "t"])
    C = cells(truth1, truth3)
    tabs = {a: tables(NAME, a) for a in ARMS}
    missing = [a for a, t in tabs.items() if t is None]
    if missing:
        raise SystemExit(f"missing score tables for {missing} (need {NAME}-<arm>-s0..s4)")
    Z, _ = B.load()
    latc = Z["val_pid"][Z["val_S"][:, S_LATC] > 0.5]
    all_p = np.unique(truth1.index.get_level_values(0))
    pops = {"all_val": all_p, "no_latc": np.setdiff1d(all_p, latc)}
    res = {"n_persons": {k: int(len(v)) for k, v in pops.items()}, "n_boot": N_BOOT, "seeds": list(SEEDS),
           "arms": {"naccft": f"{NAME}-naccft-s<k>", "base": f"{NAME}-base-s<k>"}}
    for pop, pids in pops.items():
        res[pop] = compare(prepare(C, tabs, pids), len(pids), np.random.default_rng(SEED))
    res["verdict"] = verdict(res["all_val"])
    res["training"] = training_records()
    json.dump(res, open(os.path.join(OUT, "compare.json"), "w"), indent=1, default=float)
    for k in ("p1", "p3"):
        x = res["all_val"]["diff"][k]
        v = res["all_val"]["value"]
        print(f"{k}: naccft {v['naccft'][k]:.4f} - base {v['base'][k]:.4f} = {x['diff']:+.4f} "
              f"95% [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]  Bonferroni [{x['ci_bonf2'][0]:+.4f}, {x['ci_bonf2'][1]:+.4f}]",
              flush=True)
    print("verdict:", res["verdict"], "->", OUT, flush=True)


if __name__ == "__main__":
    main()
