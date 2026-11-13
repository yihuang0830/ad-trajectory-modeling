"""mc_granularity.py -- fig3 面板 D 里"rollout 的 AUC 比直接读出低"，有多少只是因为每人只抽了 100 条轨迹？

    python mc_granularity.py      # -> packages/MC_GRANULARITY.json

做法：拿每个模型在第一次访视站点上**直接读出**的速率 λ，先按比例缩放到和 rollout 一样的风险水平
（p = 1 − exp(−c·λ·2 年)，c 使平均风险等于同一批站点上 rollout 的平均风险；只缩放，不改排序——Delphi 的直接读出
严重低估风险，不缩放的话几乎每人都抽到 0，会把颗粒度损失夸大好几倍），再抽 Binomial(100, p) / 100 当作
"100 条轨迹的 rollout 风险"，算 AUC，重复 50 次取平均。和 p 本身的 AUC 比，差值 = 只由抽样颗粒度造成的损失。
面板 D 的实际下降减去这一项，才是生成器本身（自回归漂移等）造成的。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DAYS_PER_YEAR, ENDPOINTS, RUN_DIR  # noqa: E402
from registry import DELPHI  # noqa: E402

EP = list(ENDPOINTS)


def main(n_mc=100, reps=50):
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    truth = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    base = truth.merge(pd.DataFrame({"pid": [int(p) for p in traj], "t": [traj[p]["t0"] for p in traj]}), on=["pid", "t"])
    rng = np.random.default_rng(20261008)        # 所有模型按注册表顺序共用这一串：新加的模型要加在注册表最后（见 registry.py）
    out = {}
    for name, *_ in DELPHI:
        f = os.path.join(RUN_DIR, "packages", name, "fidelity.json")
        sc_p = os.path.join(RUN_DIR, "scores", f"{name}__s42.csv")
        if not (os.path.exists(f) and os.path.exists(sc_p)):
            continue
        fid = json.load(open(f))
        b = base.merge(pd.read_csv(sc_p), on=["pid", "t"], how="left")
        roll = pd.DataFrame(json.load(open(os.path.join(RUN_DIR, "rollout", name, "fidelity_persons.json")))).set_index("pid")
        res = {}
        for e in EP:
            y = b[f"y_{e}_2y"].to_numpy()
            ok = y >= 0
            lam = np.exp(b.loc[ok, f"logit_{e}"].to_numpy()) * 2 * DAYS_PER_YEAR
            target = float(roll.loc[b.pid[ok], f"risk2y_{e}"].mean())
            lo, hi = 1e-6, 1e6
            for _ in range(200):                                  # 二分：平均风险 = rollout 的平均风险
                c = np.sqrt(lo * hi)
                if (1.0 - np.exp(-c * lam)).mean() < target:
                    lo = c
                else:
                    hi = c
            p = 1.0 - np.exp(-c * lam)
            a0 = roc_auc_score(y[ok], p)
            a_mc = np.mean([roc_auc_score(y[ok], rng.binomial(n_mc, p) / n_mc) for _ in range(reps)])
            res[e] = {"direct": float(a0), "direct_with_100_draws": float(a_mc), "granularity_loss": float(a0 - a_mc),
                      "observed_rollout": fid["auc_baseline"][e]["rollout"]}
        mac = {k: float(np.mean([res[e][k] for e in EP])) for k in ("direct", "direct_with_100_draws", "granularity_loss", "observed_rollout")}
        mac["generator_loss"] = mac["direct_with_100_draws"] - mac["observed_rollout"]
        res["macro"] = mac
        out[name] = res
        print(name, {k: round(v, 3) for k, v in mac.items()}, flush=True)
    json.dump(out, open(os.path.join(RUN_DIR, "packages", "MC_GRANULARITY.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
