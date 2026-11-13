"""val_vs_test.py -- Delphi 用 val 挑 checkpoint，val 上的数会不会偏乐观？（EVAL_PROTOCOL.md 第 8 节最后一行）

    python val_vs_test.py      # -> packages/VAL_VS_TEST.json

split811 的 Delphi 模型没在 test 段上训练过、也没用 test 挑过 checkpoint，所以同一套统一评估在 test 上再算一遍：
  val 和 test 差不多 -> 用 val 挑 checkpoint 带来的乐观偏差很小；val 明显更高 -> 主结果偏乐观。
两分的 Delphi 训练时见过 test 段的人，不能这样比，所以不在这里。
test 段的真值 = truth.py 在 snapshot 的 train.bin 上生成的那份（test 段的人在两分版里属于 train），按人筛出来。
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DELPHI_DIR, RUN_DIR, load_bin  # noqa: E402
from evaluate import endpoint_metrics, macro  # noqa: E402
from registry import DELPHI, seeds_of  # noqa: E402


def boot_macro(truth, sc, B=300, seed=20261008):
    """按人 bootstrap 的 4 终点 2 年 AUC 平均（用 rank_/logit_ 排序）。"""
    from sklearn.metrics import roc_auc_score
    pid = truth.pid.to_numpy()
    persons = np.unique(pid)
    groups = {p: np.where(pid == p)[0] for p in persons}
    rng = np.random.default_rng(seed)
    cols = [next(c for c in (f"rank_{e}", f"logit_{e}") if c in sc.columns) for e in ("ad", "death", "mmse", "cogn")]
    ys = [truth[f"y_{e}_2y"].to_numpy() for e in ("ad", "death", "mmse", "cogn")]
    out = []
    for _ in range(B):
        ii = np.concatenate([groups[p] for p in rng.choice(persons, len(persons))])
        v = []
        for y, c in zip(ys, cols):
            yy, ss = y[ii], sc[c].to_numpy()[ii]
            ok = yy >= 0
            v.append(roc_auc_score(yy[ok], ss[ok]))
        out.append(np.mean(v))
    return np.array(out)


def main():
    test_pids = set(np.unique(load_bin("rosmap_nodedup_3w", "test")[:, 0]).tolist())
    tr = pd.read_csv(os.path.join(RUN_DIR, "truth_train.csv"))
    tt = tr[tr.pid.isin(test_pids)].reset_index(drop=True)
    assert tt.pid.nunique() == len(test_pids), "test 段有人不在真值表里"
    tt.to_csv(os.path.join(RUN_DIR, "truth_test.csv"), index=False)
    tv = pd.read_csv(os.path.join(RUN_DIR, "truth_val.csv"))
    out = {"n_test_persons": len(test_pids), "n_test_landmarks": int(len(tt)), "models": {}}
    for name, dirs, dataset, *_ in DELPHI:
        if not name.endswith("split811"):
            continue
        res = {}
        for d, s in zip(dirs, seeds_of(dirs)):
            ck = os.path.join(DELPHI_DIR, d, "ckpt.pt")
            o = os.path.join(RUN_DIR, "scores", f"{name}__s{s}_test.csv")
            if not os.path.exists(o):
                subprocess.run([sys.executable, os.path.join(HERE, "score_delphi.py"), "--ckpt", ck, "--dataset", dataset,
                                "--split", "test", "--truth", os.path.join(RUN_DIR, "truth_test.csv"), "--out", o,
                                "--device", "cpu"], check=True)
            v = pd.read_csv(os.path.join(RUN_DIR, "scores", f"{name}__s{s}.csv"))
            t = pd.read_csv(o)
            sv = tv[["pid", "t"]].merge(v, on=["pid", "t"], how="left")
            st = tt[["pid", "t"]].merge(t, on=["pid", "t"], how="left")
            mv = endpoint_metrics(tv, sv)
            mt = endpoint_metrics(tt, st)
            # 两个独立样本：各自按人 bootstrap（**不同的随机种子**：两边人数相同、按 pid 排序，同一个种子会抽到同样的
            # 名次位置，两个分布被人为拉成正相关；第二轮审查后改），差值的分布 = 两边抽样分布之差
            bv, bt = boot_macro(tv, sv, seed=20261008), boot_macro(tt, st, seed=20261009)
            diff = bv - bt
            sd = float(np.std(diff))
            res[f"s{s}"] = {"val": macro(mv, 2), "test": macro(mt, 2),
                            "diff_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                            "ci_half_width": 1.96 * sd,
                            # 真有这么大的乐观偏差时，80% 的把握能检测出来（双侧 0.05）：(1.96 + 0.84)·SD
                            "mdd_80pct_power": 2.8 * sd,
                            "val_ep": {e: mv[f"{e}_2y"]["auc"] for e in ("ad", "death", "mmse", "cogn")},
                            "test_ep": {e: mt[f"{e}_2y"]["auc"] for e in ("ad", "death", "mmse", "cogn")}}
            print(name, f"s{s}", f"val {res[f's{s}']['val']:.3f}  test {res[f's{s}']['test']:.3f}", flush=True)
        dv = [r["val"] - r["test"] for r in res.values()]
        out["models"][name] = {"seeds": res, "mean_val_minus_test": float(np.mean(dv)),
                               "ci_half_width": float(np.mean([r["ci_half_width"] for r in res.values()])),
                               "mdd_80pct_power": float(np.mean([r["mdd_80pct_power"] for r in res.values()])),
                               "max_diff_ci_upper": float(max(r["diff_ci"][1] for r in res.values()))}
    json.dump(out, open(os.path.join(RUN_DIR, "packages", "VAL_VS_TEST.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
