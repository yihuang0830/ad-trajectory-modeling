"""tune_minimalrnn.py -- 按论文的搜索范围给 MinimalRNN 调参（Nguyen et al., NeuroImage 2020, Table 2；论文用 HORD，这里用随机搜索）。

  --stage search   第 0 组 = minimalrnn 打包盒用的那组（generators.MRNN_HP），第 1..N-1 组从论文的范围里随机抽；每组 seed 0
  --stage seeds    seed 0 最好的 TOP 组 + 第 0 组，再各跑 seed 1、2
  --stage select   3 个 seed 的平均最高的那组 -> chosen.json（models.py 的 minimalrnn-tuned 读它）、summary.json

打分 = test 段（调参用）上 6 题 × 2/5/10 年平均 AUC，从每个站点往后模拟 100 条轨迹（和 run.py 同一个读法、同一个随机种子）；
val 一次都不碰。每组结果一个 JSON，已经有的跳过，所以作业超时了再交一次就接着跑：
    sbatch -p general-short --time=00:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=24G \\
        ris/py.sbatch tune_minimalrnn.py --stage search --shard 0 --n_shards 8
产出（只有超参和汇总分数，可以拿回仓库）：$ZOO_RUN_DIR/tune_minimalrnn/{results/c<组>_s<seed>.json, summary.json, chosen.json}
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import bl_data as B
import generators as G
import simulator as SIM

OUT = os.path.join(B.RUN_DIR, "tune_minimalrnn")
SEED = 20261020                    # run.py 的站点模拟种子
N_CONFIGS, TOP, SEEDS = 24, 3, (1, 2)


def configs(n=N_CONFIGS, seed=20261029):
    """论文 Table 2：input / recurrent dropout 0–0.5，L2 1e-7–1e-5，学习率 1e-5–1e-2（对数均匀），1–3 层，隐层 128–512。
    论文没有输出层 dropout，L2 是 Adam 的 weight decay。"""
    rng = np.random.default_rng(seed)
    out = [{}]
    for _ in range(n - 1):
        out.append({"d": int(rng.choice(np.arange(128, 513, 64))), "n_layers": int(rng.integers(1, 4)), "dropout": 0.0,
                    "in_drop": round(float(rng.uniform(0, 0.5)), 3), "rec_drop": round(float(rng.uniform(0, 0.5)), 3),
                    "lr": float(f"{10 ** rng.uniform(-5, -2):.2e}"), "wd": float(f"{10 ** rng.uniform(-7, -5):.2e}"),
                    "opt": "adam"})
    return out


def score(model, Z, norm, dev, n_mc=100, limit=0):
    st = SIM.landmark_start(Z, "test")
    lm = B.landmarks(B.split(Z, "test"), "cur")
    if limit:
        st = {k: (v[:limit] if k in ("ii", "vv", "pid", "t") else v) for k, v in st.items()}
        lm = {k: v[:limit] for k, v in lm.items()}
    rng = np.random.default_rng(SEED)
    R = np.zeros((len(st["ii"]), B.NQ, B.NH))
    for a in range(0, len(st["ii"]), 1000):
        sub = {k: (v[a: a + 1000] if k in ("ii", "vv", "pid", "t") else v) for k, v in st.items()}
        R[a: a + 1000] = SIM.risks(G.minimalrnn_sim(model, sub, n_mc, rng, norm, dev))
    return B.macro_auc(lm["Y"], R), B.masked_bce(lm["Y"], R)


def run_one(idx, seed, hp, Z, meta, dev, args):
    path = os.path.join(args.out, "results", f"c{idx:02d}_s{seed}.json")
    if os.path.exists(path):
        return
    Dtr, Dte = B.split(Z, "train"), B.split(Z, "test")
    if args.smoke:
        Dtr = {k: v[:200] for k, v in Dtr.items()}
        Dte = {k: v[:50] for k, v in Dte.items()}
    t0 = time.time()
    model, info = G.fit_minimalrnn(Dtr, Dte, dev, seed=seed, hp=hp, max_epochs=2 if args.smoke else 150)
    t1 = time.time()
    auc, bce = score(model, Z, SIM.norm_info(meta), dev, n_mc=10 if args.smoke else 100, limit=200 if args.smoke else 0)
    rec = {"config": idx, "seed": seed, "test_macro_auc_2_5_10": auc, "test_bce": bce,
           "minutes_train": (t1 - t0) / 60, "minutes_score": (time.time() - t1) / 60, **info}
    print(f"c{idx:02d} s{seed}: test macro AUC {auc:.4f}  bce {bce:.4f}  epochs {info['epochs']}  "
          f"{rec['minutes_train']:.1f}+{rec['minutes_score']:.1f} min  {json.dumps(info['hp'])}", flush=True)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


def load_results(out):
    R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "results", "c*_s*.json")))]
    return {(r["config"], r["seed"]): r for r in R}


def candidates(res):
    s0 = sorted((r for (c, s), r in res.items() if s == 0 and c > 0), key=lambda r: -r["test_macro_auc_2_5_10"])
    return [0] + [r["config"] for r in s0[:TOP]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("search", "seeds", "select"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--budget", type=float, default=18, help="分钟：过了这么久就不再开新的一组（留给下一个作业）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：200 人训 2 轮、200 个站点 × 10 条，写进 smoke 子目录")
    args = ap.parse_args()
    args.out = os.path.join(OUT, "smoke") if args.smoke else OUT
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    C = configs()
    if args.stage == "select":
        res = load_results(args.out)
        assert all((c, 0) in res for c in range(len(C))), "search 还没跑完"
        cand = candidates(res)
        assert all((c, s) in res for c in cand for s in SEEDS), "seeds 还没跑完"
        rows = []
        for c in range(len(C)):
            a = [res[(c, s)]["test_macro_auc_2_5_10"] for s in (0,) + SEEDS if (c, s) in res]
            rows.append({"config": c, "hp": res[(c, 0)]["hp"], "auc_seeds": a, "auc_mean": float(np.mean(a)),
                         "epochs_s0": res[(c, 0)]["epochs"], "n_params": res[(c, 0)]["n_params"],
                         "minutes_train_s0": res[(c, 0)]["minutes_train"]})
        best = max((r for r in rows if r["config"] in cand), key=lambda r: r["auc_mean"])
        summary = {"criterion": "test split, 6 questions x 2/5/10 y macro AUC, rollout 100 trajectories per landmark",
                   "search_space": "Nguyen et al. 2020 Table 2 (random search, %d configs; config 0 = minimalrnn package)" % len(C),
                   "candidates": cand, "chosen": best["config"], "default_mean": rows[0]["auc_mean"], "rows": rows}
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        with open(os.path.join(args.out, "chosen.json"), "w") as f:
            json.dump({"config": best["config"], "hp": best["hp"], "auc_mean": best["auc_mean"]}, f, indent=1)
        for r in sorted(rows, key=lambda r: -r["auc_mean"]):
            print(f"c{r['config']:02d}  mean {r['auc_mean']:.4f}  seeds {np.round(r['auc_seeds'], 4).tolist()}  {json.dumps(r['hp'])}")
        print("chosen:", best["config"], "default mean", round(rows[0]["auc_mean"], 4))
        return
    import torch
    Z, meta = B.load()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.stage == "search":
        jobs = [(c, 0) for c in range(len(C))]
    else:
        res = load_results(args.out)
        assert all((c, 0) in res for c in range(len(C))), "search 还没跑完"
        jobs = [(c, s) for c in candidates(res) for s in SEEDS]
    t0 = time.time()
    for c, s in jobs[args.shard::args.n_shards]:
        if (time.time() - t0) / 60 > args.budget:
            print("budget used up, leaving the rest to the next job", flush=True)
            break
        run_one(c, s, C[c], Z, meta, dev, args)


if __name__ == "__main__":
    main()
