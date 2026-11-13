"""run_trans_all.py -- 第三部分（疾病分期转换）：在 RIS 上把注册表里的全部模型跑一遍。

    python run_trans_all.py [--only name1,name2] [--device cuda] [--rescore]

1. transition_truth.py --split val / train（若还没有；和第一部分逐格对照）
2. 每个 Delphi (模型, seed)：score_delphi_trans -> $ZOO_RUN_DIR/scores_trans/<name>__s<seed>.csv（人级，留在 RIS），
   读分位置和第一部分逐站点核对
3. evaluate_transitions -> $ZOO_RUN_DIR/packages/<name>/transitions.json
只有第 3 步的产物（聚合指标）会被拷回本地；图在本地由 figures.py 按它画。
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA_DIR, DELPHI_DIR, RUN_DIR, TRUTH_DATASET  # noqa: E402
from registry import CAL_NOTES, DELPHI, seeds_of  # noqa: E402
from run_delphi_all import fingerprint, mark, run, stale  # noqa: E402


def truth_step(py, rebuild=True):
    """真值过期就重建并打指纹。rebuild=False（--skip_truth）：只检查，过期就停 ——
    几个 --only 子集并行跑时，只能有一个进程重建真值（先单独跑一次 --truth_only），否则它们会同时改写同一个 CSV。
    直接跑 transition_truth.py 不会写指纹，后面的作业会以为它过期；所以单独建真值也要走这里（--truth_only）。"""
    for split in ("val", "train"):
        out = os.path.join(RUN_DIR, f"truth_trans_{split}.csv")
        fp = fingerprint(os.path.join(HERE, "transition_truth.py"), os.path.join(HERE, "transitions.py"),
                         os.path.join(HERE, "common.py"), os.path.join(RUN_DIR, f"truth_{split}.csv"),
                         *(os.path.join(DATA_DIR, TRUTH_DATASET, f) for f in (f"{split}.bin", "labels.csv")))
        if stale(out, fp):
            if not rebuild:
                raise SystemExit(f"{out} 过期或没有指纹：先单独跑一次 `python run_trans_all.py --truth_only`，再并行跑 --skip_truth 的子集")
            run([py, os.path.join(HERE, "transition_truth.py"), "--split", split])
            mark(out, fp)


def evaluate_pkg(py, name, scores, calib_undefined=False):
    pk = os.path.join(RUN_DIR, "packages", name)
    os.makedirs(pk, exist_ok=True)
    cmd = [py, os.path.join(HERE, "evaluate_transitions.py"), "--name", name, "--outdir", pk, "--scores", *scores]
    if calib_undefined:
        cmd.append("--calibration_undefined")
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--truth_only", action="store_true", help="只（重）建第三部分的真值并打指纹，然后退出")
    ap.add_argument("--skip_truth", action="store_true", help="不重建真值，只检查它是新的（并行子集用）")
    args = ap.parse_args()
    py = sys.executable
    sdir = os.path.join(RUN_DIR, "scores_trans")
    os.makedirs(sdir, exist_ok=True)
    truth_step(py, rebuild=not args.skip_truth)
    if args.truth_only:
        return
    truth_csv = os.path.join(RUN_DIR, "truth_trans_val.csv")
    only = None if not args.only else set(args.only.split(","))

    for name, dirs, dataset, tok, sw, note in DELPHI:
        if only and name not in only:
            continue
        scores = []
        for d, s in zip(dirs, seeds_of(dirs)):
            ck = os.path.join(DELPHI_DIR, d, "ckpt.pt")
            if not os.path.exists(ck):
                print(f"!! 缺 ckpt：{ck}，跳过这个 seed", flush=True)
                continue
            out = os.path.join(sdir, f"{name}__s{s}.csv")
            p1 = os.path.join(RUN_DIR, "scores", f"{name}__s{s}.csv")
            fp = fingerprint(os.path.join(HERE, "score_delphi_trans.py"), os.path.join(HERE, "score_delphi.py"),
                             os.path.join(HERE, "transitions.py"), os.path.join(HERE, "common.py"), ck, truth_csv,
                             *(os.path.join(DATA_DIR, dataset, f) for f in ("val.bin", "train.bin", "labels.csv")),
                             os.path.join(DELPHI_DIR, "utils.py"), os.path.join(DELPHI_DIR, "model.py"))
            if args.rescore or stale(out, fp):
                cmd = [py, os.path.join(HERE, "score_delphi_trans.py"), "--ckpt", ck, "--dataset", dataset,
                       "--out", out, "--device", args.device]
                if os.path.exists(p1):
                    cmd += ["--part1", p1]
                run(cmd)
                mark(out, fp)
            scores.append(f"s{s}={out}")
        if not scores:
            print(f"!! {name}: 一个 ckpt 都没有", flush=True)
            continue
        evaluate_pkg(py, name, scores, calib_undefined=name in CAL_NOTES)


if __name__ == "__main__":
    main()
