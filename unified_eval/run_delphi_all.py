"""run_delphi_all.py -- 在 RIS 上把注册表里的全部 Delphi ckpt 跑一遍统一评估。

    python run_delphi_all.py [--only name1,name2] [--device cuda]

1. truth.py（若还没有）
2. 每个 (模型, seed)：score_delphi -> $ZOO_RUN_DIR/scores/<name>__s<seed>.csv    （人级，留在 RIS）
3. 统一评估 -> $ZOO_RUN_DIR/packages/<name>/{metrics.json, fig*.png}
4. 每个模型导出 ckpt_info.json（配置、参数量、best val、数据规模）给 README 用
只有第 3、4 步的产物（聚合指标和图）会被拷回本地。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA_DIR, DELPHI_DIR, RUN_DIR, TRUTH_DATASET, load_bin  # noqa: E402
from registry import CAL_NOTES, DELPHI, seeds_of  # noqa: E402


def fingerprint(*paths):
    """代码 / 数据 / 真值 / ckpt 的指纹：任何一个变了，缓存的产物就作废。
    .py / .csv 按内容算；.bin、ckpt 这类大文件按大小 + 修改时间算。"""
    h = hashlib.sha256()
    for p in paths:
        if os.path.isfile(p) and p.endswith((".py", ".csv")):
            h.update(open(p, "rb").read())
        elif os.path.exists(p):
            st = os.stat(p)
            h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            h.update(f"{p}:missing".encode())
    return h.hexdigest()[:16]


def stale(out, fp):
    side = out + ".fp"
    return not (os.path.exists(out) and os.path.exists(side) and open(side).read().strip() == fp)


def mark(out, fp):
    with open(out + ".fp", "w") as f:
        f.write(fp)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ckpt_info(name, dirs, dataset):
    info = {"name": name, "dataset": dataset, "seeds": {}}
    for d, s in zip(dirs, seeds_of(dirs)):
        p = os.path.join(DELPHI_DIR, d, "ckpt.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        # 只数可训练参数：state_dict 里还有注意力掩码之类的 buffer（block_size² × 层数），会虚高
        sys.path.insert(0, DELPHI_DIR)
        from model import Delphi, DelphiConfig
        n_params = int(sum(x.numel() for x in Delphi(DelphiConfig(**ck["model_args"])).parameters()))
        info["seeds"][s] = {"ckpt": os.path.relpath(p, os.path.dirname(DELPHI_DIR)),
                            "iter_num": int(ck.get("iter_num", -1)),
                            "best_val_loss": float(ck.get("best_val_loss", float("nan"))),
                            "n_params": n_params, "model_args": ck["model_args"],
                            "config": {k: v for k, v in ck.get("config", {}).items()
                                       if isinstance(v, (int, float, str, bool, list, type(None)))}}
    st = {}
    for split in ("train", "val", "test"):
        f = os.path.join(DATA_DIR, dataset, f"{split}.bin")
        if os.path.exists(f):
            a = load_bin(dataset, split)
            n = np.unique(a[:, 0], return_counts=True)[1]
            st[split] = {"rows": int(len(a)), "persons": int(len(n)),
                         "tokens_per_person_median": float(np.median(n)), "max": int(n.max())}
    info["data"] = st
    meta = os.path.join(DATA_DIR, dataset, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta))
        info["meta"] = {k: m.get(k) for k in ("nodedup_prefixes", "no_global_dedup", "per_visit_events",
                                              "snapshot", "split_fractions", "split_counts")}
        info["meta"]["n_repeatable_tokens"] = len(m.get("repeatable_tokens_postshift", []) or [])
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    os.makedirs(os.path.join(RUN_DIR, "scores"), exist_ok=True)
    truth_csv = os.path.join(RUN_DIR, "truth_val.csv")
    # 真值读 snapshot 和交付版两份分词的 val.bin + 词表：分词重建之后必须重算
    fp_truth = fingerprint(os.path.join(HERE, "truth.py"), os.path.join(HERE, "common.py"),
                           *(os.path.join(DATA_DIR, ds, f) for ds in (TRUTH_DATASET, "rosmap") for f in ("val.bin", "labels.csv")))
    if stale(truth_csv, fp_truth):
        run([py, os.path.join(HERE, "truth.py"), "--split", "val"])
        mark(truth_csv, fp_truth)

    todo = DELPHI if not args.only else [m for m in DELPHI if m[0] in args.only.split(",")]
    for name, dirs, dataset, tok, sw, note in todo:
        scores = []
        for d, s in zip(dirs, seeds_of(dirs)):
            ck = os.path.join(DELPHI_DIR, d, "ckpt.pt")
            if not os.path.exists(ck):
                print(f"!! 缺 ckpt：{ck}，跳过这个 seed", flush=True)
                continue
            out = os.path.join(RUN_DIR, "scores", f"{name}__s{s}.csv")
            # 打分还依赖这份分词的数据（val.bin；pos 模型的对齐长度来自 train.bin）和上游 Delphi 的 get_batch / 模型代码
            fp = fingerprint(os.path.join(HERE, "score_delphi.py"), os.path.join(HERE, "common.py"), ck, truth_csv,
                             *(os.path.join(DATA_DIR, dataset, f) for f in ("val.bin", "train.bin", "labels.csv")),
                             os.path.join(DELPHI_DIR, "utils.py"), os.path.join(DELPHI_DIR, "model.py"))
            if args.rescore or stale(out, fp):
                run([py, os.path.join(HERE, "score_delphi.py"), "--ckpt", ck, "--dataset", dataset,
                     "--out", out, "--device", args.device])
                mark(out, fp)
            scores.append(f"s{s}={out}")
        if not scores:
            print(f"!! {name}: 一个 ckpt 都没有", flush=True)
            continue
        pk = os.path.join(RUN_DIR, "packages", name)
        os.makedirs(pk, exist_ok=True)
        cmd = [py, os.path.join(HERE, "evaluate.py"), "--name", name, "--outdir", pk, "--scores", *scores]
        if name in CAL_NOTES:
            cmd += ["--cal_note", CAL_NOTES[name]]
        run(cmd)
        with open(os.path.join(pk, "ckpt_info.json"), "w") as f:
            json.dump(ckpt_info(name, dirs, dataset), f, indent=1)


if __name__ == "__main__":
    main()
