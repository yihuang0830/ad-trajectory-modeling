"""
make_visit_sizes.py -- 从 train.bin 统计"每次访视发射几个临床 token"，供 engine 的访视采样用。

2026-11-01 重组时这个脚本随 figure2_eval/ 一起删掉了，而 engine 找不到 visit_sizes.npy 时会退回逐 token 采样：
Delphi rollout 的风险因此大幅下降（和已发表的结果对不上）。
这里按原样恢复（figure2_eval/make_visit_sizes.py @ babeceb^），只把路径改成重组后的布局：
数据在 $AD_BASE/delphi/data/<dataset>，默认输出在本目录（= engine 退回去找的"仓库根"，radc_delphi/ 的上一级）。

为什么需要它：模型的时间头是拿**访视间隔**训练的（mask_ties 把 dt 改成"到上一个非同龄 token 的时间"），
但生成时一次等待只发一个 token，而真实访视携带好几个。所以 rollout 的临床 token 率远低于观测。

只数 train split，因为它是**训练分布**的统计量。用 val 去估这个分布等于把评估数据喂回预测。

只数**临床** token（排除 statics 与注入的 No-event 标记）：statics 全部挤在基线那一访、且模型
从不预测它们；No-event 是合成标记，不是一次就诊。

    python make_visit_sizes.py                          # delphi/data/rosmap -> models_code/visit_sizes.npy
    python make_visit_sizes.py --dataset rosmap_nodedup # -> delphi/data/rosmap_nodedup/visit_sizes.npy

访视大小是**分词的**统计量，不是仓库的，所以非默认数据集写到该数据集目录里，engine 也是先在
data_dir 找、再退回仓库根（见 Engine.__init__）。不去重那份分词每次访视多带 2 个认知 token，
用交付版的分布去补发会静默地把访视采样调小。
"""
import os
import sys
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import DATA_DIR  # noqa: E402
from radc_delphi import vocab as V, batching as Bt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rosmap")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()   # 不叫 a：下面的循环用 a 当 age
    ddir = os.path.join(DATA_DIR, args.dataset)
    OUT = args.out or (os.path.join(HERE, "visit_sizes.npy") if args.dataset == "rosmap"
                       else os.path.join(ddir, "visit_sizes.npy"))
    # 家族/ignore 一律从**这份** labels.csv 解析，不用 live 模块（两份分词并存时会拿错）
    res = V.resolve_csv(os.path.join(ddir, "labels.csv"))
    skip = set(res.IGNORE_TOKENS) | {res.NO_EVENT}
    d = np.fromfile(os.path.join(ddir, "train.bin"), dtype=np.uint32).reshape(-1, 3)
    p2i = Bt.get_p2i(d)
    sizes, base_sizes = [], []
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        ages, toks = Bt.patient_stream(d, s, n)
        by = collections.defaultdict(int)
        for a, t in zip(ages, toks):
            if int(t) in skip:
                continue
            by[float(a)] += 1
        if not by:
            continue
        ks = sorted(by)
        # 基线那一访单独拿出来：它在 rollout 里是**前缀**，不是生成出来的，所以不该进分布。
        base_sizes.append(by[ks[0]])
        sizes += [by[a] for a in ks[1:]]
    sizes = np.array(sizes, dtype=np.int64)
    base_sizes = np.array(base_sizes, dtype=np.int64)
    np.save(OUT, sizes)

    c = collections.Counter(sizes.tolist())
    print(f"dataset={args.dataset}")
    print(f"train split：{len(sizes)} 次**非基线**访视（另有 {len(base_sizes)} 次基线访视，已排除）")
    print(f"  非基线访视的临床 token 数: 均值 {sizes.mean():.2f}  中位 {np.median(sizes):.0f}  "
          f"p95 {np.percentile(sizes, 95):.0f}  max {sizes.max()}")
    print(f"  （基线访视均值 {base_sizes.mean():.2f}，明显更大——基线一次做全套，"
          f"所以混进去会高估后续访视）")
    print("  分布: " + "  ".join(f"{k}:{v}" for k, v in sorted(c.items())[:12]) + " ...")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
