"""统一评估的公共定义：终点、时间窗、路径。所有模型共用，改这里就是改题目。

时间单位：.bin 里的年龄是**天**（uint32），访视在每个人自己的 365 天网格上
（tokenization/build.py 的 G(ab, k) = round(ab*365.25) + k*365），AD 诊断和死亡也落在同一张网格上。
所以 H 年窗口的终点取 t + H*365（不是 365.25）：恰好在第 H 年那次访视上发生的事件算在窗口内。
"""
import os

import numpy as np
import pandas as pd

AD_BASE = os.environ.get(
    "AD_BASE", "/path/to/project")
DELPHI_DIR = os.path.join(AD_BASE, "delphi")
DATA_DIR = os.path.join(DELPHI_DIR, "data")
RUN_DIR = os.environ.get("ZOO_RUN_DIR", os.path.join(AD_BASE, "model_zoo", "_runs"))  # 人级产物只留在 RIS

HORIZONS = (1, 2, 3, 5)          # 年；图只画 MAIN_H
MAIN_H = 2
GRID = 365                       # 天/年（数据网格）
DAYS_PER_YEAR = 365.25           # Delphi 速率换算风险用

# 终点：key -> (展示名, labels.csv 里的 token 名, 类型)
#   event    ：一次性事件，at-risk = 站点时还没发生
#   threshold：第一次测到某个分箱，at-risk = 到站点为止**每次**测量都不在这个箱里
ENDPOINTS = {
    "ad":    ("AD dementia",         "AD_DX",         "event"),
    "death": ("Death",               "Death",         "event"),
    "mmse":  ("MMSE < 24",           "MMSE_impaired", "threshold"),
    "cogn":  ("Global cognition < -1", "COGN_low",    "threshold"),
}
# threshold 终点的同族分箱：用来判断"这次访视测了没有"
FAMILY = {"mmse": ("MMSE_impaired", "MMSE_borderline", "MMSE_normal"),
          "cogn": ("COGN_low", "COGN_mid", "COGN_high")}

TRUTH_DATASET = "rosmap_snapshot"   # 每次访视都发射完整状态 -> 访视集合和每次的测量都在里面


def label_names(dataset="rosmap"):
    return pd.read_csv(os.path.join(DATA_DIR, dataset, "labels.csv"))["event_name"].tolist()


def pre_id(names):
    """token 名 -> 盘上（pre-shift）id。labels.csv 第 i 行（0 起）= post-shift id i = pre-shift i-1。"""
    return {n: i - 1 for i, n in enumerate(names)}


def post_id(names):
    return {n: i for i, n in enumerate(names)}


def load_bin(dataset, split="val"):
    p = os.path.join(DATA_DIR, dataset, f"{split}.bin")
    return np.fromfile(p, dtype=np.uint32).reshape(-1, 3).astype(np.int64)


def risk_cols():
    return [f"risk_{e}_{h}y" for e in ENDPOINTS for h in HORIZONS]


def label_cols():
    return [f"y_{e}_{h}y" for e in ENDPOINTS for h in HORIZONS]
