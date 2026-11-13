"""perturbations.py -- fig4 的扰动清单（按临床含义定义，所有模型共用）。

每个扰动：名字、展示名、谁有资格（perturb_cohort_val.csv 的列）、预设方向（看结果之前写死，见
EVAL_PROTOCOL.md 第 7 节；+1 = 风险应上升，不在字典里 = 不预设、不计分）。
各模型自己把扰动翻译成自己的输入：Delphi 在 rollout_delphi.apply 里改 token，其余模型在
perturb_features.apply 里改特征。
"""
from registry import DELPHI

TOKENIZATION = {m[0]: m[3] for m in DELPHI}

# 方向测试的 7 个扰动
MAIN = [
    ("apoe", "APOE e4 0->1", {"ad": 1, "mmse": 1, "cogn": 1}),
    ("mmse25", "MMSE now ->25", {"ad": 1, "death": 1, "mmse": 1, "cogn": 1}),
    ("cogn", "cognition now ->-0.5", {"ad": 1, "death": 1, "mmse": 1, "cogn": 1}),
    ("stroke", "stroke now", {"ad": 1, "death": 1, "mmse": 1, "cogn": 1}),
    ("age5", "age +5y", {"ad": 1, "death": 1, "mmse": 1, "cogn": 1}),
    ("male", "female->male", {"death": 1}),
    # 2026-10-08 修订：不预设（条件效应可反号，见 EVAL_PROTOCOL.md 修订记录）
    ("educ", "education ->10y", {}),
]
# 剂量反应：当前 MMSE 设成 29 / 27 / 25 / 23（25 与 MAIN 共用）
DOSE = ["mmse29", "mmse27", "mmse25", "mmse23"]
DOSE_VALUE = {"mmse29": 29, "mmse27": 27, "mmse25": 25, "mmse23": 23}
# 早期 vs 近期：同一个"新增卒中"放在第一次访视 vs 当前访视
EARLY = ("stroke_early", "stroke")

PERTS = ["orig"] + [p for p, _, _ in MAIN] + ["mmse29", "mmse27", "mmse23", "stroke_early"]
ELIG = {"orig": "elig_age", "apoe": "elig_apoe", "mmse25": "elig_mmse", "mmse29": "elig_mmse",
        "mmse27": "elig_mmse", "mmse23": "elig_mmse", "cogn": "elig_cogn", "stroke": "elig_stroke",
        "stroke_early": "elig_stroke", "age5": "elig_age", "male": "elig_sex", "educ": "elig_educ"}
LABEL = {p: lab for p, lab, _ in MAIN}
EXPECTED = {p: exp for p, _, exp in MAIN}


def stroke_early_stale(name, pert):
    """这个包的"早期卒中"结果是不是 2026-10-12 修正之前、翻译错了的（是 -> 不能用，等 RIS 重跑）。
    修正前一律只在第一次访视插一个 STROKE；snapshot 分词每次访视都重发已患疾病，应该第一次访视到站点之间每次都插。
    修正后的运行在 run_info.json / perturb.json 里记 stroke_early_rule，snapshot 模型缺这个字段 = 修正前的结果。
    早期和近期按构造相同的模型（看不到卒中 token）不受影响：插在哪里输出都一样。"""
    return (TOKENIZATION.get(name) == "snapshot" and not pert.get("stroke_early_rule")
            and not pert.get("early_equals_recent_by_construction"))
