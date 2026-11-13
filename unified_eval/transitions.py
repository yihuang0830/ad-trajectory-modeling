"""transitions.py -- 第三部分（疾病分期转换）的题目定义。改这里就是改第三部分的题，**不碰**第一部分。

为什么单独一个注册表、不往 common.ENDPOINTS 里加：
  第一部分的"4 终点平均 AUC"是 ENDPOINTS 上的平均，往里加题会悄悄改掉已发表的主数字；
  run_delphi_all 的缓存指纹也算上了 common.py / truth.py，动它们会让所有模型重新打分。
  这里的题、真值、打分、出图全部是新文件，旧产物逐位不变。

分期（和分词的分箱一一对应，tokenization 的切点）：
  MMSE      Normal ≥27 / Borderline 24–26 / Impaired <24      （MMSE_normal / _borderline / _impaired）
  全局认知  High ≥0 / Mid −1..0 / Low <−1                     （COGN_high / _mid / _low）
  外加两个吸收态：AD 诊断、死亡。

题目 = "首次越过某条分期线"（first passage），和第一部分的 MMSE<24、认知<−1 同一种定义：
  at-risk = 到站点为止**每次**测量都还没进入目标区间。
  为什么不用"从当前分期再往下走一期"：去重分词里每个 token 每人最多出现一次，一个人回到 Normal
  之后再掉到 Borderline，模型**结构上**发不出第二个 MMSE_borderline。用"首次越过"，每份分词都答得出来，
  比的是模型而不是分词能不能表达这道题（第一部分也是因为这个才这样定义的）。
  代价：好转（Borderline → Normal）不在题里 —— 去重分词同样表达不了它。
"""
from common import ENDPOINTS

# key -> (展示名, 家族 / 类型, 目标分箱 token 名或事件 token 名)
#   家族 mmse / cogn：threshold 题，目标 = 这些分箱里的任何一个
#   "event"：一次性事件（AD 诊断、死亡）
TRANS = {
    "mmse_lt27": ("MMSE first < 27 (Normal -> Borderline or worse)", "mmse", ("MMSE_borderline", "MMSE_impaired")),
    "mmse_lt24": ("MMSE first < 24 (-> Impaired)", "mmse", ("MMSE_impaired",)),
    "cogn_lt0": ("Global cognition first < 0 (High -> Mid or worse)", "cogn", ("COGN_mid", "COGN_low")),
    "cogn_ltm1": ("Global cognition first < -1 (-> Low)", "cogn", ("COGN_low",)),
    "ad": ("AD dementia", "event", ("AD_DX",)),
    "death": ("Death", "event", ("Death",)),
}
# 中文短名（README / LEADERBOARD 用）
TRANS_ZH = {"mmse_lt27": "MMSE 首次 <27", "mmse_lt24": "MMSE 首次 <24", "cogn_lt0": "认知首次 <0",
            "cogn_ltm1": "认知首次 <−1", "ad": "AD", "death": "死亡"}
# 和第一部分同一道题的对应关系（真值逐格对照）
SAME_AS_PART1 = {"mmse_lt24": "mmse", "cogn_ltm1": "cogn", "ad": "ad", "death": "death"}
assert set(SAME_AS_PART1.values()) == set(ENDPOINTS)

TRANS_HORIZONS = (1, 2, 3, 5, 10)        # 年；评估、出图用这 5 个，表只报 TABLE_H
TABLE_H = (2, 5, 10)
# 真值和训练标签用逐年的 1..10：逐年风险 h_1..h_10 每一年都直接受监督（只监督 1/2/3/5/10 的话，
# 第 4、6–9 年的风险只被单调性夹住、不可识别，第三部分扰动用的 10 年内平均无事件时间就会落在没监督过的年份上）。
TRUTH_HORIZONS = tuple(range(1, 11))
assert set(TRANS_HORIZONS) <= set(TRUTH_HORIZONS)
COGNITIVE = ("mmse_lt27", "mmse_lt24", "cogn_lt0", "cogn_ltm1")     # "认知分期"那 4 道

# 分期：0 = 最差 ... 2 = 最好，和 traj_truth.BIN 的箱序一致
STAGE_BINS = {"mmse": ("MMSE_impaired", "MMSE_borderline", "MMSE_normal"),
              "cogn": ("COGN_low", "COGN_mid", "COGN_high")}
STAGE_NAMES = {"mmse": ("Impaired", "Borderline", "Normal"), "cogn": ("Low", "Mid", "High")}

# 热图：行 = 站点时的当前 MMSE 分期（最近一次测量），列 = H 年内越过的线。
# 只有"这一行的人可能处在风险中"的格子才有定义：当前已经 <27 的人不会被问"首次 <27"。
HEATMAP_ROWS = {"mmse": (2, 1, 0), "cogn": (2, 1, 0)}             # Normal / Borderline / Impaired（自上而下）
HEATMAP_COLS = {"mmse": ("mmse_lt27", "mmse_lt24", "ad", "death"),
                "cogn": ("cogn_lt0", "cogn_ltm1", "ad", "death")}

# 热图每格的报告门槛（MIN_CELL 等）在 evaluate_transitions.py 里：这个文件进打分缓存的指纹，
# 只影响显示的常数放在这里会让全部 ckpt 白白重新打分。


def risk_col(q, H):
    return f"risk_{q}_{H}y"


def label_col(q, H):
    return f"y_{q}_{H}y"
