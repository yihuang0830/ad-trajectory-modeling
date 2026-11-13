"""模型注册表：打包盒名字 -> 在 RIS 上的 ckpt 目录（每个 seed 一个）+ 数据集 + 一句话说明。

名字只由模型参数组成：<框架>-<分词>[-<结构开关>][-split811]。
  分词     dedup / nodedup / fullvisit / snapshot（见 tokenization/README.md）
  结构开关 pos = 可学习绝对位置嵌入；aux = 多任务判别头；visitheads = 访视级时间/大小头；
           nojitter = 训练时不随机挪动背景 token 的年龄（其余同原配方）；
           visit1only = 注意力只看背景块 + 当前这次访视；staticonly = 只看人口学背景块
  split811 = 训练集 80% / test 10% / val 10%（val 与两分版是同一批人；train 少了 test 那批人）
"""
import glob
import os

DELPHI = [
    # name, ckpt dirs (seed 42 first), dataset, tokenization, switches, note
    ("delphi-dedup", ["Delphi-ROSMAP-gpubase", "Delphi-ROSMAP-gpubase-s43", "Delphi-ROSMAP-gpubase-s44"],
     "rosmap", "dedup", [], "交付版分词（两级去重：箱变了才发、每个 token 每人最多一次），在 H100 上重训的对照"),
    ("delphi-dedup-mps-legacy", ["Delphi-ROSMAP"], "rosmap", "dedup", [],
     "最早交付的 ckpt（Mac/MPS 训练）。同配置 H100 重训是 delphi-dedup；只作历史记录"),
    ("delphi-nodedup", ["Delphi-ROSMAP-nodedup", "Delphi-ROSMAP-nodedup-s43", "Delphi-ROSMAP-nodedup-s44"],
     "rosmap_nodedup", "nodedup", [], "MMSE 和 cogn_global 每次访视都发射（不去重），其余同 dedup"),
    ("delphi-nodedup-pos", ["Delphi-ROSMAP-nodedup-pos", "Delphi-ROSMAP-nodedup-pos-s43", "Delphi-ROSMAP-nodedup-pos-s44"],
     "rosmap_nodedup", "nodedup", ["pos"], "nodedup + 可学习绝对位置嵌入"),
    ("delphi-nodedup-aux", ["Delphi-ROSMAP-nodedup-aux"], "rosmap_nodedup", "nodedup", ["aux"],
     "nodedup + 多任务判别头（BCE，aux_lambda=0.3）；统一评估读的仍是语言模型头的速率"),
    ("delphi-nodedup-visitheads", ["Delphi-ROSMAP-nodedup-vh"], "rosmap_nodedup", "nodedup", ["visitheads"],
     "nodedup + 访视级时间头和访视大小头；lm_head 的 logit 不再是速率，校准图不可解释"),
    ("delphi-nodedup-split811", ["Delphi-ROSMAP-nodedup-3w"], "rosmap_nodedup_3w", "nodedup", [],
     "nodedup，split811 划分"),
    ("delphi-nodedup-pos-split811", ["Delphi-ROSMAP-nodedup-pos-3w", "Delphi-ROSMAP-nodedup-pos-3w-s43",
                                     "Delphi-ROSMAP-nodedup-pos-3w-s44"],
     "rosmap_nodedup_3w", "nodedup", ["pos"], "nodedup + 位置嵌入，split811 划分（figure 3 的主模型）"),
    ("delphi-nodedup-staticonly-split811", ["Delphi-ROSMAP-fig3-static", "Delphi-ROSMAP-fig3-static-s43",
                                            "Delphi-ROSMAP-fig3-static-s44"],
     "rosmap_nodedup_3w", "nodedup", ["staticonly"], "只看人口学的 transformer 基线（figure 3）"),
    ("delphi-fullvisit", ["Delphi-ROSMAP-fullvisit", "Delphi-ROSMAP-fullvisit-s43", "Delphi-ROSMAP-fullvisit-s44"],
     "rosmap_fullvisit", "fullvisit", [], "全部连续量每次访视都发射、关掉全局去重"),
    ("delphi-fullvisit-split811", ["Delphi-ROSMAP-fig3-fullvisit3w", "Delphi-ROSMAP-fig3-fullvisit3w-s43",
                                   "Delphi-ROSMAP-fig3-fullvisit3w-s44"],
     "rosmap_fullvisit_3w", "fullvisit", [], "fullvisit，split811 划分（figure 3 里'历史的价值'的上界）"),
    ("delphi-fullvisit-visit1only-split811", ["Delphi-ROSMAP-fig3-visit1", "Delphi-ROSMAP-fig3-visit1-s43",
                                              "Delphi-ROSMAP-fig3-visit1-s44"],
     "rosmap_fullvisit_3w", "fullvisit", ["visit1only"], "只看当前访视的 transformer 基线（figure 3）"),
    ("delphi-snapshot", ["Delphi-ROSMAP-snapshot", "Delphi-ROSMAP-snapshot-s43", "Delphi-ROSMAP-snapshot-s44"],
     "rosmap_snapshot", "snapshot", [], "每次访视发射完整状态快照（背景、用药状态、已患疾病、全部测量）"),
    ("delphi-snapshot-pos", ["Delphi-ROSMAP-snapshot-pos", "Delphi-ROSMAP-snapshot-pos-s43",
                             "Delphi-ROSMAP-snapshot-pos-s44"],
     "rosmap_snapshot", "snapshot", ["pos"], "snapshot + 位置嵌入"),
    ("delphi-snapshot-visit1only", ["Delphi-ROSMAP-snapshot-visit1", "Delphi-ROSMAP-snapshot-visit1-s43",
                                    "Delphi-ROSMAP-snapshot-visit1-s44"],
     "rosmap_snapshot", "snapshot", ["visit1only"], "snapshot，注意力只看当前访视（当前访视 = 完整状态）"),
    ("delphi-snapshot-staticonly", ["Delphi-ROSMAP-snapshot-static", "Delphi-ROSMAP-snapshot-static-s43",
                                    "Delphi-ROSMAP-snapshot-static-s44"],
     "rosmap_snapshot", "snapshot", ["staticonly"], "snapshot，只看人口学背景块"),
    # 新加的模型一律加在最后：mc_granularity.py 按这个顺序共用一串随机数，插在中间会让后面所有模型的 MC 颗粒度数字变一点
    # （2026-10-16 加 delphi-dedup-nojitter 时就踩过：先插在中间，后面的模型都变了，挪到最后才逐位不变）。
    ("delphi-dedup-nojitter", ["Delphi-ROSMAP-nojitter", "Delphi-ROSMAP-nojitter-s43", "Delphi-ROSMAP-nojitter-s44",
                               "Delphi-ROSMAP-nojitter-s45", "Delphi-ROSMAP-nojitter-s46"],
     "rosmap", "dedup", ["nojitter"],
     "delphi-dedup 的配方，只关掉背景 token 的年龄随机挪动（lifestyle augmentation），5 个 seed；"
     "APOE 诊断里用的就是它（APOE_DIAGNOSIS.md）"),
    # NACC 预训练（2026-10-22 加，models_code/nacc_delphi.py）：NACC 按 dedup 的规则分词，nojitter 配方在 NACC 上训 40,000 步
    ("delphi-dedup-nojitter-nacc", ["Delphi-ROSMAP-nojitter-nacc", "Delphi-ROSMAP-nojitter-nacc-s43",
                                    "Delphi-ROSMAP-nojitter-nacc-s44", "Delphi-ROSMAP-nojitter-nacc-s45",
                                    "Delphi-ROSMAP-nojitter-nacc-s46"],
     "rosmap", "dedup", ["nojitter", "nacc"],
     "delphi-dedup-nojitter 的配方，只是从 NACC 预训练的权重开始（同一个 seed），对照就是 delphi-dedup-nojitter"),
    ("delphi-dedup-nojitter-nacc-zs", ["Delphi-NACC-zs", "Delphi-NACC-zs-s43", "Delphi-NACC-zs-s44", "Delphi-NACC-zs-s45",
                                       "Delphi-NACC-zs-s46"],
     "rosmap_naccview", "dedup", ["nojitter", "nacc", "zeroshot"],
     "只在 NACC 上训练、不微调，直接读 ROSMAP（ROSMAP 的记录去掉 NACC 没有的 token：子研究、饮酒、跛行、化验、睡眠、炎症、病理）"),
]

# 两个参照基线：只进 README 的表（fig1 从 2026-10-29 起不画参照线，各模型平等）
# 不用 snapshot 的两个基线当尺子：snapshot 每次访视先重发背景块（不计损失），"两次访视之间"的位置
# 在训练里从来不受监督，统一评估读分的那个位置对它们没有定义（见 model_zoo/README.md）。
REFERENCES = [("Delphi: demographics only", "delphi-nodedup-staticonly-split811"),
              ("Delphi: current visit only", "delphi-fullvisit-visit1only-split811")]

CAL_NOTES = {"delphi-nodedup-visitheads": "visit_heads: lm_head logits are not rates here -> calibration not interpretable"}

# 本地仓库只留这 4 个 Delphi 打包盒（2026-10-20 整理）：预测最好的 dedup、修好 APOE 的 nojitter、最像真实的模拟器 nodedup、
# 只看人口学的参照。其余 13 个的聚合结果冻结在 models_self/archive/<名字>/，make_package 不再往本地拷（RIS 上的 _runs/packages 照常全量）。
LOCAL_DELPHI = ("delphi-dedup", "delphi-dedup-nojitter", "delphi-nodedup", "delphi-nodedup-staticonly-split811")
DELPHI_SIM = "delphi-dedup-sim"       # delphi-dedup 的 rollout 读法（rollout_landmarks.py），9 模型图里的 Delphi

# 本地打包盒放在哪（相对仓库根目录；2026-10-29 整理，2026-11-01 DELPHI_SIM 挪进 representative/、delphi/ 和 baselines/ 合并成 models_self/）：
#   representative/<名字>/        9 模型图里的 9 个：REPRESENTATIVE（图里画的 rollout 版）+ DELPHI_SIM
#   models_self/<名字>/           其余所有模型：LOCAL_DELPHI 和其余模型
#   models_self/archive/<名字>/   冻结的 13 个 Delphi
#   models_not_usable/<名字>/     不可用的 36 个（NOT_USABLE；2026-11-05 从 models_self/ 挪出来）
#   reports/                      跨模型的表、图、汇总 JSON（REPORT_FILES；2026-11-02 从仓库根目录挪进来）
SELF_DIR = "models_self"
ARCHIVE_DIR = "models_self/archive"
REP_DIR = "representative"
NOT_USABLE_DIR = "models_not_usable"
REPORT_DIR = "reports"
REPORT_FILES = frozenset((
    "APOE_DIAGNOSIS.md", "APOE_PROBE.json", "MODELS.md", "MODELS.png", "MODELS_calibration.png",
    "MODELS_delphi_upstream.png", "MODELS_nacc.png", "MODELS_sim.png", "MODELS_timing.png", "MODELS_traj.png",
    "EVAL_PROTOCOL.md", "LEADERBOARD.md", "leaderboard.png", "MC_GRANULARITY.json", "OVERVIEW.png", "OVERVIEW_all.png",
    "PERTURB_V2_SUMMARY.json", "SENSITIVITY.json", "SHAP_SUMMARY.json", "SHAP_SUMMARY.png", "VAL_VS_TEST.json",
    "select_model.json"))
REPRESENTATIVE = ("xgb-l2c-sim", "minimalrnn", "transformer", "fnn-l2c-sim", "ar-lr", "cox-cur-sim", "ethos", "bdsi-sim")
# 不可用的模型（用户 2026-11-05 定）：不能生成未来，只回答"H 年内会不会"——逐题逐年直接给一个概率（models.py 里 future="direct"）
# 或给一条生存曲线（future="survival"）。训练目标就是评估题本身，和要从模拟出来的未来里数答案的模型不是一回事，所以不参与评估
# （MODELS.md 的表 1–3、总图）。除了 6 个 NACC 变体（-nacc、-nacc-zs），每个都有一个能生成未来的版本留在评估里
# （-sim、ar-*；-x5 的是同一结构的自回归版 transformer / grud / tlstm）。
# compare.py 断言这个名单 = models.py 里 future 是 direct / survival 的全部。
NOT_USABLE = (
    "lr-demo", "lr-cur", "lr-l2c", "rf-l2c", "xgb-l2c", "lgbm-l2c", "tabpfn-l2c", "fnn-l2c",          # 逐题逐年的分类器
    "bdsi", "anu-adri", "caide",                                                                    # 固定权重的临床评分
    "cox-cogn", "cox-super", "cox-cur", "cox-l2c", "rsf-l2c", "xgbcox-l2c", "survivalboost-l2c",    # 生存模型
    "deepsurv-l2c", "deephit-l2c", "logistichazard-l2c", "pmf-l2c", "mtlr-l2c", "fg-cur", "fg-l2c",
    "xgbaft-l2c", "lgbmcox-l2c",
    "transformer-x5", "transformer-nacc", "transformer-nacc-zs", "grud-x5", "grud-nacc", "grud-nacc-zs",   # 序列网络的
    "tlstm-x5", "tlstm-nacc", "tlstm-nacc-zs")                                                              # 直接作答版（NACC 实验）


def pkg_dir(name):
    """打包盒名字 -> 本地目录（相对仓库根目录）。"""
    if name == DELPHI_SIM:
        return f"{REP_DIR}/{name}"
    if name.startswith("delphi") and name not in LOCAL_DELPHI:
        return f"{ARCHIVE_DIR}/{name}"
    if name in NOT_USABLE:
        return f"{NOT_USABLE_DIR}/{name}"
    return f"{REP_DIR}/{name}" if name in REPRESENTATIVE else f"{SELF_DIR}/{name}"


def zoo_pkgs(zoo):
    """Delphi 的打包盒（绝对路径，按名字排序）：models_self/ 下的 delphi-*，加上放在 representative/ 的
    DELPHI_SIM。LEADERBOARD 和各处只比 Delphi 的汇总都扫这里。"""
    ds = [d.rstrip("/") for d in glob.glob(os.path.join(zoo, SELF_DIR, "delphi-*", ""))]
    return sorted(ds + [os.path.join(zoo, pkg_dir(DELPHI_SIM))], key=os.path.basename)


def zoo_files(zoo, fname):
    """zoo_pkgs 里有 fname 的那些打包盒：[<打包盒>/fname]。"""
    return [p for p in (os.path.join(d, fname) for d in zoo_pkgs(zoo)) if os.path.exists(p)]


def rel_link(src_pkg, target):
    """从打包盒 src_pkg 的 README 指向 target 的相对链接；target = 另一个打包盒的名字（-> 它的 README），
    或仓库里的文件（含 '.' 的，如 EVAL_PROTOCOL.md；REPORT_FILES 里的自动指到 reports/）。"""
    t = (f"{REPORT_DIR}/{target}" if target in REPORT_FILES else target) if "." in target else f"{pkg_dir(target)}/README.md"
    return os.path.relpath(t, pkg_dir(src_pkg))


def report_md(text):
    """reports/ 里的 markdown：生成时链接按仓库根目录写，这里改成从 reports/ 出发的相对路径
    （REPORT_FILES 里的就在同一目录，其余加 ../）。"""
    import re

    def fix(m):
        t = m.group(1)
        if "://" in t or t.startswith(("#", "../")):
            return m.group(0)
        return "](" + (t if t.partition("#")[0] in REPORT_FILES else "../" + t) + ")"
    return re.sub(r"\]\(([^)\s]+)\)", fix, text)


def folder_md(text, folder):
    """report_md 的一般版：markdown 放在仓库的 folder/ 下，按仓库根目录写的链接改成从 folder/ 出发的相对路径
    （REPORT_FILES 里的指到 reports/）。"""
    import re

    def fix(m):
        t = m.group(1)
        if "://" in t or t.startswith(("#", "../")):
            return m.group(0)
        path, sep, frag = t.partition("#")
        return "](" + os.path.relpath(f"{REPORT_DIR}/{path}" if path in REPORT_FILES else path, folder) + sep + frag + ")"
    return re.sub(r"\]\(([^)\s]+)\)", fix, text)


def seeds_of(dirs):
    """ckpt 目录名 -> seed：结尾 -s<数字> 就是那个 seed，否则是 42（第一个目录）。"""
    import re
    out = []
    for d in dirs:
        m = re.search(r"-s(\d+)$", d)
        out.append(int(m.group(1)) if m else 42)
    return out
