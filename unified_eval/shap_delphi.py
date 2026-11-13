"""shap_delphi.py -- 上游 Delphi 的 SHAP 分析搬到 ROSMAP 的 Delphi ckpt 上：哪些输入 token 推动了第三部分
每道转换题（transitions.TRANS）的速率。目的是给反事实扰动（perturbations.py）挑靶子 —— 扰一个模型几乎
不看的 token，rollout 里看不到效应是意料之中，说明不了模型"不懂"这个临床因素。

    python shap_delphi.py run   --model delphi-dedup --seed 42[,43,44]   # 人级 SHAP（只留 RIS）
    python shap_delphi.py smoke --model delphi-dedup --seed 42 --n 6      # 回归检查 + 计时 + 总量估算
    python shap_delphi.py sens  --model delphi-dedup --seed 42 --n 20     # 上游默认 max_evals=500 的截断误差
    python shap_delphi.py aggregate                                       # 汇总 json + 图（只有聚合数字）
  进程数 --workers 默认 = SLURM_CPUS_PER_TASK；每个进程单线程。

方法（照上游 Delphi 的 SHAP notebook，只把"读什么"换成统一直接读出）：
  * 解释器：shap.Explainer(f, shap.maskers.Text(utils.shap_custom_tokenizer, output_type='str',
    mask_token='10000', collapse_mask_token=False)) -> PartitionExplainer（Owen 值；层次 = Text masker 把相邻
    token 两两合并出的近似平衡二叉树）。输入 = 这个人的 token id 用空格连起来的一个字符串；output_type='str'
    时 masker 交给 f 的是 (n, M) 的 token 字符串数组，被遮的位置是 '10000'。
  * 玩家 = 这个人在**该模型自己的 val.bin** 里年龄 <= tL 的全部真实 token（tL = 扰动队列的站点，第 4 次访视）。
    不含查询 token、哑目标、padding，也不含 get_batch 插的 no-event：后者在固定的 5 年年龄网格上、截止于哑目标
    的年龄，遮不遮玩家都不变 —— 等价于上游"no-event（value == 1）永远保留"。
  * 遮挡 = 把 token 连同它的年龄一起删掉。性别例外：删掉性别不是"没有信息"，而是训练里从没出现过的状态，
    上游的做法是翻转（Female <-> Male，年龄不动）。这里按 token id 翻，不靠"第 0 个位置"（ROSMAP 的性别和
    背景块一起在第一次访视上，不一定排第一）。
  * f = 统一直接读出（score_delphi.query_logits，和 scores_trans 同一个函数、同一种排布）：保留的 token +
    tL+1 天的 no-event 查询 + tL+2 天的哑目标，读查询位置的 logit。6 个输出（TRANS 的顺序）= 每道题目标 token
    的 logit 的 logsumexp = log Σλ（每天的 log 速率）。所以 phi 的单位是 log 速率比：exp(phi) = 这个 token 让
    速率乘了多少倍（Owen 意义下的平均边际贡献），base = 全遮（只剩翻转后的性别）时的读出。
  * max_evals = 2·M·M + 10（M = 玩家数）：足够把整棵树展开完（平衡树约 2(M²−1)/3 + 2 次）。展开完的 Owen 值
    是精确的，和优先队列的处理顺序无关；每个人都核对实际评估次数 < max_evals（没被截断）。上游默认的
    max_evals=500 在 M 稍大时会截断（没展开的节点的贡献平摊给它的叶子），sens 子命令量化这个误差。
  * 年龄不是 token，SHAP 看不见"变老"。另报"年龄进程"：全部年龄（含查询、哑目标）+1826 天的读出减原读出，
    和 rollout_delphi.apply('age5') 是同一个扰动，单位同 phi，和 token 的 SHAP 放在同一把尺子上。

聚合（每道题只在站点时处在风险中的人里算）：token 名 × 题（携带者 >= MIN_CARRIERS 才报）、临床分组、
MMSE / 认知 token 的滞后（往回数几次访视），每个 ckpt 一份，模型一级是 seed 平均（同一份数据 -> 玩家逐个对齐，
先对 phi 取平均再聚合）。

产物（人级的只留在 RIS）：
  $ZOO_RUN_DIR/shap/<model>/phi_<seed>.npz        人级：玩家 token / 年龄 / phi、base、f(全)、f(+5 岁)
  $ZOO_RUN_DIR/shap/<model>/run_s<seed>.json      聚合：回归检查、评估次数、耗时
  $ZOO_RUN_DIR/shap/<model>/shap_s<seed>.json     聚合：这个 ckpt 自己的 token / 分组 / 滞后表
  $ZOO_RUN_DIR/shap/_checks/{smoke,sens500}_*.json 聚合：冒烟测试、截断误差
  $ZOO_RUN_DIR/packages/<model>/shap.json                     交付模型（seed 平均）
  $ZOO_RUN_DIR/shap/delphi-dedup-nojitter/shap.json             无抖动重训（对照，不是注册表里的包）
  $ZOO_RUN_DIR/packages/SHAP_SUMMARY.json                     跨模型共识、seed / 模型间一致性、抖动对照（图在本地画）
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DELPHI_DIR, RUN_DIR, label_names, load_bin, post_id  # noqa: E402
from registry import DELPHI, seeds_of  # noqa: E402
from transitions import TRANS  # noqa: E402
import score_delphi as SD  # noqa: E402   （torch 在这里才 import：numpy / pandas 必须先进来）

QS = list(TRANS)                                  # 6 个输出 = 6 道题，TRANS 的顺序
READ = ("AD_DX", "Death", "MMSE_impaired", "MMSE_borderline", "COGN_low", "COGN_mid")
assert set(READ) == {t for _, _, toks in TRANS.values() for t in toks}
MASK = "10000"                                    # 上游的 mask token：词表只有 130 个，不会和真 id 撞
AGE5 = 1826                                       # 年龄 +5 岁，整天数（同 rollout_delphi.SHIFT_5Y）
BATCH = 128                                       # explainer 的 batch_size（上游同）
MIN_CARRIERS = 5                                  # token / 分组 / 滞后格子少于这么多人就不报（聚合产物要离开 RIS）
N_BOOT = 2000
SHAP_DIR = os.path.join(RUN_DIR, "shap")
CHECK_DIR = os.path.join(SHAP_DIR, "_checks")
PKG_DIR = os.path.join(RUN_DIR, "packages")
PACKAGES = ("delphi-dedup", "delphi-nodedup", "delphi-fullvisit")    # SHAP_SUMMARY 的共识只用这三个（2026-10-14 定的）
NOJIT = "delphi-dedup-nojitter"
SHORT = {"mmse_lt27": "MMSE first <27", "mmse_lt24": "MMSE first <24", "cogn_lt0": "Cognition first <0",
         "cogn_ltm1": "Cognition first <-1", "ad": "AD dementia", "death": "Death"}
GROUPS = ("current_mmse", "past_mmse", "current_cogn", "past_cogn", "apoe", "sex", "education", "other_static",
          "comorbidity", "medication", "vascular_metabolic", "sleep_inflammation", "AD_DX", "other")
CONTRAST_GROUPS = ("apoe", "education", "sex")    # 抖动只挪背景块：看它改没改模型对这些静态 token 的依赖


SKIP = {"delphi-nodedup-visitheads": "lm_head 的 logit 不是速率，SHAP 要拆的那个读数没有定义（和扰动 v2 一样）"}


def runs():
    """label -> [(seed, ckpt 目录, 数据集)]。原来的三个模型保持 2026-10-14 的 seed（SHAP_SUMMARY 的共识只用它们）；
    2026-10-15 起其余 Delphi 也算（每个包一张 SHAP 图），用注册表里所有 seed。无抖动重训不在注册表里，单独列。"""
    reg = {m[0]: m for m in DELPHI}
    keep = {"delphi-dedup": (42, 43, 44), "delphi-nodedup": (42,), "delphi-fullvisit": (42,)}
    out = {}
    for name, (_, dirs, ds, *_rest) in reg.items():
        if name in SKIP:
            continue
        out[name] = [(s, d, ds) for d, s in zip(dirs, seeds_of(dirs)) if s in keep.get(name, seeds_of(dirs))]
    if NOJIT not in out:                          # 2026-10-16 起它在注册表里（5 个 seed），这里只是老的兜底
        out[NOJIT] = [(42, "Delphi-ROSMAP-nojitter", "rosmap")]
    return out


def ckpt_spec(model, seed):
    for s, d, ds in runs()[model]:
        if s == seed:
            return os.path.join(DELPHI_DIR, d, "ckpt.pt"), ds
    raise KeyError(f"{model} 没有 seed {seed}")


def crn_seed(pid):
    return int(pid) % 2_147_483_647               # 同 run_rollout_delphi.crn_seed


def full_evals(M):
    return 2 * M * M + 10


def family_of(name):
    """测量 / 变量家族（同一个量的不同分箱）：SBP_normal / SBP_elevated / SBP_high -> SBP。"""
    return "SEX" if name in ("Female", "Male") else name.split("_")[0]


def group_of(name, at_tL):
    """临床分组。MMSE / 认知按"是不是站点那次访视的测量"分成当前 / 既往。"""
    if name.startswith("MMSE_"):
        return "current_mmse" if at_tL else "past_mmse"
    if name.startswith("COGN_"):
        return "current_cogn" if at_tL else "past_cogn"
    if name.startswith("APOE_"):
        return "apoe"
    if name in ("Female", "Male"):
        return "sex"
    if name.startswith("EDU_"):
        return "education"
    if name.startswith(("STUDY_", "RACE_", "HISP_", "SMOKING_", "ALCOHOL_")):
        return "other_static"
    if name.endswith(("_PREVALENT", "_ONSET")) or name in ("STROKE", "DEPRESSION"):
        return "comorbidity"
    if name.endswith(("_ON", "_OFF")):
        return "medication"
    if name.startswith(("SBP_", "DBP_", "BMI_", "GLU_", "HBA1C_", "HDL_", "LDL_", "GFR_")):
        return "vascular_metabolic"
    if name.startswith(("PSQI_", "APNEA_", "CRP_", "IL6_", "TNFA_")):
        return "sleep_inflammation"
    if name == "AD_DX":
        return "AD_DX"
    return "other"


# ----------------------------------------------------------------------------- 数据

def cohort():
    c = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"))
    return c[["pid", "tL"]].astype(np.int64).reset_index(drop=True)


def persons(dataset, coh):
    """队列里每人的玩家：该模型自己 val.bin 里年龄 <= tL 的真实 token，保持盘上顺序（年龄序 + 同龄时的发射序，
    get_batch 的稳定排序也保持这个顺序 -> 不遮时和标准打分路径的排布逐位相同）。"""
    arr = load_bin(dataset, "val")
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    pids, first, cnt = np.unique(arr[:, 0], return_index=True, return_counts=True)
    where = {int(p): (int(f), int(c)) for p, f, c in zip(pids, first, cnt)}
    out = []
    for pid, tL in zip(coh.pid.to_numpy(), coh.tL.to_numpy()):
        f, c = where[int(pid)]
        h = arr[f: f + c]
        assert (np.diff(h[:, 1]) >= 0).all(), "val.bin 里有人的 token 不是按年龄排的"
        h = h[h[:, 1] <= tL]
        out.append({"pid": int(pid), "tL": int(tL), "age": h[:, 1].astype(np.int64), "tok": h[:, 2] + 1})
    return out


# ----------------------------------------------------------------------------- 统一读出（f）

def make_ctx(model, ck, dataset):
    cfg = ck.get("config", {})
    assert not ck["model_args"].get("visit_heads", False), "visit_heads 的 logit 不是速率"
    post = post_id(label_names(dataset))
    lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
    block = int(ck["model_args"]["block_size"])
    # pos 模型：和 score_delphi / score_delphi_trans / perturb_v2 同一个对齐长度（训练 batch 的 x 长度中位数）。
    # query_logits 逐行对齐到 max(pos_len, 自身长度)，结果和分批方式无关，所以 query() 里复制凑长度的做法照样成立；
    # checks() 的"不遮时 = 第三部分打分表"回归会核对这一点。
    pos_len = (SD.pos_align_length(dataset, cfg, block, lr)[0] if ck["model_args"].get("pos_embedding", False) else None)
    return {"model": model, "block": block, "lr": lr, "pos_len": pos_len,
            "tok_post": [post[n] for n in READ],
            "qidx": [[READ.index(t) for t in TRANS[q][2]] for q in QS],
            "flip": {post["Female"]: post["Male"], post["Male"]: post["Female"]}}


def readout(C, keep, tok, age, t):
    """批量统一读出。keep (n, M) bool = 每条记录保留哪些玩家；tok (n, M) 模型 id（性别可能已翻转）；
    age (M,) 天；t = 站点（天）。返回 (n, 6) 的 log Σλ（每天），列 = QS。"""
    n, M = keep.shape
    jj, ii = np.nonzero(keep)                     # 行优先：同一条记录的玩家连续、保持原顺序
    j = np.arange(n)
    z = np.zeros(n, np.int64)
    data = np.concatenate([np.column_stack([jj, age[ii], tok[jj, ii] - 1]),
                           np.column_stack([j, np.full(n, t + 1), z]),        # 查询：盘上 0 = No event
                           np.column_stack([j, np.full(n, t + 2), z])]).astype(np.int64)   # 哑目标
    key = np.concatenate([ii, np.full(n, M), np.full(n, M + 1)])
    data = data[np.lexsort((key, data[:, 0]))]
    return query(C, data, keep.sum(1).astype(np.int64), np.full(n, t + 1, np.int64))


def query(C, data, hist, q_age):
    """score_delphi.query_logits + 每道题的 logsumexp（和 score_delphi_trans 同一个 token 顺序、同一个
    logaddexp.reduce）。data = (记录号, 年龄, 盘上 token)，记录号 0..n-1 连续。
    get_batch 取窗口时把起点夹在 [0, len(data) − block_size − 1]：整批数据比一个窗口还短时（f(全遮)、f(不遮)
    这种单条调用）上界是负数，负下标绕回数组尾部，同一条记录被读两遍（query_logits 的断言会炸）。把整批复制
    几份凑够长度、只取第一份的输出：不带位置编码的模型，一条记录的输出和同批里还有谁无关（cut_batch 只裁
    整列 padding；score_delphi 里验证过逐位相同）。"""
    n = len(hist)
    r = max(1, -(-(C["block"] + 1) // len(data)))
    if r > 1:
        data = np.concatenate([data + np.array([k * n, 0, 0], np.int64) for k in range(r)])
        hist, q_age = np.tile(hist, r), np.tile(q_age, r)
    lg = SD.query_logits(C["model"], data, hist, q_age, C["block"], C["tok_post"], "cpu", batch_size=256,
                         lifestyle_range=C["lr"], pos_len=C.get("pos_len"))[:n]
    return np.stack([np.logaddexp.reduce(lg[:, ix], axis=1) for ix in C["qidx"]], 1)


_W = {}                                           # 工作进程的全局状态：模型、explainer、当前这个人


def _init(ckpt, dataset):
    import torch
    torch.set_num_threads(1)                      # 并行在人上（进程），不在矩阵乘里
    import shap
    from utils import shap_custom_tokenizer       # 上游的分词器（$B/delphi/utils.py；score_delphi 已把目录放进 sys.path）
    model, ck = SD.load_model(ckpt, "cpu")
    _W["C"] = make_ctx(model, ck, dataset)
    masker = shap.maskers.Text(shap_custom_tokenizer, output_type="str", mask_token=MASK, collapse_mask_token=False)
    _W["masker"] = masker
    _W["explainer"] = shap.Explainer(_f, masker, output_names=QS)
    assert type(_W["explainer"]).__name__ == "PartitionExplainer", type(_W["explainer"])
    _W["shap_version"] = shap.__version__


def _f(ps):
    """shap 调的模型：ps (n, M) 的 token 字符串，被遮的位置是 '10000'。"""
    P, C = _W["cur"], _W["C"]
    ps = np.asarray(ps)
    if ps.ndim == 1:
        ps = ps[None, :]
    assert ps.shape[1] == len(P["tok"]), (ps.shape, len(P["tok"]))
    keep = ps != MASK
    r, c = np.nonzero(keep)
    assert (ps[r, c] == P["tok_str"][c]).all(), "masker 交回来的 token 和玩家对不上"
    tok = np.repeat(P["tok"][None, :], len(ps), 0)
    tok[~keep & P["is_sex"][None, :]] = P["sex_flipped"]      # 性别：遮 = 翻转，不删
    keep = keep | P["is_sex"][None, :]
    _W["n_evals"] += len(ps)
    return readout(C, keep, tok, P["age"], P["tL"])


def _explain(job):
    pid, tL, age, tok, max_evals, rseed = job
    C = _W["C"]
    tok = np.asarray(tok, np.int64)
    is_sex = np.isin(tok, list(C["flip"]))
    # 去重 / 不去重的分词每人一个性别 token；snapshot 每次访视都重发背景，一个人有好几个（必须是同一个值）。
    # 每个出现都是一个 player，遮掉 = 把这一个翻转（和只有一个时同一个做法）。
    assert is_sex.sum() >= 1 and len(set(tok[is_sex].tolist())) == 1, "每人应有性别 token，而且都一样"
    P = {"tL": int(tL), "age": np.asarray(age, np.int64), "tok": tok, "is_sex": is_sex,
         "tok_str": np.array([str(int(x)) for x in tok]), "sex_flipped": C["flip"][int(tok[is_sex][0])]}
    _W["cur"], _W["n_evals"] = P, 0
    s = " ".join(P["tok_str"])
    # 优先队列的并列用 np.random.randn 打破：展开完时结果和顺序无关（只差浮点求和顺序），固定种子后逐位可复现
    np.random.seed(int(rseed))
    t0 = time.time()
    ex = _W["explainer"]([s], max_evals=int(max_evals), batch_size=BATCH, silent=True)
    sec = time.time() - t0
    M = len(tok)
    phi = np.asarray(ex.values[0], np.float64)
    base = np.asarray(ex.base_values[0], np.float64)
    assert phi.shape == (M, len(QS)) and len(ex.data[0]) == M, (phi.shape, M)
    one = np.ones((1, M), bool)
    f_full = readout(C, one, tok[None, :], P["age"], P["tL"])[0]
    f_age5 = readout(C, one, tok[None, :], P["age"] + AGE5, P["tL"] + AGE5)[0]
    return {"pid": int(pid), "phi": phi, "base": base, "f_full": f_full, "f_age5": f_age5,
            "n_evals": int(_W["n_evals"]), "max_evals": int(max_evals), "sec": sec}


def _predict_job(job):
    """完整展开要评估多少次（不跑模型）：根访问 1 次，往下每层翻倍（零上下文 / 一上下文各走一遍），每次展开
    2 次评估，再加 f(全遮)、f(不遮) 两次。冒烟测试里和实测次数对一遍，再拿它估总量。"""
    s = job
    cl = np.asarray(_W["masker"].clustering(s))
    M = len(cl) + 1
    depth = np.zeros(2 * M - 1, np.int64)
    total = 2
    for i in range(len(cl) - 1, -1, -1):          # 父节点的行号总在子节点之后
        d = int(depth[M + i])
        assert cl[i, 2] >= 0, "Text masker 给了负距离：这个节点不会被展开"
        total += 2 * 2 ** d
        depth[int(cl[i, 0])] = d + 1
        depth[int(cl[i, 1])] = d + 1
    return total


def explain_all(ckpt, dataset, people, workers, max_evals_fn, rseed_fn, tag=""):
    jobs = [(p["pid"], p["tL"], p["age"], p["tok"], max_evals_fn(len(p["tok"])), rseed_fn(p["pid"])) for p in people]
    jobs.sort(key=lambda j: -len(j[3]))           # 长的先跑：池子收尾时不剩一个大个子
    res = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, min(workers, len(jobs))), initializer=_init,
                                      initargs=(ckpt, dataset)) as pool:
        for k, r in enumerate(pool.imap_unordered(_explain, jobs, chunksize=1), 1):
            res[r["pid"]] = r
            if k % 25 == 0 or k == len(jobs):
                print(f"  {tag} {k}/{len(jobs)} persons  {time.time() - t0:.0f}s", flush=True)
    return [res[p["pid"]] for p in people], time.time() - t0


def standard_readout(ckpt, dataset, coh):
    """标准打分路径（score_delphi.pseudo_records + query_logits，整批读，和 scores_trans 一样）。"""
    model, ck = SD.load_model(ckpt, "cpu")
    C = make_ctx(model, ck, dataset)
    lm = pd.DataFrame({"pid": coh.pid.to_numpy(), "t": coh.tL.to_numpy()})
    data, hist = SD.pseudo_records(load_bin(dataset, "val"), lm)
    return query(C, data, hist, lm.t.to_numpy() + 1)


def checks(model, seed, ckpt, dataset, coh, R):
    """回归检查（全部是聚合数字）。R = explain_all 的结果，顺序同 coh。"""
    f_full = np.stack([r["f_full"] for r in R])
    base = np.stack([r["base"] for r in R])
    eff = max(float(np.abs(r["phi"].sum(0) + r["base"] - r["f_full"]).max()) for r in R)
    std = standard_readout(ckpt, dataset, coh)
    out = {"vs_standard_path_max_abs": float(np.abs(std - f_full).max()),
           "efficiency_max_abs_err": eff,
           "full_expansion_all": bool(all(r["n_evals"] < r["max_evals"] - 2 for r in R)),
           "n_evals": {"median": float(np.median([r["n_evals"] for r in R])), "max": int(max(r["n_evals"] for r in R))},
           "players_M": {"median": float(np.median([len(r["phi"]) for r in R])), "max": int(max(len(r["phi"]) for r in R))},
           "base_minus_full_mean": {q: float((base - f_full)[:, k].mean()) for k, q in enumerate(QS)}}
    p = os.path.join(RUN_DIR, "scores_trans", f"{model}__s{seed}.csv")
    if os.path.exists(p):
        s = pd.read_csv(p, usecols=["pid", "t"] + [f"rank_{q}" for q in QS])
        m = pd.DataFrame({"pid": coh.pid.to_numpy(), "t": coh.tL.to_numpy()}).merge(
            s, on=["pid", "t"], how="left", validate="one_to_one")
        rk = m[[f"rank_{q}" for q in QS]].to_numpy()
        assert not np.isnan(rk).any(), "有站点不在 scores_trans 里"
        d = np.abs(rk - f_full)
        out["vs_scores_trans_max_abs"] = {q: float(d[:, k].max()) for k, q in enumerate(QS)}
        out["vs_scores_trans_ok"] = bool(d.max() < 1e-4)
    else:
        out["vs_scores_trans_max_abs"] = None       # 不是注册表里的包（无抖动重训）：只有标准路径这一项
    return out


# ----------------------------------------------------------------------------- 子命令：run / smoke / sens

def _save_npz(path, people, R, ckpt, dataset, seed):
    M = np.array([len(p["tok"]) for p in people])
    np.savez_compressed(
        path, pid=np.array([p["pid"] for p in people]), tL=np.array([p["tL"] for p in people]),
        off=np.concatenate([[0], np.cumsum(M)]), tok=np.concatenate([p["tok"] for p in people]),
        age=np.concatenate([p["age"] for p in people]), phi=np.concatenate([r["phi"] for r in R]),
        base=np.stack([r["base"] for r in R]), f_full=np.stack([r["f_full"] for r in R]),
        f_age5=np.stack([r["f_age5"] for r in R]), n_evals=np.array([r["n_evals"] for r in R]),
        max_evals=np.array([r["max_evals"] for r in R]), questions=np.array(QS), ckpt=np.array(ckpt),
        dataset=np.array(dataset), seed=np.array(seed))


def cmd_run(a):
    for seed in [int(s) for s in str(a.seed).split(",")]:
        ckpt, ds = ckpt_spec(a.model, seed)
        coh = cohort()
        if a.limit:
            coh = coh.iloc[: a.limit].reset_index(drop=True)
        people = persons(ds, coh)
        print(f"{a.model} s{seed}: {len(people)} persons, players M median {np.median([len(p['tok']) for p in people]):.0f}"
              f"  ckpt={ckpt}", flush=True)
        R, wall = explain_all(ckpt, ds, people, a.workers, full_evals, crn_seed, tag=f"{a.model} s{seed}")
        info = {"model": a.model, "seed": seed, "ckpt": ckpt, "dataset": ds, "n_persons": len(people),
                "workers": a.workers, "wall_min": wall / 60, "cpu_min_sum": sum(r["sec"] for r in R) / 60,
                "checks": checks(a.model, seed, ckpt, ds, coh, R)}
        out = os.path.join(SHAP_DIR, a.model)
        os.makedirs(out, exist_ok=True)
        sfx = f"_limit{a.limit}" if a.limit else ""
        _save_npz(os.path.join(out, f"phi_{seed}{sfx}.npz"), people, R, ckpt, ds, seed)
        json.dump(info, open(os.path.join(out, f"run_s{seed}{sfx}.json"), "w"), indent=1)
        print(json.dumps(info, indent=1), flush=True)


def cmd_smoke(a):
    ckpt, ds = ckpt_spec(a.model, a.seed)
    coh = cohort().iloc[: a.n].reset_index(drop=True)
    people = persons(ds, coh)
    R, wall = explain_all(ckpt, ds, people, a.workers, full_evals, crn_seed, tag="smoke")
    ck = checks(a.model, a.seed, ckpt, ds, coh, R)
    # 确定性：同样的 3 个人在新的进程池里再跑一遍（同种子 -> 应逐位相同）；换一个并列种子再跑一遍
    # （展开完时只剩浮点求和顺序的差别）
    sub, Rs = people[:3], R[:3]
    R2, _ = explain_all(ckpt, ds, sub, 3, full_evals, crn_seed, tag="rerun")
    R3, _ = explain_all(ckpt, ds, sub, 3, full_evals, lambda p: crn_seed(p) + 12345, tag="reseed")
    det = max(float(np.abs(x["phi"] - y["phi"]).max()) for x, y in zip(Rs, R2))
    reseed = max(float(np.abs(x["phi"] - y["phi"]).max()) for x, y in zip(Rs, R3))
    # 评估次数预测 vs 实测，再估全部 ckpt 的总量（按 M² 和序列长度粗略缩放）
    _init(ckpt, ds)
    pred = [_predict_job(" ".join(str(int(x)) for x in p["tok"])) for p in people]
    sec_per_eval = sum(r["sec"] for r in R) / sum(r["n_evals"] for r in R)
    plan = {}
    full = cohort()
    for ds2 in sorted({d for v in runs().values() for _, _, d in v}):
        pp = persons(ds2, full)
        ev = [_predict_job(" ".join(str(int(x)) for x in p["tok"])) for p in pp]
        Lm = np.mean([len(p["tok"]) for p in pp]) + 18      # + 查询 / 哑目标 / no-event（粗略）
        L0 = np.mean([len(p["tok"]) for p in people]) + 18
        plan[ds2] = {"evals_total": int(sum(ev)), "M_median": float(np.median([len(p["tok"]) for p in pp])),
                     "M_max": int(max(len(p["tok"]) for p in pp)),
                     "est_cpu_min_per_ckpt": float(sum(ev) * sec_per_eval * Lm / L0 / 60)}
    out = {"model": a.model, "seed": a.seed, "n_persons": len(people), "wall_sec": wall,
           "per_person": [{"M": len(r["phi"]), "n_evals": r["n_evals"], "predicted_evals": pv,
                           "max_evals": r["max_evals"], "sec": round(r["sec"], 2)} for r, pv in zip(R, pred)],
           "predicted_equals_actual": bool(all(r["n_evals"] == pv for r, pv in zip(R, pred))),
           "sec_per_eval": sec_per_eval, "checks": ck,
           "determinism_rerun_max_abs": det, "determinism_reseed_max_abs": reseed, "plan": plan,
           "shap_version": _W.get("shap_version")}
    os.makedirs(CHECK_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(CHECK_DIR, f"smoke_{a.model}_s{a.seed}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_sens(a):
    """上游默认 max_evals=500（np.random.seed(0)）vs 完整展开：同一批人、同一个 ckpt。"""
    ckpt, ds = ckpt_spec(a.model, a.seed)
    coh = cohort()
    np.random.seed(0)
    pick = np.random.choice(np.sort(coh.pid.to_numpy()), a.n, replace=False)
    coh = coh[coh.pid.isin(pick)].reset_index(drop=True)
    people = persons(ds, coh)
    Rf, _ = explain_all(ckpt, ds, people, a.workers, full_evals, crn_seed, tag="full")
    Rt, _ = explain_all(ckpt, ds, people, a.workers, lambda M: 500, lambda p: 0, tag="max_evals=500")
    names = label_names(ds)
    trunc = [rt["n_evals"] < rf["n_evals"] for rt, rf in zip(Rt, Rf)]     # 少评估了 = 被截断（精确判据）
    out = {"model": a.model, "seed": a.seed, "n_persons": len(people),
           "M": {"median": float(np.median([len(p["tok"]) for p in people])), "max": int(max(len(p["tok"]) for p in people))},
           "n_truncated": int(sum(trunc)),
           "full_evals": {"median": float(np.median([r["n_evals"] for r in Rf])), "max": int(max(r["n_evals"] for r in Rf))},
           "efficiency_500_max_abs_err": max(float(np.abs(r["phi"].sum(0) + r["base"] - r["f_full"]).max()) for r in Rt),
           "questions": {}}
    for k, q in enumerate(QS):
        d = np.concatenate([rt["phi"][:, k] - rf["phi"][:, k] for rt, rf in zip(Rt, Rf)])
        ref = np.concatenate([rf["phi"][:, k] for rf in Rf])
        rho = [_spearman(rt["phi"][:, k], rf["phi"][:, k]) for rt, rf, tr in zip(Rt, Rf, trunc) if tr]
        rho = [x for x in rho if x is not None]
        top1 = [int(np.argmax(np.abs(rt["phi"][:, k])) == np.argmax(np.abs(rf["phi"][:, k]))) for rt, rf in zip(Rt, Rf)]
        # 分组层面（聚合用的就是这一层）：人内按组求和后的相对 L1 误差
        gd = gref = 0.0
        for p, rt, rf in zip(people, Rt, Rf):
            grp = np.array([group_of(names[t], ag == p["tL"]) for t, ag in zip(p["tok"], p["age"])])
            for g in np.unique(grp):
                m = grp == g
                gd += abs(rt["phi"][m, k].sum() - rf["phi"][m, k].sum())
                gref += abs(rf["phi"][m, k].sum())
        out["questions"][q] = {"max_abs_err": float(np.abs(d).max()), "mean_abs_err": float(np.abs(d).mean()),
                               "rel_l1_err": float(np.abs(d).sum() / max(np.abs(ref).sum(), 1e-300)),
                               "group_rel_l1_err": float(gd / max(gref, 1e-300)),
                               "spearman_truncated_median": float(np.median(rho)) if rho else None,
                               "top1_agree_frac": float(np.mean(top1))}
    os.makedirs(CHECK_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(CHECK_DIR, f"sens500_{a.model}_s{a.seed}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


# ----------------------------------------------------------------------------- 聚合

def load_phi(model, seed):
    z = np.load(os.path.join(SHAP_DIR, model, f"phi_{seed}.npz"))
    return {k: z[k] for k in z.files}


def at_risk(Z):
    """站点 tL 时每道题的 at-risk（按人）：首次越线题 = 到站点为止每次测量都还没进入目标区间；AD = 还没诊断；死亡 = 全部。"""
    tt = pd.read_csv(os.path.join(RUN_DIR, "truth_trans_val.csv"),
                     usecols=["pid", "t", "mmse_worst_so_far", "cogn_worst_so_far"])
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    lm = pd.DataFrame({"pid": Z["pid"], "t": Z["tL"]}).merge(tt, on=["pid", "t"], how="left", validate="one_to_one")
    assert not lm.isna().any().any(), "有站点不在 truth_trans_val.csv 里"
    mw, cw = lm.mmse_worst_so_far.to_numpy(), lm.cogn_worst_so_far.to_numpy()
    ad = [traj[str(int(p))]["ad"] for p in Z["pid"]]
    no_ad = np.array([(x is None) or (x > t) for x, t in zip(ad, Z["tL"])])
    return {"mmse_lt27": mw == 2, "mmse_lt24": mw >= 1, "cogn_lt0": cw == 2, "cogn_ltm1": cw >= 1,
            "ad": no_ad, "death": np.ones(len(Z["pid"]), bool)}


def boot_ci(v, seed=0):
    v = np.asarray(v, float)
    idx = np.random.RandomState(seed).randint(0, len(v), (N_BOOT, len(v)))
    m = v[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _r(x, k=5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def summarize(Z, names, R):
    """一个 ckpt（或 seed 平均）的人级数组 -> questions 字典（全部是聚合数字，格子 < MIN_CARRIERS 人不报）。"""
    n = len(Z["pid"])
    M = np.diff(Z["off"])
    person = np.repeat(np.arange(n), M)
    tLp = np.repeat(Z["tL"], M)
    name = np.array(names, dtype=object)[Z["tok"]]
    grp = np.array([group_of(nm, a == t) for nm, a, t in zip(name, Z["age"], tLp)])
    lag = np.rint((tLp - Z["age"]) / 365).astype(int)
    fam = np.array(["mmse" if nm.startswith("MMSE_") else "cogn" if nm.startswith("COGN_") else "" for nm in name])
    famname = np.array([family_of(nm) for nm in name])
    out = {}
    for k, q in enumerate(QS):
        r = R[q]
        nr = int(r.sum())
        sel = r[person]
        ph = Z["phi"][sel, k]
        df = pd.DataFrame({"p": person[sel], "tok": name[sel], "phi": ph, "abs": np.abs(ph)})
        per = df.groupby(["tok", "p"]).agg(phi=("phi", "sum"), absphi=("abs", "sum"), n=("phi", "size")).reset_index()
        toks = []
        for t, g in per.groupby("tok"):
            if len(g) < MIN_CARRIERS:
                continue
            v = g.phi.to_numpy()
            ev = np.exp(v)
            toks.append({"token": t, "n_occ": int(g.n.sum()), "n_persons": int(len(g)),
                         "mean_phi": _r(df.phi[df.tok == t].mean()), "mean_person_phi": _r(v.mean()),
                         "mean_abs_phi_all": _r(g.absphi.sum() / nr), "frac_pos": _r((v > 0).mean(), 3),
                         "exp_phi_median": _r(np.median(ev), 4), "exp_phi_iqr": [_r(np.percentile(ev, 25), 4),
                                                                                _r(np.percentile(ev, 75), 4)]})
        toks.sort(key=lambda d: -d["mean_abs_phi_all"])
        G = pd.DataFrame({"p": person[sel], "g": grp[sel], "phi": ph}).groupby(["g", "p"]).phi.sum().reset_index()
        tot = float(np.abs(G.phi).sum())
        groups = []
        for gname in GROUPS:
            v = G.phi[G.g == gname].to_numpy()
            ok = len(v) >= MIN_CARRIERS
            groups.append({"group": gname, "n_carriers": int(len(v)) if ok else (f"<{MIN_CARRIERS}" if len(v) else 0),
                           "mean_abs_all": _r(np.abs(v).sum() / nr) if ok else None,
                           "share_abs": _r(np.abs(v).sum() / tot, 4) if ok and tot > 0 else None,
                           "mean_signed_carriers": _r(v.mean()) if ok else None,
                           "frac_pos": _r((v > 0).mean(), 3) if ok else None})
        groups.sort(key=lambda d: -(d["mean_abs_all"] or 0))
        # 家族视图：遮挡 = 删掉 token，所以 phi 里混着"这个量测过"（在场）和"测出来是多少"（取值）两件事。
        # 全遮的基线是训练里没见过的空历史，速率极低，几乎每个 token 的"在场"都把速率往上推（正常值也是）。
        # 同一家族各分箱的 mean_person_phi 之差（bin_spread）才是取值的效应（描述性：携带不同分箱的是不同的人）；
        # 设计反事实扰动应该看这一项 —— 扰动是在家族内换分箱，不是删 token。
        tokd = {d["token"]: d for d in toks}
        F = pd.DataFrame({"p": person[sel], "f": famname[sel], "phi": ph}).groupby(["f", "p"]).phi.sum().reset_index()
        fams = []
        for f, g in F.groupby("f"):
            v = g.phi.to_numpy()
            if len(v) < MIN_CARRIERS:
                continue
            bins = [{"token": t, "n_persons": tokd[t]["n_persons"], "mean_person_phi": tokd[t]["mean_person_phi"]}
                    for t in names if t in tokd and family_of(t) == f]
            bv = [b["mean_person_phi"] for b in bins]
            fams.append({"family": f, "n_carriers": int(len(v)), "mean_abs_all": _r(np.abs(v).sum() / nr),
                         "mean_signed_carriers": _r(v.mean()), "bins": bins,
                         "bin_spread": _r(max(bv) - min(bv)) if len(bv) >= 2 else None})
        fams.sort(key=lambda d: -d["mean_abs_all"])
        lagd = {}
        for f in ("mmse", "cogn"):
            m = sel & (fam == f)
            L = pd.DataFrame({"p": person[m], "lag": lag[m], "tok": name[m], "phi": Z["phi"][m, k]})
            rows = []
            for lv, g in L.groupby("lag"):
                if g.p.nunique() < MIN_CARRIERS:
                    continue
                bt = {t: {"n_occ": int(len(h)), "mean_phi": _r(h.phi.mean())}
                      for t, h in g.groupby("tok") if h.p.nunique() >= MIN_CARRIERS}
                rows.append({"visits_back": int(lv), "n_occ": int(len(g)), "n_persons": int(g.p.nunique()),
                             "mean_phi": _r(g.phi.mean()), "mean_abs_phi": _r(g.phi.abs().mean()), "by_token": bt})
            lagd[f] = rows
        d5 = (Z["f_age5"][r, k] - Z["f_full"][r, k])
        out[q] = {"display": TRANS[q][0], "n_at_risk": nr,
                  "base_value_mean": _r(Z["base"][r, k].mean()), "f_full_mean": _r(Z["f_full"][r, k].mean()),
                  "sum_phi_mean": _r((Z["f_full"][r, k] - Z["base"][r, k]).mean()),
                  "tokens": toks, "groups": groups, "families": fams, "lag": lagd,
                  "age5": {"mean": _r(d5.mean()), "ci": [_r(x) for x in boot_ci(d5)], "median": _r(np.median(d5)),
                           "rate_ratio_mean": _r(np.exp(d5.mean()), 4)}}
    return out


def _imp(Q, key="tokens", field="mean_abs_phi_all", name="token"):
    return {d[name]: d[field] for d in Q[key] if d[field] is not None}


def _agree(tabs, labels, key="tokens", field="mean_abs_phi_all", name="token"):
    """两两 Spearman（共同出现的 token / 分组上），返回 {pairs, mean, n_min}。"""
    pairs = {}
    for i in range(len(tabs)):
        for j in range(i + 1, len(tabs)):
            a, b = _imp(tabs[i], key, field, name), _imp(tabs[j], key, field, name)
            common = sorted(set(a) & set(b))
            pairs[f"{labels[i]}~{labels[j]}"] = {"rho": _spearman([a[t] for t in common], [b[t] for t in common]),
                                                 "n": len(common)}
    vals = [v["rho"] for v in pairs.values() if v["rho"] is not None]
    return {"pairs": pairs, "mean": _r(np.mean(vals), 4) if vals else None}


def aggregate_model(model, names_ds):
    seeds = [s for s, _, _ in runs()[model] if os.path.exists(os.path.join(SHAP_DIR, model, f"phi_{s}.npz"))]
    if not seeds:
        return None
    Zs = [load_phi(model, s) for s in seeds]
    for Z in Zs[1:]:
        for k in ("pid", "tL", "off", "tok", "age"):
            assert np.array_equal(Z[k], Zs[0][k]), f"{model}：seed 之间玩家对不上（{k}）"
    Zm = dict(Zs[0])
    for k in ("phi", "base", "f_full", "f_age5"):
        Zm[k] = np.mean([Z[k] for Z in Zs], 0)     # 同一份数据 -> 玩家逐个对齐：先平均 phi 再聚合
    names = label_names(names_ds)
    R = at_risk(Zm)
    per_seed = {}
    for s, Z in zip(seeds, Zs):
        per_seed[s] = summarize(Z, names, R)
        info = json.load(open(os.path.join(SHAP_DIR, model, f"run_s{s}.json")))
        json.dump({"name": model, "seed": s, "ckpt": str(Z["ckpt"]), "dataset": str(Z["dataset"]),
                   "n_persons": int(len(Z["pid"])), "regression_checks": info["checks"], "questions": per_seed[s]},
                  open(os.path.join(SHAP_DIR, model, f"shap_s{s}.json"), "w"), indent=1)
    Q = summarize(Zm, names, R) if len(seeds) > 1 else per_seed[seeds[0]]
    reg = {}
    for s in seeds:
        reg[f"s{s}"] = json.load(open(os.path.join(SHAP_DIR, model, f"run_s{s}.json")))["checks"]
    for f in sorted(os.listdir(CHECK_DIR)) if os.path.isdir(CHECK_DIR) else []:
        if f.startswith(("smoke_", "sens500_")) and f"_{model}_s" in f:
            c = json.load(open(os.path.join(CHECK_DIR, f)))
            if f.startswith("smoke_"):
                reg[f[:-5]] = {k: c[k] for k in ("n_persons", "determinism_rerun_max_abs", "determinism_reseed_max_abs",
                                                 "predicted_equals_actual", "sec_per_eval")} | {"checks": c["checks"]}
            else:
                reg[f[:-5]] = c
    sa_detail = {}
    if len(seeds) > 1:
        for q in QS:
            tabs = [per_seed[s][q] for s in seeds]
            sa_detail[q] = {"tokens": _agree(tabs, [f"s{s}" for s in seeds]),
                            "tokens_signed": _agree(tabs, [f"s{s}" for s in seeds], "tokens", "mean_person_phi", "token"),
                            "groups": _agree(tabs, [f"s{s}" for s in seeds], "groups", "mean_abs_all", "group")}
    J = {"name": model,
         "checkpoints": [{"seed": s, "ckpt": str(Z["ckpt"]), "dataset": str(Z["dataset"])} for s, Z in zip(seeds, Zs)],
         "n_persons": int(len(Zm["pid"])),
         "settings": {"explainer": "shap.Explainer(f, shap.maskers.Text(utils.shap_custom_tokenizer, output_type='str', "
                                   "mask_token='10000', collapse_mask_token=False)) -> PartitionExplainer (Owen values)",
                      "max_evals_rule": "2*M*M+10 per person (M = players): full Owen expansion, exact; "
                                        "checked n_evals < max_evals; batch_size=128",
                      "masking": "masked token removed together with its age; the sex token is flipped "
                                 "(Female<->Male) instead of removed",
                      "players": "all real tokens with age <= tL from the model's own val.bin "
                                 "(tL = 4th visit, perturbation cohort)",
                      "readout": "score_delphi.query_logits: No-event query at tL+1 d, dummy target at tL+2 d; "
                                 "output = logsumexp of the question's target-token logits = log rate per day",
                      "units": "phi in log rate ratio (exp(phi) = rate ratio); age5 = f(all ages +1826 d) - f(orig)",
                      "seed_average": "phi averaged over seeds per player, then aggregated" if len(seeds) > 1 else None,
                      "tie_break": "np.random.seed(pid % 2147483647) before each explanation",
                      "min_carriers": MIN_CARRIERS,
                      "token_fields": "n_occ, n_persons (carriers), mean_phi (per occurrence), mean_person_phi "
                                      "(sum within person, mean over carriers), mean_abs_phi_all (sum |phi| within "
                                      "person, mean over ALL at-risk persons = importance), frac_pos (carriers with "
                                      "person phi > 0), exp_phi median / IQR over carriers",
                      "group_fields": "sum within person; mean_abs_all = mean |.| over all at-risk persons; "
                                      "mean_signed_carriers = mean over carriers"},
         "regression_checks": reg,
         "questions": Q,
         "seed_agreement": {q: sa_detail[q]["tokens"]["mean"] for q in sa_detail},
         "seed_agreement_detail": sa_detail,
         "per_seed_top": {f"s{s}": {q: [{"token": d["token"], "mean_abs_phi_all": d["mean_abs_phi_all"],
                                         "mean_person_phi": d["mean_person_phi"]} for d in per_seed[s][q]["tokens"][:15]]
                                    for q in QS} for s in seeds} if len(seeds) > 1 else {},
         "_per_seed_groups": {s: {q: per_seed[s][q]["groups"] for q in QS} for s in seeds}}
    return J


def _dump(J, path):
    J = {k: v for k, v in J.items() if not k.startswith("_")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(J, open(path, "w"), indent=1)


def cmd_aggregate(a):
    J = {}
    for model, rr in runs().items():
        J[model] = aggregate_model(model, rr[0][2])
        if J[model] is None:
            print(f"  {model}: 没有 phi_*.npz，跳过", flush=True)
            continue
        d = os.path.join(PKG_DIR, model)          # 每个 Delphi 包都写自己的 shap.json（每个包一张 SHAP 图；含不抖动的那个）
        os.makedirs(d, exist_ok=True)
        _dump(J[model], os.path.join(d, "shap.json"))
        print(f"  -> {d}/shap.json", flush=True)
    have = [m for m in PACKAGES if J.get(m)]
    if not have:
        return
    S = {"models": have, "seeds": {m: [c["seed"] for c in J[m]["checkpoints"]] for m in have},
         "contrast_model": NOJIT if J.get(NOJIT) else None,
         "units": "log rate ratio per day-rate (exp = rate ratio); importance = mean_abs_phi_all",
         "consensus_rule": "rank by the mean over models of mean_abs_phi_all (a token below the carrier floor in a "
                           "model counts as 0 there); model-level values are seed-averaged",
         "questions": {}}
    for q in QS:
        tok = {}
        for m in have:
            for d in J[m]["questions"][q]["tokens"]:
                tok.setdefault(d["token"], {})[m] = d
        cons = []
        for t, per in tok.items():
            imp = [per[m]["mean_abs_phi_all"] if m in per else 0.0 for m in have]
            sg = [per[m]["mean_person_phi"] for m in per]
            cons.append({"token": t, "importance_mean": _r(np.mean(imp)), "mean_person_phi_mean": _r(np.mean(sg)),
                         "rate_ratio_carriers": _r(np.exp(np.mean(sg)), 4), "n_models": len(per),
                         "sign_agree": bool(len(set(np.sign(sg))) == 1),
                         "per_model": {m: {"mean_abs_phi_all": per[m]["mean_abs_phi_all"],
                                           "mean_person_phi": per[m]["mean_person_phi"],
                                           "n_persons": per[m]["n_persons"]} for m in per}})
        cons.sort(key=lambda d: -d["importance_mean"])
        for i, d in enumerate(cons, 1):
            d["rank"] = i
        gc = {}
        for g in GROUPS:
            per = {m: next(d for d in J[m]["questions"][q]["groups"] if d["group"] == g) for m in have}
            imp = [per[m]["mean_abs_all"] or 0.0 for m in have]
            if not any(imp):
                continue
            gc[g] = {"importance_mean": _r(np.mean(imp)),
                     "per_model": {m: {k: per[m][k] for k in ("mean_abs_all", "mean_signed_carriers", "n_carriers",
                                                                  "frac_pos")} for m in have}}
        gl = sorted(gc, key=lambda g: -gc[g]["importance_mean"])
        for i, g in enumerate(gl, 1):
            gc[g]["rank"] = i
        dd = J["delphi-dedup"] if "delphi-dedup" in have else None
        contrast = {}
        if dd and J.get(NOJIT):
            for g in CONTRAST_GROUPS:
                pick = lambda gs: next(d for d in gs if d["group"] == g)  # noqa: E731
                contrast[g] = {
                    "dedup_seedavg": {k: pick(dd["questions"][q]["groups"])[k]
                                      for k in ("mean_abs_all", "mean_signed_carriers", "n_carriers")},
                    "dedup_per_seed": {f"s{s}": pick(gs[q])["mean_abs_all"] for s, gs in dd["_per_seed_groups"].items()},
                    # 单 seed 对单 seed 才公平：seed 平均会让 seed 间不一致的 phi 相互抵消、|phi| 变小，
                    # 无抖动版只有一个 seed。图里画的是这一项（dedup 各 seed 单独算的均值，须 = seed 间范围）
                    "dedup_per_seed_mean": _r(np.mean([v for v in (pick(gs[q])["mean_abs_all"]
                                                                   for gs in dd["_per_seed_groups"].values())
                                                       if v is not None] or [np.nan])),
                    "nojitter": {k: pick(J[NOJIT]["questions"][q]["groups"])[k]
                                 for k in ("mean_abs_all", "mean_signed_carriers", "n_carriers")}}
        S["questions"][q] = {
            "display": TRANS[q][0], "n_at_risk": {m: J[m]["questions"][q]["n_at_risk"] for m in have},
            "consensus_tokens": cons[:25], "consensus_groups": [dict(group=g, **gc[g]) for g in gl],
            "age5": {m: J[m]["questions"][q]["age5"] for m in have + ([NOJIT] if J.get(NOJIT) else [])},
            "seed_agreement_dedup": (dd["seed_agreement_detail"].get(q, {}).get("tokens") if dd else None)
            or {"pairs": {}, "mean": None},
            "model_agreement": _agree([J[m]["questions"][q] for m in have], have),
            "model_agreement_groups": _agree([J[m]["questions"][q] for m in have], have, "groups", "mean_abs_all", "group"),
            "model_agreement_signed": _agree([J[m]["questions"][q] for m in have], have, "tokens", "mean_person_phi", "token"),
            "consensus_families": _consensus_families(J, have, q),
            "nojitter_contrast": contrast}
    S["regression_checks"] = {m: J[m]["regression_checks"] for m in have + ([NOJIT] if J.get(NOJIT) else [])}
    os.makedirs(PKG_DIR, exist_ok=True)
    json.dump(S, open(os.path.join(PKG_DIR, "SHAP_SUMMARY.json"), "w"), indent=1)
    print(f"  -> {PKG_DIR}/SHAP_SUMMARY.json", flush=True)
    for q in QS:
        Sq = S["questions"][q]
        print(f"[{q}] top tokens: " + ", ".join(f"{d['token']}({d['importance_mean']:.3f})" for d in Sq["consensus_tokens"][:10]))
        print(f"[{q}] groups: " + ", ".join(f"{d['group']}({d['importance_mean']:.3f})" for d in Sq["consensus_groups"]))
        print(f"[{q}] families (importance / bin spread): " + ", ".join(
            f"{d['family']}({d['importance_mean']:.3f}/{d['bin_spread_mean'] if d['bin_spread_mean'] is None else round(d['bin_spread_mean'], 3)})"
            for d in Sq["consensus_families"][:12]))


def _consensus_families(J, have, q):
    """家族的共识：重要性（在场 + 取值）按 3 个模型平均排序；bin_spread（取值）单独列出。"""
    fc = {}
    for m in have:
        for d in J[m]["questions"][q]["families"]:
            fc.setdefault(d["family"], {})[m] = d
    out = []
    for f, per in fc.items():
        sp = [per[m]["bin_spread"] for m in per if per[m]["bin_spread"] is not None]
        out.append({"family": f, "importance_mean": _r(np.mean([per[m]["mean_abs_all"] if m in per else 0.0 for m in have])),
                    "bin_spread_mean": _r(np.mean(sp)) if sp else None,
                    "per_model": {m: {"mean_abs_all": per[m]["mean_abs_all"],
                                      "mean_signed_carriers": per[m]["mean_signed_carriers"],
                                      "bin_spread": per[m]["bin_spread"],
                                      "bins": {b["token"]: b["mean_person_phi"] for b in per[m]["bins"]}} for m in per}})
    out.sort(key=lambda d: -d["importance_mean"])
    for i, d in enumerate(out, 1):
        d["rank"] = i
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    for c in ("run", "smoke", "sens"):
        p = sub.add_parser(c)
        p.add_argument("--model", required=True, choices=sorted(runs()))
        p.add_argument("--seed", default="42", help="run 可以给逗号分隔的多个 seed（同一个作业里依次跑）")
        p.add_argument("--workers", type=int, default=w)
        if c == "run":
            p.add_argument("--limit", type=int, default=0, help="只跑队列前 N 人（调试；产物文件名带 _limit<N>）")
        if c == "smoke":
            p.add_argument("--n", type=int, default=6)
        if c == "sens":
            p.add_argument("--n", type=int, default=20)
    sub.add_parser("aggregate")
    a = ap.parse_args()
    if a.cmd in ("smoke", "sens"):
        a.seed = int(a.seed)
    {"run": cmd_run, "smoke": cmd_smoke, "sens": cmd_sens, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
