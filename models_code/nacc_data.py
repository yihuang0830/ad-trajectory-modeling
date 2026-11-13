"""nacc_data.py -- NACC UDS 做成和 newmodel_data_trans.npz 逐位同一种格式的预训练数据（人级，只留在 RIS 的 $ZOO_RUN_DIR/nacc/）。

    python nacc_data.py                 # -> $ZOO_RUN_DIR/nacc/nacc_data.npz + nacc_meta.json + nacc_data_summary.json（聚合）
    python nacc_data.py --check_rosmap  # 只做自检：这里的标签函数在 ROSMAP val 上重算第三部分真值，必须和 truth_trans_val.csv 逐格相同

为什么要它：ROSMAP 训练的人不多，从零训练的序列模型 / transformer 容易过拟合。
NACC（全美各 ADRC 的统一数据集）大得多，先在它上面预训练，再到 ROSMAP 上微调。

和 ROSMAP 逐位同一种格式（每次访视 54 维、静态 25 维、手工历史特征 120 维、标签 6 道题 × 1..10 年），同一份模型代码直接能吃：
  连续量（15 个）  MMSE（NACCMMSE，只有 UDS v1/v2 有；v3 改用 MoCA，不做换算，v3 访视的 MMSE 记缺失）、
                  全局认知（下面的复合分）、收缩压 / 舒张压（BPSYS / BPDIAS）、BMI（NACCBMI）；
                  其余 10 个（血糖、HbA1c、血脂、肾小球滤过率、睡眠、炎症指标）NACC 没有，一律缺失（掩码 = 1）。
                  标准化用 ROSMAP 训练段的均值 / 标准差（meta["norm"]），所以"MMSE = 24"在两边是同一个输入值。
  全局认知        ROSMAP 的 cogn_global 是多项测验 z 分数的平均。这里同样做：每项测验按"第一次做这项测验时认知正常（NACCUDSD = 1）
                  的人"标准化成 z（连线测验取负号），一次访视的复合分 = 这次做了的测验 z 的平均（至少做了这套测验的一半）。
                  UDS v1/v2 一套 10 项、v3 一套 12 项（两套各自标准化；两套共有的类别流畅、连线测验各算各的）。
  状态（9 个）    高血压、糖尿病、心衰、冠心病（心梗 / 支架 / 搭桥）按"曾经有过"累计；跛行 NACC 没有（缺失）；
                  降压药、降脂药、降糖药、AD 用药 = NACC 的派生变量 NACCAHTN / NACCLIPL / NACCDBMD / NACCADMD。
  曾经有过（2 个） 卒中（CBSTROKE）、抑郁（DEP2YRS）。
  AD 状态         临床诊断痴呆（DEMENTED = 1）且 AD 是主要或次要病因（NACCALZP = 1 / 2），第一次出现以后一直是 1。
  时间            和 ROSMAP 同一张网格：访视按距第一次访视的年数四舍五入到整年（同一年两次只留第一次），
                  站点时刻 t = round(基线年龄 × 365.25) + k × 365（天），年龄 z = (t / 365.25 − 80) / 10。
  死亡            NACC 的死亡记录（NACCDIED、NACCYOD / NACCMOD）：落在距基线的第 ceil(年数) 年，至少比最后一次访视晚一年。
                  没有上报死亡的人一律当成"最后一次访视之后不知道"（和 ROSMAP 的规则一样，不把失访算成活着）。
  静态（25 个）   性别、教育、种族、西语裔、APOE、吸烟（第一次访视）；饮酒（ROSMAP 是终生日均饮酒量，NACC 没有）记缺失；
                  study_MAP / study_LATC = 0；"基线就有"的 5 个状态取第一次访视。
  标签            transition_truth.py 的规则原样（同一张 365 天网格、只数答案确实知道的格）；--check_rosmap 证明这份实现和它逐格相同。
划分：按人随机 90% train / 10% test（test 只给预训练早停）。NACC 里没有 ROSMAP 的 val 集，验证一律在 ROSMAP val 上做。

注意：NACC 的中心代码是匿名的，认不出 Rush ADRC；ROS / MAP 是社区队列、不交 UDS，但 Rush 的拉丁裔核心队列（ROSMAP 里的 LATC）
可能也在 NACC 里。评估时另报去掉 LATC 的人之后的结果（pretrain.py），说明结论不靠可能重叠的人。
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import GRID, RUN_DIR  # noqa: E402
from transitions import STAGE_BINS, TRANS, TRUTH_HORIZONS, label_col  # noqa: E402

NACC_CSV = os.environ.get("NACC_CSV", "/path/to/NACC/"
                                      "Delphi-2M/investigator_nacc72.csv")
OUT = os.path.join(RUN_DIR, "nacc")
SEED = 20261022

# 测验：名字 -> (合法范围, 方向)；方向 -1 = 越大越差（连线测验的秒数）
TESTS_V2 = {"LOGIMEM": (0, 25, 1), "MEMUNITS": (0, 25, 1), "DIGIF": (0, 12, 1), "DIGIB": (0, 12, 1), "ANIMALS": (0, 77, 1),
            "VEG": (0, 77, 1), "TRAILA": (1, 150, -1), "TRAILB": (1, 300, -1), "WAIS": (0, 93, 1), "BOSTON": (0, 30, 1)}
TESTS_V3 = {"CRAFTVRS": (0, 44, 1), "CRAFTDVR": (0, 44, 1), "UDSBENTC": (0, 17, 1), "UDSBENTD": (0, 17, 1),
            "DIGFORCT": (0, 14, 1), "DIGBACCT": (0, 14, 1), "ANIMALS": (0, 77, 1), "VEG": (0, 77, 1), "TRAILA": (1, 150, -1),
            "TRAILB": (1, 300, -1), "MINTTOTS": (0, 32, 1), "UDSVERTN": (0, 80, 1)}
HIST = {"hypertension_cum": ("HYPERTEN",), "dm_cum": ("DIABETES",), "chf_cum": ("CVCHF",),
        "heart_cum": ("CVHATT", "CVANGIO", "CVBYPASS")}          # 1 = 近期 / 活动，2 = 既往 -> 都算有过；0 = 没有；-4 / 9 = 不知道
RX = {"antihyp_rx": "NACCAHTN", "statin_rx": "NACCLIPL", "diabetes_rx": "NACCDBMD", "ad_rx": "NACCADMD"}
COLS = (["NACCID", "NACCADC", "FORMVER", "PACKET", "VISITYR", "VISITMO", "VISITDAY", "BIRTHYR", "BIRTHMO", "SEX", "EDUC", "RACE",
         "HISPANIC", "NACCAPOE", "NACCMMSE", "NACCUDSD", "DEMENTED", "NACCALZP", "NACCDIED", "NACCYOD", "NACCMOD", "BPSYS",
         "BPDIAS", "NACCBMI", "CBSTROKE", "DEP2YRS", "TOBAC30", "TOBAC100"]
        + sorted(set(TESTS_V2) | set(TESTS_V3)) + sorted({c for v in HIST.values() for c in v}) + list(RX.values()))


# ----------------------------------------------------------------------------- 标签（transition_truth.build 的规则）


def stage_level(raw, fam):
    """原始值 -> transitions.STAGE_BINS 的箱序（0 = 最差 .. 2 = 最好），和分词的切点一致（MMSE 24 / 27，认知 −1 / 0）。"""
    cuts = (24.0, 27.0) if fam == "mmse" else (-1.0, 0.0)
    return np.digitize(np.round(raw, 4), cuts)


TARGET = {q: (fam, np.array(sorted(STAGE_BINS[fam].index(n) for n in toks))) for q, (_, fam, toks) in TRANS.items()
          if fam != "event"}


def person_labels(ages, death, ad, meas):
    """一个人：ages (访视的网格时刻，天，升序)、death / ad（天，inf = 没有）、meas[fam] = (测量时刻, 箱序) -> (访视, 6 × 10) 标签。
    逐条照抄 transition_truth.build 的循环体（列顺序 = 题目外层、时间窗内层）。"""
    last_vis = ages.max()
    assert death > last_vis
    Q = list(TRANS)
    out = np.full((len(ages), len(Q), len(TRUTH_HORIZONS)), -1, np.int8)
    empty = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    for i, t in enumerate(ages):
        for hi, H in enumerate(TRUTH_HORIZONS):
            W = t + H * GRID
            if death <= W:
                y = 1
            elif np.isfinite(death) or last_vis >= W:
                y = 0
            else:
                y = -1
            out[i, Q.index("death"), hi] = y
            if ad <= t:
                y = -1
            elif ad <= W:
                y = 1
            elif ((ages >= W) & (ages < ad)).any():
                y = 0
            else:
                y = -1
            out[i, Q.index("ad"), hi] = y
            for q, (fam, tgt) in TARGET.items():
                a_, l_ = meas.get(fam, empty)
                hit = np.isin(l_, tgt)
                past = a_ <= t
                if not past.any() or hit[past].any():
                    y = -1
                else:
                    win = (a_ > t) & (a_ <= W)
                    after = a_ >= W
                    if hit[win].any():
                        y = 1
                    elif after.any() and not hit[after][0]:
                        y = 0
                    else:
                        y = -1
                out[i, Q.index(q), hi] = y
    return out


def check_rosmap(split="val"):
    """自检：用 transition_truth 同样的数据源（rosmap_snapshot 的 token）拿出访视 / 事件 / 测量，交给 person_labels，
    和 $ZOO_RUN_DIR/truth_trans_<split>.csv 逐格对照。"""
    from common import TRUTH_DATASET, label_names, load_bin, pre_id
    names = label_names(TRUTH_DATASET)
    pre = pre_id(names)
    s = load_bin(TRUTH_DATASET, split)
    sex_tok = {pre["Female"], pre["Male"]}
    is_sex = np.isin(s[:, 2], list(sex_tok))
    vis = pd.DataFrame({"pid": s[is_sex, 0], "age": s[is_sex, 1]}).drop_duplicates().sort_values(["pid", "age"])

    def first_age(tok):
        m = s[:, 2] == tok
        return pd.DataFrame(s[m][:, :2], columns=["pid", "age"]).groupby("pid").age.min().to_dict()
    ev = {"ad": first_age(pre["AD_DX"]), "death": first_age(pre["Death"])}
    meas = {}
    for fam, bins in STAGE_BINS.items():
        lvl = {pre[n]: k for k, n in enumerate(bins)}
        m = np.isin(s[:, 2], list(lvl))
        df = pd.DataFrame({"pid": s[m, 0], "age": s[m, 1], "lvl": [lvl[t] for t in s[m, 2]]}).sort_values("age")
        meas[fam] = {p: (g.age.to_numpy(), g.lvl.to_numpy()) for p, g in df.groupby("pid")}
    truth = pd.read_csv(os.path.join(RUN_DIR, f"truth_trans_{split}.csv")).set_index(["pid", "t"])
    cols = [label_col(q, H) for q in TRANS for H in TRUTH_HORIZONS]
    n, bad = 0, 0
    for pid, g in vis.groupby("pid"):
        ages = g.age.to_numpy()
        Y = person_labels(ages, ev["death"].get(pid, np.inf), ev["ad"].get(pid, np.inf),
                          {f: meas[f][pid] for f in meas if pid in meas[f]})
        ref = truth.loc[[(pid, int(t)) for t in ages], cols].to_numpy().reshape(Y.shape)
        bad += int((Y != ref).sum())
        n += Y.size
    print(f"check_rosmap {split}: {n} cells, {bad} mismatches", flush=True)
    assert bad == 0, "person_labels 和 transition_truth.build 的口径不一致"
    return n


# ----------------------------------------------------------------------------- NACC -> 访视表


def _valid(v, lo, hi):
    v = pd.to_numeric(v, errors="coerce")
    return v.where((v >= lo) & (v <= hi))


def load_nacc():
    d = pd.read_csv(NACC_CSV, usecols=COLS, low_memory=False)
    d["date"] = pd.to_datetime(dict(year=d.VISITYR, month=d.VISITMO, day=d.VISITDAY.clip(1, 28)), errors="coerce")
    d = d[d.date.notna()].sort_values(["NACCID", "date"]).reset_index(drop=True)
    d["mmse"] = _valid(d.NACCMMSE, 0, 30)
    d["sbp"] = _valid(d.BPSYS, 70, 250)
    d["dbp"] = _valid(d.BPDIAS, 30, 150)
    d["bmi"] = _valid(d.NACCBMI, 12, 70)
    d["v3"] = d.FORMVER >= 3
    # 复合认知：每项测验按"第一次做这项测验时认知正常"的人标准化
    comp = {}
    for ver, tests in (("v2", TESTS_V2), ("v3", TESTS_V3)):
        rows = ~d.v3 if ver == "v2" else d.v3
        Zs = []
        for c, (lo, hi, sgn) in tests.items():
            x = _valid(d[c], lo, hi).where(rows)
            first = x.notna() & ~d.loc[x.notna()].duplicated("NACCID").reindex(d.index, fill_value=True)
            ref = x[first & (d.NACCUDSD == 1)]
            Zs.append(sgn * (x - ref.mean()) / ref.std())
            comp[f"{ver}_{c}"] = {"mean": float(ref.mean()), "sd": float(ref.std()), "n_ref": int(ref.notna().sum())}
        Z = pd.concat(Zs, axis=1)
        k = Z.notna().sum(1)
        d.loc[rows, "cogn"] = Z.mean(1).where(k >= math.ceil(len(tests) / 2))[rows]
    d["ad_vis"] = (d.DEMENTED == 1) & d.NACCALZP.isin([1, 2])
    return d, comp


def static_row(r):
    """第一次访视 -> 25 维静态（和 models_code/input_data.static_features 同一顺序；饮酒 = 缺失，study = 0，基线状态另外补）。"""
    f = [1.0 if r.SEX == 1 else 0.0]
    educ = r.EDUC if 0 <= r.EDUC <= 36 else np.nan
    f += [0.0 if np.isnan(educ) else (educ - 16.0) / 4.0, float(np.isnan(educ))]
    rc = r.RACE
    f += [float(rc == 1), float(rc == 2), float(rc in (3, 4, 5, 50)), float(rc not in (1, 2, 3, 4, 5, 50))]
    f += [float(r.HISPANIC == 1), float(r.HISPANIC not in (0, 1))]
    g = r.NACCAPOE                                    # 1 e3e3, 2 e3e4, 3 e3e2, 4 e4e4, 5 e4e2, 6 e2e2, 9 缺失
    f += [float(g in (2, 5)), float(g == 4), float(g not in (1, 2, 3, 4, 5, 6)), float(g in (3, 5, 6))]
    cur, ever = r.TOBAC30, r.TOBAC100
    known = cur in (0, 1) or ever in (0, 1)
    f += [float(ever == 1 and cur != 1), float(cur == 1), float(not known)]
    f += [0.0, 1.0]                                   # 饮酒：NACC 没有 ROSMAP 的终生日均饮酒量
    f += [0.0, 0.0]                                   # study_MAP、study_LATC
    return f


def build(d, norm, split_rng):
    from input_data import CONT, EVER, PREVALENT, STATE, l2c_features
    mu = {c: norm["mu"][c] for c in CONT}
    sd = {c: norm["sd"][c] for c in CONT}
    src = {"cts_estmmse30": "mmse", "cogn_global": "cogn", "sbp_avg": "sbp", "dbp_avg": "dbp", "bmi": "bmi"}
    persons, n_drop_dup, n_young = [], 0, 0
    for pid, g in d.groupby("NACCID", sort=False):
        r0 = g.iloc[0]
        birth = pd.Timestamp(year=int(r0.BIRTHYR), month=int(r0.BIRTHMO) if 1 <= r0.BIRTHMO <= 12 else 7, day=15)
        ab = (r0.date - birth).days / 365.25
        if ab < 50:                                   # 年轻的对照 / 早发病例：和 ROSMAP（65 岁以上的社区老人）差太远
            n_young += 1
            continue
        k = np.round((g.date - r0.date).dt.days.to_numpy() / 365.25).astype(int)
        keep = ~pd.Series(k).duplicated().to_numpy()
        n_drop_dup += int((~keep).sum())
        g, k = g[keep], k[keep]
        ages = np.array([int(round(ab * 365.25)) + int(kk) * 365 for kk in k], np.int64)
        # 事件
        ad_rows = np.where(g.ad_vis.to_numpy())[0]
        ad_k = int(k[ad_rows[0]]) if len(ad_rows) else None
        ad_t = float(ages[ad_rows[0]]) if len(ad_rows) else np.inf
        death_t = np.inf
        if r0.NACCDIED == 1 or (g.NACCDIED == 1).any():
            rr = g.iloc[-1]
            if 1900 < rr.NACCYOD < 2100:
                dd = pd.Timestamp(year=int(rr.NACCYOD), month=int(rr.NACCMOD) if 1 <= rr.NACCMOD <= 12 else 7, day=15)
                kd = max(int(math.ceil((dd - r0.date).days / 365.25 - 1e-9)), int(k[-1]) + 1)
                death_t = float(int(round(ab * 365.25)) + kd * 365)
        # 逐访视
        stat = static_row(r0)
        rows, l2c, meas = [], [], {"mmse": ([], []), "cogn": ([], [])}
        hist = {c: [] for c in CONT}
        cum = {c: 0.0 for c in HIST}
        cum_known = {c: False for c in HIST}
        ever = {"r_stroke": 0.0, "r_depres": 0.0}
        prev_k = None
        base_state = None
        for j, (_, v) in enumerate(g.iterrows()):
            y = float(k[j])
            for c in CONT:
                val = v[src[c]] if c in src else np.nan
                if pd.notna(val):
                    hist[c].append((y, (val - mu[c]) / sd[c]))
            for fam, col in (("mmse", "mmse"), ("cogn", "cogn")):
                if pd.notna(v[col]):
                    meas[fam][0].append(ages[j])
                    meas[fam][1].append(int(stage_level(np.array([v[col]]), fam)[0]))
            l2c.append(l2c_features(hist, y))
            x_c = [((v[src[c]] - mu[c]) / sd[c]) if (c in src and pd.notna(v[src[c]])) else 0.0 for c in CONT]
            m_c = [0.0 if (c in src and pd.notna(v[src[c]])) else 1.0 for c in CONT]
            x_s, m_s = [], []
            for c in STATE:
                if c in HIST:
                    vals = [v[h] for h in HIST[c]]
                    pos = any(x in (1, 2) for x in vals)
                    neg = all(x == 0 for x in vals)
                    if pos:
                        cum[c], cum_known[c] = 1.0, True
                    known_now = pos or neg
                    # 累计：以前有过 = 1（知道）；这次评估了 = 评估的结果；都没有 = 缺失
                    if cum[c] > 0.5:
                        x_s.append(1.0); m_s.append(0.0)
                    elif known_now:
                        x_s.append(0.0); m_s.append(0.0)
                    else:
                        x_s.append(0.0); m_s.append(1.0)
                elif c in RX:
                    rx = v[RX[c]]
                    x_s.append(float(rx == 1)); m_s.append(float(rx not in (0, 1)))
                else:                                 # claudication_cum：NACC 没有
                    x_s.append(0.0); m_s.append(1.0)
            if base_state is None:
                base_state = {c: x_s[STATE.index(c)] if m_s[STATE.index(c)] < 0.5 else 0.0 for c in PREVALENT}
            if v.CBSTROKE in (1, 2):
                ever["r_stroke"] = 1.0
            if v.DEP2YRS == 1:
                ever["r_depres"] = 1.0
            x_e = [ever[c] for c in EVER]
            ad_state = float(ad_k is not None and k[j] >= ad_k)
            tf = [(ages[j] / 365.25 - 80.0) / 10.0, y / 10.0, 0.0 if prev_k is None else (y - prev_k)]
            rows.append(x_c + m_c + x_s + m_s + x_e + [ad_state] + tf)
            prev_k = y
        meas = {f: (np.array(a, np.int64), np.array(b, np.int64)) for f, (a, b) in meas.items() if a}
        Y = person_labels(ages, death_t, ad_t, meas)
        stat = stat + [base_state[c] for c in PREVALENT]
        persons.append({"pid": int(len(persons)), "x": np.array(rows, np.float32), "t": ages, "static": np.array(stat, np.float32),
                        "y": Y.reshape(len(ages), -1), "l2c": np.array(l2c, np.float32), "center": int(r0.NACCADC)})
    idx = split_rng.permutation(len(persons))
    n_te = len(persons) // 10
    split = {"test": sorted(idx[:n_te].tolist()), "train": sorted(idx[n_te:].tolist())}
    return persons, split, {"dropped_same_year_visits": n_drop_dup, "excluded_baseline_age_lt50": n_young}


def summarize(persons, split, comp, extra):
    Q = list(TRANS)
    Y = np.concatenate([p["y"] for p in persons]).reshape(-1, len(Q), len(TRUTH_HORIZONS))
    X = np.concatenate([p["x"] for p in persons])
    from input_data import CONT, STATE
    out = {"n_persons": len(persons), "n_train": len(split["train"]), "n_test": len(split["test"]),
           "n_landmarks": int(len(X)), "n_centers": len({p["center"] for p in persons}),
           "visits_per_person": {q: float(np.quantile([len(p["t"]) for p in persons], q)) for q in (0.25, 0.5, 0.75, 0.9, 1.0)},
           "observed_frac": {c: float(1 - X[:, len(CONT) + j].mean()) for j, c in enumerate(CONT)},
           "state_known_frac": {c: float(1 - X[:, 2 * len(CONT) + len(STATE) + j].mean()) for j, c in enumerate(STATE)},
           "ad_state_frac": float(X[:, 2 * len(CONT) + 2 * len(STATE) + 2].mean()),
           "labels": {f"{q}_{H}y": {"pos": int((Y[:, qi, hi] == 1).sum()), "neg": int((Y[:, qi, hi] == 0).sum())}
                      for qi, q in enumerate(Q) for hi, H in enumerate(TRUTH_HORIZONS) if H in (1, 2, 5, 10)},
           "cognition_composite_reference": comp, **extra}
    return out


def pack(persons, V):
    from input_data import pack as pk
    D = pk(persons, V)
    D["pid"] = np.array([p["pid"] for p in persons], np.int64)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_rosmap", action="store_true")
    args = ap.parse_args()
    n_cells = check_rosmap("val")
    if args.check_rosmap:
        return
    meta_r = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    d, comp = load_nacc()
    print(f"NACC: {d.NACCID.nunique()} persons, {len(d)} visits", flush=True)
    persons, split, extra = build(d, meta_r["norm"], np.random.default_rng(SEED))
    V = max(len(p["t"]) for p in persons)
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for s, ids in split.items():
        for kk, v in pack([persons[i] for i in ids], V).items():
            out[f"{s}_{kk}"] = v
    np.savez_compressed(os.path.join(OUT, "nacc_data.npz"), **out)
    meta = dict(meta_r, n={s: len(v) for s, v in split.items()}, max_visits=V, source=os.path.basename(NACC_CSV),
                norm_from="ROSMAP train (newmodel_data_trans_meta.json)")
    json.dump(meta, open(os.path.join(OUT, "nacc_meta.json"), "w"), indent=1)
    summ = summarize(persons, split, comp, extra)
    summ["label_check_cells_rosmap_val"] = n_cells
    json.dump(summ, open(os.path.join(OUT, "nacc_data_summary.json"), "w"), indent=1)
    print(json.dumps({k: summ[k] for k in summ if k not in ("cognition_composite_reference",)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
