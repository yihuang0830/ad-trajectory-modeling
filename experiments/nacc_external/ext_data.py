"""ext_data.py -- 第 2、3 节：NACC 外部验证的数据（人级，只留 RIS 的 $ZOO_RUN_DIR/nacc_ext/）。

    python ext_data.py

步骤：
  1. 读 NACC；排除 ext_overlap.py 找出来的人和中心。
  2. 队列：nacc_data.py 的口径（第一次访视 >= 50 岁）全部建出来，每人标 primary = 第一次访视 NACCAGE >= 65 且 NACCUDSD != 4（主分析）。
  3. 按中心划分 H / E：中心随机排列（种子 SEED），按 primary 人数累计到 >= 25% 为 H，其余为 E。
  4. 协调参数只在 H（primary）和 ROSMAP train 上拟合：
       各项测验 z 分数的参照组 = H 里第一次做这项测验时认知正常（NACCUDSD = 1）的 primary 人；
       复合分 -> ROSMAP 尺度的线性连接：锚 = 第一次访视 MMSE 28–30，H primary 的均值 / 标准差对齐 ROSMAP train 基线 MMSE 28–30 的人。
     写进 harmonization.json 以后才构建 E。
  5. 和 ROSMAP 逐位同一种格式（nacc_data.build，标准化用 ROSMAP train 的均值 / 标准差）：
       data/H.npz           H 的全部数组（含标签 Y）
       data/E_X.npz         E 的特征（X / M / T / S / L / pid / 中心 / primary / 每次访视是不是 UDS v3）；**没有 Y**
       sealed/E_Y.npz       E 的标签（pid / T / M / Y）；只有 ext_eval.py 打开
       data/{H,E}_tok.bin   Delphi 的 token（交付版 dedup 的规则，nacc_delphi.person_rows）；E 的每个站点一条截到站点为止的伪记录
                            （pid = 站点序号 + 1），伪记录里不会有站点之后的 token；H 是整条记录（检查用）
       data/E_tok_lm.npz    E 伪记录的站点表（站点序号 -> E 的 pid、t）
       ids/                 pid <-> NACCID（只留 RIS）
  只写聚合数的：harmonization.json、data_summary.json（E 只报人数、站点数，不报任何标签的统计）。
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "models_code"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "unified_eval"))
from common import AD_BASE, DATA_DIR, GRID, RUN_DIR  # noqa: E402
import nacc_data as ND  # noqa: E402

sys.path.insert(0, os.path.join(AD_BASE, "tokenization"))
RAW = os.environ.get("ROSMAP_DATA_ROOT", os.path.join(AD_BASE, "raw"))
OUT = os.path.join(RUN_DIR, "nacc_ext")
SEED = 20261101
H_FRAC = 0.25
ANCHOR = (28, 30)


def read_nacc():
    cols = sorted(set(ND.COLS) | {"NACCAGE"})
    d = pd.read_csv(ND.NACC_CSV, usecols=cols, low_memory=False)
    d["date"] = pd.to_datetime(dict(year=d.VISITYR, month=d.VISITMO, day=d.VISITDAY.clip(1, 28)), errors="coerce")
    return d[d.date.notna()].sort_values(["NACCID", "date"]).reset_index(drop=True)


def cohort(d):
    """第一次访视 -> 每人：中心、是否 >= 50 岁（nacc_data.build 的生日算法）、primary。"""
    f = d.drop_duplicates("NACCID").set_index("NACCID")
    bm = f.BIRTHMO.where(f.BIRTHMO.between(1, 12), 7)
    birth = pd.to_datetime(dict(year=f.BIRTHYR, month=bm, day=15), errors="coerce")
    ab = (f.date - birth).dt.days / 365.25
    return pd.DataFrame({"center": f.NACCADC, "ge50": ab >= 50, "hisp": f.HISPANIC == 1,
                         "primary": (f.NACCAGE >= 65) & (f.NACCUDSD != 4) & (ab >= 50)})


def split_centers(P):
    cnt = P[P.primary].center.value_counts().sort_index()
    order = np.random.default_rng(SEED).permutation(cnt.index.to_numpy())
    cum = np.cumsum(cnt.reindex(order).to_numpy())
    k = int(np.searchsorted(cum, H_FRAC * cum[-1])) + 1
    return sorted(int(c) for c in order[:k]), sorted(int(c) for c in order[k:])


def composite(d, ref_ids):
    """nacc_data.load_nacc 的复合认知分，参照组只取 ref_ids 里的人。返回 (每行的复合分, 参数)。"""
    v3 = d.FORMVER >= 3
    in_ref = d.NACCID.isin(ref_ids)
    out = pd.Series(np.nan, index=d.index)
    par = {}
    for ver, tests in (("v2", ND.TESTS_V2), ("v3", ND.TESTS_V3)):
        rows = ~v3 if ver == "v2" else v3
        Zs = []
        for c, (lo, hi, sgn) in tests.items():
            x = ND._valid(d[c], lo, hi).where(rows)
            first = x.notna() & ~d.loc[x.notna()].duplicated("NACCID").reindex(d.index, fill_value=True)
            ref = x[first & (d.NACCUDSD == 1) & in_ref]
            Zs.append(sgn * (x - ref.mean()) / ref.std())
            par[f"{ver}_{c}"] = {"mean": float(ref.mean()), "sd": float(ref.std()), "n_ref": int(ref.notna().sum())}
        Z = pd.concat(Zs, axis=1)
        k = Z.notna().sum(1)
        out[rows] = Z.mean(1).where(k >= math.ceil(len(tests) / 2))[rows]
    return out, par


def rosmap_anchor():
    """ROSMAP train 的人、基线（fu_year = 0）MMSE 28–30 的 cogn_global 均值 / 标准差。"""
    lo = pd.read_excel(f"{RAW}/longitudinal_data_gk.xlsx")
    tr = set(np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))["train_pid"].tolist())
    b = lo[(lo.fu_year == 0) & lo.projid.isin(tr) & lo.cts_estmmse30.between(*ANCHOR) & lo.cogn_global.notna()]
    return float(b.cogn_global.mean()), float(b.cogn_global.std()), int(len(b))


def visit_keep(g):
    """nacc_data.build 的同年去重：距第一次访视的年数四舍五入，同一年只留第一次。"""
    k = np.round((g.date - g.date.iloc[0]).dt.days.to_numpy() / 365.25).astype(int)
    return ~pd.Series(k).duplicated().to_numpy(), k


def build_tokens(d, ids, persons):
    """nacc_delphi.tokenize_nacc 的同一个循环（只换数据：协调过的复合分、这里的人），返回每人 (访视网格, [(年龄, token)])。"""
    import nacc_delphi as NDel
    import spec
    meta_r = json.load(open(os.path.join(DATA_DIR, "rosmap", "meta.json")))
    TOK = meta_r["tok"]
    CUT = {pre: rule for _, pre, kind, rule in spec.CONT if kind == "cut"}
    CUT["EDU"] = dict((pre, rule) for _, pre, _, rule in spec.BACKGROUND)["EDU"]
    out = []
    gb = {k: g for k, g in d[d.NACCID.isin(set(ids))].groupby("NACCID", sort=False)}
    for nid, p in zip(ids, persons):
        g = gb[nid]
        r0 = g.iloc[0]
        birth = pd.Timestamp(year=int(r0.BIRTHYR), month=int(r0.BIRTHMO) if 1 <= r0.BIRTHMO <= 12 else 7, day=15)
        ab = (r0.date - birth).days / 365.25
        keep, k = visit_keep(g)
        g, k = g[keep], k[keep]
        ad_rows = np.where(g.ad_vis.to_numpy())[0]
        ad_k = int(k[ad_rows[0]]) if len(ad_rows) else None
        death_k = None
        if r0.NACCDIED == 1 or (g.NACCDIED == 1).any():
            rr = g.iloc[-1]
            if 1900 < rr.NACCYOD < 2100:
                dd = pd.Timestamp(year=int(rr.NACCYOD), month=int(rr.NACCMOD) if 1 <= rr.NACCMOD <= 12 else 7, day=15)
                death_k = max(int(math.ceil((dd - r0.date).days / 365.25 - 1e-9)), int(k[-1]) + 1)
        ages = np.array([int(round(ab * 365.25)) + int(kk) * GRID for kk in k], np.int64)
        assert np.array_equal(ages, p["t"]), f"{nid}: token 的访视网格和 npz 不同"
        out.append(NDel.person_rows(g, k, ab, ad_k, death_k, TOK, CUT))
    return out, TOK


N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
_G = {}


def _init(d, norm):
    _G.update(d=d, norm=norm)


def _work(ids):
    """一批人：nacc_data.build（特征、标签）+ Delphi token + 每次访视是不是 v3。和整份一起跑逐人相同（build 逐人独立）。"""
    d = _G["d"]
    sub = d[d.NACCID.isin(set(ids))]
    sub = sub.set_index("NACCID").loc[ids].reset_index()                       # 保持 ids 的顺序
    persons, _, extra = ND.build(sub, _G["norm"], np.random.default_rng(0))   # 它自己的 90/10 划分不用
    assert extra["excluded_baseline_age_lt50"] == 0 and len(persons) == len(ids)
    tokens, _ = build_tokens(sub, ids, persons)
    out = {}
    for nid, p, tk, (_, g) in zip(ids, persons, tokens, sub.groupby("NACCID", sort=False)):
        keep, _ = visit_keep(g)
        out[nid] = (p, tk, g.v3.to_numpy()[keep])
    return out


def main():
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "sealed"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "ids"), exist_ok=True)
    ND.check_rosmap("val")                                     # 标签代码和 ROSMAP 第三部分真值逐格相同
    d = read_nacc()
    ex_ids = set(open(os.path.join(OUT, "overlap", "exclude_naccid.txt")).read().split())
    ex_c = set(json.load(open(os.path.join(OUT, "overlap", "exclude_centers.json")))["centers"])
    P = cohort(d)
    n0 = len(P)
    P = P[P.ge50 & ~P.index.isin(ex_ids) & ~P.center.isin(ex_c)]
    H_c, E_c = split_centers(P)
    P["seg"] = np.where(P.center.isin(H_c), "H", "E")
    # 敏感性分析（第 2.2 节）：E 里西语裔比例最高的两个中心（primary 人、第一次访视）
    hs = P[P.primary & (P.seg == "E")].groupby("center").hisp.mean().sort_values(ascending=False)
    hisp_top2 = [int(c) for c in hs.index[:2]]
    d = d[d.NACCID.isin(P.index)].reset_index(drop=True)
    # 一个人在两个中心都有访视：只按第一次访视的中心归段，这里只数一下
    multi = d.groupby("NACCID").NACCADC.nunique().gt(1).sum()

    # ---- 协调参数：只用 H 的 primary 人 + ROSMAP train
    H_ref = set(P.index[(P.seg == "H") & P.primary])
    d["cogn_raw"], test_par = composite(d, H_ref)
    f = d.drop_duplicates("NACCID")
    anc = f[f.NACCID.isin(H_ref) & ND._valid(f.NACCMMSE, 0, 30).between(*ANCHOR) & f.cogn_raw.notna()].cogn_raw
    mu_r, sd_r, n_r = rosmap_anchor()
    b = sd_r / float(anc.std())
    a = mu_r - b * float(anc.mean())
    harm = {"seed": SEED, "h_frac": H_FRAC, "H_centers": H_c, "E_centers": E_c, "excluded_centers": sorted(ex_c),
            "sensitivity_hispanic_top2_E_centers": hisp_top2, "hispanic_share_top2": [float(hs.iloc[0]), float(hs.iloc[1])],
            "n_excluded_persons_overlap": len(ex_ids), "test_reference": test_par,
            "cogn_link": {"a": a, "b": b, "anchor_mmse": ANCHOR, "nacc_H_anchor_n": int(len(anc)),
                          "nacc_H_anchor_mean": float(anc.mean()), "nacc_H_anchor_sd": float(anc.std()),
                          "rosmap_train_anchor_n": n_r, "rosmap_train_anchor_mean": mu_r, "rosmap_train_anchor_sd": sd_r},
            "norm_from": "ROSMAP train (newmodel_data_trans_meta.json)"}
    json.dump(harm, open(os.path.join(OUT, "harmonization.json"), "w"), indent=1)
    print("harmonization:", json.dumps(harm["cogn_link"]), flush=True)

    # ---- nacc_data 的其余清洗（load_nacc 的同一批列），复合分换成协调过的
    d["mmse"] = ND._valid(d.NACCMMSE, 0, 30)
    d["sbp"] = ND._valid(d.BPSYS, 70, 250)
    d["dbp"] = ND._valid(d.BPDIAS, 30, 150)
    d["bmi"] = ND._valid(d.NACCBMI, 12, 70)
    d["v3"] = d.FORMVER >= 3
    d["cogn"] = a + b * d.cogn_raw
    d["ad_vis"] = (d.DEMENTED == 1) & d.NACCALZP.isin([1, 2])

    meta_r = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    ids = d.NACCID.drop_duplicates().tolist()                                     # build 的 groupby(sort=False) 同一个顺序
    chunks = [ids[i::N_WORKERS] for i in range(N_WORKERS)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(N_WORKERS, initializer=_init, initargs=(d, meta_r["norm"])) as pool:
        res = pool.map(_work, chunks)
    by_id = {nid: r for part in res for nid, r in part.items()}
    persons = [by_id[nid][0] for nid in ids]
    tokens = [by_id[nid][1] for nid in ids]
    v3 = {nid: by_id[nid][2] for nid in ids}
    seg = P.seg.reindex(ids).to_numpy()
    prim = P.primary.reindex(ids).to_numpy()
    V = max(len(p["t"]) for p in persons)

    summ = {"n_nacc_persons_all": int(n0), "n_after_exclusion_ge50": int(len(P)), "persons_in_two_centers": int(multi),
            "max_visits": V}
    for s in ("H", "E"):
        idx = np.where(seg == s)[0]
        ps = [persons[i] for i in idx]
        D = ND.pack(ps, V)
        D["pid"] = np.arange(len(ps), dtype=np.int64)                 # 段内重新编号
        D["center"] = np.array([p["center"] for p in ps], np.int64)
        D["primary"] = prim[idx].astype(bool)
        V3 = np.zeros(D["M"].shape, bool)
        for j, i in enumerate(idx):
            V3[j, : len(v3[ids[i]])] = v3[ids[i]]
        D["V3"] = V3
        pd.DataFrame({"pid": D["pid"], "NACCID": [ids[i] for i in idx]}).to_csv(os.path.join(OUT, "ids", f"{s}.csv"), index=False)
        summ[s] = {"persons": int(len(ps)), "primary_persons": int(D["primary"].sum()), "centers": int(len(set(D["center"]))),
                   "landmarks": int(D["M"].sum()), "primary_landmarks": int(D["M"][D["primary"]].sum()),
                   "v3_landmarks": int((V3 & D["M"]).sum())}
        tk = [tokens[i] for i in idx]
        if s == "H":
            np.savez_compressed(os.path.join(OUT, "data", "H.npz"), **D)
            rows = [(j + 1, a_, t_) for j, r in enumerate(tk) for a_, t_ in r]
            np.array(rows, np.uint32).tofile(os.path.join(OUT, "data", "H_tok.bin"))
            Y = D["Y"].reshape(*D["M"].shape, 6, 10)
            summ["H"]["labels_primary_2y"] = {q: {"pos": int((Y[D["primary"]][..., qi, 1] == 1).sum()),
                                                  "neg": int((Y[D["primary"]][..., qi, 1] == 0).sum())}
                                              for qi, q in enumerate(meta_r["questions"])}
        else:
            Y = D.pop("Y")
            assert "Y" not in D
            np.savez_compressed(os.path.join(OUT, "data", "E_X.npz"), **D)
            np.savez_compressed(os.path.join(OUT, "sealed", "E_Y.npz"), Y=Y, pid=D["pid"], T=D["T"], M=D["M"])
            # Delphi：每个站点一条截到站点为止的伪记录
            chunks, lm = [], []
            for j, r in enumerate(tk):
                ra = np.array(r, np.int64).reshape(-1, 2)
                for v in np.where(D["M"][j])[0]:
                    t = int(D["T"][j, v])
                    h = ra[ra[:, 0] <= t]
                    q = len(lm)
                    chunks.append(np.column_stack([np.full(len(h), q + 1), h]))
                    lm.append((q, j, v, t))
            arr = np.concatenate(chunks).astype(np.uint32)
            arr.tofile(os.path.join(OUT, "data", "E_tok.bin"))
            lm = np.array(lm, np.int64)
            np.savez_compressed(os.path.join(OUT, "data", "E_tok_lm.npz"), q=lm[:, 0], pid=lm[:, 1], v=lm[:, 2], t=lm[:, 3])
            assert len(lm) == summ["E"]["landmarks"]
            assert (arr[:, 1].astype(np.int64) <= lm[arr[:, 0].astype(np.int64) - 1, 3]).all(), "伪记录里有站点之后的 token"
    json.dump(summ, open(os.path.join(OUT, "data_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
