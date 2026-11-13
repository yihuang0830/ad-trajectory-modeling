"""ext_overlap.py -- 第 2.2 节：找出"既在 ROSMAP 又在 NACC"的人（只用协变量，不看结局，不看模型）。

    python ext_overlap.py        # -> $ZOO_RUN_DIR/nacc_ext/overlap/exclude_naccid.txt（人级，只留 RIS）
                                 #    $ZOO_RUN_DIR/nacc_ext/overlap/overlap_summary.json（各中心计数，可以拿回仓库）

候选对：性别相同；种族、西语裔、APOE 基因型相容（任一边缺失算相容）；教育年数差 <= 1（任一边缺失算相容）；
两边的访视年龄范围重叠（容差 0.75 年）。
对上一次访视：两边各一次访视年龄差 <= 0.75 年，两边都测了的量（MMSE ±1、BMI ±0.5、收缩压 ±3、舒张压 ±3）至少 2 个且全部对得上。
疑似同一人：(a) 对上的访视 >= 3 次（两边各自数不同的访视，取小的）；或 (b) 两边都有确切的死亡年龄、差 <= 0.15 年，且至少对上 1 次。
本底：NACC 一侧的年龄（访视、死亡）整体平移 +3 / −3 年各做一遍，得到偶然对上的人数。
规则（PROTOCOL.md 第 2.2 节）：疑似的 NACC 人全部排除；某中心疑似人数 − 本底 >= 5 或 >= 该中心人数的 1%，整个中心排除。
ROSMAP 的访视年龄 = age_bl + fu_year（原始表里没有访视日期），所以容差放到 0.75 年。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "models_code"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "unified_eval"))
from common import AD_BASE, RUN_DIR  # noqa: E402
import nacc_data as ND  # noqa: E402

RAW = os.environ.get("ROSMAP_DATA_ROOT", os.path.join(AD_BASE, "raw"))
OUT = os.path.join(RUN_DIR, "nacc_ext", "overlap")
AGE_TOL, DEATH_TOL, MIN_VISITS = 0.75, 0.15, 3
TOL = {"mmse": 1.0, "bmi": 0.5, "sbp": 3.0, "dbp": 3.0}
MEAS = list(TOL)
APOE_NACC = {1: 33, 2: 34, 3: 23, 4: 44, 5: 24, 6: 22}


def rosmap():
    cs = pd.read_excel(f"{RAW}/cross-sectional-data-gk.xlsx").set_index("projid")
    lo = pd.read_excel(f"{RAW}/longitudinal_data_gk.xlsx")
    cl = pd.read_csv(f"{RAW}/ROSMAP_clinical.csv", dtype=str)
    cl["pid"] = pd.to_numeric(cl.projid, errors="coerce")
    cl = cl.dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")
    race = pd.to_numeric(cl.race, errors="coerce")
    span = pd.to_numeric(cl.spanish, errors="coerce")
    death = pd.to_numeric(cl.age_death, errors="coerce")          # "90+" -> NaN（截尾的不能用来比对）
    P = pd.DataFrame({"sex": cs.msex.map({1: 1, 0: 2}), "educ": cs.educ, "apoe": cs.apoe_genotype,
                      "study": cs.study.astype(str).str.strip()})
    P["race"] = race.reindex(P.index).map(lambda r: np.nan if pd.isna(r) else (1 if r == 1 else 2 if r == 2 else 3))
    P["hisp"] = span.reindex(P.index).map({1: 1, 2: 0})
    P["death"] = death.reindex(P.index)
    lo = lo.merge(cs[["age_bl"]], left_on="projid", right_index=True)
    V = pd.DataFrame({"pid": lo.projid, "age": lo.age_bl + lo.fu_year, "mmse": lo.cts_estmmse30, "bmi": lo.bmi,
                      "sbp": lo.sbp_avg, "dbp": lo.dbp_avg}).dropna(subset=["age"])
    return P, V


def nacc():
    cols = ["NACCID", "NACCADC", "VISITYR", "VISITMO", "VISITDAY", "BIRTHYR", "BIRTHMO", "SEX", "EDUC", "RACE", "HISPANIC",
            "NACCAPOE", "NACCMMSE", "NACCBMI", "BPSYS", "BPDIAS", "NACCYOD", "NACCMOD"]
    d = pd.read_csv(ND.NACC_CSV, usecols=cols, low_memory=False)
    d["date"] = pd.to_datetime(dict(year=d.VISITYR, month=d.VISITMO, day=d.VISITDAY.clip(1, 28)), errors="coerce")
    d = d[d.date.notna()].sort_values(["NACCID", "date"])
    bm = d.BIRTHMO.where(d.BIRTHMO.between(1, 12), 7)
    birth = pd.to_datetime(dict(year=d.BIRTHYR, month=bm, day=15), errors="coerce")
    d["age"] = (d.date - birth).dt.days / 365.25
    first = d.drop_duplicates("NACCID").set_index("NACCID")
    last = d.drop_duplicates("NACCID", keep="last").set_index("NACCID")
    dm = last.NACCMOD.where(last.NACCMOD.between(1, 12), 7)
    yod = last.NACCYOD.where(last.NACCYOD.between(1900, 2100))
    dd = pd.to_datetime(dict(year=yod, month=dm, day=15), errors="coerce")
    b0 = pd.to_datetime(dict(year=first.BIRTHYR, month=first.BIRTHMO.where(first.BIRTHMO.between(1, 12), 7), day=15),
                        errors="coerce")
    P = pd.DataFrame({"sex": first.SEX, "center": first.NACCADC,
                      "educ": first.EDUC.where(first.EDUC.between(0, 36)),
                      "apoe": first.NACCAPOE.map(APOE_NACC),
                      "race": first.RACE.map(lambda r: 1 if r == 1 else 2 if r == 2 else 3 if r in (3, 4, 5, 50) else np.nan),
                      "hisp": first.HISPANIC.map({1: 1, 0: 0}),
                      "death": ((dd - b0).dt.days / 365.25).reindex(first.index)})
    V = pd.DataFrame({"pid": d.NACCID, "age": d.age, "mmse": ND._valid(d.NACCMMSE, 0, 30), "bmi": ND._valid(d.NACCBMI, 12, 70),
                      "sbp": ND._valid(d.BPSYS, 70, 250), "dbp": ND._valid(d.BPDIAS, 30, 150)}).dropna(subset=["age"])
    return P, V


def compat(a, b):
    """逐元素：a 是标量，b 是数组；任一边缺失算相容。"""
    return np.isnan(b) | (b == a) if not np.isnan(a) else np.ones(len(b), bool)


def match(PR, VR, PN, VN, shift=0.0):
    """-> {naccid: (rosmap projid, n_visits, death_ok)} 的疑似同一人。shift：NACC 年龄整体平移（本底用）。
    每个 ROSMAP 人一次：候选人的全部访视摊平成一个数组，(ROSMAP 访视 × 候选访视) 一起比。"""
    nid = PN.index.to_numpy()
    pos = {p: i for i, p in enumerate(nid)}
    VN = VN[VN.pid.isin(pos)].sort_values(["pid", "age"])
    owner = VN.pid.map(pos).to_numpy()
    n_age = VN.age.to_numpy() + shift
    n_meas = VN[MEAS].to_numpy(float)
    death_n = PN.death.to_numpy(float) + shift
    lo_n = np.full(len(nid), np.inf)
    hi_n = np.full(len(nid), -np.inf)
    np.minimum.at(lo_n, owner, n_age)
    np.maximum.at(hi_n, owner, n_age)
    cols = {c: PN[c].to_numpy(float) for c in ("sex", "educ", "apoe", "race", "hisp")}
    tol = np.array([TOL[m] for m in MEAS])
    vr = {p: g for p, g in VR.groupby("pid")}
    hits = {}
    for rp, r in PR.iterrows():
        if rp not in vr:
            continue
        gr = vr[rp]
        ra = gr.age.to_numpy()
        rm = gr[MEAS].to_numpy(float)
        ok = cols["sex"] == r.sex
        for c in ("apoe", "race", "hisp"):
            ok &= compat(float(r[c]), cols[c])
        if not np.isnan(r.educ):
            ok &= np.isnan(cols["educ"]) | (np.abs(cols["educ"] - r.educ) <= 1)
        ok &= (lo_n <= ra.max() + AGE_TOL) & (hi_n >= ra.min() - AGE_TOL)
        sel = np.where(ok[owner] & (np.abs(n_age - ra[:, None]) <= AGE_TOL).any(0))[0]   # 候选人里年龄挨得上的访视
        if not len(sel):
            continue
        na, nm, ow = n_age[sel], n_meas[sel], owner[sel]
        near = np.abs(ra[:, None] - na[None, :]) <= AGE_TOL                                  # (R, N)
        both = ~np.isnan(rm)[:, None, :] & ~np.isnan(nm)[None, :, :]                       # (R, N, 4)
        agree = (np.abs(rm[:, None, :] - nm[None, :, :]) <= tol) | ~both
        good = near & (both.sum(-1) >= 2) & agree.all(-1)
        if not good.any():
            continue
        cand, ci = np.unique(ow, return_inverse=True)
        ri, vi = np.nonzero(good)
        r_hit = np.zeros((len(ra), len(cand)), bool)
        r_hit[ri, ci[vi]] = True
        n_hit = np.bincount(ci[np.unique(vi)], minlength=len(cand))                         # 对上的 NACC 访视数
        n_vis = np.minimum(r_hit.sum(0), n_hit)
        dn = death_n[cand]
        d_ok = ~np.isnan(dn) & (not np.isnan(r.death)) & (np.abs(dn - (r.death if not np.isnan(r.death) else 0)) <= DEATH_TOL)
        for k in np.where((n_vis >= MIN_VISITS) | (d_ok & (n_vis >= 1)))[0]:
            key = nid[cand[k]]
            prev = hits.get(key)
            if prev is None or n_vis[k] > prev[1]:
                hits[key] = (int(rp), int(n_vis[k]), bool(d_ok[k]))
    return hits


def main():
    os.makedirs(OUT, exist_ok=True)
    PR, VR = rosmap()
    PN, VN = nacc()
    print(f"ROSMAP {len(PR)} persons / {len(VR)} visits; NACC {len(PN)} persons / {len(VN)} visits", flush=True)
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))
    split_of = {int(p): s for s in ("train", "test", "val") for p in Z[f"{s}_pid"]}
    obs = match(PR, VR, PN, VN)
    print(f"observed: {len(obs)} NACC persons", flush=True)
    nulls = {s: match(PR, VR, PN, VN, shift=s) for s in (3.0, -3.0)}
    print({s: len(h) for s, h in nulls.items()}, flush=True)
    center = PN.center
    n_center = center.value_counts()
    rows = []
    for c in sorted(n_center.index):
        o = sum(center[k] == c for k in obs)
        nl = float(np.mean([sum(center[k] == c for k in h) for h in nulls.values()]))
        excess = o - nl
        rows.append({"center": int(c), "n_persons": int(n_center[c]), "observed": int(o), "null_mean": nl,
                     "exclude_center": bool(excess >= 5 or (o > 0 and excess >= 0.01 * n_center[c]))})
    ex_centers = [r["center"] for r in rows if r["exclude_center"]]
    ex_ids = sorted(obs)
    with open(os.path.join(OUT, "exclude_naccid.txt"), "w") as f:
        f.write("\n".join(ex_ids) + "\n")
    json.dump({"centers": ex_centers}, open(os.path.join(OUT, "exclude_centers.json"), "w"))
    rm = pd.Series({k: v[0] for k, v in obs.items()})
    summ = {
        "rule": "PROTOCOL.md 2.2", "age_tol": AGE_TOL, "death_tol": DEATH_TOL, "min_visits": MIN_VISITS, "tol": TOL,
        "n_rosmap_persons": int(len(PR)), "n_nacc_persons": int(len(PN)),
        "observed_nacc_persons": len(obs), "null_nacc_persons": {f"{s:+.0f}y": len(h) for s, h in nulls.items()},
        "null_by_evidence": {f"{s:+.0f}y": {"visits_ge3": int(sum(v[1] >= MIN_VISITS for v in h.values())),
                                            "death_plus_visit_only": int(sum(v[1] < MIN_VISITS for v in h.values()))}
                             for s, h in nulls.items()},
        "observed_by_evidence": {"visits_ge3": int(sum(v[1] >= MIN_VISITS for v in obs.values())),
                                 "death_plus_visit_only": int(sum(v[1] < MIN_VISITS for v in obs.values()))},
        "matched_rosmap_by_study": PR.study.reindex(rm.unique()).value_counts().to_dict(),
        "matched_rosmap_by_zoo_split": pd.Series([split_of.get(int(p), "not_in_zoo") for p in rm.unique()]).value_counts().to_dict(),
        "visits_matched_hist": pd.Series([v[1] for v in obs.values()]).value_counts().sort_index().to_dict(),
        "excluded_centers": ex_centers,
        "excluded_center_persons": int(sum(n_center[c] for c in ex_centers)),
        "centers": rows,
    }
    json.dump(summ, open(os.path.join(OUT, "overlap_summary.json"), "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in summ.items() if k != "centers"}, indent=1, default=int))
    print("centers with any observed match:", [r for r in rows if r["observed"] or r["null_mean"]])


if __name__ == "__main__":
    main()
