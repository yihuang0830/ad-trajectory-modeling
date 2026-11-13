"""ext_eval.py -- 第 7 节第 5 步：揭盲。核对封存 -> 读 E 的标签 -> 第 5、6 节的全部指标一次算完（聚合 JSON）。

    python ext_eval.py --check      # 只核对 SEAL.json（不打开标签）
    python ext_eval.py              # 揭盲；results/E_results.json 已经存在就拒绝（一次性）
    python ext_eval.py --rerun "原因"   # 出了 bug 必须重跑：原因和旧结果的哈希写进 results/deviations.json，旧结果改名保留
    python ext_eval.py --dry_on_H   # 揭盲前的演练：把 H 当考卷走一遍全部代码（不核对封存、不碰 E 的标签）-> results/DRYRUN_H.json

这是唯一打开 sealed/E_Y.npz 的脚本。所有输出都是聚合数（每格至少报 n、阳性数）。
"""
import argparse
import datetime
import json
import os
import shutil

import numpy as np

import ext_common as C
import ext_metrics as X
import bl_data as B

SEED = 20261102
N_BOOT = 2000
Q = list(B.Q)
HS_REPORT = (1, 2, 3, 5, 10)
PART1 = ("ad", "death", "mmse_lt24", "cogn_ltm1")
KEY_CELLS = [("ad", 2), ("ad", 5)]
CONFIRM_PAIR = ("xgb-l2c", "lr-cur")
def rosmap_published_ad2y():
    """PROTOCOL.md 第 1 节写定的 13 个模型在 ROSMAP val 上的 2 年 AD AUC（{模型: AUC}）。公开版不含结果数字：
    运行前把它们写进 C.OUT/rosmap_ad2y_auc.json。"""
    return json.load(open(os.path.join(C.OUT, "rosmap_ad2y_auc.json")))


def check_seal():
    seal = json.load(open(os.path.join(C.OUT, "SEAL.json")))
    from ext_delphi import CKPT
    paths = {"harmonization.json": os.path.join(C.OUT, "harmonization.json"), "delphi_ckpt": CKPT}
    bad = []
    for k, h in seal["sha256"].items():
        p = paths.get(k, os.path.join(C.OUT, k) if not k.startswith("scores/") else os.path.join(C.SCORE_DIR, k[len("scores/"):]))
        if C.sha256(p) != h:
            bad.append(k)
    for n, m in seal["models"].items():
        if m["model"] and C.sha256(os.path.join(C.MODEL_DIR, f"{n}.pt")) != m["model"]:
            bad.append(f"model {n}")
    if bad:
        raise SystemExit(f"封存核对失败：{bad}")
    print(f"封存核对通过：{len(seal['sha256'])} 个文件、{len(seal['models'])} 个模型（封存于 {seal['sealed_at']}）", flush=True)
    return seal


def stage(raw, fam):
    cuts = (24.0, 27.0) if fam == "mmse" else (-1.0, 0.0)
    return np.digitize(np.round(raw, 4), cuts)


def covariates(D, ii, vv, norm):
    """站点上的协变量（只来自特征）：年龄、年龄段、当前状态（最近一次测量的 MMSE 分期 × 认知分期，没测过 = -1）。"""
    x = D["X"][ii, vv]
    age = 80.0 + 10.0 * x[:, B.I_TF]
    L = D["L"][ii, vv]
    st = []
    for j, fam in ((0, "mmse"), (1, "cogn")):
        raw = L[:, j * 8] * norm["sd"][j] + norm["mu"][j]
        st.append(np.where(L[:, j * 8 + 1] > 0.5, stage(raw, fam), -1))
    key_state = st[0] * 10 + st[1]
    return {"age": age, "band_gap": np.clip(((age - 70) // 5).astype(int), 0, 6),
            "state_age": key_state * 1000 + np.floor(age / 5).astype(int)}


def cell_metrics(y, p, cov=None):
    k = y >= 0
    yk, pk = y[k].astype(int), p[k]
    if yk.sum() == 0 or yk.sum() == len(yk):
        return {"n": int(k.sum()), "n_pos": int(yk.sum())}
    r = X.calib(yk, pk)
    r["auc"] = X.auc(yk, pk)
    if cov is not None:
        r["auc_age_strat"] = X.strat_auc(yk, pk, cov["band_gap"][k])
        r["auc_state_age_strat"] = X.strat_auc(yk, pk, cov["state_age"][k])
    return r


def load_scores(sub, name, ii, vv):
    z = np.load(os.path.join(C.SCORE_DIR, sub, f"{name}.npz" if os.path.exists(os.path.join(C.SCORE_DIR, sub, f"{name}.npz"))
                             else f"{name}.val.npz"))
    pos = {(a, b): j for j, (a, b) in enumerate(zip(z["ii"].tolist(), z["vv"].tolist()))}
    idx = np.array([pos[(a, b)] for a, b in zip(ii.tolist(), vv.tolist())])
    return z["R"][idx]


def summary_aucs(Y, R):
    a = {f"{q}_{H}y": X.auc((Y[:, Q.index(q), H - 1] == 1)[Y[:, Q.index(q), H - 1] >= 0],
                            R[:, Q.index(q), H - 1][Y[:, Q.index(q), H - 1] >= 0]) for q in Q for H in (2, 5)}
    a["part1_2y_4endpoints"] = float(np.mean([a[f"{q}_2y"] for q in PART1]))
    a["part3_5y_6questions"] = float(np.mean([a[f"{q}_5y"] for q in Q]))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rerun", default=None)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--dry_on_H", action="store_true")
    args = ap.parse_args()
    rd = os.path.join(C.OUT, "results")
    os.makedirs(rd, exist_ok=True)
    if args.dry_on_H:
        names = [n for n in C.ALL13 + [f"{m}-nv" for m in C.NONDELPHI]
                 if os.path.exists(os.path.join(C.SCORE_DIR, "H", f"{n}.npz"))]
        seal = {"sealed_at": "DRY RUN ON H", "models": {n: {} for n in names}}
        res_path = os.path.join(rd, "DRYRUN_H.json")
    else:
        seal = check_seal()
        if args.check:
            return
        res_path = os.path.join(rd, "E_results.json")
    if os.path.exists(res_path) and not args.dry_on_H:
        if not args.rerun:
            raise SystemExit("已经揭盲过（results/E_results.json）；必须重跑就用 --rerun \"原因\"，会记进偏离记录")
        dev_p = os.path.join(rd, "deviations.json")
        dev = json.load(open(dev_p)) if os.path.exists(dev_p) else []
        old = res_path.replace(".json", f".before_rerun_{len(dev) + 1}.json")
        shutil.move(res_path, old)
        dev.append({"at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "reason": args.rerun,
                    "old_results": os.path.basename(old), "old_sha256": C.sha256(old)})
        json.dump(dev, open(dev_p, "w"), indent=1)
    meta = json.load(open(os.path.join(C.RUN_DIR, "newmodel_data_trans_meta.json")))
    import simulator as SIM
    norm = SIM.norm_info(meta)
    names = list(seal["models"])

    # ---------------------------------------------------------------- E（揭盲）
    if args.dry_on_H:
        with np.load(os.path.join(C.OUT, "data", "H.npz")) as z:
            DE = {k: z[k] for k in z.files}
        YE_all = DE.pop("Y").reshape(*DE["M"].shape, B.NQ, B.NH)
    else:
        with np.load(os.path.join(C.OUT, "data", "E_X.npz")) as z:
            DE = {k: z[k] for k in z.files}
        with np.load(os.path.join(C.OUT, "sealed", "E_Y.npz")) as z:
            assert (z["pid"] == DE["pid"]).all() and (z["T"] == DE["T"]).all()
            YE_all = z["Y"].reshape(*DE["M"].shape, B.NQ, B.NH)
    E_SUB = "H" if args.dry_on_H else "E"
    ii, vv = np.where(DE["M"] & DE["primary"][:, None])
    Y = YE_all[ii, vv]
    cov = covariates(DE, ii, vv, norm)
    center = DE["center"][ii]
    harm = json.load(open(os.path.join(C.OUT, "harmonization.json")))
    hisp2 = np.isin(center, harm["sensitivity_hispanic_top2_E_centers"])
    v12 = ~DE["V3"][ii, vv]
    R = {n: load_scores(E_SUB, n, ii, vv) for n in names}
    out = {"seal": {"sealed_at": seal["sealed_at"], "n_models": len(names)},
           "cohort": {"persons": int(len(np.unique(ii))), "landmarks": int(len(ii)), "centers": int(len(np.unique(center))),
                      "v12_landmarks": int(v12.sum()), "hispanic_top2_landmarks": int(hisp2.sum())},
           "label_counts": {f"{q}_{H}y": {"n": int((Y[:, Q.index(q), H - 1] >= 0).sum()), "pos": int((Y[:, Q.index(q), H - 1] == 1).sum())}
                            for q in Q for H in HS_REPORT},
           "models": {}}
    for n in names:
        m = {}
        for q in Q:
            for H in HS_REPORT:
                m[f"{q}_{H}y"] = cell_metrics(Y[:, Q.index(q), H - 1], R[n][:, Q.index(q), H - 1], cov if H in (2, 5) else None)
        s = summary_aucs(Y, R[n])
        m["part1_2y_4endpoints"], m["part3_5y_6questions"] = s["part1_2y_4endpoints"], s["part3_5y_6questions"]
        for lab, mask in (("excl_hispanic_top2_centers", ~hisp2), ("uds_v12_landmarks_only", v12)):
            m[lab] = {f"{q}_{H}y": cell_metrics(Y[mask, Q.index(q), H - 1], R[n][mask, Q.index(q), H - 1])
                      for q, H in KEY_CELLS + [("mmse_lt24", 2), ("death", 2)]}
            m[lab]["part1_2y_4endpoints"] = summary_aucs(Y[mask], R[n][mask])["part1_2y_4endpoints"]
        out["models"][n] = m
        print(f"{n:22s} AD 2y {m['ad_2y'].get('auc', np.nan):.3f}  AD 5y {m['ad_5y'].get('auc', np.nan):.3f}  "
              f"4ep 2y {m['part1_2y_4endpoints']:.3f}  O/E AD 2y {m['ad_2y'].get('o_e', np.nan):.2f}", flush=True)

    # ---------------------------------------------------------------- 两级 bootstrap（中心 -> 人）
    persons = np.unique(ii)
    p_idx = np.searchsorted(persons, ii)
    cop = DE["center"][persons]
    rng = np.random.default_rng(SEED)
    W = X.boot_weights(cop, args.n_boot, rng)
    boot = {}
    cells = KEY_CELLS + [(q, 2) for q in PART1 if q != "ad"] + [(q, 5) for q in Q if q != "ad"]
    main13 = [n for n in C.ALL13 if n in names]                  # bootstrap 只给 13 个主模型（-nv 是次要分析，只报点估计）
    for n in main13:
        draws = {}
        for q, H in cells:
            y = Y[:, Q.index(q), H - 1]
            k = y >= 0
            P = X.wauc_prep(y[k] == 1, R[n][k, Q.index(q), H - 1])
            draws[f"{q}_{H}y"] = X.wauc(P, W[:, p_idx[k]].astype(float))
        draws["part1_2y_4endpoints"] = np.mean([draws[f"{q}_2y"] for q in PART1], 0)
        draws["part3_5y_6questions"] = np.mean([draws[f"{q}_5y"] for q in Q], 0)
        boot[n] = draws
        for key in ("ad_2y", "ad_5y", "part1_2y_4endpoints", "part3_5y_6questions"):
            d = draws[key]
            tgt = out["models"][n][key] if isinstance(out["models"][n][key], dict) else None
            ci = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
            if tgt is not None:
                tgt["auc_ci"] = ci
            else:
                out["models"][n][f"{key}_ci"] = ci
    # 验证性比较 2：排序（ROSMAP 已发表的 2 年 AD vs E 的 2 年 AD），bootstrap 只在 E 一侧
    ros_ref = rosmap_published_ad2y()
    ros = np.array([ros_ref[n] for n in main13])
    e_pt = np.array([out["models"][n]["ad_2y"]["auc"] for n in main13])
    rho_b = [X.spearman(ros, np.array([boot[n]["ad_2y"][b] for n in main13])) for b in range(args.n_boot)]
    out["confirmatory"] = {
        "1_ad_auc": {n: {"ad_2y": [out["models"][n]["ad_2y"]["auc"]] + out["models"][n]["ad_2y"]["auc_ci"],
                         "ad_5y": [out["models"][n]["ad_5y"]["auc"]] + out["models"][n]["ad_5y"]["auc_ci"]} for n in main13},
        "2_rank_spearman_ad_2y": {"rho": X.spearman(ros, e_pt), "ci": [float(np.percentile(rho_b, 2.5)), float(np.percentile(rho_b, 97.5))],
                                  "models": main13}}
    a, b = CONFIRM_PAIR
    d = boot[a]["ad_2y"] - boot[b]["ad_2y"]
    out["confirmatory"]["3_xgb_l2c_minus_lr_cur_ad_2y"] = {
        "diff": out["models"][a]["ad_2y"]["auc"] - out["models"][b]["ad_2y"]["auc"],
        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}
    # 描述性：每个模型减 lr-cur（2 年 AD、4 终点 2 年）
    out["paired_vs_lr_cur"] = {n: {k: {"diff": (out["models"][n][k]["auc"] if isinstance(out["models"][n][k], dict) else out["models"][n][k])
                                           - (out["models"]["lr-cur"][k]["auc"] if isinstance(out["models"]["lr-cur"][k], dict) else out["models"]["lr-cur"][k]),
                                       "ci": [float(np.percentile(boot[n][k] - boot["lr-cur"][k], 2.5)),
                                              float(np.percentile(boot[n][k] - boot["lr-cur"][k], 97.5))]}
                                   for k in ("ad_2y", "part1_2y_4endpoints")} for n in main13 if n != "lr-cur"}

    # ---------------------------------------------------------------- 按中心：随机效应汇总（2 年、5 年 AD）
    out["centers"] = {}
    for q, H in KEY_CELLS:
        y = Y[:, Q.index(q), H - 1]
        res = {}
        for n in names:
            est, se, per = [], [], {}
            for c in np.unique(center):
                m = (center == c) & (y >= 0)
                yc = y[m] == 1
                if yc.sum() < 20 or (~yc).sum() < 20:
                    continue
                pc = R[n][m, Q.index(q), H - 1]
                a_ = X.auc(yc, pc)
                pp = np.unique(ii[m], return_inverse=True)[1]
                P = X.wauc_prep(yc, pc)
                Wc = np.stack([np.bincount(rng.integers(0, pp.max() + 1, pp.max() + 1), minlength=pp.max() + 1)[pp]
                               for _ in range(200)]).astype(float)
                bs = np.clip(X.wauc(P, Wc), 1e-4, 1 - 1e-4)
                lg = np.log(bs / (1 - bs))
                est.append(np.log(a_ / (1 - a_)))
                se.append(float(lg.std()))
                per[int(c)] = {"n": int(m.sum()), "n_pos": int(yc.sum()), "auc": a_}
            r = X.reml_meta(est, se)
            r["per_center"] = per
            res[n] = r
        out["centers"][f"{q}_{H}y"] = res

    # ---------------------------------------------------------------- 重新校准（H 上拟合，E 上套用）
    with np.load(os.path.join(C.OUT, "data", "H.npz")) as z:
        DH = {k: z[k] for k in z.files}
    iH, vH = np.where(DH["M"] & DH["primary"][:, None])
    YH = DH["Y"].reshape(*DH["M"].shape, B.NQ, B.NH)[iH, vH]
    out["recalibrated"] = {}
    out["H"] = {}
    for n in names:
        RH = load_scores("H", n, iH, vH)
        out["H"][n] = summary_aucs(YH, RH)
        rc = {}
        for q in Q:
            for H in (2, 5):
                j = Q.index(q)
                k = YH[:, j, H - 1] >= 0
                a_, b_ = X.recal_fit(YH[k, j, H - 1] == 1, RH[k, j, H - 1])
                ke = Y[:, j, H - 1] >= 0
                c = X.calib(Y[ke, j, H - 1] == 1, X.recal_apply(R[n][ke, j, H - 1], a_, b_))
                rc[f"{q}_{H}y"] = {"alpha": a_, "beta": b_, "o_e": c["o_e"], "cal_slope": c["cal_slope"],
                                   "cal_intercept": c["cal_intercept"], "brier": c["brier"]}
        out["recalibrated"][n] = rc

    # ---------------------------------------------------------------- 差距分解：ROSMAP val -> ROSMAP val 的 NACC 视图 -> NACC E
    Z = np.load(os.path.join(C.RUN_DIR, "newmodel_data_trans.npz"))
    iv, vvv = np.where(Z["val_M"])
    YR = Z["val_Y"][iv, vvv].reshape(-1, B.NQ, B.NH)
    out["decomposition"] = {}
    for n in names:
        row = {}
        try:
            row["rosmap_val"] = summary_aucs(YR, load_scores("rosmap", n, iv, vvv))
        except (FileNotFoundError, KeyError):
            row["rosmap_val"] = None
        try:
            row["rosmap_val_naccview"] = summary_aucs(YR, load_scores("rosmap_nv", n, iv, vvv))
        except (FileNotFoundError, KeyError):
            row["rosmap_val_naccview"] = None
        row["nacc_E"] = summary_aucs(Y, R[n])
        out["decomposition"][n] = row

    # ---------------------------------------------------------------- 人群差多远：ROSMAP val vs NACC E 的站点（只用输入特征）
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    import pretrain as PRT
    DV = PRT.nacc_view(B.split(Z, "val"))                   # 先把 ROSMAP 也变成 NACC 视图：否则只看缺失标记就能完全分开
    FR = B.featrows(DV, iv, vvv, "cur")
    FE = B.featrows(DE, ii, vv, "cur")
    Fm = StandardScaler().fit_transform(np.concatenate([FR, FE]))
    ym = np.r_[np.zeros(len(FR)), np.ones(len(FE))]
    pm = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"), Fm, ym, cv=5,
                           method="predict_proba")[:, 1]
    out["membership_c_statistic"] = {"auc": X.auc(ym == 1, pm), "n_rosmap_val": int(len(FR)), "n_nacc_E": int(len(FE)),
                                     "features": "cur (current visit 54 + static 25), ROSMAP side through pretrain.nacc_view", "cv": 5}

    out["finished_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    json.dump(out, open(res_path, "w"), indent=1, default=float)
    print("confirmatory:", json.dumps(out["confirmatory"], default=float)[:2000], flush=True)
    print(f"wrote {res_path}  sha256 {C.sha256(res_path)}", flush=True)


if __name__ == "__main__":
    main()
