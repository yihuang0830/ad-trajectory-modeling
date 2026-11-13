"""ext_score.py -- 读存盘的模型，给一个记录集打分（PROTOCOL.md 第 7 节第 3、4 步）。

    python ext_score.py --name transformer --set rosmap        # 读回核对：ROSMAP test -> val（和 run.py 同一顺序、同一个随机数），
                                                               # 和 ext_train.py 训练时内存里的风险逐位对照
    python ext_score.py --name transformer --set rosmap_nv     # ROSMAP val 的 NACC 视图（差距分解，第 5 节）
    python ext_score.py --name transformer --set H             # NACC H（协调 / 开发段）
    python ext_score.py --name transformer --set E --part 0 --n_parts 4    # NACC E（封存的考卷），分块
    python ext_score.py --name transformer --set E --merge --n_parts 4     # 合并分块 -> scores/E/<名字>.npz

防火墙：这个进程只读特征。ROSMAP / H 的数组先过 features_only（去掉 Y）；E 只读 data/E_X.npz，并断言里面没有任何标签键；
sealed/ 目录这个脚本从不打开。每条记录在站点处截断（ext_common.truncate）。
NACC 只给 primary 的人打分（--cohort secondary = 其余的人，次要队列）。分块 k 的随机数 = default_rng([SEED, 集合编号, k])。
"""
import argparse
import json
import os
import time

import numpy as np

import ext_common as C
import bl_data as B
import run as RUN
import simulator as SIM

SET_CODE = {"rosmap": 0, "rosmap_nv": 1, "H": 2, "E": 3}


def nacc_set(s, cohort):
    path = os.path.join(C.OUT, "data", f"{s}.npz" if s == "H" else "E_X.npz")
    with np.load(path) as z:
        if s == "E":
            assert not any(k.startswith("Y") for k in z.files), "E 的特征文件里有标签"
        D = C.features_only({k: z[k] for k in z.files})
        prim = z["primary"]
    keep = prim if cohort == "primary" else ~prim
    ii, vv = np.where(D["M"] & keep[:, None])
    return D, ii, vv, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", required=True, choices=list(SET_CODE))
    ap.add_argument("--cohort", default="primary", choices=["primary", "secondary"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n_parts", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    sub = args.set if args.cohort == "primary" or args.set.startswith("rosmap") else f"{args.set}_{args.cohort}"
    od = os.path.join(C.SCORE_DIR, sub)
    os.makedirs(od, exist_ok=True)
    final = os.path.join(od, f"{args.name}.npz")
    if args.merge:
        parts = [np.load(os.path.join(od, f"{args.name}.part{k}of{args.n_parts}.npz")) for k in range(args.n_parts)]
        out = {k: np.concatenate([p[k] for p in parts]) for k in ("ii", "vv", "pid", "t", "R")}
        assert len(np.unique(np.stack([out["ii"], out["vv"]]), axis=1).T) == len(out["ii"]), "分块重叠"
        np.savez_compressed(final, **out, model_sha256=parts[0]["model_sha256"])
        print(f"merged {args.n_parts} parts -> {final} ({len(out['ii'])} landmarks)  sha256 {C.sha256(final)}", flush=True)
        return
    obj, mpath = C.load_model(args.name)
    msha = C.sha256(mpath)
    info = json.load(open(os.path.join(C.MODEL_DIR, f"{args.name}.json")))
    assert msha == info["sha256"], "模型文件和训练时记的 sha256 不同"
    meta = json.load(open(os.path.join(C.RUN_DIR, "newmodel_data_trans_meta.json")))
    norm = SIM.norm_info(meta)
    t0 = time.time()
    prog = lambda a, n: print(f"  {a}/{n}  {(time.time() - t0) / 60:.1f} min", flush=True)      # noqa: E731

    if args.set == "rosmap":
        Z, _ = B.load()
        if info.get("nv"):                                  # -nv 模型训练时 test/val 也是 NACC 视图（ext_train.nacc_view_Z），核对也得用它
            import ext_train as TR
            Z = TR.nacc_view_Z(Z)
        rng = np.random.default_rng(RUN.SEED)               # run.train_and_score：先 test 再 val，同一个随机数
        ref = np.load(os.path.join(C.MODEL_DIR, f"{args.name}.risks.npz"))
        rep = {"name": args.name, "model_sha256": msha}
        for s in ("test", "val"):
            D = B.split(Z, s)
            ii, vv = np.where(D["M"])
            R = C.score(obj, C.features_only(D), ii, vv, rng, norm, log=prog)
            Y = D["Y"][ii, vv].reshape(-1, B.NQ, B.NH)
            assert (D["pid"][ii] == ref[f"{s}_pid"]).all() and (D["T"][ii, vv] == ref[f"{s}_t"]).all()
            d = np.abs(R - ref[s])
            rep[s] = {"identical": bool((d == 0).all()), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                      "auc_2y_4ep_reloaded": float(np.mean([B.masked_auc(Y[:, B.Q.index(q), 1], R[:, B.Q.index(q), 1])
                                                            for q in ("ad", "death", "mmse_lt24", "cogn_ltm1")])),
                      "auc_2y_4ep_train_time": float(np.mean([B.masked_auc(Y[:, B.Q.index(q), 1], ref[s][:, B.Q.index(q), 1])
                                                              for q in ("ad", "death", "mmse_lt24", "cogn_ltm1")]))}
            np.savez_compressed(os.path.join(od, f"{args.name}.{s}.npz"), ii=ii, vv=vv, pid=D["pid"][ii], t=D["T"][ii, vv], R=R)
        rep["minutes"] = (time.time() - t0) / 60
        json.dump(rep, open(os.path.join(od, f"{args.name}.check.json"), "w"), indent=1)
        print(json.dumps(rep), flush=True)
        return

    if args.set == "rosmap_nv":
        import pretrain as PRT
        Z, _ = B.load()
        D = C.features_only(PRT.nacc_view(B.split(Z, "val")))
        ii, vv = np.where(D["M"])
        src = "newmodel_data_trans.npz val via pretrain.nacc_view"
    else:
        D, ii, vv, src = nacc_set(args.set, args.cohort)
    idx = np.array_split(np.arange(len(ii)), args.n_parts)[args.part]
    rng = np.random.default_rng([RUN.SEED, SET_CODE[args.set], args.part])
    R = C.score(obj, D, ii[idx], vv[idx], rng, norm, log=prog)
    path = final if args.n_parts == 1 else os.path.join(od, f"{args.name}.part{args.part}of{args.n_parts}.npz")
    np.savez_compressed(path, ii=ii[idx], vv=vv[idx], pid=D["pid"][ii[idx]], t=D["T"][ii[idx], vv[idx]], R=R,
                        model_sha256=msha)
    print(json.dumps({"name": args.name, "set": sub, "source": os.path.basename(src), "part": args.part,
                      "n_parts": args.n_parts, "n": int(len(idx)), "minutes": (time.time() - t0) / 60,
                      "mean_risk_2y": R[:, :, 1].mean(0).round(4).tolist(), "out": path}), flush=True)


if __name__ == "__main__":
    main()
