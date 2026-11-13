"""ext_train.py -- 第 1 节：只用 ROSMAP 训练 11 个非 Delphi 模型，存盘（PROTOCOL.md 第 8 节第 7 条）。

    python ext_train.py --models xgb-l2c lr-cur          # -> $ZOO_RUN_DIR/nacc_ext/models/<名字>.{pt,json,risks.npz}
    python ext_train.py --models transformer --nv        # 次要分析：NACC 视图（NACC 没有的变量一律缺失）上训练 -> <名字>-nv

训练和 models_code/run.py 走同一条代码路径、同一个 seed、同一组参数（OVERRIDES = 打包盒那次运行的命令行）：
  表格分类器 = tabular.run_classifier（加了 keep，把选中那组超参的 60 个模型交回来）；其余 = run.train_and_score 返回的生成模型对象。
这个进程只读 ROSMAP 的 newmodel_data_trans.npz，不碰任何 NACC 文件。
存盘：torch.save 整个对象 -> sha256；同时存训练时内存里的 test / val 风险（risks.npz），
ext_score.py --set rosmap 读回模型重打一遍分，和这里逐位对照。和已发表的 val 打分表（scores_trans/<名字>__s0.csv）的差也记下来。
"""
import argparse
import json
import pickle
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import ext_common as C
import bl_data as B
import models as MZ
import run as RUN
from transitions import risk_col

PART1 = ("ad", "death", "mmse_lt24", "cogn_ltm1")


def nacc_view_Z(Z):
    import pretrain as PRT
    Z = dict(Z)
    for s in ("train", "test", "val"):
        v = PRT.nacc_view(B.split(Z, s))
        for k in ("X", "L", "S"):
            Z[f"{s}_{k}"] = v[k]
    return Z


def summary(Y, R):
    """第一部分 4 终点 2 年平均 AUC、AD 2 / 5 年 AUC（Y、R：(站点, 6, 10)）。"""
    out = {f"{q}_{H}y": B.masked_auc(Y[:, B.Q.index(q), H - 1], R[:, B.Q.index(q), H - 1]) for q in PART1 for H in (2, 5)}
    out["part1_2y_4endpoints"] = float(np.mean([out[f"{q}_2y"] for q in PART1]))
    return out


def published(name, lm):
    """打包盒那次运行的 val 打分表，按 (pid, t) 对齐 -> (站点, 6, 10)。"""
    p = os.path.join(C.RUN_DIR, "scores_trans", f"{name}__s0.csv")
    if not os.path.exists(p):
        return None
    tab = pd.read_csv(p).set_index(["pid", "t"])
    cols = [risk_col(q, H) for q in B.Q for H in B.HS]
    return tab.loc[list(zip(lm["pid"].tolist(), lm["t"].tolist())), cols].to_numpy().reshape(-1, B.NQ, B.NH)


def train_one(name, nv, force=False):
    import torch
    out = f"{name}-nv" if nv else name
    path = os.path.join(C.MODEL_DIR, f"{out}.pt")
    if os.path.exists(path) and not force:
        print(f"== {out}: 已经存盘，跳过", flush=True)
        return
    t0 = time.time()
    Z, meta = B.load()
    if nv:
        Z = nacc_view_Z(Z)
    a = SimpleNamespace(**{**dict(n_mc=100, n_mc_lm=100, lm_chunk=1000, n_mc_pert=100, limit=0), **C.OVERRIDES.get(name, {})})
    _, fam, impl, kw, cite, fut = MZ.BY_NAME[name]
    print(f"== {out}  impl={impl}  device={RUN.device()}  args={vars(a)}", flush=True)
    if impl == "clf":
        import tabular as TB
        keep, log = {}, []
        r = TB.run_classifier(kw["kind"], kw["feat"], Z, log, keep=keep)
        (lm_va, R_va), (lm_te, R_te) = r["val"], r["test"]
        obj = {"kind": "clf", **keep}
    else:
        lm_va, R_va, lm_te, R_te, rec, gen, _ = RUN.train_and_score(name, Z, meta, a, None)
        log = rec["log"]
        obj = {"kind": "gen", "impl": impl, "gen": gen}
    obj.update(name=out, base=name, nv=nv, n_mc=a.n_mc_lm, chunk=a.lm_chunk, seed=RUN.SEED, device=RUN.device(),
               trained_on="ROSMAP train (early stopping / grid on ROSMAP test); newmodel_data_trans.npz"
                          + (" via pretrain.nacc_view" if nv else ""))
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    tmp = path + ".tmp"                                   # 先写临时文件再改名：存到一半失败不会留下一个"已存盘"的坏文件
    try:
        torch.save(obj, tmp)
    except (AttributeError, TypeError, pickle.PicklingError):
        # 网络类定义在函数里（build_minimalrnn 等）：普通 pickle 存不了，按值序列化（cloudpickle，装在本目录的 _vendor/）
        import cloudpickle
        torch.save(obj, tmp, pickle_module=cloudpickle)
    os.replace(tmp, path)
    np.savez_compressed(os.path.join(C.MODEL_DIR, f"{out}.risks.npz"), val=R_va, test=R_te,
                        val_pid=lm_va["pid"], val_t=lm_va["t"], test_pid=lm_te["pid"], test_t=lm_te["t"])
    info = {"name": out, "base": name, "impl": impl, "args": kw, "nv": nv, "sha256": C.sha256(path),
            "bytes": os.path.getsize(path), "device": RUN.device(), "torch": torch.__version__, "run_args": vars(a),
            "minutes": (time.time() - t0) / 60, "val": summary(lm_va["Y"], R_va), "test": summary(lm_te["Y"], R_te),
            "log": log}
    P = None if nv else published(name, lm_va)
    if P is not None:
        Rc = np.clip(R_va, 1e-6, 1 - 1e-6)                    # 打分表写出前夹过
        info["published_val"] = summary(lm_va["Y"], P)
        info["vs_published"] = {"max_abs_risk_diff": float(np.abs(Rc - P).max()),
                                "identical": bool(np.allclose(Rc, P, rtol=0, atol=1e-9)),
                                "d_part1_2y_4endpoints": info["val"]["part1_2y_4endpoints"] - info["published_val"]["part1_2y_4endpoints"]}
    json.dump(info, open(os.path.join(C.MODEL_DIR, f"{out}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: info[k] for k in ("name", "sha256", "minutes", "val") if k in info} | {"vs": info.get("vs_published")},
                     default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--nv", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for m in args.models:
        assert m in C.NONDELPHI, f"{m} 不在第 1 节的名单里"
        train_one(m, args.nv, args.force)


if __name__ == "__main__":
    main()
