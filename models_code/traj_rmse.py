"""traj_rmse.py -- 轨迹模型直接比"数值"：站在 val 的每个站点上，用到站点为止的测量预测 1 / 2 / 3 / 5 年后那次访视的 MMSE 和全局认知
（真的测到了才算），报 RMSE / MAE。第一、三部分只看越线和事件，这一张表补上"轨迹本身准不准"（2026-10-22 加）。

    python traj_rmse.py      # RIS：-> $ZOO_RUN_DIR/packages/_traj/rmse.json（聚合）

比的模型（各自的轨迹部分，和打包盒里同一个实现、同一份训练）：
  locf     最近一次测到的值原样往后（参照）
  lme      lmm-joint 的线性混合模型（每个标记一个，随机截距 + 斜率）
  jm       联合估计的双变量线性混合模型（两个标记的随机效应相关），这里只用测量求后验（不加事件历史）
  gpboost  GPBoost 的树提升固定效应 + 随机截距、斜率
  merf     MERF
  gp       分层高斯过程
"""
import json
import os
import time

import numpy as np

import bl_data as B
import simulator as SIM
from bl_data import I_TF
from common import RUN_DIR

OFFSETS = (1, 2, 3, 5)
FAMS = ("mmse", "cogn")


def targets(D, norm, ii, vv):
    """每个站点 + 每个 offset：那一年的访视上测到的值（没有那次访视或没测 = nan）。"""
    import joint as JT
    V = D["X"].shape[1]
    yrs = np.round(D["X"][..., I_TF + 1] * 10).astype(int)
    out = {f: np.full((len(ii), len(OFFSETS)), np.nan) for f in FAMS}
    t0 = yrs[ii, vv]
    for k, h in enumerate(OFFSETS):
        hit = (yrs[ii] == (t0 + h)[:, None]) & D["M"][ii] & (np.arange(V)[None, :] > vv[:, None])
        has = hit.any(1)
        u = hit.argmax(1)
        for f in FAMS:
            y, obs = JT.raw_value(D, f, norm, ii, u)
            out[f][:, k] = np.where(has & obs, y, np.nan)
    return out


def locf(D, norm, ii, vv):
    import joint as JT
    out = {}
    for f in FAMS:
        T, Y, O = JT.landmark_obs(D, ii, vv, f, norm)
        last = np.where(O.any(1), O.shape[1] - 1 - O[:, ::-1].argmax(1), -1)
        v = np.where(last >= 0, Y[np.arange(len(ii)), np.maximum(last, 0)], np.nan)
        out[f] = np.repeat(v[:, None], len(OFFSETS), 1)
    return out


def predict(name, gen, D, norm, ii, vv):
    import joint as JT
    t0 = JT.years(D, ii, vv)
    tt = t0[:, None] + np.array(OFFSETS, float)[None, :]
    age0 = D["X"][ii, 0, I_TF]
    S = D["S"][ii]
    st = {"D": D, "ii": ii, "vv": vv}
    out = {}
    if name == "lme":
        post = gen.posterior(D, ii, vv, norm)
        for f in FAMS:
            m = post[f][0]
            out[f] = np.stack([gen.true_value(f, m, tt[:, k], age0, S)[0] for k in range(len(OFFSETS))], 1)
    elif name == "jm":
        d = gen.pack(D, norm, ii, vv, after_last=False)
        _, m, _ = gen.long_post(d)
        m = m.cpu().numpy()
        be = gen.beta.cpu().numpy()
        for f in FAMS:
            j = JT.J[f]
            Xd = JT.design(tt, np.repeat(age0[:, None], len(OFFSETS), 1), np.repeat(S[:, None, :], len(OFFSETS), 1))
            out[f] = Xd @ be[j] + m[:, None, 2 * j] + m[:, None, 2 * j + 1] * tt
    elif name in ("gpboost", "merf"):
        for f in FAMS:
            m, _ = gen.posterior(f, st, norm)
            fk, _ = gen._fe_grid(f, D, ii, vv, np.array(OFFSETS, float))
            out[f] = fk + m[:, :1] + m[:, 1:] * tt
    elif name == "gp":
        for f in FAMS:
            out[f] = gen.post(f, st, norm, np.array(OFFSETS, float))[0]
    return out


def score(P, Y):
    res = {}
    for f in FAMS:
        res[f] = {}
        for k, h in enumerate(OFFSETS):
            ok = np.isfinite(Y[f][:, k]) & np.isfinite(P[f][:, k])
            e = P[f][ok, k] - Y[f][ok, k]
            res[f][str(h)] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "n": int(ok.sum())}
    return res


def main():
    import generators as G
    import joint as JT
    Z, meta = B.load()
    norm = SIM.norm_info(meta)
    Dtr, Dte, Dva = (B.split(Z, s) for s in ("train", "test", "val"))
    lm = SIM.landmark_start(Z, "val")
    ii, vv = lm["ii"], lm["vv"]
    Y = targets(Dva, norm, ii, vv)
    res = {"offsets_years": list(OFFSETS), "n_val_landmarks": int(len(ii)), "models": {}}
    res["models"]["locf"] = score(locf(Dva, norm, ii, vv), Y)
    for name, make in (("lme", lambda: G.LMMJoint().fit(Dtr, Dte, norm)),
                       ("jm", lambda: JT.JointModel({}, "cpu").fit(Dtr, Dte, norm)),
                       ("gpboost", lambda: JT.MixedJoint({"kind": "gpboost"}).fit(Dtr, Dte, norm)),
                       ("merf", lambda: JT.MixedJoint({"kind": "merf"}).fit(Dtr, Dte, norm)),
                       ("gp", lambda: JT.GPJoint({}, "cpu").fit(Dtr, Dte, norm))):
        t0 = time.time()
        gen = make()
        res["models"][name] = score(predict(name, gen, Dva, norm, ii, vv), Y)
        r = res["models"][name]
        print(f"{name}: MMSE RMSE " + " / ".join(f"{r['mmse'][str(h)]['rmse']:.2f}" for h in OFFSETS)
              + "  cognition " + " / ".join(f"{r['cogn'][str(h)]['rmse']:.3f}" for h in OFFSETS)
              + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    od = os.path.join(RUN_DIR, "packages", "_traj")
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "rmse.json"), "w"), indent=1)
    print("->", od, flush=True)


if __name__ == "__main__":
    main()
