"""rollout_landmarks.py -- 一个能生成轨迹的 Delphi 模型的"rollout 版"打分表：每个站点往后生成 100 条 10 年轨迹，
数 H 年内首次越线的条数当风险（和其它模型的 -sim 版、minimalrnn、ethos 同一种读法：models_code/run.py 的 n_mc_lm）。

    python rollout_landmarks.py --name delphi-dedup --workers 16        # 生成（可续跑）+ 写打分表 + 评估
    python rollout_landmarks.py --name delphi-dedup --limit 20 --n_mc 20 # 冒烟：只写人级目录下的 smoke 文件，不评估
    python rollout_landmarks.py --name delphi-dedup --perturb --workers 16   # 扰动 v2（62 格 log OR 打分），见下
    python rollout_landmarks.py --name delphi-dedup --perturb --limit 5      # 扰动冒烟：只核对原样臂和打分表逐位相同
    python rollout_landmarks.py --name delphi-dedup --cond --workers 16      # 条件读法（量化用）-> <名字>-simcond

为什么要这一份：score_delphi 的风险是从下一个 token 的速率直接换算的（1 − exp(−rate·H)），不是生成出来的；
其它模型比的是 rollout 版，Delphi 也要按"生成未来再数"的口径打分，才是同一种比较（用户 2026-10-28：9 模型的图只比 rollout）。

站点 = 主 seed 直接读出打分表（scores/<名字>__<主 seed>.csv）里的全部 (pid, t)，和直接读出同一批。
起步、生成器、读出都和 rollout_trans 一样（查询 token 起步、_run_one / readout），只是起点从"第一次访视"换成每个站点。
随机数：crn_seed(pid) + SEED_OFFSET + t（天），每个站点一条自己的流，不和 rollout_trans（+2）、扰动（+0）、fig3（+1）共用。

产出（$ZOO_RUN_DIR）：
  rollout_landmarks/<名字>/cum.npz            人级（只留在 RIS）：每个站点 (6 题, 10 年) 的首次越线条数
  scores_trans/<名字>-sim__<主 seed>.csv     第三部分打分表（6 题 × 1..10 年）
  scores/<名字>-sim__<主 seed>.csv           第一部分打分表（4 终点 × 1/2/3/5 年）
  packages/<名字>-sim/{metrics,transitions,timing}.json   聚合数（evaluate / evaluate_transitions / timing）

--perturb（2026-10-29 加）：扰动 v2 也按这种读法，和其它生成模型（models_code/perturb.py）同一个口径：
  扰动队列 × perturb_v2 的全部臂（ATOMS + COMBOS），改法 = perturb_v2.translate_tokens（和 Delphi 的 perturb_v2 同一套 token 改法）；
  每条记录从改过的站点生成 n_mc 条，读出 = rollout_trans.readout（和打分表同一个），风险 = (k + 0.5) / (n + 1)。
  随机数 = 打分表那个站点的种子（crn_seed(pid) + SEED_OFFSET + tL），同一个人的所有臂共用（共同随机数），
  所以原样臂的计数和打分表逐位相同（direct_info.json 的回归核对）。
  -> perturb_v2/<名字>-sim/direct.npz（人级）-> perturb_v2.aggregate_model -> packages/<名字>-sim/perturb_v2.json
  perturb_v2.py 自己的 rollout 一栏（10 个臂、64 条、按 ΔRMST 打分）不受影响。
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DELPHI_DIR, HORIZONS, RUN_DIR  # noqa: E402
from registry import DELPHI  # noqa: E402
from rollout_trans import GENERATIVE, N_MC, Q, YEARS, _W, _init_worker, _run_one, simulate_from  # noqa: E402
from transitions import SAME_AS_PART1, risk_col  # noqa: E402

SEED_OFFSET = 3
CLIP = 1e-6                    # 和 bl_data.write_scores 一样：风险 0 / 1 夹到 [1e-6, 1 − 1e-6]


def landmarks(name):
    main = json.load(open(os.path.join(RUN_DIR, "packages", name, "metrics.json")))["main_seed"]
    sc = pd.read_csv(os.path.join(RUN_DIR, "scores", f"{name}__{main}.csv"), usecols=["pid", "t"])
    return sc["pid"].to_numpy(np.int64), sc["t"].to_numpy(np.int64), main


def simulate(args, ckpt, dataset, pid, t, rdir, sfx):
    import multiprocessing as mp
    from run_rollout_delphi import crn_seed
    meta = {"ckpt": ckpt, "dataset": dataset, "n_mc": args.n_mc, "years": YEARS, "seed": f"crn_seed+{SEED_OFFSET}+t",
            "start": "query", "n_landmarks": len(pid), "cond": args.cond}
    part = os.path.join(rdir, f"cum{sfx}.partial.pkl")
    done = {}
    if os.path.exists(part):
        with open(part, "rb") as f:
            m0, d0 = pickle.load(f)
        if m0 == meta:
            done = d0
            print(f"  续跑：已有 {len(done)} 个站点", flush=True)
    jobs = [(int(p), int(a), args.n_mc, (crn_seed(p) + SEED_OFFSET + int(a)) % 2_147_483_647)
            for p, a in zip(pid, t) if (int(p), int(a)) not in done]
    t0w = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(ckpt, dataset)) as pool:
        print(f"  workers {args.workers}  todo {len(jobs)}/{len(pid)}", flush=True)
        # _run_one 只回 pid；按提交顺序取（imap 保序），站点 = jobs[i]
        for i, (_, r) in enumerate(pool.imap(_run_one, jobs, chunksize=1)):
            done[(jobs[i][0], jobs[i][1])] = (r["cum"], r["sec"], r["dfirst"])
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                with open(part + ".tmp", "wb") as f:
                    pickle.dump((meta, done), f)
                os.replace(part + ".tmp", part)
                el = time.time() - t0w
                print(f"  {len(done)}/{len(pid)}  {el / 60:.1f} min  ~{el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min left",
                      flush=True)
    keys = list(zip(pid.tolist(), t.tolist()))
    assert all(k in done for k in keys), "有站点没跑完"
    cum = np.stack([done[k][0] for k in keys])
    sec = np.array([done[k][1] for k in keys])
    dfirst = np.stack([done[k][2] for k in keys])
    np.savez_compressed(os.path.join(rdir, f"cum{sfx}.npz"), pid=pid, t=t, n_mc=args.n_mc, cum=cum, sec=sec, dfirst=dfirst)
    info = dict(meta, wall_minutes=(time.time() - t0w) / 60, cpu_minutes=float(sec.sum() / 60), workers=args.workers)
    json.dump(info, open(os.path.join(rdir, f"run_info{sfx}.json"), "w"), indent=1)
    os.remove(part)
    print("simulated", json.dumps(info), flush=True)
    return cum, dfirst


def write_scores(sim_name, tag, pid, t, R):
    """R (站点, 6, 10) -> 第三部分、第一部分两份打分表（和 bl_data.write_scores 同格式）。"""
    R = np.clip(R, CLIP, 1 - CLIP)
    tab = pd.DataFrame({"pid": pid, "t": t})
    for qi, q in enumerate(Q):
        for k in range(YEARS):
            tab[risk_col(q, k + 1)] = R[:, qi, k]
    p3 = os.path.join(RUN_DIR, "scores_trans", f"{sim_name}__{tag}.csv")
    p1 = os.path.join(RUN_DIR, "scores", f"{sim_name}__{tag}.csv")
    tab.to_csv(p3, index=False)
    t1 = tab[["pid", "t"]].copy()
    for q, e in SAME_AS_PART1.items():
        for H in HORIZONS:
            t1[f"risk_{e}_{H}y"] = tab[risk_col(q, H)]
    t1.to_csv(p1, index=False)
    return p1, p3


def _perturb_one(job):
    """一个人的全部臂：token 改法照 perturb_v2，生成和读出照打分表（同一个种子 -> 原样臂 = 打分表那个站点）。"""
    import perturb_v2 as PV
    pid, tL, t_first, arms, n_mc, seed = job
    ages, toks = _W["streams"][pid]
    out, skipped = [], []
    for arm in arms:
        a, k, tl = PV.translate_tokens(_W["eng"], arm, ages, toks, tL, t_first)
        if a is None:
            skipped.append(arm)
            continue
        out.append((arm, simulate_from(a, k, tl, n_mc, seed)["cum"]))
    return pid, out, skipped


def perturb(args, ckpt, dataset, main_seed, sim_name):
    import multiprocessing as mp
    import perturb_v2 as PV
    from run_rollout_delphi import crn_seed
    C, _ = PV.cohort_table()
    smoke = bool(args.limit) or args.n_mc != N_MC
    if args.limit:
        C = C.iloc[: args.limit]
    rdir = os.path.join(PV.out_base("smoke" if smoke else None), sim_name)
    os.makedirs(rdir, exist_ok=True)
    meta = {"ckpt": ckpt, "dataset": dataset, "n_mc": args.n_mc, "years": YEARS, "seed": f"crn_seed+{SEED_OFFSET}+tL",
            "start": "query", "arms": list(PV.ARMS), "n_persons": len(C)}
    part = os.path.join(rdir, "direct.partial.pkl")
    done = {}
    if os.path.exists(part):
        with open(part, "rb") as f:
            m0, d0 = pickle.load(f)
        if m0 == meta:
            done = d0
            print(f"  续跑：已有 {len(done)} 人", flush=True)
    jobs = [(int(p), int(r.tL), int(r.t_first), tuple(a for a in PV.ARMS if r[f"elig__{a}"]), args.n_mc,
             (crn_seed(int(p)) + SEED_OFFSET + int(r.tL)) % 2_147_483_647) for p, r in C.iterrows() if int(p) not in done]
    t0w = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(ckpt, dataset)) as pool:
        print(f"  perturb workers {args.workers}  todo {len(jobs)}/{len(C)} 人", flush=True)
        for i, (pid, out, skipped) in enumerate(pool.imap_unordered(_perturb_one, jobs, chunksize=1)):
            done[pid] = (out, skipped)
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                with open(part + ".tmp", "wb") as f:
                    pickle.dump((meta, done), f)
                os.replace(part + ".tmp", part)
                el = time.time() - t0w
                print(f"  {len(done)}/{len(C)} 人  {el / 60:.1f} min  ~{el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min left",
                      flush=True)
    assert all(int(p) in done for p in C.index), "有人没跑完"
    rows = [(int(p), arm, int(C.at[p, "tL"]), cum) for p in C.index for arm, cum in done[int(p)][0]]
    pid_a = np.array([r[0] for r in rows], np.int64)
    arm_a = np.array([r[1] for r in rows])
    tl_a = np.array([r[2] for r in rows], np.int64)
    cum = np.stack([r[3] for r in rows]).astype(np.float64)                  # (记录, 6, 10)
    skipped = {}
    for p in C.index:
        for a in done[int(p)][1]:
            skipped[a] = skipped.get(a, 0) + 1
    # 回归：原样臂 = 打分表在这个站点上的值（同一个种子、同一个前缀，逐位相同；打分表夹到 [CLIP, 1 − CLIP]）
    sc = pd.read_csv(os.path.join(RUN_DIR, "scores_trans", f"{sim_name}__{main_seed}.csv"))
    o = pd.DataFrame({"pid": pid_a, "t": tl_a, "j": np.arange(len(rows))})[arm_a == "orig"]
    mm = o.merge(sc, on=["pid", "t"], how="inner", validate="one_to_one")
    R0 = np.clip(cum[mm.j.to_numpy()] / args.n_mc, CLIP, 1 - CLIP)
    dmax = max(float(np.abs(R0[:, qi, k] - mm[risk_col(q, k + 1)].to_numpy()).max()) for qi, q in enumerate(Q) for k in range(YEARS))
    reg = {"pass": bool(len(mm) == int((arm_a == "orig").sum()) and dmax < 1e-9) if not smoke or args.n_mc == N_MC else None,
           "max_abs": dmax, "n": int(len(mm))}
    np.savez_compressed(os.path.join(rdir, "direct.npz"), pid=pid_a, arm=arm_a, tland=tl_a, seeds=np.array([main_seed]),
                        F=((cum + 0.5) / (args.n_mc + 1))[None])
    info = dict(meta, name=sim_name, seeds=[main_seed], status="ok", n_records=len(rows),
                readout="rollout_trans.readout on n_mc simulated futures from the perturbed landmark; F = (k + 0.5) / (n + 1)",
                skipped_prior_token_in_own_stream=skipped, regression_scores_trans=reg,
                wall_minutes=(time.time() - t0w) / 60, workers=args.workers)
    json.dump(info, open(os.path.join(rdir, "direct_info.json"), "w"), indent=1)
    os.remove(part)
    print(f"perturb {sim_name}: {len(rows)} 条  跳过 {skipped}  原样臂 vs 打分表 {reg}", flush=True)
    if reg["pass"] is False:
        raise SystemExit(f"{sim_name}: 原样臂和打分表对不上（max {dmax:.2e}，n {len(mm)}）")
    if smoke:
        return
    meta0 = PV.model_meta
    PV.model_meta = lambda n: (dict(meta0(args.name), readout="rollout") if n == sim_name else meta0(n))
    try:
        J = PV.aggregate_model(sim_name, C, None)
    finally:
        PV.model_meta = meta0
    print(f"wrote {PV.pkg_dir(sim_name, None)}/perturb_v2.json  prespec {J['direct'].get('prespec_score')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="registry.DELPHI 里能生成轨迹的模型名")
    ap.add_argument("--n_mc", type=int, default=N_MC)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个站点（冒烟）")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--perturb", action="store_true", help="扰动 v2（见文件头），打分表要先跑好")
    ap.add_argument("--cond", action="store_true",
                    help="量化用：死亡以外的题用条件读法 cum / (n − 先死的条数)，写成 <名字>-simcond（只评估，不算时间）")
    args = ap.parse_args()
    if args.name not in GENERATIVE:
        raise SystemExit(f"{args.name} 不能生成轨迹；能生成的：{', '.join(GENERATIVE)}")
    reg = {m[0]: m for m in DELPHI}
    ckpt = os.path.join(DELPHI_DIR, reg[args.name][1][0], "ckpt.pt")
    pid, t, main_seed = landmarks(args.name)
    if args.perturb:
        return perturb(args, ckpt, reg[args.name][2], main_seed, f"{args.name}-sim")
    smoke = bool(args.limit) or args.n_mc != N_MC
    if args.limit:
        pid, t = pid[: args.limit], t[: args.limit]
    sfx = (f"_smoke_lim{args.limit}_mc{args.n_mc}" if smoke else "") + ("_cond" if args.cond else "")
    rdir = os.path.join(RUN_DIR, "rollout_landmarks", args.name)
    os.makedirs(rdir, exist_ok=True)
    cum, dfirst = simulate(args, ckpt, reg[args.name][2], pid, t, rdir, sfx)
    R = cum / np.maximum(args.n_mc - dfirst, 1) if args.cond else cum / float(args.n_mc)
    if smoke:
        print("smoke: 2 年风险均值", {q: round(float(R[:, qi, 1].mean()), 3) for qi, q in enumerate(Q)}, flush=True)
        return
    sim_name = f"{args.name}-simcond" if args.cond else f"{args.name}-sim"
    p1, p3 = write_scores(sim_name, main_seed, pid, t, R)
    pk = os.path.join(RUN_DIR, "packages", sim_name)
    os.makedirs(pk, exist_ok=True)
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "evaluate.py"), "--name", sim_name, "--scores", f"{main_seed}={p1}",
                    "--outdir", pk], check=True)
    subprocess.run([py, os.path.join(HERE, "evaluate_transitions.py"), "--name", sim_name, "--outdir", pk,
                    "--scores", f"{main_seed}={p3}"], check=True)
    if not args.cond:
        subprocess.run([py, os.path.join(HERE, "timing.py"), "--name", sim_name], check=True)
    print("wrote", pk, flush=True)


if __name__ == "__main__":
    main()
