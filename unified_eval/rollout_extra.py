"""rollout_extra.py -- 第二轮审查（2026-10-08）后的补跑，一个能生成轨迹的 Delphi 模型。人级产物只留在 RIS。

    python rollout_extra.py --name delphi-nodedup

1. fig3 重新生成（同一串随机数 crn_seed(pid)+1，同样 n_fid 条）：fidelity_row 现在多记逐年的变化对、
   "活着的这一年里一个 token 都没有"、token 间隔。旧字段和原来的 fidelity_persons.json 逐人核对
   （旧文件另存 fidelity_persons_v1.json），核对结果写进 run_info_extra.json。
2. 观测一侧：每个人自己的观测流（**同一份分词**）当成一条轨迹，用同一个读出（rollout_delphi.observed +
   fidelity_row）-> obs_persons.json。fig3 面板 B / C 两边因此是同一种读法（原来观测一侧只数"两年都测了"的
   访视对，模拟一侧数沿用上一年状态的每一年，完美的生成器也只能得 0.5–0.9）。
3. 卒中反号的机制检验：不加卒中的对照组在生成时也禁采 STROKE（orig_nostroke），同一串随机数 crn_seed(pid)。
   另把不禁采的 orig 组重跑一遍，核对能不能逐人复现 perturb_persons.csv（共同随机数还成立吗）。
   -> perturb_extra.csv
4. （--query_start）"禁止重复"被否定之后的第二个假设：rollout 第一步从前缀**最后一个 token** 读，加卒中把 STROKE 放在
   那次访视的末尾，第一步就是从一个卒中 token 的位置读的（训练里几乎没有访视以卒中结尾）。两组都在站点后 1 天加一个
   No-event 查询 token（和直接读出同一个位置）再生成，同一串随机数 -> perturb_extra_q.csv（orig_q / stroke_q）。
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DELPHI_DIR, RUN_DIR  # noqa: E402
from registry import DELPHI  # noqa: E402
import rollout_delphi as RD  # noqa: E402
from perturbations import ELIG  # noqa: E402
from run_rollout_delphi import crn_seed, fidelity_row, risks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 人（调试；不覆盖正式文件）")
    ap.add_argument("--query_start", action="store_true", help="只跑第 4 项（从查询 token 开始生成）")
    args = ap.parse_args()
    reg = {m[0]: m for m in DELPHI}
    name, dirs, dataset = reg[args.name][0], reg[args.name][1], reg[args.name][2]
    out_dir = os.path.join(RUN_DIR, "rollout", name)
    old_info = json.load(open(os.path.join(out_dir, "run_info.json")))
    n_fid, n_pert = int(old_info["n_fid"]), int(old_info["n_pert"])
    eng = RD.make_engine(os.path.join(DELPHI_DIR, dirs[0], "ckpt.pt"), dataset, args.device)
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    cohort = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"))
    pids = sorted(traj, key=int)
    sfx = ""
    if args.limit:
        pids = pids[: args.limit]
        cohort = cohort[cohort.pid.astype(str).isin(pids)]
        sfx = f"_limit{args.limit}"
    streams = RD.person_streams(dataset)
    info = {"name": name, "dataset": dataset, "n_fid": n_fid, "n_pert": n_pert, "host": os.uname().nodename}
    if args.query_start:
        query_start(eng, streams, cohort, out_dir, n_pert, sfx, info)
        return

    # ---- 1 + 2：fig3 重新生成 + 观测一侧
    t0 = time.time()
    rows, obs = [], []
    for i, pid in enumerate(pids):
        ages, toks = streams[int(pid)]
        S = RD.simulate(eng, ages, toks, traj[pid]["t0"], n_fid, seed=crn_seed(pid) + 1)
        r = fidelity_row(S)
        r["pid"] = int(pid)
        rows.append(r)
        o = fidelity_row(RD.observed(eng, ages, toks, traj[pid]["t0"]))
        o["pid"] = int(pid)
        obs.append(o)
        if i % 100 == 0:
            print(f"  fidelity {i}/{len(pids)}  {time.time() - t0:.0f}s", flush=True)
    info["fidelity_minutes"] = (time.time() - t0) / 60
    old_path = os.path.join(out_dir, "fidelity_persons.json")
    v1_path = os.path.join(out_dir, "fidelity_persons_v1.json")
    old = {r["pid"]: r for r in json.load(open(v1_path if os.path.exists(v1_path) else old_path))}
    bad = [r["pid"] for r in rows if any(r[k] != v for k, v in old[r["pid"]].items() if k != "pid")]
    info["fidelity_v1_fields_reproduced"] = len(bad) == 0
    info["fidelity_persons_not_reproduced"] = len(bad)
    print("  旧字段逐人复现：", "全部一致" if not bad else f"{len(bad)} 人不一致", flush=True)
    if not args.limit:
        if not os.path.exists(v1_path):
            shutil.copy(old_path, v1_path)
        json.dump(rows, open(old_path, "w"))
    json.dump(obs, open(os.path.join(out_dir, f"obs_persons{sfx}.json"), "w"))

    # ---- 3：卒中反号的机制检验
    t1 = time.time()
    pp = pd.read_csv(os.path.join(out_dir, "perturb_persons.csv"))
    orig_old = pp[pp.pert == "orig"].set_index("pid")
    recs, same = [], []
    for _, c in cohort.iterrows():
        if not bool(c[ELIG["stroke"]]):
            continue
        pid = int(c.pid)
        ages, toks = streams[pid]
        a, k, tl = RD.apply(eng, "orig", ages, toks, int(c.tL), int(c.t_first))
        for arm, block in (("orig_rerun", ()), ("orig_nostroke", ("STROKE",))):
            S = RD.simulate(eng, a, k, tl, n_pert, seed=crn_seed(pid), block=block)
            rec = {"pid": pid, "pert": arm, "tland": int(tl)}
            for H in (2, 5):
                for e, v in risks(S, H).items():
                    rec[f"roll{H}_{e}"] = float(v)
            recs.append(rec)
            if arm == "orig_rerun":
                # 风险是 (k+0.5)/(n+1) 的格点，真不一样至少差 1/(n+1)；容差只为吸收 CSV 读回的末位误差
                same.append(all(abs(rec[f"roll{H}_{e}"] - orig_old.loc[pid, f"roll{H}_{e}"]) < 1e-9
                                for H in (2, 5) for e in ("ad", "death", "mmse", "cogn")))
                if not same[-1] and len(same) <= 3:
                    print("  orig 组没复现", pid, {f"roll{H}_{e}": (rec[f"roll{H}_{e}"], orig_old.loc[pid, f"roll{H}_{e}"])
                                                 for H in (2,) for e in ("ad", "death")}, flush=True)
    info["orig_arm_reproduced_persons"] = int(sum(same))
    info["orig_arm_persons"] = len(same)
    info["stroke_test_minutes"] = (time.time() - t1) / 60
    pd.DataFrame(recs).to_csv(os.path.join(out_dir, f"perturb_extra{sfx}.csv"), index=False)
    json.dump(info, open(os.path.join(out_dir, f"run_info_extra{sfx}.json"), "w"), indent=1)
    print("done", json.dumps(info), flush=True)


def query_start(eng, streams, cohort, out_dir, n_pert, sfx, info):
    t1 = time.time()
    noev = int(eng.res.NO_EVENT)
    recs = []
    for _, c in cohort.iterrows():
        if not bool(c[ELIG["stroke"]]):
            continue
        pid = int(c.pid)
        ages, toks = streams[pid]
        for arm in ("orig", "stroke"):
            a, k, tl = RD.apply(eng, arm, ages, toks, int(c.tL), int(c.t_first))
            m = a <= tl
            a2, k2 = np.append(a[m], tl + 1.0), np.append(k[m], noev)      # 站点后 1 天的 No-event 查询 token
            S = RD.simulate(eng, a2, k2, tl + 1.0, n_pert, seed=crn_seed(pid))
            rec = {"pid": pid, "pert": arm + "_q", "tland": int(tl)}
            for H in (2, 5):
                for e, v in risks(S, H).items():
                    rec[f"roll{H}_{e}"] = float(v)
            recs.append(rec)
    info["query_start_minutes"] = (time.time() - t1) / 60
    pd.DataFrame(recs).to_csv(os.path.join(out_dir, f"perturb_extra_q{sfx}.csv"), index=False)
    json.dump(info, open(os.path.join(out_dir, f"run_info_extra_q{sfx}.json"), "w"), indent=1)
    print("done", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
