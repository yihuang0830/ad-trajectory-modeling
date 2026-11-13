"""run_rollout_delphi.py -- 一个 Delphi 模型的 fig3 / fig4 原始结果（人级，只留在 RIS）。

    python run_rollout_delphi.py --name delphi-nodedup --device cuda

产出 $ZOO_RUN_DIR/rollout/<name>/：
  fidelity_persons.json  每人从第一次访视模拟 5 年（n_fid 条轨迹）的汇总：逐年 AD / 死亡累计概率、
                         逐年 MMSE<24 / 认知<-1 的比例（分子分母）、相邻两年分箱变化（分子分母）、
                         4 个终点的 2 年 rollout 风险
  perturb_persons.csv    扰动队列里每人 × 每个扰动：rollout 的 2 / 5 年风险（主 seed，同一串随机数）、
                         每个 seed 直接读出的 2 年风险（站点后 1 天的查询 token，同 score_delphi）
汇总和画图在 rollout_metrics.py（所有模型共用）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import DATA_DIR, DELPHI_DIR, DAYS_PER_YEAR, ENDPOINTS, RUN_DIR, label_names, post_id  # noqa: E402
from registry import DELPHI, seeds_of  # noqa: E402
import rollout_delphi as RD  # noqa: E402
import score_delphi as SD  # noqa: E402
from perturbations import ELIG, PERTS  # noqa: E402

EP = list(ENDPOINTS)


def crn_seed(pid):
    return int(pid) % 2_147_483_647


def fidelity_row(S, years=5):
    n = len(S["death"])
    r = {}
    for k in range(1, years + 1):
        r[f"cif_ad_{k}"] = float(np.mean((S["ad"] <= k) & (S["ad"] < S["death"])))
        r[f"cif_death_{k}"] = float(np.mean(S["death"] <= k))
    for fam, key in (("mmse", "mmse_state"), ("cogn", "cogn_state")):
        st = S[key]
        for k in range(1, years + 1):
            alive = S["death"] > k
            d = alive & ~np.isnan(st[:, k])
            r[f"{fam}_low_den_{k}"] = int(d.sum())
            r[f"{fam}_low_num_{k}"] = int((d & (st[:, k] == 0)).sum())
        num = den = 0
        for k in range(years):
            ok = (S["death"] > k + 1) & ~np.isnan(st[:, k]) & ~np.isnan(st[:, k + 1])
            c = int((ok & (st[:, k] != st[:, k + 1])).sum())
            r[f"{fam}_chg_den_k{k}"], r[f"{fam}_chg_num_k{k}"] = int(ok.sum()), c      # 第 k 年 -> 第 k+1 年
            den += int(ok.sum())
            num += c
        r[f"{fam}_chg_num"], r[f"{fam}_chg_den"] = num, den
    # 访视节奏：活着的第 k 年里一个 token 都没有的比例；token 年龄之间的间隔（天）
    for k in range(1, years + 1):
        alive = S["death"] > k
        r[f"notok_den_{k}"] = int(alive.sum())
        r[f"notok_num_{k}"] = int((alive & ~S["tok_year"][:, k - 1]).sum())
    g = S["gaps"]
    r["gap_n"], r["gap_sum"], r["gap_sumsq"], r["gap_gt400"] = int(len(g)), float(g.sum()), float((g ** 2).sum()), int((g > 400).sum())
    for e in EP:
        t = S[e]
        r[f"risk2y_{e}"] = float(np.mean((t <= 2) & ((t < S["death"]) if e != "death" else True)))
    r["n_mc"] = n
    return r


def risks(S, H):
    out = {}
    n = len(S["death"])
    for e in EP:
        t = S[e]
        hit = (t <= H) & ((t < S["death"]) if e != "death" else True)
        out[e] = (hit.sum() + 0.5) / (n + 1)          # 平滑：64 条轨迹里 0 次也能取 logit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_fid", type=int, default=100)
    ap.add_argument("--n_pert", type=int, default=64)
    ap.add_argument("--skip_fidelity", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 人（调试）")
    args = ap.parse_args()
    reg = {m[0]: m for m in DELPHI}
    name, dirs, dataset = reg[args.name][0], reg[args.name][1], reg[args.name][2]
    out_dir = os.path.join(RUN_DIR, "rollout", name)
    os.makedirs(out_dir, exist_ok=True)
    ck42 = os.path.join(DELPHI_DIR, dirs[0], "ckpt.pt")
    traj = json.load(open(os.path.join(RUN_DIR, "traj_truth_val.json")))
    cohort = pd.read_csv(os.path.join(RUN_DIR, "perturb_cohort_val.csv"))
    if args.limit:
        keep = sorted(traj)[: args.limit]
        traj = {k: traj[k] for k in keep}
        cohort = cohort[cohort.pid.astype(str).isin(keep)]
    info = {"name": name, "dataset": dataset, "ckpt": ck42, "device": args.device,
            "n_fid": args.n_fid, "n_pert": args.n_pert}

    gen_ok = True
    try:
        eng = RD.make_engine(ck42, dataset, args.device)
        info["pos_align"] = eng.pos_align
        info["visit_sizes"] = eng.visit_sizes_path
    except Exception as ex:          # visit_heads、snapshot：生成器明确拒绝
        gen_ok = False
        info["generator_error"] = f"{type(ex).__name__}: {ex}"
        print("!! 不能生成轨迹：", info["generator_error"], flush=True)
    tr = eng if gen_ok else _NameOnly(dataset)          # 把扰动翻译成 token 只需要词表和发射规则
    info["mmse_per_visit"], info["cogn_per_visit"] = tr.mmse_per_visit, tr.cogn_per_visit
    info["stroke_per_visit"] = tr.stroke_per_visit
    info["stroke_early_rule"] = "every visit from the first to the landmark" if tr.stroke_per_visit else "first visit only"
    streams = RD.person_streams(dataset)

    # ------------------------------------------------------------ fig3：从第一次访视模拟 5 年
    if gen_ok and not args.skip_fidelity:
        t0 = time.time()
        rows = []
        for i, pid in enumerate(sorted(traj, key=int)):
            ages, toks = streams[int(pid)]
            S = RD.simulate(eng, ages, toks, traj[pid]["t0"], args.n_fid, seed=crn_seed(pid) + 1)
            r = fidelity_row(S)
            r["pid"] = int(pid)
            rows.append(r)
            if i % 100 == 0:
                print(f"  fidelity {i}/{len(traj)}  {time.time() - t0:.0f}s", flush=True)
        json.dump(rows, open(os.path.join(out_dir, "fidelity_persons.json"), "w"))
        info["fidelity_minutes"] = (time.time() - t0) / 60

    # ------------------------------------------------------------ fig4：扰动
    t0 = time.time()
    recs, pseudo, lands = [], [], []
    for _, c in cohort.iterrows():
        pid = int(c.pid)
        ages, toks = streams[pid]
        visits = [v[0] for v in traj[str(pid)]["visits"]]
        for pert in PERTS:
            if not bool(c[ELIG[pert]]):
                continue
            a, k, tl = RD.apply(tr, pert, ages, toks, int(c.tL), int(c.t_first), visits=visits)
            rec = {"pid": pid, "pert": pert, "tland": int(tl)}
            if gen_ok:
                S = RD.simulate(eng, a, k, tl, args.n_pert, seed=crn_seed(pid))
                for H in (2, 5):
                    for e, v in risks(S, H).items():
                        rec[f"roll{H}_{e}"] = float(v)
            j = len(recs)
            m = a <= tl
            pseudo.append(np.stack([np.full(int(m.sum()), j), a[m].astype(np.int64), k[m] - 1], 1))
            lands.append((j, int(tl)))
            recs.append(rec)
        if len(recs) % 500 < len(PERTS):
            print(f"  perturb {len(recs)} arms  {time.time() - t0:.0f}s", flush=True)
    info["perturb_rollout_minutes"] = (time.time() - t0) / 60

    # 直接读出（每个 seed）：同一批伪记录，站点后 1 天的查询 token
    arr = np.concatenate(pseudo).astype(np.int64)
    land = pd.DataFrame(lands, columns=["pid", "t"])
    data, hist_len = SD.pseudo_records(arr, land)
    q_age = land.t.to_numpy() + 1
    post = post_id(label_names(dataset))
    tok_post = [post[ENDPOINTS[e][1]] for e in EP]
    for d, s in zip(dirs, seeds_of(dirs)):
        ck = os.path.join(DELPHI_DIR, d, "ckpt.pt")
        if not os.path.exists(ck):
            continue
        model, ckd = SD.load_model(ck, args.device)
        cfg = ckd.get("config", {})
        lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
        bsz = ckd["model_args"]["block_size"]
        pos_len = SD.pos_align_length(dataset, cfg, bsz, lr)[0] if ckd["model_args"].get("pos_embedding") else None
        lg = SD.query_logits(model, data, hist_len, q_age, bsz, tok_post, args.device, lifestyle_range=lr, pos_len=pos_len)
        risk = -np.expm1(-np.exp(lg) * 2 * DAYS_PER_YEAR)
        for j, rec in enumerate(recs):
            for ei, e in enumerate(EP):
                rec[f"dir_s{s}_{e}"] = float(lg[j, ei])          # logit（速率），排序/效应用
                rec[f"dirrisk_s{s}_{e}"] = float(risk[j, ei])
    pd.DataFrame(recs).to_csv(os.path.join(out_dir, "perturb_persons.csv"), index=False)
    info["perturb_total_minutes"] = (time.time() - t0) / 60
    json.dump(info, open(os.path.join(out_dir, "run_info.json"), "w"), indent=1, default=str)
    print("done", json.dumps(info, default=str), flush=True)


class _NameOnly:
    """生成器加载失败时（visit_heads、snapshot），扰动仍要按 token 改：只需要词表和"是否每次访视都发"。"""

    def __init__(self, dataset):
        names = label_names(dataset)
        self.tok = {n: i for i, n in enumerate(names)}
        meta = json.load(open(os.path.join(DATA_DIR, dataset, "meta.json")))
        rep = set(int(t) for t in (meta.get("repeatable_tokens_postshift") or []))
        self.mmse_per_visit = self.tok["MMSE_normal"] in rep
        self.cogn_per_visit = self.tok["COGN_high"] in rep
        self.stroke_per_visit = self.tok["STROKE"] in rep


if __name__ == "__main__":
    main()
