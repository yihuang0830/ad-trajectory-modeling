"""run.py -- 在 RIS 上跑一个（或几个）模型（Delphi 以外）：训练 -> val 打分表 -> 统一评估（第一部分、第三部分）-> 生成模型另跑 10 年轨迹。

    python run.py --models lr-l2c xgb-l2c          # 按 models.BASELINES 的名字
    python run.py --models markov --limit 20 --n_mc 20   # 冒烟：只模拟 20 人 / 20 条，结果进 smoke 子目录，不评估
    python run.py --models transformer --variant geno --seed 3   # 基因实验：打包盒叫 transformer-geno-s3（见 --variant）

产出（人级的只留在 RIS）：
  $ZOO_RUN_DIR/scores{,_trans}/<name>__s0.csv            val 上的打分表
  $ZOO_RUN_DIR/packages/<name>/metrics.json, fig1/fig2   evaluate.py（第一部分）
  $ZOO_RUN_DIR/packages/<name>/transitions.json          evaluate_transitions.py（第三部分）
  $ZOO_RUN_DIR/packages/<name>/trajectory.json           生成模型：rollout_trans.aggregate（第 12 节）
  $ZOO_RUN_DIR/packages/<name>/baseline.json             训练记录：选中的超参、test 段上的分数、用时
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bl_data as B  # noqa: E402
import models as MZ  # noqa: E402
from common import RUN_DIR  # noqa: E402

UE = os.path.join(HERE, "..", "unified_eval")
TAG = "s0"
SEED = 20261020


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_rollout_version(impl, kw, Dtr, Dte, norm, quick=False, seed=0):
    """rollout_versions.py 里的模型：不能 rollout 的基线的 rollout 版（arv、seqsim），自回归的序列网络（seqar、ddhar）。
    quick：冒烟，序列网络只训 2 轮。"""
    import rollout_versions as RV
    if impl == "arv":
        return RV.ARVersion(kw, norm, device(), seed).fit(Dtr, Dte)
    m = {"seqsim": lambda: RV.SeqSimModel(kw["kind"], device()), "seqar": lambda: RV.SeqARModel(kw["kind"], device()),
         "ddhar": lambda: RV.DDHARModel(device())}[impl]()
    return m.fit(Dtr, Dte, seed=seed, max_epochs=2 if quick else 150)


def build_trfx(kw, Dtr, Dte, quick=False, seed=0):
    """TRFX 实验（2026-11-05）的配置模型（trfx/README.md）。CPU、torch 线程数按配置（8）：实验里只有这样 CPU 结果才逐位
    可复现，seed 0 和实验在 val 上评估过的那个模型相同。quick：冒烟，只训 2 轮。"""
    import torch
    sys.path.insert(0, os.path.join(HERE, "trfx"))
    import trfx_models as TX
    cfg = TX.load_config(os.path.join(HERE, "trfx", kw["config"]))
    if quick:
        cfg = TX.merge_config({"train": {"max_epochs": 2}}, cfg)
    torch.set_num_threads(int(cfg["train"]["threads"]))
    m = TX.TRFXModel(cfg, "cpu").fit(Dtr, Dte, seed=seed)
    m.info.update({"config": kw["config"], "cfg_hash": TX.config_hash(cfg), "device": "cpu",
                   "threads": torch.get_num_threads()})
    return m


ROLLOUT_IMPLS = ("msm", "lmm", "ethos", "hmm", "arv", "seqsim", "seqar", "ddhar", "ctmsm", "jm", "mixed", "gp", "seqsimnacc",
                 "ethosnacc", "trfx")                                           # 有 .sim(start, n_mc, rng, norm) 的
EXACT_IMPLS = ("msm", "ctmsm")                                                   # 站点上的风险精确算（不抽样）


def train_and_score(name, Z, meta, args, recsets=None):
    """-> (val 站点表, val 风险, test 站点表, test 风险, 训练记录, 生成模型对象或 None, 扰动记录上的风险 [v1, v2] 或 None)"""
    _, fam, impl, kw, cite, fut = MZ.BY_NAME[name]
    log, gen = [], None
    extras = [recsets[k][:2] for k in ("v1", "v2", "prs") if k in recsets] if recsets else []
    if impl == "clf":
        import tabular as TB
        r = TB.run_classifier(kw["kind"], kw["feat"], Z, log, extras=extras)
    elif impl == "score":
        import tabular as TB
        r = TB.run_score(kw["kind"], Z, meta, log, extras=extras)
    elif impl == "surv":
        import tabular as TB
        r = TB.run_survival(kw["kind"], kw["feat"], Z, log, device=device(), extras=extras)
    elif impl == "seq":
        import sequence as SQ
        r = SQ.run_sequence(kw["kind"], Z, log, device=device(), extras=extras)
    elif impl == "seqn":
        import pretrain as PRT
        r = PRT.run_seq(kw, Z, log, device=device(), extras=extras)
    else:
        import generators as G
        import simulator as SIM
        norm = SIM.norm_info(meta)
        Dtr, Dte = B.split(Z, "train"), B.split(Z, "test")
        rng = np.random.default_rng(SEED + 100 * args.seed)
        if impl == "msm":
            gen = G.MSM(kw["cov"]).fit(Dtr, norm)
            log.append({"n_pairs": gen.n_pairs, "origins_with_lr": sum(k == "lr" for k, _ in gen.rows.values())})
            risks = lambda st, n=None, sm=False: gen.landmark_risks(st, norm)                # noqa: E731
        elif impl == "lmm":
            gen = G.LMMJoint().fit(Dtr, Dte, norm)
            log.append({"lmm": {f: {"beta": p["beta"].tolist(), "G": p["G"].tolist(), "s2": p["s2"]}
                                for f, p in gen.lmm.items()}, "minutes": gen.minutes})
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl == "ar":
            gen = G.ARModels(kw["kind"]).fit(Dtr, Dte)
            log.append({"minutes": gen.minutes, "resid_sd": gen.sd.tolist()})
            risks = lambda st, n=None, sm=False: SIM.risks(G.ar_sim(gen, st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl == "hmm":
            import hmm as HM
            gen = HM.HMM().fit(Dtr, norm)
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl == "ethos":
            import ethos as ETH
            gen = ETH.Ethos(meta, device()).fit(Dtr, Dte)
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl == "minimalrnn":
            dev = device()
            model, info = G.fit_minimalrnn(Dtr, Dte, dev, seed=args.seed, hp=G.minimalrnn_hp(kw))
            log.append(info)
            if kw:
                kw = {**kw, "hp": info["hp"]}                                                   # "tuned" -> 实际用的那组
            gen = (model, dev)
            risks = lambda st, n=None, sm=False: SIM.risks(                                     # noqa: E731
                G.minimalrnn_sim(model, st, n or args.n_mc_lm, rng, norm, dev), sm)
        elif impl == "seqar" and args.variant == "naccft":
            import pretrain as PRT
            gen = PRT.SeqARNACC(kw["kind"], device()).fit(Dtr, Dte, seed=args.seed)
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl in ("arv", "seqsim", "seqar", "ddhar", "trfx"):
            gen = (build_trfx(kw, Dtr, Dte, seed=args.seed) if impl == "trfx"
                   else build_rollout_version(impl, kw, Dtr, Dte, norm, seed=args.seed))
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl in ("seqsimnacc", "ethosnacc"):
            import pretrain as PRT
            gen = (PRT.SeqSimNACC(kw["kind"], device()) if impl == "seqsimnacc" else PRT.EthosNACC(meta, device())).fit(Dtr, Dte)
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        elif impl == "ctmsm":
            import ctmsm as CT
            gen = CT.CTMSM(device()).fit(Dtr, Dte, norm)
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: gen.landmark_risks(st, norm)                # noqa: E731
        elif impl in ("jm", "mixed", "gp"):
            import joint as JT
            gen = {"jm": JT.JointModel, "mixed": JT.MixedJoint, "gp": JT.GPJoint}[impl](kw, device()).fit(Dtr, Dte, norm)
            log.append(gen.info)
            risks = lambda st, n=None, sm=False: SIM.risks(gen.sim(st, n or args.n_mc_lm, rng, norm), sm)   # noqa: E731
        else:
            raise ValueError(impl)
        out = {}
        for s in ("test", "val"):
            st = SIM.landmark_start(Z, s)
            t0 = time.time()
            R = np.zeros((len(st["ii"]), B.NQ, B.NH))
            step = args.lm_chunk
            for a in range(0, len(st["ii"]), step):
                sub = {k: (v[a: a + step] if k in ("ii", "vv", "pid", "t") else v) for k, v in st.items()}
                R[a: a + step] = risks(sub)
            log.append({"landmark_risks": s, "n": len(st["ii"]), "minutes": (time.time() - t0) / 60})
            lm = B.landmarks(B.split(Z, s), "cur")
            assert (lm["pid"] == st["pid"]).all() and (lm["t"] == st["t"]).all()
            out[s] = (lm, R)
        r = {"val": out["val"], "test": out["test"], "chosen": kw}
        if recsets:
            # 扰动：从改过的站点往后模拟（生成模型的"直接读出"就是 rollout），风险平滑成 (k + 0.5) / (n + 1)
            t0 = time.time()
            r["extras"] = []
            for Dp, vL in extras:
                st = {"D": Dp, "ii": np.arange(len(vL)), "vv": np.asarray(vL), "pid": Dp["pid"], "t": Dp["T"][np.arange(len(vL)), vL]}
                R = np.zeros((len(vL), B.NQ, B.NH))
                for a in range(0, len(vL), args.lm_chunk):
                    sub = {k: (v[a: a + args.lm_chunk] if k in ("ii", "vv", "pid", "t") else v) for k, v in st.items()}
                    R[a: a + args.lm_chunk] = risks(sub, args.n_mc_pert, True)
                r["extras"].append(R)
            log.append({"perturbation_records": [len(v) for _, v in extras], "n_mc": args.n_mc_pert,
                        "minutes": (time.time() - t0) / 60})
    lm_va, R_va = r["val"]
    lm_te, R_te = r["test"]
    rec = {"name": name, "family": fam, "citation": cite, "future": fut, "args": kw, "chosen": r["chosen"],
           "test_macro_auc_2_5_10": B.macro_auc(lm_te["Y"], R_te), "test_bce": B.masked_bce(lm_te["Y"], R_te),
           "val_landmarks": int(len(lm_va["pid"])), "log": log}
    return lm_va, R_va, lm_te, R_te, rec, gen, r.get("extras")


def evaluate(name, p1, p3):
    pk = os.path.join(RUN_DIR, "packages", name)
    os.makedirs(pk, exist_ok=True)
    # 打分用和其它模型同一个环境（ad-projection），基线环境里的 scikit-learn 版本不同
    py = os.environ.get("EVAL_PY", sys.executable)
    cmd = [py, os.path.join(UE, "evaluate.py"), "--name", name, "--scores", f"{TAG}={p1}", "--outdir", pk]
    subprocess.run(cmd, check=True)
    subprocess.run([py, os.path.join(UE, "evaluate_transitions.py"), "--name", name, "--outdir", pk,
                    "--scores", f"{TAG}={p3}"], check=True)
    return pk


def trajectories(name, gen, Z, meta, args, rec, out=None):
    import generators as G
    import rollout_trans as RT
    import simulator as SIM
    norm = SIM.norm_info(meta)
    impl = MZ.BY_NAME[name][2]
    st = SIM.first_visit_start(Z, "val")
    if args.limit:
        st = {k: (v[: args.limit] if k in ("ii", "vv", "pid", "t") else v) for k, v in st.items()}
    out = out or name
    rng = np.random.default_rng(SEED + 100 * args.seed + 1)
    t0 = time.time()
    if impl in ROLLOUT_IMPLS:
        sim = gen.sim(st, args.n_mc, rng, norm)
    elif impl == "ar":
        sim = G.ar_sim(gen, st, args.n_mc, rng, norm)
    else:
        sim = G.minimalrnn_sim(gen[0], st, args.n_mc, rng, norm, gen[1])
    minutes = (time.time() - t0) / 60
    info = {"name": out, "split": "val", "seed": SEED + 100 * args.seed + 1, "wall_minutes": minutes,
            "engine": {"simulator": rec["family"], "citation": rec["citation"], "args": rec["args"]}}
    # 完整运行一律写进打包盒（条数少的模型，比如 TabPFN 30 条，也不当冒烟）；实际条数记在 persons.npz 里
    rdir = SIM.write_persons(out, sim, st, args.n_mc, info, path_n_mc=RT.N_MC) if not args.limit else None
    if args.limit:
        print(f"smoke: simulated {len(st['pid'])} persons x {args.n_mc} in {minutes:.2f} min", flush=True)
        R = [SIM.readout(sim, i) for i in range(len(st["pid"]))]
        print("smoke occ year5 (mmse, pooled):", np.sum([r["occ"][0, 5] for r in R], 0).tolist(), flush=True)
        return
    notes = [
        "10-year window from the first visit (t0); year-k cut = t0 + k*365 days (the data grid, same as the truth tables).",
        f"Simulator: {rec['citation']}. Steps one year at a time from the first visit: samples death and AD in the "
        "coming year, then (if alive) the next annual state. See models_code/README.md for each model's step rule.",
        f"{args.n_mc} trajectories per person; seed {SEED + 100 * args.seed + 1}. A trajectory stops at death.",
        "Predicted state at year k: Dead if death <= cut, else carry-forward of the last simulated measured stage "
        "(stage from the value with the tokenization cut points: MMSE 24/27, cognition -1/0), Unknown if none.",
    ]
    RT.aggregate(out, B.DATA_FILE, f"models_code/{name}", 0, RT.N_MC, "val", TAG, notes)
    return rdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n_mc", type=int, default=100, help="第一次访视起点的轨迹数（trajectory.json）")
    ap.add_argument("--n_mc_lm", type=int, default=100, help="每个站点起点的轨迹数（生成模型的打分表）")
    ap.add_argument("--lm_chunk", type=int, default=1000, help="站点起点一次模拟多少个站点（内存）")
    ap.add_argument("--n_mc_pert", type=int, default=100, help="生成模型：每条扰动记录模拟多少条轨迹")
    ap.add_argument("--no_perturb", action="store_true", help="不做扰动评估")
    ap.add_argument("--smoke_persons", type=int, default=0, help="冒烟：只用 train 的前 N 人训练（test 用 N/4）")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只模拟前 N 人，不写打分表、不评估")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--force", action="store_true", help="打包盒里已经有 baseline.json 也重跑（默认跳过：GPU 的短作业跑完了，排在后面的同名长作业就不再重复）")
    ap.add_argument("--cond", action="store_true",
                    help="量化用：生成模型的站点风险改用条件读法（simulator.COND），写成 <名字>-cond 的打分表和打包盒；不做扰动和轨迹")
    ap.add_argument("--redo_before", type=float, default=0,
                    help="unix 时间：baseline.json 比它旧才重跑（成对提交的短作业 / 长作业用同一个值，先跑完的那个算数）")
    ap.add_argument("--variant", choices=("base", "gflag", "geno", "naccft", "geno2", "geno2nx", "geno2i", "geno2s"),
                    help="基因实验（打包盒叫 <名字>-<variant>-s<seed>）：base = 原版数据；geno = S 末尾加 PRS、5 个主成分、"
                         "无基因标志（input_data_geno.py）；gflag = 同一个文件但 PRS 和主成分清零，只留标志（标志本身代表"
                         "早入组的老队列，用来把它的作用和基因分开）。geno 另做 PRS 扰动（perturb.prs_records）。"
                         "naccft（2026-11-10）：原版数据，seqar 模型从同一个 seed 的 NACC 预训练检查点微调（pretrain.SeqARNACC），"
                         "对照是 base。geno2（2026-11-10，两层基因设计文档 的两层）：S 末尾加 APOE 六类的补充列、Tier 1 位点剂量、"
                         "6 个通路 PRS（排除 Tier 1 区域）、5 个主成分、无基因标志（input_data_geno2.py），另做每个基因变量的扰动"
                         "（perturb.geno2_records）；geno2nx = 同上但通路 PRS 不排除（敏感性）；geno2i（2026-11-11）= 和 geno2 同一份数据、"
                         "同一个 seed 重训，扰动另加基因变量 × APOE（ε3/ε3 / ε3/ε4）的 2 × 2 析因臂（交互）；geno2s（2026-11-11）= 同上的数据和 seed，"
                         "配 --save_model 把模型存下来给 geno2_perturb.py 用（打分表写成 -geno2s-，和 -geno2- 逐位比对，确认是同一个模型）")
    ap.add_argument("--save_model", action="store_true",
                    help="训练完把生成模型整个存到 $ZOO_RUN_DIR/models/<打包盒名>.pt（人级产物旁边，只留在 RIS）")
    ap.add_argument("--seed", type=int, default=0, help="训练和模拟的随机种子（0 = 原来的；非 0 只配合 --variant 用）")
    args = ap.parse_args()
    assert args.seed == 0 or args.variant, "--seed 只配合 --variant 用（原版打包盒都是 seed 0）"
    assert args.variant != "naccft" or all(MZ.BY_NAME[n][2] == "seqar" for n in args.models), "naccft 只给 seqar 的模型"
    Z, meta = B.load({"geno": "newmodel_data_trans_geno.npz", "gflag": "newmodel_data_trans_geno.npz",
                      "geno2": "newmodel_data_trans_geno2.npz", "geno2nx": "newmodel_data_trans_geno2nx.npz",
                      "geno2i": "newmodel_data_trans_geno2.npz", "geno2s": "newmodel_data_trans_geno2.npz"}.get(args.variant))
    if args.variant == "gflag":
        for s in ("train", "test", "val"):
            for g in ("prs", "pc1", "pc2", "pc3", "pc4", "pc5"):
                Z[f"{s}_S"][:, meta["static"].index(g)] = 0.0
    recsets = None
    if args.cond:
        import simulator as SIM
        SIM.COND = True
        args.no_perturb = True
    if not args.limit and not args.no_perturb:
        import perturb as PT
        recsets = PT.records(Z, meta)
        if args.variant == "geno":
            recsets["prs"] = PT.prs_records(Z, meta)
        elif args.variant in ("geno2", "geno2nx", "geno2i"):
            recsets["prs"] = PT.geno2_records(Z, meta, interaction=args.variant == "geno2i")
        print("perturbation records:", {k: len(v[1]) for k, v in recsets.items()}, flush=True)
    for name in args.models:
        assert name in MZ.BY_NAME, f"不认识的基线：{name}"
        t0 = time.time()
        out = f"{name}-{args.variant}-s{args.seed}" if args.variant else f"{name}-cond" if args.cond else name
        bj = os.path.join(RUN_DIR, "packages", out, "baseline.json")
        if not args.limit and not args.force and os.path.exists(bj) and os.path.getmtime(bj) >= args.redo_before:
            print(f"== {name}: 已经跑完（packages/{name}/baseline.json），跳过", flush=True)
            continue
        print(f"== {name}", flush=True)
        if args.limit and name in MZ.GENERATIVE:
            import generators as G
            import simulator as SIM
            impl, kw = MZ.BY_NAME[name][2], MZ.BY_NAME[name][3]
            norm = SIM.norm_info(meta)
            Dtr, Dte = B.split(Z, "train"), B.split(Z, "test")
            if args.smoke_persons:
                Dtr = {k: v[: args.smoke_persons] for k, v in Dtr.items()}
                Dte = {k: v[: args.smoke_persons // 4] for k, v in Dte.items()}
            if impl == "msm":
                gen = G.MSM(kw["cov"]).fit(Dtr, norm)
            elif impl == "lmm":
                gen = G.LMMJoint().fit(Dtr, Dte, norm)
            elif impl == "seqar" and args.variant == "naccft":
                import pretrain as PRT
                gen = PRT.SeqARNACC(kw["kind"], device()).fit(Dtr, Dte, seed=args.seed, max_epochs=2)
                print("smoke", name, json.dumps(gen.info, default=str)[:600], flush=True)
            elif impl in ("arv", "seqsim", "seqar", "ddhar", "trfx"):
                gen = (build_trfx(kw, Dtr, Dte, quick=True) if impl == "trfx"
                       else build_rollout_version(impl, kw, Dtr, Dte, norm, quick=True))
                print("smoke", name, json.dumps(gen.info, default=str)[:400], flush=True)
            elif impl == "ar":
                gen = G.ARModels(kw["kind"]).fit(Dtr, Dte)
            elif impl == "hmm":
                import hmm as HM
                gen = HM.HMM(n_iter=5).fit(Dtr, norm)
                print("smoke hmm", gen.info, flush=True)
            elif impl == "ethos":
                import ethos as ETH
                gen = ETH.Ethos(meta, device()).fit(Dtr, Dte, max_epochs=1)
                print("smoke ethos", {k: v for k, v in gen.info.items()}, flush=True)
            else:
                gen = (G.fit_minimalrnn(Dtr, Dte, device(), max_epochs=3, hp=G.minimalrnn_hp(kw))[0], device())
            rec = {"family": MZ.BY_NAME[name][1], "citation": MZ.BY_NAME[name][4], "args": kw}
            trajectories(name, gen, Z, meta, args, rec)
            st = SIM.landmark_start(Z, "val")
            st = {k: (v[: args.limit] if k in ("ii", "vv", "pid", "t") else v) for k, v in st.items()}
            rng = np.random.default_rng(0)
            if impl == "msm":
                R = gen.landmark_risks(st, norm)
            elif impl in ROLLOUT_IMPLS:
                R = SIM.risks(gen.sim(st, 20, rng, norm))
            elif impl == "ar":
                R = SIM.risks(G.ar_sim(gen, st, 20, rng, norm))
            else:
                R = SIM.risks(G.minimalrnn_sim(gen[0], st, 20, rng, norm, gen[1]))
            print("smoke landmark risks mean (q x H=1,2,5,10):",
                  np.round(R[:, :, [0, 1, 4, 9]].mean(0), 3).tolist(), flush=True)
            continue
        lm_va, R_va, lm_te, R_te, rec, gen, R_ex = train_and_score(name, Z, meta, args, recsets)
        if args.save_model:
            md = os.path.join(RUN_DIR, "models")
            os.makedirs(md, mode=0o700, exist_ok=True)
            import rollout_versions as RV
            RV.save_state(gen, os.path.join(md, f"{out}.pt.tmp"))
            back = RV.load_state(os.path.join(md, f"{out}.pt.tmp"))
            Dv = B.split(Z, "val")
            k = np.arange(64)
            pos = np.maximum(Dv["M"][k].sum(1) - 1, 0)
            o1 = gen.outputs(Dv["X"][k], Dv["M"][k], Dv["S"][k], None, pos)
            o2 = back.outputs(Dv["X"][k], Dv["M"][k], Dv["S"][k], None, pos)
            diff = max(float(np.abs(o1[0] - o2[0]).max()), float(np.abs(o1[1] - o2[1]).max()))
            assert diff == 0.0, f"存回读出来的模型输出不一样：{diff}"
            os.replace(os.path.join(md, f"{out}.pt.tmp"), os.path.join(md, f"{out}.pt"))
            print(f"saved model -> models/{out}.pt (reload check: identical outputs)", flush=True)
        p1, p3 = B.write_scores(out, TAG, lm_va["pid"], lm_va["t"], R_va)
        rec["minutes_train_score"] = (time.time() - t0) / 60
        if args.variant:
            rec.update({"out": out, "variant": args.variant, "seed": args.seed, "data_file": B.DATA_FILE})
        print(f"{name}: test macro AUC (6 q x 2/5/10y) {rec['test_macro_auc_2_5_10']:.4f}", flush=True)
        if not args.skip_eval:
            pk = evaluate(out, p1, p3)
            if name in MZ.GENERATIVE and not args.cond:
                trajectories(name, gen, Z, meta, args, rec, out)
            if R_ex:
                import perturb as PT
                PT.write(out, R_ex[0], R_ex[1], recsets, mc=name in MZ.GENERATIVE and MZ.BY_NAME[name][2] not in EXACT_IMPLS)
                if "prs" in recsets:
                    PT.write_prs(out, R_ex[2], recsets["prs"])
            rec["minutes_total"] = (time.time() - t0) / 60
            with open(os.path.join(pk, "baseline.json"), "w") as f:
                json.dump(rec, f, indent=1, default=float)
        print(f"== {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
