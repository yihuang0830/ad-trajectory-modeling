"""probe_apoe.py -- 为什么看历史的 Delphi 对 APOE 没反应（2026-10-14 诊断；只打印聚合数，在 RIS 上跑）。见 ../reports/APOE_DIAGNOSIS.md。

    python probe_apoe.py --part1 rosmap:96,rosmap_nodedup:144 \
        --ckpts dedup-s42:Delphi-ROSMAP-gpubase/ckpt.pt:rosmap nojit-s42:Delphi-ROSMAP-nojitter/ckpt.pt:rosmap

part 1  训练时的可见度：按 train.py 的方式调 get_batch（select=left、padding=random），数"预测 AD_DX（和全部临床 token）
        的那个位置能不能看到 APOE / 教育 / 性别 token"，lifestyle 年龄抖动打开（训练时）和关掉各算一遍。
        可见 = 因果、不是 padding、且和目标不同龄（mask_ties=True，所有 ROSMAP ckpt 都这样训）。
unified 和 fig4 完全同一个读出（推荐用这个）：fig4 的三个背景 token 臂 —— apoe（APOE e4 0 -> 1）、male（女 -> 男）、
        educ（受教育 16 年以上 -> 12 年以下）。站点 = 扰动队列的站点（真正的第 4 次访视，fig4 的资格列且 AD 在风险中）和第一次访视；
        RD.apply 改 token，score_delphi 的伪记录 + 查询 token 读 AD_DX 的 logit，报 log 速率比和 2 年 log OR（fig4 的效应量），
        按人 bootstrap。--fig4 delphi-dedup=dedup-s42,dedup-s43,dedup-s44 ...：这几个 ckpt 在第 4 次访视上的 2 年 log OR
        必须逐臂等于 perturb.json 的 seed_vals（同一个读出的回归检查，对不上就报错）。
part 2  （探查用的近似读法，结论以 unified 为准）每个 ckpt：第 k 个不同的 token 年龄（k = 1 基线、k = 4）之后 1 天放一个 No-event 查询 token，
        读 AD_DX 的 logit（= log 速率），比较：
          apoe        APOE e4 0 -> 1，token 留在原位（基线那天）
          apoe_early  同样的替换，但 APOE token 挪到基线前 10 年（训练时抖动常把它放在那里）
          apoe_drop   删掉 APOE token，对比 e4_0
          educ        EDU_16to19/ge20 -> EDU_lt12
          male        Female -> Male
        按人 bootstrap 的 95% CI。注意：去重分词里没有变化的访视不发 token，"第 4 个 token 年龄"常常晚于第 4 次访视；
        也没有限定 AD 在风险中、没有经过 get_batch 的排布 —— 所以它只用来找机制，报数用 unified。
"""
import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import torch

BASE = os.environ.get("AD_BASE", "/path/to/workspace/"
                      "project")
DEL = os.path.join(BASE, "delphi")
sys.path.insert(0, DEL)
from model import Delphi, DelphiConfig  # noqa: E402
from utils import get_p2i, get_batch  # noqa: E402

MASK_TIME = -10000.0
TEN_Y = 10 * 365.25


def vocab(ds):
    names = pd.read_csv(os.path.join(DEL, "data", ds, "labels.csv"))["event_name"].tolist()
    return {n: i for i, n in enumerate(names)}


# ---------------------------------------------------------------- part 1
def part1(ds, block, n_batches, seed):
    I = vocab(ds)
    data = np.memmap(os.path.join(DEL, "data", ds, "train.bin"), dtype=np.uint32, mode="r").reshape(-1, 3)
    p2i = get_p2i(data)
    groups = {
        "APOE": [I[n] for n in ("APOE_e4_0", "APOE_e4_1", "APOE_e4_2", "APOE_unk")],
        "EDU": [I[n] for n in ("EDU_lt12", "EDU_12to15", "EDU_16to19", "EDU_ge20")],
        "SEX": [I["Female"], I["Male"]],
    }
    AD = I["AD_DX"]
    ignore = np.array([0] + list(range(2, 34)))
    print(f"\n[part 1] {ds}  block={block}  {n_batches} batches x 128 (select=left, padding=random)")
    for jit in (True, False):
        g = torch.Generator().manual_seed(seed)
        n_ad = n_sup = 0
        c_ad = {k: 0 for k in groups}
        c_sup = {k: 0 for k in groups}
        for _ in range(n_batches):
            ix = torch.randint(len(p2i), (128,), generator=g)
            x, a, y, b = get_batch(ix, data, p2i, select="left", index="patient", padding="random",
                                   block_size=block, device="cpu", lifestyle_augmentations=jit,
                                   lifestyle_token_range=(3, 32), no_event_token_rate=5)
            x, a, y, b = (t.numpy() for t in (x, a, y, b))
            T = x.shape[1]
            tri = np.tril(np.ones((T, T), bool))
            for r in range(x.shape[0]):
                # key k visible to query q: causal, not padding, and not tied with the target's age
                # (mask_ties=True, which is how every ROSMAP ckpt was trained)
                vis = tri & (x[r][None, :] > 0) & (a[r][None, :] != b[r][:, None])
                sup = (y[r] > 1) & ~np.isin(y[r], ignore)
                is_ad = y[r] == AD
                for name, gi in groups.items():
                    has = vis[:, np.isin(x[r], gi)].any(1)
                    c_ad[name] += int(has[is_ad].sum())
                    c_sup[name] += int(has[sup].sum())
                n_ad += int(is_ad.sum())
                n_sup += int(sup.sum())
        lab = "jitter ON (as trained)" if jit else "jitter OFF"
        print(f"  {lab:<24s} AD_DX targets n={n_ad:<6d} " +
              "  ".join(f"{k} visible {c_ad[k] / max(n_ad, 1):.3f}" for k in groups))
        print(f"  {'':<24s} all clinical targets n={n_sup:<6d} " +
              "  ".join(f"{k} visible {c_sup[k] / max(n_sup, 1):.3f}" for k in groups))


# ---------------------------------------------------------------- part 2
def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = dict(ck["model_args"])
    valid = set(DelphiConfig.__dataclass_fields__)
    m = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    m.eval()
    return m, ck.get("iter_num"), ck.get("best_val_loss")


def subjects(ds):
    d = np.fromfile(os.path.join(DEL, "data", ds, "val.bin"), dtype=np.uint32).reshape(-1, 3)
    for s, n in get_p2i(d):
        rows = d[s:s + n]
        ages = rows[:, 1].astype(np.float64)
        toks = rows[:, 2].astype(np.int64) + 1          # disk -> model id
        o = np.argsort(ages, kind="stable")
        yield ages[o], toks[o]


@torch.no_grad()
def logits_at_query(model, seqs, col, bs=256):
    """seqs: list of (ages, toks) histories; appends a No-event query at last age + 1 day."""
    out = np.empty(len(seqs))
    for s0 in range(0, len(seqs), bs):
        chunk = seqs[s0:s0 + bs]
        L = max(len(t) for _, t in chunk) + 1
        X = np.zeros((len(chunk), L), np.int64)
        A = np.full((len(chunk), L), MASK_TIME, np.float32)
        for i, (a, t) in enumerate(chunk):
            n = len(t)
            X[i, L - n - 1:L - 1] = t
            A[i, L - n - 1:L - 1] = a
            X[i, -1] = 1                                  # No event
            A[i, -1] = a.max() + 1
        lg, _, _ = model(torch.from_numpy(X), torch.from_numpy(A))
        out[s0:s0 + len(chunk)] = lg[:, -1, col].double().numpy()
    return out


def boot(d, rng, n=2000):
    d = np.asarray(d, float)
    if d.size < 2:
        return float("nan"), float("nan"), float("nan")
    bs = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def part2(name, ckpt, ds, rng):
    I = vocab(ds)
    model, it, bvl = load(os.path.join(DEL, ckpt))
    AD = I["AD_DX"]
    dose = [I["APOE_e4_0"], I["APOE_e4_1"], I["APOE_e4_2"]]
    apoe_all = dose + [I["APOE_unk"], I["APOE_e2carrier"]]
    hi_edu = [I["EDU_16to19"], I["EDU_ge20"]]
    print(f"\n[part 2] {name}  ckpt={ckpt}  iter={it}  best_val_loss={bvl}")
    for k in (1, 4):
        pairs = {p: ([], []) for p in ("apoe", "apoe_early", "apoe_drop", "educ", "male")}
        for ages, toks in subjects(ds):
            u = np.unique(ages)
            if len(u) < k:
                continue
            keep = ages <= u[k - 1]
            a, t = ages[keep], toks[keep]
            base = u[0]
            st = set(t.tolist())
            if any(d in st for d in dose) and I["APOE_e2carrier"] not in st:
                t0 = t.copy(); t0[np.isin(t0, dose)] = I["APOE_e4_0"]
                t1 = t.copy(); t1[np.isin(t1, dose)] = I["APOE_e4_1"]
                pairs["apoe"][0].append((a, t0)); pairs["apoe"][1].append((a, t1))
                ae = a.copy(); ae[np.isin(t, apoe_all)] = base - TEN_Y
                o = np.argsort(ae, kind="stable")
                pairs["apoe_early"][0].append((ae[o], t0[o])); pairs["apoe_early"][1].append((ae[o], t1[o]))
                m = ~np.isin(t, apoe_all)
                pairs["apoe_drop"][0].append((a, t0)); pairs["apoe_drop"][1].append((a[m], t[m]))
            if any(e in st for e in hi_edu):
                te = t.copy(); te[np.isin(te, hi_edu)] = I["EDU_lt12"]
                pairs["educ"][0].append((a, t)); pairs["educ"][1].append((a, te))
            if I["Female"] in st:
                tm = t.copy(); tm[tm == I["Female"]] = I["Male"]
                pairs["male"][0].append((a, t)); pairs["male"][1].append((a, tm))
        print(f"  query after visit {k}:")
        for p, (lo, hi) in pairs.items():
            if not lo:
                continue
            d = logits_at_query(model, hi, AD) - logits_at_query(model, lo, AD)
            m, c0, c1 = boot(d, rng)
            print(f"    {p:<11s} n={len(d):<4d} ΔAD_DX logit {m:+.3f} [{c0:+.3f},{c1:+.3f}]   mean|Δ| {np.abs(d).mean():.3f}")


# ---------------------------------------------------------------- unified：和 fig4 同一个读出
# fig4 的臂名 -> (扰动队列里的资格列, 第一次访视的资格：到那天为止的 token 里有其中之一)
U_ARMS = {"apoe": ("elig_apoe", ("APOE_e4_0",)),
          "male": ("elig_sex", ("Female",)),
          "educ": ("elig_educ", ("EDU_16to19", "EDU_ge20"))}


def unified(specs, n_boot=2000, seed=20261014):
    """specs: [(label, ckpt_rel_to_delphi, dataset)]。返回 {label: {arm: {visit4: {...}, visit1: {...}}}}（只有聚合数）。"""
    sys.path.insert(0, os.path.join(BASE, "model_zoo", "unified_eval"))
    import rollout_delphi as RD
    import score_delphi as SD
    from run_rollout_delphi import _NameOnly
    from common import DAYS_PER_YEAR
    run = os.path.join(BASE, "model_zoo", "_runs")
    coh = pd.read_csv(os.path.join(run, "perturb_cohort_val.csv"))
    traj = json.load(open(os.path.join(run, "traj_truth_val.json")))
    out = {}
    cache = {}
    for label, ck_rel, ds in specs:
        if ds not in cache:
            tr = _NameOnly(ds)
            streams = RD.person_streams(ds)
            recs = {}
            for arm, (col, need) in U_ARMS.items():
                sites = {}
                # 第 4 次访视：fig4 的队列和资格（资格列 = 到站点为止有可改的 token；AD 在风险中）
                c4 = coh[coh[col].astype(bool) & coh.risk_ad.astype(bool)]
                sites["visit4"] = [(int(r.pid), int(r.tL), int(r.t_first)) for r in c4.itertuples()]
                # 第一次访视：这份分词里到第一次访视为止有可改的 token、那时还没诊断 AD 的人
                ids = [tr.tok[n] for n in need]
                v1 = []
                for pid, d in traj.items():
                    a, k = streams[int(pid)]
                    t0 = int(d["t0"])
                    if np.isin(k[a <= t0], ids).any() and (d["ad"] is None or d["ad"] > t0):
                        v1.append((int(pid), t0, t0))
                sites["visit1"] = v1
                for nm, rows in sites.items():
                    pseudo, lands = [], []
                    for pid, t, tf in rows:
                        a0, k0 = streams[pid]
                        for pert in ("orig", arm):
                            a, k, tl = RD.apply(tr, pert, a0, k0, t, tf)
                            m = a <= tl
                            j = len(lands)
                            pseudo.append(np.stack([np.full(int(m.sum()), j), a[m].astype(np.int64), k[m] - 1], 1))
                            lands.append((j, int(tl)))
                    arr = np.concatenate(pseudo).astype(np.int64)
                    land = pd.DataFrame(lands, columns=["pid", "t"])
                    data, hist = SD.pseudo_records(arr, land)
                    recs[(arm, nm)] = (data, hist, land.t.to_numpy() + 1, len(rows))
            cache[ds] = (tr, recs)
        tr, recs = cache[ds]
        ck = os.path.join(DEL, ck_rel)
        model, ckd = SD.load_model(ck, "cpu")
        cfg = ckd.get("config", {})
        lr = (cfg.get("lifestyle_token_min", 3), cfg.get("lifestyle_token_max", 32))
        bsz = ckd["model_args"]["block_size"]
        assert not ckd["model_args"].get("pos_embedding"), "pos 模型要对齐长度，这里没接"
        res = {}
        for (arm, nm), (data, hist, q_age, n) in recs.items():
            rng = np.random.default_rng(seed)          # 每个 (臂, 站点) 一样的重抽：CI 和 ckpt 的顺序、分组无关
            lg = SD.query_logits(model, data, hist, q_age, bsz, [tr.tok["AD_DX"]], "cpu", lifestyle_range=lr)[:, 0]
            lo, lp = lg[0::2], lg[1::2]
            r_o = -np.expm1(-np.exp(lo) * 2 * DAYS_PER_YEAR)
            r_p = -np.expm1(-np.exp(lp) * 2 * DAYS_PER_YEAR)
            dlog = lp - lo
            dor = np.log(r_p / (1 - r_p)) - np.log(r_o / (1 - r_o))
            bi = rng.integers(0, n, (n_boot, n))
            r = {"n": int(n), "d_lograte": float(dlog.mean()),
                 "d_lograte_ci": [float(np.percentile(dlog[bi].mean(1), 2.5)), float(np.percentile(dlog[bi].mean(1), 97.5))],
                 "d_logor_2y": float(dor.mean()),
                 "d_logor_2y_ci": [float(np.percentile(dor[bi].mean(1), 2.5)), float(np.percentile(dor[bi].mean(1), 97.5))]}
            res.setdefault(arm, {})[nm] = r
            print(f"  {label:<22s} {arm:<5s} {nm}: n={n:<4d} Δlog-rate {r['d_lograte']:+.3f} "
                  f"[{r['d_lograte_ci'][0]:+.3f},{r['d_lograte_ci'][1]:+.3f}]   Δ2y logOR {r['d_logor_2y']:+.3f}", flush=True)
        out[label] = res
    return out


def check_fig4(res, spec):
    """spec = '<打包盒>=<label>,<label>,...'（按 perturb.json 的 seeds 顺序）：这些 ckpt 在第 4 次访视上的 2 年 log OR
    必须逐臂等于 fig4 的 seed_vals，人数也要相同。"""
    pkg, labels = spec.split("=")
    pj = json.load(open(os.path.join(BASE, "model_zoo", "_runs", "packages", pkg, "perturb.json")))
    for arm in U_ARMS:
        e = pj["effects"][arm]["ad"]
        got = [res[lb][arm]["visit4"]["d_logor_2y"] for lb in labels.split(",")]
        ns = {res[lb][arm]["visit4"]["n"] for lb in labels.split(",")}
        assert ns == {e["n"]}, f"{arm}: 人数 {ns} 和 fig4 的 {e['n']} 不同"
        assert np.allclose(got, e["seed_vals"], rtol=0, atol=1e-6), f"{arm}: {got} != fig4 {e['seed_vals']}"
        print(f"  fig4 回归检查 {pkg} {arm}: n={e['n']}，{len(got)} 个 seed 一致", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", default="rosmap:96")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--ckpts", nargs="*", default=[], help="name:ckpt_rel_to_delphi:dataset")
    ap.add_argument("--unified", nargs="*", default=[], help="name:ckpt_rel_to_delphi:dataset，用和 fig4 相同的读出")
    ap.add_argument("--out", default=None, help="unified 的聚合结果写到这里（json）")
    ap.add_argument("--fig4", nargs="*", default=[], help="unified 的回归检查，例如 delphi-dedup=dedup-s42,dedup-s43,dedup-s44")
    a = ap.parse_args()
    if a.unified:
        res = unified([tuple(x.split(":")) for x in a.unified])
        for spec in a.fig4:
            check_fig4(res, spec)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        return
    torch.set_num_threads(8)
    rng = np.random.default_rng(0)
    if a.part1:
        for spec in a.part1.split(","):
            ds, blk = spec.split(":")
            part1(ds, int(blk), a.n_batches, seed=0)
    for spec in a.ckpts:
        nm, ck, ds = spec.split(":")
        part2(nm, ck, ds, rng)


if __name__ == "__main__":
    main()
