"""geno2_tier1.py -- 基因第二版（两层基因设计文档 的 Tier 1）：几个大效应位点各自成一个变量。

    python geno2_tier1.py fetch     # 登录节点（要联网）：从 1000 Genomes phase 3 远程取每个位点 ±500 kb 的欧裔基因型 -> _runs/geno/ref/tier1/
    python geno2_tier1.py run       # 作业：选代表变异、抽剂量、核对 APOE -> _runs/geno/tier1_table.csv（人级）+ tier1_qc.json（聚合）

位点和先导变异（GRCh37；来自 geno/bellenguez2022_83_stageI.tsv）：BIN1、CLU、PICALM、SORL1（常见 + 低频各一个）、ABCA7、TREM2（R47H、R62H）。
APOE 不在这里做变量（用临床记录里直接测的基因型，在 input_data_geno2.py 里编成六类），这里只用芯片的 rs429358 / rs7412 核对一致性
（文档里 NACCAPOE 核对的 ROSMAP 版本）。

选代表变异的规则（2026-11-10 和用户定的，看任何结局之前）：
- 低频变异（风险等位基因频率或其互补 < 5%）：按文档要 r² ≥ 0.9，不够就丢，不找替代（罕见变异很少有可靠的搭档，携带者也太少）；
  过了 0.9 的还要次要等位基因的携带者 ≥ 20 人（文档：携带者太少、估不稳的丢掉），然后直接用先导变异的风险等位基因剂量。
- 常见变异：候选 = 先导变异本身 + 1000G 欧裔（503 人）里和它 LD r² ≥ 0.5 的双等位 SNP。
  每个候选的"代表准确度" = 和先导变异的 LD r² × 在 ROSMAP 推断剂量里的 r²（先导变异自己的 LD r² 记 1）。
  取最高的那个；≥ 0.6 就用（文档对常见变异要 0.8，按用户的意思稍微放宽），否则丢掉。
  变量 = 代表变异上"和先导变异风险等位基因同向"的那个等位基因的剂量（0–2）。
不管留没留，所有这些位点 ±500 kb 都会从 Tier 2 的通路 PRS 里排除（geno2_score.py）。
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
from common import RUN_DIR  # noqa: E402
from geno_build import COMP, IMP, OUT, fam  # noqa: E402
from geno2_hm3 import stream  # noqa: E402

TG = os.path.join(OUT, "ref", "tier1")
KG = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
KG_VCF = KG + "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
KG_PANEL = KG + "integrated_call_samples_v3.20130502.ALL.panel"
WIN = 500_000
LD_CAND, SCORE_MIN, RARE_R2, RARE_FREQ, MIN_CARRIERS = 0.5, 0.6, 0.9, 0.05, 20
# 名字、rsID：先导变异（权重表里的）
LEADS = [("BIN1", "rs6733839"), ("CLU", "rs11787077"), ("PICALM", "rs3851179"), ("SORL1", "rs11218343"),
         ("SORL1_rare", "rs74685827"), ("ABCA7", "rs12151021"), ("TREM2_R47H", "rs75932628"), ("TREM2_R62H", "rs143332484")]
APOE_SNPS = {"rs429358": 45411941, "rs7412": 45412079}     # GRCh37；rs429358 C = ε4，rs7412 T = ε2


def leads():
    W = pd.read_csv(os.path.join(HERE, "geno", "bellenguez2022_83_stageI.tsv"), sep="\t").set_index("rsid")
    out = []
    for name, rs in LEADS:
        w = W.loc[rs]
        out.append({"name": name, "rsid": rs, "chr": int(w.chr), "pos": int(w.pos_grch37), "risk": w.risk_allele,
                    "other": w.other_allele, "raf": float(w.raf_stageI), "beta": float(w.beta_stageI)})
    return out


def fetch():
    os.makedirs(TG, mode=0o700, exist_ok=True)
    panel = os.path.join(TG, "panel.txt")
    if not os.path.exists(panel):
        subprocess.run(["curl", "-sSL", "-o", panel, KG_PANEL], check=True)
    P = pd.read_csv(panel, sep="\t")
    eur = P[P.super_pop == "EUR"]["sample"]
    eur.to_csv(os.path.join(TG, "eur.txt"), index=False, header=False)
    regions = [(L["name"], L["chr"], L["pos"]) for L in leads()] + [("APOE", 19, APOE_SNPS["rs429358"])]
    for name, c, pos in regions:
        dst = os.path.join(TG, f"{name}.vcf.gz")
        if os.path.exists(dst):
            continue
        subprocess.run(["bcftools", "view", "-r", f"{c}:{max(pos - WIN, 1)}-{pos + WIN}", "-S", os.path.join(TG, "eur.txt"),
                        "--force-samples", "-m2", "-M2", "-v", "snps", "-Oz", "-o", dst + ".tmp", KG_VCF.format(c=c)],
                       check=True, cwd=TG)
        os.replace(dst + ".tmp", dst)
        print("fetched", name, os.path.getsize(dst), flush=True)
    print("EUR samples", len(eur))


def kg_genotypes(name):
    """1000G 欧裔：每个双等位 SNP 的 ID、位置、REF、ALT、ALT 个数（人 503）。"""
    q = subprocess.run(["bcftools", "query", "-f", "%ID\t%POS\t%REF\t%ALT[\t%GT]\n", os.path.join(TG, f"{name}.vcf.gz")],
                       check=True, capture_output=True, text=True).stdout.splitlines()
    ids, pos, ref, alt, G = [], [], [], [], []
    for ln in q:
        f = ln.split("\t")
        g = np.array([sum(a == "1" for a in x.replace("|", "/").split("/")) for x in f[4:]], np.float32)
        ids.append(f[0]); pos.append(int(f[1])); ref.append(f[2]); alt.append(f[3]); G.append(g)
    return pd.DataFrame({"id": ids, "pos": pos, "ref": ref, "alt": alt}), np.stack(G)


def rs_by_pos(Ls):
    """1000G phase 3 的 VCF 没有 rsID，而 ROSMAP 剂量里有一部分行是 rsID（不带坐标）。用 Bellenguez 汇总统计（rsID + GRCh38 坐标）
    把每个区域里的 rsID 换成 GRCh37 坐标：同一个 ±500 kb 区域里两版坐标差一个常数，常数用区域里的 HapMap3 变异
    （面板有 rsID + GRCh37）定，要求 ≥ 95% 的变异给出同一个差值，否则报错。-> {名字: {GRCh37 位置: [rsID...]}}，核对数字"""
    from geno2_prep import SUMSTATS
    W = pd.read_csv(os.path.join(HERE, "geno", "bellenguez2022_83_stageI.tsv"), sep="\t").set_index("rsid")
    want = {L["name"]: (L["chr"], int(W.loc[L["rsid"]].pos_grch38)) for L in Ls}
    want["APOE"] = (19, 44908684)                                    # rs429358 的 GRCh38 坐标
    parts = []
    for ch in pd.read_csv(SUMSTATS, sep="\t", usecols=["variant_id", "chromosome", "base_pair_location"],
                          chunksize=2_000_000, dtype={"variant_id": str, "chromosome": str}):
        keep = np.zeros(len(ch), bool)
        for c, p38 in want.values():
            keep |= (ch.chromosome == str(c)).to_numpy() & ch.base_pair_location.between(p38 - WIN - 50_000, p38 + WIN + 50_000).to_numpy()
        parts.append(ch[keep])
    S = pd.concat(parts)
    hm3 = pd.read_csv(os.path.join(OUT, "ref", "ldblk_1kg_eur", "snpinfo_1kg_hm3"), sep=r"\s+").set_index("SNP").BP
    out, check = {}, {}
    for name, (c, p38) in want.items():
        s = S[(S.chromosome == str(c)) & S.base_pair_location.between(p38 - WIN - 50_000, p38 + WIN + 50_000)]
        both = s[s.variant_id.isin(hm3.index)].sort_values("base_pair_location")
        p38 = both.base_pair_location.to_numpy()
        off = p38 - hm3.loc[both.variant_id].to_numpy()
        # 两版坐标的差在区域里是分段常数（GRCh38 有局部改动）：每个变异取左右最近的 HapMap3 变异的差，两边一致才换算
        q = s.base_pair_location.to_numpy()
        r = np.clip(np.searchsorted(p38, q, "left"), 0, len(p38) - 1)
        l = np.clip(np.searchsorted(p38, q, "right") - 1, 0, len(p38) - 1)
        ok = off[l] == off[r]
        m = {}
        for rs, p in zip(s.variant_id.to_numpy()[ok], (q - off[l])[ok]):
            m.setdefault(int(p), []).append(rs)
        out[name] = m
        vc = pd.Series(off).value_counts()
        check[name] = {"offsets_38_minus_37": {int(k): int(v) for k, v in vc.head(4).items()}, "hm3_used": int(len(off)),
                       "rsids_in_region": int(len(s)), "rsids_mapped": int(ok.sum())}
    return out, check


def same_alleles(a1, a2, b1, b2):
    """+1：(a1,a2) == (b1,b2)；−1：反过来；含链翻转；0：对不上。"""
    for x1, x2 in ((a1, a2), (a1.translate(COMP), a2.translate(COMP))):
        if (x1, x2) == (b1, b2):
            return 1
        if (x1, x2) == (b2, b1):
            return -1
    return 0


def dosages(c, keys):
    """ROSMAP 推断剂量里 ID 在 keys 里的行 -> {ID: (A1, A2, 剂量)}，剂量数的是 A1。"""
    kf = os.path.join(TG, f"keys_chr{c}.txt")
    with open(kf, "w") as f:
        f.write("\n".join(sorted(keys)) + "\n")
    out = {}
    for ln in stream(c, kf):
        f = ln.split(maxsplit=3)
        out.setdefault(f[0], []).append((f[1].upper(), f[2].upper(), np.array(f[3].split(), np.float32)))
    os.remove(kf)
    return out


def r2_imp(d):
    p = d.mean() / 2
    return float(d.var() / (2 * p * (1 - p))) if 0 < p < 1 else 0.0


def choose(L, K, G, rows, pos2rs):
    """一个常见位点：返回 (选中的候选 dict 或 None, 候选表)。"""
    li = np.where((K.pos == L["pos"]) & K.apply(lambda r: same_alleles(r.ref, r.alt, L["risk"], L["other"]) != 0, axis=1))[0]
    if not len(li):
        return None, [], "lead_not_in_1000G"
    li = li[0]
    lead_risk_is_alt = same_alleles(K.ref[li], K.alt[li], L["risk"], L["other"]) == -1   # ref,alt == other,risk
    g0 = G[li] if lead_risk_is_alt else 2 - G[li]                                      # 先导风险等位基因个数
    cands = []
    for j in range(len(K)):
        if G[j].std() == 0:
            continue
        r = float(np.corrcoef(g0, G[j])[0, 1])
        ld = 1.0 if j == li else r * r
        if ld < LD_CAND:
            continue
        tag_alt = r > 0                                    # ALT 和先导风险等位基因同向？（先导自己 r = ±1）
        best = None
        for key in [f"chr{L['chr']}:{K.pos[j]}"] + pos2rs.get(int(K.pos[j]), []):
            for a1, a2, d in rows.get(key, []):
                s = same_alleles(a1, a2, K.ref[j], K.alt[j])      # +1：文件 A1 = REF
                if not s or (len(a1) == 1 and a1 == a2.translate(COMP)):
                    continue
                d_alt = 2 - d if s == 1 else d                     # ALT 的个数
                d_tag = d_alt if tag_alt else 2 - d_alt
                q = r2_imp(d)
                if best is None or q > best["imp_r2"]:
                    best = {"imp_r2": q, "dose": d_tag}
        if best is None:
            continue
        cands.append({"id": ",".join(pos2rs.get(int(K.pos[j]), [])) or K.id[j], "pos": int(K.pos[j]), "is_lead": bool(j == li), "ld_r2": round(ld, 3),
                      "imp_r2": round(best["imp_r2"], 3), "score": round(ld * min(best["imp_r2"], 1.0), 3),
                      "tag_allele": K.alt[j] if tag_alt else K.ref[j], "dose": best["dose"]})
    if not cands:
        return None, [], "no_candidate_in_rosmap"
    cands.sort(key=lambda x: -x["score"])
    top = cands[0]
    return (top if top["score"] >= SCORE_MIN else None), cands, ("ok" if top["score"] >= SCORE_MIN else "score_below_min")


def apoe_check(rows, ids):
    """芯片推出的 APOE（ε4 个数 = rs429358 C 的个数；ε2 个数 = rs7412 T 的个数，取最近整数）对临床 apoe_genotype。只出聚合数。"""
    dose = {}
    for rs, pos in APOE_SNPS.items():
        allele = {"rs429358": "C", "rs7412": "T"}[rs]
        got = rows.get(rs, []) + rows.get(f"chr19:{pos}", [])
        got = [(a1, a2, d) for a1, a2, d in got if allele in (a1, a2, a1.translate(COMP), a2.translate(COMP))]
        if not got:
            return {"status": f"{rs}_not_in_imputed_data"}
        a1, a2, d = got[0]
        dose[rs] = (d if a1 in (allele, allele.translate(COMP)) else 2 - d, r2_imp(d))
    Z = np.load(os.path.join(RUN_DIR, "newmodel_data_trans.npz"))
    meta = json.load(open(os.path.join(RUN_DIR, "newmodel_data_trans_meta.json")))
    si = meta["static"].index
    pid = np.concatenate([Z[f"{s}_pid"] for s in ("train", "test", "val")])
    S = np.concatenate([Z[f"{s}_S"] for s in ("train", "test", "val")])
    pos = {int(p): i for i, p in enumerate(ids)}
    m = np.array([int(p) in pos for p in pid]) & (S[:, si("apoe_na")] < 0.5)
    k = np.array([pos[int(p)] for p in pid[m]])
    e4_chip = np.rint(dose["rs429358"][0][k])
    e2_chip = np.rint(dose["rs7412"][0][k])
    e4_clin = S[m, si("apoe_e4x1")] + 2 * S[m, si("apoe_e4x2")]
    e2c_clin = S[m, si("apoe_e2carrier")]
    return {"status": "ok", "n_compared": int(m.sum()),
            "imp_r2": {rs: round(v[1], 3) for rs, v in dose.items()},
            "e4_count_agree": round(float((e4_chip == e4_clin).mean()), 4),
            "e2_carrier_agree": round(float(((e2_chip > 0) == (e2c_clin > 0.5)).mean()), 4)}


def run():
    ids = fam(IMP + ".fam")
    res, table = {"rule": {"ld_candidate_min": LD_CAND, "score_min": SCORE_MIN, "rare_r2_min": RARE_R2,
                           "rare_freq": RARE_FREQ, "window_bp": WIN}, "loci": []}, {"projid": ids}
    by_chr = {}
    rsmap, res["grch38_to_37_check"] = rs_by_pos(leads())
    print("rsID 换坐标的核对", res["grch38_to_37_check"], flush=True)
    for L in leads():
        by_chr.setdefault(L["chr"], []).append(L)
    for c, Ls in sorted(by_chr.items()):
        kg = {L["name"]: kg_genotypes(L["name"]) for L in Ls}
        keys = {L["rsid"] for L in Ls} | {f"chr{c}:{L['pos']}" for L in Ls}
        for nm, (K, _) in kg.items():
            keys |= {f"chr{c}:{p}" for p in K.pos} | {r for p in K.pos for r in rsmap[nm].get(int(p), [])}
        if c == 19:
            keys |= set(APOE_SNPS) | {f"chr19:{p}" for p in APOE_SNPS.values()}
        rows = dosages(c, keys)
        for L in Ls:
            rec = {k: L[k] for k in ("name", "rsid", "chr", "pos", "risk", "raf", "beta")}
            rare = min(L["raf"], 1 - L["raf"]) < RARE_FREQ
            if rare:
                # 只认等位基因对得上的行（同一位置可能有别的变异）；剂量换成风险等位基因的个数
                best = None
                for key in (L["rsid"], f"chr{c}:{L['pos']}"):
                    for a1, a2, d in rows.get(key, []):
                        s = same_alleles(a1, a2, L["risk"], L["other"])
                        if s and (best is None or r2_imp(d) > best[0]):
                            best = (r2_imp(d), d if s == 1 else 2 - d)
                q = best[0] if best else 0.0
                minor = None if best is None else (best[1] if L["raf"] < 0.5 else 2 - best[1])     # 次要等位基因的个数
                n_carr = 0 if minor is None else int((minor > 0.5).sum())
                ok = q >= RARE_R2 and n_carr >= MIN_CARRIERS
                rec.update({"rare": True, "imp_r2": round(q, 3), "used": ok, "minor_allele_carriers": n_carr,
                            "why": "ok" if ok else ("rare_low_r2" if q < RARE_R2 else "too_few_carriers"),
                            "expected_carriers": round(2 * min(L["raf"], 1 - L["raf"]) * len(ids), 1)})
                if ok:
                    col = f"t1_{L['name']}"
                    table[col] = best[1]
                    rec.update({"column": col, "mean_dose": round(float(best[1].mean()), 3),
                                "chosen": {"id": L["rsid"], "pos": L["pos"], "is_lead": True, "imp_r2": round(q, 3),
                                           "tag_allele": L["risk"]}})
            else:
                K, G = kg[L["name"]]
                top, cands, why = choose(L, K, G, rows, rsmap[L["name"]])
                rec.update({"rare": False, "used": top is not None, "why": why, "n_candidates": len(cands),
                            "candidates_top5": [{k: v for k, v in x.items() if k != "dose"} for x in cands[:5]]})
                if top is not None:
                    rec["chosen"] = {k: v for k, v in top.items() if k != "dose"}
                    col = f"t1_{L['name']}"
                    table[col] = top["dose"]
                    rec["column"] = col
                    rec["mean_dose"] = round(float(top["dose"].mean()), 3)
            res["loci"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "candidates_top5"}), flush=True)
        if c == 19:
            res["apoe_chip_check"] = apoe_check(rows, ids)
            print("APOE check", res["apoe_chip_check"], flush=True)
    pd.DataFrame(table).to_csv(os.path.join(OUT, "tier1_table.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "tier1_qc.json"), "w"), indent=1)
    print("columns", [k for k in table if k != "projid"])


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
