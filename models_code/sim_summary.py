"""sim_summary.py -- 不能 rollout 的基线 vs 它们的 rollout 版，一张对比表（只读打包盒里的聚合 JSON）。

    python sim_summary.py                 # RIS：读 $ZOO_RUN_DIR/packages，写 packages/_rollout_versions/summary.{json,md}
    python sim_summary.py --local         # 本地：读仓库里的打包盒（models_self/、representative/，见 registry.pkg_dir）
    python sim_summary.py --pkg <目录>    # 读一个平铺的 packages 目录（每个模型一个子目录）
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unified_eval"))
import models as MZ  # noqa: E402
from report_sections import pv2_logor_scores, traj_summary  # noqa: E402


def _j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def row(d):
    m, t, tr, pv = (_j(os.path.join(d, f)) for f in ("metrics.json", "transitions.json", "trajectory.json", "perturb_v2.json"))
    if not m:
        return None
    r = {"auc2": m["macro_auc"]["2y"], "ci2": m["macro_auc_ci_2y"],
         "oe": sum(m["endpoints"][f"{e}_2y"]["o_e"] for e in ("ad", "death", "mmse", "cogn")) / 4}
    if t:
        r["auc5"] = t["macro"]["5y"]["auc"]
    if tr and tr.get("applicable") and tr.get("matrix_5y"):
        s = traj_summary(tr)
        r.update(gap=s["matrix_gap"], ad_sim=s["ad_sim"], ad_obs=s["ad_obs"], d_sim=s["death_sim"], d_obs=s["death_obs"])
    if pv and (pv.get("direct") or {}).get("status") == "ok":
        lo, (n_sync, n_def) = pv2_logor_scores(pv["direct"])
        r["pv2"] = [lo["right"], lo["no_response"], lo["wrong"]]
        e = pv["direct"]["effects"].get("cog2", {}).get("ad", {})
        r["cog2_ad"] = e.get("d_rmst") if e.get("defined") else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=None)
    ap.add_argument("--local", action="store_true")
    args = ap.parse_args()
    if args.local:
        from registry import pkg_dir
        zoo = os.path.dirname(HERE)
        loc, pkg = (lambda n: os.path.join(zoo, pkg_dir(n))), os.path.join(zoo, "models_self")
    else:
        if args.pkg:
            pkg = args.pkg
        else:
            from common import RUN_DIR
            pkg = os.path.join(RUN_DIR, "packages")
        loc = lambda n: os.path.join(pkg, n)        # noqa: E731
    out = []
    for orig, sim in MZ.SIM_OF.items():
        o, s = row(loc(orig)), row(loc(sim))
        out.append({"orig": orig, "sim": sim, "orig_metrics": o, "sim_metrics": s,
                    "status": "ok" if (o and s) else ("missing sim" if o else "missing orig")})
    od = os.path.join(pkg, "_rollout_versions")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "summary.json"), "w"), indent=1)
    f = lambda x, k="{:.3f}": "—" if x is None else k.format(x)     # noqa: E731
    L = ["| 原模型 | rollout 版 | 2 年 AUC 原 → rollout | 5 年分期 AUC 原 → rollout | O/E 原 → rollout | 5 年矩阵差（点） | "
         "10 年 AD 模拟 / 实际 | 10 年死亡 模拟 / 实际 | 扰动 对/无/错：原 → rollout |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        o, s = r["orig_metrics"] or {}, r["sim_metrics"] or {}
        pv = lambda x: "—" if not x.get("pv2") else "/".join(str(v) for v in x["pv2"])   # noqa: E731
        L.append(f"| {r['orig']} | {r['sim']} | {f(o.get('auc2'))} → {f(s.get('auc2'))} | {f(o.get('auc5'))} → {f(s.get('auc5'))} | "
                 f"{f(o.get('oe'), '{:.2f}')} → {f(s.get('oe'), '{:.2f}')} | {f(s.get('gap') and 100 * s['gap'], '{:.0f}')} | "
                 f"{f(s.get('ad_sim'), '{:.0%}')} / {f(s.get('ad_obs'), '{:.0%}')} | {f(s.get('d_sim'), '{:.0%}')} / {f(s.get('d_obs'), '{:.0%}')} | "
                 f"{pv(o)} → {pv(s)} |")
    open(os.path.join(od, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"{sum(r['status'] == 'ok' for r in out)}/{len(out)} pairs complete -> {od}")


if __name__ == "__main__":
    main()
