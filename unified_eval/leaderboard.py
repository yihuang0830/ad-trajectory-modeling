"""leaderboard.py -- 把所有打包盒的 metrics.json 汇成一张表 + 一张图（横向对比只看这里）。

    python leaderboard.py      # -> model_zoo/reports/LEADERBOARD.md, model_zoo/reports/leaderboard.png
"""
import json
import os
import sys

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import ENDPOINTS, MAIN_H  # noqa: E402
from registry import REFERENCES, REPORT_DIR, pkg_dir, report_md, zoo_files  # noqa: E402
from transitions import TABLE_H, TRANS  # noqa: E402

# 不进排名的：读分在这个模型上没有定义（见各自 README 的"注意"）
UNRANKED_REASON = {
    "delphi-snapshot": "snapshot：读分位置训练中不受监督",
    "delphi-snapshot-pos": "snapshot：读分位置训练中不受监督",
    "delphi-snapshot-visit1only": "snapshot：读分位置训练中不受监督",
    "delphi-snapshot-staticonly": "snapshot：读分位置训练中不受监督",
    "delphi-dedup-sim": "delphi-dedup 同一个 ckpt 的 rollout 读法（9 模型图用），不另排名",
}
ROLE = {"delphi-nodedup-staticonly-split811": "参照：只看人口学",
        "delphi-fullvisit-visit1only-split811": "参照：只看当前访视",
        "delphi-nodedup-visitheads": "logit 不是速率：AUC 有效，校准和扰动读法无定义"}
NO_CALIB = {"delphi-nodedup-visitheads"}


def load_all():
    out = []
    for p in zoo_files(ZOO, "metrics.json"):
        m = json.load(open(p))
        m["_dir"] = os.path.basename(os.path.dirname(p))
        out.append(m)
    return out


def _j(d, f):
    p = os.path.join(ZOO, pkg_dir(d) if d else REPORT_DIR, f)    # d = "" -> reports/ 里的跨模型汇总
    return json.load(open(p)) if os.path.exists(p) else None


def _f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def table_perturbation(ms):
    """表 2a：模拟逼真度（第二部分；只有能生成轨迹的模型）；表 2b：扰动可信度（第二部分，所有模型同一种读法）。"""
    L = ["## 表 2a：模拟逼真度（第二部分；只有能生成轨迹的模型；EVAL_PROTOCOL.md 第 6 节）", "",
         "两边同一种读法：真实一侧 = 这个人自己的观测流（同一份分词），和生成的轨迹用同一个读出函数；只数那一年还在随访里的人年。"
         "1 = 和真实一样。方括号 = 按人 bootstrap 的 95% CI。", "",
         "| 模型 | 5 年 AD 模拟÷真实（对称删失；原口径） | 5 年死亡 模拟÷真实（对称删失；原口径） | MMSE 箱逐年变化 模拟÷真实 | 认知箱逐年变化 | "
         "一整年没有 token 模拟 / 真实 | 第一次访视 2 年 AUC 直接 → rollout | 扰动经过生成器：斜率；同号 / 反号 / 丢失 / 凭空出现 | "
         "卒中→死亡：rollout / 对照组也禁采卒中 / 从查询 token 开始生成 / 直接读出 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        d = m["_dir"]
        f, p = _j(d, "fidelity.json"), _j(d, "perturb.json")
        if not f:
            continue
        ro = (p or {}).get("rollout") or {}

        def ratio(k):
            x = f.get(k) or {}
            ci = x.get("ratio_ci")
            return _f(x.get("ratio")) + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")

        def cif(kind):
            """对称删失；原口径。AD 的对称口径是 2026-10-12 加的，还没重算的写'待重跑'。"""
            c = f[f"cif_{kind}"]
            return ("待重跑" if "ratio_5y_sym" not in c else _f(c["ratio_5y_sym"])) + "；" + _f(c["ratio_5y"])
        nt = f.get("no_token_year")
        sa = ro.get("sign_agreement")
        st = ((ro.get("stroke_norepeat_test") or {}).get("effects") or {}).get("death")
        L.append(f"| [{d}]({pkg_dir(d)}/README.md) | {cif('ad')} | {cif('death')} | "
                 f"{ratio('change_mmse')} | {ratio('change_cogn')} | "
                 f"{'—' if not nt else '%.2f / %.2f' % (nt['sim'], nt['obs'])} | "
                 f"{f['auc_baseline']['macro_direct']:.3f} → {f['auc_baseline']['macro_rollout']:.3f} | "
                 f"{_f(ro.get('slope_vs_direct'))}；{'—' if not sa else '%d / %d / %d / %d' % (sa['same'], sa['opposite'], sa['lost'], sa['rollout_only'])} | "
                 f"{'—' if not st else '%+.2f / %+.2f / %s / %s' % (st['vs_orig'], st['vs_nostroke'], ('%+.2f [%+.2f, %+.2f]' % (st['query_start'], *st['query_start_ci'])) if 'query_start' in st else '—', _f(st.get('direct_pooled')))} |")
    pending = any("ratio_5y_sym" not in (_j(m["_dir"], "fidelity.json") or {"cif_ad": {"ratio_5y_sym": 0}})["cif_ad"] for m in ms)
    L += ["", "- 对称删失（结论以它为准）：最后一次访视之后超过 1.5 年才去世的人当作失访，和失访后还活着的人一样删失在最后一次访视。"
          "原口径把他们一直跟到去世：死亡那条线把失访之后的死亡也算上，AD 那条线把失访那段观察不到的时间也算成'没得 AD'。"
          + ("AD 的对称口径是 2026-10-12 代码审查后加的，'待重跑' = 还没有在 RIS 上重算。" if pending else ""),
          "- 斜率：rollout 的效应对同一个模型直接读出的效应（同一个合并估计量、同一风险水平）回归；1 = 原样传下去。"
          "逐格（7 个扰动 × 4 个终点）：'有效应' = rollout |log OR| ≥ 0.05 且 CI 不含 0，直接读出 |log OR| ≥ 0.05。两边都有 -> 同号 / 反号；"
          "只有直接读出有 -> 丢失；只有 rollout 有 -> 凭空出现（生成器自己造出来的效应）；其余两边都没有。",
          "- 卒中→死亡：'对照组也禁采卒中' = 不加卒中的那组在生成时也不许再出现卒中（检验'禁止重复'假设，被否定）；"
          "'从查询 token 开始生成' = 两组都在站点后 1 天加一个 No-event 查询 token（和直接读出同一个位置）再生成"
          "（原来生成的第一步从前缀最后一个 token 读，加卒中把 STROKE 放在了那次访视的末尾）。", ""]

    L += ["## 表 2b：扰动可信度（第二部分，单个改动、2 年风险；所有模型同一种读法，可以横向比；EVAL_PROTOCOL.md 第 7 节）", "",
          "直接读出，每个训练 seed 单独算，再平均。'无响应' = |log OR| < 0.05，不算对也不算错。", "",
          "| 模型 | 预设方向 对 / 无响应 / 错（20 格） | 原始 24 格 对 | 数据同号 对（共 {nd} 格） | 剂量单调 | 和数据 r（描述） | seed 波动 |",
          "|---|---|---|---|---|---|---|"]
    nd = next((_j(m["_dir"], "perturb.json")["direction"]["data"]["cells"] for m in ms
               if (_j(m["_dir"], "perturb.json") or {}).get("direction")), 0)
    L[-2] = L[-2].replace("{nd}", str(nd))
    for m in ms:
        d = m["_dir"]
        p = _j(d, "perturb.json")
        if p is None:
            continue
        if not p.get("readout_defined", True):
            L.append(f"| [{d}]({pkg_dir(d)}/README.md) | 读法无定义（lm_head 的 logit 不是速率，风险饱和成 1） | — | — | — | — | — |")
            continue
        D = p["direction"]
        ag = p["agreement"] or {}
        st = p["seed_stability"]["median_sd_over_mean"]
        pr = D["prespec"]
        L.append(f"| [{d}]({pkg_dir(d)}/README.md) | {pr['right']:.1f} / {pr['no_response']:.1f} / {pr['wrong']:.1f} | "
                 f"{D['prespec_original24']['right']:.1f} / 24 | {D['data']['right']:.1f} | {_f(p['dose'].get('monotone_frac'))} | "
                 f"{_f(ag.get('r'))} | {_f(st)} |")
    return L + [""]


def load_trans():
    """所有带 transitions.json 的打包盒。"""
    out = []
    for p in zoo_files(ZOO, "transitions.json"):
        t = json.load(open(p))
        t["_dir"] = os.path.basename(os.path.dirname(p))
        out.append(t)
    return out


def _m(t, H, f="auc"):
    v = (t.get("macro") or {}).get(f"{H}y", {}).get(f)
    return v


def table_transitions(ts):
    """表 4：疾病分期转换（第三部分；EVAL_PROTOCOL.md 第 9–11 节）。所有数字都从 transitions.json 读。"""
    rank_ok = [t for t in ts if t.get("answers_all") and t["_dir"] not in UNRANKED_REASON]
    rest = [t for t in ts if t not in rank_ok]
    rank_ok.sort(key=lambda t: -_m(t, 5))
    rest.sort(key=lambda t: -(_m(t, 5) or 0))
    hs = "/".join(map(str, TABLE_H))
    L = ["## 表 4：疾病分期转换（第三部分；2 / 5 / 10 年）", "",
         "站在每一次访视上，预测 H 年内会不会**第一次越过**一条分期线：MMSE <27、<24，全局认知 <0、<−1，AD 诊断，死亡（6 道题）。"
         "同一份真值（和第一部分逐格对照过）、同一份打分代码。排名按 5 年的 6 题平均 AUC。", "",
         "- '同一当前状态 + 年龄段内' = 只在站点时 MMSE 分期、认知分期、5 岁年龄段都相同的站点之间比。只会读出当前**分期**的预测在这里是 0.5；"
         "剩下的是分期内更细的当前值、背景（APOE、教育、性别）和病史 —— 不全是动态，要和下面的尺子、只看人口学的 Delphi 比。",
         "- '5 年抓到的转变者' = 把 80% 的'5 年内不越线'的站点判为稳定时，'5 年内会越线'的站点里被标出来的比例（6 题平均；"
         "按站点算，不是按人；随机是 20%）。",
         "- 竞争风险敏感性：窗口内没越线就先死了的站点算阴性（主口径只数答案已知的站点，把它们排除）。", "",
         f"| # | 模型 | 6 题平均 AUC（{hs} 年；5 年带 95% CI） | 竞争风险敏感性（{hs} 年） | "
         f"同一当前状态 + 年龄段内 AUC（{hs} 年） | 5 年抓到的转变者 | 备注 |",
         "|---|---|---|---|---|---|---|"]

    def row(t, rank, note):
        if t.get("answers_all"):
            ci = (t["macro"]["5y"].get("auc_ci") or [None, None])
            auc = "%s / **%s** [%s, %s] / %s" % (_f(_m(t, 2), 3), _f(_m(t, 5), 3), _f(ci[0], 3), _f(ci[1], 3), _f(_m(t, 10), 3))
            cr = " / ".join(_f(_m(t, H, "auc_cr"), 3) for H in TABLE_H)
            wa = " / ".join(_f(_m(t, H, "auc_within_state_age"), 3) for H in TABLE_H)
            sp = _m(t, 5, "sens_at_spec80")
            cap = "—" if sp is None else f"{sp:.0%}"
        else:
            auc = cr = wa = cap = "—"
            answered = [q for q in TRANS if t["answered"][q]]
            note = (note + "；" if note else "") + "只答 %d/6 道题（%s 年）" % (len(answered), "/".join(
                str(H) for H in sorted({H for q in answered for H in t["answered"][q]})))
        return f"| {rank} | [{t['_dir']}]({pkg_dir(t['_dir'])}/README.md) | {auc} | {cr} | {wa} | {cap} | {note} |"

    for i, t in enumerate(rank_ok, 1):
        L.append(row(t, str(i), ROLE.get(t["_dir"], "")))
    for t in rest:
        L.append(row(t, "", UNRANKED_REASON.get(t["_dir"], ROLE.get(t["_dir"], ""))))
    ref = next((t for t in ts if "snapshot_baseline_macro" in t), None)
    if ref:
        def bm(B, f):
            return " / ".join(_f((B.get(f"{H}y") or {}).get(f), 3) for H in TABLE_H)
        rulers = [("*快照基线*（训练段逻辑回归：当前 MMSE 分期、认知分期、年龄、性别）", "snapshot_baseline_macro", "不是模型：只会读出'现在在哪个分期、多大年纪'"),
                  ("*连续值快照*（同上，但用当前访视的 MMSE、全局认知连续值）", "snapshot_cont_baseline_macro", "不是模型：读出'现在的分数、多大年纪'，不看病史"),
                  ("*分期基线*（训练段里同一当前状态的发生率）", "stage_baseline_macro", "同一当前状态内按构造是 0.5")]
        for lab, key, why in rulers:
            B = ref.get(key)
            if B:
                sp = (B.get("5y") or {}).get("sens_at_spec80")
                L.append(f"| | {lab} | {bm(B, 'auc')} | {bm(B, 'auc_cr')} | {bm(B, 'auc_within_state_age')} | "
                         f"{'—' if sp is None else f'{sp:.0%}'} | {why} |")
        # 结论句从数据里生成：尺子 vs 参与排名的 Delphi
        delphi = [t for t in rank_ok if t["_dir"].startswith("delphi-")]
        if delphi:
            sb = ref["snapshot_baseline_macro"]["5y"]["auc"]
            above = [t["_dir"] for t in delphi if _m(t, 5) > sb]
            L += ["", ("**5 年 6 题平均：快照基线（%.3f）高于所有参与排名的 Delphi**（最高 %.3f）。" % (sb, max(_m(t, 5) for t in delphi)))
                  if not above else ("5 年 6 题平均高于快照基线（%.3f）的 Delphi：%s。" % (sb, "、".join(above)))]
        cm = ref.get("snapshot_cont_baseline_macro")
        top = rank_ok[0] if rank_ok else None
        if cm and top is not None:
            d5 = round(_m(top, 5), 3) - round(cm["5y"]["auc"], 3)      # 用显示出来的两个数相减，读者自己减也对得上
            if d5 < 0:
                L += ["**连续值快照（%.3f）高于所有参与排名的模型**：排第一的 %s 是 %.3f（%+.3f）；"
                      "同一当前状态 + 年龄段内 %.3f 对 %.3f（5 年）。" % (
                          cm["5y"]["auc"], top["_dir"], _m(top, 5), d5, cm["5y"]["auc_within_state_age"],
                          _m(top, 5, "auc_within_state_age"))]
            else:
                L += ["**连续值快照（%.3f）和排第一的 %s（%.3f）只差 %+.3f**：在这 6 道题上，排名靠前主要靠'用了当前访视的连续值'，"
                      "病史和变化只添了这么多；同一当前状态 + 年龄段内也是 %.3f 对 %.3f（5 年）。" % (
                          cm["5y"]["auc"], top["_dir"], _m(top, 5), d5, cm["5y"]["auc_within_state_age"],
                          _m(top, 5, "auc_within_state_age"))]
    # 计数从 JSON 读（站点 vs 人）
    if ref:
        r2, r10 = ref["results"].get("ad_2y", {}), ref["results"].get("ad_10y", {})
        if r2.get("n_persons_pos") is not None and r10.get("n_persons_pos") is not None:
            L += ["", "- 10 年的阳性多是同一批人在更多站点上重复计数（AD：2 年 %d 个阳性站点来自 %d 人，10 年 %d 个来自 %d 人）。"
                  % (r2["n_pos"], r2["n_persons_pos"], r10["n_pos"], r10["n_persons_pos"])]
    L += ["- Delphi 的直接读出用同一个速率给所有时间窗排序。不同时间窗的 AUC 比的是不同的一批站点（答案已知的），"
          "不能直接比高低。", ""]
    return L


def table_trajectory():
    """表 5：10 年轨迹（第三部分；只有能生成轨迹的模型）。所有数字从 <包>/trajectory.json 读。"""
    from report_sections import traj_summary
    rows = []
    for p in zoo_files(ZOO, "trajectory.json"):
        t = json.load(open(p))
        if t.get("applicable") is False or not t.get("matrix_5y"):
            continue
        rows.append((os.path.basename(os.path.dirname(p)), traj_summary(t)))
    if not rows:
        return []
    rows.sort(key=lambda r: r[1]["matrix_gap"] if r[1]["matrix_gap"] is not None else 9)

    def pc(x):
        return "—" if x is None else f"{x:.0%}"
    L = ["## 表 5：10 年轨迹（第三部分；只有能生成轨迹的模型）", "",
         "每人从第一次访视出发模拟 100 条 10 年的未来，和这个人实际的后续比（每个包的 fig4、fig5）。"
         "按 5 年转移矩阵的平均差距排序（越小越像）。实际一侧每行人数有限，光是抽样误差，完美的模拟器也会差几个百分点；"
         "差距在抽样误差以内的先后没有意义。", "",
         "| 模型 | 5 年转移矩阵：模拟和实际平均每格相差（百分点，三行等权） | 10 年 AD 累计发生率：实际 / 模拟 | 10 年死亡：实际 / 模拟 |",
         "|---|---|---|---|"]
    for name, x in rows:
        gap = "—" if x["matrix_gap"] is None else f"{100 * x['matrix_gap']:.0f}"
        L.append(f"| [{name}]({pkg_dir(name)}/README.md) | {gap} | {pc(x['ad_obs'])} / {pc(x['ad_sim'])} | "
                 f"{pc(x['death_obs'])} / {pc(x['death_sim'])} |")
    return L + ["", "delphi-dedup-sim 的轨迹就是 delphi-dedup 那一行。实际一侧按失访校正"
                "（Aalen–Johansen，EVAL_PROTOCOL.md 第 12 节）。", ""]


def table_perturb_v2():
    """表 6：扰动 v2（第三部分；SHAP 引导、10 年、多终点同步）。描述性，不排名。所有数字从 <包>/perturb_v2.json 读。"""
    rows = []
    for p in zoo_files(ZOO, "perturb_v2.json"):
        J = json.load(open(p))
        J["_dir"] = os.path.basename(os.path.dirname(p))
        rows.append(J)
    if not rows:
        return []

    def eff(X, a, q):
        c = ((X.get("effects") or {}).get(a) or {}).get(q) or {}
        return c.get("d_rmst") if c.get("defined") else None

    def score(X):
        t = X.get("prespec_score") or {}
        return "%s / %s / %s" % (t.get("right", "—"), t.get("no_response", "—"), t.get("wrong", "—"))

    def sync(X):
        sy = X.get("synchrony") or {}
        n_def = [a for a, v in sy.items() if v.get("ad_defined") and v.get("cog_defined")]   # 同 pv2_logor_scores
        n_ok = [a for a in n_def if sy[a].get("synchronous_advance")]
        return f"{len(n_ok)} / {len(n_def)}"

    from report_sections import PV2_LOGOR_H, pv2_logor_scores
    ok = [J for J in rows if (J.get("direct") or {}).get("status") == "ok"]
    for J in ok:
        J["_lo"] = pv2_logor_scores(J["direct"])
    # 按不依赖校准的 log OR 得分排（按年数的得分受 AD 基线风险影响，跨家族不可比；两个都列出来）
    ok.sort(key=lambda J: (-J["_lo"][0]["right"], J["_lo"][0]["wrong"]))
    n_ro = sum((J.get("rollout") or {}).get("status") == "ok" for J in rows)
    L = ["## 表 6：扰动 v2 —— SHAP 引导的反事实，10 年里 AD 和认知分期是不是一起变坏（第三部分；描述性，不排名）", "",
         "EVAL_PROTOCOL.md 第 13 节。扰动队列（第 4 次访视），单个改动 + 组合；效应 = 10 年内平均无事件时间（RMST）的变化（年，**负 = 缩短**；按全部人平均，多出来的病例占大头，不是同一批病例提前了这么多年）。"
         "'预设方向'只数结果出来之前写死的格子；'同步' = AD 和每条有定义的认知线都往坏的方向动（分母 = AD 和至少一条认知线都有定义的扰动）。"
         f"rollout 跑了 {n_ro} 个能生成轨迹的模型（seed 42，每人每臂 64 条轨迹，查询 token 起步，同一串随机数）。", "",
         f"直接读出给两种得分。**按年数**（|ΔRMST| < 0.05 年 = 无响应）是预先定的，但它依赖校准：Delphi 直接读出的 AD 基线风险接近 0"
         f"，同样大的相对效应在 nodedup、"
         f"fullvisit 族只挪动很少的年数，会被判成'无响应'。**按 {PV2_LOGOR_H} 年 log OR**"
         "（|Δ| < 0.05 = 无响应，和表 2b 同一个门槛）不依赖校准，是结果出来之后加的；跨模型家族比较看这一列，表按它排序。", "",
         f"| 模型 | 直接读出，按 {PV2_LOGOR_H} 年 log OR：对 / 无响应 / 错 | 同步（log OR） | 直接读出，按年数：对 / 无响应 / 错 | 同步（年数） | "
         "ΔAD 与 ΔMMSE<24 的 Spearman | MMSE + 认知下降：AD / MMSE<24 的 ΔRMST（年） | 三种共病：AD / 死亡的 ΔRMST（年） | rollout（按年数，36 格）：对 / 无响应 / 错 | rollout 同步 | "
         "rollout，MMSE + 认知下降：AD / MMSE<24 的 ΔRMST（年） |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for J in ok:
        D, R = J["direct"], J.get("rollout") or {}
        sp = (D.get("spearman_ad_vs") or {}).get("mmse_lt24") or {}
        rok = R.get("status") == "ok"
        note = "" if J.get("ranked", True) else "（不排名）"
        tl, (ns, na) = J["_lo"]
        L.append(f"| [{J['_dir']}]({pkg_dir(J['_dir'])}/README.md){note} | {tl['right']} / {tl['no_response']} / {tl['wrong']} | {ns} / {na} | "
                 f"{score(D)} | {sync(D)} | {_f(sp.get('rho'))} | "
                 f"{_f(eff(D, 'cog2', 'ad'))} / {_f(eff(D, 'cog2', 'mmse_lt24'))} | {_f(eff(D, 'comorb3', 'ad'))} / {_f(eff(D, 'comorb3', 'death'))} | "
                 f"{score(R) if rok else '—'} | {sync(R) if rok else '—'} | "
                 f"{(_f(eff(R, 'cog2', 'ad')) + ' / ' + _f(eff(R, 'cog2', 'mmse_lt24'))) if rok else '—'} |")
    bad = [J["_dir"] for J in rows if (J.get("direct") or {}).get("status") != "ok"]
    L += ["", "SHAP（选扰动靶子的依据）见 [SHAP_SUMMARY.png](SHAP_SUMMARY.png)，汇总数字在 SHAP_SUMMARY.json、PERTURB_V2_SUMMARY.json。"
          + ("" if not bad else " 直接读出没有定义的：" + "、".join(bad) + "（logit 不是速率）。"), ""]
    return L


def table_sensitivity(ms):
    """表 3：防偏倚的敏感性分析（4 终点 2 年 AUC 平均）。"""
    L = ["## 表 3：敏感性分析（4 终点 2 年 AUC 平均；EVAL_PROTOCOL.md 第 8 节）", "",
         "| 模型 | 主结果 | IPCW [两端] | 年龄分层 | 每人等权 | 只取第一次访视 | 和 delphi-dedup 的配对差 [95% CI] | val − test [95% CI]（80% 把握可检测的差） |",
         "|---|---|---|---|---|---|---|---|"]
    vt = _j("", "VAL_VS_TEST.json") or {"models": {}}
    for m in ms:
        d = m["_dir"]
        s = _j(d, "sensitivity.json")
        if s is None:
            continue
        mc = s["macro"]
        pdv = s.get("paired_vs_delphi-dedup")
        pds = "（参照）" if pdv is None else f"{pdv['diff']:+.3f} [{pdv['ci'][0]:+.3f}, {pdv['ci'][1]:+.3f}]"
        v = vt["models"].get(d)
        if v is None:
            vts = "—"
        else:
            cis = [r["diff_ci"] for r in v["seeds"].values()]
            vts = "%+.3f [%+.3f, %+.3f]（±%.3f）" % (v["mean_val_minus_test"], min(c[0] for c in cis), max(c[1] for c in cis),
                                                  v.get("mdd_80pct_power", v.get("min_detectable_diff", float("nan"))))
        ip = f"{mc['ipcw']:.3f}" + (f" [{min(mc['ipcw_case1'], mc['ipcw_casep']):.3f}, {max(mc['ipcw_case1'], mc['ipcw_casep']):.3f}]"
                                    if "ipcw_case1" in mc else "")
        L.append(f"| [{d}]({pkg_dir(d)}/README.md) | {mc['main']:.3f} | {ip} | {mc['age_strat']:.3f} | "
                 f"{mc['person_weight']:.3f} | {mc['first_visit']:.3f} | {pds} | {vts} |")
    S = _j("", "SENSITIVITY.json") or {}
    pairs = S.get("pairs") or {}
    L += ["", "IPCW：删失模型只在窗口内没发生事件的站点上拟合；病例权重取决于事件那一刻被观察到的概率，主口径按窗口内删失速率恒定插值，"
          "方括号是两个极端（病例权重 1 和 1/p）。",
          "val − test 只对 split811 的 Delphi 有意义：它们没在 test 段上训练、也没用它挑 checkpoint。方括号是各 seed 差值 95% CI 的并集，"
          "上界就是能排除的乐观偏差；括号里是 80% 把握能检测出的差（2.8 × bootstrap SD，两个 N 人的独立样本）。", ""]
    if pairs:
        L += ["分词之间的配对差（同一批人重抽样 1000 次）：" + "；".join(
            "%s %+.3f [%+.3f, %+.3f]" % (k.replace(" - ", " − "), v["diff"], v["ci"][0], v["ci"][1]) for k, v in pairs.items()) + "。", ""]
    return L


def fig_overview(ms, ts, path):
    """leaderboard.png：每个模型一行、四个面板（figures.leaderboard_figure）。行按第一部分的名次。"""
    import figures
    tby = {t["_dir"]: t for t in ts}
    rows = [(m["_dir"], m, False) for m in ms]
    trajby = {r[0]: _j(r[0], "trajectory.json") for r in rows}
    figures._style()
    roles = {"delphi-nodedup-staticonly-split811": "demographics only", "delphi-fullvisit-visit1only-split811": "current visit only"}
    figures.leaderboard_figure(rows, tby, {k: v for k, v in trajby.items() if v}, set(UNRANKED_REASON), path,
                               roles={k: v for k, v in roles.items() if k in {r[0] for r in rows}})


def main():
    ms = load_all()
    ms.sort(key=lambda m: -m["macro_auc"][f"{MAIN_H}y"])
    L = ["# 横向对比（统一评估，验证集 N 人）", "",
         "题目和打分方法见 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)。每一行都用同一份真值、同一份打分代码；",
         "CI = 按人 bootstrap 1000 次的 95% 区间；'seed 均值 ± sd' 是同一配置不同训练 seed 的波动。", "",
         "- 第一部分，预测（2 年风险）：表 1、表 3；第二部分，给扰动引擎用：表 2a、2b；",
         "- 第三部分，疾病分期转换、10 年轨迹、SHAP 引导的扰动（2026-10-14 加）：表 4、5、6。", "",
         "![leaderboard](leaderboard.png)", "",
         "每个模型一行。A：未来 2 年的 4 个终点（第一部分）。B：6 道分期转换题、5 年（第三部分），"
         "C：模拟出来的 10 年 AD（紫）和死亡（灰）"
         "累计发生率 ÷ 实际（1 = 对）。D：5 年 MMSE 转移矩阵，模拟和实际平均每格差几个百分点（0 = 对；表 5）。C、D 只有能生成轨迹的模型。"
         "行按第一部分的名次；第三部分的名次见表 4。", "",
         "## 表 1：预测（第一部分；2 年风险）", "",
         "| # | 模型 | 4 终点平均 AUC [95% CI] | seed 均值 ± sd | " +
         " | ".join(ENDPOINTS[e][0] for e in ENDPOINTS) + " | 平均 O/E | 备注 |",
         "|---|---|---|---|" + "---|" * len(ENDPOINTS) + "---|---|"]
    k = 0
    for m in ms:
        ep = m["endpoints"]
        name = m["_dir"]
        rank = "" if name in UNRANKED_REASON else str(k := k + 1)
        mc = m.get("macro_auc_ci_2y")
        mac = f"**{m['macro_auc']['2y']:.3f}**" + (f" [{mc[0]:.3f}, {mc[1]:.3f}]" if mc else "")
        sd = ""
        if "across_seeds" in m:
            a = m["across_seeds"][f"macro_{MAIN_H}y_auc"]
            sd = f"{a['mean']:.3f} ± {a['sd']:.3f} (n={len(a['values'])})"
        oe = np.mean([ep[f"{e}_{MAIN_H}y"]["o_e"] for e in ENDPOINTS])
        oe_s = "—" if name in NO_CALIB else f"{oe:.2f}"
        note = UNRANKED_REASON.get(name, ROLE.get(name, ""))
        L.append(f"| {rank} | [{name}]({pkg_dir(name)}/README.md) | {mac} | {sd} | " +
                 " | ".join(f"{ep[f'{e}_{MAIN_H}y']['auc']:.3f}" for e in ENDPOINTS) + f" | {oe_s} | {note} |")
    L += ["", "O/E = 实际发生数 / 预测风险之和（1 = 总量上校准；>1 = 低估风险）。",
          "没有名次的行：数字只作记录，原因见备注。", ""]
    L += table_perturbation(ms) + table_sensitivity(ms)
    ts = load_trans()
    if ts:
        L += table_transitions(ts)
    L += table_trajectory() + table_perturb_v2()
    with open(os.path.join(ZOO, REPORT_DIR, "LEADERBOARD.md"), "w") as f:
        f.write(report_md("\n".join(L)))

    fig_overview(ms, ts, os.path.join(ZOO, REPORT_DIR, "leaderboard.png"))
    n = len(ms)
    print("wrote LEADERBOARD.md + leaderboard.png,", n, "models")


if __name__ == "__main__":
    main()
