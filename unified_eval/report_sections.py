"""report_sections.py -- 打包盒 README 里各节的文字（make_package 和 models_code/compare.py 共用）。

2026-10-15 改版：每节一张图（figures.py 画）+ 两三行关键数字，细节在 JSON 和 LEADERBOARD 的表里。
  1 预测（fig1、fig2，各自的 README 生成器写）
  2 分期转换（fig3）    3 转移矩阵（fig4）    4 10 年轨迹（fig5）    5 扰动（fig6）    附：敏感性分析（表）
只读打包盒里的聚合 JSON，本地就能重生成。
"""
import json
import math
import os

ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPN = {"ad": "AD", "death": "死亡", "mmse": "MMSE<24", "cogn": "认知<−1"}
EP = list(EPN)


def _f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _j(d, f):
    p = os.path.join(d, f)
    return json.load(open(p)) if os.path.exists(p) else None


# ----------------------------------------------------------------------------- 附：敏感性分析（第一部分，只有表）

def sensitivity_section(d, vvt=None):
    s = _j(d, "sensitivity.json")
    if not s:
        return []
    names = [("main", "主结果"), ("ipcw", "IPCW（失访加权，病例权重取中间设定）"), ("ipcw_case1", "IPCW：病例权重 1（一端）"),
             ("ipcw_casep", "IPCW：病例权重 1/p（另一端）"), ("age_strat", "年龄分层"), ("person_weight", "每人等权"),
             ("first_visit", "只取第一次访视")]
    names = [x for x in names if x[0] in s["macro"]]
    L = ["## 附：敏感性分析（第一部分）", "", "主结果的每个口径选择都换一种重算 2 年 AUC，看结论变不变（EVAL_PROTOCOL.md 第 8 节）。", "",
         "| 口径 | 4 终点平均 | " + " | ".join(EPN[e] for e in EP) + " |", "|---|---|" + "---|" * len(EP)]
    for k, lab in names:
        L.append(f"| {lab} | {_f(s['macro'][k])} | " + " | ".join(_f(s[e].get(k)) for e in EP) + " |")
    L.append(f"| AD：先死亡算阴性（多计入 {s['ad'].get('n_added_by_cr', 0)} 个站点） | | {_f(s['ad'].get('ad_cr'))} | | | |")
    x = s.get("paired_vs_delphi-dedup")
    if x:
        L += ["", "和参照模型 delphi-dedup 的配对差值（同一批人重抽样 1000 次）：**%+.3f** [%+.3f, %+.3f]。" % (x["diff"], x["ci"][0], x["ci"][1])]
    xi = s.get("paired_vs_delphi-dedup_ipcw")
    if xi:
        L += ["", "IPCW 口径下的配对差值：%+.3f [%+.3f, %+.3f]。" % (xi["diff"], xi["ci"][0], xi["ci"][1])]
    xe = s.get("paired_vs_delphi-dedup_ensemble")
    if xe:
        L += ["", "上面用的是单个 seed（s0），和 Delphi 的单 seed 可比；5 seed 集成的配对差值是 %+.3f [%+.3f, %+.3f]，"
              "集成的 4 终点平均是 %.3f。" % (xe["diff"], xe["ci"][0], xe["ci"][1], s["macro_ensemble"]["main"])]
    if vvt:
        L += ["", "val 对 test（这个模型没在 test 段上训练、也没用它挑 checkpoint）：" + "；".join(
            "%s val %.3f / test %.3f（差的 95%% CI [%+.3f, %+.3f]）" % (k, v["val"], v["test"], v["diff_ci"][0], v["diff_ci"][1])
            for k, v in vvt["seeds"].items())
              + "。val − test 平均 %+.3f。两个 N 人的独立样本：真有约 %.3f 的乐观偏差时才有 80%% 的把握检测出来，"
                "能排除的只是比差值 CI 上界（最大 %+.3f）更大的偏差。" % (
                  vvt["mean_val_minus_test"], vvt.get("mdd_80pct_power", vvt.get("min_detectable_diff", float("nan"))),
                  vvt.get("max_diff_ci_upper", float("nan")))]
    return L + [""]


# ----------------------------------------------------------------------------- 2 分期转换（fig3）

def sec_transitions(d, num="2"):
    t = _j(d, "transitions.json")
    if not t:
        return []
    L = [f"## {num}. 分期转换（fig3）", "", "![stage transitions](fig3_transitions.png)", "",
         "站在每次访视上，预测 2 / 5 / 10 年内会不会**第一次**越过 6 条线：MMSE <27、<24，全局认知 <0、<−1，AD，死亡"
         "（EVAL_PROTOCOL.md 第 9–11 节）。", ""]
    mac = t["macro"]
    if t.get("answers_all"):
        ci = mac["5y"].get("auc_ci")
        L.append(f"- **6 题平均 AUC**：2 年 {mac['2y']['auc']:.3f}，5 年 {mac['5y']['auc']:.3f}"
                 + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + f"，10 年 {mac['10y']['auc']:.3f}。")
        L.append("- 每道题三根柱子是 2 / 5 / 10 年（柱子从 0.5 = 瞎猜往右长），各自只算答案已知的起点，不能看成一条趋势线；"
                 "5 年那根带 95% 置信区间，"
                 "行名下是 5 年内越线的人数。")
        cr = mac["10y"].get("auc_cr")
        if cr is not None:
            L.append(f"- 10 年的数偏高：窗口内没越线就先去世的起点被排除了，把它们算成\"没越线\"，10 年平均从 {mac['10y']['auc']:.2f} "
                     f"掉到 {cr:.2f}。")
    else:
        got = [q for q in t["answered"] if t["answered"][q]]
        L.append(f"- 这个模型只训练过其中 {len(got)} 道题（1 / 2 / 3 / 5 年），没答的题图上写 not answered，6 题平均不报。")
    return L + [""]


# ----------------------------------------------------------------------------- 3 转移矩阵（fig4）、4 10 年轨迹（fig5）

TRAJ_NA_ZH = {"snapshot tokenization": "snapshot 分词每次访视都重发 AD_DX（和所有已患疾病），生成的轨迹里'首次 AD 诊断'没有定义，"
                                       "figure 2 的生成器拒绝这种分词",
              "visit_heads": "visit_heads 模型的 lm_head logit 不是速率（等待时间在 time_head 里），figure 2 的生成器不支持"}


def _na_reason(tr):
    why = (tr or {}).get("reason") or ""
    return next((v for k, v in TRAJ_NA_ZH.items() if k in why), why or "这个模型不能生成轨迹")


STAGE_ZH = ("正常", "临界", "受损")
STATE_ZH = ("正常", "临界", "受损", "死亡")


def matrix_gap(M):
    """5 年矩阵：模拟和实际的差。返回 (平均每格绝对差, 最明显的一格 (行, 列, 实际, 模拟))。
    平均差三行等权：每个起始分期算一道题。按人数加权会被人数最多的 Normal 行主导，几乎不让人换分期的模型反而显得最像。
    "最明显的一格"按差除以抽样误差挑（假如模拟的比例是对的，这一行 n 个人的实际比例会抖多少）：行人数少时，
    相当大的差也还在噪声里。
    fig4 的标题、各模型 README、LEADERBOARD 表 5 和 leaderboard.png 的面板 D 都用这一个定义。"""
    gaps, main, best_z = [], None, -1.0
    n_rows = M.get("n_approx") or [None] * 3
    for i, (o, p) in enumerate(zip(M["observed"], M["predicted"])):
        if o is None:
            continue
        n = n_rows[i] or 30
        for j, (x, y) in enumerate(zip(o, p)):
            gaps.append(abs(x - y))
            z = abs(x - y) / math.sqrt(max(y * (1 - y), 0.01) / n)     # 误差按模拟的比例算：假如模型是对的，实际比例会抖多少
            if z > best_z:
                best_z, main = z, (i, j, x, y)
    return (sum(gaps) / len(gaps) if gaps else float("nan")), main


def matrix_gap_floor(M, draws=4000, seed=0):
    """一个完美的模拟器（模拟 = 真实的转移概率）因为实际一侧的抽样误差也会有的平均每格差：每行按 n_approx 人
    多项抽样，和抽样前的比例比。是下限：活着那部分的分期只来自第 5 年还在随访的人，人数比 n_approx 少。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = [(np.array(o, float) / sum(o), n) for o, n in zip(M["observed"], M.get("n_approx") or []) if o is not None and n]
    if not rows:
        return None
    tot = 0.0
    for _ in range(draws):
        tot += np.mean([np.mean(np.abs(rng.multinomial(n, p) / n - p)) for p, n in rows])
    return float(tot / draws)


FAM_ZH = {"mmse": ("MMSE", STAGE_ZH, STATE_ZH),
          "cogn": ("全局认知", ("高", "中", "低"), ("高", "中", "低", "死亡"))}


def sec_matrix(d, num="3"):
    tr = _j(d, "trajectory.json")
    if tr is None:
        return []
    L = [f"## {num}. 转移矩阵（fig4）", ""]
    if not tr.get("applicable") or not tr.get("matrix_5y"):
        return L + [f"**不适用**：{_na_reason(tr)}。", ""]
    L += ["![transition matrix](fig4_transition_matrix.png)", "",
          "第一次访视时的分期 → 5 年后在哪个状态。上面一行是 MMSE（正常 ≥27 / 临界 24–26 / 受损 <24），下面一行是全局认知"
          "（高 ≥0 / 中 −1 到 0 / 低 <−1），死亡单独一列。左边是实际发生的（灰），右边是模型模拟的（蓝；同一批人，每人 100 条未来）。", ""]
    for fam in ("mmse", "cogn"):
        M = tr["matrix_5y"].get(fam)
        if not M:
            continue
        name, rows, cols = FAM_ZH[fam]
        stay = [f"{lab} {M['observed'][i][i]:.0%} 对 {M['predicted'][i][i]:.0%}"
                for i, lab in enumerate(rows) if M["observed"][i] is not None]
        if not stay:
            continue
        gap, main = matrix_gap(M)
        floor = matrix_gap_floor(M)
        i, j, o, p = main
        L.append(f"- **{name}**：5 年后还在原来的分期（实际 对 模拟）{'，'.join(stay)}；平均每格相差 {100 * gap:.0f} 个百分点"
                 + (f"（噪声下限约 {100 * floor:.0f}）" if floor else "") + f"；最明显的差别是 {rows[i]} → {cols[j]}，实际 {o:.0%}，模拟 {p:.0%}。")
    L.append("- 平均每格的差三行等权；最明显的差别按抽样误差衡量。实际一侧：5 年内死亡的比例用 Kaplan–Meier 估计（考虑失访），"
             "活着的那部分按第 5 年还在随访的人的分期分配。")
    return L + [""]


def sec_trajectories(d, num="4"):
    tr = _j(d, "trajectory.json")
    if tr is None:
        return []
    L = [f"## {num}. 10 年轨迹（fig5）", ""]
    if not tr.get("applicable") or not tr.get("incidence"):
        return L + [f"**不适用**：{_na_reason(tr)}。", ""]
    inc = tr["incidence"]
    L += ["![trajectories](fig5_trajectories.png)", "",
          f"验证集 {tr.get('n_persons')} 人，每人从第一次访视起模拟 {tr.get('n_mc')} 条 10 年的未来，和实际发生的比（EVAL_PROTOCOL.md 第 12 节）。", ""]
    for q, lab in (("ad", "10 年 AD 累计发生率"), ("death", "10 年死亡累计发生率")):
        c = inc[q]
        L.append(f"- **{lab}**：实际 {c['observed'][-1]:.0%}（95% CI {c['observed_lo'][-1]:.0%}–{c['observed_hi'][-1]:.0%}），"
                 f"模拟 {c['simulated'][-1]:.0%}。")
    occ = (tr.get("stage_over_time") or {}).get("mmse")
    if occ:
        def normal_share(r):
            return None if r is None or None in r[:3] or sum(r[:3]) <= 0 else r[0] / sum(r[:3])
        o, p = normal_share(occ["observed"][-1]), normal_share(occ["predicted"][-1])
        if o is not None and p is not None:
            L.append(f"- 第 10 年还活着的人里 MMSE 正常的比例：实际 {o:.0%}，模拟 {p:.0%}。")
    return L + [""]


# ----------------------------------------------------------------------------- 5 扰动（fig6）

# 扰动 v2 的第二个方向得分（结果出来之后加的，EVAL_PROTOCOL 13.4）：按 5 年 log OR 打分，不依赖校准。
# 按年数（ΔRMST）打分时，Delphi 直接读出的 AD 基线风险接近 0，
# 同样大的相对效应在年数上只挪动很少，会被 0.05 年的门槛判成"无响应"。
# 门槛 0.05 和第二部分扰动（表 2b）的"无响应"相同。
PV2_LOGOR_H, PV2_LOGOR_EPS = "5", 0.05
PV2_COG = ("mmse_lt27", "mmse_lt24", "cogn_lt0", "cogn_ltm1")


def pv2_logor_scores(D, H=PV2_LOGOR_H, eps=PV2_LOGOR_EPS):
    """直接读出的 effects -> ({right, no_response, wrong, n_scored}, (同步提前的扰动数, AD 有定义的扰动数))。
    log OR 为正 = 风险升高 = 提前。只数有预设方向、有定义的格子（和 ΔRMST 的得分同一批格子）。
    同步提前的分母 = AD 和至少一道认知题都有定义的扰动：cog2_severe 把 4 道认知题都直接越过了，原来只凭 AD 就算"同步"（2026-10-31 改）。"""
    t = {"right": 0, "no_response": 0, "wrong": 0}
    n_ad = n_sync = 0
    for a, e in (D.get("effects") or {}).items():
        for q, c in e.items():
            if c.get("prespec") and c.get("defined"):
                v = c["d_logor"][H]
                t["right" if v > eps else ("wrong" if v < -eps else "no_response")] += 1
        cog = [q for q in PV2_COG if (e.get(q) or {}).get("defined")]
        if (e.get("ad") or {}).get("defined") and cog:
            n_ad += 1
            n_sync += all(e[q]["d_logor"][H] > eps for q in ["ad"] + cog)
    t["n_scored"] = t["right"] + t["no_response"] + t["wrong"]
    return t, (n_sync, n_ad)


def _pv2_status(st):
    """perturb_v2.json 里的状态是英文；README 用中文，按开头对上，对不上就原样给。"""
    st = st or "没有"
    if st.startswith("undefined: visit_heads"):
        return "没有定义（visit_heads 模型的 lm_head logit 不是速率，风险和时间都没有定义）"
    if st.startswith("undefined: non-generative"):
        return "不适用（不是生成模型）"
    if st == "not run":
        return "没有跑"
    return st


def sec_perturbation(d, num="5", note_missing=None):
    J = _j(d, "perturb_v2.json")
    L = [f"## {num}. 扰动（fig6）", ""]
    if J is None:
        return (L + [note_missing, ""]) if note_missing else []
    D, R = J.get("direct") or {}, J.get("rollout") or {}
    if D.get("status") != "ok":
        return L + [f"**不适用**：直接读出{_pv2_status(D.get('status'))}。", ""]
    tl, (ns, na) = pv2_logor_scores(D)
    shown = {a: {q: c for q, c in e.items() if q in ("ad", "mmse_lt24", "death")}
             for a, e in D["effects"].items() if a in ("mmse_b", "cogn_m", "cog2", "age5", "stroke", "comorb3", "apoe")}
    ts, _ = pv2_logor_scores({"effects": shown})
    gen = R.get("status") == "ok"
    c2ad = (D["effects"].get("cog2") or {}).get("ad") or {}
    responsive = bool(c2ad.get("defined")) and abs(c2ad["d_logor"][PV2_LOGOR_H]) >= PV2_LOGOR_EPS
    if gen and responsive:
        cap = ("；图 B 是 MMSE 和认知一起下降之后，**模拟的未来**里 10 年的 AD、MMSE<24 累计风险。两张图读法不同：A 读的是改完那一刻"
               "查询位置的速率换算成的风险（Delphi 的直接读法），B 是往后生成的未来，后面的访视会重新生成。Delphi 直接读出的 AD 风险接近 0，"
               "所以 A 里的优势比很大，B 里的风险挪得少。")
    elif gen:
        cap = ("；图 B 是 MMSE 和认知一起下降之后，模拟的未来里 10 年的 AD、MMSE<24 累计风险。这个模型只看人口学和年龄，"
               "认知改动在图 A、图 B 里都没有反应（符合预期）。")
    else:
        cap = None
    what = ("图 A 是直接读出的 5 年风险优势比" + cap) if cap else "图是直接读出的 5 年风险优势比（这个模型不能模拟未来，没有时间进程的图）。"
    panel = "图 A 上" if cap else "图上"
    L += ["![perturbation](fig6_perturbation.png)", "",
          "在第 4 次访视上改记录（单个改动或组合），看未来怎么变（扰动队列 N 人；改哪些按 SHAP 的结果挑，见 "
          f"[SHAP_SUMMARY.png]({os.path.relpath(os.path.join(ZOO, 'reports', 'SHAP_SUMMARY.png'), d)})；EVAL_PROTOCOL.md 第 13 节）。" + what, "",
          f"- **方向**：{panel}画出的 7 个改动 × 3 条线里，事先写好预期方向的 {ts['n_scored']} 格对 {ts['right']}；"
          f"全部 18 个改动 × 6 条线里有预设方向的 {tl['n_scored']} 格，对 {tl['right']}、没反应 {tl['no_response']}、错 {tl['wrong']}"
          f"（按 5 年优势比）；按 5 年优势比，AD 和每条有定义的认知线风险一起升高的改动 {ns} / {na} 个。"]
    c2 = D["effects"].get("cog2") or {}
    if (c2.get("ad") or {}).get("defined") and (c2.get("mmse_lt24") or {}).get("defined"):
        L.append(f"- **MMSE 和认知一起下降**：直接读出，5 年内 AD 的优势比 ×{math.exp(c2['ad']['d_logor']['5']):.1f}，"
                 f"MMSE<24 ×{math.exp(c2['mmse_lt24']['d_logor']['5']):.1f}。")
        src = R.get("effects", {}).get("cog2") if gen else None
        if src and (src.get("ad") or {}).get("cif_orig") is not None:
            how = "模拟的未来里" if gen else "按模型给的风险"
            ad, m24 = src["ad"], src["mmse_lt24"]
            L.append(f"  {how}（图 B），10 年 AD 累计风险 {ad['cif_orig'][-1]:.0%} → {ad['cif_pert'][-1]:.0%}，"
                     f"MMSE<24 {m24['cif_orig'][-1]:.0%} → {m24['cif_pert'][-1]:.0%}；"
                     f"10 年内平均无 AD 时间少 {-ad['d_rmst']:.1f} 年，无 MMSE<24 时间少 {-m24['d_rmst']:.1f} 年（按全部人平均，多出来的病例占大头，不是同一批病例提前了这么多年）。")
    extra = "按年数的打分、rollout 的打分" if R.get("status") == "ok" else "按年数的打分"
    L.append(f"- 逐格的数字（包括{extra}）在 `perturb_v2.json` 和 LEADERBOARD 表 6。")
    return L + [""]


# ----------------------------------------------------------------------------- 拼起来 + 页首目录

SHAP_ZH = {"MMSE": "MMSE", "COGN": "认知", "DBP": "舒张压"}
BIN_ZH = {"impaired": "受损", "borderline": "临界", "normal": "正常", "low": "低", "mid": "中", "high": "高",
          "elevated": "偏高"}


def sec_shap(d, num="6"):
    """fig7：这个模型判断 AD 时看什么（SHAP）。没算 SHAP 的包写原因。"""
    J = _j(d, "shap.json")
    L = [f"## {num}. 模型看什么（fig7，SHAP）", ""]
    if J is None:
        why = ("visit_heads 的 logit 不是速率，SHAP 要拆的那个读数没有定义" if "visitheads" in os.path.basename(d.rstrip("/"))
               else "没有算 SHAP")
        return L + [f"**不适用**：{why}。", ""]
    Q = J["questions"]["ad"]
    lm = J.get("family") == "landmark"
    L += ["![SHAP](fig7_shap.png)", "",
          ("把模型给的 AD 风险拆成每个变量的贡献（SHAP）：拿掉一个变量 = 到第 4 次访视为止都当没测过；按这个人最近一次测到的值分档。"
           if lm else "照上游 Delphi 的 SHAP 方法，把模型给的 AD 速率拆成每条记录的贡献（拿掉一条记录，看速率变多少）。")
          + f"对象是第 4 次访视时还没得 AD 的 {Q['n_at_risk']} 人；×k = 有这条{'变量' if lm else '记录'}时"
          + ("5 年累积风险率（风险小时约等于风险）" if lm else "速率") + "是没有时的 k 倍"
          "（EVAL_PROTOCOL.md 13.1）。", ""]
    from figures import _tok_label                 # 和图上同一套标签（figures 已经 import 了本模块，这里晚一点 import 不会绕圈）
    top = sorted([t for t in Q["tokens"] if t["n_persons"] >= 20], key=lambda t: -t["mean_person_phi"])[:3]
    if top:
        L.append("- 推得最高的三种：" + "，".join(f"{_tok_label(t['token'])} ×{math.exp(t['mean_person_phi']):.1f}（{t['n_persons']} 人）"
                                              for t in top) + "。")
    fam = {f["family"]: {b["token"]: b["mean_person_phi"] for b in f["bins"]} for f in Q.get("families", [])}
    parts = []
    for g, (worst, best) in (("MMSE", ("impaired", "normal")), ("COGN", ("low", "high")), ("DBP", ("high", "normal"))):
        w, b_ = fam.get(g, {}).get(f"{g}_{worst}"), fam.get(g, {}).get(f"{g}_{best}")
        if w is not None and b_ is not None:
            parts.append(f"{SHAP_ZH[g]} {BIN_ZH[worst]} ×{math.exp(w):.1f}、{BIN_ZH[best]} ×{math.exp(b_):.1f}")
    if parts:
        L.append("- 看值还是看在不在（图 B）：" + "；".join(parts) + "。值越差推得越高 = 模型在看数值；\"正常\"也推得高 = 只在意测没测过。")
    a5 = Q.get("age5", {}).get("mean")
    if a5 is not None:
        L.append(f"- 参照：整段病史老 5 岁，AD {'5 年累积风险率' if lm else '速率'} ×{math.exp(a5):.2f}。")
    L.append("- 这不是打分：SHAP 只说明模型靠什么判断，没有对错，不参与排名。数字在 `shap.json`。")
    return L + [""]


TIMING_ZH = {"ad": "AD", "mmse_lt24": "MMSE<24", "death": "死亡"}


def sec_timing(d, num="7."):
    """fig8：真的发生了的人，模型说的"什么时候"差几年（timing.py）。"""
    J = _j(d, "timing.json")
    L = [f"## {num} 时间预测得准不准（fig8）", ""]
    if J is None:
        return L + ["**不适用**：没有算（timing.json 不在）。", ""]
    if not J.get("applicable"):
        return L + [f"**不适用**：{J['reason']}。", ""]
    Y = J["years"]
    src = "模拟的未来里 k 年内发生的比例" if J["source"] == "rollout" else "模型在第一次访视时给的 k 年风险"
    pred = (f"逐年累计风险 F(1)…F({Y})（{src}）第一次达到 F({Y}) 一半的那个整年"
            f"（不插值；即 {Y} 年内会发生的前提下，按整年算的发生时间中位数）")
    L += ["![timing](fig8_timing.png)", "",
          f"从第一次访视起，只看 {Y} 年内真的发生了的人：预测时间 = {pred}；误差 = 预测 − 实际（年）。"
          "尺子 = 所有人都猜同一个数（实际时间的中位数）。平均误差比尺子小，才说明模型分得出谁早谁晚。", "",
          "| 题 | 人数 | 模型平均误差（年）[95% CI] | 人人猜同一个数的平均误差（年） | 模型 − 猜同一个数 [95% CI] | 预测和实际的秩相关 |",
          "|---|---|---|---|---|---|"]
    for q in ("ad", "mmse_lt24", "death"):
        r = J["questions"].get(q) or {}
        if "mae" not in r:
            continue
        ci, dc = r["mae_ci"], r["mae_minus_constant_ci"]
        L.append(f"| {TIMING_ZH[q]} | {r['n']} | {r['mae']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {r['constant_mae']:.2f} | "
                 f"{r['mae'] - r['constant_mae']:+.2f} [{dc[0]:+.2f}, {dc[1]:+.2f}] | "
                 f"{'—（所有人预测一样）' if r['spearman'] is None else format(r['spearman'], '.2f')} |")
    return L + ["", "另外三道越线题（MMSE<27、认知<0、认知<−1）的数字在 `timing.json`。", ""]


def main_sections(d, perturb_note=None):
    """2–7 节，固定顺序。"""
    return (sec_transitions(d) + sec_matrix(d) + sec_trajectories(d) + sec_perturbation(d, note_missing=perturb_note)
            + sec_shap(d) + sec_timing(d))


def toc_line(d):
    return ("本页：1 预测（fig1、fig2）→ 2 分期转换（fig3）→ 3 转移矩阵（fig4）→ 4 10 年轨迹（fig5）→ 5 扰动（fig6）"
            "→ 6 模型看什么（fig7，SHAP）→ 7 时间预测（fig8）→ 附：敏感性分析 → 实现。不适用的节只写原因。")


# ----------------------------------------------------------------------------- LEADERBOARD 表 5 用的汇总

def traj_summary(t):
    """trajectory.json -> 表 5 的几个数（字段缺了给 None）：5 年矩阵平均相差、10 年 AD / 死亡 实际 vs 模拟。"""
    out = {"matrix_gap": None, "ad_obs": None, "ad_sim": None, "death_obs": None, "death_sim": None}
    M = (t.get("matrix_5y") or {}).get("mmse")
    if M:
        out["matrix_gap"] = matrix_gap(M)[0]
    inc = t.get("incidence") or {}
    for q in ("ad", "death"):
        if q in inc:
            out[f"{q}_obs"], out[f"{q}_sim"] = inc[q]["observed"][-1], inc[q]["simulated"][-1]
    return out
