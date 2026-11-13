"""make_package.py -- 在本地把 RIS 拉回来的聚合产物装成打包盒，并生成 Delphi 打包盒的 README。

    python make_package.py --fetched <从 RIS 拉回的 packages 目录>
    python make_package.py --fetched ..        # 不拷文件，只按仓库里现有的 JSON 重新生成 README

每个 Delphi 打包盒 model_zoo/<name>/：
  README.md               一句话 + 1 预测 + 2 分期转换 + 3 转移矩阵 + 4 10 年轨迹 + 5 扰动 + 敏感性 + 实现细节 + 注意事项
  metrics.json            第一部分的统一指标（evaluate.py 产出）
  fig1_auc_2y.png、fig2_calibration_2y.png        第一部分的两张图（RIS 上 evaluate.py 画，本地可 replot.py 重画）
  fig3_transitions.png ... fig8_timing.png  第三部分的图（本地 figures.py 按下面的 JSON 画）
  transitions.json / trajectory.json / perturb_v2.json / shap.json / timing.json   第三部分的聚合数字
  fidelity.json / perturb.json / rollout_info.json / sensitivity.json / pos_leak.json   第二部分和敏感性的聚合数字（只进表，不出图）
  ckpt_info.json          ckpt 里的配置、参数量、best val、数据规模
  prior_results.md        统一评估之前算过的指标（collect_prior.py 生成；口径各异，不可直接比）
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ZOO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from common import ENDPOINTS, HORIZONS, MAIN_H  # noqa: E402
from registry import CAL_NOTES, DELPHI, LOCAL_DELPHI, REFERENCES, REPORT_DIR, pkg_dir, rel_link  # noqa: E402
from report_sections import main_sections, sensitivity_section, toc_line  # noqa: E402

# 从 RIS 拉回来要拷的文件：第一部分的两张图 + 所有聚合 JSON。第三部分的图（fig3–fig6）由本地 figures.py 按 JSON 画。
FILES = ("metrics.json", "fig1_auc_2y.png", "fig2_calibration_2y.png", "ckpt_info.json",
         "pos_leak.json", "fidelity.json", "perturb.json", "rollout_info.json", "sensitivity.json",
         "transitions.json", "trajectory.json", "shap.json", "perturb_v2.json", "timing.json")

TOK_DESC = {
    "dedup": ("`python build.py --out data_rosmap --dataset rosmap`",
              "两级去重：连续量只在分箱变化时发射；每个 token 每人最多出现一次（Delphi 的 first-occurrence 语义）。"
              "认知量表的往复（正常→受损→正常）在序列里结构上不存在。"),
    "nodedup": ("`python build.py --nodedup MMSE,COGN ...`",
                "MMSE 和 cogn_global 每次访视都发射一个分箱 token（两级去重都豁免），其余同 dedup。"),
    "fullvisit": ("`python build.py --nodedup all --no-global-dedup ...`",
                  "15 个连续量全部每次访视都发射，并关掉全局去重（用药 ON/OFF 可以重复）。背景块只在基线发一次。"),
    "snapshot": ("`python build.py --snapshot ...`",
                 "每次访视发射完整状态快照：背景块、用药当前状态（ON/OFF）、已患疾病（ONSET / AD_DX 从首次起每次都发）、全部测量。"),
}
SW_DESC = {
    "pos": "`pos_embedding=True`：可学习绝对位置嵌入（block_size 行）。",
    "aux": "`aux_head=True, aux_lambda=0.3`：多任务判别头（每个位置预测若干终点 h 年内是否发生，BCE），与语言模型头联合训练。",
    "visitheads": "`visit_heads=True`：访视级时间头 + 访视大小头；等待时间不再来自 lm_head 的 logsumexp。",
    "visit1only": "`attn_visits=1`：每个位置只能注意背景块 + 当前这次访视（model.py 改动 C），等价于'只看现在'。",
    "staticonly": "`static_only=True`：非背景 token 在输入里抹成 no-event、只能注意背景块和自己，等价于'只看人口学 + 当前年龄'。",
    "nojitter": "训练配置 `lifestyle_token_max=2`（原配方是 32）：随机挪动年龄的 token 范围是 3..2，空的，所以背景块（研究队列、教育、"
                "种族、西语裔、APOE、吸烟、饮酒、基线已患病）都放在真实年龄上，不再加 U[−20, +40) 年的随机偏移；其余超参和 delphi-dedup 相同"
                "（max_iters 12000）。读分时用 ckpt 里存的这份配置，和训练一致。",
}
CONFIG_FILE = {
    "delphi-dedup": "train_delphi_rosmap.py（out_dir 覆盖为 Delphi-ROSMAP-gpubase）",
    "delphi-dedup-mps-legacy": "train_delphi_rosmap.py（--device=mps）",
    "delphi-dedup-nojitter": "train_delphi_rosmap.py（--lifestyle_token_max=2，--seed=42..46，out_dir = Delphi-ROSMAP-nojitter[-sNN]）",
    "delphi-nodedup": "train_delphi_rosmap_nodedup.py",
    "delphi-nodedup-pos": "train_delphi_rosmap_nodedup_pos.py",
    "delphi-nodedup-aux": "train_delphi_rosmap_nodedup_aux.py",
    "delphi-nodedup-visitheads": "train_delphi_rosmap_nodedup_vh.py",
    "delphi-nodedup-split811": "train_delphi_rosmap_nodedup_3w.py",
    "delphi-nodedup-pos-split811": "train_delphi_rosmap_nodedup_pos_3w.py",
    "delphi-nodedup-staticonly-split811": "train_delphi_rosmap_fig3_static.py",
    "delphi-fullvisit": "train_delphi_rosmap_fullvisit.py",
    "delphi-fullvisit-split811": "train_delphi_rosmap_fig3_fullvisit3w.py",
    "delphi-fullvisit-visit1only-split811": "train_delphi_rosmap_fig3_visit1.py",
    "delphi-snapshot": "train_delphi_rosmap_snapshot.py",
    "delphi-snapshot-pos": "train_delphi_rosmap_snapshot_pos.py",
    "delphi-snapshot-visit1only": "train_delphi_rosmap_snapshot_visit1.py",
    "delphi-snapshot-staticonly": "train_delphi_rosmap_snapshot_static.py",
}
CAVEATS = {
    "snapshot": "**统一评估的读分对 snapshot 分词没有定义好**：snapshot 每次访视都先重发性别等背景 token，"
                "背景 token 不计损失，所以'两次访视之间'这个位置（统一评估在站点后 1 天读速率的地方）在训练里"
                "从来不受监督。这里的 AUC 偏低主要反映'Delphi 的速率读分在这种分词下不可用'，不等于表征差。",
    "pos": "带位置编码：训练时每个 batch 右对齐到这一批最长的那条（cut_batch=True），位置下标和'后面还剩几个 token'挂钩。"
           "统一评估只喂到站点为止的记录，并按训练方式右对齐到 max(自身长度, 训练 batch 长度中位数)，"
           "查询 token 落在训练里最常见的'记录末尾'那一格。以前在完整记录上读分的影响见上面'这个模型独有的检查'一节。",
    "visitheads": "visit_heads 打开后 lm_head 的 logit 不再是速率：AUC 用 logit 排序照常有效，"
                  "但'风险 = 1−exp(−速率×H)'没有意义（饱和成 1），校准图不可解释，第二部分扰动（表 2b）和扰动 v2（第 5 节）的读法也没有定义（不打分）。",
    "split811": "split811：训练集少 N 人（N vs N），val 是同一批人，所以和两分模型的 val 数字可直接比。",
    "legacy": "Mac/MPS 上训练的最早版本；同配置同 seed 在 H100 上重训的是 delphi-dedup，以那个为准。",
    "nojitter": "和 delphi-dedup 只差一处：训练时**不**随机挪动背景 token（APOE、教育、种族、吸烟等；性别本来就不在内）的年龄。"
                "原配方的这个增广让背景 token 在很多 AD 预测位置上看不见，看历史的模型就不用 APOE；关掉之后 APOE 的效应回来了"
                "（[APOE_DIAGNOSIS.md](../../reports/APOE_DIAGNOSIS.md)）。5 个 seed（42–46），指标按 seed 42 报、seed 均值 ± sd 另列。",
}


def fmt_ci(r):
    ci = r.get("auc_ci")
    return f"{r['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{r['auc']:.3f}"


def results_section(m, refs):
    ep = m["endpoints"]
    L = []
    L.append(f"验证集 {m['n_persons']} 人、{m['n_landmarks']} 个站点；主 seed = {m['main_seed']}"
             + (f"，另有 {len(m['seeds']) - 1} 个 seed" if len(m["seeds"]) > 1 else "") + "。\n")
    head = "| 终点（2 年） | AUC [95% CI] | 站点数（阳性） | O/E | ECE |"
    if "across_seeds" in m:
        head += " AUC 各 seed 均值 ± sd |"
    L += [head, "|" + "---|" * (head.count("|") - 1)]
    for e in ENDPOINTS:
        r = ep[f"{e}_{MAIN_H}y"]
        row = f"| {ENDPOINTS[e][0]} | {fmt_ci(r)} | {r['n']} ({r['n_pos']}) | {r['o_e']:.2f} | {r['ece']:.3f} |"
        if "across_seeds" in m:
            a = m["across_seeds"][f"{e}_{MAIN_H}y_auc"]
            row += f" {a['mean']:.3f} ± {a['sd']:.3f} |"
        L.append(row)
    mc = m.get("macro_auc_ci_2y")
    row = f"| **4 个终点平均** | **{m['macro_auc']['2y']:.3f}**" + (f" [{mc[0]:.3f}, {mc[1]:.3f}]" if mc else "") + " | | | |"
    if "across_seeds" in m:
        a = m["across_seeds"][f"macro_{MAIN_H}y_auc"]
        row += f" {a['mean']:.3f} ± {a['sd']:.3f} |"
    L.append(row)
    L.append("")
    main_lab = "5 个 seed 的集成" if m["main_seed"] == "ens" else f"主 seed {m['main_seed']}"
    L.append(f"其它时间窗的 AUC（{main_lab}）：")
    L.append("")
    L.append("| 终点 | " + " | ".join(f"{H} 年" for H in HORIZONS) + " |")
    L.append("|---|" + "---|" * len(HORIZONS))
    for e in ENDPOINTS:
        L.append(f"| {ENDPOINTS[e][0]} | " + " | ".join(f"{ep[f'{e}_{H}y']['auc']:.3f}" for H in HORIZONS) + " |")
    L.append("| 平均 | " + " | ".join(f"{m['macro_auc'][f'{H}y']:.3f}" for H in HORIZONS) + " |")
    if refs:
        L.append("")
        L.append("参照（fig1 里的两条线，seed 42）：" + "；".join(
            f"{lab} = {rm['macro_auc']['2y']:.3f}（{rn}）" for lab, rn, rm in refs))
    return "\n".join(L)


def impl_section(name, tok, sw, info):
    s42 = info["seeds"].get("42") or next(iter(info["seeds"].values()))
    ma, cfg = s42["model_args"], s42["config"]
    cmd, desc = TOK_DESC[tok]
    L = ["### 数据 / 分词", "", f"- 数据集 `{info['dataset']}`（RIS: `delphi/data/{info['dataset']}/`），生成命令 {cmd}"
         + ("，加 `--split 0.8,0.1,0.1`" if name.endswith("split811") else ""),
         f"- {desc}"]
    d = info.get("data", {})
    if d:
        L.append("- 规模：" + "；".join(f"{k} {v['persons']} 人 / {v['rows']:,} 行（每人中位 {v['tokens_per_person_median']:.0f} 个 token，最长 {v['max']}）"
                                    for k, v in d.items()))
    meta = info.get("meta") or {}
    if meta.get("n_repeatable_tokens") is not None:
        L.append(f"- 实测可重复出现的 token：{meta['n_repeatable_tokens']} 个")
    L += ["", "### 模型结构", "",
          f"- Delphi-2M 式 GPT（上游 gerstung-lab/delphi，本仓库 `delphi/model.py`）：{ma['n_layer']} 层、{ma['n_head']} 头、"
          f"宽度 {ma['n_embd']}、block_size {ma['block_size']}、词表 {ma['vocab_size']}、参数量 {s42['n_params'] / 1e6:.2f}M",
          "- 输入 = token 嵌入 + 年龄的正弦编码（以天为单位）" + ("+ 可学习位置嵌入" if ma.get("pos_embedding") else "（没有位置嵌入）"),
          f"- 损失：下一个 token 的交叉熵 + 指数等待时间的对数似然（`mask_ties={ma.get('mask_ties')}`：同龄 token 互相看不见），"
          f"`t_min={ma.get('t_min')}`；背景块 / 性别（post-shift 2–33）不作为预测目标"]
    for k in sw:
        L.append(f"- {SW_DESC[k]}")
    L += ["", "### 训练", "",
          f"- 配置文件 `delphi/config/{CONFIG_FILE[name]}`",
          f"- batch {cfg.get('batch_size')}、lr {cfg.get('learning_rate')}（warmup {cfg.get('warmup_iters')}，余弦衰减到 {cfg.get('min_lr')}）、"
          f"{cfg.get('max_iters')} iter、dropout {cfg.get('dropout')}、weight decay {cfg.get('weight_decay')}、"
          f"no-event 插值率 每 {cfg.get('no_event_token_rate')} 年、"
          + ("背景块年龄随机偏移（lifestyle augmentation）"
             if (cfg.get("lifestyle_token_max") or 0) >= (cfg.get("lifestyle_token_min") or 0) else
             f"背景块年龄**不**随机偏移（lifestyle_token_max={cfg.get('lifestyle_token_max')}，范围为空）"),
          "- 按 val loss 挑 checkpoint（`always_save_checkpoint=False`）",
          "", "| seed | best val loss | 在第几 iter | ckpt（RIS，相对 项目目录/） |", "|---|---|---|---|"]
    for s, v in sorted(info["seeds"].items(), key=lambda kv: int(kv[0])):
        L.append(f"| {s} | {v['best_val_loss']:.4f} | {v['iter_num']} | `{v['ckpt']}` |")
    L += ["", "注意：不同分词的 val loss 不可横向比（token 组成完全不同）。", "",
          "### 复现", "", "```bash",
          *(["# 训练（RIS GPU）：每个 seed 一次（42–46）；train.sbatch 把多余的参数原样传给 train.py",
             "cd delphi && for s in 42 43 44 45 46; do d=Delphi-ROSMAP-nojitter; [ $s = 42 ] || d=$d-s$s",
             "  CFG=config/train_delphi_rosmap.py sbatch ris/train.sbatch --lifestyle_token_max=2 --seed=$s --out_dir=$d; done"]
            if "nojitter" in sw else
            ["# 训练（RIS GPU）：seed 43/44 加 --seed=43 --out_dir=<目录>-s43",
             f"cd delphi && CFG=config/{CONFIG_FILE[name].split('（')[0]} sbatch ris/train.sbatch"]),
          "# 统一评估（RIS CPU）",
          f"cd model_zoo/unified_eval && sbatch ris/run_delphi_all.sbatch --only {name}",
          "```"]
    return "\n".join(L)


def caveats(name, tok, sw):
    out = []
    if name.endswith("legacy"):
        out.append(CAVEATS["legacy"])
    if tok == "snapshot":
        out.append(CAVEATS["snapshot"])
    if "pos" in sw:
        out.append(CAVEATS["pos"])
    if "visitheads" in sw:
        out.append(CAVEATS["visitheads"])
    if "nojitter" in sw:
        out.append(CAVEATS["nojitter"])
    if name.endswith("split811"):
        out.append(CAVEATS["split811"])
    return out


def copy_files(src, dst, files):
    """把 src 里有的文件拷到 dst；src 就是 dst（--fetched 指向仓库本身）时什么都不做。"""
    os.makedirs(dst, exist_ok=True)
    for f in files:
        s, t = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(s) and not (os.path.exists(t) and os.path.samefile(s, t)):
            shutil.copy(s, t)


# 跨模型的汇总（RIS 上写在 _runs/packages/ 下，仓库里放在 reports/）。GAP_AUC.json 是另一项检查，不放进仓库。
ROOT_FILES = ("SENSITIVITY.json", "VAL_VS_TEST.json", "MC_GRANULARITY.json",
              "SHAP_SUMMARY.json", "PERTURB_V2_SUMMARY.json")          # SHAP_SUMMARY.png 在本地由 figures.py 画，不从 RIS 拷


def build(fetched, archive=False):
    # --fetched ..：读仓库里现有的打包盒，它们按 registry.pkg_dir 分在子目录里；RIS 拉回来的 packages/ 是平铺的
    local = os.path.samefile(fetched, ZOO)
    src_of = lambda n: os.path.join(ZOO, pkg_dir(n)) if local else os.path.join(fetched, n)    # noqa: E731
    copy_files(fetched, os.path.join(ZOO, REPORT_DIR), ROOT_FILES)
    vp = os.path.join(os.path.join(ZOO, REPORT_DIR) if local else fetched, "VAL_VS_TEST.json")
    vvt = json.load(open(vp))["models"] if os.path.exists(vp) else {}
    ref_metrics = {}
    for lab, rn in REFERENCES:
        p = os.path.join(src_of(rn), "metrics.json")
        if os.path.exists(p):
            ref_metrics[rn] = (lab, rn, json.load(open(p)))
    for name, dirs, dataset, tok, sw, note in DELPHI:
        if name not in LOCAL_DELPHI and not (archive and local):
            print("-- 已归档（models_self/archive/，本地不再打包）：", name)
            continue
        src = src_of(name)
        if not os.path.exists(os.path.join(src, "metrics.json")):
            print("!! 没有结果：", name)
            continue
        dst = os.path.join(ZOO, pkg_dir(name))
        copy_files(src, dst, FILES)
        m = json.load(open(os.path.join(src, "metrics.json")))
        info = json.load(open(os.path.join(src, "ckpt_info.json")))
        refs = [ref_metrics[rn] for _, rn in REFERENCES if rn in ref_metrics and rn != name]
        L = [f"# {name}", "", note + "。", "",
             f"**统一评估：4 个终点 2 年 AUC 平均 = {m['macro_auc']['2y']:.3f}**（题目见 [EVAL_PROTOCOL.md]({rel_link(name, 'EVAL_PROTOCOL.md')})，"
             f"横向对比见 [LEADERBOARD.md]({rel_link(name, 'LEADERBOARD.md')})）", "",
             toc_line(dst), "",
             "## 1. 预测（fig1、fig2）", "", "![AUC](fig1_auc_2y.png)", "", "![calibration](fig2_calibration_2y.png)", "",
             results_section(m, refs), ""]
        # 顺序：1 预测 -> 2 分期转换 -> 3 转移矩阵 -> 4 10 年轨迹 -> 5 扰动 -> 附：敏感性 -> 实现
        L += main_sections(dst)
        L += sensitivity_section(dst, vvt.get(name))
        L += ["## 实现", "", impl_section(name, tok, sw, info), ""]
        pl = os.path.join(src, "pos_leak.json")
        if os.path.exists(pl) and "pos" in sw:
            d = json.load(open(pl))
            ba = d.get("by_align", {})
            Ls = list(ba)
            L += ["## 这个模型独有的检查：位置编码会不会漏未来", "",
                  "同一个 ckpt、同一批站点、同一个查询 token，只换输入排布：**只给到站点为止的记录**（统一评估）"
                  " vs **完整记录**（以前的读法：因果注意力看不到后面的 token，但右对齐下位置下标和'后面还剩几个 token'挂钩）。"
                  "两种排布都按训练时的方式对齐（训练 batch 长度的第 5/50/95 百分位 = " + "/".join(Ls) + "）。"
                  f"中位数对齐下 logit 最多相差 {d['max_abs_logit_diff']:.2f}（不带位置编码的 delphi-nodedup 对照为 0）。", "",
                  "| 终点（2 年） | 只到站点（统一评估） | 完整记录 | 差（中位数对齐） | 差在 3 种对齐下的范围 | '站点后还剩几个 token' 单独的 AUC |",
                  "|---|---|---|---|---|---|"]
            big, rest = [], []
            for e in ENDPOINTS:
                r = d["endpoints"][e]
                ds = [ba[x]["endpoints"][e]["leaky"] - ba[x]["endpoints"][e]["honest"] for x in Ls]
                L.append(f"| {ENDPOINTS[e][0]} | {r['honest']:.3f} | {r['leaky']:.3f} | {r['leaky'] - r['honest']:+.3f} | "
                         f"[{min(ds):+.3f}, {max(ds):+.3f}] | {r['future_length_alone']:.3f} |")
                (big if max(abs(x) for x in ds) > 0.05 else rest).append((ENDPOINTS[e][0], ds))
            txt = "结论：位置确实让输出依赖未来的长度（logit 会变）。"
            if big:
                txt += "主要影响在 " + "、".join(f"{n}（{min(ds):+.3f} ~ {max(ds):+.3f}）" for n, ds in big) + "，"
                txt += "旧读法（完整记录）下这些终点的 AUC 被位置里的未来信息抬高了，不可靠。"
            if rest:
                worst = max(abs(x) for _, ds in rest for x in ds)
                txt += "其余终点" + ("的差在 ±%.2f 以内。" % (worst + 0.005))
            L += ["", txt, ""]
        cv = caveats(name, tok, sw)
        if name in CAL_NOTES and not any("visit_heads" in c for c in cv):
            cv.append(CAL_NOTES[name])
        if cv:
            L += ["## 注意", ""] + [f"- {c}" for c in cv] + [""]
        if os.path.exists(os.path.join(dst, "prior_results.md")):        # 统一评估之后才加的包没有旧指标
            L += ["## 以前的指标", "", "见 [prior_results.md](prior_results.md)（口径各不相同，不能和上面的数字直接比）。", ""]
        with open(os.path.join(dst, "README.md"), "w") as f:
            f.write("\n".join(L))
        print("packaged", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetched", required=True)
    ap.add_argument("--archive", action="store_true",
                    help="--fetched .. 时也按仓库里的 JSON 重写 models_self/archive/ 下的 README（模板或 JSON 改了之后用）")
    a = ap.parse_args()
    build(a.fetched, a.archive)
