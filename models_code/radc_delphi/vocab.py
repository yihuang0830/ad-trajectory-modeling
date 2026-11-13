"""
vocab.py -- the ROSMAP token table, resolved into the id families Figure 2 asks for.

This replaces delphi-fine-tuning's `radc_delphi/vocab.py`, which hard-codes a 50/56-token RADC
vocabulary. Ours is the 129-token table built by `../tokenization/build.py`, so the NAMES are
read from `labels.csv` rather than typed out here, and every family is derived from that table
by NAME. Nothing downstream of this file may hard-code a token id -- that rule is inherited
from upstream and is the reason a checkpoint cannot be scored against a table it never saw.

TOKEN SPACE. Everything this package exports is MODEL space, which is also the labels.csv row
index. The .bin on disk stores model_id - 1; that +1 shift belongs to batching.py and to
nothing else (upstream delphi/utils.py:get_batch does the same at its last line).

------------------------------------------------------------------------------------------
THE FAMILIES, AND WHY EACH ONE EXISTS

  SCALES            ordinal bins. Order inside each family is labels.csv order, which is
                    low -> high on the underlying measurement. For MMSE and COG that means
                    WORST FIRST, and perdomain.py depends on it (adverse = lower index); the
                    two are asserted in check(). NOTE these are NOT exempt from the rollout's
                    no-repeat rule -- see REPEATABLE_TOKENS below for why that changed.
  KEEP_FIRST_IDS    incident events that happen once (STROKE, DEPRESSION, the five *_ONSET).
  MED_IDS           medication starts and stops. Repeat by construction -- they are a pair.
  ONSET_IDS         EMPTY here, and deliberately so. Upstream has two GRADED onset families
                    (`Stroke, probable` / `Stroke, possible`) that revert and re-fire; our
                    tokenizer emits a single once-only onset token per condition instead, so
                    those tokens are in KEEP_FIRST_IDS and this family has no members. The name
                    is kept because figure2/ imports it.
  STATIC_IDS        sex + the background block (ids 2..33). These are the model's
                    `ignore_tokens`: still attended to, never predicted.
  PATHOLOGY_IDS     the 34 post-mortem tokens. Upstream has NO analogue -- see below.

------------------------------------------------------------------------------------------
PATHOLOGY IS BLOCKED FROM ROLLOUTS, AND THAT IS A CHOICE

`../tokenization/README.md` records that the autopsy tokens are emitted at the SAME age as
Death and after it, because placing them earlier would be leakage. Death terminates a rollout,
so a sampled pathology token can only ever appear BEFORE death -- i.e. in a position the
training data never contains. Leaving them sampleable would let a living 80-year-old be
assigned a Braak stage.

Blocking them is not free: the sampler is an exponential race over per-token rates, so removing
a token's rate from the race makes every other event happen slightly sooner. The size of that
distortion is measured, not assumed -- `engine.Engine.blocked_mass()` reports the share of
pre-death probability mass the block removes, and the run's README quotes it.
"""
import json
import os

import numpy as np
import pandas as pd

from common import DATA_DIR  # noqa: E402  (unified_eval/common.py)

LABELS_CSV = os.environ.get("ROSMAP_LABELS", os.path.join(DATA_DIR, "rosmap", "labels.csv"))


def repeatable_ids_for(labels_csv):
    """这份 build **实测**为可重复的 token id（model space），没有就返回 None。

    build.py 在写出 .bin 之后数了一遍"哪些 token 对某个人出现过 >1 次"，写进
    meta.json 的 `repeatable_tokens_postshift`。这比前缀可靠：`--no-global-dedup` 会让
    用药 ON/OFF 也重复，而它不在任何 `--nodedup` 前缀里，按前缀推会**漏**——漏掉的后果是
    生成侧把第二次 `STATIN_ON` 永久封掉，数据里明明有。

    返回 None 表示这份 meta.json 是旧格式（没有这个键），调用方退回前缀规则。
    """
    meta = os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "meta.json")
    if not os.path.exists(meta):
        return None
    with open(meta) as fh:
        m = json.load(fh)
    v = m.get("repeatable_tokens_postshift")
    return None if v is None else tuple(int(x) for x in v)


def repeatable_prefixes_for(labels_csv):
    """这份 build 声明为"可重复"的 token 名前缀。

    规则跟着**数据**走，不跟着代码走：../../tokenization/build.py 把 `--nodedup` 的前缀写进
    同目录的 meta.json（`repeatable_token_prefixes`），这里读回来。交付版那份 meta.json 没有
    这个键 -> () -> 全局不重复，也就是下面 REPEATABLE_TOKENS 那段注释描述的原样行为。

    为什么不写成模块常量：两个分词现在要并存（data/rosmap 与 data/rosmap_nodedup），而
    rollout 的 no-repeat 规则是**分词的性质**。写死在代码里就得靠人记得改，而忘了改不会报错，
    只会让不去重的那个模型在生成侧永远发不出第二个 MMSE token —— 正是这次实验要看的东西。
    ROSMAP_REPEATABLE_PREFIXES 可以覆盖，只为调试用。
    """
    env = os.environ.get("ROSMAP_REPEATABLE_PREFIXES")
    if env is not None:
        return tuple(x for x in (y.strip() for y in env.split(",")) if x)
    meta = os.path.join(os.path.dirname(os.path.abspath(labels_csv)), "meta.json")
    if os.path.exists(meta):
        with open(meta) as fh:
            return tuple(json.load(fh).get("repeatable_token_prefixes", ()))
    return ()

# Ordinal families, by the prefix their level names share. EXPLICIT, not derived by splitting
# on "_": half this vocabulary has an underscore in it (STUDY_ROS, APOE_e4_0, HTN_ONSET,
# BRAAK_low), so a blind prefix split would invent a dozen scales that are not scales and would
# make them repeatable in the sampler.
#
# The keys are upstream's names where a counterpart exists ("COG" for our COGN_*), so that
# radc_states.AUX_SCALES and plotting_style._SCALE_COLORS keep working unedited.
SCALE_PREFIX = {
    "MMSE": "MMSE_", "COG": "COGN_", "BMI": "BMI_",
    "SBP": "SBP_", "DBP": "DBP_", "GLU": "GLU_", "HBA1C": "HBA1C_",
    "HDL": "HDL_", "LDL": "LDL_", "GFR": "GFR_",
    "PSQI": "PSQI_", "APNEA": "APNEA_",
    "CRP": "CRP_", "IL6": "IL6_", "TNFA": "TNFA_",
}

# Ordered worst -> best is NOT what labels.csv gives; it gives low -> high on the measurement.
# These two are the families where the two orders coincide (lower MMSE and lower cognition are
# both worse), which is what lets perdomain.py score "worsens >= 1 bin" as "moves to a lower
# index". Every other family is left out of SEVERITY_ORDER and scored as "moves >= 1 bin",
# exactly as upstream does for BMI: asking whether HDL "worsened" needs a clinical direction
# this tokenization does not carry.
SEVERITY_ORDERED = ("MMSE", "COG")

# Post-mortem block: everything after Death in the table. Named by prefix for the same reason
# the scales are.
PATHOLOGY_PREFIX = ("BRAAK_", "CERAD_", "GPATH_", "AMYL_", "TANG_", "TDP_", "LEWY_",
                    "ARTSCL_", "CAA_", "CVDA_", "MICROINF_", "INFARCT_")

# Human-readable names for the tokens that reach a figure axis. The raw labels are
# SCREAMING_SNAKE ids and would otherwise be printed as row names.
#
# THE CUT POINTS ARE PART OF THE NAME, and that is the point. ../../tokenization/spec.py bins
# with `searchsorted(edges, v, side="right")`, i.e. each bin is closed on the LEFT: MMSE
# `borderline` is 24 <= x < 27, not "24-26 inclusive". Writing the interval into the label is
# what stops a reader importing upstream's Folstein boundaries, which are closed the other way.
DISPLAY = {
    "MMSE_normal": "MMSE ≥27", "MMSE_borderline": "MMSE 24–26", "MMSE_impaired": "MMSE <24",
    "COGN_high": "Global cognition ≥0", "COGN_mid": "Global cognition −1.0..0",
    "COGN_low": "Global cognition <−1.0",
    "STROKE": "Stroke", "DEPRESSION": "Depression",
    "HTN_ONSET": "Hypertension onset", "DM_ONSET": "Diabetes onset",
    "CHF_ONSET": "Heart failure onset", "CLAUD_ONSET": "Claudication onset",
    "HEART_ONSET": "Heart condition onset",
    "ANTIHYP_ON": "Antihypertensive started", "ANTIHYP_OFF": "Antihypertensive stopped",
    "STATIN_ON": "Statin started", "STATIN_OFF": "Statin stopped",
    "DIABRX_ON": "Diabetes Rx started", "DIABRX_OFF": "Diabetes Rx stopped",
    "ADRX_ON": "AD Rx started", "ADRX_OFF": "AD Rx stopped",
    "AD_DX": "AD diagnosis", "Death": "Death",
}


class Resolved:
    """Id families for ONE label table. Attribute names mirror upstream's `Resolved`."""

    def __init__(self, labels, repeatable_prefixes=None, repeatable_ids=None):
        self.NAMES = [str(x) for x in labels]
        self.VOCAB_SIZE = len(self.NAMES)
        self.ID = {n: i for i, n in enumerate(self.NAMES)}
        need = ("Padding", "No event", "Death", "AD_DX")
        missing = [n for n in need if n not in self.ID]
        if missing:
            raise ValueError(f"label table is missing {missing}; not a ROSMAP tokenization")
        self.PADDING = self.ID["Padding"]
        self.NO_EVENT = self.ID["No event"]
        self.DEATH = self.ID["Death"]
        self.AD_DX = self.ID["AD_DX"]
        self.ENDPOINT_IDS = (self.AD_DX, self.DEATH)
        # Death is absorbing. The AD diagnosis is NOT: subjects go on being observed after it,
        # and dying afterwards is the competing risk the evaluation has to respect.
        self.TERMINATION_TOKENS = (self.DEATH,)

        self.SCALES = {}
        for key, pre in SCALE_PREFIX.items():
            ids = tuple(i for i, n in enumerate(self.NAMES) if n.startswith(pre))
            if ids:
                self.SCALES[key] = ids
        self.SEVERITY_ORDER = {k: self.SCALES[k] for k in SEVERITY_ORDERED if k in self.SCALES}
        self.SCALE_OF = {t: k for k, ids in self.SCALES.items() for t in ids}

        self.PATHOLOGY_IDS = tuple(i for i, n in enumerate(self.NAMES)
                                   if n.startswith(PATHOLOGY_PREFIX))
        self.MED_IDS = tuple(i for i, n in enumerate(self.NAMES)
                             if n.endswith(("_ON", "_OFF")))
        # once-only incident events: the graded onsets plus the two standalone diagnoses
        self.KEEP_FIRST_IDS = tuple(i for i, n in enumerate(self.NAMES)
                                    if n.endswith("_ONSET") or n in ("STROKE", "DEPRESSION"))
        self.ONSET_IDS = ()          # see the module docstring: no graded families here

        # 默认为空，**由数据决定**（见 repeatable_prefixes_for）。下面这段注释讲的是交付版
        # （data/rosmap）为什么是空的；`--nodedup MMSE,COGN` 那份分词会在这里得到 6 个
        # MMSE_*/COGN_* token，因为它的 .bin 里这些 token 确实每人多次出现。
        #
        # EMPTY, AND THAT IS A CORRECTION. 早先这里把序数箱和用药事件列为"可重复"，理由是
        # 它们编码当前状态、状态会反复。那个理由对**上游**的分词成立，对**我们的**不成立：
        # ../../tokenization/build.py:204 在 run-length 去重之后还有第二级全局去重
        #
        #     seen = set()                      # Delphi 语义: first occurrence only
        #     for age_d, tid in ev:
        #         if tid in seen: continue
        #
        # 所以**每个 token id 每人最多出现一次**（实测没有违规）。一个人
        # normal -> impaired -> normal 在 .bin 里只剩 `normal, impaired`，最后那次回归被丢掉；
        # MMSE 路径因此从不含重复字母，最长 3 步。上游 Delphi 的 `no_repeat=True`（封掉所有
        # 已发射 token）**恰好就是这份数据的规则**，不需要豁免。
        #
        # 代价说清楚：这也意味着数据本身**低计**了真实的往复。一个在 26 分附近来回跳的人，
        # 真实轨迹是 N-B-N-B，.bin 里只有 N-B。要恢复它得改 build.py 去掉第二级去重
        # （并相应改 spec，因为 Delphi 的 no_repeat 语义也会跟着变）。
        pre = tuple(REPEATABLE_PREFIXES if repeatable_prefixes is None else repeatable_prefixes)
        self.REPEATABLE_PREFIXES = pre
        if repeatable_ids is not None:
            # 实测的 id 列表优先。它是 .bin 的事实，前缀只是命令行参数的回声。
            self.REPEATABLE_TOKENS = tuple(sorted(int(t) for t in repeatable_ids
                                                  if 0 <= int(t) < self.VOCAB_SIZE))
        else:
            self.REPEATABLE_TOKENS = tuple(i for i, n in enumerate(self.NAMES)
                                           if pre and n.startswith(pre))

        # Statics are whatever is left, and they must be contiguous -- if they are not, a token
        # was renamed out of a family above and would be silently treated as a covariate.
        # 直接用各家族，**不要**再经由 REPEATABLE_TOKENS：后者现在是空的（见上），
        # 借它来认量表/用药会让这些 token 落进"未分类"，把下面的连续性检查搞崩。
        classified = (set(t for ids in self.SCALES.values() for t in ids)
                      | set(self.MED_IDS) | set(self.ONSET_IDS)
                      | set(self.KEEP_FIRST_IDS) | set(self.PATHOLOGY_IDS)
                      | {self.PADDING, self.NO_EVENT, self.AD_DX, self.DEATH})
        statics = sorted(set(range(self.VOCAB_SIZE)) - classified)
        if statics != list(range(statics[0], statics[-1] + 1)):
            raise ValueError(
                f"the unclassified tokens are not contiguous: {statics}. Either a token was "
                f"renamed out of one of the families above, or a new family was added and this "
                f"resolver has not been told about it.")
        self.STATIC_FIRST, self.STATIC_LAST = statics[0], statics[-1]
        self.STATIC_IDS = tuple(statics)
        self.IGNORE_TOKENS = [self.PADDING] + list(statics)
        self.DT_IGNORE_TOKENS = list(self.IGNORE_TOKENS) + [self.NO_EVENT]
        # Never sampled in a rollout: see "PATHOLOGY IS BLOCKED" in the module docstring.
        self.SAMPLE_BLOCKED = tuple(self.PATHOLOGY_IDS)

        assert classified | set(statics) == set(range(self.VOCAB_SIZE))
        self._check()

    def _check(self):
        # The two orders perdomain.py depends on. It scores "worsened" as "moved to a LOWER bin
        # index", which is only true while labels.csv lists these two families worst-first.
        for key, worst, best in (("MMSE", "MMSE_impaired", "MMSE_normal"),
                                 ("COG", "COGN_low", "COGN_high")):
            ids = self.SCALES.get(key)
            if not ids:
                continue
            assert self.NAMES[ids[0]] == worst and self.NAMES[ids[-1]] == best, (
                f"{key} is not ordered worst-first in labels.csv: "
                f"{[self.NAMES[t] for t in ids]}. perdomain.py would score recovery as decline.")
        assert self.DEATH not in self.REPEATABLE_TOKENS
        assert self.AD_DX not in self.REPEATABLE_TOKENS
        assert not (set(self.SAMPLE_BLOCKED) & set(self.REPEATABLE_TOKENS))
        # every token the figure scores must be one the model is trained to emit
        for t in (self.AD_DX, self.DEATH) + tuple(self.SCALES.get("MMSE", ())):
            assert t not in self.IGNORE_TOKENS and t != self.NO_EVENT
        return True

    def display(self, tok):
        """Readable name for a token id, falling back to the raw label."""
        n = self.NAMES[int(tok)]
        return DISPLAY.get(n, n)


def resolve(labels, repeatable_prefixes=None, repeatable_ids=None):
    """`Resolved` for a label table (a list of names in id order, i.e. labels.csv's column)."""
    return Resolved(labels, repeatable_prefixes, repeatable_ids)


def resolve_csv(path):
    """`Resolved` for a build's labels.csv, with that build's own no-repeat rule."""
    return Resolved([str(x) for x in pd.read_csv(path)["event_name"].tolist()],
                    repeatable_prefixes_for(path), repeatable_ids_for(path))


# ---------------------------------------------------------------- the live table
# Bound at import to the build under data/, so code that does not care about cross-tokenization
# scoring can use the module constants. figure2_core re-resolves against the checkpoint's own
# labels.csv regardless.
# 模块默认值：Resolved(labels) 不带路径时（radc_states.configure 的调用点）用这个。
REPEATABLE_PREFIXES = repeatable_prefixes_for(LABELS_CSV)
_LIVE = resolve_csv(LABELS_CSV)

NAMES = _LIVE.NAMES
ID = _LIVE.ID
VOCAB_SIZE = _LIVE.VOCAB_SIZE
PADDING = _LIVE.PADDING
NO_EVENT = _LIVE.NO_EVENT
DEATH = _LIVE.DEATH
AD_DX = _LIVE.AD_DX
SCALES = _LIVE.SCALES
SEVERITY_ORDER = _LIVE.SEVERITY_ORDER
IGNORE_TOKENS = _LIVE.IGNORE_TOKENS
STATIC_IDS = _LIVE.STATIC_IDS
REPEATABLE_TOKENS = _LIVE.REPEATABLE_TOKENS
TERMINATION_TOKENS = _LIVE.TERMINATION_TOKENS
KEEP_FIRST_IDS = _LIVE.KEEP_FIRST_IDS
MED_IDS = _LIVE.MED_IDS
ONSET_IDS = _LIVE.ONSET_IDS
PATHOLOGY_IDS = _LIVE.PATHOLOGY_IDS
SAMPLE_BLOCKED = _LIVE.SAMPLE_BLOCKED
# The ordinal staging scale. MMSE is the staging analogue for this cohort for the same reason
# upstream gives: cogdx / dcfdx_lv are end-of-study summaries and both leak, so
# ../tokenization/spec.py excludes them (see meta.json "excluded").
STAGE_TOKENS_DISK = tuple(t - 1 for t in SCALES["MMSE"])


def describe():
    r = _LIVE
    return (f"ROSMAP vocabulary: {r.VOCAB_SIZE} tokens\n"
            f"  statics      {r.STATIC_FIRST}..{r.STATIC_LAST} ({len(r.STATIC_IDS)})\n"
            f"  scales       {len(r.SCALES)} families, "
            f"{sum(len(v) for v in r.SCALES.values())} bins\n"
            f"  keep-first   {len(r.KEEP_FIRST_IDS)}   meds {len(r.MED_IDS)}\n"
            f"  pathology    {len(r.PATHOLOGY_IDS)} (blocked from rollouts)\n"
            f"  repeatable   {len(r.REPEATABLE_TOKENS)} "
            f"{[r.NAMES[t] for t in r.REPEATABLE_TOKENS]} "
            f"(prefixes {list(r.REPEATABLE_PREFIXES) or 'none'})\n"
            f"  AD_DX={r.AD_DX}  Death={r.DEATH}  No event={r.NO_EVENT}")


if __name__ == "__main__":
    print(describe())
