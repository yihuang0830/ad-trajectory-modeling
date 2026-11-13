"""
engine.py -- inference. The ONE place the ported evaluation code is allowed to touch the model.

    from radc_delphi.engine import load
    eng = load("../delphi/Delphi-ROSMAP/ckpt.pt", data_dir="data/rosmap", device="cpu")
    T, A = eng.simulate(tokens, ages, n_mc=100)

The model is upstream gerstung-lab Delphi (`../delphi/model.py`), loaded unmodified. What this
file owns is the ROLLOUT, because upstream's `Delphi.generate` does not have the semantics the
figure needs and the difference is silent rather than loud:

  1. TERMINATION. Death must be absorbing. Upstream supports this via `termination_tokens` but
     DEFAULTS IT TO `[1269]` -- a UKB id that in our 129-token table does not exist -- behind a
     warnings.warn. A rollout that inherits that default is an immortal cohort that keeps
     accruing AD hazard for as long as it is allowed to run.

  2. NO-REPEAT IS ALL-OR-NOTHING UPSTREAM. `generate(no_repeat=True)` blocks EVERY token the
     trajectory has already emitted. That is right for once-only events (a second STROKE token
     means nothing) and wrong for the 52 ordinal bins and medication switches, which encode a
     CURRENT STATE and must be able to recur: under upstream's rule a subject who goes
     MMSE_normal -> MMSE_borderline can never come back, so recovery is structurally
     unsamplable and panel a's "MMSE_normal (back to normal)" row would be identically zero.
     This loop takes an explicit `repeatable_tokens` set instead.

  3. THE POST-MORTEM BLOCK. See vocab.py. Blocked from the draw; the cost is measured by
     `blocked_mass()` rather than assumed to be zero.

  3b. NO-REPEAT IS THE DATA'S RULE, NOT AN OPTION -- AND THE DATA NOW SAYS IT PER BUILD.
     The delivered tokenization (data/rosmap) applies a GLOBAL per-subject dedup after its
     run-length pass ("Delphi 语义: first occurrence only"), so every token id appears at most
     once per subject -- no violations in any subject. Upstream Delphi's blanket
     `no_repeat=True` is exactly right for THAT build, and `vocab.REPEATABLE_TOKENS` resolves
     to empty for it. An earlier version of this file exempted the ordinal bins and medication
     events on the grounds that "states recur"; they do in reality, but not in that .bin, and
     the exemption let a share of simulated clinical tokens (a larger share of medication
     tokens) be repeats the data cannot contain.

     The ablation build (data/rosmap_nodedup, `build.py --nodedup MMSE,COGN`) emits one MMSE
     and one cogn_global token PER VISIT and exempts both from either dedup, so for that .bin
     those 6 tokens DO recur and blocking them would make recovery unsamplable. The rule is
     therefore read from the build's own meta.json (`repeatable_token_prefixes`) rather than
     typed here -- see vocab.repeatable_prefixes_for. Getting it wrong is silent in exactly
     one direction: a model trained on repeats, scored under blanket no-repeat, emits at most
     one MMSE token per trajectory and every MMSE row of panel a collapses.

  4. CONTEXT WINDOW. Upstream feeds the whole growing sequence back in, and its own block-size
     assert is commented out (model.py:214). Our attention mask buffer is built at
     block_size=96, so a prefix plus 96 sampled tokens would index past it. This slides
     a block_size window instead. That is sound HERE and would not be in a GPT: this model has
     NO positional embedding (model.py:165 -- `wpe` is commented out), position enters only
     through the age embedding, so a shifted window is not a shifted representation. The
     no-repeat bookkeeping still reads the FULL history, not the window.

TOKEN SPACE. Everything crossing this API is MODEL space (= labels.csv row index). The +1 disk
shift belongs to batching.py.
"""
import os
import sys
import hashlib

import numpy as np
import pandas as pd
import torch

from . import vocab as V
from .batching import get_p2i, patient_stream

from common import DELPHI_DIR as _DELPHI  # noqa: E402  (unified_eval/common.py；调用方都已把 unified_eval 放进 sys.path)
if _DELPHI not in sys.path:
    sys.path.insert(0, _DELPHI)
from model import Delphi, DelphiConfig  # noqa: E402  (upstream delphi/model.py)

DAYS_PER_YEAR = 365.25
# What simulate() writes into a position that is past death or past max_age. Any code reading a
# simulated age must exclude these; they are not ages. Matches upstream's `mask_time`.
PAD_AGE = -10000.0
# Upstream's clamp on a single sampled waiting time. Kept identical so the time model is the
# one the checkpoint was trained under.
MAX_WAIT_DAYS = 365 * 80


def _fingerprint(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:12]


def load(ckpt_path, data_dir=None, device="cpu", strict_vocab=True, vocab_labels=None):
    """Load a checkpoint into an Engine, verifying it against the vocabulary it was trained on.

    UPSTREAM CHECKS A FINGERPRINT HERE AND WE CANNOT. delphi-fine-tuning's train.py records the
    md5 of labels.csv in the checkpoint (`vocab_sig`), so its loader can refuse a checkpoint
    scored against a different tokenization. Our `../delphi/train.py` is upstream Delphi and
    writes no such field, so the strongest check available is the VOCAB SIZE plus the
    checkpoint's own `ignore_tokens` -- which does pin the static block, and therefore catches
    a re-tokenization that moved it. `strict_vocab` controls whether a mismatch raises.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = dict(ck["model_args"])
    lp = vocab_labels or (os.path.join(data_dir, "labels.csv") if data_dir else None)
    if lp and os.path.exists(lp):
        labels = [str(x) for x in pd.read_csv(lp)["event_name"].tolist()]
    else:
        labels = list(V.NAMES)
    if args["vocab_size"] != len(labels):
        raise RuntimeError(
            f"checkpoint vocab_size {args['vocab_size']} != {len(labels)} rows in {lp}. That "
            f"table is not the one this checkpoint was trained on.")
    # no-repeat 规则跟着**这份**分词走（meta.json 与 labels.csv 同目录），不跟着 live 模块走。
    rep = V.repeatable_prefixes_for(lp) if lp else tuple(V.REPEATABLE_PREFIXES)
    rep_ids = V.repeatable_ids_for(lp) if lp else None
    res = V.resolve(labels, rep, rep_ids)
    ck_ignore = sorted(int(t) for t in args.get("ignore_tokens", res.IGNORE_TOKENS))
    if strict_vocab and ck_ignore != sorted(res.IGNORE_TOKENS):
        raise RuntimeError(
            f"the checkpoint's ignore_tokens {ck_ignore[:5]}...({len(ck_ignore)}) do not match "
            f"the static block this label table resolves to "
            f"{sorted(res.IGNORE_TOKENS)[:5]}...({len(res.IGNORE_TOKENS)}). The checkpoint and "
            f"{lp} are different tokenizations. Pass strict_vocab=False only if you know why.")
    valid = {f for f in DelphiConfig.__dataclass_fields__}
    model = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return Engine(model, args, device, ckpt=ck, data_dir=data_dir, ckpt_path=ckpt_path,
                  ckpt_sig=_fingerprint(ckpt_path), labels=labels, repeatable_prefixes=rep,
                  repeatable_ids=rep_ids)


class Engine:
    def __init__(self, model, args, device, ckpt=None, data_dir=None, ckpt_path=None,
                 ckpt_sig=None, labels=None, allow_no_event=True, repeatable_prefixes=None,
                 repeatable_ids=None):
        self.model = model
        self.args = args
        self.device = device
        self.block_size = int(args["block_size"])
        self.ckpt = ckpt or {}
        self.data_dir = data_dir
        self.ckpt_path = ckpt_path
        # md5 of the checkpoint file. Every cached rollout is keyed on this: a cache hit against
        # a different set of weights is silent and produces plausible numbers.
        self.ckpt_sig = ckpt_sig
        self.labels = list(labels) if labels is not None else list(V.NAMES)
        self.vocab_size = len(self.labels)
        self.res = V.resolve(self.labels, repeatable_prefixes, repeatable_ids)
        # `ignore_tokens` comes off the CHECKPOINT, so a model trained against a different
        # static block keeps its own rather than inheriting the live one.
        ignore = set(int(t) for t in args.get("ignore_tokens", self.res.IGNORE_TOKENS))
        self.ignore_tokens = sorted(ignore)
        self.content_ids = np.array([t for t in range(self.vocab_size) if t not in ignore],
                                    dtype=np.int64)

        # WHY No event STAYS SAMPLEABLE, against upstream's choice.
        # delphi-fine-tuning blocks it, on the grounds that it is a synthetic marker rather than
        # a clinical event. That is right for their model and wrong for ours, because the marker
        # is part of THIS model's time model: `config/train_delphi_rosmap.py` sets
        # no_event_token_rate=5, so the checkpoint was fitted with one injected marker per five
        # years and learned a rate for it. Blocking a token removes its rate from the
        # exponential race, which makes every real event fire sooner -- i.e. blocking it would
        # bias timing, in exchange for removing tokens the figure ignores anyway (No event is
        # not in GRID_TOKENS and reaches no panel). Measured both ways in ../README.md.
        self.allow_no_event = allow_no_event

        blocked = set(self.ignore_tokens) | set(self.res.SAMPLE_BLOCKED)
        if not allow_no_event:
            blocked.add(self.res.NO_EVENT)
        self.sample_blocked = sorted(blocked)
        self._blocked_t = torch.tensor(self.sample_blocked, dtype=torch.long, device=device)
        self._repeat_t = torch.tensor(sorted(set(self.res.REPEATABLE_TOKENS) |
                                             {self.res.PADDING, self.res.NO_EVENT}),
                                      dtype=torch.long, device=device)
        self._term_t = torch.tensor(sorted(self.res.TERMINATION_TOKENS), dtype=torch.long,
                                    device=device)

        # 经验访视大小（见 make_visit_sizes.py / README 3.2）。缺文件就退回逐 token 采样，
        # 并在 simulate() 里按 use_visit_sizes 明确分支——不要静默改变生成语义。
        # 先看**数据集目录**里那份，再退回仓库根那份。访视大小是分词的统计量，不是仓库的：
        # 不去重那份分词每次访视多带 2 个认知 token（均值随之变大），用交付版的分布去补发
        # 会静默地把访视采样调小，而 README 3.2 量到的每一行提升几乎正好等于访视大小。
        self.visit_sizes = None
        self.visit_sizes_path = None
        cands = ([os.path.join(data_dir, "visit_sizes.npy")] if data_dir else []) + [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "visit_sizes.npy")]
        for vs in cands:
            if os.path.exists(vs):
                self.visit_sizes = torch.as_tensor(np.load(vs).astype(np.int64).ravel(),
                                                   dtype=torch.long, device=device)
                self.visit_sizes_path = vs
                break
        # token -> 量表家族序号（-1 = 不属于任何序数家族）。用于"一次访视内同一家族最多一个
        # token"这条约束：train.bin 的所有访视里 0 次违规，是硬不变量。
        names = list(self.res.SCALES)
        fam = np.full(self.vocab_size, -1, dtype=np.int64)
        for fi, nm in enumerate(names):
            for t in self.res.SCALES[nm]:
                fam[t] = fi
        self._fam_of = torch.tensor(fam, dtype=torch.long, device=device)
        self._n_fam = len(names)
        # 这些 token 不开启一次"访视"：No-event 是合成标记；Death 终止 rollout，而它在数据里
        # 的同日伙伴全是病理 token，而病理被禁止采样，所以死亡那一"访视"只应有 Death 本身。
        self._no_visit_t = torch.tensor(sorted({self.res.NO_EVENT, self.res.DEATH}),
                                        dtype=torch.long, device=device)


    # ------------------------------------------------------------------ data access
    def load_split(self, split, data_dir=None):
        """(data, p2i, subjects) for one split.

        `subjects` is None: upstream carries a per-subject table (subjects.csv, with the split
        assignment and the covariates its cohort code reads); `../tokenization/build.py` writes
        only the .bin files and the split IS the file. Nothing in the ported figure2 code reads
        the third element -- it is destructured as `_sub` at all three call sites -- so the slot
        is kept and left empty rather than faked.
        """
        d = data_dir or self.data_dir
        arr = np.fromfile(os.path.join(d, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
        return arr, get_p2i(arr), None

    @staticmethod
    def stream(data, p2i, k):
        """(ages_days, model_tokens, projid) for the k-th subject of a split."""
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        ages, toks = patient_stream(data, s, n)
        return ages, toks, int(data[s, 0])

    @staticmethod
    def prefix_at(ages, toks, cut_day):
        """The subject's stream truncated to tokens at or before `cut_day`."""
        m = ages <= cut_day
        return ages[m], toks[m]

    # ------------------------------------------------------------------ mask_ties
    @torch.no_grad()
    def _logits(self, idx, age, mask_ties=True):
        """最后一个位置的 logits，可选地按**训练时的 mask_ties 语义**打分。

        WHY. model.py 里这段只在传 targets 时生效：

            if targets is not None and self.config.mask_ties:
                attn_mask *= (age != targets_age)      # 屏蔽与目标同龄的 token

        本分词把一次访视的所有 token 放在同一天（大部分 Delta-t = 0），所以训练时**每个位置**
        的隐状态都是在"看不见自己这次访视的兄弟"的条件下算出来的。生成时不传 targets，掩码整个
        失效，模型处在一个从未训练过的注意力配置里 —— 实测同一批位置、同一个模型，
        `MMSE_normal` 的预测从接近真值掉到严重低估。

        HOW. 掩码只依赖 `targets_age`，不依赖 targets 的内容，所以把"每个位置下一个 token 的
        年龄"喂进去就能精确复现；targets 随便给一个合法值，它只进损失，而损失这里不用。
        最后一个位置的下一个年龄未知，但必然**大于**当前所有年龄，填一个哨兵值即可 —— 那一位
        因此不屏蔽任何 key，这正是训练时"预测新访视第一个 token"的配置。

        **默认 False，因为实测更差，而不是因为它不忠实。** 在 rollout 里打开它，at-risk 组的
        `MMSE_normal` 占比进一步下降（离真值和训练口径更远）。

        为什么"更忠实"反而更差：训练口径那个值是在**真实病史**上测的；rollout 的上下文是
        模型**自己生成的**病史，而那段历史本身已经离群 —— 同一次测量里 rollout 发出的
        `MMSE_impaired` 的占比远高于真值。在一段过度恶化的历史上再屏蔽掉同访视信息，
        只会让下一步更恶化。这是自回归漂移，不是掩码能修的。

        保留这个开关是为了让上面这句话可复现，不是为了让人打开它。
        """
        if not mask_ties:
            logits, _, _ = self.model(idx, age)
            return logits[:, -1, :].float()
        big = torch.full((idx.shape[0], 1), 1e9, dtype=age.dtype, device=age.device)
        tgt_age = torch.cat([age[:, 1:], big], dim=1)
        logits, _, _ = self.model(idx, age, targets=idx, targets_age=tgt_age)
        return logits[:, -1, :].float()

    # ------------------------------------------------------------------ single-step
    @torch.no_grad()
    def next_token_probs(self, tokens, ages, renormalise=True, mask_ties=False):
        """P(next token) after the given history, over the trained content columns."""
        idx = torch.as_tensor(np.asarray(tokens[-self.block_size:])[None, :],
                              dtype=torch.long, device=self.device)
        age = torch.as_tensor(np.asarray(ages[-self.block_size:])[None, :],
                              dtype=torch.float32, device=self.device)
        p = torch.softmax(self._logits(idx, age, mask_ties)[0], -1).cpu().numpy()
        if renormalise:
            keep = np.zeros_like(p)
            keep[self.content_ids] = p[self.content_ids]
            tot = keep.sum()
            p = keep / tot if tot > 0 else keep
        return p

    def rate_profile(self, tokens, ages):
        """Total event rate and its composition at one prefix -- the time model, laid bare.

        Returns a dict with:
          mean_wait_days  exp(-logsumexp(logits)) over the sampleable columns. This IS the
                          sampler's mean waiting time per emitted token: the race
                          min_k Exp(exp(logit_k)) is Exp(sum_k exp(logit_k)).
          noevent_share   the No-event marker's share of that total rate.
          clinical_per_yr real (non-marker) tokens per year = (1 - noevent_share) / wait.

        WHY THIS IS THE DIAGNOSTIC THAT MATTERS. model.py trains the total rate against
        `dt`, and under mask_ties `dt` is "time from the last UNTIED token" -- i.e. the gap to
        the previous VISIT, never 0. So the model learns "the next token arrives in about one
        visit gap" while generation emits ONE token per waiting time, and a real visit carries
        several clinical tokens. That single mismatch, not the token choice, is what deflates every
        rate in Figure 2's panel a2. See README section 3.2.
        """
        idx = torch.as_tensor(np.asarray(tokens[-self.block_size:])[None, :],
                              dtype=torch.long, device=self.device)
        age = torch.as_tensor(np.asarray(ages[-self.block_size:])[None, :],
                              dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _, _ = self.model(idx, age)
        lg = logits[0, -1].double().clone()
        lg[torch.tensor(self.sample_blocked, device=lg.device)] = -torch.inf
        tot = float(torch.exp(torch.logsumexp(lg, -1)))
        r = torch.exp(lg)
        ne = float(r[self.res.NO_EVENT]) / tot if tot > 0 else float("nan")
        wait = 1.0 / tot if tot > 0 else float("inf")
        return {"mean_wait_days": wait, "noevent_share": ne,
                "clinical_per_yr": (1.0 - ne) / wait * DAYS_PER_YEAR}

    def blocked_mass(self, tokens, ages):
        """Share of the next-token RATE that the post-mortem block removes at this prefix.

        The sampler is an exponential race over per-token rates exp(logit), so the probability
        that the next token is a pathology token is exactly its share of the total rate. This
        is the size of the distortion vocab.py's docstring promises to measure rather than wave
        away: run it over a sample of live prefixes and quote the distribution.
        """
        idx = torch.as_tensor(np.asarray(tokens[-self.block_size:])[None, :],
                              dtype=torch.long, device=self.device)
        age = torch.as_tensor(np.asarray(ages[-self.block_size:])[None, :],
                              dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _, _ = self.model(idx, age)
        r = torch.exp(logits[0, -1].double())
        # the statics are never candidates in the first place, so they are out of the
        # denominator too -- otherwise the share is diluted by columns nothing could sample
        r[torch.tensor(self.ignore_tokens, device=r.device)] = 0.0
        tot = float(r.sum())
        path = float(r[torch.tensor(list(self.res.PATHOLOGY_IDS), device=r.device)].sum())
        return path / tot if tot > 0 else float("nan")

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def simulate(self, tokens, ages, n_mc=200, until_age_years=110.0, max_new_tokens=96,
                 seed=0, use_visit_sizes=True, mask_ties=False):
        """Monte-Carlo forward simulation from a prefix.

        Returns (sim_tokens, sim_ages) of shape (n_mc, T) in MODEL space, WITH the prefix
        included. Positions past death or past max_age carry token 0 and age PAD_AGE.

        `use_visit_sizes=True`（默认）让生成以**一次访视**为单位，而不是一个 token 为单位。

        WHY. `model.py` 的时间头是拿 `dt` 训练的，而 mask_ties 把 `dt` 换成"到上一个非同龄
        token 的时间"，即**访视间隔**。所以模型学到的 `exp(-logsumexp(logits))`
        ≈ 观测访视间隔中位数。但逐 token 采样每等一次只发一个 token，而真实的非基线访视
        携带**多个**临床 token（make_visit_sizes.py）。差的那一个因子就是 README 3.2 量到的
        临床 token 率缺口（模型的每年 token 率明显低于观测）。

        HOW. 采一个等待时间发出第一个 token 之后，从经验分布抽 k，再在**同一个 age** 上补发
        k-1 个 token，时间不走。

        补发的 token 复用**同一次 forward 的 logits**，这不是近似：mask_ties 下一个 token 被
        训练成不 attend 同一次访视的兄弟 token，所以按模型自己的因子分解，一次访视内的 token
        在给定历史下条件独立。从一次 forward i.i.d. 抽 k 个正是训练目标规定的做法。

        三条约束，都是数据的硬不变量，缺一条就会生成数据里不存在的流：
          * 全局不重复：每个 token 每人最多一次（vocab.REPEATABLE_TOKENS 为空），补发时要排除
            已发射过的、**包括本次访视里刚发的**；
          * 一次访视内同一量表家族最多一个 token（训练集所有访视 0 次违规）；
          * No-event 与 Death 不开启访视（见 __init__ 的 _no_visit_t）。

        k 是**逐轨迹**抽的，不是整批共用一个。上游明确把"整批共用"列为简化，代价是同一步内各
        条轨迹相关；这里为每条轨迹独立抽 k，短于最大 k 的轨迹在多余位置填 padding（PAD_AGE），
        最终掩码会把它们排除，且不打乱剩余位置的时间顺序。
                """
        # ../../delphi/model.py 的 `visit_heads` 打开之后，速率在 time_head、访视大小在
        # size_head，lm_head 的 logits 只剩"发什么"、**不再是速率**。下面那句
        # `-exp(-logits)*log(u)` 照样能跑完，只会给出一条时间尺度完全错的轨迹 —— 没有报错、
        # 没有 NaN、AUC 还是个像样的数。所以在这里硬炸。
        # 只挡 rollout：probe_head / direct_head / probe_loss 只用编码器，那些路径不受影响，
        # 因此这个检查**不能**放进 load()。
        if self.args.get("visit_heads"):
            raise RuntimeError(
                "这个 checkpoint 是用 visit_heads=True 训练的：等待时间在 model.time_head、"
                "访视大小在 model.size_head，simulate() 里的指数竞赛已经失效。需要的采样是 "
                "wait ~ Exp(time_head(x))、k ~ Categorical(size_head(x))，再从 softmax(lm_head) "
                "抽 k 个 token 放在同一个 age 上（即把下面 use_visit_sizes 分支里的 "
                "self.visit_sizes 经验分布换成 size_head）。判别头（aux_head）不受影响，"
                "它不参与生成。")
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        pre_t = np.asarray(tokens)[-self.block_size:]
        pre_a = np.asarray(ages)[-self.block_size:]
        idx = torch.as_tensor(np.tile(pre_t, (n_mc, 1)), dtype=torch.long, device=self.device)
        age = torch.as_tensor(np.tile(pre_a, (n_mc, 1)), dtype=torch.float32, device=self.device)
        max_age = until_age_years * DAYS_PER_YEAR
        B = self.block_size
        sizes = self.visit_sizes if (use_visit_sizes and self.visit_sizes is not None) else None
        # max_new_tokens 数的是"步"（= 采样了几次等待时间 = 几次访视），**不是张量列数**。
        # 第一版按列计数，而访视补发是按全批最大 k 追加列的，于是每步吃掉好几列的
        # 预算，192 的预算只换来很少几步，轨迹在死亡之前就被截断 —— P(死亡 ever) 大幅塌掉。
        # 列数另有一个宽松的硬上限，只为防内存失控。
        n_steps = 0
        col_cap = int(max_new_tokens) * (1 + (int(sizes.max().item()) if sizes is not None else 0))

        # 每条轨迹的"当前年龄"，单独维护，**不要**从 age[:, -1] 读。
        # 访视补发会给未激活的行写入 PAD_AGE，所以最后一列不再是那一行的真实年龄；第一版就是
        # 这么写的，结果大部分轨迹从 -10000 起算、几乎不往前走。
        cur_age = age[:, -1].clone()

        while n_steps < max_new_tokens and idx.shape[1] < col_cap:
            logits = self._logits(idx[:, -B:], age[:, -B:], mask_ties)
            logits[:, self._blocked_t] = -torch.inf
            # 全局不重复（vocab.REPEATABLE_TOKENS 为空，所以这里封掉一切已发射的）。
            # padding / No-event 映射到第 0 列，那一列本来就是 -inf，等于不封。
            fill = idx.clone()
            fill[torch.isin(fill, self._repeat_t)] = self.res.PADDING
            logits = logits.scatter(1, fill, -torch.inf)
            # 指数竞赛：t_k ~ Exp(exp(logit_k))，下一个 token = argmin。与上游 generate() 一致，
            # 包括那个 clamp，所以采样的时间模型就是 checkpoint 训练时的那个。
            u = torch.rand(logits.shape, generator=g).to(logits.device)
            t = torch.clamp(-torch.exp(-logits) * torch.log(u), min=0, max=MAX_WAIT_DAYS)
            wait, nxt = t.min(1)
            cur_age = cur_age + wait
            idx = torch.cat((idx, nxt[:, None]), dim=1)
            age = torch.cat((age, cur_age[:, None]), dim=1)
            n_steps += 1

            # ---- 这一次访视的其余 token，落在同一个 age 上 ----------------------------
            # k 是**整批共用**一个，不是逐轨迹抽。上游也是这么做的，并把"同一步内各行相关"
            # 列为明示的简化；我一开始改成逐轨迹抽，给未激活的行填 PAD，结果踩进一个很隐蔽的
            # 坑：模型的 attn_mask 让 padding 位置**只能 attend 自己**（model.py 的
            # "Except for padding" 那行），于是那一行下一步的 logits[:, -1, :] 是从一个什么都
            # 看不见的位置取的 —— 死亡速率因此塌掉，P(死亡 ever) 明显下降。
            # 共用 k 不产生空洞，访视大小的**边际分布**依然精确等于 visit_sizes。
            if sizes is not None:
                k_shared = int(sizes[torch.randint(len(sizes), (1,), generator=g)].item())
                opens = ~torch.isin(nxt, self._no_visit_t)
                fam_used = torch.zeros((n_mc, self._n_fam), dtype=torch.bool,
                                       device=self.device)
                f0 = self._fam_of[nxt]
                h0 = f0 >= 0
                if bool(h0.any()):
                    fam_used[h0, f0[h0]] = True
                for _j in range(max(k_shared - 1, 0)):
                    if idx.shape[1] >= col_cap:
                        break
                    p = torch.softmax(logits, -1)          # 被屏蔽列已是 -inf -> 0
                    fill2 = idx.clone()                    # 含本次访视里刚发的
                    fill2[torch.isin(fill2, self._repeat_t)] = self.res.PADDING
                    p = p.scatter(1, fill2, 0.0)
                    # 同一访视内，已用过的量表家族整族屏蔽（数据里所有访视 0 次违规）
                    fam_blocked = fam_used.gather(
                        1, self._fam_of.clamp(min=0).expand(n_mc, -1))
                    p = torch.where((self._fam_of >= 0).expand(n_mc, -1) & fam_blocked,
                                    torch.zeros_like(p), p)
                    p = p.scatter(1, self._no_visit_t.expand(n_mc, -1), 0.0)
                    tot = p.sum(1, keepdim=True)
                    # ok=False 的行照样追加一列，但**填的是它自己刚发的那个 token**是错的，
                    # 填 padding 又会造成上面说的空洞。这里的解法是：整批同步，只有当
                    # "这一步没有任何行该发"时才停 —— 不该发的行用 PAD，但它们本来就没开访视
                    # （opens=False），而 opens 只在 No-event / Death 时为 False，那两种情况
                    # 下一步的预测本来也不该从它们之后继续同访视，所以用 nxt 自身占位即可。
                    ok = opens & (tot.squeeze(1) > 0)
                    if not bool(ok.any()):
                        break
                    pn = torch.where(tot > 0, p / tot.clamp(min=1e-30),
                                     torch.full_like(p, 1.0 / p.shape[1]))
                    d2 = torch.multinomial(pn, 1, generator=g).squeeze(1)
                    # 不开访视的行重复放它自己的 nxt 会违反全局不重复；放 No-event 既不违反
                    # 不变量（它是唯一允许重复的 token），也不会制造 attention 空洞。
                    draw = torch.where(ok, d2, torch.full_like(d2, self.res.NO_EVENT))
                    idx = torch.cat((idx, draw[:, None]), dim=1)
                    age = torch.cat((age, cur_age[:, None]), dim=1)
                    fj = self._fam_of[draw]
                    hj = ok & (fj >= 0)
                    if bool(hj.any()):
                        fam_used[hj, fj[hj]] = True

            done = torch.isin(idx, self._term_t).any(-1) | (cur_age > max_age)
            if bool(done.all()):
                break

        # Mask everything strictly after the first termination token, and anything past max_age.
        # Upstream's expression, kept verbatim: the inner cumsum marks positions at or after the
        # first Death, the outer one makes ">1" mean "strictly after it", so the Death token
        # itself survives and is the last real position of the trajectory.
        term = torch.isin(idx, self._term_t)
        pad = (torch.cumsum(torch.cumsum(term, 1).bool().int(), 1) > 1) | (age > max_age)
        idx = idx.masked_fill(pad, self.res.PADDING)
        age = age.masked_fill(pad, PAD_AGE)
        return idx.cpu().numpy(), age.cpu().numpy()


# ---------------------------------------------------------------------- batch readers
def first_event_ages(sim_tokens, sim_ages, token_id, after_day=None):
    """Age (YEARS) at which `token_id` first fires in each trajectory; NaN if never.

    Reads a batch simulate() already produced rather than re-simulating: for the competing-risk
    endpoints the event age and the death age must come from THE SAME draw.
    """
    big = 1e18
    hit = (sim_tokens == token_id) & (sim_ages > PAD_AGE + 1)
    if after_day is not None:
        hit &= sim_ages > after_day
    first = np.where(hit, sim_ages, big).min(1)
    return np.where(first >= big, np.nan, first / DAYS_PER_YEAR)


def sim_end_ages(sim_ages):
    """Age (YEARS) of the last real position of each trajectory.

    A rollout that hit max_new_tokens without dying simply stops emitting. It is not a subject
    observed to be stable forever -- any state read past this age is carried-forward padding.
    """
    return np.where(sim_ages > PAD_AGE + 1, sim_ages, -np.inf).max(1) / DAYS_PER_YEAR


# ---------------------------------------------------------------------- observed side
def aalen_johansen(event_age, event_type, entry_age, grid_ages):
    """Cause-specific cumulative incidence of event_type == 1, with 2 as the competing event.

    LEFT TRUNCATION IS NOT OPTIONAL. Subjects enter at their baseline age, so nobody is at risk
    on the age axis before they enrol; ignoring that puts the whole cohort in the denominator
    from birth and drives every incidence estimate towards zero. All ages in YEARS.
    """
    event_age = np.asarray(event_age, float)
    event_type = np.asarray(event_type, int)
    entry_age = np.asarray(entry_age, float)
    order = np.argsort(event_age)
    ea, et, en = event_age[order], event_type[order], entry_age[order]
    times = np.unique(ea[et > 0])
    surv, cif, curve = 1.0, 0.0, []
    for t in times:
        at_risk = int(((en <= t) & (ea >= t)).sum())
        if at_risk == 0:
            continue
        d1 = int(((ea == t) & (et == 1)).sum())
        d2 = int(((ea == t) & (et == 2)).sum())
        cif += surv * d1 / at_risk                # increment uses S BEFORE this time
        surv *= (1.0 - (d1 + d2) / at_risk)
        curve.append((t, cif))
    if not curve:
        return np.zeros(len(grid_ages)), []
    ct = np.array([c[0] for c in curve])
    cv = np.array([c[1] for c in curve])
    out = np.array([cv[ct <= g][-1] if (ct <= g).any() else 0.0 for g in grid_ages])
    return out, curve


def risk_set_size(entry_age, exit_age, at_age):
    """How many subjects are actually at risk at `at_age`. Quote this with every CIF value."""
    return int(((np.asarray(entry_age) <= at_age) & (np.asarray(exit_age) >= at_age)).sum())
