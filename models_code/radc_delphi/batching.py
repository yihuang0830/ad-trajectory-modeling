"""
batching.py -- the .bin layout, and the ONE place the disk -> model token shift happens.

Our `.bin` is `np.uint32` triples `(patient_id, age_days, disk_token)`, written by
`../tokenization/build.py`. Upstream delphi's `utils.get_batch` shifts tokens by +1 on its last
line so that model id 0 can mean Padding; every id that leaves this module is therefore
MODEL space (= the labels.csv row index), and nothing downstream needs to know about the shift.

`get_p2i` is re-exported from upstream `delphi/utils.py` rather than reimplemented -- the
training loop and the evaluation must agree on where each subject's rows start, and two copies
of that rule is exactly the drift upstream's own docstring warns about.
"""
import sys

import numpy as np

from common import DELPHI_DIR as _DELPHI  # noqa: E402  (unified_eval/common.py；调用方都已把 unified_eval 放进 sys.path)
if _DELPHI not in sys.path:
    sys.path.insert(0, _DELPHI)

from utils import get_p2i  # noqa: E402,F401  (upstream delphi/utils.py)


def patient_stream(data, start, n):
    """(ages_days, model_tokens) for the `n` rows starting at `start`, age-ascending.

    STABLE sort, and that is not cosmetic: `../tokenization/README.md` measures most
    Delta-t as exactly 0 because the ROSMAP time axis is an annual grid and every token of one
    visit carries the same age. A non-stable sort reorders tied tokens differently across
    platforms, which is the same defect that moved upstream `evaluate_auc`'s Death AUC
    between macOS/MPS and Linux/CPU until `stable=True` was added to get_batch.
    """
    rows = data[start:start + n]
    ages = rows[:, 1].astype(np.float64)
    toks = rows[:, 2].astype(np.int64) + 1          # disk -> model
    o = np.argsort(ages, kind="stable")
    return ages[o], toks[o]


def n_visits(ages, toks, ignored):
    """Distinct VISIT ages in one stream, counting only non-ignored tokens.

    "Distinct age" alone is the wrong count. In this build the statics sit at exactly the
    baseline visit age, so they do not add a phantom visit the way upstream's (age_bl - 1 day)
    placement does -- but a subject whose only non-static content is a single background row
    would still be counted as having a visit to predict. Counting content tokens only makes the
    rule mean what it says in both builds.
    """
    a = np.asarray(ages, float)
    t = np.asarray(toks)
    content = ~np.isin(t, list(ignored))
    return int(np.unique(a[(a > 0) & content]).size)


def filter_cohort(data, p2i, min_visits=2, short_min_visits=2, stage_disk=(), ignored_disk=()):
    """Rows of `p2i` for subjects inside the training cohort.

    UPSTREAM HAS A REAL FILTER HERE AND WE DO NOT. delphi-fine-tuning's `train.py` drops
    subjects with too little follow-up before fitting (N -> N' on its train split), so
    its Figure 2 must be able to restrict scoring to the same population -- that is what its
    "matched" cohort means. Our `../delphi/train.py` is upstream gerstung-lab Delphi and has no
    such filter: it fits on every subject in train.bin.

    The rule is implemented anyway, with the same signature, so that `--cohort matched` remains
    meaningful if a filter is ever added to training. With the defaults below it keeps everyone
    who has something to predict, and `figure2_core.training_cohort_mask` reports how many rows
    it dropped rather than letting "matched" quietly mean "all".

    `stage_disk` / `ignored_disk` are DISK ids, because this reads the raw .bin.
    """
    stage = set(int(t) for t in stage_disk)
    ignored = set(int(t) for t in ignored_disk)
    keep = []
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        rows = data[s:s + n]
        a, t = rows[:, 1].astype(float), rows[:, 2].astype(np.int64)
        content = ~np.isin(t, list(ignored)) if ignored else np.ones(len(t), bool)
        nv = int(np.unique(a[(a > 0) & content]).size)
        n_stage = int(np.isin(t, list(stage)).sum()) if stage else 0
        if nv >= min_visits or (nv >= short_min_visits and n_stage >= 2):
            keep.append(p2i[k])
    return np.array(keep) if keep else np.empty((0, 2), dtype=p2i.dtype)
