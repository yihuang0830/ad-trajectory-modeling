"""
radc_delphi -- the Delphi engine behind unified_eval's Delphi rollouts (rollout_delphi.py, rollout_metrics.py).

Moved here from the old figure2_eval/ (removed 2026-11-01). The package name comes from
an internal delphi-fine-tuning repository, whose figure code imported `radc_delphi.{vocab,engine,batching}`;
this project runs the upstream gerstung-lab Delphi on the 129-token ROSMAP tokenization instead, and
every adaptation to that model and vocabulary is confined to these three modules:

    vocab.py     our labels.csv, resolved into the token id families
    batching.py  our .bin layout (pid, age_days, disk_token) and the disk -> model +1 shift
    engine.py    our checkpoint, and a rollout sampler (death terminates, per-tokenization no-repeat rules)

Upstream delphi/model.py and utils.py are not in this repo: they live in $AD_BASE/delphi on RIS
(common.DELPHI_DIR), and callers must already have unified_eval/ on sys.path.
"""
from . import vocab  # noqa: F401
