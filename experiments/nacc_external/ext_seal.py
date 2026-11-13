"""ext_seal.py -- 第 7 节第 4 步：E 上全部风险算完以后封存（揭盲之前）。

    python ext_seal.py            # -> $ZOO_RUN_DIR/nacc_ext/SEAL.json（只有哈希和计数，提交进仓库）

核对：每个模型在 E 上的风险文件存在、恰好覆盖 E 的全部 primary 站点、每个站点 6 × 10 个有限的风险；风险文件里记的模型 sha256
和模型文件一致。然后记下：模型文件、Delphi 检查点、协调参数、E 的特征和 token、每个风险文件、E 的标签文件（只算字节的哈希，不解析）
的 sha256。已经有揭盲结果（results/E_results.json）就拒绝重新封存。
"""
import datetime
import json
import os

import numpy as np

import ext_common as C
from ext_delphi import CKPT


def main():
    assert not os.path.exists(os.path.join(C.OUT, "results", "E_results.json")), "已经揭盲，不能重新封存"
    with np.load(os.path.join(C.OUT, "data", "E_X.npz")) as z:
        assert not any(k.startswith("Y") for k in z.files)
        M, prim = z["M"], z["primary"]
    ii, vv = np.where(M & prim[:, None])
    want = set(zip(ii.tolist(), vv.tolist()))
    names = C.ALL13 + [f"{m}-nv" for m in C.NONDELPHI if os.path.exists(os.path.join(C.SCORE_DIR, "E", f"{m}-nv.npz"))]
    files = {"harmonization.json": os.path.join(C.OUT, "harmonization.json"),
             "data/E_X.npz": os.path.join(C.OUT, "data", "E_X.npz"),
             "data/E_tok.bin": os.path.join(C.OUT, "data", "E_tok.bin"),
             "data/E_tok_lm.npz": os.path.join(C.OUT, "data", "E_tok_lm.npz"),
             "data/H.npz": os.path.join(C.OUT, "data", "H.npz"),
             "sealed/E_Y.npz": os.path.join(C.OUT, "sealed", "E_Y.npz"),
             "delphi_ckpt": CKPT}
    models = {}
    for n in names:
        sp = os.path.join(C.SCORE_DIR, "E", f"{n}.npz")
        z = np.load(sp)
        got = set(zip(z["ii"].tolist(), z["vv"].tolist()))
        assert got == want and len(z["ii"]) == len(want), f"{n}：E 的站点不全或有多余（{len(got)} / {len(want)}）"
        assert z["R"].shape == (len(want), 6, 10) and np.isfinite(z["R"]).all(), f"{n}：风险的形状或数值不对"
        if n.startswith("delphi"):
            assert str(z["model_sha256"]) == C.sha256(CKPT)
            mf = None
        else:
            mf = os.path.join(C.MODEL_DIR, f"{n}.pt")
            info = json.load(open(os.path.join(C.MODEL_DIR, f"{n}.json")))
            assert str(z["model_sha256"]) == info["sha256"] == C.sha256(mf), f"{n}：风险文件不是这个模型打的"
        models[n] = {"scores_E": C.sha256(sp), "model": None if mf is None else C.sha256(mf), "n_landmarks": len(want)}
        files[f"scores/E/{n}.npz"] = sp
    rev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GIT_REV")
    seal = {"sealed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "deployed_git": open(rev).read().strip() if os.path.exists(rev) else None,
            "n_E_primary_landmarks": len(want), "n_E_primary_persons": int(prim.sum()), "models": models,
            "sha256": {k: C.sha256(p) for k, p in files.items()}}
    json.dump(seal, open(os.path.join(C.OUT, "SEAL.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in seal.items() if k != "sha256"}, indent=1))


if __name__ == "__main__":
    main()
