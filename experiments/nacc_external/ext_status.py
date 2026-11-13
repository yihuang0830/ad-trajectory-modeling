"""ext_status.py -- 进度一览（只读模型记录、核对结果、文件是否存在；不读任何 NACC 标签）。

    python ext_status.py
"""
import glob
import json
import os

import ext_common as C


def main():
    print(f"{'model':20s} {'dev':4s} {'val 4ep':>8s} {'vs pub':>16s} {'reload test/val':>16s}  H  E  rosmap_nv")
    names = C.NONDELPHI + [f"{m}-nv" for m in C.NONDELPHI]
    for n in names:
        p = os.path.join(C.MODEL_DIR, f"{n}.json")
        if not os.path.exists(p):
            print(f"{n:20s} (not trained)")
            continue
        i = json.load(open(p))
        vp = i.get("vs_published")
        vs = "-" if not vp else ("identical" if vp["identical"] else f"d={vp['d_part1_2y_4endpoints']:+.4f}")
        ck = os.path.join(C.SCORE_DIR, "rosmap", f"{n}.check.json")
        rl = "-"
        if os.path.exists(ck):
            c = json.load(open(ck))
            rl = "/".join("same" if c[s]["identical"] else f"{c[s]['max_abs_diff']:.1e}" for s in ("test", "val"))
        has = lambda s: "y" if os.path.exists(os.path.join(C.SCORE_DIR, s, f"{n}.npz")) else (       # noqa: E731
            f"{len(glob.glob(os.path.join(C.SCORE_DIR, s, f'{n}.part*')))}p" if glob.glob(os.path.join(C.SCORE_DIR, s, f"{n}.part*")) else ".")
        print(f"{n:20s} {i['device'][:4]:4s} {i['val']['part1_2y_4endpoints']:8.4f} {vs:>16s} {rl:>16s}  {has('H')}  {has('E')}  {has('rosmap_nv')}")
    for n in C.DELPHI:
        infos = sorted(glob.glob(os.path.join(C.SCORE_DIR, "*", f"{n}*info.json")))
        for f in infos:
            j = json.load(open(f))
            print(f"{n:20s} {os.path.relpath(f, C.SCORE_DIR):40s} n={j['n']} {j['minutes']:.1f}min {j.get('vs_published', '')}")
    print("SEAL:", os.path.exists(os.path.join(C.OUT, "SEAL.json")), " results:", os.listdir(os.path.join(C.OUT, "results"))
          if os.path.exists(os.path.join(C.OUT, "results")) else None)


if __name__ == "__main__":
    main()
