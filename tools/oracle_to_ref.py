#!/usr/bin/env python3
"""Turn per-frame oracle dumps into an all-frames reference results file (brief, section 5),
so real run.py outputs can be scored end-to-end with tools/compare.py.

    python tools/oracle_to_ref.py oracle_fast_49.npz [--source stock] > reference_49.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import CLASS_NAMES, median_floor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--source", default="fast", choices=["fast", "stock"])
    args = ap.parse_args()
    vids = defaultdict(dict)
    for p in args.npz:
        z = np.load(p)
        for k in z.files:
            v, f = k.split("|")
            vids[v].setdefault(f, z[k])
    for v, d in sorted(vids.items()):
        if args.source + "_pres" not in d:
            continue
        secs = d["secs"]
        pres = d[args.source + "_pres"].astype(bool)
        ppl = d[args.source + "_ppl"].astype(int)
        idx = defaultdict(list)
        for i, s in enumerate(secs.tolist()):
            idx[s].append(i)
        for s in range(int(secs.max()) + 1):
            ix = idx.get(s)
            if ix:
                p = pres[ix]
                objs = [CLASS_NAMES[c] for c in range(5) if 2 * p[:, c].sum() >= len(ix)]
                n = median_floor(ppl[ix].tolist())
            else:
                objs, n = [], 0
            print(json.dumps({"video": v, "second": s, "objects": objs, "people": n}))


if __name__ == "__main__":
    main()
