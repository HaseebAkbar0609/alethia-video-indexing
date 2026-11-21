#!/usr/bin/env python3
"""Score a results file against a reference file with the brief's three checks.

    python tools/compare.py reference.jsonl results.jsonl
Only videos present in the reference are scored.
"""
import json
import sys
from collections import Counter


def load(p):
    secs, errs = {}, set()
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                errs.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (set(d["objects"]), d["people"])
    return secs, errs


def score(ref_path, res_path, verbose=True):
    ref, ref_err = load(ref_path)
    res, res_err = load(res_path)
    vids = {v for v, _ in ref}
    tp = fp = fn = 0
    exact = within1 = n = 0
    missing = extra = 0
    per_class = Counter()
    for key, (ro, rp) in ref.items():
        n += 1
        if key not in res:
            missing += 1
            fn += len(ro)
            continue
        so, sp = res[key]
        tp += len(ro & so)
        fp += len(so - ro)
        fn += len(ro - so)
        for c in so - ro:
            per_class["fp:" + c] += 1
        for c in ro - so:
            per_class["fn:" + c] += 1
        exact += sp == rp
        within1 += abs(sp - rp) <= 1
    for key in res:
        if key[0] in vids and key not in ref:
            extra += 1
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    out = {
        "videos": len(vids),
        "seconds": n,
        "objects_f1": round(f1, 4),
        "people_exact": round(exact / max(n, 1), 4),
        "people_within1": round(within1 / max(n, 1), 4),
        "missing_seconds": missing,
        "extra_seconds": extra,
        "pass": f1 >= 0.95 and exact / max(n, 1) >= 0.85 and within1 / max(n, 1) >= 0.95,
        "errors_ref": len(ref_err),
        "errors_res": len(res_err & vids) if vids else len(res_err),
        "class_errors": dict(per_class.most_common()),
    }
    if verbose:
        print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    score(sys.argv[1], sys.argv[2])
