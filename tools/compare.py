#!/usr/bin/env python3
"""Score a results file against a reference file with the brief's three checks.

    python tools/compare.py reference.jsonl results.jsonl [--detail]

What is scored (only videos present in the reference count):
  objects_f1      class PRESENCE per second, micro-averaged: every (second, class) pair is one
                  decision, TP/FP/FN are pooled over all seconds and all 10 classes. People
                  COUNTS are not part of it.
  people_exact    share of seconds whose person count equals the reference exactly
  people_within1  share of seconds whose person count is off by at most one
--detail adds what the pooled numbers can hide: per-class precision/recall (rare classes),
macro F1, people accuracy by crowd size, and the worst videos.
"""
import argparse
import json
from collections import Counter, defaultdict


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


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def crowd(n):
    return "0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


def score(ref_path, res_path, verbose=True, detail=False, worst=10):
    ref, ref_err = load(ref_path)
    res, res_err = load(res_path)
    vids = {v for v, _ in ref}
    tp = fp = fn = 0
    exact = within1 = n = 0
    missing = extra = 0
    per_class = Counter()
    cls = defaultdict(Counter)                      # class -> tp/fp/fn
    vid = defaultdict(Counter)                      # video -> tp/fp/fn/n/exact/within1
    crowd_b = defaultdict(Counter)                  # crowd bucket -> n/exact/within1
    for key, (ro, rp) in ref.items():
        n += 1
        v = vid[key[0]]
        v["n"] += 1
        b = crowd_b[crowd(rp)]
        b["n"] += 1
        if key not in res:
            missing += 1
            fn += len(ro)
            v["fn"] += len(ro)
            for c in ro:
                cls[c]["fn"] += 1
            continue
        so, sp = res[key]
        for c in ro & so:
            cls[c]["tp"] += 1
        for c in so - ro:
            cls[c]["fp"] += 1
            per_class["fp:" + c] += 1
        for c in ro - so:
            cls[c]["fn"] += 1
            per_class["fn:" + c] += 1
        t, p_, n_ = len(ro & so), len(so - ro), len(ro - so)
        tp += t
        fp += p_
        fn += n_
        v["tp"] += t
        v["fp"] += p_
        v["fn"] += n_
        e, w = sp == rp, abs(sp - rp) <= 1
        exact += e
        within1 += w
        v["exact"] += e
        v["within1"] += w
        b["exact"] += e
        b["within1"] += w
    for key in res:
        if key[0] in vids and key not in ref:
            extra += 1
    F1 = f1(tp, fp, fn)
    out = {
        "videos": len(vids),
        "seconds": n,
        "objects_f1": round(F1, 4),
        "people_exact": round(exact / max(n, 1), 4),
        "people_within1": round(within1 / max(n, 1), 4),
        "missing_seconds": missing,
        "extra_seconds": extra,
        "pass": F1 >= 0.95 and exact / max(n, 1) >= 0.85 and within1 / max(n, 1) >= 0.95,
        "errors_ref": len(ref_err),
        "errors_res": len(res_err & vids) if vids else len(res_err),
        "class_errors": dict(per_class.most_common()),
    }
    if detail:
        rows = {}
        for c, k in sorted(cls.items(), key=lambda x: -(x[1]["tp"] + x[1]["fn"])):
            rows[c] = {
                "ref_seconds": k["tp"] + k["fn"],
                "precision": round(k["tp"] / max(k["tp"] + k["fp"], 1), 4),
                "recall": round(k["tp"] / max(k["tp"] + k["fn"], 1), 4),
                "f1": round(f1(k["tp"], k["fp"], k["fn"]), 4),
                "fp": k["fp"], "fn": k["fn"],
            }
        out["per_class"] = rows
        present = [r["f1"] for r in rows.values() if r["ref_seconds"] > 0]
        out["macro_f1"] = round(sum(present) / max(len(present), 1), 4)
        out["people_by_crowd"] = {
            k: {"seconds": b["n"], "exact": round(b["exact"] / b["n"], 4), "within1": round(b["within1"] / b["n"], 4)}
            for k, b in sorted(crowd_b.items(), key=lambda x: ["0", "1", "2", "3-5", "6-10", "11+"].index(x[0]))
        }
        vrows = []
        for name, k in vid.items():
            vrows.append({
                "video": name, "seconds": k["n"],
                "objects_f1": round(f1(k["tp"], k["fp"], k["fn"]), 4) if k["tp"] + k["fp"] + k["fn"] else None,
                "people_exact": round(k["exact"] / k["n"], 4), "people_within1": round(k["within1"] / k["n"], 4),
                "fp": k["fp"], "fn": k["fn"],
            })
        out["worst_videos_objects"] = sorted([r for r in vrows if r["objects_f1"] is not None],
                                             key=lambda r: r["objects_f1"])[:worst]
        out["worst_videos_people"] = sorted(vrows, key=lambda r: r["people_exact"])[:worst]
        out["videos_failing_any_threshold"] = sum(
            1 for r in vrows if (r["objects_f1"] is not None and r["objects_f1"] < 0.95)
            or r["people_exact"] < 0.85 or r["people_within1"] < 0.95)
    if verbose:
        print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("results")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--worst", type=int, default=10)
    a = ap.parse_args()
    score(a.reference, a.results, detail=a.detail, worst=a.worst)
