#!/usr/bin/env python3
"""Offline policy search on per-frame oracle dumps (tools/oracle.py output).

For every candidate sampling policy it reproduces what run.py would output and scores it
with the brief's metrics against the all-frames reference, plus the inference cost.

  reference : per-second labels from ALL frames (stock FP32 Ultralytics if present in the
              dump, else the fast detector) -- exactly the rule in section 5 of the brief
  policy    : pass 1 = K uniform samples per second; seconds whose samples disagree are
              refined with all their frames (or E extra frames)

    python tools/simulate.py oracle_fast_49.npz [oracle_stock_49.npz ...]
"""
import sys
from collections import defaultdict

import numpy as np


def load(paths):
    vids = defaultdict(dict)
    for p in paths:
        z = np.load(p)
        for k in z.files:
            v, f = k.split("|")
            vids[v].setdefault(f, z[k])
    return vids


def median_floor(x):
    x = np.sort(x)
    n = len(x)
    return int(x[n // 2]) if n % 2 else int((int(x[n // 2 - 1]) + int(x[n // 2])) // 2)


def per_second(secs, pres, ppl, idx_by_sec, chosen=None):
    out = {}
    for s, idx in idx_by_sec.items():
        if chosen is not None:
            idx = chosen[s]
        p = pres[idx]
        n = len(idx)
        out[s] = (frozenset(np.nonzero(2 * p.sum(0) >= n)[0].tolist()), median_floor(ppl[idx]))
    return out


def score(ref, res):
    tp = fp = fn = ex = w1 = n = 0
    for k, (ro, rp) in ref.items():
        so, sp = res[k]
        tp += len(ro & so)
        fp += len(so - ro)
        fn += len(ro - so)
        ex += sp == rp
        w1 += abs(sp - rp) <= 1
        n += 1
    return 2 * tp / max(2 * tp + fp + fn, 1), ex / n, w1 / n


def pick_uniform(m, k):
    """indices of frames chosen by run.py's UniformSampler for a second holding m frames"""
    if k >= m:
        return list(range(m))
    out = []
    for j in range(k):
        i = int(np.ceil(j * m / k - 1e-9))
        if not out or i > out[-1]:
            out.append(min(i, m - 1))
    return out


def simulate(v, src, k, refine, tol):
    secs = v["secs"]
    pres, ppl = v[src + "_pres"].astype(bool), v[src + "_ppl"].astype(int)
    idx_by_sec = defaultdict(list)
    for i, s in enumerate(secs.tolist()):
        idx_by_sec[s].append(i)
    chosen = {}
    cost = 0
    refined = 0
    for s, idx in idx_by_sec.items():
        sel = [idx[i] for i in pick_uniform(len(idx), k)]
        cnt = pres[sel].sum(0)
        unsure = bool(((cnt > 0) & (cnt < len(sel))).any()) or (ppl[sel].max() - ppl[sel].min() > tol)
        if refine and unsure:
            refined += 1
            if refine == "all":
                sel = idx
            else:
                e = int(refine)
                extra = [idx[i] for i in pick_uniform(len(idx), k + e) if idx[i] not in sel]
                sel = sorted(set(sel) | set(extra))
        chosen[s] = sel
        cost += len(sel)
    return per_second(secs, pres, ppl, idx_by_sec, chosen), cost, refined, len(idx_by_sec)


def main():
    vids = load(sys.argv[1:])
    has_stock = [v for v, d in vids.items() if "stock_pres" in d]
    print(f"videos {len(vids)}, with stock FP32 reference {len(has_stock)}")
    # 1) fast detector on all frames vs stock FP32 on all frames (best achievable)
    if has_stock:
        agree_p = agree_c = nfr = 0
        R, F = {}, {}
        for v in has_stock:
            d = vids[v]
            if len(d["stock_ppl"]) != len(d["fast_ppl"]):
                print("  length mismatch", v)
                continue
            agree_p += (d["stock_pres"] == d["fast_pres"]).all(1).sum()
            agree_c += (d["stock_ppl"] == d["fast_ppl"]).sum()
            nfr += len(d["secs"])
            idx = defaultdict(list)
            for i, s in enumerate(d["secs"].tolist()):
                idx[s].append(i)
            for s, val in per_second(d["secs"], d["stock_pres"].astype(bool), d["stock_ppl"].astype(int), idx).items():
                R[(v, s)] = val
            for s, val in per_second(d["secs"], d["fast_pres"].astype(bool), d["fast_ppl"].astype(int), idx).items():
                F[(v, s)] = val
        f1, ex, w1 = score(R, F)
        print(f"per-frame agreement fast vs stock: presence {agree_p / nfr:.4f}, people {agree_c / nfr:.4f} ({nfr} frames)")
        print(f"ALL frames, fast vs stock reference: F1 {f1:.4f} exact {ex:.4f} within1 {w1:.4f} ({len(R)} seconds)")

    # 2) policy sweep. Reference = fast detector on all frames (isolates the sampling error),
    #    and, on the stock subset, the stock reference (sampling + numeric error together).
    print("\nK refine tol | F1 exact within1 | frames/sec-of-video  refined% | (stock ref: F1 exact within1)")
    total_frames = sum(len(d["secs"]) for d in vids.values())
    for k in (1, 2, 3, 4, 6):
        for refine in (None, "4", "8", "all"):
            for tol in (0, 1):
                if refine is None and tol == 1:
                    continue
                R, S, R2, S2 = {}, {}, {}, {}
                cost = refined = nsec = 0
                for v, d in vids.items():
                    out, c, r, n = simulate(d, "fast", k, refine, tol)
                    idx = defaultdict(list)
                    for i, s in enumerate(d["secs"].tolist()):
                        idx[s].append(i)
                    ref = per_second(d["secs"], d["fast_pres"].astype(bool), d["fast_ppl"].astype(int), idx)
                    for s in ref:
                        R[(v, s)] = ref[s]
                        S[(v, s)] = out[s]
                    if "stock_pres" in d and len(d["stock_ppl"]) == len(d["fast_ppl"]):
                        sref = per_second(d["secs"], d["stock_pres"].astype(bool), d["stock_ppl"].astype(int), idx)
                        for s in sref:
                            R2[(v, s)] = sref[s]
                            S2[(v, s)] = out[s]
                    cost += c
                    refined += r
                    nsec += n
                f1, ex, w1 = score(R, S)
                line = f"{k} {str(refine):>4} {tol} | {f1:.4f} {ex:.4f} {w1:.4f} | {cost / nsec:6.2f} ({cost / total_frames:5.1%} of frames) {refined / nsec:5.1%}"
                if R2:
                    f1b, exb, w1b = score(R2, S2)
                    line += f" | {f1b:.4f} {exb:.4f} {w1b:.4f}"
                print(line, flush=True)


if __name__ == "__main__":
    main()
