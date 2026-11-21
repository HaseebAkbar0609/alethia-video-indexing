#!/usr/bin/env python3
"""Slow reference: run YOLO11l on EVERY frame with the stock Ultralytics API and build
per-second labels exactly as the brief (section 5) describes. Used only to check run.py.

    python tools/reference.py --input <folder> --output reference.jsonl [--limit N] [--videos file:row,...]
"""
import argparse
import io
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import CLASS_NAMES, COCO_IDS, list_tasks, median_floor  # noqa: E402


def frames_of(mp4):
    import av

    c = av.open(io.BytesIO(mp4))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    tb = s.time_base
    t0 = None
    prev = None
    fps = float(s.average_rate or 30)
    k = 0
    for fr in c.decode(s):
        t = float(fr.pts * tb) if fr.pts is not None else ((prev + 1 / fps) if prev is not None else 0.0)
        prev = t
        if t0 is None:
            t0 = t
            k = int(round((fr.rotation or 0) / 90.0)) % 4
        img = fr.to_ndarray(format="bgr24")
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
        yield max(t - t0, 0.0), img
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--videos", default="")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from ultralytics import YOLO

    model = YOLO("yolo11l.pt")
    tasks, _ = list_tasks(args.input)
    if args.videos:
        want = set(args.videos.split(","))
        tasks = [t for t in tasks if f"{os.path.basename(t[0])}:{t[3]}" in want]
    if args.limit:
        tasks = tasks[: args.limit]
    out = open(args.output, "w")
    for path, rg, i, row in tasks:
        vid = f"{os.path.basename(path)}:{row}"
        mp4 = pq.ParquetFile(path).read_row_group(rg, columns=["mp4"]).column("mp4")[i].as_py()
        per_sec_p = {}
        per_sec_n = {}
        batch, secs = [], []
        last = None

        def flush():
            kw = {"half": True} if args.half else {}
            res = model.predict(batch, imgsz=640, conf=0.4, verbose=False, device=args.device, **kw)
            for s, r in zip(secs, res):
                cls = r.boxes.cls.int().tolist()
                per_sec_p.setdefault(s, []).append([c in cls for c in COCO_IDS])
                per_sec_n.setdefault(s, []).append(sum(1 for c in cls if c == 0))

        try:
            for t, img in frames_of(mp4):
                s = int(math.floor(t + 1e-9))
                last = s if last is None else max(last, s)
                batch.append(img)
                secs.append(s)
                if len(batch) == args.batch:
                    flush()
                    batch, secs = [], []
            if batch:
                flush()
            err = None if last is not None else "could not decode video"
        except Exception as e:
            err = f"{type(e).__name__}"
        if last is not None:
            for s in range(last + 1):
                if s in per_sec_p:
                    arr = np.asarray(per_sec_p[s])
                    n = len(arr)
                    objs = [CLASS_NAMES[c] for c in range(5) if 2 * arr[:, c].sum() >= n]
                    ppl = median_floor(per_sec_n[s])
                else:
                    objs, ppl = [], 0
                out.write(json.dumps({"video": vid, "second": s, "objects": objs, "people": ppl, "nframes": len(per_sec_n.get(s, []))}) + "\n")
        if err:
            out.write(json.dumps({"video": vid, "error": err}) + "\n")
        out.flush()
        print(vid, "done", last, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
