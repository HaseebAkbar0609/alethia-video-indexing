#!/usr/bin/env python3
"""Per-frame dump ("oracle"): run the detector on EVERY frame of some videos and save, per
frame, its second, the 5 presence flags and the person count. Saved per backend:

  fast  : run.py's own Detector (TensorRT/PyTorch FP16, our letterbox + GPU post-processing)
  stock : stock Ultralytics API, model.predict(frame, imgsz=640, conf=0.4) in FP32 -- this is
          how the graders describe their reference

Any sampling policy can then be simulated offline (tools/simulate.py) without a GPU.

    python tools/oracle.py --input DIR --out oracle.npz [--limit N] [--stock-limit M]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run  # noqa: E402


def stream_fast(mp4, det, batch):
    """Decode every frame, run the fast detector chunk by chunk (constant memory)."""
    import torch

    secs, P, C = [], [], []
    shape = [None]

    def emit(kind, payload):
        fr, sc = payload
        shape[0] = fr.shape[1:3]
        for s in range(0, len(fr), batch):
            x = torch.from_numpy(fr[s : s + batch])
            p, c = det.infer(x.pin_memory() if torch.cuda.is_available() else x)
            P.append(p)
            C.append(c)
        secs.append(sc)

    run.WORKER_DECODE_THREADS = 2
    info = run.decode_pass(mp4, lambda idx, rel, sec: True, emit, batch)
    if not secs:
        return None
    return np.concatenate(secs), np.concatenate(P), np.concatenate(C), shape[0], info


def stream_stock(mp4, stock, batch=32):
    import io

    import av

    c = av.open(io.BytesIO(mp4))
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    sp, sc, imgs = [], [], []
    rk = None

    def flush():
        for r in stock.predict(imgs, imgsz=640, conf=0.4, verbose=False, device=0 if __import__("torch").cuda.is_available() else "cpu"):
            cls = r.boxes.cls.int().tolist()
            sp.append([cc in cls for cc in run.COCO_IDS])
            sc.append(sum(1 for cc in cls if cc == 0))
        imgs.clear()

    for f in c.decode(st):
        if rk is None:
            rk = int(round((f.rotation or 0) / 90.0)) % 4
        a = f.to_ndarray(format="bgr24")
        imgs.append(np.ascontiguousarray(np.rot90(a, rk)) if rk else a)
        if len(imgs) == batch:
            flush()
    if imgs:
        flush()
    c.close()
    return np.asarray(sp, dtype=bool), np.asarray(sc, dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stock-limit", type=int, default=0, help="also run stock FP32 Ultralytics on the first M videos")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    import torch

    tasks, _ = run.list_tasks(args.input)
    if args.limit:
        tasks = tasks[: args.limit]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    det = run.Detector(os.path.join(run.HERE, "yolo11l.pt"), os.path.join(run.HERE, "yolo11l.engine"), dev, True, args.batch, args.backend)
    print("backend", det.backend_name, flush=True)
    stock = None
    if args.stock_limit:
        from ultralytics import YOLO

        stock = YOLO(os.path.join(run.HERE, "yolo11l.pt"))
    cache = run.RowGroupCache()
    res = {}
    t_fast = t_stock = n_fast = n_stock = 0
    for k, (path, rg, i, row) in enumerate(tasks):
        vid = f"{os.path.basename(path)}:{row}"
        mp4 = cache.get(path, rg, i)
        t = time.time()
        try:
            r = stream_fast(mp4, det, args.batch)
        except Exception as e:
            print(vid, "decode error", e, flush=True)
            continue
        if r is None:
            continue
        secs, fp, fc, shp, info = r
        if dev != "cpu":
            torch.cuda.synchronize()
        t_fast += time.time() - t
        n_fast += len(secs)
        res[vid + "|secs"] = secs.astype(np.int32)
        res[vid + "|fast_pres"] = fp
        res[vid + "|fast_ppl"] = fc.astype(np.int16)
        res[vid + "|shape"] = np.asarray(shp)
        if stock is not None and k < args.stock_limit:
            t = time.time()
            sp, sc = stream_stock(mp4, stock)
            t_stock += time.time() - t
            n_stock += len(sp)
            if len(sp) != len(secs):
                print(vid, "frame count mismatch", len(sp), len(secs), flush=True)
            res[vid + "|stock_pres"] = sp
            res[vid + "|stock_ppl"] = sc
        print(f"{k + 1}/{len(tasks)} {vid} frames={len(secs)} fast {n_fast / max(t_fast, 1e-6):.0f} fr/s (incl. decode)"
              + (f" stock {n_stock / max(t_stock, 1e-6):.0f} fr/s" if n_stock else ""), flush=True)
        np.savez_compressed(args.out, **res)
    np.savez_compressed(args.out, **res)


if __name__ == "__main__":
    main()
