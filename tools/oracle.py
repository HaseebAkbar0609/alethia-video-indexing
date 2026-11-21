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


def decode_all(mp4):
    frames, secs, raw = [], [], []

    def emit(kind, payload):
        if kind == "frames":
            frames.append(payload[0])
            secs.append(payload[1])

    run.WORKER_DECODE_THREADS = 2
    # 1000 targets per second == every frame is "sampled"
    run.sample_and_decode(mp4, 1000, emit, 256)
    if not frames:
        return None, None
    return np.concatenate(frames), np.concatenate(secs)


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
    det = run.Detector(os.path.join(run.HERE, "yolo11l.pt"), os.path.join(run.HERE, "yolo11l.engine"), "cuda:0", True, args.batch, args.backend)
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
        try:
            fr, secs = decode_all(mp4)
        except Exception as e:
            print(vid, "decode error", e, flush=True)
            continue
        if fr is None:
            continue
        t = time.time()
        P, C = [], []
        for s in range(0, len(fr), args.batch):
            p, c = det.infer(torch.from_numpy(fr[s : s + args.batch]).pin_memory())
            P.append(p)
            C.append(c)
        torch.cuda.synchronize()
        t_fast += time.time() - t
        n_fast += len(fr)
        res[vid + "|secs"] = secs.astype(np.int32)
        res[vid + "|fast_pres"] = np.concatenate(P)
        res[vid + "|fast_ppl"] = np.concatenate(C).astype(np.int16)
        res[vid + "|shape"] = np.asarray(fr.shape[1:3])
        if stock is not None and k < args.stock_limit:
            # stock API on the ORIGINAL frames (re-decode without letterbox)
            import io

            import av

            c = av.open(io.BytesIO(mp4))
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            imgs = []
            rk = None
            for f in c.decode(st):
                if rk is None:
                    rk = int(round((f.rotation or 0) / 90.0)) % 4
                a = f.to_ndarray(format="bgr24")
                imgs.append(np.ascontiguousarray(np.rot90(a, rk)) if rk else a)
            c.close()
            t = time.time()
            sp, sc = [], []
            for s in range(0, len(imgs), 32):
                for r in stock.predict(imgs[s : s + 32], imgsz=640, conf=0.4, verbose=False, device=0):
                    cls = r.boxes.cls.int().tolist()
                    sp.append([cc in cls for cc in run.COCO_IDS])
                    sc.append(sum(1 for cc in cls if cc == 0))
            t_stock += time.time() - t
            n_stock += len(imgs)
            if len(sp) != len(fr):
                print(vid, "frame count mismatch", len(sp), len(fr), flush=True)
            res[vid + "|stock_pres"] = np.asarray(sp, dtype=bool)
            res[vid + "|stock_ppl"] = np.asarray(sc, dtype=np.int16)
        print(f"{k + 1}/{len(tasks)} {vid} frames={len(fr)} fast {n_fast / max(t_fast, 1e-6):.0f} fr/s"
              + (f" stock {n_stock / max(t_stock, 1e-6):.0f} fr/s" if n_stock else ""), flush=True)
        np.savez_compressed(args.out, **res)
    np.savez_compressed(args.out, **res)


if __name__ == "__main__":
    main()
