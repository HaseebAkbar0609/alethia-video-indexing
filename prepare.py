#!/usr/bin/env python3
"""One-time preparation (needs internet, not timed):
  1. download the official yolo11l.pt next to run.py
  2. build a TensorRT FP16 engine (dynamic batch/shape) for this GPU -> yolo11l.engine
  3. check the engine against the PyTorch model on real-looking input; if TensorRT is not
     usable or disagrees, the engine is removed and run.py falls back to PyTorch FP16.

    python prepare.py [--batch 64] [--no-engine]
"""
import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-engine", action="store_true")
    args = ap.parse_args()
    os.chdir(HERE)
    from ultralytics import YOLO

    pt = os.path.join(HERE, "yolo11l.pt")
    if not os.path.exists(pt):
        YOLO("yolo11l.pt")  # Ultralytics downloads the official weights into the cwd
    assert os.path.exists(pt), "yolo11l.pt download failed"
    print("[prepare] weights ok", os.path.getsize(pt))
    engine = os.path.join(HERE, "yolo11l.engine")
    if args.no_engine:
        return
    import torch

    if not torch.cuda.is_available():
        print("[prepare] no CUDA GPU, skipping TensorRT engine")
        return
    t = time.time()
    try:
        out = YOLO(pt).export(format="engine", half=True, dynamic=True, batch=args.batch, imgsz=640, device=0, verbose=False)
        if os.path.abspath(out) != engine:
            shutil.move(out, engine)
        print(f"[prepare] engine built in {time.time() - t:.0f}s")
    except Exception as e:
        print(f"[prepare] TensorRT export failed ({e}); run.py will use PyTorch FP16", file=sys.stderr)
        if os.path.exists(engine):
            os.remove(engine)
        return
    # sanity check: engine vs PyTorch on the same frames (synthetic but structured input)
    try:
        sys.path.insert(0, HERE)
        import numpy as np

        import run

        rng = np.random.default_rng(0)
        x = (rng.random((16, 384, 640, 3)) * 255).astype(np.uint8)
        a = run.Detector(pt, engine, "cuda:0", True, args.batch, "trt")
        b = run.Detector(pt, engine, "cuda:0", True, args.batch, "torch")
        ya = a.model(torch.from_numpy(x).cuda().flip(-1).permute(0, 3, 1, 2).half().div(255).contiguous())
        yb = b.model(torch.from_numpy(x).cuda().flip(-1).permute(0, 3, 1, 2).half().div(255).contiguous())
        ya = ya[0] if isinstance(ya, (list, tuple)) else ya
        yb = yb[0] if isinstance(yb, (list, tuple)) else yb
        err = (ya.float()[:, 4:] - yb.float()[:, 4:]).abs().max().item()
        print(f"[prepare] engine vs pytorch max class-score diff {err:.4f}")
        if not np.isfinite(err) or err > 0.05:
            raise RuntimeError(f"engine output differs too much ({err})")
    except Exception as e:
        print(f"[prepare] engine check failed ({e}); removing engine", file=sys.stderr)
        if os.path.exists(engine):
            os.remove(engine)


if __name__ == "__main__":
    main()
