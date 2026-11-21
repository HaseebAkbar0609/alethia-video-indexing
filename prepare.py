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


def build_engine(pt, engine, batch):
    """ONNX (Ultralytics exporter, FP32, dynamic) -> TensorRT FP16 engine with a tight profile:
    batch 1..B, H/W 32..640 (the letterbox never exceeds 640), optimised for B x 384 x 640, the
    shape of 16:9 video. Written with the Ultralytics metadata header so AutoBackend loads it."""
    import json

    import tensorrt as trt
    from ultralytics import YOLO

    model = YOLO(pt)
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True, imgsz=640, batch=batch, verbose=False)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    major = int(trt.__version__.split(".")[0])
    flags = 0 if major >= 10 else (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if not hasattr(trt.BuilderFlag, "FP16"):
        raise RuntimeError("this TensorRT has no FP16 builder flag (TensorRT >= 11)")
    config.set_flag(trt.BuilderFlag.FP16)
    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, 3, 32, 32), (batch, 3, 384, 640), (batch, 3, 640, 640))
    config.add_optimization_profile(profile)
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("TensorRT build returned None")
    meta = {
        "description": "YOLO11l FP16 TensorRT (prepare.py)",
        "stride": 32,
        "task": "detect",
        "batch": batch,
        "imgsz": [640, 640],
        "names": {int(k): v for k, v in model.names.items()},
        "half": True,
        "dynamic": True,
        "args": {"batch": batch, "half": True, "dynamic": True, "nms": False},
    }
    m = json.dumps(meta).encode()
    with open(engine, "wb") as f:
        f.write(len(m).to_bytes(4, byteorder="little", signed=True))
        f.write(m)
        f.write(ser)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild the engine even if one exists")
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
        if os.path.exists(engine) and not args.force:
            print("[prepare] existing engine found, checking it (use --force to rebuild)", flush=True)
        else:
            build_engine(pt, engine, args.batch)
        print(f"[prepare] engine built in {time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print(f"[prepare] own TensorRT build failed ({e}); trying Ultralytics export", file=sys.stderr, flush=True)
        try:
            out = YOLO(pt).export(format="engine", half=True, dynamic=True, batch=args.batch, imgsz=640, device=0, verbose=False)
            if os.path.abspath(out) != engine:
                shutil.move(out, engine)
            print(f"[prepare] engine built in {time.time() - t:.0f}s", flush=True)
        except Exception as e2:
            print(f"[prepare] TensorRT export failed ({e2}); run.py will use PyTorch FP16", file=sys.stderr)
            if os.path.exists(engine):
                os.remove(engine)
            return
    # sanity check: engine vs PyTorch on real images, through run.py's own code path
    try:
        sys.path.insert(0, HERE)
        import cv2
        import numpy as np
        from ultralytics.utils import ASSETS

        import run

        imgs = [cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")]
        a = run.Detector(pt, engine, "cuda:0", True, args.batch, "trt")
        b = run.Detector(pt, engine, "cuda:0", True, args.batch, "torch")
        worst = 0.0
        for im in imgs:
            x = run.letterbox(im, run.letterbox_params(*im.shape[:2]))
            xb = torch.from_numpy(np.repeat(x[None], 8, 0))
            pa, ca = a.infer(xb)
            pb, cb = b.infer(xb)
            print(f"[prepare] check {im.shape}: engine flags {pa[0].tolist()} people {int(ca[0])} | "
                  f"pytorch flags {pb[0].tolist()} people {int(cb[0])}", flush=True)
            if (pa != pb).any() or abs(int(ca[0]) - int(cb[0])) > 1:
                worst = 1.0
        if worst:
            raise RuntimeError("engine and PyTorch disagree on the check images")
        print(f"[prepare] engine OK ({a.backend_name}, input {'fp16' if a.in_half else 'fp32'})", flush=True)
    except Exception as e:
        print(f"[prepare] engine check failed ({e}); removing engine", file=sys.stderr)
        if os.path.exists(engine):
            os.remove(engine)

if __name__ == "__main__":
    main()
