#!/usr/bin/env python3
"""Per-second object indexing of FineVideo parquet shards with YOLO11l.

Usage:
    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline (see README.md for details):
  * CPU worker processes read one video at a time from the parquet files, decode it with
    PyAV, pick a few frames per second, letterbox them exactly like Ultralytics does and
    send them to the main process through shared memory.
  * The main process batches frames of the same shape, runs YOLO11l on the GPU in FP16
    (TensorRT engine if prepared, otherwise PyTorch), and reduces each frame on the GPU to
    5 presence flags + a person count (after class-aware NMS, IoU 0.7, conf >= 0.4).
  * Per second: object present if seen in >= half of the sampled frames; people = median
    (rounded down at .5) of the sampled per-frame counts. Lines are written as soon as a
    video is finished.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue as queue_mod
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

CLASS_NAMES = ["person", "car", "dog", "laptop", "cell phone"]
COCO_IDS = [0, 2, 16, 63, 67]  # yolo11l (COCO) class ids for CLASS_NAMES
CONF = 0.4
IOU = 0.7
IMGSZ = 640
STRIDE = 32
PAD_VALUE = 114

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# Letterbox (identical arithmetic to ultralytics.data.augment.LetterBox with auto=True,
# which is what YOLO('yolo11l.pt').predict(frame, imgsz=640) uses for a single image)
# --------------------------------------------------------------------------------------
def letterbox_params(h: int, w: int, imgsz: int = IMGSZ, stride: int = STRIDE):
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = imgsz - new_w, imgsz - new_h
    dw, dh = np.mod(dw, stride) / 2, np.mod(dh, stride) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return new_w, new_h, top, bottom, left, right


def letterbox(img, params):
    import cv2

    new_w, new_h, top, bottom, left, right = params
    if img.shape[1] != new_w or img.shape[0] != new_h:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)


# --------------------------------------------------------------------------------------
# Worker side: read parquet rows, decode, sample, letterbox
# --------------------------------------------------------------------------------------
def list_tasks(input_dir: str):
    """Return (tasks, early_errors). A task is (path, row_group, row_in_group, global_row)."""
    import pyarrow.parquet as pq

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
    tasks, errors = [], []
    for name in files:
        path = os.path.join(input_dir, name)
        try:
            pf = pq.ParquetFile(path)
            md = pf.metadata
            row = 0
            for rg in range(md.num_row_groups):
                n = md.row_group(rg).num_rows
                for i in range(n):
                    tasks.append((path, rg, i, row))
                    row += 1
        except Exception as e:  # unreadable footer: we cannot know how many videos it holds
            errors.append({"video": f"{name}:0", "error": f"could not read parquet file: {type(e).__name__}"})
    return tasks, errors


class RowGroupCache:
    """Each worker keeps the last row group it read (mp4 column only)."""

    def __init__(self):
        self.key = None
        self.col = None
        self.pf = {}

    def get(self, path, rg, i):
        import pyarrow.parquet as pq

        if self.key != (path, rg):
            self.col = None
            if path not in self.pf:
                self.pf = {path: pq.ParquetFile(path)}
            tbl = self.pf[path].read_row_group(rg, columns=["mp4"])
            self.col = tbl.column("mp4")
            self.key = (path, rg)
        v = self.col[i].as_py()
        return v


def sample_and_decode(mp4: bytes, fps_samples: int, emit, max_chunk: int):
    """Decode a video, emit sampled letterboxed frames in chunks.

    emit(kind, payload) is called with:
        ("frames", (frames_uint8[N,H,W,3], seconds[N]))
        ("done", {"last_second": int})            -- normal end
        ("error", {"last_second": int|None, "msg": str})
    """
    import io

    import av

    container = av.open(io.BytesIO(mp4), mode="r", options={"err_detect": "ignore_err"})
    try:
        if not container.streams.video:
            raise RuntimeError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = WORKER_DECODE_THREADS
        tb = stream.time_base
        t0 = None
        last_t = None
        cur_sec = -1
        next_j = 0
        params = None
        shape_key = None
        rot_k = 0
        buf, secs = [], []
        prev_t = None
        n_frames = 0
        fallback_fps = float(stream.average_rate or stream.guessed_rate or 30.0) or 30.0
        try:
            for frame in container.decode(stream):
                # timestamp relative to the first decoded frame
                if frame.pts is not None:
                    t = float(frame.pts * tb)
                elif prev_t is not None:
                    t = prev_t + 1.0 / fallback_fps
                else:
                    t = 0.0
                prev_t = t
                if t0 is None:
                    t0 = t
                    try:
                        rot = frame.rotation or 0
                    except Exception:
                        rot = 0
                    rot_k = int(round(rot / 90.0)) % 4
                rel = t - t0
                if rel < 0:  # out-of-order / pre-roll frame before the first decoded one
                    rel = 0.0
                n_frames += 1
                sec = int(math.floor(rel + 1e-9))
                if last_t is None or rel > last_t:
                    last_t = rel
                if sec != cur_sec:
                    if sec < cur_sec:  # non-monotonic timestamps: just treat as current second
                        sec = cur_sec
                    else:
                        cur_sec = sec
                        next_j = 0
                # targets at sec + j/K, j = 0..K-1 ; first frame of each second always sampled
                if next_j < fps_samples and rel + 1e-9 >= sec + next_j / fps_samples:
                    while next_j < fps_samples and rel + 1e-9 >= sec + next_j / fps_samples:
                        next_j += 1
                    img = frame.to_ndarray(format="bgr24")
                    if rot_k:
                        img = np.ascontiguousarray(np.rot90(img, rot_k))
                    if shape_key != img.shape[:2]:
                        if buf:
                            emit("frames", (np.stack(buf), np.asarray(secs, dtype=np.int32)))
                            buf, secs = [], []
                        shape_key = img.shape[:2]
                        params = letterbox_params(*shape_key)
                    buf.append(letterbox(img, params))
                    secs.append(sec)
                    if len(buf) >= max_chunk:
                        emit("frames", (np.stack(buf), np.asarray(secs, dtype=np.int32)))
                        buf, secs = [], []
        except Exception as e:  # damaged stream part-way
            if buf:
                emit("frames", (np.stack(buf), np.asarray(secs, dtype=np.int32)))
            last = int(math.floor(last_t + 1e-9)) if last_t is not None else None
            emit("error", {"last_second": last, "msg": f"decode failed after {n_frames} frames: {type(e).__name__}"})
            return
        if buf:
            emit("frames", (np.stack(buf), np.asarray(secs, dtype=np.int32)))
        if n_frames == 0:
            emit("error", {"last_second": None, "msg": "could not decode video"})
            return
        emit("done", {"last_second": int(math.floor(last_t + 1e-9))})
    finally:
        container.close()


WORKER_DECODE_THREADS = 1


def worker_main(task_q, out_q, fps_samples, max_chunk, decode_threads, shutdown=None):
    global WORKER_DECODE_THREADS
    WORKER_DECODE_THREADS = decode_threads
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    cache = RowGroupCache()
    while True:
        task = task_q.get()
        if task is None:
            out_q.put(None)
            if shutdown is not None:
                shutdown.wait(timeout=24 * 3600)
            return
        vid, path, rg, i = task

        def emit(kind, payload, _vid=vid):
            if kind == "frames":
                frames, secs = payload
                out_q.put((_vid, "frames", torch.from_numpy(frames), torch.from_numpy(secs)))
            else:
                out_q.put((_vid, kind, payload, None))

        try:
            mp4 = cache.get(path, rg, i)
        except Exception as e:
            emit("error", {"last_second": None, "msg": f"could not read row: {type(e).__name__}"})
            continue
        if not mp4:
            emit("error", {"last_second": None, "msg": "empty mp4 field"})
            continue
        try:
            sample_and_decode(mp4, fps_samples, emit, max_chunk)
        except Exception as e:
            emit("error", {"last_second": None, "msg": f"could not decode video: {type(e).__name__}"})
        del mp4


# --------------------------------------------------------------------------------------
# GPU side
# --------------------------------------------------------------------------------------
class Detector:
    def __init__(self, weights, engine, device, half, batch, backend):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        self.backend_name = None
        from ultralytics.nn.autobackend import AutoBackend

        use_engine = backend in ("auto", "trt") and engine and os.path.exists(engine) and self.device.type == "cuda"
        if backend == "trt" and not use_engine:
            raise SystemExit(f"TensorRT engine not found: {engine}")
        model = None
        if use_engine:
            try:
                model = AutoBackend(engine, device=self.device, fp16=True, verbose=False)
                self.backend_name = "tensorrt-fp16"
                self.half = True
            except Exception as e:
                print(f"[run.py] TensorRT engine failed to load ({e}); falling back to PyTorch", file=sys.stderr)
                model = None
        if model is None:
            model = AutoBackend(weights, device=self.device, fp16=self.half, fuse=True, verbose=False)
            self.backend_name = "pytorch-fp16" if self.half else "pytorch-fp32"
        model.eval()
        self.model = model
        self.class_idx = torch.tensor(COCO_IDS, device=self.device)
        self.max_batch = batch
        self.engine_max_hw = None
        if use_engine and self.backend_name.startswith("tensorrt"):
            self.engine_max_hw = (IMGSZ, IMGSZ)

    def infer(self, frames_u8):
        """frames_u8: uint8 tensor [N,H,W,3] BGR (CPU, ideally pinned). Returns (presence[N,5] bool, people[N])."""
        torch = self.torch
        from torchvision.ops import batched_nms

        x = frames_u8.to(self.device, non_blocking=True)
        x = x.flip(-1).permute(0, 3, 1, 2)  # BGR->RGB, NHWC->NCHW (as ultralytics preprocess)
        x = x.half() if self.half else x.float()
        x = x.div_(255.0).contiguous()
        with torch.inference_mode():
            y = self.model(x)
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.float()  # [N, 84, A]
            scores = y[:, 4:, :]  # [N, 80, A]
            best, cls = scores.max(1)  # [N, A] -- ultralytics NMS uses the best class per box
            keep = best >= CONF  # candidates
            # presence: a class is present iff some candidate has it as best class (NMS keeps the top box)
            pres = torch.stack([(keep & (cls == c)).any(1) for c in COCO_IDS], 1)
            # people: class-aware NMS among person candidates, counted per image
            pmask = keep & (cls == 0)
            idx_img, idx_anchor = pmask.nonzero(as_tuple=True)
            people = torch.zeros(y.shape[0], dtype=torch.int64, device=y.device)
            if idx_img.numel():
                b = y[idx_img, :4, idx_anchor]  # xywh
                xyxy = torch.empty_like(b)
                xyxy[:, 0] = b[:, 0] - b[:, 2] / 2
                xyxy[:, 1] = b[:, 1] - b[:, 3] / 2
                xyxy[:, 2] = b[:, 0] + b[:, 2] / 2
                xyxy[:, 3] = b[:, 1] + b[:, 3] / 2
                sc = best[idx_img, idx_anchor]
                k = batched_nms(xyxy, sc, idx_img, IOU)
                people.index_add_(0, idx_img[k], torch.ones_like(k))
                people.clamp_(max=300)  # ultralytics max_det
        return pres.cpu().numpy(), people.cpu().numpy()


class VideoAcc:
    __slots__ = ("pres", "people", "end", "outstanding")

    def __init__(self):
        self.pres = defaultdict(list)  # sec -> list of presence rows
        self.people = defaultdict(list)
        self.end = None  # ("done"|"error", payload)
        self.outstanding = 0  # frames queued but not yet inferred


def median_floor(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return int(s[n // 2])
    return int((s[n // 2 - 1] + s[n // 2]) // 2)


def video_lines(vid, acc: VideoAcc):
    kind, payload = acc.end
    last = payload.get("last_second")
    lines = []
    if last is not None:
        for s in range(last + 1):
            pr = acc.pres.get(s)
            if pr:
                arr = np.asarray(pr)
                cnt = arr.sum(0)
                n = arr.shape[0]
                objs = [CLASS_NAMES[c] for c in range(5) if 2 * cnt[c] >= n]
                ppl = median_floor(acc.people[s])
            else:
                objs, ppl = [], 0
            lines.append({"video": vid, "second": s, "objects": objs, "people": ppl})
    if kind == "error":
        lines.append({"video": vid, "error": payload.get("msg", "error")})
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--samples-per-second", type=int, default=int(os.environ.get("VI_SPS", 6)))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("VI_BATCH", 64)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("VI_WORKERS", 0)), help="0 = auto")
    ap.add_argument("--decode-threads", type=int, default=int(os.environ.get("VI_DECODE_THREADS", 1)))
    ap.add_argument("--backend", choices=["auto", "trt", "torch"], default=os.environ.get("VI_BACKEND", "auto"))
    ap.add_argument("--weights", default=os.path.join(HERE, "yolo11l.pt"))
    ap.add_argument("--engine", default=os.path.join(HERE, "yolo11l.engine"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp32", action="store_true", help="disable FP16 (PyTorch backend only)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N videos")
    ap.add_argument("--stats", default="", help="optional path to write timing stats JSON")
    args = ap.parse_args()

    t_start = time.time()
    import torch
    import torch.multiprocessing as mp

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[run.py] CUDA not available, running on CPU (slow)", file=sys.stderr)
        args.device = "cpu"

    weights = args.weights if os.path.exists(args.weights) else "yolo11l.pt"
    tasks, early_errors = list_tasks(args.input)
    if args.limit:
        tasks = tasks[: args.limit]
    n_videos = len(tasks)
    ncpu = os.cpu_count() or 4
    workers = args.workers or max(1, min(ncpu - 1, 30))
    workers = max(1, min(workers, max(1, n_videos)))

    try:  # received shared-memory tensors each hold a file descriptor for a short while
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(hard, 65536)), hard))
    except Exception:
        pass
    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()  # workers stay alive until the end so their shared tensors stay readable
    task_q = ctx.Queue()
    out_q = ctx.Queue(maxsize=workers * 2 + 8)
    vids = {}
    for path, rg, i, row in tasks:
        vid = f"{os.path.basename(path)}:{row}"
        vids[vid] = (path, rg, i)
        task_q.put((vid, path, rg, i))
    for _ in range(workers):
        task_q.put(None)
    def spawn():
        p = ctx.Process(
            target=worker_main,
            args=(task_q, out_q, args.samples_per_second, 32, args.decode_threads, shutdown),
            daemon=True,
        )
        p.start()
        return p

    procs = [spawn() for _ in range(workers)]
    n_respawn = 0

    det = Detector(weights, args.engine, args.device, not args.fp32, args.batch, args.backend)
    # warm-up (also triggers cuDNN autotune for the common 384x640 shape)
    if det.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        for hw in ((384, 640),):
            det.infer(torch.zeros((min(args.batch, 8), hw[0], hw[1], 3), dtype=torch.uint8))
    t_model = time.time()

    out = open(args.output, "w", buffering=1 << 20)
    for e in early_errors:
        out.write(json.dumps(e) + "\n")

    accs: dict[str, VideoAcc] = {}
    written: set[str] = set()
    buffers: dict[tuple, list] = defaultdict(list)  # (H,W) -> list of (vid, frames, secs)
    buf_count: dict[tuple, int] = defaultdict(int)
    finished_workers = 0
    n_done = 0
    n_err = 0
    n_frames = 0
    t_gpu = 0.0
    t_wait = 0.0
    last_report = time.time()

    def finalize(vid):
        nonlocal n_done, n_err
        acc = accs.pop(vid)
        written.add(vid)
        for ln in video_lines(vid, acc):
            out.write(json.dumps(ln) + "\n")
        n_done += 1
        if acc.end[0] == "error":
            n_err += 1

    def run_buffer(key):
        nonlocal n_frames, t_gpu
        items = buffers.pop(key)
        buf_count.pop(key, None)
        frames = torch.cat([it[1] for it in items]) if len(items) > 1 else items[0][1]
        t = time.time()
        for s0 in range(0, frames.shape[0], args.batch):
            chunk = frames[s0 : s0 + args.batch]
            if det.device.type == "cuda":
                chunk = chunk.pin_memory()
            pres, ppl = det.infer(chunk)
            if s0 == 0:
                allp, allc = [pres], [ppl]
            else:
                allp.append(pres)
                allc.append(ppl)
        t_gpu += time.time() - t
        pres = np.concatenate(allp)
        ppl = np.concatenate(allc)
        n_frames += frames.shape[0]
        off = 0
        touched = []
        for vid, fr, secs in items:
            n = fr.shape[0]
            acc = accs[vid]
            for j, s in enumerate(secs.tolist()):
                acc.pres[s].append(pres[off + j])
                acc.people[s].append(int(ppl[off + j]))
            acc.outstanding -= n
            off += n
            touched.append(vid)
        for vid in touched:
            acc = accs.get(vid)
            if acc is not None and acc.end is not None and acc.outstanding == 0:
                finalize(vid)

    while finished_workers < workers:
        # replace workers that crashed (e.g. a decoder segfault on a damaged video)
        for k, p in enumerate(procs):
            if p.exitcode not in (None, 0) and n_respawn < 1000:
                procs[k] = spawn()
                n_respawn += 1
                print(f"[run.py] worker died (exit {p.exitcode}), respawned", file=sys.stderr, flush=True)
        tw = time.time()
        try:
            msg = out_q.get(timeout=1.0)
        except (EOFError, OSError, RuntimeError) as e:  # message from a worker that crashed mid-send
            print(f"[run.py] dropped a message: {e}", file=sys.stderr, flush=True)
            continue
        except queue_mod.Empty:
            t_wait += time.time() - tw
            # nothing arriving: flush the biggest partial buffer so the GPU is not idle
            if buffers:
                run_buffer(max(buf_count, key=buf_count.get))
            if all(p.exitcode is not None for p in procs) and out_q.empty():
                break
            continue
        t_wait += time.time() - tw
        if msg is None:
            finished_workers += 1
            continue
        vid, kind, a, b = msg
        acc = accs.get(vid)
        if acc is None:
            acc = accs[vid] = VideoAcc()
        if kind == "frames":
            frames, secs = a, b
            key = tuple(frames.shape[1:3])
            buffers[key].append((vid, frames, secs))
            buf_count[key] += frames.shape[0]
            acc.outstanding += frames.shape[0]
            if buf_count[key] >= args.batch:
                run_buffer(key)
        else:
            acc.end = (kind, a)
            if acc.outstanding == 0:
                finalize(vid)
            else:
                # flush buffers holding this video's frames so its lines can be written now
                for key in [k for k, its in buffers.items() if any(it[0] == vid for it in its)]:
                    if key in buffers:
                        run_buffer(key)
        if time.time() - last_report > 30:
            last_report = time.time()
            el = time.time() - t_start
            print(
                f"[run.py] {n_done}/{n_videos} videos, {n_frames} frames inferred, "
                f"{n_frames / max(el, 1e-6):.0f} fr/s overall, gpu busy {t_gpu / max(el, 1e-6):.0%}, {el:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    for key in list(buffers):
        if key in buffers:
            run_buffer(key)
    # any video whose worker died without reporting
    for vid in list(accs):
        acc = accs[vid]
        if acc.end is None:
            acc.end = ("error", {"last_second": max(acc.pres) if acc.pres else None, "msg": "worker crashed"})
        finalize(vid)
    # videos that never produced any message (worker died before reporting anything)
    for vid in vids:
        if vid not in written:
            out.write(json.dumps({"video": vid, "error": "video was not processed (worker crashed)"}) + "\n")
            n_err += 1
    seen = n_done
    shutdown.set()
    for p in procs:
        p.join(timeout=5)
    out.close()
    el = time.time() - t_start
    stats = {
        "videos": n_videos,
        "videos_written": seen,
        "errors": n_err + len(early_errors),
        "frames_inferred": n_frames,
        "wall_s": round(el, 1),
        "model_load_s": round(t_model - t_start, 1),
        "gpu_infer_s": round(t_gpu, 1),
        "main_wait_s": round(t_wait, 1),
        "backend": det.backend_name,
        "workers": workers,
        "worker_respawns": n_respawn,
        "samples_per_second": args.samples_per_second,
        "batch": args.batch,
    }
    print("[run.py] done " + json.dumps(stats), file=sys.stderr, flush=True)
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
