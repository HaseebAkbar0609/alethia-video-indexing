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


class UniformSampler:
    """Pass-1 selection: in every second pick the first frame at/after sec + j/K (j=0..K-1).
    The first frame of every second is always picked, so no second is left without a sample."""

    def __init__(self, k):
        self.k = k
        self.cur = -1
        self.j = 0
        self.picked = []  # decode indices of picked frames

    def __call__(self, idx, rel, sec):
        if sec > self.cur:
            self.cur, self.j = sec, 0
        elif sec < self.cur:  # timestamps went backwards: never sample, pass 2 can still use it
            return False
        k = self.k
        if self.j < k and rel + 1e-9 >= sec + self.j / k:
            while self.j < k and rel + 1e-9 >= sec + self.j / k:
                self.j += 1
            self.picked.append(idx)
            return True
        return False


class RefineSelector:
    """Pass-2 selection: frames of the 'uncertain' seconds that pass 1 did not already use.
    max_extra = 0 -> every frame of those seconds; otherwise at most max_extra extra frames
    per second, spread evenly (by timestamp)."""

    def __init__(self, seconds, skip, max_extra=0):
        self.seconds = set(seconds)
        self.skip = set(skip)
        self.max_extra = max_extra
        self.taken = {}
        self.cur = -1
        self.j = 0

    def __call__(self, idx, rel, sec):
        if sec not in self.seconds or idx in self.skip:
            return False
        if not self.max_extra:
            return True
        if sec != self.cur:
            self.cur, self.j = sec, 0
        k = self.max_extra
        # extra targets placed between pass-1 targets: sec + (j + 0.5)/k
        if self.j < k and rel + 1e-9 >= sec + (self.j + 0.5) / k:
            while self.j < k and rel + 1e-9 >= sec + (self.j + 0.5) / k:
                self.j += 1
            return True
        return False


def decode_pass(mp4: bytes, select, emit, max_chunk: int):
    """Decode a whole video once; letterbox and emit the frames select() accepts.

    Times are relative to the first decoded frame (pts * time_base). Returns a dict:
        {"last_second": int|None, "n_frames": int, "error": str|None}
    emit("frames", (frames_uint8[N,H,W,3], seconds[N])) is called in chunks.
    """
    import io

    import av

    info = {"last_second": None, "n_frames": 0, "error": None}
    try:
        container = av.open(io.BytesIO(mp4), mode="r")
    except Exception as e:
        info["error"] = f"could not open video: {type(e).__name__}"
        return info
    buf, secs = [], []

    def flush():
        nonlocal buf, secs
        if buf:
            emit("frames", (np.stack(buf), np.asarray(secs, dtype=np.int32)))
            buf, secs = [], []

    try:
        if not container.streams.video:
            info["error"] = "no video stream"
            return info
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = WORKER_DECODE_THREADS
        tb = stream.time_base
        t0 = None
        last_t = None
        params = None
        shape_key = None
        rot_k = 0
        prev_t = None
        idx = 0
        fallback_fps = float(stream.average_rate or stream.guessed_rate or 30.0) or 30.0
        try:
            for frame in container.decode(stream):
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
                rel = max(t - t0, 0.0)
                sec = int(math.floor(rel + 1e-9))
                if last_t is None or rel > last_t:
                    last_t = rel
                    info["last_second"] = sec
                if select(idx, rel, sec):
                    img = frame.to_ndarray(format="bgr24")
                    if rot_k:
                        img = np.ascontiguousarray(np.rot90(img, rot_k))
                    if shape_key != img.shape[:2]:
                        flush()
                        shape_key = img.shape[:2]
                        params = letterbox_params(*shape_key)
                    buf.append(letterbox(img, params))
                    secs.append(sec)
                    if len(buf) >= max_chunk:
                        flush()
                idx += 1
                info["n_frames"] = idx
        except Exception as e:  # damaged stream part-way
            info["error"] = f"decode failed after {idx} frames: {type(e).__name__}"
        flush()
        if info["n_frames"] == 0 and not info["error"]:
            info["error"] = "could not decode video"
        return info
    finally:
        container.close()


WORKER_DECODE_THREADS = 1


def worker_main(wid, task_q, out_q, reply_q, cfg, shutdown=None):
    """Decode worker. For every video: pass 1 (uniform samples) -> wait for the main
    process to say which seconds are uncertain -> pass 2 (re-decode, extra frames of those
    seconds only) -> final message."""
    global WORKER_DECODE_THREADS
    WORKER_DECODE_THREADS = cfg["decode_threads"]
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    cache = RowGroupCache()
    max_chunk = cfg["chunk"]
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
                out_q.put((_vid, "frames", torch.from_numpy(frames), torch.from_numpy(secs), wid))
            else:
                out_q.put((_vid, kind, payload, None, wid))

        try:
            mp4 = cache.get(path, rg, i)
        except Exception as e:
            emit("final", {"last_second": None, "error": f"could not read row: {type(e).__name__}"})
            continue
        if not mp4:
            emit("final", {"last_second": None, "error": "empty mp4 field"})
            continue
        try:
            sampler = UniformSampler(cfg["sps"])
            info = decode_pass(mp4, sampler, emit, max_chunk)
            if info["error"] or not cfg["refine"]:
                emit("final", info)
                continue
            emit("pass1", info)
            while True:  # wait for the decision about this video
                r_vid, seconds = reply_q.get()
                if r_vid == vid:
                    break
            if seconds:
                info2 = decode_pass(mp4, RefineSelector(seconds, sampler.picked, cfg["refine_max"]), emit, max_chunk)
                if info2["error"] and not info["error"]:
                    info["error"] = None  # pass 1 succeeded; a pass-2 failure only loses extra frames
            emit("final", info)
        except Exception as e:
            emit("final", {"last_second": None, "error": f"could not decode video: {type(e).__name__}"})
        finally:
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
    __slots__ = ("pres", "people", "end", "pass1", "decided", "wid", "outstanding", "refined")

    def __init__(self):
        self.pres = defaultdict(list)  # sec -> list of presence rows (bool[5])
        self.people = defaultdict(list)  # sec -> list of person counts
        self.end = None  # final info dict
        self.pass1 = None  # pass-1 info dict (set when the worker waits for a decision)
        self.decided = False
        self.wid = None
        self.outstanding = 0  # frames queued but not yet inferred
        self.refined = 0


def median_floor(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return int(s[n // 2])
    return int((s[n // 2 - 1] + s[n // 2]) // 2)


def uncertain_seconds(acc: "VideoAcc", ppl_tol: int):
    """Seconds whose pass-1 samples do not agree. A second is certain only if every sampled
    frame agrees on all five presence flags and the person counts differ by <= ppl_tol."""
    out = []
    for s, rows in acc.pres.items():
        if len(rows) < 2:
            if len(rows) == 1 and acc.people[s][0] > 0 and ppl_tol < 0:
                out.append(s)
            continue
        arr = np.asarray(rows)
        cnt = arr.sum(0)
        if ((cnt > 0) & (cnt < len(rows))).any():
            out.append(s)
            continue
        ppl = acc.people[s]
        if max(ppl) - min(ppl) > ppl_tol:
            out.append(s)
    return sorted(out)


def video_lines(vid, acc: VideoAcc):
    info = acc.end
    last = info.get("last_second")
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
    if info.get("error"):
        lines.append({"video": vid, "error": info["error"]})
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--samples-per-second", type=int, default=int(os.environ.get("VI_SPS", 3)),
                    help="pass-1 frames per second of video")
    ap.add_argument("--refine", type=int, default=int(os.environ.get("VI_REFINE", 1)),
                    help="1: re-run uncertain seconds on more frames (pass 2), 0: pass 1 only")
    ap.add_argument("--refine-max", type=int, default=int(os.environ.get("VI_REFINE_MAX", 0)),
                    help="extra frames per uncertain second (0 = all its frames)")
    ap.add_argument("--ppl-tol", type=int, default=int(os.environ.get("VI_PPL_TOL", 0)),
                    help="a second is uncertain if its sampled person counts differ by more than this")
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
    # workers block while the GPU decides about their video, so use a few more than cores
    workers = args.workers or max(2, min(ncpu + 2, 40))
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
    reply_qs = [ctx.Queue() for _ in range(workers)]
    cfg = {
        "sps": args.samples_per_second,
        "refine": bool(args.refine),
        "refine_max": args.refine_max,
        "decode_threads": args.decode_threads,
        "chunk": 32,
    }
    vids = []
    for path, rg, i, row in tasks:
        vid = f"{os.path.basename(path)}:{row}"
        vids.append(vid)
        task_q.put((vid, path, rg, i))
    for _ in range(workers):
        task_q.put(None)

    def spawn(wid):
        p = ctx.Process(target=worker_main, args=(wid, task_q, out_q, reply_qs[wid], cfg, shutdown), daemon=True)
        p.start()
        return p

    procs = [spawn(w) for w in range(workers)]
    n_respawn = 0

    det = Detector(weights, args.engine, args.device, not args.fp32, args.batch, args.backend)
    if det.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        det.infer(torch.zeros((min(args.batch, 8), 384, 640, 3), dtype=torch.uint8))  # warm-up
    t_model = time.time()

    out = open(args.output, "w", buffering=1 << 20)
    for e in early_errors:
        out.write(json.dumps(e) + "\n")

    accs: dict[str, VideoAcc] = {}
    written: set[str] = set()
    buffers: dict[tuple, list] = defaultdict(list)  # (H,W) -> list of (vid, frames, secs)
    buf_count: dict[tuple, int] = defaultdict(int)
    finished_workers = 0
    st = defaultdict(float)
    last_report = time.time()

    def finalize(vid):
        acc = accs.pop(vid)
        written.add(vid)
        for ln in video_lines(vid, acc):
            out.write(json.dumps(ln) + "\n")
        st["videos_done"] += 1
        st["seconds"] += len(acc.pres) if acc.end.get("last_second") is not None else 0
        st["refined_seconds"] += acc.refined
        if acc.end.get("error"):
            st["errors"] += 1

    def decide(vid):
        acc = accs[vid]
        secs = uncertain_seconds(acc, args.ppl_tol)
        acc.refined = len(secs)
        acc.decided = True
        reply_qs[acc.wid].put((vid, secs))

    def check(vid):
        acc = accs.get(vid)
        if acc is None or acc.outstanding:
            return
        if acc.end is not None:
            finalize(vid)
        elif acc.pass1 is not None and not acc.decided:
            decide(vid)

    def run_buffer(key):
        items = buffers.pop(key)
        buf_count.pop(key, None)
        frames = torch.cat([it[1] for it in items]) if len(items) > 1 else items[0][1]
        t = time.time()
        allp, allc = [], []
        for s0 in range(0, frames.shape[0], args.batch):
            chunk = frames[s0 : s0 + args.batch]
            if det.device.type == "cuda":
                chunk = chunk.pin_memory()
            pres, ppl = det.infer(chunk)
            allp.append(pres)
            allc.append(ppl)
        st["gpu_s"] += time.time() - t
        pres = np.concatenate(allp)
        ppl = np.concatenate(allc)
        st["frames"] += frames.shape[0]
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
        for vid in dict.fromkeys(touched):
            check(vid)

    def flush_video(vid):
        for key in [k for k, its in buffers.items() if any(it[0] == vid for it in its)]:
            if key in buffers:
                run_buffer(key)

    while finished_workers < workers:
        for k, p in enumerate(procs):  # replace workers that crashed (e.g. decoder segfault)
            if p.exitcode not in (None, 0) and n_respawn < 1000:
                procs[k] = spawn(k)
                n_respawn += 1
                print(f"[run.py] worker {k} died (exit {p.exitcode}), respawned", file=sys.stderr, flush=True)
                for vid, acc in list(accs.items()):  # its in-flight video is lost: close it
                    if acc.wid == k and acc.end is None:
                        acc.end = {"last_second": acc.pass1["last_second"] if acc.pass1 else (max(acc.pres) if acc.pres else None),
                                   "error": "worker crashed while decoding"}
                        flush_video(vid)
                        check(vid)
        tw = time.time()
        try:
            msg = out_q.get(timeout=0.5)
        except queue_mod.Empty:
            st["wait_s"] += time.time() - tw
            if buffers:  # nothing arriving: run the biggest partial batch so nobody waits
                run_buffer(max(buf_count, key=buf_count.get))
            if all(p.exitcode is not None for p in procs) and out_q.empty():
                break
            continue
        except Exception as e:  # message from a worker that crashed mid-send
            print(f"[run.py] dropped a message: {e}", file=sys.stderr, flush=True)
            continue
        st["wait_s"] += time.time() - tw
        if msg is None:
            finished_workers += 1
            continue
        vid, kind, a, b, wid = msg
        acc = accs.get(vid)
        if acc is None:
            acc = accs[vid] = VideoAcc()
        acc.wid = wid
        if kind == "frames":
            key = tuple(a.shape[1:3])
            buffers[key].append((vid, a, b))
            buf_count[key] += a.shape[0]
            acc.outstanding += a.shape[0]
            if acc.pass1 is not None:
                st["frames_pass2"] += a.shape[0]
            if buf_count[key] >= args.batch:
                run_buffer(key)
        elif kind == "pass1":
            acc.pass1 = a
            if acc.outstanding:
                flush_video(vid)  # the worker is waiting: decide as soon as possible
            check(vid)
        else:  # final
            acc.end = a
            if acc.outstanding:
                flush_video(vid)
            check(vid)
        if time.time() - last_report > 30:
            last_report = time.time()
            el = time.time() - t_start
            print(
                f"[run.py] {int(st['videos_done'])}/{n_videos} videos, {int(st['frames'])} frames "
                f"({int(st['frames_pass2'])} in pass 2), {st['frames'] / max(el, 1e-6):.0f} fr/s, "
                f"gpu busy {st['gpu_s'] / max(el, 1e-6):.0%}, {el:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    for key in list(buffers):
        if key in buffers:
            run_buffer(key)
    for vid in list(accs):  # anything left open (worker died without reporting)
        acc = accs[vid]
        if acc.end is None:
            acc.end = {"last_second": max(acc.pres) if acc.pres else None, "error": "worker crashed"}
        acc.outstanding = 0
        finalize(vid)
    for vid in vids:
        if vid not in written:
            out.write(json.dumps({"video": vid, "error": "video was not processed (worker crashed)"}) + "\n")
            st["errors"] += 1
    shutdown.set()
    for p in procs:
        p.join(timeout=5)
    out.close()
    el = time.time() - t_start
    stats = {
        "videos": n_videos,
        "errors": int(st["errors"]) + len(early_errors),
        "seconds_with_frames": int(st["seconds"]),
        "refined_seconds": int(st["refined_seconds"]),
        "frames_inferred": int(st["frames"]),
        "frames_pass2": int(st["frames_pass2"]),
        "wall_s": round(el, 1),
        "startup_s": round(t_model - t_start, 1),
        "gpu_infer_s": round(st["gpu_s"], 1),
        "main_wait_s": round(st["wait_s"], 1),
        "backend": det.backend_name,
        "workers": workers,
        "worker_respawns": n_respawn,
        "samples_per_second": args.samples_per_second,
        "refine": args.refine,
        "refine_max": args.refine_max,
        "ppl_tol": args.ppl_tol,
        "batch": args.batch,
    }
    print("[run.py] done " + json.dumps(stats), file=sys.stderr, flush=True)
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
