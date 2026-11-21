"""Correctness tests (CPU is enough; a GPU is used if present).

  * letterbox: identical bytes to ultralytics.data.augment.LetterBox(auto=True)
  * detector:  run.Detector (custom batched preprocessing + presence/NMS reduction on the
               tensor) gives exactly the same presence flags and person count per frame as
               the stock API  YOLO('yolo11l.pt').predict(frame, imgsz=640, conf=0.4)
  * seconds:   per-second aggregation rules (>= half of frames, median rounded down)
  * damaged input produces error lines instead of crashing

    FINEVIDEO_SHARD=/path/to/train-000xx-of-01357.parquet pytest -q tests
"""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run  # noqa: E402

SHARD = os.environ.get("FINEVIDEO_SHARD", "")


def real_frames(n=24):
    """A few frames from different videos of a real shard (or synthetic noise if none)."""
    if SHARD and os.path.exists(SHARD):
        import av
        import pyarrow.parquet as pq

        col = pq.ParquetFile(SHARD).read_row_group(0, columns=["mp4"]).column("mp4")
        out = []
        for r in range(min(len(col), 8)):
            c = av.open(io.BytesIO(col[r].as_py()))
            for k, f in enumerate(c.decode(video=0)):
                if k % 97 == 0:
                    out.append(f.to_ndarray(format="bgr24"))
                if len(out) >= (r + 1) * n // 8:
                    break
            c.close()
        return out
    rng = np.random.default_rng(0)
    return [(rng.random((360, 640, 3)) * 255).astype(np.uint8) for _ in range(4)]


@pytest.mark.parametrize("hw", [(360, 640), (640, 360), (240, 320), (1080, 1920), (334, 640), (360, 202), (17, 999)])
def test_letterbox_matches_ultralytics(hw):
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(1)
    img = (rng.random((*hw, 3)) * 255).astype(np.uint8)
    ref = LetterBox((640, 640), auto=True, stride=32)(image=img)
    ours = run.letterbox(img, run.letterbox_params(*hw))
    assert ref.shape == ours.shape
    assert np.array_equal(ref, ours)


def test_detector_matches_stock_api():
    import torch
    from ultralytics import YOLO

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    frames = real_frames()
    stock = YOLO(os.path.join(ROOT, "yolo11l.pt"))
    det = run.Detector(os.path.join(ROOT, "yolo11l.pt"), "", dev, False, 16, "torch")  # FP32 like the stock API
    mism = 0
    for img in frames:
        r = stock.predict(img, imgsz=640, conf=0.4, verbose=False, device=dev)[0]
        cls = r.boxes.cls.int().tolist()
        ref_p = [c in cls for c in run.COCO_IDS]
        ref_n = sum(1 for c in cls if c == 0)
        x = run.letterbox(img, run.letterbox_params(*img.shape[:2]))
        p, n = det.infer(torch.from_numpy(x[None]))
        mism += (list(p[0]) != ref_p) or int(n[0]) != ref_n
    assert mism == 0, f"{mism}/{len(frames)} frames differ from the stock Ultralytics output"


def test_second_rules():
    acc = run.VideoAcc()
    T, F = True, False
    acc.pres[0] = [np.array([T, F, F, F, F])] * 2 + [np.array([F, F, F, F, F])] * 2  # exactly half -> present
    acc.people[0] = [1, 1, 2, 2]  # median 1.5 -> 1
    acc.pres[1] = [np.array([T, T, F, F, F])] + [np.array([F, F, F, F, F])] * 2  # 1/3 -> absent
    acc.people[1] = [3, 0, 0]
    acc.end = {"last_second": 2, "error": None}  # second 2 has no frames -> empty line
    lines = run.video_lines("x.parquet:0", acc)
    assert lines[0] == {"video": "x.parquet:0", "second": 0, "objects": ["person"], "people": 1}
    assert lines[1] == {"video": "x.parquet:0", "second": 1, "objects": [], "people": 0}
    assert lines[2] == {"video": "x.parquet:0", "second": 2, "objects": [], "people": 0}
    assert run.median_floor([0, 3]) == 1 and run.median_floor([2, 2, 5]) == 2


def test_damaged_inputs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    good = os.path.join(ROOT, "tests", "data", "tiny.mp4")
    if not os.path.exists(good):
        pytest.skip("tests/data/tiny.mp4 missing")
    b = open(good, "rb").read()
    rows = [b, b[: len(b) // 3], b"not a video", None, b[:200] + bytes(1000) + b[1200:]]
    pq.write_table(pa.table({"mp4": pa.array(rows, pa.binary()), "json": pa.array(["{}"] * len(rows))}),
                   tmp_path / "train-x.parquet", row_group_size=2)
    (tmp_path / "broken.parquet").write_bytes(b"PAR1garbage")
    out = tmp_path / "r.jsonl"
    cmd = [sys.executable, os.path.join(ROOT, "run.py"), "--input", str(tmp_path), "--output", str(out),
           "--workers", "2", "--batch", "8", "--samples-per-second", "1", "--refine", "0"]
    subprocess.run(cmd, check=True, timeout=900)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    vids = {d["video"] for d in lines}
    assert {"broken.parquet:0", "train-x.parquet:0", "train-x.parquet:1", "train-x.parquet:2",
            "train-x.parquet:3", "train-x.parquet:4"} <= vids
    assert any(d["video"] == "train-x.parquet:0" and "second" in d for d in lines)
    for v in ("broken.parquet:0", "train-x.parquet:2", "train-x.parquet:3"):
        assert any(d["video"] == v and "error" in d for d in lines)
