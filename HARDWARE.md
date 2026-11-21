# Hardware used for the full run (50 shards, 125+ hours)

**Where:** Kaggle notebook (interactive session), `kaggle/job.sh` in `full` mode, on 21 Nov 2026.

| | |
|---|---|
| GPU | 2 × NVIDIA Tesla T4, 16 GB each (15,360 MiB), driver 580.178.04 |
| CPU | Intel Xeon @ 2.00 GHz, 4 vCPUs (2 cores × 2 threads) |
| RAM | 31 GB |
| Disk | Kaggle persistent disk (SSD, non-rotational); shards were downloaded to `/tmp` and deleted after use |
| OS | Ubuntu 24.04.4 LTS (Kaggle container); the script ran in a Python 3.11.18 venv built from `requirements.txt` |
| Inference backend | TensorRT 10.16 FP16 engine built by `prepare.py` (engine check passed) |

## How long it took

| Step | Time |
|---|---|
| Environment (Python 3.11 venv + `pip install -r requirements.txt`) | 2 min 13 s |
| Preparation (`prepare.py`: weights + ONNX export + TensorRT engine build and check) | 17 min 19 s |
| **Full run, all 50 shards, start to finish (including downloading the 25 GB of shards from Hugging Face)** | **2 h 25 min 5 s (8,705 s)** |
| Everything above together | 2 h 44 min 40 s |

How it was run:
- One `python run.py --input <shard dir> --output <part>.jsonl` per shard.
- Two in parallel, one per T4; GPU *g* took shards *n* with *n* mod 2 = *g*.
- 3 decode workers per process.
- A downloader fetched shards a few ahead of the GPUs.

Per-shard wall time was 262–457 s, mean 346 s.

What was processed:
- 1,650 videos, 461,757 seconds of video (128.3 h), 0 errors.
- 78,331 seconds (17 %) were refined in pass 2.
- 1,548,830 frames went through YOLO11l: about 12 % of all frames.

Where the time went on this machine: the GPUs were busy only about 35 % of the time (6,062 GPU-seconds over 2 GPUs × 8,705 s). The 4 slow vCPUs, decoding video for two processes at once, were the bottleneck. On the grading machine (32 threads, RTX 4090) decoding has 8× more cores, so the balance is very different.

Per-shard statistics and the full job log are in `docs/kaggle_run/`.
