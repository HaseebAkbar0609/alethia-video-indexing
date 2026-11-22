#!/usr/bin/env bash
# Kaggle driver (2x T4). The notebook only does:
#   HF_TOKEN from Kaggle secrets -> git clone this repo -> bash kaggle/job.sh
# What runs is chosen by kaggle/mode.txt so the notebook never has to change.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/kaggle/working/out
DATA=/tmp/fv
mkdir -p "$OUT" "$DATA"
MODE="${MODE:-$(cat "$REPO/kaggle/mode.txt" | tr -d '[:space:]')}"
echo "mode=$MODE repo=$REPO commit=$(git -C "$REPO" rev-parse --short HEAD)" | tee "$OUT/job.log"
T_JOB=$(date +%s)

# ---- hardware report -----------------------------------------------------------------
{
  echo "date: $(date -u)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
  free -g | head -n 2
  df -h /tmp /kaggle/working | tail -n 2
  cat /etc/os-release | grep PRETTY
  lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null
} > "$OUT/hardware.txt" 2>&1

# ---- environment: same as the graders (Python 3.11 venv + requirements.txt) ------------
T0=$(date +%s)
pip install -q uv >/dev/null 2>&1
uv python install 3.11 >/dev/null 2>&1
unset UV_SYSTEM_PYTHON UV_PYTHON
uv venv -q --clear -p 3.11 /tmp/venv
export VIRTUAL_ENV=/tmp/venv PATH=/tmp/venv/bin:$PATH
uv pip install -q --python /tmp/venv/bin/python -r "$REPO/requirements.txt" --index-strategy unsafe-best-match 2>&1 | tail -n 5
python -c "import torch, av, ultralytics, tensorrt; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'av', av.__version__, 'trt', tensorrt.__version__)" | tee -a "$OUT/job.log"
echo "env_setup_s=$(( $(date +%s) - T0 ))" | tee -a "$OUT/job.log"

fetch() {  # fetch shard index N into $DATA/N/
  local n=$1 d="$DATA/$1"
  mkdir -p "$d"
  local name; name=$(printf "data/train-%05d-of-01357.parquet" "$n")
  for try in 1 2 3 4 5; do
    hf download HuggingFaceFV/finevideo --repo-type dataset --include "$name" --local-dir "$d/dl" >/dev/null 2>&1 && break
    sleep 10
  done
  mv "$d"/dl/data/*.parquet "$d"/ 2>/dev/null
  rm -rf "$d/dl"
}

# ---- one-time preparation (weights + TensorRT engine) ----------------------------------
T0=$(date +%s)
mkdir -p /kaggle/working/cache
[ -f /kaggle/working/cache/yolo11l.engine ] && cp /kaggle/working/cache/yolo11l.engine "$REPO/"
( cd "$REPO" && python -u prepare.py --batch 64 ) 2>&1 | grep --line-buffered -v "^ " | tee -a "$OUT/job.log"
echo "prepare_s=$(( $(date +%s) - T0 ))" | tee -a "$OUT/job.log"
[ -f "$REPO/yolo11l.engine" ] && cp "$REPO/yolo11l.engine" /kaggle/working/cache/
ls -la "$REPO"/yolo11l.* | tee -a "$OUT/job.log"

case "$MODE" in
oracle)
  fetch 49
  ( CUDA_VISIBLE_DEVICES=0 python "$REPO/tools/oracle.py" --input "$DATA/49" --out "$OUT/oracle_fast_49.npz" > "$OUT/oracle_fast.log" 2>&1 ) &
  ( CUDA_VISIBLE_DEVICES=1 python "$REPO/tools/oracle.py" --input "$DATA/49" --out "$OUT/oracle_stock_49.npz" --limit 8 --stock-limit 8 > "$OUT/oracle_stock.log" 2>&1 ) &
  wait
  tail -n 3 "$OUT"/oracle_*.log | tee -a "$OUT/job.log"
  python "$REPO/tools/simulate.py" "$OUT/oracle_fast_49.npz" "$OUT/oracle_stock_49.npz" 2>&1 | tee "$OUT/simulate.txt"
  # end-to-end benchmark of run.py on the same shard, one GPU
  for sps in 3; do
    CUDA_VISIBLE_DEVICES=0 python "$REPO/run.py" --input "$DATA/49" --output "$OUT/bench_sps$sps.jsonl" \
      --samples-per-second $sps --stats "$OUT/bench_sps$sps.json" 2>&1 | grep -v "^ " | tail -n 4 | tee -a "$OUT/job.log"
  done
  ;;
full)
  bash "$REPO/kaggle/full.sh" 2>&1 | tee -a "$OUT/job.log"
  ;;
heldout)
  # every-frame oracles for shards the settings were NOT tuned on (tuning used shard 49 only);
  # shards fixed in advance: the first (0) and the middle (25) of the 50
  fetch 0 & fetch 25 & wait
  ( CUDA_VISIBLE_DEVICES=0 python "$REPO/tools/oracle.py" --input "$DATA/0" --out "$OUT/oracle_0.npz" --stock-limit 8 > "$OUT/oracle_0.log" 2>&1 ) &
  ( CUDA_VISIBLE_DEVICES=1 python "$REPO/tools/oracle.py" --input "$DATA/25" --out "$OUT/oracle_25.npz" --stock-limit 8 > "$OUT/oracle_25.log" 2>&1 ) &
  wait
  tail -n 3 "$OUT"/oracle_*.log | tee -a "$OUT/job.log"
  # score the production output of the full run (results.jsonl.gz, unchanged settings) on these shards
  for s in 0 25; do
    f=$(printf "train-%05d-of-01357" "$s")
    zcat "$REPO/results.jsonl.gz" | grep "\"$f" > "$OUT/res_$s.jsonl"
    for src in fast stock; do
      python "$REPO/tools/oracle_to_ref.py" "$OUT/oracle_$s.npz" --source $src > "$OUT/ref_${s}_$src.jsonl"
      python "$REPO/tools/compare.py" "$OUT/ref_${s}_$src.jsonl" "$OUT/res_$s.jsonl" --detail > "$OUT/heldout_${s}_$src.json"
    done
  done
  python - "$OUT" <<'PY' 2>&1 | tee -a "$OUT/job.log"
import json, sys
o = sys.argv[1]
for s in (0, 25):
    for src in ("fast", "stock"):
        d = json.load(open(f"{o}/heldout_{s}_{src}.json"))
        print("HELDOUT", json.dumps({"shard": s, "ref": src, **d}, separators=(",", ":")))
PY
  ;;
esac
echo "job_total_s=$(( $(date +%s) - T_JOB ))" | tee -a "$OUT/job.log"
