#!/usr/bin/env bash
# Full run on the first 50 FineVideo shards on Kaggle (2x T4, 4 vCPU, limited disk).
# A downloader keeps a few shards ahead; GPU g processes shards n with n % 2 == g, each
# with an unmodified `python run.py --input <dir> --output <file>`; shards are deleted
# after use. Per-shard outputs are concatenated into results.jsonl.gz at the end.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/kaggle/working/out
DATA=/tmp/fv
PARTS=$OUT/parts
mkdir -p "$PARTS" "$DATA"
FIRST=${FIRST:-0}
LAST=${LAST:-49}
AHEAD=${AHEAD:-6}
WORKERS=${WORKERS:-3}
T0=$(date +%s)

fetch() {
  local n=$1 d="$DATA/$1"
  mkdir -p "$d"
  local name; name=$(printf "data/train-%05d-of-01357.parquet" "$n")
  for try in 1 2 3 4 5 6; do
    hf download HuggingFaceFV/finevideo --repo-type dataset --include "$name" --local-dir "$d/dl" >/dev/null 2>&1 && break
    sleep 15
  done
  mv "$d"/dl/data/*.parquet "$d"/ 2>/dev/null
  rm -rf "$d/dl"
  touch "$d/.ready"
}

downloader() {
  for n in $(seq "$FIRST" "$LAST"); do
    # stay at most $AHEAD shards ahead of the slowest consumer
    while [ "$(ls -d "$DATA"/*/ 2>/dev/null | wc -l)" -ge "$AHEAD" ]; do sleep 2; done
    if [ -s "$PARTS/$n.done" ]; then continue; fi
    t=$(date +%s); fetch "$n"; echo "fetched $n in $(( $(date +%s) - t ))s"
  done
}

consumer() {
  local g=$1
  for n in $(seq "$FIRST" "$LAST"); do
    [ $(( n % 2 )) -eq "$g" ] || continue
    [ -s "$PARTS/$n.done" ] && continue
    while [ ! -f "$DATA/$n/.ready" ]; do sleep 1; done
    t=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g python "$REPO/run.py" --input "$DATA/$n" --output "$PARTS/$n.jsonl" \
      --workers "$WORKERS" --stats "$PARTS/$n.stats.json" > "$PARTS/$n.log" 2>&1
    rc=$?
    echo "gpu$g shard $n rc=$rc $(( $(date +%s) - t ))s $(tail -c 400 "$PARTS/$n.stats.json" 2>/dev/null | tr -d '\n ' | head -c 300)"
    [ $rc -eq 0 ] && date +%s > "$PARTS/$n.done"
    rm -rf "${DATA:?}/$n"
  done
}

downloader &
DL=$!
consumer 0 &
C0=$!
consumer 1 &
C1=$!
wait $C0 $C1
kill $DL 2>/dev/null
T1=$(date +%s)

# assemble
cat $(for n in $(seq "$FIRST" "$LAST"); do echo "$PARTS/$n.jsonl"; done) | gzip -9 > "$OUT/results.jsonl.gz"
python - "$PARTS" "$FIRST" "$LAST" "$(( T1 - T0 ))" > "$OUT/full_summary.json" <<'EOF'
import json, sys, glob, os
parts, a, b, wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tot = {}
for n in range(a, b + 1):
    p = f"{parts}/{n}.stats.json"
    if os.path.exists(p):
        for k, v in json.load(open(p)).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                tot[k] = tot.get(k, 0) + v
tot["wall_s_including_downloads"] = wall
tot["shards_done"] = len(glob.glob(f"{parts}/*.done"))
print(json.dumps(tot, indent=1))
EOF
cat "$OUT/full_summary.json"
zcat "$OUT/results.jsonl.gz" | wc -l
ls -la "$OUT/results.jsonl.gz"
