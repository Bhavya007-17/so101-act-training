#!/usr/bin/env bash
# Run a command while sampling GPU memory from nvidia-smi, then report the peak.
#
# torch.cuda.max_memory_reserved() only sees the allocator's own pool; the CUDA
# context, cuDNN workspaces and kernel code are outside it. nvidia-smi sees the
# real process footprint, which is what actually has to fit in 8 GB.
#
# usage: run_with_vram.sh <label> <cmd...>
set -uo pipefail

LABEL="$1"; shift
SAMPLES="logs/vram_${LABEL}.csv"
mkdir -p logs
echo "timestamp_s,used_mib" > "$SAMPLES"

(
  while true; do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${U:-}" ] && echo "$(date +%s),$U" >> "$SAMPLES"
    sleep 0.5
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

START=$(date +%s)
"$@"
RC=$?
END=$(date +%s)

kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null

PEAK=$(awk -F, 'NR>1 && $2+0>m {m=$2+0} END {print m+0}' "$SAMPLES")
N=$(( $(wc -l < "$SAMPLES") - 1 ))
echo
echo "--------------------------------------------------------------"
echo "  label            : $LABEL"
echo "  exit code        : $RC"
echo "  wall clock       : $((END-START)) s"
echo "  PEAK GPU MEM     : ${PEAK} MiB   (nvidia-smi, ${N} samples @ 0.5s)"
echo "  samples          : $SAMPLES"
echo "--------------------------------------------------------------"
exit $RC
