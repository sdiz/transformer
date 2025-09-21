#!/usr/bin/env bash
set -euo pipefail

# Runner (Vaswani Big-angelehnt): d_model=1024, heads=16, layers=6, d_ff=4096, dropout=0.3
# Hinweis: sehr ressourcenintensiv. MAX_JOBS=1 und ggf. Batch-Size reduzieren.

MAX_JOBS=${MAX_JOBS:-1}
PY=python

OOS_YEARS=(1987 1998 2001 2008 2011 2015 2020 2022)

SEQ_LENGTHS=(30 60 120)
Z_WINDOWS=(63 126 252)

LAYERS=6
HEAD_SIZE=64             # 64*16 = 1024
NUM_HEADS=16
FF_MULT=4                # 4 * d_model
REFITS=(Y)
CALIBS=(none)
USE_RETURNS_FLAGS=(--use-returns)
HEADS=(--cls-head --last-token)

join_by_space() { local IFS=" "; echo "$*"; }
OOS_ARG=(--oos-years $(join_by_space "${OOS_YEARS[@]}"))

job_counter=0
run_job() {
  local cmd=$1
  echo "[RUN] $cmd"
  if [[ "$MAX_JOBS" -eq 1 ]]; then
    bash -c "$cmd"
  else
    bash -c "$cmd" &
    job_counter=$((job_counter+1))
    if [[ $job_counter -ge $MAX_JOBS ]]; then
      if wait -n 2>/dev/null; then :; else wait || true; fi
      job_counter=$((job_counter-1))
    fi
  fi
}

DMODEL=$((HEAD_SIZE*NUM_HEADS))

for sl in "${SEQ_LENGTHS[@]}"; do
  for zw in "${Z_WINDOWS[@]}"; do
    for rf in "${REFITS[@]}"; do
      for cal in "${CALIBS[@]}"; do
        for uh in "${HEADS[@]}"; do
          for ur in "${USE_RETURNS_FLAGS[@]}"; do
            ff=$((FF_MULT*DMODEL))
            cmd="$PY transformer.py ${OOS_ARG[*]} \
              --plots \
              --seq-length $sl --z-window $zw \
              --layers $LAYERS --head-size $HEAD_SIZE --num-heads $NUM_HEADS --ff-dim $ff \
              $uh $ur \
              --refit $rf --calibration $cal \
              --lr 1e-3 --batch-size 32 --epochs 20 --val-split 0.2 --clipnorm 1.0 \
              --es-patience 5 --rlrop-patience 3 --rlrop-factor 0.5"
            run_job "$cmd"
          done
        done
      done
    done
  done
done

wait || true
echo "All jobs finished."


