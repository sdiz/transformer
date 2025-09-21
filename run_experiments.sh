#!/usr/bin/env bash
set -euo pipefail

# Grid runner for transformer.py
# Usage:
#   bash run_experiments.sh
# Optional env vars:
#   MAX_JOBS: parallel jobs (default 2)
#   FULL=1   : expand grid (adds more values)

MAX_JOBS=${MAX_JOBS:-1}
PY=python

# Scientifically motivated OOS years (crisis/regime episodes)
OOS_YEARS=(1987 1998 2001 2008 2011 2015 2020 2022)

# Full grid (immer aktiv)
SEQ_LENGTHS=(30 60 120)
Z_WINDOWS=(63 126 252)
LAYERS=(2 4)
HEAD_SIZE=16
NUM_HEADS=4
FF_MULTS=(2 4)         # ff_dim = FF_MULT * (HEAD_SIZE*NUM_HEADS)
DROPOUTS=(0.1 0.2 0.3)
ATTN_DROPS=(0.0 0.1 0.2)
REFITS=(Y)
CALIBS=(none)
# USE_RETURNS_FLAGS=(--no-returns --use-returns)
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
    for lyr in "${LAYERS[@]}"; do
      for ffm in "${FF_MULTS[@]}"; do
        ff=$((ffm*DMODEL))
        for dr in "${DROPOUTS[@]}"; do
          for adr in "${ATTN_DROPS[@]}"; do
            for rf in "${REFITS[@]}"; do
              for cal in "${CALIBS[@]}"; do
                for uh in "${HEADS[@]}"; do
                  for ur in "${USE_RETURNS_FLAGS[@]}"; do
                    cmd="$PY transformer.py ${OOS_ARG[*]} \
                      --plots \
                      --seq-length $sl --z-window $zw \
                      --layers $lyr --head-size $HEAD_SIZE --num-heads $NUM_HEADS --ff-dim $ff \
                      --dropout $dr --attn-dropout $adr $uh $ur \
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
      done
    done
  done
done

wait || true
echo "All jobs finished."


