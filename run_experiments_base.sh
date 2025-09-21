#!/usr/bin/env bash
set -euo pipefail

###############################
# Base-Runner: nutzt ausschließlich die Defaults aus transformer.py
# (layers=4, head-size=16, num-heads=4, ff-dim=128, dropout=0.2, attn-dropout=0.1, CLS-Head)
# Wir scannen lediglich Sequenz-/Z-Fenster. Keine Architektur-Overrides.
###############################

MAX_JOBS=${MAX_JOBS:-1}
PY=python

# OOS years (crisis/regime)
OOS_YEARS=(1987 1998 2001 2008 2011 2015 2020 2022)

# Grid: Daten-/Fenster-Parameter + Head-Variante
SEQ_LENGTHS=(30 60 120)
Z_WINDOWS=(63 126 252)
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

for sl in "${SEQ_LENGTHS[@]}"; do
  for zw in "${Z_WINDOWS[@]}"; do
    for uh in "${HEADS[@]}"; do
      cmd="$PY transformer.py ${OOS_ARG[*]} \
        --plots \
        --seq-length $sl --z-window $zw $uh \
        --refit Y --calibration none \
        --lr 1e-3 --batch-size 32 --epochs 20 --val-split 0.2 --clipnorm 1.0 \
        --es-patience 5 --rlrop-patience 3 --rlrop-factor 0.5"
      run_job "$cmd"
    done
  done
done

wait || true
echo "All jobs finished."


