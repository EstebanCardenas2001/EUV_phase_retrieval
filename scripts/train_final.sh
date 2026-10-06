#!/usr/bin/env bash
# Final ensemble: 5 members (seeds 0-4), trained one after another so the GPU never idles.
# Run from the repo root inside tmux:  tmux new -d -s final 'bash scripts/train_final.sh'
# A member whose final.pth already exists is skipped, so the script can simply be restarted
# after an interruption (the interrupted member restarts from scratch; there is no mid-run resume).
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p logs saved_models/final training_progress/final

EPOCHS=250
SAMPLES_PER_EPOCH=4096
VAL_SAMPLES=1024

for SEED in 0 1 2 3 4; do
  OUT=saved_models/final/m$SEED
  if [ -f "$OUT/final.pth" ]; then
    echo "$(date '+%F %T') m$SEED already finished, skipping" | tee -a logs/final_driver.log
    continue
  fi
  echo "$(date '+%F %T') starting m$SEED" | tee -a logs/final_driver.log
  python ai/train.py --epochs $EPOCHS --batch-size 32 --samples-per-epoch $SAMPLES_PER_EPOCH \
      --lr 3e-4 --scheduler cosine --noise-aug-max 30 --diversity=-1.0,0.0,1.0 --mode hybrid \
      --val-samples $VAL_SAMPLES --val-seed 12345 --r2-every 10 --seed $SEED \
      --save-dir $OUT --progress-dir training_progress/final/m$SEED > logs/final_m$SEED.log 2>&1
  STATUS=$?
  echo "$(date '+%F %T') m$SEED exited with status $STATUS" | tee -a logs/final_driver.log
  if [ $STATUS -ne 0 ]; then
    echo "stopping: m$SEED failed (see logs/final_m$SEED.log)" | tee -a logs/final_driver.log
    exit $STATUS
  fi
done
echo "$(date '+%F %T') all members done" | tee -a logs/final_driver.log
