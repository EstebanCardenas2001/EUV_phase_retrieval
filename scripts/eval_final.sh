#!/usr/bin/env bash
# Phase 4 evaluation: identical settings for every model family, then studies, figures and benchmarks.
# Run from the repo root inside tmux:  tmux new -d -s final_eval 'bash scripts/eval_final.sh'
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
LOG=logs/final_eval.log
step() { echo "$(date '+%F %T') $*" | tee -a $LOG; }

K3="saved_models/k3_pm1/best.pth"
ENS="saved_models/ens/m0/best.pth saved_models/ens/m1/best.pth saved_models/ens/m2/best.pth saved_models/ens/m3/best.pth saved_models/ens/m4/best.pth"
FM0="saved_models/final/m0/best.pth"
FENS="saved_models/final/m0/best.pth saved_models/final/m1/best.pth saved_models/final/m2/best.pth saved_models/final/m3/best.pth saved_models/final/m4/best.pth"

run() { step "START $1"; shift; "$@" >> $LOG 2>&1 || { step "FAILED (status $?)"; exit 1; }; }

for spec in "k3_pm1|$K3|30" "ens5|$ENS|0" "final_m0|$FM0|30" "final_ens5|$FENS|0"; do
  IFS='|' read -r NAME CKPTS MC <<< "$spec"
  run "evaluate $NAME x1"  python ai/evaluate.py --checkpoint $CKPTS --out-dir eval/$NAME
  run "evaluate $NAME x10" python ai/evaluate.py --checkpoint $CKPTS --noise-mult 10 --out-dir eval/${NAME}_n10
  run "uq_calibration $NAME (mc $MC)" python ai/uq_calibration.py --checkpoint $CKPTS --mc-passes $MC --out-dir eval/$NAME
done
run "uq_calibration ens5 x MC6"       python ai/uq_calibration.py --checkpoint $ENS --mc-passes 6 --out-dir eval/ens5_mc
run "uq_calibration final_ens5 x MC6" python ai/uq_calibration.py --checkpoint $FENS --mc-passes 6 --out-dir eval/final_ens5_mc

run "benchmark final_ens5" python ai/benchmark_speed.py --checkpoint $FENS --num-samples 256 --out-dir eval/benchmark
run "benchmark final_m0"   python ai/benchmark_speed.py --checkpoint $FM0 --num-samples 256 --out-dir eval/benchmark_final_m0

run "diversity study" python ai/diversity_study.py --num-phases 96 --iterations 1000 --out-dir eval
run "solver figure K=1" python solver/gradient_descent.py --diversity 0 --out docs/figures/solver_single_plane.png
run "solver figure K=3" python solver/gradient_descent.py --out docs/figures/solver_diversity.png
step "ALL_DONE"
