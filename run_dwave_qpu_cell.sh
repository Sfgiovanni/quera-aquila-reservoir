#!/usr/bin/env bash
# The annealing reservoir on real D-Wave hardware, in three stages, under a hard cumulative
# QPU-time ceiling.
#
# Operating point is the published SQA cell's, unchanged, so the hardware number pairs against
# `results/qrc_fusion_fair/generation_cells_anneal_dwave_sqa_full/` (FID 73.90 +/- 1.62, 6 seeds):
#   12 spins, one 5 ns probe, 8 reads, h_scale 2.0, j_scale 0.5, full K12 couplings,
#   advantage2-fast schedule, n_train 10000, samples 10000.
# 5 ns is exactly the floor of Advantage2's `fast_anneal_time_range` and below the standard
# `annealing_time_range` floor of 0.5 us, so this arm submits with fast_anneal=True.
#
# WHY THREE STAGES. The quota is not the binding constraint (6 seeds cost ~6.6 of 24 minutes);
# wall-clock is (~1-3 h per seed). Firing all six blind risks discovering a bad embedding after
# 18 h and 6.6 unrecoverable minutes. Each stage answers one question before the next one pays:
#
#   --smoke   3.8 s QPU   does the path work on real hardware at all?
#   --seed0   65.7 s      are chain breaks low and the FID plausible?
#   --rest    5.5 min     the publishable 6-seed result
#
# BUDGET. Every submission is charged to $LEDGER before it leaves the process and settled against
# the solver's reported `qpu_access_time` afterwards. The run raises BudgetExceeded rather than
# crossing CAP*(1-MARGIN). The ledger is cumulative across runs AND processes (flock'd read-modify-
# write), which is what makes --rest safe to run in parallel. To start a fresh quota month, move
# the ledger aside -- do not raise CAP.
#
# Usage:
#   dwave config create                        # once, or export DWAVE_API_TOKEN
#   MOCK=1 ./run_dwave_qpu_cell.sh --dry-run   # plan with no token and no QPU time
#   ./run_dwave_qpu_cell.sh --dry-run          # plan against the real solver (still no QPU time)
#   ./run_dwave_qpu_cell.sh --smoke
#   ./run_dwave_qpu_cell.sh --seed0
#   ./run_dwave_qpu_cell.sh --rest             # seeds 1-5, PAR at a time
#   ./run_dwave_qpu_cell.sh --status           # ledger balance, spends nothing
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
DRAW=${DRAW:-0}
N_TRAIN=${N_TRAIN:-10000}
SAMPLES=${SAMPLES:-10000}
READS=${READS:-8}
ANNEAL_US=${ANNEAL_US:-0.005}
CAP=${CAP:-1440}            # 24 minutes, the whole quota
MARGIN=${MARGIN:-0.10}      # never spend the last 10%
LEDGER=${LEDGER:-results/dwave_qpu_ledger.json}
CELLS=${CELLS:-generation_cells_anneal_dwave_qpu}
PAR=${PAR:-2}               # concurrent seed processes in --rest
REST_SEEDS=${REST_SEEDS:-"1 2 3 4 5"}
# Above this mean chain-break fraction, a cell is reporting majority-vote artefacts rather than
# annealing, and --rest refuses to spend the remaining quota on more of them.
MAX_CHAIN_BREAK=${MAX_CHAIN_BREAK:-0.05}
MOCK_FLAG=""; [ "${MOCK:-0}" = 1 ] && MOCK_FLAG="--mock"

usage() {
  sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

ledger_summary() {
  "$PY" -c "
from pathlib import Path
from anneal.qpu_budget import QpuBudget
import json; print(json.dumps(QpuBudget(Path('$LEDGER')).summary(), indent=1))"
}

# preflight <n_train> <samples> <plan_path> -- consumes no QPU time; returns non-zero if it does
# not fit the REMAINING budget or asks for an anneal time the solver cannot reach.
preflight() {
  local nt=$1 samples=$2 plan=$3
  echo "== preflight (consumes no QPU time) =="
  "$PY" -m experiments.dwave_qpu_preflight \
    --n-train "$nt" --samples "$samples" --reads "$READS" \
    --anneal-times-us "$ANNEAL_US" --cap-seconds "$CAP" \
    --ledger "$LEDGER" --out "$plan" $MOCK_FLAG || return 1
  "$PY" -c "
import json,sys
plan=json.load(open('$plan'))
sys.exit(0 if plan['fits'] and not plan['unreachable_times_us'] else 1)" || {
    echo "preflight: this cell does not fit the remaining budget -- not submitting." >&2
    return 1; }
}

# run_cell <seed> <n_train> <samples> <cells_dir> <tag>
run_cell() {
  local seed=$1 nt=$2 samples=$3 cells=$4 tag=$5
  "$PY" -m experiments.qrc_fusion_fair_generation \
    --dataset fashionmnist --method qrc --reservoir anneal \
    --anneal-backend dwave-qpu --anneal-times-us "$ANNEAL_US" --anneal-reads "$READS" \
    --anneal-schedule advantage2-fast --anneal-h-scale 2.0 --anneal-j-scale 0.5 \
    --shots "$READS" --seed "$seed" --draw "$DRAW" \
    --n-train "$nt" --samples "$samples" \
    --qpu-cap-seconds "$CAP" --qpu-margin "$MARGIN" --qpu-ledger "$LEDGER" \
    --cells-dir "$cells" --tag "$tag"
}

# Gate on stage 2's result before stage 3 spends the rest of the quota.
check_seed0() {
  "$PY" - "$CELLS" "$MAX_CHAIN_BREAK" <<'EOF'
import sys, glob
import pandas as pd
cells, limit = sys.argv[1], float(sys.argv[2])
hits = glob.glob(f'results/qrc_fusion_fair/{cells}/*_s0_*anneal_dwave_qpu*.parquet')
if not hits:
    sys.exit("no seed-0 cell found: run './run_dwave_qpu_cell.sh --seed0' first")
d = pd.read_parquet(hits[0])
cb = float(d.get('qpu_mean_chain_break_fraction', pd.Series([0.0])).iloc[0])
fid = float(d['fid'].iloc[0])
print(f"seed 0: FID {fid:.3f}  chain breaks {cb:.3%}  tiles {int(d['qpu_tiles'].iloc[0])}  "
      f"QPU {float(d['qpu_access_time_s'].iloc[0]):.1f}s")
if bool(d['degenerate'].iloc[0]):
    sys.exit("seed 0 tripped the degeneracy guard -- do not spend more quota on this config")
if cb > limit:
    sys.exit(f"seed 0 chain-break fraction {cb:.3%} exceeds {limit:.1%}: the cell is reporting "
             f"majority-vote artefacts, not annealing. Lower --anneal-j-scale (a weaker J needs a "
             f"weaker chain) rather than raising chain strength back over the solver's range.")
EOF
}

case "${1:-}" in
  --status)
    ledger_summary; exit 0 ;;

  --dry-run)
    preflight "$N_TRAIN" "$SAMPLES" results/qrc_fusion_fair/dwave_qpu_plan.json
    echo "dry run: stopping before any submission"; exit 0 ;;

  --smoke)
    echo "== stage 1/3: smoke (nt=500, samples=500, ~4 s of QPU) =="
    preflight 500 500 results/qrc_fusion_fair/dwave_qpu_plan_smoke.json || exit 1
    run_cell 0 500 500 "${CELLS}_smoke" anneal_dwave_qpu_smoke || exit 1
    echo; ledger_summary
    echo; echo "next: ./run_dwave_qpu_cell.sh --seed0" ;;

  --seed0)
    echo "== stage 2/3: seed 0 at full scale (~66 s of QPU) =="
    preflight "$N_TRAIN" "$SAMPLES" results/qrc_fusion_fair/dwave_qpu_plan.json || exit 1
    run_cell 0 "$N_TRAIN" "$SAMPLES" "$CELLS" anneal_dwave_qpu || exit 1
    echo; check_seed0
    echo; ledger_summary
    echo; echo "next: ./run_dwave_qpu_cell.sh --rest" ;;

  --rest)
    echo "== stage 3/3: seeds $REST_SEEDS, $PAR at a time =="
    # The QPU serialises annealing regardless, but network latency overlaps across processes --
    # which is the actual bottleneck. Concurrency is safe because the ledger is flock'd.
    check_seed0 || exit 1
    preflight "$N_TRAIN" "$SAMPLES" results/qrc_fusion_fair/dwave_qpu_plan.json || exit 1
    export PY DRAW N_TRAIN SAMPLES READS ANNEAL_US CAP MARGIN LEDGER CELLS
    printf '%s\n' $REST_SEEDS | xargs -P "$PAR" -I{} bash -c '
      "$PY" -m experiments.qrc_fusion_fair_generation \
        --dataset fashionmnist --method qrc --reservoir anneal \
        --anneal-backend dwave-qpu --anneal-times-us "$ANNEAL_US" --anneal-reads "$READS" \
        --anneal-schedule advantage2-fast --anneal-h-scale 2.0 --anneal-j-scale 0.5 \
        --shots "$READS" --seed {} --draw "$DRAW" \
        --n-train "$N_TRAIN" --samples "$SAMPLES" \
        --qpu-cap-seconds "$CAP" --qpu-margin "$MARGIN" --qpu-ledger "$LEDGER" \
        --cells-dir "$CELLS" --tag anneal_dwave_qpu || echo "seed {} stopped" >&2'
    echo; ledger_summary ;;

  ""|--help|-h) usage 0 ;;
  *) echo "unknown option: $1" >&2; usage 1 ;;
esac
