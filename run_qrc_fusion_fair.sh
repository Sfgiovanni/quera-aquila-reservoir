#!/usr/bin/env bash
# Stage 2 of the bias-corrected fusion comparison: a few cells at a time on the GPU.
# n_train=500 chains to the published protocol; n_train=5000 is where the supervised sweep put the
# signal. Arms: plain classical, QRC observables, dimension-matched random map, and the structured
# classical interaction block that the supervised stage showed dominates the QRC.
#
# Each job covers the three data seeds sequentially, so one process reuses its loaded autoencoder
# and Inception weights across them.
set -uo pipefail
cd "$(dirname "$0")"
PAR=${PAR:-3}
jobs_file=$(mktemp)
for dataset in fashionmnist breastmnist; do
  samples=10000; [ "$dataset" = breastmnist ] && samples=1000
  for n_train in 500 5000; do
    printf '%s classical -1 %s %s\n' "$dataset" "$n_train" "$samples" >> "$jobs_file"
    printf '%s interaction -1 %s %s\n' "$dataset" "$n_train" "$samples" >> "$jobs_file"
    for draw in 0 1 2 3 4; do
      printf '%s qrc %s %s %s\n' "$dataset" "$draw" "$n_train" "$samples" >> "$jobs_file"
      printf '%s random %s %s %s\n' "$dataset" "$draw" "$n_train" "$samples" >> "$jobs_file"
    done
  done
done
echo "jobs queued: $(wc -l < "$jobs_file") (x3 seeds each)"
xargs -P "$PAR" -n 5 bash -c '
  for s in 0 1 2; do
    conda run -n qrc python -m experiments.qrc_fusion_fair_generation \
      --dataset "$0" --method "$1" --draw "$2" --n-train "$3" --samples "$4" \
      --seed "$s" --device cuda || echo "FAILED $0 $1 draw=$2 nt=$3 seed=$s"
  done' < "$jobs_file"
