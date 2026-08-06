#!/usr/bin/env bash
# BreastMNIST n_train sweep for the two arms the question is about: ridge only and QRC + ridge.
#
# Caveat that belongs next to every number this produces: BreastMNIST has 546 training images and 78
# validation images. `balanced_pairs` samples with replacement once count exceeds that, so n_train
# past ~546 buys more (timestep, epsilon) realisations per image, NOT more image diversity. The
# sweep therefore measures saturation in noise coverage, not in data.
set -uo pipefail
cd "$(dirname "$0")"
PAR=${PAR:-3}
jobs_file=$(mktemp)
for n_train in 250 500 1000 2000 5000 10000 20000; do
  printf 'classical -1 %s\n' "$n_train" >> "$jobs_file"
  for draw in 0 1 2 3 4; do printf 'qrc %s %s\n' "$draw" "$n_train" >> "$jobs_file"; done
done
echo "jobs queued: $(wc -l < "$jobs_file") (x3 seeds each)"
xargs -P "$PAR" -n 3 bash -c '
  for s in 0 1 2; do
    conda run -n qrc python -m experiments.qrc_fusion_fair_generation \
      --dataset breastmnist --method "$0" --draw "$1" --n-train "$2" \
      --samples 1000 --seed "$s" --device cuda || echo "FAILED $0 draw=$1 nt=$2 seed=$s"
  done' < "$jobs_file"
