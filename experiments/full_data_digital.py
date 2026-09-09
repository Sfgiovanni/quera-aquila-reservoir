"""Digital/classical grid at the full training set, 6 seeds.

Motivated by the n_train dependence already in the results: `classical` improves from FID 72.35
(n_train=500) to 59.16 (n_train=5000), a jump larger than any between-arm difference, and it is the
*best* arm at 5000. Whether that keeps going decides whether a Rydberg cell at full data is worth
its ~9h, so the cheap half of the question is answered first.

## `n_train=15000` is the whole dataset here, and 60000 is not

`qrc_fusion_fair_supervised.load_latents` encodes `tr[:20000]` and splits it `z[:15000]` /
`z[15000:]` -- so this pipeline holds **15,000 training latents and 5,000 validation latents**, not
the official 60k/10k. `balanced_pairs` draws `image_id` with `replace=count > len(z)`, so:

- `n_train=15000` uses every training image **exactly once** (`replace=False`). True full data.
- `n_train=60000` would resample the same 15,000 images ~4x with different `(t, epsilon)` draws.
  That is a different experiment -- readout-fit saturation in noise realizations, not more data --
  and it should not be labelled "the full dataset".

Going past 15,000 distinct images means re-encoding a larger slice of `tr`, which moves the
train/val boundary and breaks comparability with every committed cell. Not done here.

## Cost

`n_train` barely moves the digital arms: features come off the GPU in seconds and the ~3 min per
cell is almost entirely Inception over the 10,000 generated images, which is fixed. 102 cells at
n_train=15000 is a few hours, against ~9h for a *single* Rydberg cell.

## Order

`classical` and `interaction` first, all 6 seeds: they are the two reference arms and 12 cells
already answer whether the n_train trend continues. Then `qrc` digital, then the two random
controls. Stopping after any block leaves a coherent result.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

CELLS = Path('results/qrc_fusion_fair/generation_cells')
SEEDS = (0, 1, 2, 3, 4, 5)
GEOM = ['--spacing-um', '9.756844364507792', '--omega', '6.283', '--s-detuning', '9.0',
        '--v-slices', '4', '--t-scale', '1.0']


def jobs(n_train: int):
    nt = str(n_train)
    stem = f'fashionmnist_%s_s%s_d%s_n10000_nt{n_train}%s.parquet'
    out = []
    for s in SEEDS:                                    # the two reference arms
        out.append((f'classical s{s}', stem % ('classical', s, -1, ''),
                    ['--method', 'classical', '--seed', str(s), '--draw', '-1']))
        out.append((f'interaction s{s}', stem % ('interaction', s, -1, ''),
                    ['--method', 'interaction', '--seed', str(s), '--draw', '-1']))
    for s in SEEDS:
        for dr in range(5):
            out.append((f'qrc digital s{s} d{dr}', stem % ('qrc', s, dr, ''),
                        ['--method', 'qrc', '--seed', str(s), '--draw', str(dr)]))
    for s in SEEDS:
        for dr in range(5):
            out.append((f'random84 s{s} d{dr}', stem % ('random', s, dr, ''),
                        ['--method', 'random', '--seed', str(s), '--draw', str(dr)]))
    for s in SEEDS:                                    # width-matched to the rydberg arm (312)
        for dr in range(5):
            out.append((f'random312 s{s} d{dr}', stem % ('random', s, dr, '_rydberg_gate6'),
                        ['--method', 'random', '--reservoir', 'rydberg', '--seed', str(s),
                         '--draw', str(dr), '--tag', 'gate6', *GEOM]))
    return [(name, cell, args + ['--n-train', nt]) for name, cell, args in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-train', type=int, default=15000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    base = ['--dataset', 'fashionmnist', '--samples', '10000', '--device', a.device]

    pending = [(n, args) for n, cell, args in jobs(a.n_train) if not (CELLS / cell).exists()]
    print(f'n_train={a.n_train}: {len(pending)} cells to run', flush=True)
    if a.dry_run:
        for n, args in pending:
            print(' ', n, ' '.join(args))
        return
    for i, (name, args) in enumerate(pending, 1):
        cmd = [sys.executable, '-u', '-m', 'experiments.qrc_fusion_fair_generation', *base, *args]
        print(f'\n[{i}/{len(pending)}] {name}', flush=True)
        started = perf_counter()
        rc = subprocess.run(cmd, env=dict(os.environ)).returncode
        print(f'[{i}/{len(pending)}] {name} rc={rc} ({perf_counter() - started:.0f}s)', flush=True)
        if rc != 0:
            print(f'FAILED {name} -- continuing', flush=True)
    print(f'\nfull-data digital queue drained (n_train={a.n_train})', flush=True)


if __name__ == '__main__':
    main()
