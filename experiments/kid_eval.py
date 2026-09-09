"""KID over the n_train=15000 grid, as a robustness check on the FID headline.

FID is a Gaussian fit to 2048-d Inception features: it assumes the two populations differ only in
mean and covariance, and it is biased at finite sample size. KID (`metrics.kernel_distance`) drops
both assumptions -- polynomial-kernel MMD^2, unbiased, no Gaussian fit -- so it is the standard
check on whether a small FID gap survives the estimator that produced it.

## The gap under test

At n_train=15000 the full-Pauli readout is the first digital QRC configuration to beat `classical`
(FID 56.31 vs 58.52, 0/6 seeds unfavorable), while the width-matched random control at the same 612
columns does not (63.38, 6/6 unfavorable). Both legs need KID or neither does.

`interaction` is in the grid despite not being in the headline: it is the arm where FID already
disagrees with val_mse and diversity (best val_mse of any arm at 0.4677, yet FID 65.30), which
makes it the one most likely to reorder under a different metric. Six cells is a cheap way to find
out.

Out of the grid on purpose: `multibase`/full (its FID sits 0.09 from quadrature/full, inside noise
-- nothing to check), and `random84`/`random312`, which no part of the current claim rests on.

## Why cells are re-run rather than re-read

Only the scalar FID was ever persisted; the 10000x2048 generated features it came from were not.
So KID needs the rollout again. Every re-run therefore also writes those features to
`inception_features/` (float32, ~82MB/cell, ~8.4GB for the grid) -- after this, any further
distribution metric costs seconds instead of three hours.

Output goes to `kid_cells/`, never `generation_cells/`: the committed cells stay byte-identical and
the re-run's FID can be diffed against them as a check that the pipeline still reproduces. The
gate for that check was `classical s0` (58.638182 committed) and `qrc quadrature/full s0 d0`
(53.888633).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

OUT = Path('results/qrc_fusion_fair')
CELLS = OUT / 'kid_cells'
SEEDS = (0, 1, 2, 3, 4, 5)
GEOM = ['--spacing-um', '9.756844364507792', '--omega', '6.283', '--s-detuning', '9.0',
        '--v-slices', '4', '--t-scale', '1.0']


def jobs(n_train: int):
    nt = str(n_train)
    stem = f'fashionmnist_%s_s%s_d%s_n10000_nt{n_train}%s.parquet'
    out = []
    for s in SEEDS:
        out.append((f'classical s{s}', stem % ('classical', s, -1, ''),
                    ['--method', 'classical', '--seed', str(s), '--draw', '-1']))
        out.append((f'interaction s{s}', stem % ('interaction', s, -1, ''),
                    ['--method', 'interaction', '--seed', str(s), '--draw', '-1']))
    for s in SEEDS:
        for dr in range(5):
            out.append((f'qrc full s{s} d{dr}', stem % ('qrc', s, dr, '_enc_quadrature_full'),
                        ['--method', 'qrc', '--seed', str(s), '--draw', str(dr),
                         '--encoding', 'quadrature', '--observables', 'full',
                         '--tag', 'enc_quadrature_full']))
    for s in SEEDS:
        for dr in range(5):
            out.append((f'qrc zz s{s} d{dr}', stem % ('qrc', s, dr, ''),
                        ['--method', 'qrc', '--seed', str(s), '--draw', str(dr)]))
    for s in SEEDS:                                    # width-matched to the full-Pauli readout
        for dr in range(5):
            out.append((f'random612 s{s} d{dr}', stem % ('random', s, dr, '_rand612'),
                        ['--method', 'random', '--seed', str(s), '--draw', str(dr),
                         '--random-width', '612', '--tag', 'rand612']))
    return [(name, cell, args + ['--n-train', nt]) for name, cell, args in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-train', type=int, default=15000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    base = ['--dataset', 'fashionmnist', '--samples', '10000', '--device', a.device, '--kid',
            '--cells-dir', 'kid_cells', '--feature-cache', 'inception_features']

    pending = [(n, args) for n, cell, args in jobs(a.n_train) if not (CELLS / cell).exists()]
    print(f'KID n_train={a.n_train}: {len(pending)} cells to run', flush=True)
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
    print(f'\nKID queue drained (n_train={a.n_train})', flush=True)


if __name__ == '__main__':
    main()
