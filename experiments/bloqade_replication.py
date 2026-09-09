"""The remaining full-scale cells with QuEra's bloqade-analog doing the Schrodinger solve.

`docs/AQUILA_PORT.md`'s Gate 6 numbers were produced by `quera/emulator.py`. For a paper the
physics should be attributable to the vendor's own emulator, so this re-runs the same cells with
`--backend bloqade`: identical geometry, identical protocol, identical downstream code (readout,
DDIM, PRDC, FID) -- the *only* component swapped is the solver. That is what makes the claim exact
rather than "two implementations agreed": everything after `rydberg_features` is the same code
either way, so agreement at the feature level propagates to the FID by construction.

The `samples=500` cell already ran and reproduced `85.4065` against `85.4018` -- a 0.005 FID
difference, against the 2.91 FID seed-to-seed SD Gate 6 measured, i.e. ~600x inside the
experiment's own noise. These five extend that to the full-scale, 3-seed grid.

## Resource notes, learned the hard way

`--bloqade-workers 20` with `BLOQADE_MAXTASKS=50` is not a guess: bloqade grows from ~250MB to
700MB+ RSS per worker over a few thousand solves (it appears to cache compiled programs, and `h`
differs on every task), so 22 workers with loose recycling walked a 15.7GB machine down to 400MB
free in four minutes. This configuration sits on a flat ~6GB-available plateau. Throughput is
~3.3-3.6 cache entries/min, i.e. **~5h per cell**, which is ~78% of the pure-solve floor; the rest
is IPC and the per-worker re-forks, and it does not improve by keeping workers alive longer (that
was measured, see `quera/_bloqade_worker.py`).

Cells run strictly one at a time -- two concurrent pools do not fit in memory. Each cell writes its
own parquet and `qrc_fusion_fair_generation.py` skips existing ones, so killing this driver and
re-running it resumes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

CELLS = Path('results/qrc_fusion_fair/generation_cells')
GEOM = ['--spacing-um', '9.756844364507792', '--omega', '6.283', '--s-detuning', '9.0',
        '--v-slices', '4', '--t-scale', '1.0']
COMMON = ['--dataset', 'fashionmnist', '--method', 'qrc', '--reservoir', 'rydberg',
          '--draw', '-1', '--n-train', '500', '--samples', '10000', '--rk4-safety', '0.2',
          '--device', 'cuda', '--backend', 'bloqade', '--bloqade-workers', '20']

# Exact seeds first: they complete a 3-seed bloqade arm against Gate 6's finished 3-seed torch
# arm, which is the comparison a reviewer actually needs. The shots arm follows.
JOBS = [(1, None), (2, None), (0, 1000), (1, 1000), (2, 1000)]


def main():
    env = dict(os.environ, BLOQADE_MAXTASKS=os.environ.get('BLOQADE_MAXTASKS', '50'))
    pending = []
    for seed, shots in JOBS:
        tag = 'bloqade_gate6' if shots is None else f'bloqade_gate6_shots{shots}'
        cell = CELLS / f'fashionmnist_qrc_s{seed}_d-1_n10000_nt500_rydberg_{tag}.parquet'
        (pending.append((seed, shots, tag)) if not cell.exists()
         else print(f'SKIP s{seed} shots={shots} (on disk)', flush=True))

    print(f'{len(pending)} cells to run, ~5h each (~{5*len(pending)}h total)', flush=True)
    for i, (seed, shots, tag) in enumerate(pending, 1):
        extra = [] if shots is None else ['--shots', str(shots), '--vacancy-rate', '0.01']
        cmd = [sys.executable, '-u', '-m', 'experiments.qrc_fusion_fair_generation',
               *COMMON, *GEOM, '--seed', str(seed), '--tag', tag, *extra]
        print(f'\n[{i}/{len(pending)}] seed={seed} shots={shots}\n  $ {" ".join(cmd)}', flush=True)
        started = perf_counter()
        rc = subprocess.run(cmd, env=env).returncode
        print(f'[{i}/{len(pending)}] seed={seed} shots={shots} rc={rc} '
              f'({perf_counter() - started:.0f}s)', flush=True)
        if rc != 0:
            print(f'FAILED seed={seed} shots={shots} -- continuing', flush=True)
    print('\nbloqade replication queue drained', flush=True)


if __name__ == '__main__':
    main()
