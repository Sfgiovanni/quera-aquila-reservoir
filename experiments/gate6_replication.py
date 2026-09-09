"""Gate 6: full-scale (samples=10000) FID/PRDC replication over 3 data seeds, at the Gate 4
operating point (a=9.757um i.e. a/R_b=1.0, Omega=6.283, S=9.0, V=4, t_scale=1.0).

Gates 3-5 all rest on **one seed at samples=500** -- both their own writeups say so explicitly.
Gate 6 is the replication that turns those single points into numbers with a spread attached.
It re-measures two things at full scale and over seeds {0,1,2}:

  - `exact`   -- the Gate 4 headline (rydberg beats classical and interaction on FID)
  - `shots=1000` with `vacancy_rate=0.01` -- the Gate 5 headline (that gain does not survive
    the hardware shot ceiling)

Gate 6 is replication at the *selected* operating point, not more operating-point search:
`s_detuning=4.5` (docs/AQUILA_PORT.md's "strongest untested candidate") is deliberately out of
scope here -- adding it would be a new Gate 4 branch, not a replication of the existing one.

## What this script does and does not run

The digital arms at this exact cell definition (`fashionmnist`, `n_train=500`, `samples=10000`,
seeds 0/1/2) are **already committed** under `results/qrc_fusion_fair/generation_cells/` from the
original fair-comparison study -- `classical`, `interaction`, digital `qrc` (5 draws) and digital
`random` (5 draws, width 84). Re-running them would burn GPU hours to reproduce data that is
already in git, so this driver does not queue them; `gate6_analyze.py` reads them straight off
disk. Their cross-machine comparability was checked directly rather than assumed: re-running
`classical`/`seed=0` on this machine gave `fid=70.604` against the committed `70.769` (delta
0.165, versus a seed-to-seed SD of ~2.9 for that same arm), consistent with the float32 BLAS
difference `docs/AQUILA_PORT.md` already documents for the supervised numbers.

What the committed cells do *not* cover is a random control **matched to the rydberg arm's
width**: the digital `random` cells are 84-dim (the digital reservoir's width), while the
rydberg arm emits `n_features(12)*V = 312`. Gate 3's "beats random by a wide margin" claim is a
claim about a dimension-matched control, so this driver queues 312-dim random cells (15 of them,
3 seeds x 5 draws -- they cost seconds of generation each, so there is no reason to accept a
single-draw estimate of that arm's spread).

## Cost, and why the order is what it is

Measured on this machine at the Gate 4 geometry: 2343s of generation for 500 samples, i.e.
**4.69 s/sample**, so one rydberg cell at `samples=10000` is **~13 h** and the 6 rydberg cells
are **~78 h (~3.3 days)** of GPU. That is the real budget; there is no way to shrink it that
keeps the "full-scale" in Gate 6 (`rk4_safety` is already at Gates 3-5's relaxed 0.2, and
per-sample cost is flat in `--batch` past 256 -- see docs/AQUILA_PORT.md's performance-fix note).

So the queue is ordered by decision value, and every cell is written to its own parquet the
moment it finishes: `qrc_fusion_fair_generation.py` SKIPs any cell whose file already exists, so
killing this driver at any point and re-running it resumes rather than restarts. The order is
cheap-controls -> (seed 0 exact, seed 0 shots) -> seed 1 pair -> seed 2 pair, so that the run is
useful if stopped early: after ~26 h there is a complete full-scale seed-0 pair (a directly
comparable full-scale replacement for the Gate 4 and Gate 5 tables), and each later seed
completes a *within-seed* exact-vs-shots pair rather than leaving a half pair behind.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter

OUT = Path('results/qrc_fusion_fair/generation_cells')
DATASET = 'fashionmnist'
N_TRAIN = 500
SAMPLES = 10000
SEEDS = (0, 1, 2)

# Gate 4's selected operating point, byte-identical to gate5_shot_noise.GEOM.
GEOM = dict(spacing_um=9.756844364507792, omega=6.283, s_detuning=9.0, v_slices=4, t_scale=1.0)

# Gates 3-5's generation protocol. rk4_safety=0.2 (not the CLI default 0.05) is what every
# generation cell in this port has used; its ~2.2e-6 discretization error sits under Gate 1's own
# ~1e-5 cross-validation floor, so the relaxation trades away precision the references could not
# back up anyway. Changing it here would also break comparability with the Gate 4/5 tables.
RK4_SAFETY = 0.2
SHOTS = 1000          # device.MAX_SHOTS -- the single-task hardware ceiling
VACANCY_RATE = 0.01   # task spec's stated per-site atom-loss rate; shots arm only


def cell_path(method, seed, draw, tag, reservoir='rydberg'):
    suffix = '' if reservoir == 'digital' else f'_{reservoir}'
    if tag:
        suffix += f'_{tag}'
    return OUT / f'{DATASET}_{method}_s{seed}_d{draw}_n{SAMPLES}_nt{N_TRAIN}{suffix}.parquet'


def geom_args():
    return ['--spacing-um', repr(GEOM['spacing_um']), '--omega', repr(GEOM['omega']),
            '--s-detuning', repr(GEOM['s_detuning']), '--v-slices', str(GEOM['v_slices']),
            '--t-scale', repr(GEOM['t_scale'])]


def queue():
    """(name, method, seed, draw, tag, extra_args), cheapest-and-most-informative first."""
    jobs = []
    # 312-dim random control, matched to the rydberg arm's width. Seconds of generation each.
    for seed in SEEDS:
        for draw in range(5):
            jobs.append((f'random312 s{seed} d{draw}', 'random', seed, draw, 'gate6', []))
    # The expensive half: within-seed (exact, shots=1000) pairs, seed 0 first.
    for seed in SEEDS:
        jobs.append((f'rydberg exact s{seed}', 'qrc', seed, -1, 'gate6',
                     ['--rk4-safety', repr(RK4_SAFETY)]))
        jobs.append((f'rydberg shots={SHOTS} s{seed}', 'qrc', seed, -1, f'gate6_shots{SHOTS}',
                     ['--rk4-safety', repr(RK4_SAFETY), '--shots', str(SHOTS),
                      '--vacancy-rate', repr(VACANCY_RATE)]))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true', help='print the queue and exit')
    a = p.parse_args()

    jobs = queue()
    pending = [j for j in jobs if not cell_path(j[1], j[2], j[3], j[4]).exists()]
    print(f'{len(jobs)} cells queued, {len(jobs) - len(pending)} already on disk, '
          f'{len(pending)} to run', flush=True)
    if a.dry_run:
        for name, method, seed, draw, tag, extra in pending:
            print(f'  {name:28s} -> {cell_path(method, seed, draw, tag).name}')
        return

    for i, (name, method, seed, draw, tag, extra) in enumerate(pending, 1):
        cmd = [sys.executable, '-m', 'experiments.qrc_fusion_fair_generation',
               '--dataset', DATASET, '--method', method, '--reservoir', 'rydberg',
               '--seed', str(seed), '--draw', str(draw), '--n-train', str(N_TRAIN),
               '--samples', str(SAMPLES), '--tag', tag, '--device', a.device,
               *geom_args(), *extra]
        print(f'\n[{i}/{len(pending)}] {name}\n  $ {" ".join(cmd)}', flush=True)
        started = perf_counter()
        # One process per cell: the reservoir's statevector batches and Inception's weights do
        # not coexist comfortably in 11GB, and a crashed cell must not take the queue with it.
        rc = subprocess.run(cmd).returncode
        print(f'[{i}/{len(pending)}] {name} rc={rc} ({perf_counter() - started:.0f}s)', flush=True)
        if rc != 0:
            print(f'FAILED {name} -- continuing to the next cell', flush=True)

    print('\nqueue drained; run experiments/gate6_analyze.py', flush=True)


if __name__ == '__main__':
    main()
