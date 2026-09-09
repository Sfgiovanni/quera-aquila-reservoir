"""Seeds 3-5: doubling Gate 6's replication from 3 to 6 data seeds.

Gate 6 ran the 3 seeds the original fair-comparison study used, which is what the committed digital
cells cover. Three is enough to read *sign* consistency and nothing finer: at n=3 the SD is itself
~50% uncertain, which is why this port reports per-seed deltas rather than a p-value for the
Rydberg arm. The `shots=1000` result -- the one carrying the practical conclusion -- currently sits
at `+5.29 +/- 4.04`, a mean and a spread of the same order. Six seeds make that SD estimable.

**The Rydberg cells here use `--backend bloqade`**, QuEra's own emulator, for two reasons: it runs a
full-scale cell in ~7.5h against ~13h for `quera/emulator.py`, and it makes seeds 3-5 attributable
to the vendor's solver. Seeds 0-2 are being produced *both* ways, so the combined 6-seed arm is
all-bloqade while its first half also carries a torch cross-check (agreement measured at 0.0001 and
0.0013 FID, versus a 2.91 FID seed-to-seed SD).

## What this does not fix

`qrc_fusion_fair_generation.py:76` draws the generation noise from a fixed `default_rng(ROOT+999)`
shared by every arm and seed, so `--seed` varies the train/val draw and the readout fitted from it,
never the sampler's noise. Six seeds measure readout-fit variability six times; generation-noise
variability stays unsampled at n=1. That is kept deliberately (changing it breaks comparability
with every committed cell) and is a limit on what any number of seeds here can establish.

## Order

Digital backdrop first (~3 min per cell, all of it Inception -- the rollouts themselves take
seconds), because the paired deltas need `classical` and `interaction` at these seeds before any
Rydberg cell means anything. Then the `shots=1000` arm, which carries the main conclusion, then
the exact arm. Stopping after any block leaves a coherent result.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

CELLS = Path('results/qrc_fusion_fair/generation_cells')
SEEDS = (3, 4, 5)
BASE = ['--dataset', 'fashionmnist', '--n-train', '500', '--samples', '10000', '--device', 'cuda']
GEOM = ['--spacing-um', '9.756844364507792', '--omega', '6.283', '--s-detuning', '9.0',
        '--v-slices', '4', '--t-scale', '1.0']
RYD = ['--method', 'qrc', '--reservoir', 'rydberg', '--draw', '-1', '--rk4-safety', '0.2',
       '--backend', 'bloqade', '--bloqade-workers', '20', *GEOM]


def jobs():
    out = []
    for s in SEEDS:                                    # digital backdrop, minutes each
        out.append((f'classical s{s}', f'fashionmnist_classical_s{s}_d-1_n10000_nt500.parquet',
                    ['--method', 'classical', '--seed', str(s), '--draw', '-1']))
        out.append((f'interaction s{s}', f'fashionmnist_interaction_s{s}_d-1_n10000_nt500.parquet',
                    ['--method', 'interaction', '--seed', str(s), '--draw', '-1']))
        for dr in range(5):
            out.append((f'qrc digital s{s} d{dr}',
                        f'fashionmnist_qrc_s{s}_d{dr}_n10000_nt500.parquet',
                        ['--method', 'qrc', '--seed', str(s), '--draw', str(dr)]))
            out.append((f'random84 s{s} d{dr}',
                        f'fashionmnist_random_s{s}_d{dr}_n10000_nt500.parquet',
                        ['--method', 'random', '--seed', str(s), '--draw', str(dr)]))
            out.append((f'random312 s{s} d{dr}',
                        f'fashionmnist_random_s{s}_d{dr}_n10000_nt500_rydberg_gate6.parquet',
                        ['--method', 'random', '--reservoir', 'rydberg', '--seed', str(s),
                         '--draw', str(dr), '--tag', 'gate6', *GEOM]))
    for s in SEEDS:                                    # the arm carrying the conclusion
        out.append((f'rydberg shots=1000 s{s}',
                    f'fashionmnist_qrc_s{s}_d-1_n10000_nt500_rydberg_bloqade_gate6_shots1000.parquet',
                    [*RYD, '--seed', str(s), '--tag', 'bloqade_gate6_shots1000',
                     '--shots', '1000', '--vacancy-rate', '0.01']))
    for s in SEEDS:
        out.append((f'rydberg exact s{s}',
                    f'fashionmnist_qrc_s{s}_d-1_n10000_nt500_rydberg_bloqade_gate6.parquet',
                    [*RYD, '--seed', str(s), '--tag', 'bloqade_gate6']))
    return out


def main():
    env = dict(os.environ, BLOQADE_MAXTASKS=os.environ.get('BLOQADE_MAXTASKS', '50'))
    pending = [(n, a) for n, cell, a in jobs() if not (CELLS / cell).exists()]
    print(f'{len(pending)} cells to run '
          f'({sum("bloqade" in " ".join(a) for _, a in pending)} of them ~7.5h each)', flush=True)
    for i, (name, args) in enumerate(pending, 1):
        cmd = [sys.executable, '-u', '-m', 'experiments.qrc_fusion_fair_generation', *BASE, *args]
        print(f'\n[{i}/{len(pending)}] {name}', flush=True)
        started = perf_counter()
        rc = subprocess.run(cmd, env=env).returncode
        print(f'[{i}/{len(pending)}] {name} rc={rc} ({perf_counter() - started:.0f}s)', flush=True)
        if rc != 0:
            print(f'FAILED {name} -- continuing', flush=True)
    print('\nseeds 3-5 queue drained', flush=True)


if __name__ == '__main__':
    main()
