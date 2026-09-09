"""`multibase` against the `quadrature` encoding the whole study used, digital reservoir only.

`denoiser_qrc._input_density` supports three encodings and every committed cell hardcoded
`quadrature`. The comparison matters because the reservoir's *only* nonlinearity is the tensor
product `rho_1 (x) ... (x) rho_n`, whose cross terms scale with the per-qubit Bloch radius squared
-- and the encodings differ precisely in that radius:

- `quadrature`: `tanh(x)/sqrt(2)` in pairs, `z` closes the sphere -> qubits on the surface.
- `multibase`:  `x` normalised to unit norm, `z = 0` -> qubits in the equatorial plane, and the
  latent's *magnitude* is discarded (only its direction enters).

Run at both n_train values, because this session established that n_train reorders arms: an
encoding that helps at 500 need not help at 15000.

## `single` is not in this grid

`data_qubits` returns `x.shape[1] = 10` for `single` against `5` for the other two, so it needs
`N_QUBITS >= 11` where the study uses 6 -- `reset_data` computes `dm = d // dd = 64 // 1024 = 0`
and the reshape fails. Raising the register to 11 changes the arm on two axes at once (264
features instead of 84, an 11-qubit register instead of 6), so it would not be an encoding
comparison. Cost is measured separately; see the docs section.
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


def jobs(n_trains, encoding, observables):
    tag = f'enc_{encoding}' + ('_full' if observables == 'full' else '')
    out = []
    for nt in n_trains:
        for s in SEEDS:
            for dr in range(5):
                out.append((f'{encoding}/{observables} s{s} d{dr} nt{nt}',
                            f'fashionmnist_qrc_s{s}_d{dr}_n10000_nt{nt}_{tag}.parquet',
                            ['--method', 'qrc', '--seed', str(s), '--draw', str(dr),
                             '--n-train', str(nt), '--encoding', encoding,
                             '--observables', observables, '--tag', tag]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoding', default='multibase')
    p.add_argument('--observables', choices=('zz', 'full'), default='zz')
    p.add_argument('--n-train', type=int, nargs='+', default=[500, 15000])
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    base = ['--dataset', 'fashionmnist', '--samples', '10000', '--device', a.device]

    pending = [(n, args) for n, cell, args in jobs(a.n_train, a.encoding, a.observables)
               if not (CELLS / cell).exists()]
    print(f'{a.encoding}/{a.observables}: {len(pending)} cells to run', flush=True)
    if a.dry_run:
        return
    for i, (name, args) in enumerate(pending, 1):
        cmd = [sys.executable, '-u', '-m', 'experiments.qrc_fusion_fair_generation', *base, *args]
        print(f'\n[{i}/{len(pending)}] {name}', flush=True)
        started = perf_counter()
        rc = subprocess.run(cmd, env=dict(os.environ)).returncode
        print(f'[{i}/{len(pending)}] {name} rc={rc} ({perf_counter() - started:.0f}s)', flush=True)
        if rc != 0:
            print(f'FAILED {name} -- continuing', flush=True)
    print(f'\n{a.encoding}/{a.observables} queue drained', flush=True)


if __name__ == '__main__':
    main()
