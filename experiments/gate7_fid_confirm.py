"""Gate 7 part 2: the FID confirmation run the sweep is not allowed to substitute for.

`gate7_snr_sweep.py` estimates that `s_detuning=27` at Gate 4's geometry lands ~1.78 FID below the
Gate 4 operating point at `shots=1000`, i.e. ~91.4 against `classical`'s 92.31. That estimate
chains **two extrapolations, each fitted on two points** -- Gate 5's `gap(S) = 283.5*S^-0.521`, and
a rate of 54 FID per unit val_mse calibrated from the two configurations Gate 4 confirmed with real
FID runs. The second is the weaker link: it was calibrated across a *geometry* change and is being
applied to a *detuning* change. Every gate in this port has required a real FID run before a cheap
proxy became a choice (Gate 3's val_mse reversal, Gate 4's block-correlation reversal); this is
that run.

Protocol is Gate 5's, unchanged, so the results drop straight into its table: `seed=0`,
`n_train=500`, `samples=500`, `rk4_safety=0.2`, `shots=1000`, `vacancy_rate=0.01`, Gate 4's
geometry. The comparison points already on disk are `qrc rydberg shots=1000` at **93.18**,
`classical` at **92.31**, `interaction` at **88.46**, and the Gate 4 exact cell at **85.40**.

## The three cells, in the order they run

1. `s27_shots1000` -- **the verdict**. Directly against the existing 93.18 cell, one knob changed.
2. `s27_exact` -- **the diagnosis**. The decomposition predicts `85.40 + 2.95 = ~88.35` here. If
   cell 1 disagrees with its prediction, this says which half of the chain broke: an on-prediction
   exact cell means the 54 FID/val_mse rate transferred and the noise-robustness law did not, and
   vice versa. Without it a failed cell 1 is uninterpretable.
3. `s9_ts025_shots1000` -- **the one axis still standing**. This slot originally held
   `S=27 + t_scale=0.25`, to ask whether the two axes add. Cell 1 answered that question by
   removing it: at **99.93** measured against 91.4 predicted, `S=27` is strongly negative, and
   stacking anything onto it tests nothing. `t_scale=0.25` at Gate 4's own `S=9` is what replaced
   it -- the only configuration in the sweep that improved exact-feature val_mse at no SNR cost,
   and one that no FID run has touched.

Ordered so that killing this driver after any cell leaves a coherent result rather than a
half-answered question, and each cell is skipped if its parquet already exists.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from time import perf_counter

CELLS = Path('results/qrc_fusion_fair/generation_cells')
GEOM = dict(spacing_um=9.756844364507792, omega=6.283, v_slices=4)
COMMON = ['--dataset', 'fashionmnist', '--method', 'qrc', '--reservoir', 'rydberg',
          '--seed', '0', '--draw', '-1', '--n-train', '500', '--samples', '500',
          '--rk4-safety', '0.2', '--device', 'cuda']

JOBS = [
    ('s27_shots1000', ['--s-detuning', '27.0', '--t-scale', '1.0',
                       '--shots', '1000', '--vacancy-rate', '0.01']),
    ('s27_exact', ['--s-detuning', '27.0', '--t-scale', '1.0']),
    # Originally `s27_ts025_shots1000` -- stacking t_scale onto S=27. Replaced after cell 1 came
    # in at **99.93** against a predicted 91.4: with S=27 measured as strongly negative, stacking a
    # second axis onto it answers nothing. `t_scale=0.25` at Gate 4's own S=9 is the live question
    # instead -- it was the one config in the sweep that improved exact-feature val_mse at no SNR
    # cost (net -0.41 FID, entirely on the exact side) and it has never been checked against FID.
    ('s9_ts025_shots1000', ['--s-detuning', '9.0', '--t-scale', '0.25',
                            '--shots', '1000', '--vacancy-rate', '0.01']),
]


def main():
    for i, (tag, extra) in enumerate(JOBS, 1):
        cell = CELLS / f'fashionmnist_qrc_s0_d-1_n500_nt500_rydberg_{tag}.parquet'
        if cell.exists():
            print(f'[{i}/{len(JOBS)}] SKIP {tag} (on disk)', flush=True)
            continue
        cmd = [sys.executable, '-m', 'experiments.qrc_fusion_fair_generation', *COMMON,
               '--spacing-um', repr(GEOM['spacing_um']), '--omega', repr(GEOM['omega']),
               '--v-slices', str(GEOM['v_slices']), '--tag', tag, *extra]
        print(f'\n[{i}/{len(JOBS)}] {tag}\n  $ {" ".join(cmd)}', flush=True)
        started = perf_counter()
        rc = subprocess.run(cmd).returncode
        print(f'[{i}/{len(JOBS)}] {tag} rc={rc} ({perf_counter() - started:.0f}s)', flush=True)
        if rc != 0:
            print(f'FAILED {tag} -- continuing', flush=True)
    print('\ngate7 FID confirmation queue drained', flush=True)


if __name__ == '__main__':
    main()
