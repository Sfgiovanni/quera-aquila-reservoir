"""Gate 6 aggregation: per-seed deltas, seed-to-seed spread, and the FID null floor.

Reads every `samples=10000`, `n_train=500`, `fashionmnist` generation cell on disk -- the
committed digital arms plus whatever `gate6_replication.py` has finished -- and reports the
comparison the way this port's own rules require:

  - **Per-seed deltas and their mean/SD, not a headline p-value.** `docs/AQUILA_PORT.md`'s
    wiring section fixed this rule up front: the rydberg arm has no `draw` ensemble, so it gets
    exactly one cell per (seed, n_train) -- 3 independent pairs, not the digital arm's 15
    correlated ones. A p-value from n=3 is not comparable to the audit's tables, so none is
    printed. What n=3 *can* support is a paired delta per seed plus the spread across seeds,
    which is what decides whether the Gate 4/5 single-point gaps were real.
  - **Arms with a draw ensemble are collapsed to a per-seed mean first.** Averaging all 15
    digital `qrc` cells directly would weight seeds by draw count and understate the
    seed-to-seed spread that the pairing is about.
  - **The FID null floor**, `metrics.fid_null_floor` -- what a *perfect* generator scores against
    a finite real reference. Reported with an explicit caveat about how it is constructed here
    (see `null_floor` below); it bounds how small a gap could possibly mean anything.

`PRDC_SUBSAMPLE=1000` in `qrc_fusion_fair_generation.py` caps precision/recall/density/coverage
at 1000 generated samples regardless of `--samples`, so the PRDC columns below are *not*
full-scale in the way FID is. That constant is left alone (changing it would break comparability
with every cell already committed); it is stated here so the two metric families are not read as
having the same sample backing.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import fid_null_floor

OUT = Path('results/qrc_fusion_fair')
CELLS = OUT / 'generation_cells'
DATASET = 'fashionmnist'
N_TRAIN = 500
SAMPLES = 10000
METRICS = ('fid', 'precision', 'recall', 'density', 'coverage', 'diversity')

# Every arm Gate 6 compares, as (label, filename glob). The rydberg arms carry Gate 6's own tags;
# the digital arms are the committed cells from the original fair-comparison study.
ARMS = {
    'rydberg exact':      f'{DATASET}_qrc_s*_d-1_n{SAMPLES}_nt{N_TRAIN}_rydberg_gate6.parquet',
    'rydberg shots=1000': f'{DATASET}_qrc_s*_d-1_n{SAMPLES}_nt{N_TRAIN}_rydberg_gate6_shots1000.parquet',
    'classical':          f'{DATASET}_classical_s*_d-1_n{SAMPLES}_nt{N_TRAIN}.parquet',
    'interaction':        f'{DATASET}_interaction_s*_d-1_n{SAMPLES}_nt{N_TRAIN}.parquet',
    'qrc digital':        f'{DATASET}_qrc_s*_d[0-9]_n{SAMPLES}_nt{N_TRAIN}.parquet',
    'random 84d':         f'{DATASET}_random_s*_d[0-9]_n{SAMPLES}_nt{N_TRAIN}.parquet',
    'random 312d':        f'{DATASET}_random_s*_d[0-9]_n{SAMPLES}_nt{N_TRAIN}_rydberg_gate6.parquet',
}

REFERENCE_ARM = 'classical'
CONTRASTS = ('rydberg exact', 'rydberg shots=1000')


def load():
    rows = []
    for label, pattern in ARMS.items():
        for f in sorted(glob.glob(str(CELLS / pattern))):
            df = pd.read_parquet(f)
            df['arm'] = label
            rows.append(df)
    if not rows:
        raise SystemExit('no Gate 6 cells found')
    return pd.concat(rows, ignore_index=True)


def per_seed(df):
    """One row per (arm, seed). Draw ensembles are averaged within a seed first, so that every
    arm contributes the same number of independent units to the seed-level statistics."""
    g = df.groupby(['arm', 'data_seed'], as_index=False)
    out = g[list(METRICS)].mean()
    out['n_cells'] = g.size()['size'].values
    return out


def null_floor(device='cuda'):
    """FID between two disjoint halves of the real reference, at n_generated=SAMPLES.

    Construction caveat, stated rather than buried: the reference set has exactly `SAMPLES`
    images, so a floor "at n=10000 against a 10000-image reference" cannot be built from disjoint
    halves -- this splits 5000/5000 and bootstraps `SAMPLES` draws from the second half. Both
    departures (a half-size reference, and resampling with replacement) push plug-in FID *up*,
    so this is a conservative floor: the true floor for the reported cells is no larger than
    this. It is used as an order-of-magnitude bound on "how small a gap could still be real",
    not as a number to subtract from the FIDs.
    """
    from experiments.qrc_fusion_fair_generation import reference
    real = reference(DATASET, device)
    half = len(real) // 2
    return fid_null_floor(real[:half], real[half:], SAMPLES)


def main():
    df = load()
    df = df[(df.n_train == N_TRAIN) & (df.n_samples == SAMPLES)]
    assert not df.degenerate.any(), f'degenerate cells present: {df[df.degenerate].arm.tolist()}'
    seeds = per_seed(df)

    print('=== per-seed cells (draw ensembles averaged within seed) ===')
    print(seeds.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print('\n=== arm summary over seeds (mean +/- SD across the 3 data seeds) ===')
    lines = []
    for arm in ARMS:
        s = seeds[seeds.arm == arm]
        if s.empty:
            print(f'{arm:20s} -- not yet on disk')
            continue
        sd = s.fid.std(ddof=1) if len(s) > 1 else np.nan
        print(f'{arm:20s} n_seeds={len(s)}  fid={s.fid.mean():7.3f} +/- {sd:5.3f}  '
              f'P={s.precision.mean():.3f} R={s.recall.mean():.3f} '
              f'D={s.density.mean():.3f} C={s.coverage.mean():.3f}')
        lines.append(dict(arm=arm, n_seeds=len(s), fid_mean=float(s.fid.mean()),
                          fid_sd=float(sd), **{m: float(s[m].mean()) for m in METRICS}))

    print(f'\n=== paired per-seed deltas vs {REFERENCE_ARM} (negative = better FID) ===')
    ref = seeds[seeds.arm == REFERENCE_ARM].set_index('data_seed')
    deltas = []
    for arm in CONTRASTS + ('interaction', 'qrc digital', 'random 312d', 'random 84d'):
        s = seeds[seeds.arm == arm].set_index('data_seed')
        common = sorted(set(s.index) & set(ref.index))
        if not common:
            print(f'{arm:20s} -- not yet on disk')
            continue
        d = np.array([s.fid[k] - ref.fid[k] for k in common])
        sd = d.std(ddof=1) if len(d) > 1 else np.nan
        print(f'{arm:20s} ' + '  '.join(f's{k}={v:+7.3f}' for k, v in zip(common, d))
              + f'   mean={d.mean():+7.3f} +/- {sd:5.3f}')
        deltas.append(dict(arm=arm, seeds=common, delta_fid=d.tolist(),
                           mean=float(d.mean()), sd=float(sd)))

    floor = null_floor()
    print(f"\n=== FID null floor (conservative; see null_floor docstring) ===\n"
          f"  {floor['fid_floor_mean']:.3f} +/- {floor['fid_floor_sd']:.3f} "
          f"over {floor['fid_floor_repeats']} resamples at n_generated={floor['fid_floor_n_generated']}")

    path = OUT / 'gate6_summary.json'
    path.write_text(json.dumps(dict(arms=lines, deltas_vs_classical=deltas, null_floor=floor,
                                    n_train=N_TRAIN, n_samples=SAMPLES, dataset=DATASET),
                               indent=2) + '\n')
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
