"""Stage 3: paired generation statistics for the bias-corrected fusion comparison.

Pairing rule. The classical and interaction arms do not depend on the unitary draw, so a QRC cell is
paired against the same-(dataset, n_train, data_seed) baseline cell and the pair unit is
(data_seed, unitary_draw) -- the same rule the published report used. Degenerate cells are reported
separately and never averaged into a headline number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.time_injection import mde

OUT = Path('results/qrc_fusion_fair')
DRAWLESS = ('classical', 'interaction')


def load():
    cells = sorted((OUT / 'generation_cells').glob('*.parquet'))
    if not cells:
        raise SystemExit('no generation cells yet')
    return pd.concat([pd.read_parquet(c) for c in cells], ignore_index=True)


def paired(frame, arm, baseline, metric, lower_is_better=True):
    """Difference arm minus baseline, one row per (data_seed, unitary_draw)."""
    b = frame[frame.method == baseline].set_index('data_seed')[metric]
    a = frame[frame.method == arm]
    if arm in DRAWLESS:
        a = a.assign(unitary_draw=-1)
    d = a.assign(base=a.data_seed.map(b)).dropna(subset=[metric, 'base'])
    if len(d) < 2:
        return None
    v = (d[metric] - d.base).values
    ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v)) if len(v) > 1 else (np.nan, np.nan)
    wins = int((v < 0).sum() if lower_is_better else (v > 0).sum())
    return dict(arm=arm, baseline=baseline, metric=metric, n_pairs=int(len(v)),
                arm_mean=float(d[metric].mean()), baseline_mean=float(d.base.mean()),
                mean_difference=float(v.mean()), sd=float(v.std(ddof=1)),
                ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                p_value=float(stats.ttest_1samp(v, 0.).pvalue),
                mde=mde(float(v.std(ddof=1)), len(v)), wins=wins)


def run(a):
    frame = load()
    print(f'cells: {len(frame)}')
    deg = frame[frame.degenerate]
    print(f'degenerate cells: {len(deg)}' + (f' -> {deg.method.value_counts().to_dict()}' if len(deg) else ' (none)'))
    print(f'max probe |z| over all cells: {frame.probe_max_abs_z.max():.2f} sigma  '
          f'(published rollout reached 1.3e7)')
    print(f'max GPU-vs-numpy readout diff: {frame.readout_max_diff.max():.2e}\n')

    rows = []
    for (dataset, n_train, n_samples), sub in frame.groupby(['dataset', 'n_train', 'n_samples']):
        avail = set(sub.method)
        print(f'--- {dataset}  n_train={n_train}  n_generated={n_samples}')
        tab = (sub.groupby('method')
                  .agg(fid=('fid', 'mean'), fid_sd=('fid', 'std'), IS=('inception_score', 'mean'),
                       recall=('recall', 'mean'), diversity=('diversity', 'mean'),
                       clip=('frac_at_clip', 'mean'), cells=('fid', 'size'))
                  .sort_values('fid'))
        print(tab.to_string(float_format=lambda v: f'{v:.4f}'))
        for baseline in ('classical', 'interaction'):
            if baseline not in avail:
                continue
            for arm in ('qrc', 'random', 'interaction', 'classical'):
                if arm == baseline or arm not in avail:
                    continue
                for metric, lower in (('fid', True), ('inception_score', False)):
                    if sub[metric].isna().all():
                        continue
                    s = paired(sub, arm, baseline, metric, lower)
                    if s:
                        rows.append(dict(dataset=dataset, n_train=int(n_train),
                                         n_generated=int(n_samples), **s))
        print()

    if rows:
        out = pd.DataFrame(rows)
        out.to_parquet(OUT / 'generation_paired.parquet', index=False)
        (OUT / 'generation_paired.json').write_text(json.dumps(rows, indent=2, default=float) + '\n')
        key = out[(out.metric == 'fid') & (out.baseline.isin(('classical', 'interaction')))]
        print('=== paired FID differences (negative = arm better than baseline) ===')
        for _, r in key.iterrows():
            print(f"{r.dataset:13s} nt={r.n_train:<5d} {r.arm:12s} vs {r.baseline:12s} "
                  f"dFID={r.mean_difference:+9.3f} sd={r.sd:8.3f} "
                  f"ci95=[{r.ci95_low:+9.3f},{r.ci95_high:+9.3f}] p={r.p_value:.3g} "
                  f"mde={r.mde:7.3f} wins={r.wins}/{r.n_pairs}")


def main():
    p = argparse.ArgumentParser()
    run(p.parse_args())


if __name__ == '__main__':
    main()
