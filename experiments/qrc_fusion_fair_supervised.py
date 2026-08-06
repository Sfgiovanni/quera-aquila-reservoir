"""Stage 1: bias-corrected supervised comparison, and the gate on the FID grid.

Three arms, all sharing the same pairs, standardization, lambda grid and splits:
  ridge_classical   [x_t, te]                      20 features
  ridge_qrc         [x_t, h_QRC(84), te]          104 features
  ridge_random      [x_t, random_tanh(84), te]    104 features   <- dimension-matched control

Swept over n_train, because n=500 with 104 features is the leading candidate explanation for the
published +0.0146 gap: the classical model is NESTED in the hybrid, so with enough data and a
properly tuned lambda the hybrid cannot lose. If the gap closes as n grows, the published negative
was sample-starved. That decision selects the n_train the FID stage runs at.

`--clamp-train` is a separate arm, not the default: clamping x_t at fit time changes the supervised
problem and would break comparability with the published MSE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import (LAMBDAS, ROOT, balanced_pairs, fit_readout,
                                              interaction_extra, plain_design, qrc_features,
                                              random_map, set_variance_floor)

OUT = Path('results/qrc_fusion_fair')


def load_latents(dataset, device):
    """Disjoint splits. Fashion uses the 15k/5k correction the kernel experiment introduced."""
    from autoencoder import encode_numpy, load_autoencoder
    if dataset == 'breastmnist':
        from data import load_breast_mnist
        (tr, _), (va, _), (te, _) = load_breast_mnist()
        ck = 'checkpoints/breastmnist_sweep/B_dim10_hflip_600ep_seed0.pt'
        ae = load_autoencoder(ck, device)
        a, b, c = encode_numpy(ae, tr), encode_numpy(ae, va), encode_numpy(ae, te)
    else:
        from data import load_fashion_mnist
        tr, _, te, _ = load_fashion_mnist('data/fashion-mnist/raw')
        ck = 'checkpoints/autoencoder_d10.pt'
        ae = load_autoencoder(ck, device)
        z = encode_numpy(ae, tr[:20000])
        a, b, c = z[:15000], z[15000:], encode_numpy(ae, te)
    sigma = float(a.std())
    return a / sigma, b / sigma, c / sigma, ae, ck, sigma


def cell(dataset, seed, draw, n_train, ztr, zv, zt, alpha_bar, device, clamp_train):
    ds = ROOT + seed
    n_val = max(500, n_train // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, n_train, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    xt, yt, tt, _ = balanced_pairs(zt, alpha_bar, min(2000, 4 * len(zt)), ds + 200)
    c, cv, ct = plain_design(x, t), plain_design(xv, tv), plain_design(xt, tt)
    raw = lambda xx, e, cc: np.column_stack([xx, e, cc[:, -10:]])

    started = perf_counter()
    q, qv, qt = (qrc_features(x, t, alpha_bar, draw, device, clamp=clamp_train),
                 qrc_features(xv, tv, alpha_bar, draw, device, clamp=clamp_train),
                 qrc_features(xt, tt, alpha_bar, draw, device, clamp=clamp_train))
    extract_s = perf_counter() - started

    mapper = random_map(c, q.shape[1], ROOT + 50000 + 100 * seed + draw)
    r, rv, rt = mapper(c), mapper(cv), mapper(ct)
    i_, iv, it = interaction_extra(c), interaction_extra(cv), interaction_extra(ct)

    arms = {}
    base, bval, blam = fit_readout(c, y, cv, yv)
    arms['ridge_classical'] = (base, bval, blam, base.predict(ct), 20, 0)
    for name, (tr_e, va_e, te_e) in (('ridge_qrc', (q, qv, qt)), ('ridge_random', (r, rv, rt)),
                                     ('ridge_interaction', (i_, iv, it))):
        m, val, lam = fit_readout(c, y, cv, yv, tr_e, va_e)
        arms[name] = (m, val, lam, m.predict(raw(xt, te_e, ct)), 20 + tr_e.shape[1],
                      m.extra.n_floored)

    rows = []
    for name, (m, val, lam, pred, nf, floored) in arms.items():
        err = pred - yt
        rows.append(dict(dataset=dataset, data_seed=seed, unitary_draw=draw, n_train=n_train,
                         n_val=n_val, n_test=len(yt), clamp_train=clamp_train, method=name,
                         val_mse=val, test_mse=float(np.mean(err ** 2)),
                         test_mae=float(np.mean(np.abs(err))), lambda_=lam, n_features=nf,
                         lambda_at_ceiling=bool(lam >= max(LAMBDAS)), n_floored_columns=floored,
                         feature_extraction_s=extract_s if name == 'ridge_qrc' else 0.))
    return rows


def paired(frame, arm, metric):
    w = frame.pivot_table(index=['dataset', 'data_seed', 'unitary_draw', 'n_train'],
                          columns='method', values=metric)
    d = (w[arm] - w['ridge_classical']).dropna()
    if len(d) < 2:
        return None
    v = d.values
    ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
    return dict(arm=arm, metric=metric, n_pairs=int(len(v)), mean=float(v.mean()),
                sd=float(v.std(ddof=1)), ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                p_value=float(stats.ttest_1samp(v, 0.).pvalue),
                wins_vs_classical=int((v < 0).sum()))


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    device = a.device
    _, alpha_bar = cosine_schedule(200)
    set_variance_floor(a.variance_floor)
    tag = 'clamptrain' if a.clamp_train else 'main'
    if a.variance_floor != 1e-6:
        tag += f'_floor{a.variance_floor:g}'
    path = OUT / f'supervised_{tag}.parquet'
    rows = pd.read_parquet(path).to_dict('records') if path.exists() else []
    done = {(r['dataset'], r['data_seed'], r['unitary_draw'], r['n_train']) for r in rows}
    for dataset in a.datasets:
        ztr, zv, zt, _, ck, sigma = load_latents(dataset, device)
        print(f'{dataset}: train/val/test latents {len(ztr)}/{len(zv)}/{len(zt)} sigma={sigma:.5f}',
              flush=True)
        for n_train in a.n_train:
            for seed in range(a.seeds):
                for draw in range(a.draws):
                    if (dataset, seed, draw, n_train) in done:
                        continue
                    new = cell(dataset, seed, draw, n_train, ztr, zv, zt, alpha_bar, device,
                               a.clamp_train)
                    rows.extend(new)
                    pd.DataFrame(rows).to_parquet(path, index=False)
                    d = {r['method']: r['test_mse'] for r in new}
                    b = d['ridge_classical']
                    print(f"{dataset} n={n_train} s={seed} d={draw} classical={b:.5f} "
                          f"qrc={d['ridge_qrc']-b:+.5f} random={d['ridge_random']-b:+.5f} "
                          f"interaction={d['ridge_interaction']-b:+.5f} "
                          f"floored={new[1]['n_floored_columns']}", flush=True)

    frame = pd.DataFrame(rows)
    summary = []
    for dataset in frame.dataset.unique():
        for n_train in sorted(frame.n_train.unique()):
            sub = frame[(frame.dataset == dataset) & (frame.n_train == n_train)]
            for arm in ('ridge_qrc', 'ridge_random', 'ridge_interaction'):
                for metric in ('test_mse', 'val_mse'):
                    s = paired(sub, arm, metric)
                    if s:
                        summary.append(dict(dataset=dataset, n_train=int(n_train), **s))
    (OUT / f'supervised_{tag}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print()
    for s in summary:
        if s['metric'] == 'test_mse':
            print(f"{s['dataset']:14s} n_train={s['n_train']:<5d} {s['arm']:13s} "
                  f"delta_test_mse={s['mean']:+.5f} sd={s['sd']:.5f} "
                  f"ci95=[{s['ci95_low']:+.5f},{s['ci95_high']:+.5f}] p={s['p_value']:.3g} "
                  f"wins={s['wins_vs_classical']}/{s['n_pairs']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['fashionmnist', 'breastmnist'])
    p.add_argument('--n-train', nargs='+', type=int, default=[500, 2000, 5000])
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--draws', type=int, default=5)
    p.add_argument('--clamp-train', action='store_true')
    p.add_argument('--variance-floor', type=float, default=1e-6,
                   help='attribution ablation only; 1e-10 reproduces the published Standardizer')
    p.add_argument('--device', default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
