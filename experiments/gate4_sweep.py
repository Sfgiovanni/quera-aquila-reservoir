"""Gate 4: coordinate sweep over (a/R_b, Omega, S, V, t_scale), cheap enough to run in full.

Per the task spec, Gate 4 selects "por validacao, nao por teste" -- but Gate 3 (see
docs/AQUILA_PORT.md) already showed val_mse ranks this reservoir *below a random projection*
on a configuration whose FID beats classical, i.e. val_mse is not merely noisy here, it is
actively anti-correlated with the metric that matters. So this script does NOT select a
winner by val_mse. It is a diagnostic pass: one-factor-at-a-time from the Gate 0-3 baseline
(a=8um, Omega=6.283, S=9.0, V=4, t_scale=1.0), n_train=200 (val fixed at 500 per
qrc_fusion_fair_core's own n_val=max(500,n_train//2) rule) for speed, recording per config:
  - val_mse delta vs classical and vs the dimension-matched random control (the two numbers
    Gate 3 showed pulling in opposite directions from FID)
  - n_floored_columns (dead-feature diagnostic)
  - mean |corr| between probe-time blocks on the *train* design matrix -- the mechanism
    Gate 3's writeup pins the MSE damage on (four probe times of the same underlying
    trajectory are correlated; collapsing that redundancy is a physically motivated reason to
    prefer a config independent of what it does to val_mse)

Full factorial over 5 knobs x ~3 values would be a week of wall clock (each rydberg_features
call is 100+ RK4 solves); this sweep is intentionally not that. The output is a shortlist for
a handful of actual FID runs via qrc_fusion_fair_generation.py, not a final answer on its own.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import ROOT, balanced_pairs, fit_readout, plain_design, random_map
from experiments.qrc_fusion_fair_supervised import load_latents
from quera.device import blockade_radius
from quera.encoding import fit_encoding
from quera.features import ReservoirParams, rydberg_features

OUT = Path('results/qrc_fusion_fair/gate4_sweep.json')
DEVICE = 'cuda'
N_TRAIN = 200
DATASET = 'fashionmnist'
SEED = 0

BASELINE = dict(spacing_um=8.0, omega=6.283, s_detuning=9.0, v_slices=4, t_scale=1.0)


def block_correlation(q: np.ndarray, v_slices: int) -> float:
    """Mean |Pearson r| between same-column features at different probe times, averaged over
    all C(V,2) block pairs and all 78 columns -- the redundancy Gate 3 blames for the MSE
    damage from combining 4 probe times of the same trajectory."""
    n = q.shape[0]
    blocks = q.reshape(n, v_slices, -1)
    corrs = []
    for i in range(v_slices):
        for j in range(i + 1, v_slices):
            a, b = blocks[:, i, :], blocks[:, j, :]
            ac, bc = a - a.mean(0), b - b.mean(0)
            num = (ac * bc).sum(0)
            den = np.sqrt((ac**2).sum(0) * (bc**2).sum(0)) + 1e-12
            corrs.append(np.abs(num / den))
    return float(np.mean(corrs)) if corrs else float('nan')


def run_config(name, spacing_um, omega, s_detuning, v_slices, t_scale, ztr, zv, alpha_bar):
    started = perf_counter()
    ds = ROOT + SEED
    n_val = max(500, N_TRAIN // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, N_TRAIN, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    c, cv = plain_design(x, t), plain_design(xv, tv)

    encoding = fit_encoding(x, t, t_scale=t_scale)
    params = ReservoirParams(encoding=encoding, spacing_um=spacing_um, omega=omega,
                             s_detuning=s_detuning, v_slices=v_slices)
    q = rydberg_features(x, t, params, DEVICE)
    qv = rydberg_features(xv, tv, params, DEVICE)

    base, bval, _ = fit_readout(c, y, cv, yv)
    model, val, lam = fit_readout(c, y, cv, yv, q, qv)

    mapper = random_map(c, q.shape[1], ROOT + 90000)
    r, rv = mapper(c), mapper(cv)
    _, rval, _ = fit_readout(c, y, cv, yv, r, rv)

    corr = block_correlation(q, v_slices)
    elapsed = perf_counter() - started
    row = dict(name=name, spacing_um=spacing_um, omega=omega, s_detuning=s_detuning,
              v_slices=v_slices, t_scale=t_scale, val_mse=val, val_mse_classical=bval,
              val_mse_random=rval, delta_vs_classical=val - bval, delta_vs_random=val - rval,
              lambda_=lam, n_floored=model.extra.n_floored, block_corr=corr, seconds=elapsed)
    print(f"{name:22s} val_mse={val:.5f} d_classical={row['delta_vs_classical']:+.5f} "
          f"d_random={row['delta_vs_random']:+.5f} floored={row['n_floored']} "
          f"block_corr={corr:.4f} ({elapsed:.1f}s)", flush=True)
    return row


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ztr, zv, _, _, _, _ = load_latents(DATASET, DEVICE)
    _, alpha_bar = cosine_schedule(200)

    configs = [dict(name='baseline', **BASELINE)]
    rb_base = blockade_radius(BASELINE['omega'])
    ratio_base = BASELINE['spacing_um'] / rb_base
    for ratio in (0.7, 1.0):
        cfg = dict(BASELINE); cfg['spacing_um'] = ratio * rb_base
        configs.append(dict(name=f'ratio={ratio}', **cfg))
    for omega in (3.14, 12.0):
        cfg = dict(BASELINE); cfg['omega'] = omega
        cfg['spacing_um'] = ratio_base * blockade_radius(omega)  # hold a/R_b fixed
        configs.append(dict(name=f'omega={omega}', **cfg))
    for s in (4.5, 18.0):
        cfg = dict(BASELINE); cfg['s_detuning'] = s
        configs.append(dict(name=f's_detuning={s}', **cfg))
    for v in (2, 6):
        cfg = dict(BASELINE); cfg['v_slices'] = v
        configs.append(dict(name=f'v_slices={v}', **cfg))
    for ts in (0.5, 2.0):
        cfg = dict(BASELINE); cfg['t_scale'] = ts
        configs.append(dict(name=f't_scale={ts}', **cfg))

    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r['name'] for r in rows}
    for cfg in configs:
        if cfg['name'] in done:
            print(f"SKIP {cfg['name']} (already done)", flush=True)
            continue
        row = run_config(cfg['name'], cfg['spacing_um'], cfg['omega'], cfg['s_detuning'],
                         cfg['v_slices'], cfg['t_scale'], ztr, zv, alpha_bar)
        rows.append(row)
        OUT.write_text(json.dumps(rows, indent=2) + '\n')

    print(f"\nwrote {len(rows)} rows to {OUT}", flush=True)


if __name__ == '__main__':
    main()
