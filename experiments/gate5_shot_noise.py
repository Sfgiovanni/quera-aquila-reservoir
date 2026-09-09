"""Gate 5: shot noise end-to-end, at the Gate 4-selected operating point
(a=9.757um i.e. a/R_b=1.0, Omega=6.283, S=9.0, V=4, t_scale=1.0).

Sweeps shots in {50, 100, 300, 1000} (MAX_SHOTS=1000 is the hardware ceiling; larger values
would not be a single-task submission) x two arms:
  - `shots`: real multinomial shot noise (`quera.sampling`) with `vacancy_rate=0.01`, the task
    spec's own stated per-site atom-loss rate. `fp_rate`/`fn_rate` are left at 0 -- the spec
    gives no concrete detection-error rate, and the task's hard rule is to ask rather than
    invent hardware numbers, so false-positive/negative detection error is out of scope for
    this gate pending an explicit value.
  - `gaussian`: `quera.features.gaussian_noise_like_shots`, the same per-feature variance with
    none of real shot noise's within-probe-time structure -- Gate 5's control for "is S-
    dependence generic noise-as-regularizer, or does it need real measurement statistics."

Like Gate 4, this is a cheap val_mse diagnostic pass (n_train=200), not the selector on its
own: Gate 3/4 already showed val_mse can rank this reservoir backwards relative to FID, so the
"minimum S" this script finds is a *candidate*, confirmed with actual FID runs afterward
(see docs/AQUILA_PORT.md's Gate 5 section) before being reported as the answer.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import ROOT, balanced_pairs, fit_readout, plain_design, random_map
from experiments.qrc_fusion_fair_supervised import load_latents
from quera.device import MAX_SHOTS
from quera.encoding import fit_encoding
from quera.features import ReservoirParams, rydberg_features

OUT = Path('results/qrc_fusion_fair/gate5_sweep.json')
DEVICE = 'cuda'
N_TRAIN = 200
DATASET = 'fashionmnist'
SEED = 0
VACANCY_RATE = 0.01  # task spec's own stated per-site atom-loss rate

# Gate 4's selected operating point.
GEOM = dict(spacing_um=9.756844364507792, omega=6.283, s_detuning=9.0, v_slices=4, t_scale=1.0)

SHOT_COUNTS = (50, 100, 300, MAX_SHOTS)


def run_config(name, shots, noise_mode, ztr, zv, alpha_bar):
    started = perf_counter()
    ds = ROOT + SEED
    n_val = max(500, N_TRAIN // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, N_TRAIN, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    c, cv = plain_design(x, t), plain_design(xv, tv)

    encoding = fit_encoding(x, t, t_scale=GEOM['t_scale'])
    vacancy = VACANCY_RATE if (shots is not None and noise_mode == 'shots') else 0.0
    params = ReservoirParams(encoding=encoding, spacing_um=GEOM['spacing_um'], omega=GEOM['omega'],
                             s_detuning=GEOM['s_detuning'], v_slices=GEOM['v_slices'],
                             vacancy_rate=vacancy, noise_mode=noise_mode)
    q = rydberg_features(x, t, params, DEVICE, shots=shots)
    qv = rydberg_features(xv, tv, params, DEVICE, shots=shots)

    base, bval, _ = fit_readout(c, y, cv, yv)
    model, val, lam = fit_readout(c, y, cv, yv, q, qv)

    mapper = random_map(c, q.shape[1], ROOT + 90000)
    r, rv = mapper(c), mapper(cv)
    _, rval, _ = fit_readout(c, y, cv, yv, r, rv)

    elapsed = perf_counter() - started
    row = dict(name=name, shots=shots, noise_mode=noise_mode, vacancy_rate=vacancy,
              val_mse=val, val_mse_classical=bval, val_mse_random=rval,
              delta_vs_classical=val - bval, delta_vs_random=val - rval,
              lambda_=lam, n_floored=model.extra.n_floored, seconds=elapsed)
    print(f"{name:24s} val_mse={val:.5f} d_classical={row['delta_vs_classical']:+.5f} "
          f"d_random={row['delta_vs_random']:+.5f} floored={row['n_floored']} ({elapsed:.1f}s)", flush=True)
    return row


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ztr, zv, _, _, _, _ = load_latents(DATASET, DEVICE)
    _, alpha_bar = cosine_schedule(200)

    configs = [dict(name='exact', shots=None, noise_mode='shots')]
    for s in SHOT_COUNTS:
        configs.append(dict(name=f'shots={s}', shots=s, noise_mode='shots'))
        configs.append(dict(name=f'gaussian={s}', shots=s, noise_mode='gaussian'))

    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {r['name'] for r in rows}
    for cfg in configs:
        if cfg['name'] in done:
            print(f"SKIP {cfg['name']} (already done)", flush=True)
            continue
        row = run_config(cfg['name'], cfg['shots'], cfg['noise_mode'], ztr, zv, alpha_bar)
        rows.append(row)
        OUT.write_text(json.dumps(rows, indent=2) + '\n')

    print(f"\nwrote {len(rows)} rows to {OUT}", flush=True)


if __name__ == '__main__':
    main()
