"""Gate 7, part 2: pick the operating point by shot-noise SNR, not by exact-feature performance.

Gate 4 selected `a/R_b=1.0` by sweeping **exact** features. Nothing in that sweep could have
noticed shot-efficiency, because at `shots=None` there is none to notice -- and Gate 5 then found
the winner it picked does not survive `shots<=1000`. The knob Gate 4 could not see is in the
variance formula itself:

    sigma^2 = (1 - <O>^2) / S

**The noise on an observable depends on the observable's own value.** An operating point that
drives `<Z>` toward +/-1 is intrinsically cheaper to measure: a saturated observable costs no
shots at all, one sitting at `<O>=0` costs the most. Going from `<O>~0` to `<O>~0.7` cuts sigma by
~30%, and since Gate 5's measured FID law is `gap(S) = 283.5 * S^-0.521` -- an exponent that lands
on the `1/sqrt(S)` counting law Gate 2 measured independently -- a 30% cut in sigma is worth
roughly **2x the effective shots**, i.e. it buys the `S~1250` that Gate 5's extrapolation says
would tie `classical`, without leaving the single-task shot ceiling.

Polarization alone is *not* the objective, and that is the trap this script is built to avoid: a
config could in principle polarize its observables by evolving so little that every atom stays in
`|g>`, giving `<Z>=+1`, zero noise, and zero information. (An earlier version of this docstring
named `t_scale -> 0` as that degenerate case. That was wrong: `t_scale` is a post-standardization
multiplier on the two *timestep* channels of the encoding -- see `quera/encoding.py:41` -- and does
not shorten the evolution at all. Evolution duration is `t_max_us/v_slices`, which this sweep
varies only through `v_slices`. The measured `polar_t_scale=0.25` row bears the correction out: its
polarization is 0.129, identical to the baseline's.) What matters is the ratio of the feature's
informative variation across inputs to its shot noise:

    SNR_j = std_over_samples(<O_j>) / mean_over_samples(sqrt((1 - <O_j>^2)/S_eff))

Gate 3's recorded diagnostic puts the numerator at 0.065-0.158 and, at `S=1000`, the denominator
at ~0.032 -- a per-feature SNR of roughly 2:1 to 5:1 feeding a 312-column ridge. That number, not
val_mse, is what this pass maximizes.

**This is a diagnostic, not a selector** -- the same standing rule as Gates 4 and 5. Gate 3 showed
val_mse can rank this reservoir backwards against FID, and Gate 4 showed block correlation can do
the same one level down; SNR is a third cheap proxy and gets no more trust than the other two.
Any shortlist it produces needs an FID confirmation run before it is called a choice.

## Cost

Pass B (below) is **free**: it reuses the exact feature arrays Gate 5 already computed and cached
at this operating point (`quera/cache.py` is content-addressed on geometry, `h`, probe time and
`shots`, so re-issuing Gate 5's calls is a cache hit, not a recomputation). Pass A is free for the
11 configs Gate 4 already ran and ~280s of GPU per config beyond them, which is why new configs
are opt-in behind `--extra` rather than on by default: Gate 6's replication queue owns the GPU for
~3 days and this pass must not compete with it for no reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ddpm import cosine_schedule
from experiments.gate7_noise_readout import feature_noise_variance, fit_readout_noise_aware
from experiments.qrc_fusion_fair_core import ROOT, balanced_pairs, fit_readout, plain_design
from experiments.qrc_fusion_fair_supervised import load_latents
from quera.device import MAX_SHOTS, blockade_radius
from quera.encoding import fit_encoding
from quera.features import ReservoirParams, rydberg_features

OUT = Path('results/qrc_fusion_fair/gate7_sweep.json')
DEVICE = 'cuda'
N_TRAIN = 200
DATASET = 'fashionmnist'
SEED = 0
VACANCY_RATE = 0.01
N_SITES = 12

# Gate 4's selected point, byte-identical to gate5_shot_noise.GEOM so the cache keys line up.
GEOM = dict(spacing_um=9.756844364507792, omega=6.283, s_detuning=9.0, v_slices=4, t_scale=1.0)
SHOT_COUNTS = (50, 100, 300, MAX_SHOTS)


def snr_metrics(exact: np.ndarray, shots: int = MAX_SHOTS, vacancy_rate: float = VACANCY_RATE):
    """The shot-efficiency diagnostic: per-feature signal-to-shot-noise, plus what it is worth.

    `effective_shots_multiplier` translates an SNR ratio into the shot count it substitutes for.
    Because sigma ~ 1/sqrt(S), improving SNR by a factor k is worth k^2 shots -- that is the whole
    reason a modest polarization gain is interesting at a hard shot ceiling.
    """
    exact = np.asarray(exact, np.float64)
    signal = exact.std(axis=0)
    noise = np.sqrt(feature_noise_variance(exact, shots, vacancy_rate, N_SITES))
    snr = signal / np.clip(noise, 1e-12, None)
    return dict(snr_mean=float(snr.mean()), snr_median=float(np.median(snr)),
                snr_p10=float(np.percentile(snr, 10)),
                polarization=float(np.abs(exact).mean()),
                signal_std_mean=float(signal.mean()), noise_std_mean=float(noise.mean()),
                frac_snr_below_2=float((snr < 2).mean()))


def features_for(cfg, ztr, zv, alpha_bar, shots=None):
    ds = ROOT + SEED
    n_val = max(500, N_TRAIN // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, N_TRAIN, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    encoding = fit_encoding(x, t, t_scale=cfg['t_scale'])
    params = ReservoirParams(encoding=encoding, spacing_um=cfg['spacing_um'], omega=cfg['omega'],
                             s_detuning=cfg['s_detuning'], v_slices=cfg['v_slices'],
                             vacancy_rate=VACANCY_RATE if shots else 0.0)
    q = rydberg_features(x, t, params, DEVICE, shots=shots)
    qv = rydberg_features(xv, tv, params, DEVICE, shots=shots)
    return (plain_design(x, t), y, plain_design(xv, tv), yv, q, qv)


def pass_b(ztr, zv, alpha_bar):
    """Does the noise-aware readout recover any of what Gate 5 lost? Cached features only."""
    rows = []
    for shots in (None, *SHOT_COUNTS):
        started = perf_counter()
        c, y, cv, yv, q, qv = features_for(GEOM, ztr, zv, alpha_bar, shots=shots)
        _, plain_val, plain_lam = fit_readout(c, y, cv, yv, q, qv)
        _, aware_val, aware_lam, tau = fit_readout_noise_aware(
            c, y, cv, yv, q, qv, shots=shots, vacancy_rate=VACANCY_RATE if shots else 0.0)
        row = dict(shots=shots, val_mse_plain=plain_val, val_mse_noise_aware=aware_val,
                   improvement=plain_val - aware_val, lambda_plain=plain_lam,
                   lambda_aware=aware_lam, tau=tau, seconds=perf_counter() - started,
                   **snr_metrics(q, shots or MAX_SHOTS, VACANCY_RATE if shots else 0.0))
        print(f"  shots={str(shots):5s} plain={plain_val:.5f} aware={aware_val:.5f} "
              f"delta={row['improvement']:+.5f} tau={tau:.2f} snr_med={row['snr_median']:.2f}",
              flush=True)
        rows.append(row)
    return rows


def candidate_configs(extra: bool):
    """Gate 4's 11 configs (already cached) plus, with --extra, points chosen to raise
    polarization: stronger detuning drives population toward a definite Rydberg/ground state,
    which is where sigma^2=(1-<O>^2)/S is small."""
    cfgs = [dict(name='gate4_selected', **GEOM)]
    base8 = dict(GEOM, spacing_um=8.0)
    cfgs.append(dict(name='gate03_baseline', **base8))
    rb = blockade_radius(GEOM['omega'])
    for ratio in (0.7,):
        cfgs.append(dict(name=f'ratio={ratio}', **dict(base8, spacing_um=ratio * rb)))
    for om in (3.14, 12.0):
        cfgs.append(dict(name=f'omega={om}', **dict(base8, omega=om,
                                                    spacing_um=(8.0 / rb) * blockade_radius(om))))
    for s in (4.5, 18.0):
        cfgs.append(dict(name=f's_detuning={s}', **dict(base8, s_detuning=s)))
    for v in (2, 6):
        cfgs.append(dict(name=f'v_slices={v}', **dict(base8, v_slices=v)))
    for ts in (0.5, 2.0):
        cfgs.append(dict(name=f't_scale={ts}', **dict(base8, t_scale=ts)))
    if extra:
        for s in (27.0, 36.0):
            cfgs.append(dict(name=f'polar_s={s}', **dict(GEOM, s_detuning=s)))
        cfgs.append(dict(name='polar_s=18_at_gate4', **dict(GEOM, s_detuning=18.0)))
        cfgs.append(dict(name='polar_t_scale=0.25', **dict(GEOM, t_scale=0.25)))
    return cfgs


def pass_a(cfgs, ztr, zv, alpha_bar, blob):
    """Incremental saves write the *whole* blob, never the bare row list: this pass takes
    ~280s per uncached config, so a kill mid-sweep must not cost Pass B's results as well."""
    rows = blob.setdefault('pass_a', [])
    done = {r['name'] for r in rows}
    for cfg in cfgs:
        if cfg['name'] in done:
            print(f"  SKIP {cfg['name']} (already done)", flush=True)
            continue
        started = perf_counter()
        name = cfg['name']
        c, y, cv, yv, q, qv = features_for(cfg, ztr, zv, alpha_bar, shots=None)
        _, val, lam = fit_readout(c, y, cv, yv, q, qv)
        row = dict(name=name, val_mse_exact=val, lambda_=lam,
                   seconds=perf_counter() - started,
                   **{k: cfg[k] for k in GEOM}, **snr_metrics(q))
        print(f"  {name:22s} snr_med={row['snr_median']:5.2f} pol={row['polarization']:.3f} "
              f"below2={row['frac_snr_below_2']:.2f} val_mse={val:.5f} ({row['seconds']:.0f}s)",
              flush=True)
        rows.append(row)
        OUT.write_text(json.dumps(blob, indent=2) + '\n')
    return rows


def main():
    global DEVICE
    p = argparse.ArgumentParser()
    p.add_argument('--extra', action='store_true',
                   help='also run polarization-targeted configs Gate 4 never tried (GPU cost: '
                        '~280s each; Gate 6 owns the GPU by default)')
    p.add_argument('--skip-pass-a', action='store_true')
    p.add_argument('--skip-pass-b', action='store_true')
    p.add_argument('--only', default='',
                   help='comma-separated config-name prefixes; run only those in Pass A. Lets a '
                        'decision-relevant subset jump the queue when the GPU is contended, '
                        'without re-running what is already on disk.')
    p.add_argument('--device', default=DEVICE)
    a = p.parse_args()
    DEVICE = a.device

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ztr, zv, _, _, _, _ = load_latents(DATASET, DEVICE)
    _, alpha_bar = cosine_schedule(200)
    blob = json.loads(OUT.read_text()) if OUT.exists() else {}
    if isinstance(blob, list):  # pre-fix runs wrote the bare pass_a list; keep their rows
        blob = {'pass_a': blob}

    if not a.skip_pass_b:
        print('Pass B: noise-aware readout vs plain, at the Gate 4 operating point')
        blob['pass_b'] = pass_b(ztr, zv, alpha_bar)
        OUT.write_text(json.dumps(blob, indent=2) + '\n')

    if not a.skip_pass_a:
        print('\nPass A: shot-noise SNR across operating points')
        cfgs = candidate_configs(a.extra)
        if a.only:
            wanted = tuple(w.strip() for w in a.only.split(',') if w.strip())
            cfgs = [c for c in cfgs if c['name'].startswith(wanted)]
            print(f"Pass A restricted to {len(cfgs)} config(s): "
                  f"{', '.join(c['name'] for c in cfgs)}")
        blob['pass_a'] = pass_a(cfgs, ztr, zv, alpha_bar, blob)
        OUT.write_text(json.dumps(blob, indent=2) + '\n')
    print(f'\nwrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
