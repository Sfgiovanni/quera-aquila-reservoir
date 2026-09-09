"""Stage 2: bias-corrected FID. One process per cell so the GPU stays busy across arms.

Every correction from `QRC_FUSION_FID_BIAS_AUDIT.md` is live here: reservoir reset before each DDIM
step, variance floor on the observable block, lambda grid to 1e4, dimension-matched random control,
and a guard rail that flags a degenerate rollout instead of quietly reporting its FID.

Two assertions run before any sampling:
  * `GpuReadout.check` -- the device readout must agree with the numpy/sklearn path.
  * the probe trace -- with the reset in place, standardized observables must stay within a few
    sigma across all 50 steps. The published rollout hit 1.6e5 at step 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from autoencoder import decode_numpy
from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import (ROOT, GpuReadout, balanced_pairs, degenerate,
                                              fit_readout, gpu_interaction, gpu_random_map,
                                              interaction_extra, plain_design, qrc_features,
                                              random_map, rollout)
from experiments.qrc_fusion_fair_supervised import load_latents
from experiments.time_injection import energy, prdc
from metrics import (fid_null_floor, fmnist_probs, frechet_distance_lowrank,
                     inception_features_and_probs, inception_score, kernel_distance)

OUT = Path('results/qrc_fusion_fair')
PRDC_SUBSAMPLE = 1000


def qpu_telemetry(a) -> dict:
    """Hardware provenance for a `dwave-qpu` cell; empty for every simulated arm.

    A QPU result is not reproducible from its parameters alone -- what the chip actually did
    depends on which solver ran it, how the problem was embedded, and how often chains broke.
    `mean_chain_break_fraction` in particular is the number that separates "the physics gave this"
    from "the embedding did": a cell with high chain breaks is reporting majority-vote artefacts,
    not annealing, and must not be averaged into a headline FID without saying so.
    """
    if getattr(a, 'anneal_backend', None) != 'dwave-qpu':
        return {}
    from anneal.qpu_backend import get_reservoir
    report = get_reservoir().report()
    return dict(qpu_solver=getattr(getattr(get_reservoir().sampler, 'solver', None), 'name', ''),
                qpu_tiles=report['n_tiles'], qpu_submissions=report['submissions'],
                qpu_rows=report['rows'], qpu_access_time_s=report['qpu_access_time_s'],
                qpu_mean_chain_break_fraction=report['mean_chain_break_fraction'])


def reference(dataset, device):
    """Inception features of the real test images. Fashion reuses the cached 10k reference."""
    cache = OUT / f'reference_{dataset}.npz'
    if dataset == 'fashionmnist' and Path('results/magic_fid_reference_features_n10000.npz').exists():
        return np.load('results/magic_fid_reference_features_n10000.npz')['features']
    if cache.exists():
        return np.load(cache)['features']
    if dataset == 'breastmnist':
        from data import load_breast_mnist
        _, _, (test_x, _) = load_breast_mnist()
    else:
        from data import load_fashion_mnist
        _, _, test_x, _ = load_fashion_mnist('data/fashion-mnist/raw')
    f, _ = inception_features_and_probs(test_x, batch_size=256, device=device)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=f)
    return f


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '' if a.reservoir == 'digital' else f'_{a.reservoir}'
    if a.tag:
        suffix += f'_{a.tag}'
    stem = f'{a.dataset}_{a.method}_s{a.seed}_d{a.draw}_n{a.samples}_nt{a.n_train}{suffix}'
    cell = OUT / a.cells_dir / f'{stem}.parquet'
    cell.parent.mkdir(parents=True, exist_ok=True)
    if cell.exists():
        print(f'SKIP {cell.name}', flush=True)
        return

    device = a.device
    ztr, zv, _, ae, ck, sigma = load_latents(a.dataset, device)
    _, alpha_bar = cosine_schedule(200)
    x0_clip = .5 / sigma
    ds = ROOT + a.seed
    n_val = max(500, a.n_train // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, a.n_train, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    c, cv = plain_design(x, t), plain_design(xv, tv)
    noise = np.random.default_rng(ROOT + 999).normal(size=(a.samples, 10))

    draw_kw, extra_fn, floored = {}, None, 0
    started = perf_counter()
    if a.method == 'classical':
        model, val, lam = fit_readout(c, y, cv, yv)
        raw_check = np.column_stack([x[:256], c[:256, -10:]])
    elif a.method == 'qrc':
        reservoir_kwargs = {}
        if a.reservoir == 'rydberg':
            from quera.encoding import fit_encoding
            from quera.features import ReservoirParams
            reservoir_kwargs = dict(
                reservoir='rydberg', shots=a.shots,
                reservoir_params=ReservoirParams(
                    encoding=fit_encoding(x, t, t_scale=a.t_scale), spacing_um=a.spacing_um,
                    omega=a.omega, s_detuning=a.s_detuning, v_slices=a.v_slices,
                    rk4_safety=a.rk4_safety, vacancy_rate=a.vacancy_rate, fp_rate=a.fp_rate,
                    fn_rate=a.fn_rate, noise_mode=a.noise_mode, backend=a.backend,
                    bloqade_workers=a.bloqade_workers, observables=a.observables))
        elif a.reservoir == 'anneal':
            from anneal.features import AnnealParams
            from anneal.schedule import schedule_from_name
            from quera.encoding import fit_encoding
            if a.anneal_backend == 'dwave-qpu':
                # Build the reservoir up front so the tiling search, the solver handshake and the
                # budget check all happen before any fitting work -- an unaffordable cell should
                # fail in seconds, not after the training features are already paid for.
                from anneal.qpu_backend import QpuReservoir, set_reservoir
                from anneal.qpu_budget import QpuBudget
                budget = QpuBudget(Path(a.qpu_ledger), cap_seconds=a.qpu_cap_seconds,
                                   margin=a.qpu_margin)
                reservoir = QpuReservoir(n_spins=12, budget=budget, max_tiles=a.qpu_max_tiles,
                                         label=f'{a.dataset}-s{a.seed}-d{a.draw}')
                set_reservoir(reservoir)
                print(f"QPU {getattr(reservoir.sampler.solver, 'name', '?')}: "
                      f"{len(reservoir.tiles)} tiles, "
                      f"{budget.remaining_seconds():.1f}s of budget remaining", flush=True)
            reservoir_kwargs = dict(
                reservoir='anneal', shots=a.shots,
                reservoir_params=AnnealParams(
                    encoding=fit_encoding(x, t, t_scale=a.t_scale), n_spins=12,
                    draw=a.draw, t_anneal_us=tuple(a.anneal_times_us), h_scale=a.anneal_h_scale,
                    j_scale=a.anneal_j_scale, j_density=a.anneal_j_density, schedule=schedule_from_name(a.anneal_schedule), backend=a.anneal_backend, num_reads=a.anneal_reads, dephasing_rate=a.anneal_dephasing_rate, relaxation_rate=a.anneal_relaxation_rate, temperature_k=a.anneal_temperature_k, sweeps_per_us=a.anneal_sweeps_per_us, seed=a.seed))
        if a.reservoir == 'digital':
            reservoir_kwargs['encoding'] = a.encoding
            reservoir_kwargs['observables'] = a.observables
        q, qvv = (qrc_features(x, t, alpha_bar, a.draw, device, **reservoir_kwargs),
                  qrc_features(xv, tv, alpha_bar, a.draw, device, **reservoir_kwargs))
        model, val, lam = fit_readout(c, y, cv, yv, q, qvv)
        floored = model.extra.n_floored
        draw_kw = (dict(**reservoir_kwargs) if a.reservoir in ('rydberg', 'anneal')
                   else dict(draw=a.draw, encoding=a.encoding, observables=a.observables))
        raw_check = np.column_stack([x[:256], q[:256], c[:256, -10:]])
    elif a.method == 'random':
        from quera.features import n_features
        analogue_width = (n_features(12) * len(a.anneal_times_us)
                          if a.reservoir == 'anneal' else n_features(12) * a.v_slices)
        mapper = random_map(c, model_width := (a.random_width or
                                              (analogue_width if a.reservoir in ('rydberg', 'anneal') else 84)),
                            ROOT + 50000 + 100 * a.seed + a.draw)
        r, rv = mapper(c), mapper(cv)
        model, val, lam = fit_readout(c, y, cv, yv, r, rv)
        floored = model.extra.n_floored
        extra_fn = gpu_random_map(mapper, device)
        raw_check = np.column_stack([x[:256], r[:256], c[:256, -10:]])
    else:
        i_, iv = interaction_extra(c), interaction_extra(cv)
        model, val, lam = fit_readout(c, y, cv, yv, i_, iv)
        floored = model.extra.n_floored
        extra_fn = gpu_interaction(device)
        raw_check = np.column_stack([x[:256], i_[:256], c[:256, -10:]])
    fit_s = perf_counter() - started

    gpu = GpuReadout(model, device)
    readout_max_diff = gpu.check(model, raw_check, device)

    started = perf_counter()
    latent, trace = rollout(noise, gpu, alpha_bar, x0_clip, device, extra_fn=extra_fn,
                            batch=a.batch, probe=True, reset_every_step=True, **draw_kw)
    gen_s = perf_counter() - started
    probe_max = max((mx for _, _, _, mx, _ in trace), default=0.)
    probe_mean = float(np.mean([m for _, _, m, _, _ in trace])) if trace else 0.
    # With the reset in place the rollout's observables must stay in the distribution the readout
    # was fitted on. The published run reached 1.3e7 at step 1; anything past a few tens of sigma
    # means the alignment failed and the FID below would be meaningless. Fail rather than record it.
    if probe_max > a.probe_limit:
        raise AssertionError(
            f'rollout features left the training distribution: max|z_extra|={probe_max:.3e} '
            f'> {a.probe_limit:g}. Trace (step,t,mean,max,max|x|): '
            + ' | '.join(f'{i}:{tv}:{m:.2f}:{mx:.2e}:{xm:.2f}' for i, tv, m, mx, xm in trace[:6]))

    guard = degenerate(latent, x0_clip)
    enc = latent * sigma
    real = reference(a.dataset, device)
    images = decode_numpy(ae, enc)
    # The reservoir's density matrices are done with; Inception needs the room. Without this a
    # second worker on the same GPU OOMs while loading its weights.
    import torch
    torch.cuda.empty_cache()
    gf, _ = inception_features_and_probs(images, batch_size=a.inception_batch, device=device)
    torch.cuda.empty_cache()
    # float32, matching what FID is computed from: the cache has to reproduce the headline metric
    # exactly, or it is not a substitute for re-running the cell.
    if a.feature_cache:
        fc = OUT / a.feature_cache
        fc.mkdir(parents=True, exist_ok=True)
        np.savez(fc / f'{stem}.npz', features=gf.astype(np.float32))
    kid, kid_sd = kernel_distance(real, gf) if a.kid else (np.nan, np.nan)

    # Inception Score over the Fashion-MNIST domain classifier. BreastMNIST is excluded on purpose:
    # it has no domain classifier and ImageNet classes carry no meaning for ultrasound.
    is_mean = is_sd = np.nan
    if a.dataset == 'fashionmnist' and Path('results/fmnist_classifier.pt').exists():
        try:
            is_mean, is_sd = inception_score(
                fmnist_probs(images, 'results/fmnist_classifier.pt',
                             batch_size=a.inception_batch, device=device))
        except ModuleNotFoundError as e:
            # Pre-existing repo scoping gap, not introduced by the quera/ port: the checkpoint
            # is present (README lists it as included) but its defining module
            # (`experiments.phase_f.FashionCNN`) is one of the phase0-phase5 files the README
            # says were deliberately left out. FID/PRDC below don't depend on this classifier
            # at all, so degrade to NaN (the same fallback already used when the checkpoint is
            # absent) rather than blocking the run over a metric this task never asked for.
            print(f'WARN: Inception Score unavailable ({e}); FID/PRDC unaffected.', flush=True)

    rng = np.random.default_rng(0)
    center = real.mean(0)
    basis = np.linalg.svd(real[rng.choice(len(real), min(1000, len(real)), False)] - center,
                          full_matrices=False)[2][:32]
    idx_r = rng.choice(len(real), min(PRDC_SUBSAMPLE, len(real)), False)
    idx_g = rng.choice(len(gf), min(PRDC_SUBSAMPLE, len(gf)), False)
    p, r_, d_, cov = prdc((real[idx_r] - center) @ basis.T, (gf[idx_g] - center) @ basis.T, k=5)
    half = len(real) // 2
    floor = fid_null_floor(real[:half], real[half:], a.samples) if a.dataset == 'breastmnist' else np.nan

    row = dict(dataset=a.dataset, method=a.method, reservoir=a.reservoir, data_seed=a.seed,
               unitary_draw=a.draw, n_samples=a.samples, n_train=a.n_train, val_mse=val, lambda_=lam,
               n_floored_columns=floored, fid=frechet_distance_lowrank(real, gf),
               kid=kid, kid_subset_sd=kid_sd,
               spacing_um=a.spacing_um, omega=a.omega, s_detuning=a.s_detuning,
               v_slices=a.v_slices, t_scale=a.t_scale, shots=a.shots, vacancy_rate=a.vacancy_rate,
               fp_rate=a.fp_rate, fn_rate=a.fn_rate, noise_mode=a.noise_mode, backend=a.backend,
               tag=a.tag, encoding=a.encoding, observables=a.observables,
               anneal_spins=12, anneal_times_us=json.dumps(a.anneal_times_us),
               anneal_h_scale=a.anneal_h_scale, anneal_j_scale=a.anneal_j_scale,
               anneal_j_density=a.anneal_j_density, anneal_backend=a.anneal_backend, anneal_reads=a.anneal_reads, anneal_dephasing_rate=a.anneal_dephasing_rate, anneal_relaxation_rate=a.anneal_relaxation_rate, anneal_temperature_k=a.anneal_temperature_k, anneal_sweeps_per_us=a.anneal_sweeps_per_us, anneal_schedule=a.anneal_schedule,
               inception_score=is_mean, inception_score_sd=is_sd,
               fid_null_floor=floor, latent_energy=energy(ztr * sigma, enc, size=500),
               precision=p, recall=r_, density=d_, coverage=cov,
               probe_mean_abs_z=probe_mean, probe_max_abs_z=probe_max,
               readout_max_diff=readout_max_diff, fit_seconds=fit_s, generation_seconds=gen_s,
               checkpoint=ck, **guard, **qpu_telemetry(a))
    pd.DataFrame([row]).to_parquet(cell, index=False)
    print(f"DONE {a.dataset} {a.method} s={a.seed} d={a.draw} nt={a.n_train} lam={lam:g} "
          f"fid={row['fid']:.3f} kid={kid * 1e3:.3f}e-3 is={is_mean:.3f} div={row['diversity']:.4f} clip={row['frac_at_clip']:.3f} "
          f"probe_max={probe_max:.1f} floored={floored} "
          f"{'DEGENERATE' if row['degenerate'] else ''} gen={gen_s:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=('breastmnist', 'fashionmnist'), required=True)
    p.add_argument('--method', choices=('classical', 'qrc', 'random', 'interaction'), required=True)
    p.add_argument('--reservoir', choices=('digital', 'rydberg', 'anneal'), default='digital')
    p.add_argument('--rk4-safety', type=float, default=0.05,
                   help='rydberg only: ||H||*dt bound per RK4 sub-step, see quera/emulator.py')
    p.add_argument('--spacing-um', type=float, default=8.0, help='rydberg only: cluster spacing a')
    p.add_argument('--omega', type=float, default=6.283, help='rydberg only: probe pulse Omega')
    p.add_argument('--s-detuning', type=float, default=9.0, help='rydberg only: probe pulse S')
    p.add_argument('--v-slices', type=int, default=4, help='rydberg only: number of probe times V')
    p.add_argument('--t-scale', type=float, default=1.0,
                   help='rydberg only: post-standardization scale on the timestep channels')
    p.add_argument('--shots', type=int, default=None,
                   help='rydberg only: finite shots per probe time (default: exact expectations)')
    p.add_argument('--vacancy-rate', type=float, default=0.0, help='rydberg only: per-site atom loss rate')
    p.add_argument('--fp-rate', type=float, default=0.0, help='rydberg only: false-positive detection rate')
    p.add_argument('--fn-rate', type=float, default=0.0, help='rydberg only: false-negative detection rate')
    p.add_argument('--backend', choices=('torch', 'bloqade'), default='torch',
                   help="rydberg only: 'torch' uses quera/emulator.py; 'bloqade' does the "
                        "Schrodinger solve with QuEra's bloqade-analog in the .venv-gate1 "
                        'subprocess (see quera/bloqade_backend.py)')
    p.add_argument('--bloqade-workers', type=int, default=16,
                   help='--backend bloqade only: CPU processes in the solver pool')
    p.add_argument('--noise-mode', choices=('shots', 'gaussian'), default='shots',
                   help='rydberg only, with --shots set: real shot noise vs. the variance-matched '
                        'Gaussian control (Gate 5)')
    p.add_argument('--anneal-times-us', type=float, nargs='+', default=[0.005, 0.02, 0.05, 0.1],
                   help='anneal only: physical durations in microseconds')
    p.add_argument('--anneal-backend', choices=('dwave-sqa', 'qutip-open', 'ocean-sa', 'dwave-qpu'), default='dwave-sqa')
    p.add_argument('--anneal-reads', type=int, default=100)
    p.add_argument('--anneal-dephasing-rate', type=float, default=0.0)
    p.add_argument('--anneal-relaxation-rate', type=float, default=0.0)
    p.add_argument('--anneal-temperature-k', type=float, default=0.018)
    p.add_argument('--anneal-sweeps-per-us', type=float, default=20000.0)
    p.add_argument('--anneal-schedule', choices=('advantage2-fast', 'advantage2-standard', 'linear'), default='advantage2-fast')
    p.add_argument('--qpu-cap-seconds', type=float, default=24 * 60.0,
                   help='hard ceiling on CUMULATIVE QPU access time across every run sharing '
                        '--qpu-ledger. The run raises BudgetExceeded rather than crossing it.')
    p.add_argument('--qpu-margin', type=float, default=0.10,
                   help='fraction of the cap held back and never spent, absorbing the gap '
                        'between the pre-submission estimate and what the solver bills')
    p.add_argument('--qpu-ledger', default='results/dwave_qpu_ledger.json')
    p.add_argument('--qpu-max-tiles', type=int, default=None,
                   help='cap parallel problem copies per submission (default: as many as fit)')
    p.add_argument('--anneal-h-scale', type=float, default=2.0,
                   help='anneal only: local-field scale, clipped to the D-Wave range')
    p.add_argument('--anneal-j-scale', type=float, default=0.5,
                   help='anneal only: standard deviation of frozen random couplings')
    p.add_argument('--anneal-j-density', type=float, default=1.0,
                   help='anneal only: fraction of nonzero frozen couplings')
    p.add_argument('--tag', default='', help='free-form output filename suffix, e.g. gate4 sweep configs')
    p.add_argument('--random-width', type=int, default=0,
                   help='random arm only: width of the tanh projection. 0 keeps the default that '
                        'matches the reservoir being controlled (84 digital, 312 rydberg); set it '
                        'explicitly to match a wider readout, e.g. 612 for --observables full.')
    p.add_argument('--observables', choices=('zz', 'full'), default='zz',
                   help="digital reservoir only: 'zz' is the study's 84 columns (<Z_q>, <Z_qZ_r> "
                        "per slice); 'full' is every weight-<=2 Pauli, 612 columns. Same dynamics, "
                        "same states -- only which operators are read off each slice.")
    p.add_argument('--encoding', choices=('quadrature', 'multibase'), default='quadrature',
                   help="digital reservoir only: how the latent enters the density matrix. "
                        "`denoiser_qrc` also implements 'single', deliberately NOT offered here: "
                        "it consumes 10 data qubits against this study's N_QUBITS=6, so it needs an "
                        "11-qubit register -- measured at 46.9 ms/sample against 0.16 (~6.5h per "
                        "cell) and it would change the arm's width too (264 features, not 84), "
                        "making it a capacity comparison rather than an encoding one.")
    p.add_argument('--probe-limit', type=float, default=50.,
                   help='fail the cell if standardized rollout features exceed this many sigma')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--draw', type=int, default=-1)
    p.add_argument('--n-train', type=int, default=500)
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--inception-batch', type=int, default=96,
                   help='keep low enough that PAR workers fit in GPU memory together')
    p.add_argument('--cells-dir', default='generation_cells',
                   help='subdirectory of results/qrc_fusion_fair/ the cell parquet lands in. Point '
                        'a re-run at a fresh directory to add columns without touching the '
                        'committed cells.')
    p.add_argument('--feature-cache', default='',
                   help='if set, also write the 10000x2048 Inception features of the generated '
                        'images to this subdirectory (float32, ~82MB/cell). Any further '
                        'distribution metric then costs seconds instead of a re-run.')
    p.add_argument('--kid', action='store_true',
                   help='also compute KID (polynomial-kernel MMD^2, 100 subsets of 1000). ~35s on '
                        'top of the cell; off by default so committed cells stay reproducible.')
    p.add_argument('--device', default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
