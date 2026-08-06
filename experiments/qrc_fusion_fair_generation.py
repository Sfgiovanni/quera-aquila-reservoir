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
                     inception_features_and_probs, inception_score)

OUT = Path('results/qrc_fusion_fair')
PRDC_SUBSAMPLE = 1000


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
    cell = OUT / 'generation_cells' / f'{a.dataset}_{a.method}_s{a.seed}_d{a.draw}_n{a.samples}_nt{a.n_train}.parquet'
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
        q, qvv = (qrc_features(x, t, alpha_bar, a.draw, device),
                  qrc_features(xv, tv, alpha_bar, a.draw, device))
        model, val, lam = fit_readout(c, y, cv, yv, q, qvv)
        floored = model.extra.n_floored
        draw_kw = dict(draw=a.draw)
        raw_check = np.column_stack([x[:256], q[:256], c[:256, -10:]])
    elif a.method == 'random':
        mapper = random_map(c, 84, ROOT + 50000 + 100 * a.seed + a.draw)
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

    # Inception Score over the Fashion-MNIST domain classifier. BreastMNIST is excluded on purpose:
    # it has no domain classifier and ImageNet classes carry no meaning for ultrasound.
    if a.dataset == 'fashionmnist' and Path('results/fmnist_classifier.pt').exists():
        is_mean, is_sd = inception_score(
            fmnist_probs(images, 'results/fmnist_classifier.pt',
                         batch_size=a.inception_batch, device=device))
    else:
        is_mean = is_sd = np.nan

    rng = np.random.default_rng(0)
    center = real.mean(0)
    basis = np.linalg.svd(real[rng.choice(len(real), min(1000, len(real)), False)] - center,
                          full_matrices=False)[2][:32]
    idx_r = rng.choice(len(real), min(PRDC_SUBSAMPLE, len(real)), False)
    idx_g = rng.choice(len(gf), min(PRDC_SUBSAMPLE, len(gf)), False)
    p, r_, d_, cov = prdc((real[idx_r] - center) @ basis.T, (gf[idx_g] - center) @ basis.T, k=5)
    half = len(real) // 2
    floor = fid_null_floor(real[:half], real[half:], a.samples) if a.dataset == 'breastmnist' else np.nan

    row = dict(dataset=a.dataset, method=a.method, data_seed=a.seed, unitary_draw=a.draw,
               n_samples=a.samples, n_train=a.n_train, val_mse=val, lambda_=lam,
               n_floored_columns=floored, fid=frechet_distance_lowrank(real, gf),
               inception_score=is_mean, inception_score_sd=is_sd,
               fid_null_floor=floor, latent_energy=energy(ztr * sigma, enc, size=500),
               precision=p, recall=r_, density=d_, coverage=cov,
               probe_mean_abs_z=probe_mean, probe_max_abs_z=probe_max,
               readout_max_diff=readout_max_diff, fit_seconds=fit_s, generation_seconds=gen_s,
               checkpoint=ck, **guard)
    pd.DataFrame([row]).to_parquet(cell, index=False)
    print(f"DONE {a.dataset} {a.method} s={a.seed} d={a.draw} nt={a.n_train} lam={lam:g} "
          f"fid={row['fid']:.3f} is={is_mean:.3f} div={row['diversity']:.4f} clip={row['frac_at_clip']:.3f} "
          f"probe_max={probe_max:.1f} floored={floored} "
          f"{'DEGENERATE' if row['degenerate'] else ''} gen={gen_s:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=('breastmnist', 'fashionmnist'), required=True)
    p.add_argument('--method', choices=('classical', 'qrc', 'random', 'interaction'), required=True)
    p.add_argument('--probe-limit', type=float, default=50.,
                   help='fail the cell if standardized rollout features exceed this many sigma')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--draw', type=int, default=-1)
    p.add_argument('--n-train', type=int, default=500)
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--inception-batch', type=int, default=96,
                   help='keep low enough that PAR workers fit in GPU memory together')
    p.add_argument('--device', default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
