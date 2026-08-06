"""Paired sample sheets for the bias-corrected fusion comparison.

Four rows, and only two of them are paired with each other:

  real            real test images -- NOT in correspondence with anything below. These models are
                  unconditional, so a generated image has no matching original.
  autoencoder     reconstructions of exactly those real images. This is the ceiling: no sampler can
                  beat what the frozen decoder can represent.
  ridge           generated from initial noise i
  qrc + ridge     generated from the SAME initial noise i

Because both samplers consume `rng(ROOT+999)`, column i of the last two rows starts from an
identical latent, so they can be read against each other image by image.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import decode_numpy, encode_numpy
from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import (ROOT, GpuReadout, balanced_pairs, fit_readout,
                                              plain_design, qrc_features, rollout)
from experiments.qrc_fusion_fair_supervised import load_latents

OUT = Path('figures/qrc_fusion_fair')


def sheet(dataset, n_train, seed, draw, n_show, device):
    ztr, zv, _, ae, _, sigma = load_latents(dataset, device)
    _, alpha_bar = cosine_schedule(200)
    x0_clip = .5 / sigma
    ds = ROOT + seed
    n_val = max(500, n_train // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, n_train, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    c, cv = plain_design(x, t), plain_design(xv, tv)
    noise = np.random.default_rng(ROOT + 999).normal(size=(n_show, 10))

    base, _, base_lam = fit_readout(c, y, cv, yv)
    lat_base, _ = rollout(noise, GpuReadout(base, device), alpha_bar, x0_clip, device, batch=n_show)

    q, qvv = qrc_features(x, t, alpha_bar, draw, device), qrc_features(xv, tv, alpha_bar, draw, device)
    hyb, _, hyb_lam = fit_readout(c, y, cv, yv, q, qvv)
    lat_qrc, _ = rollout(noise, GpuReadout(hyb, device), alpha_bar, x0_clip, device, draw=draw,
                         batch=n_show)

    if dataset == 'breastmnist':
        from data import load_breast_mnist
        _, _, (test_x, _) = load_breast_mnist()
    else:
        from data import load_fashion_mnist
        _, _, test_x, _ = load_fashion_mnist('data/fashion-mnist/raw')
    real = test_x[np.random.default_rng(7).choice(len(test_x), n_show, replace=False)]
    recon = decode_numpy(ae, encode_numpy(ae, real))

    return [('real (teste)', real),
            ('autoencoder (teto)', recon),
            (f'ridge apenas  λ={base_lam:g}', decode_numpy(ae, lat_base * sigma)),
            (f'QRC + ridge  λ={hyb_lam:g}', decode_numpy(ae, lat_qrc * sigma))]


def draw_sheet(rows, title, path, n_show):
    fig, ax = plt.subplots(len(rows), n_show, figsize=(1.05 * n_show, 1.18 * len(rows)))
    for r, (label, images) in enumerate(rows):
        for k in range(n_show):
            a = ax[r, k]
            a.imshow(np.squeeze(images[k]), cmap='gray', vmin=-1, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(.4); s.set_color('0.75')
        ax[r, 0].set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8.5,
                            labelpad=8)
    fig.suptitle(title, fontsize=10, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}', flush=True)


def draw_grid(images, title, path, side):
    """One arm, side x side images, no labels -- for judging a single model on its own."""
    fig, ax = plt.subplots(side, side, figsize=(side * 1.05, side * 1.05))
    for i in range(side):
        for j in range(side):
            a = ax[i, j]
            a.imshow(np.squeeze(images[i * side + j]), cmap='gray', vmin=-1, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(.3); s.set_color('0.8')
    fig.suptitle(title, fontsize=11, y=.997)
    fig.subplots_adjust(wspace=.04, hspace=.04, top=.955, left=.01, right=.99, bottom=.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}', flush=True)


def grids(dataset, n_train, seed, draw, side, device):
    """Two side x side grids from the SAME initial noise, so position i,j corresponds between them."""
    rows = sheet(dataset, n_train, seed, draw, side * side, device)
    for label, fname in (('ridge apenas', 'ridge'), ('QRC + ridge', 'qrc')):
        img = next(im for lab, im in rows if lab.startswith(label))
        lam = next(lab for lab, _ in rows if lab.startswith(label)).split('λ=')[-1]
        draw_grid(img, f'{dataset} — {label} (λ={lam}) — n_train={n_train}, '
                       f'data_seed={seed}' + (f', unitary_draw={draw}' if fname == 'qrc' else ''),
                  OUT / f'grid{side}x{side}_{dataset}_{fname}_nt{n_train}_s{seed}_d{draw}.png', side)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['breastmnist', 'fashionmnist'])
    p.add_argument('--n-train', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--draw', type=int, default=0)
    p.add_argument('--n-show', type=int, default=12)
    p.add_argument('--grid', type=int, default=0,
                   help='emit one NxN grid per arm instead of the four-row comparison sheet')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.grid:
        for dataset in a.datasets:
            grids(dataset, a.n_train, a.seed, a.draw, a.grid, a.device)
        return
    for dataset in a.datasets:
        rows = sheet(dataset, a.n_train, a.seed, a.draw, a.n_show, a.device)
        title = (f'{dataset} — n_train={a.n_train}, data_seed={a.seed}, unitary_draw={a.draw}. '
                 'As duas últimas linhas partem do mesmo ruído inicial por coluna; '
                 'as duas primeiras não correspondem a elas.')
        draw_sheet(rows, title, OUT / f'samples_{dataset}_nt{a.n_train}_s{a.seed}_d{a.draw}.png',
                   a.n_show)


if __name__ == '__main__':
    main()
