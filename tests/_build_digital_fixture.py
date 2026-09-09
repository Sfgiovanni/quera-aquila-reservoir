"""One-off generator for `tests/fixtures/digital_arm_fixture.npz` -- NOT a test itself.

Re-run this only if the digital arm's *intended* behavior changes (it should not, per the
task's hard rule: `experiments/qrc_fusion_fair_core.py` may only grow a `reservoir=`
parameter, never change the digital arm's numerics). `tests/test_digital_arm_regression.py`
reruns the same computation and asserts bit-for-bit equality against this file, on whatever
machine the test runs on -- see that file's docstring for why "bit-for-bit" is scoped to a
single machine/torch build, not claimed across machines.
"""
from __future__ import annotations

import numpy as np

from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import ROOT, balanced_pairs, fit_readout, interaction_extra, \
    plain_design, qrc_features, random_map
from experiments.qrc_fusion_fair_supervised import load_latents

OUT = "tests/fixtures/digital_arm_fixture.npz"


def main():
    device = "cuda"
    dataset, seed, draw, n_train = "fashionmnist", 0, 0, 500
    _, alpha_bar = cosine_schedule(200)
    ztr, zv, zt, ae, ck, sigma = load_latents(dataset, device)

    ds = ROOT + seed
    n_val = max(500, n_train // 2)
    x, y, t, _ = balanced_pairs(ztr, alpha_bar, n_train, ds)
    xv, yv, tv, _ = balanced_pairs(zv, alpha_bar, n_val, ds + 100)
    c, cv = plain_design(x, t), plain_design(xv, tv)

    q = qrc_features(x, t, alpha_bar, draw, device)
    qv = qrc_features(xv, tv, alpha_bar, draw, device)
    readout, val_mse, lam = fit_readout(c, y, cv, yv, q, qv)

    mapper = random_map(c, q.shape[1], ROOT + 50000 + 100 * seed + draw)
    r, rv = mapper(c), mapper(cv)
    readout_random, val_mse_random, lam_random = fit_readout(c, y, cv, yv, r, rv)

    i_, iv = interaction_extra(c), interaction_extra(cv)
    readout_interaction, val_mse_interaction, lam_interaction = fit_readout(c, y, cv, yv, i_, iv)

    readout_classical, val_mse_classical, lam_classical = fit_readout(c, y, cv, yv)

    np.savez(
        OUT,
        dataset=dataset, seed=seed, draw=draw, n_train=n_train,
        q_train=q, q_val=qv,
        qrc_coef=readout.model.coef_, qrc_intercept=readout.model.intercept_,
        qrc_classical_mean=readout.classical.mean_, qrc_classical_scale=readout.classical.scale_,
        qrc_extra_mean=readout.extra.mean_, qrc_extra_scale=readout.extra.scale_,
        qrc_n_floored=readout.extra.n_floored, qrc_val_mse=val_mse, qrc_lambda=lam,
        classical_coef=readout_classical.model.coef_, classical_intercept=readout_classical.model.intercept_,
        classical_val_mse=val_mse_classical, classical_lambda=lam_classical,
        random_coef=readout_random.model.coef_, random_intercept=readout_random.model.intercept_,
        random_val_mse=val_mse_random, random_lambda=lam_random,
        interaction_coef=readout_interaction.model.coef_, interaction_intercept=readout_interaction.model.intercept_,
        interaction_val_mse=val_mse_interaction, interaction_lambda=lam_interaction,
    )
    print(f"wrote {OUT}: q_train={q.shape} qrc_val_mse={val_mse!r} lambda={lam!r} "
         f"n_floored={readout.extra.n_floored}")


if __name__ == "__main__":
    main()
