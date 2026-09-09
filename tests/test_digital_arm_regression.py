"""The digital arm must keep reproducing bit-for-bit as `reservoir=` is added to
`qrc_fusion_fair_core.py`/`qrc_fusion_fair_supervised.py` -- this is the task's hard rule
("O braco digital tem que continuar reproduzindo bit a bit").

"Bit-for-bit" is checked against `tests/fixtures/digital_arm_fixture.npz`, generated on
*this* machine by `tests/_build_digital_fixture.py`, not against the parquets committed to
`results/qrc_fusion_fair/` -- those were produced on a different GPU/torch build (see
`README.md`/`QRC_FUSION_FID_BIAS_AUDIT.md`), and re-running the identical computation here
reproduces them only to ~5e-7 relative, not bit-for-bit (float32 matmul reductions are not
portable across BLAS/cuDNN versions). That ~5e-7 gap was measured once, directly, as the
cross-machine sanity check this file's docstring promises -- it is evidence the pipeline is
unchanged, not something this test re-verifies on every run (it would be flaky: this
machine's own repeated runs agree exactly, but nothing guarantees the committed parquet's
originating machine still exists to compare against). Cross-machine measurement, for the
fashionmnist/seed=0/draw=0/n_train=500 cell, `ridge_qrc` test_mse:
committed=0.5817008392756552, this machine=0.5817003376932807, diff=-5.02e-07.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ddpm import cosine_schedule
from experiments.qrc_fusion_fair_core import (ROOT, balanced_pairs, fit_readout, interaction_extra,
                                              plain_design, qrc_features, random_map)
from experiments.qrc_fusion_fair_supervised import load_latents

FIXTURE = Path(__file__).parent / "fixtures" / "digital_arm_fixture.npz"


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} missing -- run tests/_build_digital_fixture.py once per machine")
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def rerun(fixture):
    device = "cuda"
    dataset, seed, draw, n_train = (str(fixture["dataset"]), int(fixture["seed"]),
                                    int(fixture["draw"]), int(fixture["n_train"]))
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

    return dict(q=q, qv=qv, readout=readout, val_mse=val_mse, lam=lam,
               readout_random=readout_random, val_mse_random=val_mse_random, lam_random=lam_random,
               readout_interaction=readout_interaction, val_mse_interaction=val_mse_interaction,
               lam_interaction=lam_interaction,
               readout_classical=readout_classical, val_mse_classical=val_mse_classical,
               lam_classical=lam_classical)


def test_qrc_features_bit_exact(fixture, rerun):
    np.testing.assert_array_equal(rerun["q"], fixture["q_train"])
    np.testing.assert_array_equal(rerun["qv"], fixture["q_val"])


def test_qrc_readout_bit_exact(fixture, rerun):
    r = rerun["readout"]
    np.testing.assert_array_equal(r.model.coef_, fixture["qrc_coef"])
    np.testing.assert_array_equal(r.model.intercept_, fixture["qrc_intercept"])
    np.testing.assert_array_equal(r.classical.mean_, fixture["qrc_classical_mean"])
    np.testing.assert_array_equal(r.classical.scale_, fixture["qrc_classical_scale"])
    np.testing.assert_array_equal(r.extra.mean_, fixture["qrc_extra_mean"])
    np.testing.assert_array_equal(r.extra.scale_, fixture["qrc_extra_scale"])
    assert r.extra.n_floored == int(fixture["qrc_n_floored"])
    assert rerun["val_mse"] == float(fixture["qrc_val_mse"])
    assert rerun["lam"] == float(fixture["qrc_lambda"])


def test_classical_arm_bit_exact(fixture, rerun):
    r = rerun["readout_classical"]
    np.testing.assert_array_equal(r.model.coef_, fixture["classical_coef"])
    np.testing.assert_array_equal(r.model.intercept_, fixture["classical_intercept"])
    assert rerun["val_mse_classical"] == float(fixture["classical_val_mse"])
    assert rerun["lam_classical"] == float(fixture["classical_lambda"])


def test_random_arm_bit_exact(fixture, rerun):
    r = rerun["readout_random"]
    np.testing.assert_array_equal(r.model.coef_, fixture["random_coef"])
    np.testing.assert_array_equal(r.model.intercept_, fixture["random_intercept"])
    assert rerun["val_mse_random"] == float(fixture["random_val_mse"])
    assert rerun["lam_random"] == float(fixture["random_lambda"])


def test_interaction_arm_bit_exact(fixture, rerun):
    r = rerun["readout_interaction"]
    np.testing.assert_array_equal(r.model.coef_, fixture["interaction_coef"])
    np.testing.assert_array_equal(r.model.intercept_, fixture["interaction_intercept"])
    assert rerun["val_mse_interaction"] == float(fixture["interaction_val_mse"])
    assert rerun["lam_interaction"] == float(fixture["interaction_lambda"])
