"""Gate 7 part 1: the noise-aware readout must generalize the existing one, not replace it.

The load-bearing test is `test_tau_zero_reproduces_plain_ridge`. The whole design argument for
`fit_readout_noise_aware` is that `tau=0` is in its grid and is exactly the readout Gate 5 used,
so validation selection can only pick the correction when it genuinely helps. If that identity
does not hold numerically, any Gate 7 "gain" could be an artifact of a differently-conditioned
solver rather than of the errors-in-variables correction, and the comparison would be worthless.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from experiments.gate7_noise_readout import (LinearModel, _solve, feature_noise_variance,
                                             fit_readout_noise_aware)
from experiments.qrc_fusion_fair_core import FloorStandardizer, fit_readout


def _data(seed=0, n=400, n_val=200, d_c=20, d_e=60, d_y=10):
    rng = np.random.default_rng(seed)
    ct, cv = rng.normal(size=(n, d_c)), rng.normal(size=(n_val, d_c))
    et, ev = rng.normal(scale=.3, size=(n, d_e)), rng.normal(scale=.3, size=(n_val, d_e))
    w_c, w_e = rng.normal(size=(d_c, d_y)), rng.normal(scale=.2, size=(d_e, d_y))
    f = lambda c, e: c @ w_c + e @ w_e + rng.normal(scale=.1, size=(len(c), d_y))
    return ct, f(ct, et), cv, f(cv, ev), et, ev


def test_tau_zero_reproduces_plain_ridge():
    """tau=0 must be the incumbent readout to float precision, not merely close to it."""
    ct, yt, cv, yv, et, ev = _data()
    cs, es = FloorStandardizer().fit(ct), FloorStandardizer().fit(et)
    dt = np.column_stack([cs.transform(ct), es.transform(et)])
    dv = np.column_stack([cs.transform(cv), es.transform(ev)])
    noise = np.zeros(dt.shape[1])
    for lam in (1e-2, 1., 100., 1e4):
        mine = _solve(dt, yt, noise, 0.0, lam)
        ref = Ridge(alpha=lam).fit(dt, yt)
        assert np.allclose(mine.coef_, ref.coef_, rtol=1e-7, atol=1e-9)
        assert np.allclose(mine.intercept_, ref.intercept_, rtol=1e-7, atol=1e-9)
        assert np.allclose(mine.predict(dv), ref.predict(dv), rtol=1e-7, atol=1e-9)


def test_shots_none_matches_fit_readout_exactly():
    """With no shot noise there is nothing to correct, so the whole gate must collapse onto
    `fit_readout`'s selected model -- same lambda, same validation MSE, tau=0."""
    ct, yt, cv, yv, et, ev = _data(seed=1)
    _, val_ref, lam_ref = fit_readout(ct, yt, cv, yv, et, ev)
    _, val, lam, tau = fit_readout_noise_aware(ct, yt, cv, yv, et, ev, shots=None)
    assert tau == 0.0
    assert lam == lam_ref
    assert val == pytest.approx(val_ref, rel=1e-9)


def test_noise_variance_follows_one_over_s_and_vacancy():
    """sigma^2 = (1-<O>^2)/S_eff, with S_eff the vacancy-filtered usable count."""
    rng = np.random.default_rng(0)
    f = np.clip(rng.normal(scale=.4, size=(2000, 30)), -1, 1)
    v1000 = feature_noise_variance(f, 1000)
    assert np.allclose(feature_noise_variance(f, 100), v1000 * 10, rtol=1e-12)
    # 12 sites at 1% loss -> 0.99**12 usable, the figure quera/sampling.py is tested against.
    got = feature_noise_variance(f, 1000, vacancy_rate=0.01)
    assert np.allclose(got, v1000 / 0.99 ** 12, rtol=1e-12)
    assert feature_noise_variance(f, None).tolist() == [0.0] * 30
    # A saturated observable (<O>=+/-1) is free to estimate: no counting noise at all.
    assert feature_noise_variance(np.ones((10, 3)), 1000).tolist() == [0.0] * 3


def test_correction_undoes_attenuation_on_known_noise():
    """The point of the correction: with features observed through known noise, plain ridge
    shrinks the true coefficients toward zero and the EIV correction recovers more of them."""
    rng = np.random.default_rng(7)
    n, d = 4000, 12
    x_true = rng.normal(size=(n, d))
    w = rng.normal(size=(d, 1))
    y = x_true @ w
    sigma2 = 0.25
    x_obs = x_true + rng.normal(scale=np.sqrt(sigma2), size=(n, d))
    noise = np.full(d, sigma2)
    plain = _solve(x_obs, y, noise, 0.0, 1e-6)
    corrected = _solve(x_obs, y, noise, 1.0, 1e-6)
    err = lambda m: float(np.linalg.norm(m.coef_.ravel() - w.ravel()))
    # Attenuation is toward zero, so the plain fit is systematically short.
    assert np.linalg.norm(plain.coef_) < np.linalg.norm(w.ravel())
    assert err(corrected) < err(plain) / 2


def test_gram_correction_stays_solvable_when_noise_dominates():
    """A column that is almost pure noise must not blow the solve up: the clamp plus the PSD
    projection have to keep the system finite even at tau=1."""
    ct, yt, cv, yv, et, ev = _data(seed=3)
    et[:, :10] = 1e-9 * ct[:, :10]  # near-dead columns, the case FloorStandardizer exists for
    readout, val, lam, tau = fit_readout_noise_aware(ct, yt, cv, yv, et, ev, shots=50,
                                                     vacancy_rate=0.01)
    assert np.isfinite(val)
    assert np.isfinite(readout.model.coef_).all()


def test_returns_gpu_readout_compatible_model():
    """GpuReadout reads .coef_/.intercept_ off the model directly; the layout must match Ridge's
    (n_targets, n_features) or the rollout would silently transpose."""
    ct, yt, cv, yv, et, ev = _data(seed=5)
    readout, _, _, _ = fit_readout_noise_aware(ct, yt, cv, yv, et, ev, shots=1000)
    assert isinstance(readout.model, LinearModel)
    assert readout.model.coef_.shape == (yt.shape[1], ct.shape[1] + et.shape[1])
    assert readout.model.intercept_.shape == (yt.shape[1],)
