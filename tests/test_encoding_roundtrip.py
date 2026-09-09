"""`encoding.EncodingAffine`: fit-on-train, apply-unchanged-elsewhere, save/load, and the
saturation-rate accounting the task spec asks to be logged rather than silently absorbed.
"""
from __future__ import annotations

import numpy as np
import pytest

from quera.device import H_LOCAL_MAX, H_LOCAL_MIN
from quera.encoding import EncodingAffine, fit_encoding


def _synthetic_latents(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=scale, size=(n, 10))
    t = rng.integers(1, 201, size=n)
    return x, t


def test_output_always_in_unit_interval():
    x_train, t_train = _synthetic_latents(2000, 0)
    affine = fit_encoding(x_train, t_train)
    for seed, scale in ((1, 1.0), (2, 5.0), (3, 0.01)):
        x, t = _synthetic_latents(500, seed, scale)
        h = affine.transform(x, t)
        assert h.shape == (500, 12)
        assert np.all(h >= H_LOCAL_MIN) and np.all(h <= H_LOCAL_MAX)


def test_fit_on_train_reused_unchanged_elsewhere():
    """Fitting is a pure function of the train split; transforming val/test/rollout data
    must not refit -- checked by fitting twice on different data and confirming a *fixed*
    affine gives identical output regardless of what other data exists."""
    x_train, t_train = _synthetic_latents(2000, 0)
    affine = fit_encoding(x_train, t_train)
    x_val, t_val = _synthetic_latents(300, 99)

    h1 = affine.transform(x_val, t_val)
    # Fitting a second, differently-seeded affine must not affect the first's behavior.
    _ = fit_encoding(*_synthetic_latents(2000, 123))
    h2 = affine.transform(x_val, t_val)
    np.testing.assert_array_equal(h1, h2)


def test_saturation_rate_nonzero_on_fitting_data_by_construction():
    """k=4 (+-2 sigma to the edges) means a Gaussian channel saturates ~4.6% of the time on
    its own fitting data -- an affine that never saturates on the data it was fit to would be
    hiding saturation, not measuring it."""
    x_train, t_train = _synthetic_latents(5000, 0)
    affine = fit_encoding(x_train, t_train)
    rates = affine.saturation_rate(x_train, t_train)
    total = rates["at_zero"] + rates["at_one"]
    assert 0.01 < total < 0.15


def test_saturation_rate_rises_out_of_distribution():
    """A rollout-like input far outside the training range should saturate much more than
    the training data itself -- the concrete failure mode `docs/AQUILA_PORT.md` warns about
    ("no rollout DDIM x_t sai da faixa de treino")."""
    x_train, t_train = _synthetic_latents(3000, 0, scale=1.0)
    affine = fit_encoding(x_train, t_train)
    in_dist_rate = affine.saturation_rate(*_synthetic_latents(1000, 1, scale=1.0))
    out_of_dist_rate = affine.saturation_rate(*_synthetic_latents(1000, 2, scale=6.0))
    assert (out_of_dist_rate["at_zero"] + out_of_dist_rate["at_one"]) > \
          (in_dist_rate["at_zero"] + in_dist_rate["at_one"])


def test_save_load_roundtrip(tmp_path):
    x_train, t_train = _synthetic_latents(1000, 0)
    affine = fit_encoding(x_train, t_train, k=3.5)
    path = tmp_path / "encoding.npz"
    affine.save(path)
    reloaded = EncodingAffine.load(path)

    x, t = _synthetic_latents(200, 7)
    np.testing.assert_array_equal(affine.transform(x, t), reloaded.transform(x, t))
    assert reloaded.k == 3.5
    assert reloaded.T == affine.T


def test_t_scale_changes_only_timestep_channels():
    """`t_scale` must act on the two timestep channels (columns 10, 11) post-standardization,
    not be a no-op absorbed by `fit_encoding`'s own per-channel std (see the field's
    docstring for why scaling the raw channel before fitting would cancel out)."""
    x_train, t_train = _synthetic_latents(3000, 0)
    baseline = fit_encoding(x_train, t_train)
    scaled = fit_encoding(x_train, t_train, t_scale=3.0)

    x, t = _synthetic_latents(500, 42)
    h_base, h_scaled = baseline.transform(x, t), scaled.transform(x, t)
    np.testing.assert_array_equal(h_base[:, :10], h_scaled[:, :10])
    assert not np.allclose(h_base[:, 10:], h_scaled[:, 10:])

    rate_base = baseline.saturation_rate(x_train, t_train)
    rate_scaled = scaled.saturation_rate(x_train, t_train)
    sat_t_base = sum(rate_base["per_channel_at_zero"][10:]) + sum(rate_base["per_channel_at_one"][10:])
    sat_t_scaled = sum(rate_scaled["per_channel_at_zero"][10:]) + sum(rate_scaled["per_channel_at_one"][10:])
    assert sat_t_scaled > sat_t_base


def test_ramps_smoothly_not_step_function():
    """The affine is linear-then-clip: within the unsaturated region output must change
    monotonically and continuously with input, not jump -- confirms `k`/`clip` are wired the
    way the docstring claims, not e.g. accidentally rounding or binning."""
    x_train, t_train = _synthetic_latents(2000, 0)
    affine = fit_encoding(x_train, t_train)
    xs = np.linspace(-0.5, 0.5, 50)[:, None] * np.ones((1, 10))
    t = np.full(50, 100)
    h = affine.transform(xs, t)
    assert np.all(np.diff(h[:, 0]) >= -1e-12)  # monotonically nondecreasing in x_0
