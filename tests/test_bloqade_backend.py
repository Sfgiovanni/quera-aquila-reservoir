"""`backend='bloqade'` must produce the same features as `backend='torch'`, on real programs.

This is Gate 1's cross-validation moved from 20 hand-built programs to the actual production
feature map: same geometry, same pulse schedule, same encoded `h`, compared at the level of the
312 numbers that reach the readout rather than at the level of raw populations. It is what lets a
paper say the reported FIDs could have been produced by QuEra's own emulator -- everything after
`rydberg_features` is identical code either way, so agreement here propagates exactly.

Skipped when `.venv-gate1` is absent, the same way `test_emulator_vs_reference.py` handles it:
the production environment deliberately has no bloqade (see docs/AQUILA_PORT.md, Environment).

The tolerance is `2e-4` on features in `[-1, 1]`. That is looser than Gate 1's `2e-5` on `<n_i>`
for two compounding reasons, both intentional: production runs at `rk4_safety=0.2` rather than the
default `0.05` (see the Gate 3 deviations note), and a `<Z_i Z_j>` correlator accumulates the error
of two single-site terms plus a joint term. The measured worst case sits well inside it.
"""
from __future__ import annotations

import numpy as np
import pytest

from quera.bloqade_backend import DEFAULT_PYTHON
from quera.encoding import fit_encoding
from quera.features import ReservoirParams, n_features, rydberg_features

pytestmark = pytest.mark.skipif(not DEFAULT_PYTHON.exists(),
                                reason='.venv-gate1 (bloqade-analog) not installed')

GEOM = dict(spacing_um=9.756844364507792, omega=6.283, s_detuning=9.0, v_slices=4)
N = 6


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.8, size=(N, 10))
    t = rng.integers(1, 200, N)
    return x, t, fit_encoding(x, t, t_scale=1.0)


def _params(encoding, backend, **kw):
    # `use_cache=False` is load-bearing, not tidiness. With the cache on, a second run of this
    # file reads both arms straight off disk and passes in 0.7s without touching either solver --
    # it would keep passing after an arbitrary change to the worker. The whole point here is to
    # exercise bloqade every time, so the cache is off and the test costs ~30s.
    return ReservoirParams(encoding=encoding, backend=backend, rk4_safety=0.2, use_cache=False,
                           dtype='complex128', bloqade_workers=6, **GEOM, **kw)


def test_exact_features_match_torch_backend():
    x, t, enc = _inputs()
    mine = rydberg_features(x, t, _params(enc, 'torch'), 'cpu')
    theirs = rydberg_features(x, t, _params(enc, 'bloqade'), 'cpu')
    assert mine.shape == theirs.shape == (N, n_features(12) * GEOM['v_slices'])
    worst = float(np.abs(mine - theirs).max())
    assert worst < 2e-4, f'worst feature disagreement {worst:.3e}'
    # A permuted site convention would still pass a loose bound on the *mean*; the per-site
    # <Z_i> block is where it would show up, so pin that separately and tightly.
    assert float(np.abs(mine[:, :12] - theirs[:, :12]).max()) < 5e-5


def test_features_are_in_range_and_not_degenerate():
    """Guards the failure mode a wrong index reversal produces: a valid-looking but scrambled
    state whose features are still in [-1,1] and still sum correctly."""
    x, t, enc = _inputs(seed=3)
    f = rydberg_features(x, t, _params(enc, 'bloqade'), 'cpu')
    assert np.isfinite(f).all() and np.abs(f).max() <= 1.0 + 1e-9
    assert f.std(axis=0).min() > 0, 'a constant column means the encoding never reached the solver'


def test_cache_key_separates_backends(tmp_path):
    """The feature cache is content-addressed; if `backend` were left out of the key a bloqade
    run would silently reuse torch-computed features and the whole comparison would be circular."""
    from quera.cache import content_key
    base = dict(probe_index=1, h=np.zeros((2, 12)), shots=None)
    assert content_key(**base, backend='torch') != content_key(**base, backend='bloqade')
