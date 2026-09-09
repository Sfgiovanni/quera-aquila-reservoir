"""Gate 7, part 1: a readout that knows its features are shot-noise estimates.

Gate 5 established that the Rydberg arm's FID gain dies at `shots<=1000`, and that the damage is
generic noise *magnitude* (the variance-matched Gaussian control tracked real shot noise within
~0.5%). What neither Gate 5 nor anything before it did is tell the *readout* that the features are
noisy. `fit_readout` standardizes the observable block with `FloorStandardizer`, whose
`VARIANCE_FLOOR=1e-6` was calibrated for **exact** observables (see `QRC_FUSION_FID_BIAS_AUDIT.md`);
at `shots=1000` each feature carries a noise std of ~0.032 while its informative variation across
samples is only 0.065-0.158 (Gate 3's recorded diagnostic), i.e. a per-feature SNR of roughly 2:1
to 5:1. The current readout treats those columns as if they were exact.

`quera.features.physical_variance_floor` already computes the right per-feature variance and, as
Gate 2's section notes, "is implemented and tested but not wired into anything yet". This module
wires it in -- for the Rydberg arm only, as a separate function. `qrc_fusion_fair_core.fit_readout`
is not touched, per the standing rule that the digital arm's behavior must not change.

## What the correction actually is

Noisy features make ridge regression *attenuate*: with `X = X_true + E` and `E` zero-mean with
known per-feature variance, `E[X'X] = X_true'X_true + n*Sigma`, so the Gram matrix is inflated
along the diagonal by exactly the noise the features carry. Subtracting it back is the classical
errors-in-variables (total-least-squares / regression-calibration) correction, and the reason it
is worth doing *here* rather than in a generic setting is that `Sigma` is **known analytically**
rather than estimated: for any +/-1-valued observable, `sigma^2 = (1 - <O>^2)/S` exactly. That is
a rare luxury and it is the whole argument for this gate.

The correction is unbiased but higher-variance (undoing attenuation inflates coefficients on the
noisiest columns, which is exactly where the noise lives), so it is not applied at full strength
on faith. A shrinkage `tau` in `[0, 1]` scales it and is selected on validation *jointly with*
`lambda`, and **`tau=0` is in the grid and reproduces plain ridge exactly** (asserted in
`tests/test_gate7_readout.py`). That makes this a strict generalization: on validation it cannot
do worse than the readout it replaces, and any gain is measured against that identical baseline
rather than against a re-tuned one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.qrc_fusion_fair_core import LAMBDAS, FloorStandardizer, Readout

# Shrinkage grid for the errors-in-variables correction. 0.0 (plain ridge) is deliberately first
# and deliberately present: it is the null hypothesis this gate has to beat on validation.
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)

# The Gram correction subtracts tau*n*sigma^2 from a standardized diagonal whose entries are ~1.
# A column whose noise variance exceeds this fraction of its *total* variance carries essentially
# no signal, and letting the subtraction approach 1.0 would drive that column's corrected variance
# to zero and its fitted coefficient to infinity. Clamp instead of trusting the PSD projection to
# clean it up afterwards.
MAX_NOISE_FRACTION = 0.9


def feature_noise_variance(features: np.ndarray, shots: int, vacancy_rate: float = 0.0,
                           n_sites: int = 12) -> np.ndarray:
    """Per-column shot-noise variance of `features`, averaged over samples.

    `sigma^2 = (1 - <O>^2)/S_eff` -- exact for any +/-1-valued observable, so it holds for the
    `<Z_i>` singles and the `<Z_i Z_j>` correlators alike (`Z_i Z_j` is itself +/-1-valued), which
    is why one formula covers all 78 observables per probe time.

    `S_eff` accounts for the vacancy post-selection the sampler actually performs: a shot is usable
    only if all `n_sites` atoms survived, so the expected usable count is `S*(1-v)^n_sites` -- the
    same `0.99**12 = 88.6%` figure `quera/sampling.py` is tested against, not a new assumption.

    `<O>` is taken from the (noisy) observed features. That is slightly conservative:
    `E[O_hat^2] = <O>^2 + sigma^2`, so `1 - O_hat^2` under-estimates `1 - <O>^2` and the resulting
    sigma^2 is a mild under-estimate. Correcting it would need the true `<O>` this function exists
    to avoid needing; the bias is `O(1/S)` and is absorbed by the `tau` grid.
    """
    if shots is None:
        return np.zeros(features.shape[1])
    s_eff = shots * (1.0 - vacancy_rate) ** n_sites
    return np.mean(np.clip(1.0 - np.asarray(features, np.float64) ** 2, 0.0, None), axis=0) / s_eff


@dataclass
class LinearModel:
    """`sklearn.Ridge`'s predict contract, over coefficients solved here.

    Matches `Ridge`'s array layout exactly (`coef_` is `(n_targets, n_features)`) so that
    `Readout` and, critically, `GpuReadout` -- which reads `.coef_`/`.intercept_` directly and
    re-checks them against the host path before any sampling -- work against this unchanged.
    """
    coef_: np.ndarray
    intercept_: np.ndarray

    def predict(self, design):
        return np.asarray(design, np.float64) @ self.coef_.T + self.intercept_


def _solve(design, y, noise_var, tau, lam):
    """Ridge on a Gram matrix with `tau * n * diag(noise_var)` subtracted back off.

    The subtraction can in principle push the Gram indefinite (it is only guaranteed PSD in
    expectation), which would turn a solvable system into nonsense rather than into an error, so
    the corrected Gram is projected back onto the PSD cone by eigenvalue clipping before `lambda`
    is added. `eigh` is safe here: the Gram is symmetric by construction and is symmetrized
    explicitly anyway to kill float asymmetry.
    """
    n = len(design)
    dm, ym = design.mean(0), y.mean(0)
    dc, yc = design - dm, y - ym
    gram = dc.T @ dc
    if tau > 0 and noise_var.any():
        gram = gram - tau * n * np.diag(noise_var)
        gram = (gram + gram.T) / 2
        w, v = np.linalg.eigh(gram)
        gram = (v * np.clip(w, 0.0, None)) @ v.T
    coef = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), dc.T @ yc)
    return LinearModel(coef.T, ym - dm @ coef)


def fit_readout_noise_aware(c_train, y_train, c_val, y_val, e_train, e_val, shots,
                            vacancy_rate: float = 0.0, taus=TAUS, lambdas=LAMBDAS):
    """`fit_readout`'s contract -- `(Readout, val_mse, lambda)` -- plus the selected `tau`.

    Returns `(readout, val_mse, lambda, tau)`. Selection is on validation only, exactly as
    `fit_readout` does it; `tau` joins `lambda` in the same grid search rather than being fixed
    ahead of time, because which of the two is doing the regularizing is precisely what is unknown.
    """
    cs = FloorStandardizer().fit(c_train)
    es = FloorStandardizer().fit(e_train)
    dt = np.column_stack([cs.transform(c_train), es.transform(e_train)])
    dv = np.column_stack([cs.transform(c_val), es.transform(e_val)])

    # Noise lives only in the observable block, and the standardizer divided that block by
    # `es.scale_`, so the variance must be divided by `scale_**2` to land in the same coordinates.
    # Columns the floor neutralised keep scale_=1 and are excluded outright -- they carry no signal
    # and correcting them would be correcting pure float noise.
    raw_var = feature_noise_variance(e_train, shots, vacancy_rate)
    ext_var = np.zeros_like(raw_var) if shots is None else raw_var / es.scale_ ** 2
    if getattr(es, 'floored_', None) is not None:
        ext_var[es.floored_] = 0.0
    ext_var = np.clip(ext_var, 0.0, MAX_NOISE_FRACTION)
    noise_var = np.concatenate([np.zeros(cs.scale_.shape[0]), ext_var])

    best = None
    for tau in taus:
        for lam in lambdas:
            model = _solve(dt, y_train, noise_var, tau, float(lam))
            score = float(np.mean((model.predict(dv) - y_val) ** 2))
            if best is None or score < best[0]:
                best = (score, float(lam), float(tau), model)
    score, lam, tau, model = best
    return Readout(cs, es, model), score, lam, tau
