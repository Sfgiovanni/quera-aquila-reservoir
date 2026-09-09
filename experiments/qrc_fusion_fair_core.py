"""Shared pieces for the bias-corrected classical/QRC fusion comparison.

Every correction identified in `QRC_FUSION_FID_BIAS_AUDIT.md` is applied here:

1. `FloorStandardizer` uses a variance floor of 1e-6 instead of 1e-10, so the 12-30 observables
   that are exactly constant while the memory qubit sits at |0> are neutralised rather than
   divided by ~1e-8. The dropped count is recorded per draw, not assumed.
2. `LAMBDAS` spans 1e-4..1e4. Every published cell selected 10.0, the old ceiling.
3. `rollout` resets the reservoir before every DDIM step, so the features driving the sampler come
   from the same distribution the readout was fitted on. `probe=True` returns the standardized
   magnitude per step so the alignment is verified rather than trusted.
4. `random_map` builds a dimension-matched control: the same 84 extra columns, from a fixed random
   tanh projection instead of a reservoir. It separates "84 nonlinear features" from "quantum".
5. `GpuReadout` evaluates the whole design on the GPU, so the 50-step rollout never round-trips to
   the host. `GpuReadout.check` asserts agreement with the numpy/sklearn path before it is used.

The input clamp asymmetry (`features()` sees raw `x_t`, the samplers clamp to +-1) is deliberately
NOT changed here: clamping at training time alters the supervised problem and would break
comparability with the published MSE. It is exposed as `clamp_train` and run as its own arm.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from sklearn.linear_model import Ridge

from denoiser_classical import sinusoidal_embedding
from denoiser_qrc import fixed_slices, initial_state, reset_data, step, time_rotation
from experiments.qrc_kernel_core import Standardizer

ROOT = 20260802
T = 200
DDIM_STEPS = 50
N_QUBITS = 6
V_SLICES = 4
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1e3, 1e4, 1e5)
VARIANCE_FLOOR = 1e-6


def set_variance_floor(value):
    """Only for the attribution ablation: 1e-10 reproduces the parent Standardizer's behaviour."""
    global VARIANCE_FLOOR
    VARIANCE_FLOOR = float(value)


def ddim_grid(steps=DDIM_STEPS, t_max=T):
    return np.rint(np.linspace(1, t_max, steps)).astype(int)[::-1]


class FloorStandardizer(Standardizer):
    """Standardizer with a floor that actually catches float32-constant columns.

    The parent floors at 1e-10. Observables that are deterministic given a |0> memory qubit have a
    train-time std around 1e-8 -- real float32 noise, not a real scale -- so they pass the parent's
    test and get amplified by 1e8. `n_floored` records how many columns were neutralised.
    """

    def fit(self, x):
        super().fit(x)
        dead = self.scale_ < VARIANCE_FLOOR
        self.n_floored = int(dead.sum())
        self.floored_ = dead
        self.scale_[dead] = 1.
        return self


@dataclass
class RandomMap:
    """Dimension-matched control: a fixed random tanh projection to the same width as the QRC block."""
    standardizer: Standardizer
    weight: np.ndarray
    bias: np.ndarray

    def __call__(self, plain):
        return np.tanh(self.standardizer.transform(plain) @ self.weight + self.bias)


def random_map(plain_train, width, seed):
    rng = np.random.default_rng(seed)
    st = Standardizer().fit(plain_train)
    d = plain_train.shape[1]
    return RandomMap(st, rng.normal(scale=1 / np.sqrt(d), size=(d, width)), rng.normal(scale=.1, size=width))


def plain_design(x, t):
    return np.column_stack([x, sinusoidal_embedding(torch.as_tensor(t), 10, T).numpy()])


def interaction_extra(plain):
    """The 100 `x_t (x) te(t)` products -- the structured classical control.

    This is the baseline the time-injection and timestep-selective experiments used
    (`time_injection.Readout` with `use_interaction=True`). It matters here because a 104-feature
    model beating a 20-feature linear one is the nesting argument, not a quantum result: the honest
    question is whether the reservoir beats a classical block of comparable capacity.
    """
    x, te = plain[:, :10], plain[:, -10:]
    return (x[:, :, None] * te[:, None, :]).reshape(len(x), -1)


def gpu_interaction(device):
    """`interaction_extra` in the `extra_fn` signature `rollout` expects."""
    return lambda xin, te: (xin[:, :, None] * te[:, None, :]).reshape(len(xin), -1)


def balanced_pairs(z, alpha_bar, count, seed):
    """Pairs balanced over the exact DDIM grid. Identical to the published sampler-side protocol."""
    rng = np.random.default_rng(seed)
    t = np.resize(ddim_grid(DDIM_STEPS, len(alpha_bar)), count)
    rng.shuffle(t)
    image_id = rng.choice(len(z), count, replace=count > len(z))
    epsilon = rng.normal(size=(count, z.shape[1]))
    a = alpha_bar[t - 1, None]
    return np.sqrt(a) * z[image_id] + np.sqrt(1 - a) * epsilon, epsilon, t, image_id


@torch.inference_mode()
def qrc_features(x, t, alpha_bar, draw, device, batch=2048, clamp=False, reservoir='digital',
                 reservoir_params=None, shots=None, encoding='quadrature', observables='zz'):
    """The 84 Z/ZZ observables, one fresh reservoir per row -- `phase4.features` without the
    surrounding [x_t, ., te] columns. `clamp` mirrors the sampler's input convention when asked.

    `reservoir='rydberg'` dispatches to `quera.features.rydberg_features` (312 features,
    `V_SLICES` probe-time programs of 78 each) instead of the digital circuit above; `draw`
    is unused on that path -- the Rydberg map has no unitary ensemble, see
    `docs/AQUILA_PORT.md`'s Gate 3 section. `reservoir='digital'` (the default) is
    byte-for-byte the pre-existing function, unchanged.
    """
    if reservoir == 'rydberg':
        from quera.features import rydberg_features
        xb = np.clip(x, -1, 1) if clamp else x
        return rydberg_features(xb, t, reservoir_params, device, shots=shots)
    if reservoir == 'anneal':
        from anneal.features import anneal_features
        xb = np.clip(x, -1, 1) if clamp else x
        return anneal_features(xb, t, reservoir_params, device, shots=shots)
    slices = fixed_slices(V_SLICES, ROOT + 500 + draw, device, N_QUBITS, np.pi / 4, 4, 'alpha_dial')
    out = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch], dtype=torch.float32, device=device)
        if clamp:
            xb = torch.clamp(xb, -1, 1)
        tb = torch.as_tensor(t[i:i + batch], device=device)
        ph = time_rotation(tb, 2 ** N_QUBITS, len(alpha_bar), np.pi / 2)
        if observables == 'full':
            _, h = step_full(initial_state(len(xb), device, N_QUBITS), xb, slices, encoding, ph,
                             pauli_ops(N_QUBITS, device))
        else:
            _, h = step(initial_state(len(xb), device, N_QUBITS), xb, slices,
                        encoding, True, 1.0, time_phase=ph)
        out.append(h.cpu().numpy())
    return np.concatenate(out)


_PAULI_CACHE = {}


def pauli_ops(n_qubits, device):
    """All weight-1 and weight-2 Pauli operators on `n_qubits`, as one `(n_ops, d, d)` tensor.

    The study's readout is `<Z_q>` and `<Z_q Z_r>` only -- 21 operators for 6 qubits. This is the
    full weight-<=2 family: `3*n` single-site plus `9*C(n,2)` two-site, 153 for n=6. Measured
    directly on the post-unitary density matrix, so it costs one extra einsum per slice and no
    extra evolution.
    """
    key = (n_qubits, str(device))
    if key in _PAULI_CACHE:
        return _PAULI_CACHE[key]
    i2 = torch.eye(2, dtype=torch.complex64, device=device)
    p1 = {'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device),
          'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device),
          'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)}

    def build(assign):
        m = torch.ones((1, 1), dtype=torch.complex64, device=device)
        for q in range(n_qubits):
            m = torch.kron(m, p1[assign[q]] if q in assign else i2)
        return m

    ops = [build({q: a}) for a in 'XYZ' for q in range(n_qubits)]
    ops += [build({q: a, r: b}) for q in range(n_qubits) for r in range(q + 1, n_qubits)
            for a in 'XYZ' for b in 'XYZ']
    out = torch.stack(ops)
    _PAULI_CACHE[key] = out
    return out


@torch.inference_mode()
def step_full(state, x, slices, encoding, time_phase, ops):
    """`denoiser_qrc.step` with the full weight-<=2 Pauli readout instead of Z/ZZ.

    Identical dynamics -- same reset, same R(t), same `U rho U^dagger` per slice -- so the only
    difference from `step` is which observables come off each slice.
    """
    state = reset_data(state, x, encoding, 1.0)
    feats = []
    for U in slices:
        if time_phase is not None:
            state = time_phase[:, :, None] * state * time_phase.conj()[:, None, :]
        state = U @ state @ U.mH
        feats.append(torch.einsum('bij,kji->bk', state, ops).real)
    return state, torch.cat(feats, 1)


# ---------------------------------------------------------------------------- readouts


@dataclass
class Readout:
    """Ridge over [standardized classical | standardized extra], with the sklearn predict contract.

    `raw` is always laid out as the sampler builds it: [x_t (10) | extra (width) | te (10)].
    """
    classical: FloorStandardizer
    extra: FloorStandardizer | None
    model: Ridge

    def design(self, raw):
        if self.extra is None:
            return self.classical.transform(raw)
        c = np.column_stack([raw[:, :10], raw[:, -10:]])
        return np.column_stack([self.classical.transform(c), self.extra.transform(raw[:, 10:-10])])

    def predict(self, raw):
        return self.model.predict(self.design(raw))


def fit_readout(c_train, y_train, c_val, y_val, e_train=None, e_val=None):
    """Lambda chosen on validation only; the validation MSE is returned for selection, never as the
    headline number (the caller reports held-out test MSE)."""
    cs = FloorStandardizer().fit(c_train)
    es = None
    if e_train is None:
        dt, dv = cs.transform(c_train), cs.transform(c_val)
    else:
        es = FloorStandardizer().fit(e_train)
        dt = np.column_stack([cs.transform(c_train), es.transform(e_train)])
        dv = np.column_stack([cs.transform(c_val), es.transform(e_val)])
    best = None
    for lam in LAMBDAS:
        m = Ridge(alpha=lam).fit(dt, y_train)
        s = float(np.mean((m.predict(dv) - y_val) ** 2))
        if best is None or s < best[0]:
            best = (s, float(lam), m)
    return Readout(cs, es, best[2]), best[0], best[1]


class GpuReadout:
    """`Readout` as pure GPU tensor ops, so the rollout never leaves the device.

    Standardization is done in float64 on the host inside `Readout`; here it runs in float32 on the
    device. `check` asserts the two agree before any sampling happens -- a silent column-order or
    dtype slip would produce plausible-looking wrong FIDs rather than an error.
    """

    def __init__(self, readout: Readout, device):
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        self.c_mean, self.c_scale = f(readout.classical.mean_), f(readout.classical.scale_)
        self.has_extra = readout.extra is not None
        if self.has_extra:
            self.e_mean, self.e_scale = f(readout.extra.mean_), f(readout.extra.scale_)
        self.w = f(readout.model.coef_).T
        self.b = f(readout.model.intercept_)

    def predict(self, x, extra, te):
        c = (torch.cat([x, te], 1) - self.c_mean) / self.c_scale
        if self.has_extra:
            c = torch.cat([c, (extra - self.e_mean) / self.e_scale], 1)
        return c @ self.w + self.b

    def check(self, readout: Readout, raw, device, atol=1e-2):
        """`atol` raised from 2e-4 to 1e-2 for the n_train=15000 grid, on two measurements.

        **The device path is arithmetically correct.** Repeating `predict` in float64 on device
        reproduces the host path *exactly* at every n_train, while the float32 residual tracks
        the coefficient scale:

            n_train   lambda  |coef|max    err_f32    err_f64
                500       10       0.75  2.036e-06  0.000e+00
               5000       10       0.76  9.510e-07  0.000e+00
              15000    1e-04      20.61  2.598e-04  0.000e+00

        So the slip this check exists to catch -- a column-order or dtype error, which produces
        O(1) disagreement -- is excluded. `|coef|max` grows because the validation curve is flat
        in lambda at n_train=15000 (val_mse varies by 4e-5 across 8 decades while `|coef|max`
        varies 27x), so the argmin selector lands on weakly-regularized fits.

        **A residual at this scale does not move FID.** `random84 s1 d0` at n_train=15000,
        perturbing the fitted coefficients so the induced prediction shift matches the float32
        residual, then ten times that:

            perturbation   FID
              none         64.546
              1e-3         64.544   (delta 0.002)
              1e-2         64.657   (delta 0.111)

        Against a seed-to-seed SD of order 1 FID for this arm, 1e-3 is nothing and 1e-2 is still
        small -- so `atol=1e-2` leaves the guard rail four orders of magnitude below the O(1)
        disagreement it exists to catch, with the amplification question settled by measurement
        rather than assumed. Per-cell `readout_max_diff` stays in the parquet, so the actual
        fidelity of every cell remains auditable after the fact.
        """
        x = torch.as_tensor(raw[:, :10], dtype=torch.float32, device=device)
        te = torch.as_tensor(raw[:, -10:], dtype=torch.float32, device=device)
        extra = torch.as_tensor(raw[:, 10:-10], dtype=torch.float32, device=device) if self.has_extra else None
        got = self.predict(x, extra, te).cpu().numpy()
        want = readout.predict(raw)
        err = float(np.abs(got - want).max())
        if not err < atol:
            raise AssertionError(f'GpuReadout disagrees with the numpy path: max|diff|={err:.3e}')
        return err


# ---------------------------------------------------------------------------- rollout


@torch.inference_mode()
def rollout(noise, gpu_readout, alpha_bar, x0_clip, device, draw=None, extra_fn=None,
            batch=2048, steps=DDIM_STEPS, reset_every_step=True, probe=False,
            reservoir='digital', reservoir_params=None, shots=None, encoding='quadrature',
            observables='zz'):
    """Deterministic (eta=0) DDIM rollout, entirely on device.

    `extra_fn` supplies the middle block: a reservoir step when `draw` is given, a random map when
    `extra_fn` is given, or nothing for the classical arm. With `reset_every_step` the reservoir is
    re-prepared from |0><0| before each step, which is exactly what `qrc_features` does at fit time.

    `reservoir='rydberg'` computes `extra` via `quera.features.rydberg_features` each DDIM
    step (a fresh set of `V_SLICES` probe-time programs from the current `(x_t, t)`, batched
    over samples like every other arm -- DDIM stays sequential in steps, parallel in
    samples). It is stateless by construction (there is no persistent reservoir state to
    carry, only re-encoded `h`), so `reset_every_step=False` is rejected rather than silently
    promising state that is never carried.
    """
    if reservoir in ('rydberg', 'anneal') and not reset_every_step:
        raise ValueError(f"reservoir={reservoir!r} is stateless by construction; "
                         "reset_every_step=False is not meaningful for it")
    grid = ddim_grid(steps, len(alpha_bar))
    slices = fixed_slices(V_SLICES, ROOT + 500 + draw, device, N_QUBITS, np.pi / 4, 4,
                          'alpha_dial') if (reservoir == 'digital' and draw is not None) else None
    out, trace = [], []
    for s0 in range(0, len(noise), batch):
        x = torch.as_tensor(noise[s0:s0 + batch], dtype=torch.float32, device=device)
        state = initial_state(len(x), device, N_QUBITS) if slices is not None else None
        for i, tv in enumerate(grid):
            t = torch.full((len(x),), int(tv), device=device, dtype=torch.long)
            te = sinusoidal_embedding(t, 10, len(alpha_bar))
            xin = torch.clamp(x, -1, 1)
            if reservoir == 'rydberg':
                from quera.features import rydberg_features
                extra_np = rydberg_features(xin.cpu().numpy(), t.cpu().numpy(), reservoir_params,
                                            device, shots=shots)
                extra = torch.as_tensor(extra_np, dtype=torch.float32, device=device)
            elif reservoir == 'anneal':
                from anneal.features import anneal_features
                extra_np = anneal_features(xin.cpu().numpy(), t.cpu().numpy(), reservoir_params,
                                           device, shots=shots)
                extra = torch.as_tensor(extra_np, dtype=torch.float32, device=device)
            elif slices is not None:
                if reset_every_step:
                    state = initial_state(len(x), device, N_QUBITS)
                ph = time_rotation(t, 2 ** N_QUBITS, len(alpha_bar), np.pi / 2)
                if observables == 'full':
                    state, extra = step_full(state, xin, slices, encoding, ph,
                                             pauli_ops(N_QUBITS, device))
                else:
                    state, extra = step(state, xin, slices, encoding, True, 1.0, time_phase=ph)
            elif extra_fn is not None:
                extra = extra_fn(xin, te)
            else:
                extra = None
            if probe and extra is not None and gpu_readout.has_extra:
                z = (extra - gpu_readout.e_mean) / gpu_readout.e_scale
                trace.append((i, int(tv), float(z.abs().mean()), float(z.abs().max()),
                              float(x.abs().max())))
            eps = gpu_readout.predict(xin, extra, te)
            a = float(alpha_bar[tv - 1])
            prev = int(grid[i + 1]) if i + 1 < len(grid) else 0
            ap = 1. if prev == 0 else float(alpha_bar[prev - 1])
            x0 = torch.clamp((x - np.sqrt(1 - a) * eps) / np.sqrt(a), -x0_clip, x0_clip)
            x = np.sqrt(ap) * x0 + np.sqrt(max(1 - ap, 0.)) * eps
        out.append(x.cpu().numpy())
    return np.concatenate(out), trace


def gpu_random_map(mapper: RandomMap, device):
    """`RandomMap` as device tensors, matching the `extra_fn` signature `rollout` expects."""
    f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    mean, scale, w, b = f(mapper.standardizer.mean_), f(mapper.standardizer.scale_), f(mapper.weight), f(mapper.bias)
    return lambda xin, te: torch.tanh((torch.cat([xin, te], 1) - mean) / scale @ w + b)


def degenerate(latent, x0_clip):
    """Guard rail. A cell that trips this is reported, never averaged into a headline FID."""
    diversity = float(np.mean(np.std(latent, axis=0)))
    frac = float(np.mean(np.abs(latent) >= x0_clip * .999))
    return dict(diversity=diversity, frac_at_clip=frac, finite=bool(np.isfinite(latent).all()),
                degenerate=bool(diversity < 1e-4 or frac > .5 or not np.isfinite(latent).all()))
