"""The 78 Z/ZZ observables (`n_sites` `<Z_i>` + `n_sites*(n_sites-1)/2` `<Z_i Z_j>`) for one
probe-time program -- the Rydberg analogue of `denoiser_qrc.step`'s Z/ZZ block, in the
occupation-to-spin convention `Z = 1 - 2n` (`Z=+1` for |g>, `Z=-1` for |r>, matching the
usual qubit convention with |r> playing the role of |1>).

`zz_features_exact` uses the statevector directly (`shots=None` in `rydberg_features` below);
`zz_features_sampled` counts from noisy bitstrings, exactly
like the hardware would report them. `tests/test_shot_convergence.py` is the evidence that
the two agree in the `shots -> infinity` limit at the `1/sqrt(shots)` rate shot noise
predicts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quera.cache import cached
from quera.device import MAX_DURATION_US, MIN_SEGMENT_US
from quera.emulator import DEFAULT_RK4_SAFETY, GeometryCache, build_geometry_cache, evolve, occupation_moments
from quera.encoding import EncodingAffine
from quera.layout import DEFAULT_SPACING_UM, cluster_3x4
from quera.pulses import build_probe_program
from quera.sampling import apply_readout_noise, sample_bitstrings


def _pair_indices(n_sites: int, device):
    return torch.triu_indices(n_sites, n_sites, offset=1, device=device)


def n_features(n_sites: int) -> int:
    return n_sites + n_sites * (n_sites - 1) // 2


def zz_features_exact(psi: torch.Tensor, geom: GeometryCache) -> torch.Tensor:
    """`(batch, n_features(n_sites))`, exact expectations from the statevector."""
    n_i, joint = occupation_moments(psi, geom)
    iu = _pair_indices(geom.n_sites, psi.device)
    z_i = 1 - 2 * n_i
    n_ij = joint[:, iu[0], iu[1]]
    z_ij = 1 - 2 * n_i[:, iu[0]] - 2 * n_i[:, iu[1]] + 4 * n_ij
    return torch.cat([z_i, z_ij], dim=1)


def _pauli_coeff(kind, bits_q, dtype, device):
    """Row-index-dependent matrix element of a single-site Pauli, in the `|g>=|0>, |r>=|1>` basis.

    `(P)[i_q, j_q]` for `X` is 1 whenever the bit differs; for `Y` it is `i*(2*i_q - 1)` (so
    `Y[1,0]=+i`, `Y[0,1]=-i`); for `Z` it is `(1 - 2*i_q)` with no bit flip. Everything depends on
    the bits of the *row* index, which is what makes the gather formulation below exact rather
    than approximate.
    """
    if kind == 'X':
        return torch.ones_like(bits_q, dtype=dtype)
    if kind == 'Y':
        return (1j * (2 * bits_q - 1)).to(dtype)
    return (1 - 2 * bits_q).to(dtype)


def pauli_features_exact(psi: torch.Tensor, geom: GeometryCache) -> torch.Tensor:
    """All weight-1 and weight-2 Pauli expectations, `(batch, 3n + 9*C(n,2))`.

    `zz_features_exact` reads only `Z` and `ZZ`, which for a Rydberg array is exactly what site
    occupation gives you. `X` and `Y` are the ground-Rydberg coherences: real information in the
    state, invisible to an occupation measurement. This computes all of them from the statevector
    by bit-flip gathers (no dense operator is ever materialised -- at `n_sites=12` the full set
    would be 84 GB).

    Ordering: `[X_q, Y_q, Z_q for q] + [P_q P_r for q<r, P in XYZ x XYZ]`, matching
    `qrc_fusion_fair_core.pauli_ops` so the two arms' blocks are read the same way.
    """
    dtype = psi.dtype
    n, dim = geom.n_sites, geom.dim
    bits = geom.bits.to(psi.device)                       # (n, dim) 0/1
    flip = geom.flip_index.to(psi.device)                 # (n, dim)
    conj = psi.conj()
    idx_id = torch.arange(dim, device=psi.device)

    def expect(coeff, idx):
        return torch.einsum('bi,i,bi->b', conj, coeff, psi[:, idx]).real

    feats = []
    for kind in ('X', 'Y', 'Z'):
        for q in range(n):
            c = _pauli_coeff(kind, bits[q], dtype, psi.device)
            feats.append(expect(c, flip[q] if kind in 'XY' else idx_id))
    for q in range(n):
        for r in range(q + 1, n):
            for a in 'XYZ':
                ca = _pauli_coeff(a, bits[q], dtype, psi.device)
                for b in 'XYZ':
                    cb = _pauli_coeff(b, bits[r], dtype, psi.device)
                    idx = idx_id
                    if a in 'XY':
                        idx = flip[q][idx]
                    if b in 'XY':
                        idx = flip[r][idx]
                    feats.append(expect(ca * cb, idx))
    return torch.stack(feats, dim=1)


def n_pauli_features(n_sites: int) -> int:
    return 3 * n_sites + 9 * n_sites * (n_sites - 1) // 2


def zz_features_sampled(psi: torch.Tensor, geom: GeometryCache, shots: int, generator=None,
                        vacancy_rate: float = 0.0, fp_rate: float = 0.0, fn_rate: float = 0.0):
    """`(features, n_usable)`: features counted from `shots` noisy samples per batch row,
    `n_usable` (`(batch,)`) the number that survived the vacancy filter (see
    `sampling.apply_readout_noise`) -- the shots actually averaged over, since a fixed
    `shots` argument produces a random *usable* count once `vacancy_rate>0`.
    """
    bits = sample_bitstrings(psi, geom, shots, generator=generator)
    noisy, usable = apply_readout_noise(bits, vacancy_rate, fp_rate, fn_rate, generator=generator)
    noisy_f = noisy.to(torch.float64)
    usable_f = usable.to(torch.float64).unsqueeze(-1)  # (batch, shots, 1)
    n_usable = usable.to(torch.float64).sum(dim=1).clamp_min(1.0)  # (batch,)

    p_i = (noisy_f * usable_f).sum(dim=1) / n_usable[:, None]  # (batch, n_sites)
    iu = _pair_indices(geom.n_sites, psi.device)
    pair_prod = noisy_f[:, :, iu[0]] * noisy_f[:, :, iu[1]]  # (batch, shots, n_pairs)
    p_ij = (pair_prod * usable_f).sum(dim=1) / n_usable[:, None]

    z_i = 1 - 2 * p_i
    z_ij = 1 - 2 * p_i[:, iu[0]] - 2 * p_i[:, iu[1]] + 4 * p_ij
    return torch.cat([z_i, z_ij], dim=1), n_usable


def gaussian_noise_like_shots(exact: torch.Tensor, shots: int, generator=None) -> torch.Tensor:
    """`exact` (`(batch, n_features)`) plus i.i.d. Gaussian noise whose per-feature variance
    matches `physical_variance_floor` at `shots` -- same magnitude as real shot noise, none of
    its structure (`zz_features_sampled`'s `Z_i`/`Z_iZ_j` at a probe time come from the *same*
    multinomial draws, so real shot noise is correlated within a probe time in a way this
    control deliberately is not). Gate 5's null hypothesis: if this control performs the same
    as real shot noise downstream, whatever `S`-dependence shows up is generic noise-as-
    regularizer, not something specific to projective measurement statistics."""
    sigma = physical_variance_floor(exact, shots).clamp_min(0.0).sqrt()
    noise = torch.randn(exact.shape, generator=generator, device=exact.device, dtype=torch.float32)
    return exact + noise.to(exact.dtype) * sigma


def physical_variance_floor(features: torch.Tensor, shots) -> torch.Tensor:
    """`sigma^2 ~= (1 - <O>^2) / shots`, the correct floor for counting-based observables.

    Contrast with the digital arm's `experiments/qrc_fusion_fair_core.VARIANCE_FLOOR=1e-6`,
    which was calibrated for *exact* observables (see `QRC_FUSION_FID_BIAS_AUDIT.md`) and is
    left untouched -- this is a separate, opt-in floor for the Rydberg feature pipeline once
    it standardizes sampled features, not a replacement for the digital arm's constant.
    """
    return (1 - features**2).clamp_min(0.0) / shots


@dataclass
class ReservoirParams:
    """Everything `rydberg_features` needs beyond `(x, t)`: the fitted encoding, the fixed
    cluster geometry/pulse operating point, and the emulator's own numerics. Deliberately
    flat and JSON-primitive-or-array (see `cache.content_key`'s docstring) so the whole
    struct can be dropped straight into a cache key.
    """
    encoding: EncodingAffine
    spacing_um: float = DEFAULT_SPACING_UM
    omega: float = 6.283
    s_detuning: float = 9.0
    ramp_us: float = MIN_SEGMENT_US
    v_slices: int = 4
    t_max_us: float = MAX_DURATION_US
    rk4_safety: float = DEFAULT_RK4_SAFETY
    dtype: str = "complex64"
    vacancy_rate: float = 0.0
    fp_rate: float = 0.0
    fn_rate: float = 0.0
    backend: str = "torch"
    """`torch` (default): this project's `quera/emulator.py`. `bloqade`: QuEra's own
    bloqade-analog, via `quera.bloqade_backend`, which does the Schrodinger solve in the
    `.venv-gate1` subprocess and hands back populations -- every downstream step (Z/ZZ
    observables, shot sampling, vacancy filtering) stays in this project's tested code, so the
    only component swapped is the solver. See `quera/_bloqade_worker.py` for why the split is
    drawn there and for the index-convention reversal it requires."""
    bloqade_workers: int = 16
    noise_mode: str = "shots"
    """`shots` (default): real multinomial shot noise via `zz_features_sampled`, with
    `vacancy_rate`/`fp_rate`/`fn_rate` applied. `gaussian`: exact features plus
    `gaussian_noise_like_shots`'s variance-matched i.i.d. noise instead -- the Gate 5 control
    that isolates shot noise's magnitude from its structure. `vacancy_rate`/`fp_rate`/
    `fn_rate` don't apply in `gaussian` mode (they are shot-population artifacts with no
    Gaussian analogue); only meaningful together with an integer `shots` argument to
    `rydberg_features` -- ignored when `shots is None` (exact features, no noise at all)."""
    seed: int = 0
    observables: str = 'zz'
    use_cache: bool = True


def rydberg_features(x: np.ndarray, t, params: ReservoirParams, device, shots: int | None = None) -> np.ndarray:
    if params.observables == 'full' and shots is not None:
        raise ValueError(
            "observables='full' with shots is not implemented: <X> and <Y> are ground-Rydberg "
            "coherences and cannot be counted from occupation bitstrings. On hardware they need a "
            "terminal basis-rotation pulse and their own shot budget; see docs/AQUILA_PORT.md.")
    """`(n, 78*v_slices)`, the Rydberg analogue of `qrc_features(x, t, alpha_bar, draw, device)`
    -- same position in the pipeline, called the same way, but **no `draw`**: the digital
    arm's `draw` selects a random circuit from an ensemble of reservoir unitaries; the
    Rydberg map has no such ensemble; it is fully determined, up to shot noise, by the fixed
    physical operating point in `params` (see `docs/AQUILA_PORT.md`'s Gate 3 section for what
    this means for the paired statistics downstream).

    Each of the `v_slices` probe-time programs is its own from-scratch pulse
    (`T_k = k * t_max_us / v_slices`, `k=1..v_slices` -- `k=0` would be a zero-duration,
    trivially-`|gg...g>` program, so it is skipped rather than included as a degenerate
    slice), matching `pulses.py`'s "no snapshots of one long trajectory" rule. `shots=None`
    gives exact expectations; an integer sample-counts features from noisy bitstrings using
    `params.vacancy_rate`/`fp_rate`/`fn_rate`.
    """
    x = np.asarray(x, dtype=np.float64)
    h = params.encoding.transform(x, t)  # (n, n_sites), already clipped to [0, 1]
    positions = cluster_3x4(params.spacing_um)
    real_dtype = torch.float32 if params.dtype == "complex64" else torch.float64
    geom = build_geometry_cache(positions, device, real_dtype)
    torch_dtype = getattr(torch, params.dtype)
    placeholder_h = np.full(geom.n_sites, 0.5)  # program.h is a validation placeholder only;
                                                # the real per-sample h is checked in evolve()

    blocks = []
    for k in range(1, params.v_slices + 1):
        t_k = k * params.t_max_us / params.v_slices
        plateau_us = max(0.0, t_k - 2 * params.ramp_us)
        program = build_probe_program(positions, placeholder_h, plateau_us=plateau_us,
                                      omega=params.omega, s_detuning=params.s_detuning,
                                      ramp_us=params.ramp_us)

        def compute(program=program, h=h, k=k):
            if params.backend == "bloqade":
                from quera.bloqade_backend import shared_pool
                probs = shared_pool(params.bloqade_workers).probs(program, h)
                # Downstream only ever reads |psi|^2 (`occupation_moments`, `sample_bitstrings`),
                # so a real sqrt(probs) is a faithful stand-in for the amplitude vector and lets
                # the whole feature/sampling path run unmodified against bloqade's populations.
                psi = torch.sqrt(torch.as_tensor(probs, dtype=real_dtype, device=device))
            else:
                psi = evolve(program, h, device, dtype=torch_dtype, geometry=geom,
                             rk4_safety=params.rk4_safety)
            if shots is None:
                feat = (pauli_features_exact(psi, geom) if params.observables == 'full'
                        else zz_features_exact(psi, geom))
            elif params.noise_mode == "gaussian":
                exact = zz_features_exact(psi, geom)
                gen = torch.Generator(device=device).manual_seed(params.seed * 1_000_003 + k + 500_009)
                feat = gaussian_noise_like_shots(exact, shots, generator=gen)
            else:
                gen = torch.Generator(device=device).manual_seed(params.seed * 1_000_003 + k)
                feat, _ = zz_features_sampled(psi, geom, shots, generator=gen,
                                              vacancy_rate=params.vacancy_rate,
                                              fp_rate=params.fp_rate, fn_rate=params.fn_rate)
            return feat.cpu().numpy()

        if params.use_cache:
            block = cached(compute, probe_index=k, positions_um=positions, times_us=program.times_us,
                           omega=program.omega, phase=program.phase, delta_global=program.delta_global,
                           delta_local=program.delta_local, h=h, shots=shots, seed=params.seed,
                           rk4_safety=params.rk4_safety, dtype=params.dtype, backend=params.backend,
                           noise_mode=params.noise_mode, observables=params.observables,
                           vacancy_rate=params.vacancy_rate, fp_rate=params.fp_rate, fn_rate=params.fn_rate)
        else:
            block = compute()
        blocks.append(block)
    return np.concatenate(blocks, axis=1)
