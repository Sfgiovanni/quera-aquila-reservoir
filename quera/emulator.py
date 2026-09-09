"""Batched statevector propagator for the Rydberg Hamiltonian, in torch.

No dissipative channels and a fixed initial state (`|gg...g>`) make the evolution unitary
pure, so this works directly with statevectors (batch, 2^n) rather than density matrices --
2^12 = 4096 amplitudes, 32 KB/sample at complex64, matching the task spec's budget.

Hamiltonian, restated from the task spec::

    H(t)/hbar = sum_j (Omega(t)/2)(e^{i phi(t)} |g><r|_j + h.c.)
              - sum_j [Delta_global(t) + h_j*Delta_local(t)] n_j
              + sum_{i<j} (C6 / r_ij^6) n_i n_j

with n_j = |r><r|_j. Bit convention: index bit q (site q, q=0 = most significant, matching
`denoiser_qrc.step`'s `idx >> (n-1-q)` convention) is 1 for |r> and 0 for |g>.

Matvec without assembling a matrix
-----------------------------------
The drive term acting on qubit q connects basis index `i` only to `flip_q(i)` (`i` with
bit q toggled) -- a fixed index permutation, applied with `torch.gather`. The diagonal term
(detuning + van der Waals) is a `(batch, 2^n)` real vector, built once per program from three
precomputed, geometry-only pieces (`popcount`, `pair_energy`) and one per-sample piece
(`f_h = h @ bits`) that does not depend on time, since `Delta_global`/`Delta_local` are held
constant across an entire probe-time program (see `pulses.py`). Only `Omega(t)` (and, if
nonzero, `phi(t)`) varies within a program, so the diagonal is computed exactly once per
`evolve()` call and reused at every Runge-Kutta stage.

Propagator choice
------------------
RK4 with a fixed sub-step count per pulse segment, chosen from a Gershgorin bound on
`||H||` (`max|diagonal| + n*max(Omega)/2` -- the diagonal magnitude plus the n possible
single-flip off-diagonal contributions per row) so that `||H||*dt` stays under a fixed
safety threshold. Krylov/Lanczos would need fewer matvecs per unit accuracy at the
`||H||` this problem reaches (interaction energies of several hundred rad/us at the default
`a=8um` cluster spacing -- deliberately close to the blockade radius, so the diagonal is
large even though it is usually strongly suppressed by the blockade in the states the drive
actually populates), but RK4 was chosen over it here because the matvec is already cheap
(O(n) gathers, no orthogonalization bookkeeping) and fully batched RK4 is a few lines of
tensor algebra with no subspace-size hyperparameter to tune per-batch. `tests/test_emulator_vs_reference.py`
checks this against two independent reference solvers; `test_shot_convergence`-adjacent
convergence checks (halved step count) live alongside the emulator's own unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from quera.device import Program

DEFAULT_RK4_SAFETY = 0.05  # ||H||*dt kept under this (dimensionless) bound per RK4 sub-step
# Measured on the drive-factorization free correctness test (n=12, a=8um cluster): global
# error ~2.2e-6 at safety=0.2, ~8e-9 at safety=0.05 -- consistent with RK4's dt^4 scaling
# (halving the safety threshold roughly halves dt and cuts error by ~16x). 0.05 keeps a
# comfortable margin under Gate 1's 1e-6 cross-validation tolerance at a modest cost
# (~0.3s for a single 12-atom, 0.8us program on CPU); pass `rk4_safety` to `evolve` to
# trade accuracy for speed in bulk production runs once Gate 3+ shows what margin is needed.


@dataclass
class GeometryCache:
    """Everything about a program that depends on `positions_um` alone, not on `h` or time.

    Reused across an entire batch of samples (same cluster, many different latents) and,
    within `rollout`-style calls, across every DDIM step too -- geometry never changes.
    """
    n_sites: int
    dim: int
    bits: torch.Tensor          # (n_sites, dim) float, 0/1
    flip_index: torch.Tensor    # (n_sites, dim) int64, flip_index[q,i] = i with bit q toggled
    flip_index_flat: torch.Tensor  # (n_sites*dim,) = flip_index.reshape(-1), for one batched gather
    popcount: torch.Tensor      # (dim,) float
    pair_energy: torch.Tensor   # (dim,) float, sum_{p<q} C6/r_pq^6 * bit_p(i)*bit_q(i)


def build_geometry_cache(positions_um: np.ndarray, device, real_dtype=torch.float32) -> GeometryCache:
    from quera.device import C6_RAD_US_UM6

    pos = np.asarray(positions_um, dtype=np.float64)
    n = pos.shape[0]
    dim = 1 << n
    idx = np.arange(dim)
    bits_np = np.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)]).astype(np.float64)  # (n, dim)
    flip_np = np.stack([idx ^ (1 << (n - 1 - q)) for q in range(n)])  # (n, dim)

    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((diff**2).sum(-1))
    with np.errstate(divide="ignore"):
        j_pair = np.where(r > 0, C6_RAD_US_UM6 / np.where(r > 0, r, 1.0) ** 6, 0.0)
    pair_energy_np = np.zeros(dim)
    for p in range(n):
        for q in range(p + 1, n):
            pair_energy_np += j_pair[p, q] * bits_np[p] * bits_np[q]

    f = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    flip_index = f(flip_np, torch.int64)
    return GeometryCache(
        n_sites=n, dim=dim,
        bits=f(bits_np, real_dtype),
        flip_index=flip_index,
        flip_index_flat=flip_index.reshape(-1).contiguous(),
        popcount=f(bits_np.sum(0), real_dtype),
        pair_energy=f(pair_energy_np, real_dtype),
    )


def zero_state(batch: int, dim: int, device, dtype=torch.complex64) -> torch.Tensor:
    """`|gg...g>`, i.e. index 0, for every sample in the batch."""
    psi = torch.zeros((batch, dim), dtype=dtype, device=device)
    psi[:, 0] = 1.0
    return psi


def _diagonal(geom: GeometryCache, h_batch: torch.Tensor, delta_global: float, delta_local: float,
             dtype) -> torch.Tensor:
    """`(batch, dim)`, constant for the whole program (Delta_global/Delta_local don't vary)."""
    f_h = h_batch.to(geom.bits.dtype) @ geom.bits  # (batch, dim)
    diag = (-delta_global) * geom.popcount[None, :] + (-delta_local) * f_h + geom.pair_energy[None, :]
    return diag.to(dtype)


def _hamiltonian_matvec(psi: torch.Tensor, geom: GeometryCache, diag: torch.Tensor, omega: float,
                        phase: float, phase_coeff: torch.Tensor | None) -> torch.Tensor:
    """`phase_coeff` is `None` for `phase==0` (the real, symmetric `Omega/2 * X_q` drive --
    the only case `pulses.py` currently emits) or a precomputed `(n_sites, dim)` table of
    `e^{+-i*phase}` otherwise (see `_phase_coeff_table`). Precomputing it outside the RK4
    loop, and gathering all `n_sites` flips in one call instead of `n_sites` sequential
    `torch.gather`s, is what makes this GPU-batch-friendly: the per-step cost was
    launch-bound (many tiny kernels), not compute-bound (see `docs/AQUILA_PORT.md`'s Gate 1
    performance note) -- ~30x faster on a `batch=2048` production-size call after this change.
    """
    out = diag * psi
    if omega != 0.0:
        half_omega = omega / 2.0
        idx = geom.flip_index_flat.unsqueeze(0).expand(psi.shape[0], -1)  # (batch, n_sites*dim)
        flipped = torch.gather(psi, 1, idx).view(psi.shape[0], geom.n_sites, geom.dim)
        if phase_coeff is None:
            drive = half_omega * flipped.sum(dim=1)
        else:
            drive = half_omega * (phase_coeff[None, :, :] * flipped).sum(dim=1)
        out = out + drive
    return -1j * out


def _phase_coeff_table(geom: GeometryCache, phase: float, dtype) -> torch.Tensor:
    """`(n_sites, dim)`: `e^{i phase}` where site q is |g> (raising term), `e^{-i phase}`
    where site q is |r> (lowering term) -- see the module docstring's basis convention."""
    eip, eim = complex(np.cos(phase), np.sin(phase)), complex(np.cos(phase), -np.sin(phase))
    return torch.where(geom.bits.bool(), torch.full_like(geom.bits, eim, dtype=dtype),
                       torch.full_like(geom.bits, eip, dtype=dtype))


def _segment_steps(geom: GeometryCache, diag: torch.Tensor, omega_lo: float, omega_hi: float,
                   duration_us: float, rk4_safety: float) -> int:
    """RK4 sub-step count so `||H||*dt <= rk4_safety` under a Gershgorin bound on `||H||`."""
    diag_max = float(diag.abs().amax().real) if torch.is_complex(diag) else float(diag.abs().amax())
    drive_bound = geom.n_sites * max(abs(omega_lo), abs(omega_hi)) / 2.0
    h_bound = diag_max + drive_bound
    if h_bound <= 0.0:
        return 1
    dt_target = rk4_safety / h_bound
    return max(1, int(np.ceil(duration_us / dt_target)))


def evolve(program: Program, h_batch, device, dtype=torch.complex64, geometry: GeometryCache | None = None,
          norm_drift_tol: float = 1e-5, rk4_safety: float = DEFAULT_RK4_SAFETY):
    """Propagate `|gg...g>` through `program`, one shared pulse, batched over `h_batch`.

    `h_batch` is `(batch, n_sites)`, one local-detuning pattern per sample; the pulse
    (`omega`/`phase`/`delta_global`/`delta_local` on `program.times_us`) is identical for
    every sample in the batch, matching the Hamiltonian's data dependence living only in
    the detuning term. Returns the final `(batch, dim)` statevector.
    """
    from quera.device import H_LOCAL_MAX, H_LOCAL_MIN, ProgramValidationError, validate
    validate(program)  # only checks program.h, a single placeholder row -- see below

    geom = geometry if geometry is not None else build_geometry_cache(
        program.positions_um, device, torch.float32 if dtype == torch.complex64 else torch.float64)
    h_t = torch.as_tensor(np.asarray(h_batch, dtype=np.float64), device=device)
    if h_t.shape[1] != geom.n_sites:
        raise ProgramValidationError(f"h_batch has {h_t.shape[1]} sites, geometry has {geom.n_sites}")
    # `validate(program)` above only range-checks `program.h`, which callers batching many
    # samples (e.g. `features.rydberg_features`) treat as a placeholder -- the real per-sample
    # data is `h_batch`, checked here instead of trusting every caller's upstream encoding.
    h_min, h_max = float(h_t.min()), float(h_t.max())
    if h_min < H_LOCAL_MIN - 1e-9 or h_max > H_LOCAL_MAX + 1e-9:
        raise ProgramValidationError(f"h_batch out of [{H_LOCAL_MIN}, {H_LOCAL_MAX}]: min={h_min:.4f} max={h_max:.4f}")
    batch = h_t.shape[0]

    # Delta_global/Delta_local are constant across the whole program by construction
    # (pulses.py), so the diagonal is built once and reused at every RK4 stage.
    diag = _diagonal(geom, h_t, float(program.delta_global[0]), float(program.delta_local[0]), dtype)

    psi = zero_state(batch, geom.dim, device, dtype)
    times = program.times_us
    for seg in range(len(times) - 1):
        t0, t1 = float(times[seg]), float(times[seg + 1])
        o0, o1 = float(program.omega[seg]), float(program.omega[seg + 1])
        phi = float(program.phase[seg])  # constant within a program in pulses.py's builder
        phase_coeff = None if phi == 0.0 else _phase_coeff_table(geom, phi, dtype)
        n_steps = _segment_steps(geom, diag, o0, o1, t1 - t0, rk4_safety)
        dt = (t1 - t0) / n_steps
        for step in range(n_steps):
            t_a = t0 + step * dt
            omega_a = o0 + (o1 - o0) * (t_a - t0) / (t1 - t0) if t1 > t0 else o0
            omega_mid = o0 + (o1 - o0) * (t_a + dt / 2 - t0) / (t1 - t0) if t1 > t0 else o0
            omega_b = o0 + (o1 - o0) * (t_a + dt - t0) / (t1 - t0) if t1 > t0 else o0
            k1 = _hamiltonian_matvec(psi, geom, diag, omega_a, phi, phase_coeff)
            k2 = _hamiltonian_matvec(psi + dt / 2 * k1, geom, diag, omega_mid, phi, phase_coeff)
            k3 = _hamiltonian_matvec(psi + dt / 2 * k2, geom, diag, omega_mid, phi, phase_coeff)
            k4 = _hamiltonian_matvec(psi + dt * k3, geom, diag, omega_b, phi, phase_coeff)
            psi = psi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    drift = (psi.norm(dim=1) - 1.0).abs().amax().item()
    if drift > norm_drift_tol:
        raise AssertionError(f"statevector norm drifted by {drift:.3e} > {norm_drift_tol:g}; "
                             f"tighten rk4_safety or check the Hamiltonian is Hermitian")
    return psi


def occupation_moments(psi: torch.Tensor, geom: GeometryCache):
    """`<n_i>` (batch, n_sites) and `<n_i n_j>` (batch, n_sites, n_sites) upper triangle,
    directly from the population `|psi|^2` -- the observables Gate 1 compares against the
    reference simulators, in the occupation basis (not the Z basis `features.py` will use)."""
    probs = (psi.conj() * psi).real  # (batch, dim)
    n_i = probs @ geom.bits.T  # (batch, n_sites)
    joint = torch.einsum("bi,qi,ri->bqr", probs, geom.bits, geom.bits)  # (batch, n, n), diag = <n_i>
    return n_i, joint
