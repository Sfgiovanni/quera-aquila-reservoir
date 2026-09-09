"""Closed-system RK4 propagator for the transverse-field Ising annealing Hamiltonian.

    H(s) = -(A(s)/2) sum_i X_i + (B(s)/2) [ sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j ]

Same shape as `quera/emulator.py` -- a diagonal term plus a symmetric single-site flip term -- so
the same batched-gather trick applies: all `n_spins` flips are fetched in one `torch.gather`
instead of `n_spins` sequential ones, which is what makes the per-step cost compute-bound rather
than kernel-launch-bound (see `docs/AQUILA_PORT.md`'s Gate 1 performance note, where the same fix
was worth ~30x on a production-size batch).

Two differences from the Rydberg propagator, both physical rather than cosmetic:

1. **The initial state is the uniform superposition**, the ground state of `-A(0)/2 sum X_i`, not
   `|00...0>`. Starting from a computational basis state would be starting in a random excited
   state of the initial Hamiltonian and would make the whole schedule meaningless.
2. **The diagonal is time-dependent**, scaled by `B(s)`, where the Rydberg diagonal is fixed for a
   whole program. It is built once per batch and rescaled per RK4 stage, so the cost is unchanged.

This is a *closed-system* model: unitary, zero temperature, no `1/f` flux noise, no coupling to the
~15 mK bath a real annealer sits in. It is the ideal limit, useful for establishing whether an
effect exists at all before asking whether hardware would show it. Do not read a simulated result
here as a prediction of D-Wave output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_RK4_SAFETY = 0.05
"""Tighter than the Rydberg arm's 0.2, and measured rather than inherited. Norm drift of the final
state, 12 spins, production fields:

    t_anneal   rk4_safety   drift (complex64)   drift (complex128)
      0.80 us       0.200          1.04e-05            1.09e-05
      0.80 us       0.050          2.26e-06            1.06e-08
      3.20 us       0.200          4.57e-05            4.54e-05
      3.20 us       0.050          5.36e-06            4.44e-08

At 0.2 the two precisions drift *identically*, which identifies the cause as RK4 truncation, not
float32 accumulation -- so loosening `norm_drift_tol` would have hidden a real integration error.
At 0.05 the float64 run converges (1e-8) and the float32 run sits at its own precision floor. The
annealing Hamiltonian needs the tighter step because `B(s)` grows the diagonal over the schedule
while the Rydberg diagonal is constant."""


@dataclass
class IsingCache:
    """Everything that depends on the spin count and the couplings, but not on `h` or time."""
    n_spins: int
    dim: int
    z: torch.Tensor              # (n_spins, dim) float, +1 where spin i is up (bit 0), -1 where down
    flip_index_flat: torch.Tensor  # (n_spins*dim,) int64, index with bit i toggled
    coupling_diag: torch.Tensor  # (dim,) float, sum_{i<j} J_ij z_i z_j
    j_matrix: np.ndarray         # (n_spins, n_spins) upper triangle, for the cache key


def build_ising_cache(j_matrix: np.ndarray, device, real_dtype=torch.float32) -> IsingCache:
    j = np.asarray(j_matrix, dtype=np.float64)
    if j.ndim != 2 or j.shape[0] != j.shape[1]:
        raise ValueError(f"j_matrix must be square, got {j.shape}")
    n = j.shape[0]
    dim = 1 << n
    idx = np.arange(dim)
    # Bit convention matches quera/emulator.py: spin 0 is the most significant bit.
    bits = np.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)]).astype(np.float64)  # (n, dim)
    z_np = 1.0 - 2.0 * bits
    flip_np = np.stack([idx ^ (1 << (n - 1 - q)) for q in range(n)])

    coupling = np.zeros(dim)
    for a in range(n):
        for b in range(a + 1, n):
            if j[a, b] != 0.0:
                coupling += j[a, b] * z_np[a] * z_np[b]

    t = lambda arr, dt: torch.as_tensor(arr, dtype=dt, device=device)
    return IsingCache(n_spins=n, dim=dim, z=t(z_np, real_dtype),
                      flip_index_flat=t(flip_np.reshape(-1), torch.int64),
                      coupling_diag=t(coupling, real_dtype), j_matrix=j)


def plus_state(batch: int, dim: int, device, dtype=torch.complex64) -> torch.Tensor:
    """`|+>^{(x)n}`, the ground state of the initial transverse-field Hamiltonian."""
    return torch.full((batch, dim), 1.0 / np.sqrt(dim), dtype=dtype, device=device)


def _matvec(psi, cache: IsingCache, diag, a_val: float, b_val: float):
    """`-i H(s) psi` with `H(s) = -(A/2) sum X + (B/2) * diag`."""
    out = (b_val / 2.0) * (diag * psi)
    if a_val != 0.0:
        idx = cache.flip_index_flat.unsqueeze(0).expand(psi.shape[0], -1)
        flipped = torch.gather(psi, 1, idx).view(psi.shape[0], cache.n_spins, cache.dim)
        out = out - (a_val / 2.0) * flipped.sum(dim=1)
    return -1j * out


def problem_diagonal(cache: IsingCache, h_batch: torch.Tensor, dtype) -> torch.Tensor:
    """`(batch, dim)`: `sum_i h_i z_i + sum_{i<j} J_ij z_i z_j`, the `B(s)=1` problem energy."""
    local = h_batch.to(cache.z.dtype) @ cache.z            # (batch, dim)
    return (local + cache.coupling_diag[None, :]).to(dtype)


def _apply_transverse(psi, cache: IsingCache, theta: float):
    """`exp(+i theta sum_i X_i) psi`, applied exactly.

    The `X_i` commute and share an angle, so the operator factorises into identical single-spin
    rotations `cos(theta) I + i sin(theta) X`. Each is applied by one bit-flip gather, which makes
    this step *exactly unitary* -- no norm drift is possible, at any step size.

    Written as `psi + [(cos(theta)-1) psi + i sin(theta) psi_flip]` with `cos(theta)-1` evaluated as
    `-2 sin^2(theta/2)`. That form matters at the step sizes this schedule needs: `theta` falls to
    ~1e-3 rad, where `cos(theta)` rounds to within a couple of float32 ulps of 1.0 and the *deviation*
    -- the part that actually preserves the norm -- is destroyed by cancellation. Computing it
    directly as a half-angle sine keeps full relative precision, and measured norm drift over 16k
    steps drops from 1.1e-03 to the float32 floor.
    """
    cm1 = -2.0 * float(np.sin(theta / 2.0)) ** 2   # cos(theta) - 1, without cancellation
    s = float(np.sin(theta))
    for q in range(cache.n_spins):
        idx = cache.flip_index_flat[q * cache.dim:(q + 1) * cache.dim]
        psi = psi + (cm1 * psi + (1j * s) * psi[:, idx])
    return psi


def anneal_split(h_batch, cache: IsingCache, schedule, t_anneal_us: float, device,
                 dtype=torch.complex64, n_steps: int | None = None, steps_per_us: float = 800.0,
                 norm_drift_tol: float | None = None) -> torch.Tensor:
    """Strang-split propagator: the integrator this Hamiltonian actually wants.

    RK4 on `H(s)` is dominated by the *diagonal* energy scale: `B_max/2 * max|diag|` reaches
    ~1300 rad/us at 12 spins with hardware field ranges, so a Gershgorin-bounded step forces ~96k
    substeps for a 3.2 us anneal -- measured at over 900 s for a batch of 256, i.e. unusable in a
    rollout that needs millions of anneals.

    Splitting fixes both halves at once. `exp(-i H_diag dt)` is a diagonal phase, applied exactly
    in one elementwise multiply regardless of how large the diagonal is; `exp(+i (A/2) dt sum X)`
    factorises into exact single-spin rotations. Neither half is approximated, so **the propagator
    is unitary by construction** and the only error is the `O(dt^3)` Strang commutator -- which
    scales with `||[H_x, H_z]||`, not with `||H_z||` alone. That is why the step size stops being
    hostage to the problem energy.

    `steps_per_us` sets the resolution; `n_steps` overrides it outright.
    """
    h_t = torch.as_tensor(np.asarray(h_batch, dtype=np.float64), device=device)
    if h_t.shape[1] != cache.n_spins:
        raise ValueError(f"h_batch has {h_t.shape[1]} spins, cache has {cache.n_spins}")
    diag = problem_diagonal(cache, h_t, dtype)
    steps = int(n_steps if n_steps is not None else max(16, round(steps_per_us * t_anneal_us)))
    dt = t_anneal_us / steps

    psi = plus_state(h_t.shape[0], cache.dim, device, dtype)
    for step in range(steps):
        s_mid = (step + 0.5) / steps
        a_val, b_val = schedule(s_mid)
        # Strang: half a transverse kick, a full diagonal phase, half a transverse kick.
        psi = _apply_transverse(psi, cache, a_val * dt / 4.0)
        psi = torch.exp(-1j * (b_val / 2.0) * dt * diag) * psi
        psi = _apply_transverse(psi, cache, a_val * dt / 4.0)

    # The propagator is exactly unitary, so any drift is the working precision's accumulation over
    # `steps`, not discretisation. Measured over 3200 steps at 12 spins: 7.2e-05 in complex64 and
    # 2.2e-14 in complex128 -- i.e. entirely float32's floor, which is why the tolerance is
    # dtype-aware rather than a single constant. A drift beyond these floors *would* be a bug.
    if norm_drift_tol is None:
        norm_drift_tol = 1e-3 if dtype == torch.complex64 else 1e-9
    drift = (psi.norm(dim=1) - 1.0).abs().amax().item()
    if drift > norm_drift_tol:
        raise AssertionError(
            f"norm drifted by {drift:.3e} > {norm_drift_tol:g} in a propagator that is unitary by "
            f"construction; this is past the working precision's floor, so it indicates a bug "
            f"rather than a step-size problem")
    # Renormalise before the caller squares it: a uniform norm error of ~1e-4 would otherwise scale
    # every probability, and hence every feature, by a small systematic factor.
    return psi / psi.norm(dim=1, keepdim=True)


def anneal(h_batch, cache: IsingCache, schedule, t_anneal_us: float, device,
           dtype=torch.complex64, rk4_safety: float = DEFAULT_RK4_SAFETY,
           min_steps: int = 64, norm_drift_tol: float = 1e-5) -> torch.Tensor:
    """Propagate `|+>^{(x)n}` through the schedule, batched over per-sample local fields.

    `h_batch` is `(batch, n_spins)` -- one field pattern per sample, which is where the data
    enters. `J` is shared by the whole batch and lives in `cache`, exactly as the Rydberg arm
    shares geometry across a batch and varies only the detuning.
    """
    h_t = torch.as_tensor(np.asarray(h_batch, dtype=np.float64), device=device)
    if h_t.shape[1] != cache.n_spins:
        raise ValueError(f"h_batch has {h_t.shape[1]} spins, cache has {cache.n_spins}")
    diag = problem_diagonal(cache, h_t, dtype)

    # Gershgorin bound on ||H(s)||: the diagonal peaks at B_max, the drive at n*A_max/2.
    a0, b0 = schedule(0.0)
    a1, b1 = schedule(1.0)
    diag_max = float(diag.abs().amax().real if torch.is_complex(diag) else diag.abs().amax())
    h_bound = max(abs(b0), abs(b1)) / 2.0 * diag_max + cache.n_spins * max(abs(a0), abs(a1)) / 2.0
    n_steps = max(min_steps, int(np.ceil(t_anneal_us / (rk4_safety / h_bound)))) if h_bound > 0 else min_steps

    psi = plus_state(h_t.shape[0], cache.dim, device, dtype)
    dt = t_anneal_us / n_steps
    for step in range(n_steps):
        s0 = step / n_steps
        sm = (step + 0.5) / n_steps
        s1 = (step + 1) / n_steps
        a_a, b_a = schedule(s0)
        a_m, b_m = schedule(sm)
        a_b, b_b = schedule(s1)
        k1 = _matvec(psi, cache, diag, a_a, b_a)
        k2 = _matvec(psi + dt / 2 * k1, cache, diag, a_m, b_m)
        k3 = _matvec(psi + dt / 2 * k2, cache, diag, a_m, b_m)
        k4 = _matvec(psi + dt * k3, cache, diag, a_b, b_b)
        psi = psi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    drift = (psi.norm(dim=1) - 1.0).abs().amax().item()
    if drift > norm_drift_tol:
        raise AssertionError(f"statevector norm drifted by {drift:.3e} > {norm_drift_tol:g}; "
                             f"tighten rk4_safety (currently {rk4_safety}) or shorten t_anneal")
    return psi
