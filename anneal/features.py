"""The annealing reservoir's feature map: `(x_t, t) -> local fields -> anneal -> <Z>, <ZZ>`.

Deliberately the same shape as `quera/features.rydberg_features`, so the annealing arm slots into
`qrc_fusion_fair_core` exactly where the Rydberg one does and every downstream stage -- readout,
DDIM rollout, PRDC, FID -- is literally unchanged code. The parallel is close because the physics
is: data enters through single-site energies, the dynamics is fixed and untrained, and the readout
is whatever a computational-basis measurement gives you.

    Rydberg (Aquila)                    annealing (D-Wave)
    data -> per-site detuning           data -> local fields h_i
    fixed Omega, C6/r^6 interactions    A(s) transverse field, fixed J_ij
    occupation -> <Z_i>, <Z_i Z_j>      bitstrings -> <Z_i>, <Z_i Z_j>

## Where the probe times come from

The Rydberg arm reads the state at `v_slices` probe *times* within one pulse. Here the analogue is
`t_anneal_us`: a tuple of annealing durations, each a separate anneal read out at the end. That is
not an arbitrary transplant -- it is the paper's own knob. Sakurai et al. (Phys. Rev. Research,
2026) report that **short annealing times give higher classification accuracy and long ones lower
accuracy at lower sampling cost**, so sweeping this tuple is the direct test of whether their
finding survives the move from classification to generation. This project has measured five times
that supervised gains need not transfer to FID, so it is an open question rather than a formality.

## What the couplings are, and why there is a `draw`

`J` is drawn once from a fixed seed and then frozen -- the annealing analogue of the digital arm's
random reservoir unitaries. That gives this arm a `draw` ensemble, which the Rydberg arm does not
have (a Rydberg operating point is a single fixed Hamiltonian, hence `unitary_draw=-1` there). So
the annealing arm pairs against the digital arm's 5-draw protocol rather than the Rydberg arm's
single cell, and its per-seed statistics have the same structure as `ridge_qrc`'s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from anneal.ocean_backend import sample_ising as sample_ising_ocean
from anneal.qutip_backend import sample_ising as sample_ising_qutip
from anneal.sqa_backend import sample_ising as sample_ising_sqa
from anneal.emulator import IsingCache, anneal_split, build_ising_cache
from anneal.schedule import Linear
from quera.cache import cached
from quera.encoding import EncodingAffine

# Advantage2 accepts h in [-4, 4] and J in [-1, 1] after auto-scaling. Staying inside the hardware
# ranges from the first simulated cell means a later hardware run is a backend swap, not a redesign.
H_FIELD_MAX = 4.0
J_COUPLING_MAX = 1.0


def n_features(n_spins: int) -> int:
    return n_spins + n_spins * (n_spins - 1) // 2


def random_couplings(n_spins: int, seed: int, density: float = 1.0, scale: float = 0.5):
    """Fixed `J` upper triangle, clipped to the hardware range. `density` thins the graph.

    A fully connected 12-spin problem needs a size-12 clique embedding on Zephyr, which Advantage2
    has room for; `density < 1` is there for testing sparser, cheaper-to-embed graphs.
    """
    rng = np.random.default_rng(seed)
    j = np.triu(rng.normal(scale=scale, size=(n_spins, n_spins)), 1)
    if density < 1.0:
        j *= (rng.random((n_spins, n_spins)) < density)
    return np.clip(j, -J_COUPLING_MAX, J_COUPLING_MAX)


@dataclass
class AnnealParams:
    """Everything the feature map depends on. Flat and JSON-primitive so it drops into a cache key."""
    encoding: EncodingAffine
    n_spins: int = 12
    draw: int = 0
    t_anneal_us: tuple = (0.00005, 0.0002, 0.0005, 0.001)
    """0.05-5 ns: the window where consecutive durations still give decorrelated feature maps
    (see `schedule.INFORMATIVE_ANNEAL_US` for the measurement). Being short, it is also the cheap
    end -- step count scales with duration, so the informative tuple costs ~8x less than the
    saturated 5-50 ns one it replaced."""
    h_scale: float = 2.0
    j_scale: float = 0.5
    j_density: float = 1.0
    schedule: object = field(default_factory=Linear)
    steps_per_us: float = 1.6e6
    """Strang substeps per microsecond. Set from a convergence measurement at D-Wave energy scales:
    against a 3.2e6 reference, features differ by 3.6e-03 at 2e5, 2.2e-04 at 8e5 and 4.0e-05 at
    1.6e6. The earlier 1e3 default was calibrated before the GHz conversion was fixed and is 3
    orders of magnitude too coarse for the corrected Hamiltonian."""
    dtype: str = "complex64"
    seed: int = 0
    use_cache: bool = True
    backend: str = "dwave-sqa"
    num_reads: int = 100
    dephasing_rate: float = 0.0
    relaxation_rate: float = 0.0
    temperature_k: float = 0.018
    sweeps_per_us: float = 20_000.0
    _cache: IsingCache | None = field(default=None, repr=False, compare=False)

    def couplings(self):
        return random_couplings(self.n_spins, 20260902 + self.draw, self.j_density, self.j_scale)

    def ising_cache(self, device, real_dtype):
        if self._cache is None:
            object.__setattr__(self, "_cache", build_ising_cache(self.couplings(), device, real_dtype))
        return self._cache


def zz_from_probs(probs: torch.Tensor, cache: IsingCache) -> torch.Tensor:
    """`<Z_i>` and `<Z_i Z_j>` (i<j) from the measurement distribution -- all a Z-basis readout gives.

    This is the same restriction Aquila imposes (site occupation only), so the observation that a
    full weight-<=2 Pauli readout is worth ~5 FID on the digital arm applies here too: available in
    simulation, unavailable on the device without basis-rotation machinery neither annealer exposes.
    """
    z = cache.z.to(probs.dtype)                    # (n, dim), +-1
    z_i = probs @ z.T                              # (batch, n)
    iu = torch.triu_indices(cache.n_spins, cache.n_spins, offset=1, device=probs.device)
    zz = torch.einsum("bd,id,jd->bij", probs, z, z)
    return torch.cat([z_i, zz[:, iu[0], iu[1]]], dim=1)


def anneal_features(x: np.ndarray, t, params: AnnealParams, device,
                    shots: int | None = None) -> np.ndarray:
    """Generate Ising features with D-Wave Ocean's official simulator by default."""
    if params.backend in ("dwave-sqa", "qutip-open", "ocean-sa", "dwave-qpu"):
        u = params.encoding.transform(x, t)
        h = np.clip((2.0 * u - 1.0) * params.h_scale, -H_FIELD_MAX, H_FIELD_MAX)
        reads = params.num_reads if shots is None else shots
        iu = np.triu_indices(params.n_spins, 1)
        blocks = []
        # Ocean SA uses sweeps, not physical time. Preserve the configured feature width by
        # mapping the former duration slots monotonically to algorithmic annealing effort.
        for k, anneal_time in enumerate(params.t_anneal_us):
            def block(anneal_time=anneal_time, k=k):
                if params.backend == "dwave-sqa":
                    samples = sample_ising_sqa(
                        h, params.couplings(), params.schedule, num_reads=reads,
                        annealing_time_us=float(anneal_time), temperature_k=params.temperature_k,
                        sweeps_per_us=params.sweeps_per_us, seed=params.seed * 1_000_003 + k)
                elif params.backend == "qutip-open":
                    samples = sample_ising_qutip(
                        h, params.couplings(), params.schedule,
                        annealing_time_us=float(anneal_time), num_reads=reads,
                        dephasing_rate=params.dephasing_rate,
                        relaxation_rate=params.relaxation_rate,
                        seed=params.seed * 1_000_003 + k)
                elif params.backend == "dwave-qpu":
                    from anneal.qpu_backend import sample_ising as sample_ising_qpu
                    samples = sample_ising_qpu(h, params.couplings(), num_reads=reads,
                                               annealing_time_us=float(anneal_time))
                else:
                    sweeps = max(1, round(float(anneal_time) * 1e7))
                    samples = sample_ising_ocean(h, params.couplings(), num_reads=reads,
                                                 num_sweeps=sweeps,
                                                 seed=params.seed * 1_000_003 + k)
                z = samples.mean(axis=1)
                zz = (samples[:, :, :, None] * samples[:, :, None, :]).mean(axis=1)
                return np.concatenate([z, zz[:, iu[0], iu[1]]], axis=1)

            # Only the QPU path is cached. Quota spent on a row is unrecoverable, so a cell that
            # crashes or gets re-run must not pay twice for fields it already annealed; the
            # simulated backends recompute more cheaply than the cache would cost in disk.
            if params.backend == "dwave-qpu":
                blocks.append(cached(block, probe_index=k, t_anneal_us=float(anneal_time), h=h,
                                     j_matrix=params.couplings(), reads=int(reads),
                                     n_spins=int(params.n_spins), reservoir="anneal-qpu"))
            else:
                blocks.append(block())
        return np.concatenate(blocks, axis=1)
    if params.backend != "statevector":
        raise ValueError(f"unknown anneal backend {params.backend!r}")
    # Legacy backend retained only to reproduce artifacts created before the Ocean migration.
    real_dtype = torch.float32 if params.dtype == "complex64" else torch.float64
    torch_dtype = torch.complex64 if params.dtype == "complex64" else torch.complex128
    cache = params.ising_cache(device, real_dtype)

    # `h` in [0,1] from the shared affine encoder, mapped to the symmetric hardware field range.
    u = params.encoding.transform(x, t)
    h = np.clip((2.0 * u - 1.0) * params.h_scale, -H_FIELD_MAX, H_FIELD_MAX)

    blocks = []
    for k, t_us in enumerate(params.t_anneal_us):
        def compute(t_us=t_us, k=k):
            psi = anneal_split(h, cache, params.schedule, float(t_us), device,
                               dtype=torch_dtype, steps_per_us=params.steps_per_us)
            probs = (psi.conj() * psi).real
            if shots is not None:
                gen = torch.Generator(device=device).manual_seed(params.seed * 1_000_003 + k)
                counts = torch.multinomial(probs, shots, replacement=True, generator=gen)
                probs = torch.zeros_like(probs).scatter_add_(
                    1, counts, torch.ones_like(counts, dtype=probs.dtype)) / shots
            return zz_from_probs(probs, cache).cpu().numpy()

        if params.use_cache:
            block = cached(compute, probe_index=k, t_anneal_us=float(t_us), h=h,
                           j_matrix=cache.j_matrix, schedule=params.schedule.key,
                           shots=shots, seed=params.seed, steps_per_us=params.steps_per_us,
                           dtype=params.dtype, reservoir="anneal")
        else:
            block = compute()
        blocks.append(block)
    return np.concatenate(blocks, axis=1)
