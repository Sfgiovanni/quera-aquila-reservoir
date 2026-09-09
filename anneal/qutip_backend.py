"""Free open-system quantum-annealing approximation implemented with QuTiP."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _operators(n_spins: int):
    import qutip as qt

    ident, sx, sz, sm = qt.qeye(2), qt.sigmax(), qt.sigmaz(), qt.sigmam()

    def local(op, site):
        factors = [ident] * n_spins
        factors[site] = op
        return qt.tensor(factors)

    xs = tuple(local(sx, i) for i in range(n_spins))
    zs = tuple(local(sz, i) for i in range(n_spins))
    sms = tuple(local(sm, i) for i in range(n_spins))
    plus = qt.tensor([qt.basis(2, 0) + qt.basis(2, 1) for _ in range(n_spins)]).unit()
    return xs, zs, sms, plus


def sample_ising(h_batch: np.ndarray, j_matrix: np.ndarray, schedule, *,
                 annealing_time_us: float, num_reads: int, dephasing_rate: float,
                 relaxation_rate: float, seed: int, n_time_points: int = 101) -> np.ndarray:
    """Return computational-basis samples from Monte-Carlo open-system trajectories.

    Rates are phenomenological inverse-microsecond rates.  This is a D-Wave-inspired model,
    not a calibrated reproduction of a particular QPU.
    """
    import qutip as qt

    h_batch = np.asarray(h_batch, dtype=float)
    n_spins = h_batch.shape[1]
    xs, zs, sms, plus = _operators(n_spins)
    hx = -0.5 * sum(xs)
    c_ops = ([np.sqrt(dephasing_rate) * z for z in zs] if dephasing_rate else [])
    c_ops += ([np.sqrt(relaxation_rate) * sm for sm in sms] if relaxation_rate else [])
    times = np.linspace(0.0, float(annealing_time_us), n_time_points)
    rng = np.random.default_rng(seed)
    output = []

    for row_index, fields in enumerate(h_batch):
        hz = sum(float(fields[i]) * zs[i] for i in range(n_spins))
        for i in range(n_spins):
            for j in range(i + 1, n_spins):
                if j_matrix[i, j]:
                    hz += float(j_matrix[i, j]) * zs[i] * zs[j]

        def a_coeff(t, _args=None):
            return schedule(t / annealing_time_us)[0]

        def b_coeff(t, _args=None):
            return 0.5 * schedule(t / annealing_time_us)[1]

        result = qt.mcsolve(
            [[hx, a_coeff], [hz, b_coeff]], plus, times, c_ops, ntraj=num_reads,
            seeds=[int(x) for x in rng.integers(1, 2**31 - 1, size=num_reads)],
            options={"store_states": False, "store_final_state": True,
                     "keep_runs_results": True, "progress_bar": ""},
        )
        states = result.runs_final_states
        draws = []
        for state in states:
            probs = np.abs(state.full().ravel()) ** 2
            basis_index = rng.choice(len(probs), p=probs / probs.sum())
            draws.append([1 - 2 * ((basis_index >> (n_spins - 1 - q)) & 1)
                          for q in range(n_spins)])
        output.append(draws)
    return np.asarray(output, dtype=np.int8)
