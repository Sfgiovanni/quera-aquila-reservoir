"""D-Wave's free path-integral quantum-annealing simulator."""
from __future__ import annotations

import numpy as np

KB_OVER_H_GHZ_PER_K = 20.8366191


def sample_ising(h_batch: np.ndarray, j_matrix: np.ndarray, schedule, *, num_reads: int,
                 annealing_time_us: float, temperature_k: float, sweeps_per_us: float,
                 seed: int, schedule_points: int = 101) -> np.ndarray:
    from dwave.samplers import PathIntegralAnnealingSampler

    h_batch = np.asarray(h_batch, dtype=float)
    n_spins = h_batch.shape[1]
    couplings = {(i, j): float(j_matrix[i, j]) for i in range(n_spins)
                 for j in range(i + 1, n_spins) if j_matrix[i, j] != 0.0}
    s_grid = np.linspace(0.0, 1.0, schedule_points)
    ab_ghz = np.asarray([schedule(float(s)) for s in s_grid]) / (2 * np.pi * 1000.0)
    beta_ghz = 1.0 / (KB_OVER_H_GHZ_PER_K * temperature_k)
    hp_field = beta_ghz * ab_ghz[:, 1] / 2.0
    hd_field = beta_ghz * ab_ghz[:, 0] / 2.0
    sweeps_per_point = max(1, round(annealing_time_us * sweeps_per_us / schedule_points))
    sampler = PathIntegralAnnealingSampler()
    output = []
    for row_index, fields in enumerate(h_batch):
        response = sampler.sample_ising(
            {i: float(fields[i]) for i in range(n_spins)}, couplings,
            num_reads=num_reads, beta_schedule_type="custom",
            Hp_field=hp_field, Hd_field=hd_field,
            num_sweeps_per_beta=sweeps_per_point,
            seed=(seed + row_index) % (2**31 - 1))
        order = np.argsort(np.asarray(response.variables))
        output.append(response.record.sample[:, order])
    return np.asarray(output, dtype=np.int8)
