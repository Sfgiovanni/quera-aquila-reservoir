"""Adapter for D-Wave's official Ocean simulated-annealing sampler."""
from __future__ import annotations

import numpy as np


def sample_ising(h_batch: np.ndarray, j_matrix: np.ndarray, *, num_reads: int,
                 num_sweeps: int, seed: int) -> np.ndarray:
    """Return one ``(num_reads, n_spins)`` spin array for each input row."""
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError as exc:
        raise RuntimeError("Install the official sampler: python -m pip install dwave-samplers") from exc
    h_batch = np.asarray(h_batch, dtype=np.float64)
    j_matrix = np.asarray(j_matrix, dtype=np.float64)
    n_spins = h_batch.shape[1]
    couplings = {(i, j): float(j_matrix[i, j]) for i in range(n_spins)
                 for j in range(i + 1, n_spins) if j_matrix[i, j] != 0.0}
    sampler = SimulatedAnnealingSampler()
    samples = []
    for row_index, fields in enumerate(h_batch):
        response = sampler.sample_ising(
            {i: float(fields[i]) for i in range(n_spins)}, couplings,
            num_reads=num_reads, num_sweeps=num_sweeps,
            seed=(seed + row_index) % (2**31 - 1),
        )
        order = np.argsort(np.asarray(response.variables))
        samples.append(response.record.sample[:, order])
    return np.stack(samples).astype(np.int8, copy=False)
