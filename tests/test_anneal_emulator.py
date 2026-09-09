from __future__ import annotations

import numpy as np
import torch

from anneal.emulator import anneal_split, build_ising_cache
from anneal.features import AnnealParams, anneal_features, n_features
from anneal.schedule import Linear
from quera.encoding import fit_encoding


def test_zero_duration_keeps_plus_state():
    cache = build_ising_cache(np.zeros((3, 3)), "cpu", torch.float64)
    psi = anneal_split(np.zeros((2, 3)), cache, Linear(), 0.0, "cpu",
                       dtype=torch.complex128, n_steps=16)
    expected = torch.full_like(psi, 1 / np.sqrt(8))
    assert torch.allclose(psi, expected, atol=1e-12)


def test_split_step_halving_converges():
    j = np.array([[0, .3, -.2], [0, 0, .25], [0, 0, 0.]])
    cache = build_ising_cache(j, "cpu", torch.float64)
    h = np.array([[.2, -.4, .7]])
    states = [anneal_split(h, cache, Linear(), .0005, "cpu", dtype=torch.complex128,
                           n_steps=n) for n in (32, 64, 128)]
    coarse = torch.max(torch.abs(states[0] - states[2])).item()
    fine = torch.max(torch.abs(states[1] - states[2])).item()
    assert fine > 0
    assert coarse / fine > 3.0


def test_feature_shape_bounds_and_shot_reproducibility():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 10))
    t = np.array([1, 50, 100, 150])
    params = AnnealParams(encoding=fit_encoding(x, t), n_spins=12, draw=2,
                          t_anneal_us=(.00005, .0001), backend="ocean-sa",
                          seed=11, use_cache=False)
    assert AnnealParams(encoding=fit_encoding(x, t)).backend == "dwave-sqa"
    exact = anneal_features(x, t, params, "cpu")
    sampled_a = anneal_features(x, t, params, "cpu", shots=100)
    sampled_b = anneal_features(x, t, params, "cpu", shots=100)
    assert exact.shape == (4, n_features(12) * 2)
    assert np.all(np.isfinite(exact)) and np.max(np.abs(exact)) <= 1.000001
    assert np.array_equal(sampled_a, sampled_b)
def test_qutip_open_backend_returns_quantum_measurements():
    from anneal.qutip_backend import sample_ising
    samples = sample_ising(np.array([[.2, -.1]]), np.array([[0., .3], [0., 0.]]),
                           Linear(), annealing_time_us=.0001, num_reads=4,
                           dephasing_rate=.01, relaxation_rate=.01, seed=7, n_time_points=11)
    assert samples.shape == (1, 4, 2)
    assert set(np.unique(samples)) <= {-1, 1}
def test_official_dwave_sqa_with_measured_schedule():
    from anneal.schedule import schedule_from_name
    from anneal.sqa_backend import sample_ising
    schedule = schedule_from_name("advantage2-fast")
    samples = sample_ising(np.array([[.2, -.1]]), np.array([[0., .3], [0., 0.]]),
                           schedule, num_reads=4, annealing_time_us=.005,
                           temperature_k=.0154, sweeps_per_us=20000, seed=7)
    assert samples.shape == (1, 4, 2)
    assert set(np.unique(samples)) <= {-1, 1}
