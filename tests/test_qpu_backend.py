"""Hardware-path tests that never touch a QPU.

`MockDWaveSampler` exposes the same properties, node/edge lists and `sample()` contract as a real
solver, so everything except the physics is exercised here: tiling, embedding, the packing of many
rows into one submission, unembedding back to per-row spins, and -- most importantly -- that the
budget guard refuses to submit rather than overspending.
"""
from __future__ import annotations

import numpy as np
import pytest

from anneal.qpu_budget import BudgetExceeded, QpuBudget, estimate_access_time_us
from anneal.qpu_backend import QpuReservoir, find_tiles

N_SPINS = 6          # small: keeps the mock's own annealing fast, tiling logic is size-independent


@pytest.fixture(scope="module")
def sampler():
    from dwave.system.testing import MockDWaveSampler
    # Zephyr Z4 is far smaller than Advantage2 but has the same topology family, so embeddings
    # have the same shape; the tests care about counts and wiring, not chip size.
    return MockDWaveSampler(topology_type="zephyr", topology_shape=[4, 4])


@pytest.fixture
def reservoir(sampler, tmp_path):
    return QpuReservoir(n_spins=N_SPINS, sampler=sampler, max_tiles=8,
                        budget=QpuBudget(tmp_path / "ledger.json"),
                        tiles=find_tiles(sampler.to_networkx_graph(), N_SPINS, max_tiles=8))


def test_tiles_are_disjoint(reservoir):
    """Tiles must share no qubits, or rows submitted together would couple into one problem."""
    seen = set()
    for tile in reservoir.tiles:
        qubits = {q for chain in tile.values() for q in chain}
        assert not (qubits & seen), "tiles overlap: rows would contaminate each other"
        seen |= qubits


def test_tiles_cover_all_logical_spins(reservoir):
    for tile in reservoir.tiles:
        assert set(tile) == set(range(N_SPINS))


def test_sample_shape_and_spin_values(reservoir):
    h = np.random.default_rng(0).uniform(-2, 2, size=(5, N_SPINS))
    j = np.zeros((N_SPINS, N_SPINS))
    out = reservoir.sample(h, j, num_reads=4, anneal_us=0.005)
    assert out.shape == (5, 4, N_SPINS)
    assert set(np.unique(out)).issubset({-1, 1})


def test_strong_fields_polarise_each_spin_independently(reservoir):
    """The physics check: with `J = 0` and large `|h|`, spin `i` must sit at `-sign(h_i)`.

    This is what proves the *plumbing* -- that row `r`'s fields reached tile `r` and came back
    labelled as row `r`. A transposed batch, a mis-ordered unembedding or a tile mix-up all break
    this while still returning a correctly shaped array of valid spins.
    """
    h = np.array([[+3.9] * N_SPINS,
                  [-3.9] * N_SPINS,
                  [+3.9, -3.9] * (N_SPINS // 2)], dtype=float)
    j = np.zeros((N_SPINS, N_SPINS))
    out = reservoir.sample(h, j, num_reads=16, anneal_us=0.005)
    mean_z = out.mean(axis=1)
    assert np.all(mean_z[0] < -0.5), "positive h must drive <Z> negative"
    assert np.all(mean_z[1] > +0.5), "negative h must drive <Z> positive"
    assert np.all(np.sign(mean_z[2]) == -np.sign(h[2])), "per-site fields got scrambled"


def test_rows_beyond_tile_count_are_split_across_submissions(reservoir):
    """More rows than tiles must still come back one-to-one, in order."""
    n_rows = len(reservoir.tiles) * 2 + 3
    h = np.tile(np.array([+3.9, -3.9] * (N_SPINS // 2)), (n_rows, 1))
    h[::2] *= -1.0
    out = reservoir.sample(h, np.zeros((N_SPINS, N_SPINS)), num_reads=8, anneal_us=0.005)
    assert out.shape == (n_rows, 8, N_SPINS)
    assert np.all(np.sign(out.mean(axis=1)) == -np.sign(h)), "row order lost across submissions"
    assert reservoir.telemetry["submissions"] == 3
    assert reservoir.telemetry["rows"] == n_rows


def test_unreachable_anneal_time_is_refused(reservoir):
    """The simulated arm's informative window is below every hardware floor -- fail, don't round."""
    with pytest.raises(ValueError, match="outside this solver"):
        reservoir.sample(np.zeros((1, N_SPINS)), np.zeros((N_SPINS, N_SPINS)),
                         num_reads=4, anneal_us=0.00005)


# -- budget ------------------------------------------------------------------------------------
def test_budget_refuses_before_submitting(sampler, tmp_path):
    """A cell that would breach the cap must raise instead of spending."""
    tiny = QpuBudget(tmp_path / "tiny.json", cap_seconds=0.001)
    res = QpuReservoir(n_spins=N_SPINS, sampler=sampler, max_tiles=4, budget=tiny,
                       tiles=find_tiles(sampler.to_networkx_graph(), N_SPINS, max_tiles=4))
    with pytest.raises(BudgetExceeded):
        res.sample(np.zeros((1, N_SPINS)), np.zeros((N_SPINS, N_SPINS)),
                   num_reads=8, anneal_us=0.005)
    assert res.telemetry["submissions"] == 0, "nothing may be submitted once the cap is hit"


def test_budget_accumulates_across_instances(tmp_path):
    """Two runs sharing a ledger see one balance -- the quota is monthly, not per-process."""
    path = tmp_path / "shared.json"
    first = QpuBudget(path, cap_seconds=10.0, margin=0.0)
    with first.charge(4e6, "a") as record:
        record(4e6)
    second = QpuBudget(path, cap_seconds=10.0, margin=0.0)
    assert second.spent_seconds() == pytest.approx(4.0)
    assert second.remaining_seconds() == pytest.approx(6.0)


def test_crash_leaves_estimate_charged(tmp_path):
    """An unreported submission stays charged at its estimate: over-count, never under-count."""
    budget = QpuBudget(tmp_path / "crash.json", cap_seconds=10.0, margin=0.0)
    with pytest.raises(RuntimeError):
        with budget.charge(2e6, "boom"):
            raise RuntimeError("network died after the problem was accepted")
    assert budget.spent_seconds() == pytest.approx(2.0)


def test_margin_is_untouchable(tmp_path):
    budget = QpuBudget(tmp_path / "margin.json", cap_seconds=100.0, margin=0.10)
    assert budget.remaining_seconds() == pytest.approx(90.0)


def test_estimate_prefers_solver_timing_data():
    """When the solver publishes its timing model, the estimate must use it over the fallbacks."""
    props = {"problem_timing_data": {
        "typical_programming_time": 1000.0, "default_programming_thermalization": 0.0,
        "qpu_delay_time_per_sample": 20.0, "readout_time_model_parameters": [30.0, 40.0]}}
    got = estimate_access_time_us(props, num_reads=10, anneal_us=0.005)
    assert got == pytest.approx(1000.0 + 10 * (0.005 + 40.0 + 20.0))
    bare = estimate_access_time_us({}, num_reads=10, anneal_us=0.005)
    assert bare > got, "fallbacks must be the conservative side"


# -- pipeline integration ----------------------------------------------------------------------
def test_anneal_features_matches_simulated_arm_width(tmp_path):
    """The QPU arm must produce the same feature block the SQA arm does, or the readout, the
    dimension-matched random control and every published contrast stop lining up."""
    from dwave.system.testing import MockDWaveSampler
    from anneal.features import AnnealParams, anneal_features, n_features
    from anneal.qpu_backend import QpuReservoir, set_reservoir
    from anneal.schedule import schedule_from_name
    from quera.encoding import fit_encoding

    mock = MockDWaveSampler(topology_type="zephyr", topology_shape=[6, 4])
    tiles = find_tiles(mock.to_networkx_graph(), 12, max_tiles=12)
    reservoir = QpuReservoir(n_spins=12, sampler=mock, tiles=tiles,
                             budget=QpuBudget(tmp_path / "ledger.json"))
    set_reservoir(reservoir)
    try:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(24, 10)).astype(np.float32)
        t = rng.integers(1, 200, 24)
        params = AnnealParams(encoding=fit_encoding(x, t, t_scale=1.0), n_spins=12, draw=0,
                              t_anneal_us=(0.005,), backend="dwave-qpu", num_reads=8,
                              schedule=schedule_from_name("advantage2-fast"), seed=0)
        features = anneal_features(x, t, params, "cpu")
        assert features.shape == (24, n_features(12))
        assert np.isfinite(features).all()
        assert np.abs(features).max() <= 1.0, "<Z> and <ZZ> are bounded by 1"

        # Quota spent is unrecoverable, so a repeated call must hit the cache, not the solver.
        before = reservoir.telemetry["submissions"]
        again = anneal_features(x, t, params, "cpu")
        assert np.array_equal(features, again)
        assert reservoir.telemetry["submissions"] == before, "re-run re-paid for cached fields"
    finally:
        set_reservoir(None)


# -- field scaling -----------------------------------------------------------------------------
def test_chain_strength_capped_to_extended_j_range(sampler, tmp_path):
    """The production `J` wants a chain strength above what the solver takes without rescaling.

    Uncapped, `auto_scale` would shrink the data-carrying `h` from +-4.0 to +-1.74 -- a 2.3x loss of
    signal that appears only on hardware and reads as decoherence. Capping trades that for chain
    breaks, which are measured.
    """
    import dimod
    from anneal.features import random_couplings

    res = QpuReservoir(n_spins=12, sampler=sampler, max_tiles=1,
                       budget=QpuBudget(tmp_path / "l.json"),
                       tiles=find_tiles(sampler.to_networkx_graph(), 12, max_tiles=1))
    j = random_couplings(12, 20260902, 1.0, 0.5)
    couplings = {(i, k): float(j[i, k]) for i in range(12) for k in range(i + 1, 12) if j[i, k]}
    source = dimod.BinaryQuadraticModel.from_ising({i: 4.0 for i in range(12)}, couplings)
    strength = res._resolve_chain_strength(source, res.tiles[0])
    assert strength <= res.max_chain_strength()
    assert res.max_chain_strength() == pytest.approx(2.0)   # extended_j_range floor, not j_range's
    assert res.telemetry["chain_strength_capped"] == 1


def test_explicit_chain_strength_is_respected(sampler, tmp_path):
    import dimod
    res = QpuReservoir(n_spins=N_SPINS, sampler=sampler, max_tiles=1, chain_strength=0.75,
                       budget=QpuBudget(tmp_path / "l.json"),
                       tiles=find_tiles(sampler.to_networkx_graph(), N_SPINS, max_tiles=1))
    source = dimod.BinaryQuadraticModel.from_ising({i: 1.0 for i in range(N_SPINS)}, {})
    assert res._resolve_chain_strength(source, res.tiles[0]) == pytest.approx(0.75)


def test_out_of_range_fields_are_refused(reservoir):
    """With auto_scale off, an over-range field must fail here rather than at the solver."""
    with pytest.raises(ValueError, match="h_range"):
        reservoir.sample(np.full((1, N_SPINS), 99.0), np.zeros((N_SPINS, N_SPINS)),
                         num_reads=4, anneal_us=0.005)
    with pytest.raises(ValueError, match="usable range"):
        reservoir.sample(np.zeros((1, N_SPINS)), np.full((N_SPINS, N_SPINS), 5.0),
                         num_reads=4, anneal_us=0.005)
