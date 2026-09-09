"""Real-hardware backend: the annealing reservoir on a D-Wave QPU.

Signature-compatible with `anneal.ocean_backend` and `anneal.sqa_backend`, so `anneal/features.py`
dispatches to it with a three-line branch and every downstream stage -- readout, DDIM rollout,
PRDC, FID -- is unchanged code.

## Why tiling is the whole design

Data enters this reservoir through the local fields `h_i`, so **every row of the batch is a
different Ising problem**. On a QPU a different problem means a different programming cycle, and
programming (~15 ms) dominates everything else: at the published operating point -- one 5 ns probe,
8 reads -- the anneal-and-readout part of a submission is ~0.3 ms against ~16 ms of programming.
Submitting one row at a time would spend 98% of the quota programming the chip.

The fix is that the reservoir is *small* and the chip is *large*. A 12-spin problem needs a K12
clique embedding, which on Zephyr takes ~35 qubits (chains of ~3). An Advantage2 has thousands, so
~100+ disjoint copies of the problem fit side by side. `find_tiles` packs them greedily, and one
submission then carries one row per tile: the programming cost is paid once and amortised across
all of them. Measured on a defect-free Zephyr Z15,4 graph this is **148 tiles**, which turns a
515k-row cell from ~2.2 h of QPU time into ~1 min.

The tiles are independent by construction -- disjoint qubit sets, no couplers between them -- so
this is not an approximation. It is the same physics run in parallel regions of one chip.

## What differs from the simulated arms, physically

`anneal/emulator.py` is closed-system, unitary and at zero temperature. A QPU is none of those: it
sits at ~15 mK, it decoheres, and its `h`/`J` are analog values with calibration error. It also
breaks chains -- a logical spin represented by several physical qubits can come back disagreeing --
which `unembed` resolves by majority vote. `chain_break_fraction` is returned alongside the samples
rather than discarded, because it is the one number that says whether the embedding, not the
physics, produced the result.

**Annealing time.** The published cell uses a single 5 ns probe. That is exactly the bottom of
Advantage2's `fast_anneal_time_range` and *below* its standard `annealing_time_range` floor of
0.5 us, so this backend submits with `fast_anneal=True` and refuses a duration the solver cannot
reach rather than silently rounding it up to something 100x longer.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from anneal.qpu_budget import QpuBudget, estimate_access_time_us

DEFAULT_TILE_CACHE = Path(__file__).resolve().parent / "data" / "qpu_tiles"


# ---------------------------------------------------------------------------------------------
# embedding / tiling
# ---------------------------------------------------------------------------------------------
def _graph_fingerprint(nodelist, edgelist) -> str:
    """Identify a specific chip *calibration*, not just a solver name.

    Qubits and couplers get taken out of service between calibrations, and an embedding cached
    against yesterday's working graph can reference a qubit that no longer exists. Keying the cache
    on the actual node/edge sets makes that a cache miss instead of a submission error.
    """
    digest = hashlib.sha256()
    digest.update(repr(sorted(nodelist)).encode())
    digest.update(repr(sorted(map(sorted, edgelist))).encode())
    return digest.hexdigest()[:16]


def find_tiles(target_graph, n_spins: int, max_tiles: int | None = None, verbose: bool = False):
    """Greedily pack disjoint `n_spins`-clique embeddings into `target_graph`.

    Returns a list of embeddings, each a `{logical_index: [physical_qubits]}` dict. Greedy is the
    right algorithm here: the tiles are interchangeable (every one runs the same `J`, only `h`
    differs), so there is nothing to gain from a globally optimal packing -- only the count matters,
    and each additional tile is a strict win.
    """
    from minorminer.busclique import find_clique_embedding

    remaining = target_graph.copy()
    tiles: list[dict] = []
    while max_tiles is None or len(tiles) < max_tiles:
        try:
            embedding = find_clique_embedding(n_spins, remaining)
        except Exception:                       # busclique raises when the graph is exhausted
            break
        if not embedding:
            break
        qubits = [q for chain in embedding.values() for q in chain]
        tiles.append({int(k): [int(q) for q in v] for k, v in embedding.items()})
        remaining.remove_nodes_from(qubits)
        if verbose and len(tiles) % 25 == 0:
            print(f"  {len(tiles)} tiles packed")
    return tiles


def load_or_build_tiles(sampler, n_spins: int, cache_dir: Path = DEFAULT_TILE_CACHE,
                        max_tiles: int | None = None, verbose: bool = False):
    """Tiles for this chip's current calibration, from disk when they are still valid."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _graph_fingerprint(sampler.nodelist, sampler.edgelist)
    name = getattr(getattr(sampler, "solver", None), "name", "unknown")
    path = cache_dir / f"{name}_n{n_spins}_{fingerprint}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        tiles = [{int(k): v for k, v in tile.items()} for tile in payload["tiles"]]
        if max_tiles is not None:
            tiles = tiles[:max_tiles]
        return tiles
    tiles = find_tiles(sampler.to_networkx_graph(), n_spins, max_tiles, verbose)
    path.write_text(json.dumps({"solver": name, "fingerprint": fingerprint,
                                "n_spins": n_spins, "tiles": tiles}, indent=1))
    return tiles


# ---------------------------------------------------------------------------------------------
# the reservoir
# ---------------------------------------------------------------------------------------------
@dataclass
class QpuReservoir:
    """A configured QPU sampler plus its tiling, its budget, and the run's telemetry."""

    n_spins: int = 12
    budget: QpuBudget | None = None
    max_tiles: int | None = None
    chain_strength: float | None = None
    fast_anneal: bool = True
    label: str = "anneal-reservoir"
    sampler: object | None = None
    tiles: list = field(default_factory=list)
    telemetry: dict = field(default_factory=lambda: {
        "submissions": 0, "rows": 0, "qpu_access_time_us": 0.0,
        "chain_break_fraction_sum": 0.0, "chain_break_rows": 0,
        "chain_strength": 0.0, "chain_strength_capped": 0})
    _adjacency: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.sampler is None:
            from dwave.system import DWaveSampler
            try:
                self.sampler = DWaveSampler()
            except ValueError as exc:
                if "token" not in str(exc).lower():
                    raise
                raise SystemExit(
                    "No D-Wave API token configured, so this arm has no QPU to run on.\n"
                    "  Fix:  dwave config create      (or: export DWAVE_API_TOKEN=...)\n"
                    "  Plan without one: MOCK=1 ./run_dwave_qpu_cell.sh --dry-run") from exc
        if not self.tiles:
            self.tiles = load_or_build_tiles(self.sampler, self.n_spins, max_tiles=self.max_tiles)
        if not self.tiles:
            raise RuntimeError(f"no {self.n_spins}-spin clique embedding fits this solver")

    @property
    def adjacency(self) -> dict:
        """Target adjacency, built once. `to_networkx_graph()` walks the whole chip, and this is
        needed on every submission -- rebuilding it per batch was ~3,500 chip-sized graph builds."""
        if self._adjacency is None:
            graph = self.sampler.to_networkx_graph()
            self._adjacency = {q: set(graph[q]) for q in self.sampler.nodelist}
        return self._adjacency

    def max_chain_strength(self) -> float:
        """Strongest ferromagnetic chain coupler this solver takes without rescaling the problem.

        Chain couplers are negative, so the binding limit is the *floor* of `extended_j_range`
        (typically -2.0) rather than `j_range`'s -1.0. Staying inside it is what lets us submit with
        `auto_scale=False` -- see `_resolve_chain_strength` for why that matters.
        """
        props = self.sampler.properties
        window = props.get("extended_j_range") or props.get("j_range") or [-1.0, 1.0]
        return abs(float(window[0]))

    def _resolve_chain_strength(self, source, tile) -> float:
        """Chain strength, capped so the embedded problem never triggers auto-scaling.

        `uniform_torque_compensation` asks for ~2.30 on the production `J` (12 spins, `j_scale`
        0.5), above even the extended range's 2.0. Left alone with `auto_scale=True`, Ocean would
        rescale the *entire* problem to fit -- shrinking the data-carrying fields `h` from +-4.0 to
        +-1.74, a 2.3x cut in signal against a fixed thermal noise floor, silently and only on
        hardware. That would degrade the hardware arm for a reason that has nothing to do with
        annealing physics and would read as decoherence.

        So we cap instead, and submit with `auto_scale=False`. The cap buys weaker chains, which
        cost chain *breaks* -- and unlike a silent rescaling, chain breaks are measured and reported
        on every cell as `qpu_mean_chain_break_fraction`.
        """
        from dwave.embedding.chain_strength import uniform_torque_compensation

        if self.chain_strength is not None:
            return float(self.chain_strength)
        wanted = float(uniform_torque_compensation(source, tile))
        allowed = self.max_chain_strength()
        if wanted > allowed:
            self.telemetry["chain_strength_capped"] += 1
            return allowed
        return wanted

    # -- validation ------------------------------------------------------------------------
    def check_anneal_time(self, anneal_us: float) -> float:
        """Reject a duration the hardware cannot produce instead of letting it be rounded."""
        props = self.sampler.properties
        key = "fast_anneal_time_range" if self.fast_anneal else "annealing_time_range"
        window = props.get(key)
        if window is None:
            return float(anneal_us)
        low, high = float(window[0]), float(window[1])
        if not (low <= float(anneal_us) <= high):
            raise ValueError(
                f"annealing_time={anneal_us} us is outside this solver's {key} [{low}, {high}]. "
                f"The simulated arm's INFORMATIVE_ANNEAL_US window (0.05-1 ns) is below every "
                f"hardware floor; use --anneal-times-us 0.005 (the published cell's operating "
                f"point, exactly the fast-anneal minimum) or a duration inside the range above.")
        return float(anneal_us)

    def _check_ranges(self, h_batch: np.ndarray, j_matrix: np.ndarray) -> None:
        """Fail loudly on out-of-range fields instead of letting the solver quietly rescale them.

        With `auto_scale=False` an out-of-range value is an error rather than a silent shrink, but
        the solver's message points at physical qubits after embedding; this one points at the
        feature map's own clipping constants, which is where the fix belongs.
        """
        props = self.sampler.properties
        h_low, h_high = props.get("h_range", [-4.0, 4.0])
        j_low, j_high = props.get("extended_j_range") or props.get("j_range") or [-1.0, 1.0]
        if h_batch.size and (h_batch.min() < h_low or h_batch.max() > h_high):
            raise ValueError(
                f"h in [{h_batch.min():.3f}, {h_batch.max():.3f}] is outside the solver's h_range "
                f"[{h_low}, {h_high}]; lower AnnealParams.h_scale or H_FIELD_MAX in anneal/features.py")
        off = j_matrix[np.triu_indices(self.n_spins, 1)]
        if off.size and (off.min() < j_low or off.max() > j_high):
            raise ValueError(
                f"J in [{off.min():.3f}, {off.max():.3f}] is outside the solver's usable range "
                f"[{j_low}, {j_high}]; lower AnnealParams.j_scale or J_COUPLING_MAX")

    def estimate_us(self, n_rows: int, num_reads: int, anneal_us: float) -> float:
        """Projected QPU access time for `n_rows`, at the current tiling."""
        submissions = int(np.ceil(n_rows / len(self.tiles)))
        per = estimate_access_time_us(self.sampler.properties, num_reads=num_reads,
                                      anneal_us=anneal_us)
        return submissions * per

    # -- sampling --------------------------------------------------------------------------
    def sample(self, h_batch: np.ndarray, j_matrix: np.ndarray, *, num_reads: int,
               anneal_us: float) -> np.ndarray:
        """`(n_rows, num_reads, n_spins)` spins, one row per input field vector."""
        import dimod
        from dwave.embedding import embed_bqm, unembed_sampleset

        h_batch = np.atleast_2d(np.asarray(h_batch, dtype=float))
        j_matrix = np.asarray(j_matrix, dtype=float)
        n_rows = h_batch.shape[0]
        anneal_us = self.check_anneal_time(anneal_us)
        self._check_ranges(h_batch, j_matrix)

        couplings = {(i, j): float(j_matrix[i, j])
                     for i in range(self.n_spins)
                     for j in range(i + 1, self.n_spins) if j_matrix[i, j] != 0.0}
        adjacency = self.adjacency

        out = np.empty((n_rows, num_reads, self.n_spins), dtype=np.int8)
        n_tiles = len(self.tiles)

        for start in range(0, n_rows, n_tiles):
            group = h_batch[start:start + n_tiles]
            sources, combined = [], dimod.BinaryQuadraticModel.empty(dimod.SPIN)
            for slot, fields in enumerate(group):
                source = dimod.BinaryQuadraticModel.from_ising(
                    {i: float(fields[i]) for i in range(self.n_spins)}, couplings)
                strength = self._resolve_chain_strength(source, self.tiles[slot])
                self.telemetry["chain_strength"] = strength
                combined.update(embed_bqm(source, self.tiles[slot], adjacency,
                                          chain_strength=strength))
                sources.append(source)

            estimate = estimate_access_time_us(self.sampler.properties, num_reads=num_reads,
                                               anneal_us=anneal_us)
            budget = self.budget or QpuBudget(Path("results/dwave_qpu_ledger.json"))
            with budget.charge(estimate, f"{self.label} rows {start}-{start + len(group)}") as record:
                sampleset = self.sampler.sample(
                    combined, num_reads=num_reads, answer_mode="raw",
                    annealing_time=anneal_us, fast_anneal=self.fast_anneal,
                    auto_scale=False,   # see `_resolve_chain_strength`: rescaling would shrink `h`
                    label=f"{self.label}[{start}:{start + len(group)}]")
                sampleset.resolve()
                access = float(sampleset.info.get("timing", {}).get("qpu_access_time", 0.0))
                if access:
                    record(access)
                self.telemetry["qpu_access_time_us"] += access or estimate
                self.telemetry["submissions"] += 1

            for slot, source in enumerate(sources):
                unembedded = unembed_sampleset(sampleset, self.tiles[slot], source,
                                               chain_break_fraction=True)
                spins = unembedded.record.sample[:, np.argsort(unembedded.variables)]
                if spins.shape[0] < num_reads:      # defensive: raw mode should give exactly n
                    raise RuntimeError(f"solver returned {spins.shape[0]} of {num_reads} reads")
                out[start + slot] = spins[:num_reads].astype(np.int8, copy=False)
                if "chain_break_fraction" in unembedded.record.dtype.names:
                    self.telemetry["chain_break_fraction_sum"] += float(
                        unembedded.record.chain_break_fraction.mean())
                    self.telemetry["chain_break_rows"] += 1

            self.telemetry["rows"] += len(group)

        return out

    def report(self) -> dict:
        t = dict(self.telemetry)
        t["n_tiles"] = len(self.tiles)
        t["qpu_access_time_s"] = t["qpu_access_time_us"] / 1e6
        t["mean_chain_break_fraction"] = (
            t["chain_break_fraction_sum"] / t["chain_break_rows"]
            if t["chain_break_rows"] else 0.0)
        t["max_chain_strength"] = self.max_chain_strength()
        return t


_RESERVOIR: QpuReservoir | None = None


def get_reservoir(**kwargs) -> QpuReservoir:
    """Process-wide reservoir. Built once: tiling and the solver handshake are not per-batch work."""
    global _RESERVOIR
    if _RESERVOIR is None:
        _RESERVOIR = QpuReservoir(**kwargs)
    return _RESERVOIR


def set_reservoir(reservoir: QpuReservoir | None) -> None:
    """Inject a configured (or mock) reservoir; `None` clears it. Used by the drivers and tests."""
    global _RESERVOIR
    _RESERVOIR = reservoir


def sample_ising(h_batch: np.ndarray, j_matrix: np.ndarray, *, num_reads: int,
                 annealing_time_us: float, seed: int = 0, **kwargs) -> np.ndarray:
    """Backend entry point, shaped like `ocean_backend.sample_ising`.

    `seed` is accepted and ignored: a QPU's randomness is physical, and pretending otherwise by
    threading a seed through would misrepresent the arm as reproducible run-to-run.
    """
    if seed and not getattr(sample_ising, "_warned", False):
        warnings.warn("anneal.qpu_backend ignores `seed`: QPU sampling is not seedable",
                      stacklevel=2)
        sample_ising._warned = True
    return get_reservoir(**kwargs).sample(h_batch, j_matrix, num_reads=num_reads,
                                          anneal_us=annealing_time_us)
