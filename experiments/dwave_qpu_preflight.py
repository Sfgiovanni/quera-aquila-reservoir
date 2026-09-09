"""Price a hardware annealing cell before spending any quota.

Consumes **zero QPU time**: solver properties and the node/edge lists are metadata, and the tiling
search is a classical graph computation on a local copy of the chip graph. Run this first, read the
projection, then run the cell.

    python -m experiments.dwave_qpu_preflight --mock            # no token needed
    python -m experiments.dwave_qpu_preflight                   # against the real solver
    python -m experiments.dwave_qpu_preflight --samples 10000 --n-train 10000

What it answers, in order:

1. Can this solver even reach the operating point? The published cell anneals for 5 ns, which is
   below every standard `annealing_time_range` floor and exactly at the `fast_anneal` minimum.
2. How many copies of the 12-spin problem fit on the chip at once? This is the budget's dominant
   term -- see `anneal/qpu_backend.py` on why programming time, not annealing, is what a QPU cell
   actually costs.
3. Does the requested cell fit in the remaining budget, and if not, what size does?

The projection is deliberately the same code path the run itself charges against
(`anneal.qpu_budget.estimate_access_time_us`), so a cell that preflights as affordable cannot be
refused mid-run by a differently-computed estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anneal.qpu_budget import DEFAULT_CAP_SECONDS, QpuBudget, estimate_access_time_us
from anneal.qpu_backend import load_or_build_tiles

DDIM_STEPS = 50   # mirrors experiments.qrc_fusion_fair_core.DDIM_STEPS


def rows_for_cell(n_train: int, samples: int, ddim_steps: int = DDIM_STEPS) -> dict:
    """Field vectors the pipeline will ask the reservoir for, split by stage.

    The rollout dominates by more than an order of magnitude: fitting touches each training pair
    once, but generation re-encodes every sample at all `ddim_steps` denoising steps.
    """
    n_val = max(500, n_train // 2)
    fit = n_train + n_val
    generation = samples * ddim_steps
    return {"fit": fit, "generation": generation, "total": fit + generation}


def project(properties: dict, n_tiles: int, rows: int, *, num_reads: int, anneal_us: float,
            n_probes: int) -> dict:
    per_submission = estimate_access_time_us(properties, num_reads=num_reads, anneal_us=anneal_us)
    submissions = int(np.ceil(rows / n_tiles)) * n_probes
    total_us = submissions * per_submission
    return {"submissions": submissions, "per_submission_us": per_submission,
            "qpu_seconds": total_us / 1e6, "qpu_minutes": total_us / 6e7}


def max_samples_within(properties: dict, n_tiles: int, budget_seconds: float, *, n_train: int,
                       num_reads: int, anneal_us: float, n_probes: int,
                       ddim_steps: int = DDIM_STEPS) -> int:
    """Largest `--samples` whose projected cost stays inside `budget_seconds`."""
    per_submission = estimate_access_time_us(properties, num_reads=num_reads, anneal_us=anneal_us)
    affordable = int(budget_seconds * 1e6 / (per_submission * n_probes))
    fit_rows = rows_for_cell(n_train, 0, ddim_steps)["fit"]
    rows_left = affordable * n_tiles - fit_rows
    return max(0, rows_left // ddim_steps)


def build_sampler(mock: bool):
    if mock:
        from dwave.system.testing import MockDWaveSampler
        # Zephyr Z15,4 is Advantage2's topology. The mock graph has no calibration defects, so the
        # tile count it reports is an UPPER bound on a real chip's -- stated in the output.
        return MockDWaveSampler(topology_type="zephyr", topology_shape=[15, 4])
    from dwave.system import DWaveSampler
    try:
        return DWaveSampler()
    except ValueError as exc:
        if "token" not in str(exc).lower():
            raise
        raise SystemExit(
            "No D-Wave API token configured, so there is no solver to price.\n"
            "  Fix:  dwave config create        (or: export DWAVE_API_TOKEN=...)\n"
            "  Or:   rerun with --mock to plan against a defect-free Zephyr graph, which needs\n"
            "        no token and reports an UPPER bound on the tile count.") from exc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true",
                   help="use a defect-free Zephyr mock: no token, no network, upper-bound tiling")
    p.add_argument("--n-spins", type=int, default=12)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--ddim-steps", type=int, default=DDIM_STEPS)
    p.add_argument("--anneal-times-us", type=float, nargs="+", default=[0.005],
                   help="the published cell uses a single 5 ns probe")
    p.add_argument("--reads", type=int, default=8,
                   help="the published SQA cell used 8; matching it keeps the arms comparable")
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--cap-seconds", type=float, default=DEFAULT_CAP_SECONDS)
    p.add_argument("--ledger", default="results/dwave_qpu_ledger.json")
    p.add_argument("--out", default="results/qrc_fusion_fair/dwave_qpu_plan.json")
    a = p.parse_args()

    sampler = build_sampler(a.mock)
    props = sampler.properties
    name = getattr(getattr(sampler, "solver", None), "name", "mock-zephyr")

    print(f"solver              {name}")
    print(f"qubits              {len(sampler.nodelist)}  couplers {len(sampler.edgelist)}")
    print(f"topology            {props.get('topology')}")
    fast = props.get("fast_anneal_time_range")
    standard = props.get("annealing_time_range")
    print(f"annealing_time      standard {standard}   fast {fast}")
    print(f"h_range / j_range   {props.get('h_range')} / {props.get('j_range')}")
    print(f"num_reads_range     {props.get('num_reads_range')}")
    print(f"problem_timing_data {'present' if props.get('problem_timing_data') else 'ABSENT -> conservative fallbacks'}")

    # 1. reachability of the operating point
    print("\n-- operating point --")
    for t in a.anneal_times_us:
        ok_fast = fast and fast[0] <= t <= fast[1]
        ok_std = standard and standard[0] <= t <= standard[1]
        verdict = ("fast_anneal" if ok_fast else "standard" if ok_std else "UNREACHABLE")
        print(f"  {t} us -> {verdict}")
    unreachable = [t for t in a.anneal_times_us
                   if not ((fast and fast[0] <= t <= fast[1])
                           or (standard and standard[0] <= t <= standard[1]))]
    if unreachable:
        print(f"  !! {unreachable} cannot be produced by this solver; the run would be refused.")

    # 2. tiling -- the dominant budget term
    print("\n-- tiling --")
    tiles = load_or_build_tiles(sampler, a.n_spins, max_tiles=a.max_tiles, verbose=True)
    chains = [len(c) for tile in tiles for c in tile.values()]
    print(f"  tiles               {len(tiles)}  (rows carried per programming cycle)")
    print(f"  chain length        mean {np.mean(chains):.2f}  max {max(chains)}")
    print(f"  qubits used         {sum(chains)} of {len(sampler.nodelist)}")
    if a.mock:
        print("  NOTE: mock graph is defect-free; a real chip will tile somewhat lower.")

    # 3. does the cell fit
    rows = rows_for_cell(a.n_train, a.samples, a.ddim_steps)
    proj = project(props, len(tiles), rows["total"], num_reads=a.reads,
                   anneal_us=min(a.anneal_times_us), n_probes=len(a.anneal_times_us))
    budget = QpuBudget(Path(a.ledger), cap_seconds=a.cap_seconds)
    state = budget.summary()

    print("\n-- projected cell --")
    print(f"  rows  fit {rows['fit']:,}  generation {rows['generation']:,}  total {rows['total']:,}")
    print(f"  submissions         {proj['submissions']:,}")
    print(f"  per submission      {proj['per_submission_us'] / 1000:.2f} ms")
    print(f"  QPU time            {proj['qpu_seconds']:.1f} s  ({proj['qpu_minutes']:.2f} min)")
    print(f"\n  budget cap          {state['cap_seconds']:.0f} s ({state['cap_seconds'] / 60:.0f} min)")
    print(f"  usable (less margin){state['usable_seconds']:>8.0f} s")
    print(f"  already spent       {state['spent_seconds']:.1f} s over {state['submissions']} submissions")
    print(f"  remaining           {state['remaining_seconds']:.1f} s")

    fits = proj["qpu_seconds"] <= state["remaining_seconds"]
    print(f"\n  VERDICT             {'FITS' if fits else 'DOES NOT FIT'}")
    if not fits:
        best = max_samples_within(props, len(tiles), state["remaining_seconds"],
                                  n_train=a.n_train, num_reads=a.reads,
                                  anneal_us=min(a.anneal_times_us),
                                  n_probes=len(a.anneal_times_us), ddim_steps=a.ddim_steps)
        print(f"  largest --samples that fits: {best:,}")

    plan = {"solver": name, "qubits": len(sampler.nodelist), "n_tiles": len(tiles),
            "chain_len_mean": float(np.mean(chains)), "chain_len_max": int(max(chains)),
            "mock": bool(a.mock), "n_train": a.n_train, "samples": a.samples,
            "ddim_steps": a.ddim_steps, "reads": a.reads,
            "anneal_times_us": a.anneal_times_us, "rows": rows, "projection": proj,
            "budget": state, "fits": bool(fits), "unreachable_times_us": unreachable}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1))
    print(f"\nplan written to {out}")


if __name__ == "__main__":
    main()
