"""Gate 1 helper: computes <n_i>/<n_i n_j> from bloqade-analog and braket_ahs for a batch of
programs. Runs under `.venv-gate1` (needs bloqade-analog + amazon-braket-sdk, NOT torch);
`tests/test_emulator_vs_reference.py` runs under the production `.venv` and shells out to
this script, so neither environment needs the other's dependencies.

Site-index conventions were verified empirically (see `docs/AQUILA_PORT.md`): bloqade labels
site j as bit position j of `state_vector.space.configurations` (LSB = site 0, from
`AtomType.integer_to_string`'s `state_int % n_level` peeling off site 0 first); braket's
`get_blockade_configurations` strings put site j at character index j. Both put the atom
added first (`positions_um` row 0) at site 0, matching `program.py`'s export order, and both
were cross-checked against each other on an asymmetric two-atom program before being trusted
here (0.528/0.458 agreement, see docs). Neither convention matches this project's own
`emulator.py` (MSB-first, see its module docstring) -- that is fine, because this script
never compares raw amplitude vectors across implementations, only the physically-labeled
per-site moments computed by each implementation from its own state and its own convention.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def bloqade_moments(program_json):
    from quera.program import to_bloqade

    program = _program_from_json(program_json)
    routine = to_bloqade(program)
    emu = routine.bloqade.python()

    def cb(state_vector, metadata, hamiltonian, *a):
        return np.asarray(state_vector.data), np.asarray(state_vector.space.configurations)

    data, configs = emu.run_callback(cb)[0]
    probs = np.abs(data) ** 2
    n = program.h.shape[0]
    bits = np.stack([(configs >> j) & 1 for j in range(n)]).astype(np.float64)  # (n, dim)
    n_i = (bits * probs[None, :]).sum(1)
    n_ij = np.einsum("qi,ri,i->qr", bits, bits, probs)
    return n_i, n_ij


def braket_moments(program_json, c6_rad_us_um6):
    from braket.analog_hamiltonian_simulator.rydberg.constants import (
        SPACE_UNIT, TIME_UNIT, capabilities_constants,
    )
    from braket.analog_hamiltonian_simulator.rydberg.numpy_solver import rk_run
    from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
        get_blockade_configurations,
    )
    from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import convert_unit
    from braket.analog_hamiltonian_simulator.rydberg.scipy_solver import scipy_integrate_ode_run
    from braket.analog_hamiltonian_simulator.rydberg.validators.blockade_radius import (
        validate_blockade_radius,
    )
    from braket.analog_hamiltonian_simulator.rydberg.validators.ir_validator import validate_program
    from braket.analog_hamiltonian_simulator.rydberg.validators.rydberg_coefficient import (
        validate_rydberg_interaction_coef,
    )

    from quera.program import to_braket_ahs

    program = _program_from_json(program_json)
    ahs = to_braket_ahs(program)
    ir_program = ahs.to_ir()

    c6_si = c6_rad_us_um6 * 1e-30  # rad/us.um^6 -> rad/s.m^6, matching quera.device's units
    coef = validate_rydberg_interaction_coef(c6_si) / ((SPACE_UNIT**6) / TIME_UNIT)
    blockade_radius = validate_blockade_radius(0.0) / SPACE_UNIT
    validate_program(ir_program, capabilities_constants())
    p = convert_unit(ir_program)
    duration = float(p.hamiltonian.drivingFields[0].amplitude.time_series.times[-1])
    configs = get_blockade_configurations(p.setup.ahs_register, blockade_radius)

    # rk_run (fixed-step RK6) only below the 1000-configuration threshold the public
    # RydbergAtomSimulator.run() itself uses; above it, scipy_integrate_ode_run's adaptive
    # solver is both faster and more accurate here (see docs/AQUILA_PORT.md's timing note).
    if len(configs) <= 1000:
        sim_times = np.linspace(0, duration, 20000)
        states = rk_run(p, configs, sim_times, coef)
    else:
        sim_times = np.linspace(0, duration, 1000)
        states = scipy_integrate_ode_run(p, configs, sim_times, coef, atol=1e-10, rtol=1e-10)
    final = states[-1]
    probs = np.abs(final) ** 2

    n = program.h.shape[0]
    bits = np.array([[1.0 if c[site] == "r" else 0.0 for c in configs] for site in range(n)])
    n_i = (bits * probs[None, :]).sum(1)
    n_ij = np.einsum("qi,ri,i->qr", bits, bits, probs)
    return n_i, n_ij


def _program_from_json(d):
    from quera.device import Program
    return Program(
        positions_um=np.array(d["positions_um"]),
        times_us=np.array(d["times_us"]),
        omega=np.array(d["omega"]),
        phase=np.array(d["phase"]),
        delta_global=np.array(d["delta_global"]),
        delta_local=np.array(d["delta_local"]),
        h=np.array(d["h"]),
        shots=None,
    )


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path) as f:
        payload = json.load(f)
    c6 = payload["c6_rad_us_um6"]

    results = []
    for program_json in payload["programs"]:
        bq_ni, bq_nij = bloqade_moments(program_json)
        bk_ni, bk_nij = braket_moments(program_json, c6)
        results.append(dict(bloqade_n_i=bq_ni.tolist(), bloqade_n_ij=bq_nij.tolist(),
                            braket_n_i=bk_ni.tolist(), braket_n_ij=bk_nij.tolist()))

    with open(out_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
