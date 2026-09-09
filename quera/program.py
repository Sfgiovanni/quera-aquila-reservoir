"""Export a `device.Program` to bloqade-analog and to `braket.ahs`.

Used by the Gate 1 cross-validation test today, and is what a future hardware submission
would build on: `to_braket_ahs` produces the same `AnalogHamiltonianSimulation` object
`AwsDevice("...Aquila").run(...)` takes, so nothing here needs to change when the "no
credentials, zero spend" constraint of this task is lifted.

`bloqade-analog` and `amazon-braket-sdk` are optional dependencies (see
`docs/AQUILA_PORT.md`) -- both are imported lazily inside these functions so importing
`quera.program` never fails in the production environment, which doesn't install either.

Both exports assume `program.phase` is uniformly 0, the only value `pulses.py` currently
emits; a nonzero phase raises rather than silently dropping the phase channel.
"""
from __future__ import annotations

import numpy as np

from quera.device import Program, validate

_UM_TO_M = 1e-6
_US_TO_S = 1e-6
_RAD_PER_US_TO_RAD_PER_S = 1e6


def _check_uniform_phase(program: Program) -> None:
    if np.any(np.abs(program.phase) > 1e-12):
        raise NotImplementedError("nonzero phase is not exported yet -- pulses.py never emits it")


def to_bloqade(program: Program):
    """A `bloqade.analog` routine, ready for `.bloqade.python()` (or `.braket.local_emulator()`).

    bloqade's native units are already rad/us and um (see `device.py`'s module docstring),
    so no conversion happens here -- this export is closer to a straight relabeling than
    `to_braket_ahs`'s SI conversion.
    """
    from bloqade.analog import start

    validate(program)
    _check_uniform_phase(program)

    durations = list(np.diff(program.times_us))
    omega = list(program.omega)
    delta_global = list(program.delta_global)
    delta_local = list(program.delta_local)
    positions = [(float(x), float(y)) for x, y in program.positions_um]
    h = [float(v) for v in program.h]

    routine = (
        start.add_position(positions)
        .rydberg.rabi.amplitude.uniform.piecewise_linear(durations, omega)
        .detuning.uniform.piecewise_linear(durations, delta_global)
        .detuning.scale(h).piecewise_linear(durations, delta_local)
    )
    return routine


def to_braket_ahs(program: Program):
    """A `braket.ahs.AnalogHamiltonianSimulation`, in Braket's SI units (rad/s, m)."""
    from braket.ahs import AnalogHamiltonianSimulation, AtomArrangement, DrivingField, LocalDetuning

    validate(program)
    _check_uniform_phase(program)

    times_s = [t * _US_TO_S for t in program.times_us]
    omega_si = [v * _RAD_PER_US_TO_RAD_PER_S for v in program.omega]
    phase_si = [float(v) for v in program.phase]
    delta_global_si = [v * _RAD_PER_US_TO_RAD_PER_S for v in program.delta_global]
    delta_local_si = [v * _RAD_PER_US_TO_RAD_PER_S for v in program.delta_local]

    register = AtomArrangement()
    for x, y in program.positions_um:
        register.add((float(x) * _UM_TO_M, float(y) * _UM_TO_M))

    drive = DrivingField.from_lists(times=times_s, amplitudes=omega_si, detunings=delta_global_si,
                                    phases=phase_si)
    local = LocalDetuning.from_lists(times=times_s, values=delta_local_si, pattern=list(program.h))
    hamiltonian = drive + local

    return AnalogHamiltonianSimulation(register=register, hamiltonian=hamiltonian)
