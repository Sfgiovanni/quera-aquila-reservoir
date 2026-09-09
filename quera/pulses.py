"""Per-probe-time pulse programs: Omega/Delta_global/Delta_local segments, with ramps,
quantized to the 50 ns grid.

Every probe time `T_k` in the encoding is its own **independent** program -- ramp up,
plateau, ramp down -- not a snapshot of one long trajectory (see `docs/AQUILA_PORT.md`).
Only Omega ramps; the task spec gives no start/end-zero requirement for the detunings, so
Delta_global and Delta_local are held constant for the whole program, including through the
Omega ramps. That keeps the dominant "trecho" of the pulse -- the plateau -- exactly
constant-H, per the emulator spec; the two 50 ns ramp edges are short, fixed-duration
corrections that `emulator.py`'s integrator handles like any other time-dependent segment.

Quantization always happens before validation, never after: rounding a value that already
passed a check can silently reintroduce the violation it just cleared (e.g. two positions
4.003 um apart quantizing down to 3.99 um). See `device.validate`'s docstring.
"""
from __future__ import annotations

import numpy as np

from quera.device import (
    MIN_SEGMENT_US,
    TIME_RESOLUTION_US,
    Program,
    ProgramValidationError,
    validate,
)


def quantize_time(t_us: float) -> float:
    """Round to the nearest 1 ns tick."""
    return float(np.round(t_us / TIME_RESOLUTION_US)) * TIME_RESOLUTION_US


def build_probe_program(positions_um, h, plateau_us, omega=6.283, s_detuning=9.0, phi=0.0,
                        ramp_us=MIN_SEGMENT_US, shots=None):
    """Ramp-up / plateau / ramp-down program at fixed Omega, recentred detuning.

    `s_detuning` is the envelope amplitude `S` from the encoding spec: Delta_global is
    held at `+S/2` and Delta_local at `-S`, so the per-site detuning
    `Delta_global + h[j]*Delta_local = S/2 - S*h[j]` sits in `[-S/2, +S/2]` while
    Delta_local itself stays <= 0, satisfying the "local detuning is negative-only"
    constraint. `plateau_us=0` collapses the plateau breakpoint, giving a pure
    ramp-up/ramp-down triangle -- needed for small `V_SLICES` probe times.

    Raises `ProgramValidationError` (via `device.validate`) if the requested combination
    of `plateau_us`/`ramp_us`/`omega`/`s_detuning` cannot be realized on the Aquila grid --
    e.g. a ramp shorter than the 50 ns minimum segment, or a plateau that quantizes to a
    duration under 50 ns instead of to exactly 0.
    """
    ramp_q = quantize_time(ramp_us)
    if ramp_q < MIN_SEGMENT_US - 1e-9:
        raise ProgramValidationError(
            f"ramp_us={ramp_us} quantizes to {ramp_q * 1000:.3f} ns, below the "
            f"{MIN_SEGMENT_US * 1000:g} ns minimum segment")
    plateau_q = quantize_time(plateau_us)
    if 0.0 < plateau_q < MIN_SEGMENT_US - 1e-9:
        raise ProgramValidationError(
            f"plateau_us={plateau_us} quantizes to {plateau_q * 1000:.3f} ns, which is neither "
            f"0 nor >= the {MIN_SEGMENT_US * 1000:g} ns minimum segment")

    if plateau_q > 0.0:
        times = np.array([0.0, ramp_q, ramp_q + plateau_q, 2 * ramp_q + plateau_q])
        omega_vals = np.array([0.0, omega, omega, 0.0])
    else:
        times = np.array([0.0, ramp_q, 2 * ramp_q])
        omega_vals = np.array([0.0, omega, 0.0])

    delta_global = np.full_like(times, s_detuning / 2.0)
    delta_local = np.full_like(times, -s_detuning)
    phase = np.full_like(times, phi)

    program = Program(
        positions_um=np.array(positions_um, dtype=np.float64, copy=True),
        times_us=times,
        omega=omega_vals,
        phase=phase,
        delta_global=delta_global,
        delta_local=delta_local,
        h=np.array(h, dtype=np.float64, copy=True),
        shots=shots,
    )
    validate(program)
    return program
