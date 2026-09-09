"""Gate 0: the Aquila validator must reject every constraint violation it is meant to catch.

Each test starts from `pulses.build_probe_program`'s valid output and breaks exactly one
constraint, so a failure here isolates which check regressed.
"""
from __future__ import annotations

import numpy as np
import pytest

from quera import device
from quera.layout import cluster_3x4
from quera.pulses import build_probe_program

POSITIONS = cluster_3x4()
H = np.linspace(0.0, 1.0, len(POSITIONS))


def valid_program():
    return build_probe_program(POSITIONS, H, plateau_us=1.0)


def test_valid_program_passes():
    program = valid_program()
    device.validate(program)  # must not raise
    assert program.omega[0] == 0.0 and program.omega[-1] == 0.0


def test_missing_ramp_fails():
    """Omega that does not return to 0 at the end must be rejected."""
    program = valid_program()
    program.omega[-1] = program.omega[-2]
    with pytest.raises(device.ProgramValidationError, match="Omega must be 0"):
        device.validate(program)


def test_30ns_segment_fails():
    """A 30 ns segment is below the 50 ns minimum."""
    program = valid_program()
    program.times_us = np.array([0.0, 0.03, 1.0, 1.05])
    with pytest.raises(device.ProgramValidationError, match="segment duration"):
        device.validate(program)


def test_5us_duration_fails():
    """5 us exceeds the 4 us maximum program duration."""
    program = valid_program()
    program.times_us = program.times_us * (5.0 / program.duration_us)
    with pytest.raises(device.ProgramValidationError, match="exceeds MAX_DURATION_US"):
        device.validate(program)


def test_3um_spacing_fails():
    """3 um is below the 4 um minimum pairwise spacing."""
    positions = cluster_3x4(spacing_um=3.0)
    with pytest.raises(device.ProgramValidationError, match="minimum pairwise spacing"):
        build_probe_program(positions, np.linspace(0, 1, len(positions)), plateau_us=1.0)


def test_positive_delta_local_fails():
    """Delta_local must stay <= 0; the shifting field is negative-only."""
    program = valid_program()
    program.delta_local[:] = 1.0
    with pytest.raises(device.ProgramValidationError, match="Delta_local"):
        device.validate(program)


def test_omega_out_of_range_fails():
    program = valid_program()
    program.omega[1] = device.OMEGA_MAX + 1.0
    with pytest.raises(device.ProgramValidationError, match="Omega out of"):
        device.validate(program)


def test_omega_slew_violation_fails():
    """MIN_SEGMENT_US * OMEGA_SLEW_MAX = 20 rad/us > OMEGA_MAX = 15.8 rad/us, so a program
    that also respects the 50 ns minimum segment can never violate slew -- the two limits
    are jointly consistent by construction. Exercise `_validate_omega` directly instead of
    going through `validate`, which would reject the short segment first."""
    program = valid_program()
    program.times_us = np.array([0.0, 0.01, 1.0, 1.01])  # 10 ns ramp, below MIN_SEGMENT_US
    program.omega = np.array([0.0, 10.0, 10.0, 0.0])  # slew = 10/0.01 = 1000 rad/us^2
    with pytest.raises(device.ProgramValidationError, match="slew"):
        device._validate_omega(program)


def test_delta_global_out_of_range_fails():
    program = valid_program()
    program.delta_global[:] = device.DELTA_GLOBAL_MAX + 1.0
    with pytest.raises(device.ProgramValidationError, match="Delta_global"):
        device.validate(program)


def test_h_out_of_range_fails():
    program = valid_program()
    program.h[0] = 1.5
    with pytest.raises(device.ProgramValidationError, match=r"h out of"):
        device.validate(program)


def test_too_many_sites_fails():
    positions = np.zeros((device.MAX_SITES + 1, 2))
    positions[:, 0] = np.arange(device.MAX_SITES + 1) * device.MIN_SPACING_UM
    with pytest.raises(device.ProgramValidationError, match="MAX_SITES"):
        build_probe_program(positions, np.zeros(device.MAX_SITES + 1), plateau_us=1.0)


def test_shots_over_limit_fails():
    program = valid_program()
    program.shots = device.MAX_SHOTS + 1
    with pytest.raises(device.ProgramValidationError, match="shots"):
        device.validate(program)


def test_unquantized_position_fails():
    program = valid_program()
    program.positions_um = program.positions_um.copy()
    program.positions_um[0, 0] += 0.0001  # 0.1 nm off the 10 nm grid
    with pytest.raises(device.ProgramValidationError, match="10 nm grid|not on the"):
        device.validate(program)


def test_unquantized_time_fails():
    program = valid_program()
    program.times_us = program.times_us.copy()
    program.times_us[1] += 0.0001  # 0.1 ns off the 1 ns grid
    with pytest.raises(device.ProgramValidationError, match="1 ns grid|not on the"):
        device.validate(program)


def test_zero_plateau_collapses_to_triangle():
    program = build_probe_program(POSITIONS, H, plateau_us=0.0)
    assert len(program.times_us) == 3
    device.validate(program)


def test_short_ramp_rejected_by_builder():
    with pytest.raises(device.ProgramValidationError, match="minimum segment"):
        build_probe_program(POSITIONS, H, plateau_us=1.0, ramp_us=0.03)
