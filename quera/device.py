"""Single source of truth for the QuEra Aquila numeric constraints.

Every constant here is copied verbatim from the task's "Especificacao do Aquila" table --
none is estimated, looked up in a datasheet, or derived from a different source. If a
number this module needs is not in that table, it does not exist for this project and must
be requested rather than guessed.

Internal units are rad/us for frequencies and um for lengths (Bloqade's convention).
Braket's SI units (rad/s, m) are converted only at the `program.py` export boundary, so
every constant below stays in the units actually used by `pulses.py` and `emulator.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------- layout

MAX_SITES = 256
FOV_X_UM = 75.0
FOV_Y_UM = 76.0
MIN_SPACING_UM = 4.0            # radial and vertical -- enforced as all-pairs distance,
                                 # which subsumes both (row spacing is a pair of sites too).
POSITION_RESOLUTION_UM = 0.01   # 10 nm

# ---------------------------------------------------------------------------- drive (Omega, phase)

OMEGA_MIN = 0.0
OMEGA_MAX = 15.8                # rad/us
OMEGA_SLEW_MAX = 400.0          # rad/us^2
# Omega must be exactly 0 at the first and last waveform breakpoint of every program.

# ---------------------------------------------------------------------------- detuning

DELTA_GLOBAL_MIN = -125.0       # rad/us
DELTA_GLOBAL_MAX = 125.0        # rad/us

DELTA_LOCAL_MIN = -125.0        # rad/us -- local detuning is negative-only by construction
DELTA_LOCAL_MAX = 0.0
LOCAL_DETUNING_MAX_SITES = 256

H_LOCAL_MIN = 0.0
H_LOCAL_MAX = 1.0

# ---------------------------------------------------------------------------- timing

MAX_DURATION_US = 4.0
TIME_RESOLUTION_US = 0.001      # 1 ns
MIN_SEGMENT_US = 0.05           # 50 ns

# ---------------------------------------------------------------------------- shots

MAX_SHOTS = 1000

# ---------------------------------------------------------------------------- interaction

C6_RAD_US_UM6 = 5.4203e6        # rad/us . um^6 (== 862690 * 2*pi MHz.um^6, the task's own identity)

# Numeric tolerances for validation, not physical parameters: they only absorb float64
# round-off from the quantization helpers in `layout.py`/`pulses.py`, which is why they are
# many orders below the quantities they compare against (10 nm position grid, 1 ns time grid).
_POSITION_EPS_UM = 1e-7
_TIME_EPS_US = 1e-9
_VALUE_EPS = 1e-6


def blockade_radius(omega: float) -> float:
    """R_b(Omega) = (C6/Omega)^(1/6), in um.

    9.76 um at Omega=2*pi rad/us and 8.37 um at Omega=OMEGA_MAX, matching the task spec.
    """
    return (C6_RAD_US_UM6 / omega) ** (1.0 / 6.0)


def pair_coupling(r_um: float) -> float:
    """J(r) = C6 / r^6, in rad/us -- the van der Waals coupling between two clusters."""
    return C6_RAD_US_UM6 / r_um**6


@dataclass
class Program:
    """A complete Aquila-submittable analog program.

    `times_us` is one shared breakpoint grid for `omega`, `phase`, `delta_global` and
    `delta_local`, all piecewise-linear in that grid -- matching the shared-timeseries
    convention `braket.ahs.DrivingField.from_lists` uses for amplitude/detuning/phase.
    `delta_local` is the time-dependent *magnitude* of the shifting field; the per-site
    spatial pattern is `h` (dimensionless, in [0,1]), so the detuning applied to site j at
    time t is `delta_global[t] + h[j] * delta_local[t]`.
    """

    positions_um: np.ndarray     # (n_sites, 2)
    times_us: np.ndarray         # (n_points,), times_us[0] == 0
    omega: np.ndarray            # (n_points,) rad/us
    phase: np.ndarray            # (n_points,) rad
    delta_global: np.ndarray     # (n_points,) rad/us
    delta_local: np.ndarray      # (n_points,) rad/us, <= 0
    h: np.ndarray                # (n_sites,) in [0, 1]
    shots: int | None = None

    @property
    def n_sites(self) -> int:
        return int(self.positions_um.shape[0])

    @property
    def duration_us(self) -> float:
        return float(self.times_us[-1])


class ProgramValidationError(ValueError):
    """Raised by `validate` for any Aquila constraint violation. Never a warning."""


def validate(program: Program) -> None:
    """Raise `ProgramValidationError` on the first Aquila constraint the program violates.

    Runs unconditionally in both emulation and (eventually) hardware submission, so a
    program that would fail on the real device fails here first, deterministically.
    Positions and times must already be on their respective quantization grids -- quantize
    in `layout.py`/`pulses.py` *before* calling this, never after: validating a
    pre-quantization value and then rounding can silently reintroduce a violation (e.g. two
    sites 4.003 um apart that quantize down to 4.00 um exactly, or up to 3.99).
    """
    _validate_layout(program)
    _validate_timing(program)
    _validate_omega(program)
    _validate_detuning(program)
    _validate_shots(program)


def _validate_layout(program: Program) -> None:
    pos = np.asarray(program.positions_um, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] < 1:
        raise ProgramValidationError(f"positions_um must be (n_sites, 2) with n_sites>=1, got {pos.shape}")
    n = pos.shape[0]
    if n > MAX_SITES:
        raise ProgramValidationError(f"{n} sites exceeds MAX_SITES={MAX_SITES}")
    if n > LOCAL_DETUNING_MAX_SITES:
        raise ProgramValidationError(f"{n} sites exceeds LOCAL_DETUNING_MAX_SITES={LOCAL_DETUNING_MAX_SITES}")

    ticks = np.round(pos / POSITION_RESOLUTION_UM)
    if not np.allclose(ticks * POSITION_RESOLUTION_UM, pos, atol=_POSITION_EPS_UM):
        bad = np.abs(ticks * POSITION_RESOLUTION_UM - pos).max()
        raise ProgramValidationError(
            f"positions not on the {POSITION_RESOLUTION_UM * 1000:g} nm grid: max deviation {bad:.3e} um")

    width = pos[:, 0].max() - pos[:, 0].min()
    height = pos[:, 1].max() - pos[:, 1].min()
    if width > FOV_X_UM + _POSITION_EPS_UM or height > FOV_Y_UM + _POSITION_EPS_UM:
        raise ProgramValidationError(
            f"layout bounding box {width:.3f}x{height:.3f} um exceeds FOV {FOV_X_UM}x{FOV_Y_UM} um")

    if n > 1:
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        closest = dist.min()
        if closest < MIN_SPACING_UM - _POSITION_EPS_UM:
            raise ProgramValidationError(
                f"minimum pairwise spacing {closest:.4f} um is below MIN_SPACING_UM={MIN_SPACING_UM} um")

    if program.h.shape != (n,):
        raise ProgramValidationError(f"h must have shape ({n},), got {program.h.shape}")
    if np.any(program.h < H_LOCAL_MIN - _VALUE_EPS) or np.any(program.h > H_LOCAL_MAX + _VALUE_EPS):
        raise ProgramValidationError(
            f"h out of [{H_LOCAL_MIN}, {H_LOCAL_MAX}]: min={program.h.min():.4f} max={program.h.max():.4f}")


def _validate_timing(program: Program) -> None:
    t = np.asarray(program.times_us, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2:
        raise ProgramValidationError(f"times_us must have >=2 breakpoints, got shape {t.shape}")
    if abs(t[0]) > _TIME_EPS_US:
        raise ProgramValidationError(f"times_us must start at 0, got {t[0]!r}")
    if t[-1] > MAX_DURATION_US + _TIME_EPS_US:
        raise ProgramValidationError(f"duration {t[-1]:.6f} us exceeds MAX_DURATION_US={MAX_DURATION_US}")

    ticks = np.round(t / TIME_RESOLUTION_US)
    if not np.allclose(ticks * TIME_RESOLUTION_US, t, atol=_TIME_EPS_US):
        bad = np.abs(ticks * TIME_RESOLUTION_US - t).max()
        raise ProgramValidationError(
            f"times_us not on the {TIME_RESOLUTION_US * 1000:g} ns grid: max deviation {bad:.3e} us")

    durations = np.diff(t)
    if np.any(durations <= _TIME_EPS_US):
        raise ProgramValidationError("times_us must be strictly increasing")
    short = durations < MIN_SEGMENT_US - _TIME_EPS_US
    if np.any(short):
        raise ProgramValidationError(
            f"segment duration {durations[short].min() * 1000:.3f} ns is below "
            f"MIN_SEGMENT_US={MIN_SEGMENT_US * 1000:g} ns")

    for name, arr in (("omega", program.omega), ("phase", program.phase),
                      ("delta_global", program.delta_global), ("delta_local", program.delta_local)):
        if np.asarray(arr).shape != t.shape:
            raise ProgramValidationError(f"{name} must share times_us's shape {t.shape}, got {np.shape(arr)}")


def _validate_omega(program: Program) -> None:
    omega = np.asarray(program.omega, dtype=np.float64)
    if abs(omega[0]) > _VALUE_EPS or abs(omega[-1]) > _VALUE_EPS:
        raise ProgramValidationError(
            f"Omega must be 0 at the start and end of the program, got omega[0]={omega[0]!r} "
            f"omega[-1]={omega[-1]!r}")
    if np.any(omega < OMEGA_MIN - _VALUE_EPS) or np.any(omega > OMEGA_MAX + _VALUE_EPS):
        raise ProgramValidationError(
            f"Omega out of [{OMEGA_MIN}, {OMEGA_MAX}] rad/us: min={omega.min():.4f} max={omega.max():.4f}")

    durations = np.diff(program.times_us)
    slew = np.abs(np.diff(omega)) / durations
    if np.any(slew > OMEGA_SLEW_MAX + _VALUE_EPS):
        raise ProgramValidationError(
            f"Omega slew {slew.max():.3f} rad/us^2 exceeds OMEGA_SLEW_MAX={OMEGA_SLEW_MAX}")


def _validate_detuning(program: Program) -> None:
    dg = np.asarray(program.delta_global, dtype=np.float64)
    if np.any(dg < DELTA_GLOBAL_MIN - _VALUE_EPS) or np.any(dg > DELTA_GLOBAL_MAX + _VALUE_EPS):
        raise ProgramValidationError(
            f"Delta_global out of [{DELTA_GLOBAL_MIN}, {DELTA_GLOBAL_MAX}] rad/us: "
            f"min={dg.min():.4f} max={dg.max():.4f}")

    dl = np.asarray(program.delta_local, dtype=np.float64)
    if np.any(dl < DELTA_LOCAL_MIN - _VALUE_EPS) or np.any(dl > DELTA_LOCAL_MAX + _VALUE_EPS):
        raise ProgramValidationError(
            f"Delta_local out of [{DELTA_LOCAL_MIN}, {DELTA_LOCAL_MAX}] rad/us (local detuning is "
            f"negative-only): min={dl.min():.4f} max={dl.max():.4f}")


def _validate_shots(program: Program) -> None:
    if program.shots is None:
        return
    if not (1 <= program.shots <= MAX_SHOTS):
        raise ProgramValidationError(f"shots={program.shots} out of [1, {MAX_SHOTS}]")
