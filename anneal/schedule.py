"""Annealing schedules `A(s)`, `B(s)` for the transverse-field Ising Hamiltonian

    H(s) = -(A(s)/2) sum_i X_i + (B(s)/2) [ sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j ]

with `s = t / t_anneal` in `[0, 1]`. `A` falls from its maximum to ~0, `B` rises from ~0 to its
maximum: the system starts in the ground state of the transverse field (the uniform superposition,
NOT `|00...0>` -- this is the one structural difference from `quera/emulator.py`, whose Rydberg
programs start in `|gg...g>`) and ends governed by the problem Hamiltonian.

**The default schedule here is the linear idealisation, not D-Wave's measured one.** Advantage2
ships its `A(s)`/`B(s)` as tabulated curves per system; they are smooth, monotone, and cross
somewhere around `s ~ 0.4`, but they are not straight lines. `Tabulated` accepts those curves
directly, so a hardware-faithful run is a data swap rather than a code change. Anything claimed
about *absolute* annealing times should use the tabulated schedule; the linear one is for
establishing that the pipeline works and for sweeping shape-independent trends.

Units: `A`, `B` in rad/us, matching `quera/`'s convention so `rk4_safety` means the same thing in
both emulators and the two arms' step-size accounting is comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# D-Wave quotes A(0)/h and B(1)/h in GHz -- order 5 GHz and 6 GHz respectively. In this module's
# rad/us units that is `2*pi * GHz * 1000`, since 1 GHz is 1000 cycles per microsecond.
#
# **Getting this conversion wrong by the factor of 1000 is easy and was done here first**: with
# `2*pi*5` instead of `2*pi*5000`, a "0.05 us" anneal accumulates only ~1.5 rad of phase and the
# state barely leaves |+>, which showed up as the shortest probe carrying almost no signal
# (std of <ZZ> = 0.009 against 0.83 at the longest). The physical scale is what makes anneal
# *durations* mean anything, so it is worth stating plainly rather than burying in a constant.
GHZ_TO_RAD_US = 2 * np.pi * 1000.0
A_MAX_RAD_US = 5.0 * GHZ_TO_RAD_US
B_MAX_RAD_US = 6.0 * GHZ_TO_RAD_US

# At these energies a hardware-realistic anneal is enormously oscillatory: D-Wave's default 20 us
# accumulates ~6e5 rad of phase, which no exact statevector propagator resolves cheaply.
#
# Where the *informative* window sits was measured rather than assumed -- correlation between the
# feature vectors of consecutive anneal durations, 12 spins, production fields:
#
#     t         std<Z>   std<ZZ>   corr with previous
#     0.05 ns   0.1075   0.0129    --
#     0.20 ns   0.5422   0.3099    0.686
#     1.00 ns   0.9104   0.7924    0.761
#     5.00 ns   0.9762   0.8902    0.969
#    20.00 ns   0.9875   0.9081    0.995
#
# The map saturates past ~1 ns: at 5 and 20 ns consecutive probes correlate at 0.97 and 0.99, so
# they are near-duplicates paying full price. The durations that actually produce *different*
# feature maps are sub-nanosecond -- which is **below Advantage2's ~5 ns fast-anneal floor**. In
# this closed-system model the knob the paper reports on is therefore exhausted before hardware can
# reach it; whether that survives the decoherence a real annealer has at those timescales is
# exactly what the closed-system caveat in `emulator.py` is about.
# Cost scales with total duration (step count = steps_per_us * sum(t)), so the 5 ns probe alone was
# 80% of the compute while correlating 0.969 with the 1 ns one -- paying four fifths of the bill for
# a near-duplicate. Spreading four probes *inside* the decorrelated window instead keeps the 312
# feature width and cuts a cell from ~25 h to ~7 h.
INFORMATIVE_ANNEAL_US = (0.00005, 0.0002, 0.0005, 0.001)


@dataclass(frozen=True)
class Linear:
    """`A(s) = a_max (1-s)`, `B(s) = b_max s`. The textbook idealisation."""
    a_max: float = A_MAX_RAD_US
    b_max: float = B_MAX_RAD_US

    def __call__(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), 1.0)
        return self.a_max * (1.0 - s), self.b_max * s

    @property
    def key(self) -> dict:
        return {"kind": "linear", "a_max": float(self.a_max), "b_max": float(self.b_max)}


@dataclass(frozen=True)
class Tabulated:
    """D-Wave's measured `A(s)`/`B(s)`, linearly interpolated between breakpoints.

    `s_points` must be sorted and span `[0, 1]`. Supply `a_points`/`b_points` in rad/us.
    """
    s_points: tuple
    a_points: tuple
    b_points: tuple

    def __call__(self, s: float) -> tuple[float, float]:
        s = min(max(s, 0.0), 1.0)
        return (float(np.interp(s, self.s_points, self.a_points)),
                float(np.interp(s, self.s_points, self.b_points)))

    @property
    def key(self) -> dict:
        return {"kind": "tabulated", "s": list(map(float, self.s_points)),
                "a": list(map(float, self.a_points)), "b": list(map(float, self.b_points))}


def schedule_from_name(name: str):
    if name == "linear":
        return Linear()
    files = {"advantage2-standard": "advantage2_system4_standard.csv",
             "advantage2-fast": "advantage2_system4_fast.csv"}
    if name not in files:
        raise ValueError(f"unknown schedule {name!r}")
    values = np.loadtxt(Path(__file__).parent / "data" / files[name], delimiter=",", skiprows=1)
    return Tabulated(tuple(values[:, 0]), tuple(values[:, 1] * GHZ_TO_RAD_US),
                     tuple(values[:, 2] * GHZ_TO_RAD_US))
