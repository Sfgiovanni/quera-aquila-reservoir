"""Emulator unit tests that need no optional dependency: closed-form physics checks and
RK4 step-halving convergence. `tests/test_emulator_vs_reference.py` is the actual Gate 1
cross-validation against bloqade-analog/braket_ahs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from quera.device import Program
from quera.emulator import build_geometry_cache, evolve, occupation_moments
from quera.layout import cluster_3x4
from quera.pulses import build_probe_program

DEVICE = "cpu"
POSITIONS = cluster_3x4()
N = len(POSITIONS)


def test_drive_factorizes_into_independent_rotations():
    """phi=0, Delta_global=Delta_local=0 and interaction turned off: H(t) is proportional to
    the *same* fixed operator sum_q X_q at every instant, so the time-ordered exponential is
    exact -- U = exp(-i*theta*X_q) per qubit, theta = integral of Omega(t)dt (the pulse
    area, exact by the trapezoid rule since Omega is piecewise-linear). This isolates the
    drive/index-permutation half of the matvec from the diagonal half, with no reference
    simulator involved.
    """
    omega, ramp_us, plateau_us = 5.0, 0.05, 0.7
    program = build_probe_program(POSITIONS, np.zeros(N), plateau_us=plateau_us, omega=omega,
                                  s_detuning=0.0, ramp_us=ramp_us)
    geom = build_geometry_cache(POSITIONS, DEVICE, real_dtype=torch.float64)
    geom.pair_energy.zero_()  # isolate the drive term from the (nonzero at this geometry) interaction

    psi = evolve(program, np.zeros((1, N)), DEVICE, dtype=torch.complex128, geometry=geom)

    # Drive coefficient is Omega/2 (see the Hamiltonian in emulator.py's module docstring),
    # so the rotation angle is half the pulse area, not the area itself.
    theta = 0.5 * omega * (ramp_us + plateau_us)  # exact pulse area of the ramp-plateau-ramp trapezoid
    # exp(-i*theta*X)|0> = cos(theta)|0> - i*sin(theta)|1>
    c, s = np.cos(theta), -1j * np.sin(theta)
    dim = 1 << N
    idx = np.arange(dim)
    popcount = np.array([bin(i).count("1") for i in idx])
    expected = (c ** (N - popcount)) * (s ** popcount)

    got = psi[0].cpu().numpy()
    err = np.abs(got - expected).max()
    assert err < 1e-7, f"max|psi - exact product state| = {err:.3e}"


def test_step_halving_converges():
    """Halving `rk4_safety` (i.e. doubling the sub-step count) should shrink the change in
    the final state by ~16x, the RK4 global-error scaling -- not just "gets a bit better"."""
    program = build_probe_program(POSITIONS, np.linspace(0, 1, N), plateau_us=0.5)
    h = np.linspace(0, 1, N)[None, :]

    results = {safety: evolve(program, h, DEVICE, dtype=torch.complex128, rk4_safety=safety).cpu().numpy()
              for safety in (0.4, 0.2, 0.1, 0.05)}

    err_coarse = np.abs(results[0.4] - results[0.1]).max()
    err_fine = np.abs(results[0.2] - results[0.1]).max()
    assert err_fine > 0, "test is vacuous if halving makes no difference at this precision"
    ratio = err_coarse / err_fine
    assert ratio > 8, f"error did not shrink like RK4 global error under step halving: ratio={ratio:.2f}"


def test_norm_preserved():
    program = build_probe_program(POSITIONS, np.random.default_rng(0).uniform(0, 1, N), plateau_us=1.0)
    psi = evolve(program, np.random.default_rng(1).uniform(0, 1, (8, N)), DEVICE, dtype=torch.complex128)
    norms = psi.norm(dim=1).cpu().numpy()
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_occupation_moments_shapes_and_bounds():
    program = build_probe_program(POSITIONS, np.random.default_rng(2).uniform(0, 1, N), plateau_us=1.0)
    geom = build_geometry_cache(POSITIONS, DEVICE, real_dtype=torch.float64)
    psi = evolve(program, np.random.default_rng(3).uniform(0, 1, (5, N)), DEVICE, dtype=torch.complex128,
                geometry=geom)
    n_i, joint = occupation_moments(psi, geom)
    assert n_i.shape == (5, N)
    assert joint.shape == (5, N, N)
    assert torch.all(n_i >= -1e-9) and torch.all(n_i <= 1 + 1e-9)
    diag = torch.diagonal(joint, dim1=1, dim2=2)
    assert torch.allclose(diag, n_i, atol=1e-8)


def test_zero_plateau_program_runs():
    program = build_probe_program(POSITIONS, np.zeros(N), plateau_us=0.0)
    psi = evolve(program, np.zeros((1, N)), DEVICE, dtype=torch.complex128)
    assert psi.shape == (1, 1 << N)
