"""Unitary circuit families used as reservoir dynamics and as state preparations.

The three reservoir families implement the magic dial of the study:

``clifford``
    Random Clifford circuits.  Zero dynamics-generated magic by the
    Gottesman-Knill structure: a Clifford conjugates every Pauli string to
    another Pauli string, so the Pauli spectrum is permuted (up to sign) and
    ``Mtilde_2`` is exactly invariant.  This is the H2 arm.
``alpha_dial``
    ``U(alpha) = R_z(alpha) C`` with fixed Clifford ``C`` and
    ``R_z(alpha) = exp[-i (alpha/2) sum_j Z_j]``, the paper's own dial
    (arXiv:2607.12035, Sec. II.3 / Fig. 1).  **Depth is identical at every
    alpha**, so it controls for the "deeper circuits are not a magic control"
    confound.  Primary dial.
``doped``
    Clifford + ``m`` injected T gates.  Monotone in nominal T-count but *not*
    depth-matched; secondary dial, reported alongside.
``haar``
    Haar-random unitaries and a chaotic Ising Floquet unitary; near-maximal
    magic.

Note on Clifford sampling
-------------------------
We generate random Clifford *circuits* (random products of H, S, CNOT) rather
than sampling the Clifford group uniformly via the symplectic algorithm.  For
every claim we make this is sufficient: the defining property used in H2/E2 is
that the dynamics *is* Clifford, which holds exactly for any such product, and
zero dynamics-generated magic follows from that alone, not from the sampling
distribution.  Where the paper's *ensemble-averaged* constants (the 3/4 of
Theorem 1) would be at stake we do not claim agreement anyway -- see
``docs/theory_notes.md`` Sec. 2, HYPOTHESIS GAP 1.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "H_GATE",
    "S_GATE",
    "T_GATE",
    "embed_1q",
    "embed_cnot",
    "random_clifford_circuit",
    "alpha_dial_unitary",
    "doped_clifford_circuit",
    "haar_random_unitary",
    "chaotic_ising_unitary",
    "build_reservoir_unitary",
]

_SQ2 = 1.0 / np.sqrt(2.0)

H_GATE: np.ndarray = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex)
S_GATE: np.ndarray = np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)
T_GATE: np.ndarray = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=complex)

_PAULI_Z: np.ndarray = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def embed_1q(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit ``gate`` on ``qubit`` into the ``n_qubits`` register.

    Qubit 0 is the most significant tensor factor, matching the index ordering
    of :func:`magicqrc.magic.pauli_spectrum`.
    """
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def embed_cnot(control: int, target: int, n_qubits: int) -> np.ndarray:
    """CNOT between ``control`` and ``target`` on an ``n_qubits`` register."""
    if control == target:
        raise ValueError("control and target must differ")
    dim = 2**n_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for basis in range(dim):
        bits = [(basis >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control]:
            bits[target] ^= 1
        image = sum(bit << (n_qubits - 1 - q) for q, bit in enumerate(bits))
        out[image, basis] = 1.0
    return out


def random_clifford_circuit(
    n_qubits: int, depth: int, rng: np.random.Generator
) -> np.ndarray:
    """Random Clifford unitary as a depth-``depth`` product of H, S and CNOT.

    Each layer applies one uniformly chosen gate from {H, S} on a random qubit
    and, for ``n_qubits >= 2``, a CNOT on a random ordered pair.
    """
    dim = 2**n_qubits
    unitary = np.eye(dim, dtype=complex)
    for _ in range(depth):
        gate = H_GATE if rng.integers(2) == 0 else S_GATE
        unitary = embed_1q(gate, int(rng.integers(n_qubits)), n_qubits) @ unitary
        if n_qubits >= 2:
            control, target = rng.choice(n_qubits, size=2, replace=False)
            unitary = embed_cnot(int(control), int(target), n_qubits) @ unitary
    return unitary


def alpha_dial_unitary(
    n_qubits: int, alpha: float, depth: int, rng: np.random.Generator
) -> np.ndarray:
    r"""``U(alpha) = R_z(alpha) C``, the paper's depth-matched magic dial.

    ``R_z(alpha) = exp[-i (alpha/2) sum_j Z_j]`` (arXiv:2607.12035, Sec. II.3).
    At ``alpha = 0`` and ``alpha = pi/2`` the global rotation is Clifford, so the
    dynamics-generated magic returns to its floor; the paper reports the magic to
    be non-monotonic in ``alpha`` with minima exactly there.  The Clifford factor
    ``C`` is drawn from ``rng`` at fixed ``depth``, so sweeping ``alpha`` changes
    magic **at constant circuit depth** -- the property that makes this dial, and
    not the T-count ladder, the confound-free control.
    """
    clifford = random_clifford_circuit(n_qubits, depth, rng)
    generator = sum(embed_1q(_PAULI_Z, q, n_qubits) for q in range(n_qubits))
    rotation = expm(-1j * (alpha / 2.0) * generator)
    return rotation @ clifford


def doped_clifford_circuit(
    n_qubits: int, depth: int, n_t_gates: int, rng: np.random.Generator
) -> np.ndarray:
    """Clifford circuit doped with ``n_t_gates`` T gates at random positions.

    Secondary magic dial.  NOT depth-matched across ``n_t_gates``: the doped
    circuit has ``depth + n_t_gates`` gate layers, so a bare comparison across
    ``m`` confounds magic with depth.  E1 therefore reports it only alongside the
    depth-matched :func:`alpha_dial_unitary` sweep.
    """
    dim = 2**n_qubits
    unitary = np.eye(dim, dtype=complex)
    t_positions = set(rng.choice(depth, size=min(n_t_gates, depth), replace=False).tolist())
    remaining = n_t_gates - len(t_positions)
    for layer in range(depth):
        gate = H_GATE if rng.integers(2) == 0 else S_GATE
        unitary = embed_1q(gate, int(rng.integers(n_qubits)), n_qubits) @ unitary
        if layer in t_positions:
            unitary = embed_1q(T_GATE, int(rng.integers(n_qubits)), n_qubits) @ unitary
        if n_qubits >= 2:
            control, target = rng.choice(n_qubits, size=2, replace=False)
            unitary = embed_cnot(int(control), int(target), n_qubits) @ unitary
    for _ in range(max(remaining, 0)):  # more T gates than layers: append the rest
        unitary = embed_1q(T_GATE, int(rng.integers(n_qubits)), n_qubits) @ unitary
    return unitary


def haar_random_unitary(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR decomposition of a Ginibre matrix."""
    dim = 2**n_qubits
    ginibre = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q_mat, r_mat = np.linalg.qr(ginibre)
    return q_mat * (np.diag(r_mat) / np.abs(np.diag(r_mat)))


def chaotic_ising_unitary(
    n_qubits: int,
    t_evolve: float = 1.0,
    j_coupling: float = 1.0,
    h_x: float = 1.4,
    h_z: float = 0.9045,
) -> np.ndarray:
    """Floquet unitary of the chaotic transverse+longitudinal-field Ising model.

    ``H = J sum_j Z_j Z_{j+1} + h_x sum_j X_j + h_z sum_j Z_j`` with the standard
    non-integrable point ``(h_x, h_z) = (1.4, 0.9045)``; open boundary conditions.
    Deterministic given its arguments -- used as a seed-independent near-maximal
    magic reference in the E1 ladder.
    """
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    dim = 2**n_qubits
    ham = np.zeros((dim, dim), dtype=complex)
    for q in range(n_qubits - 1):
        ham += j_coupling * (
            embed_1q(_PAULI_Z, q, n_qubits) @ embed_1q(_PAULI_Z, q + 1, n_qubits)
        )
    for q in range(n_qubits):
        ham += h_x * embed_1q(pauli_x, q, n_qubits)
        ham += h_z * embed_1q(_PAULI_Z, q, n_qubits)
    return expm(-1j * t_evolve * ham)


def build_reservoir_unitary(
    family: str, n_qubits: int, rng: np.random.Generator, **kwargs
) -> np.ndarray:
    """Dispatch to a reservoir unitary family by name.

    Parameters
    ----------
    family
        One of ``"clifford"``, ``"alpha_dial"``, ``"doped"``, ``"haar"``,
        ``"chaotic_ising"``.
    kwargs
        ``depth`` (all Clifford-based families), ``alpha`` (``alpha_dial``),
        ``n_t_gates`` (``doped``), ``t_evolve`` (``chaotic_ising``).
    """
    depth = int(kwargs.get("depth", 20))
    if family == "clifford":
        return random_clifford_circuit(n_qubits, depth, rng)
    if family == "alpha_dial":
        return alpha_dial_unitary(n_qubits, float(kwargs["alpha"]), depth, rng)
    if family == "doped":
        return doped_clifford_circuit(n_qubits, depth, int(kwargs["n_t_gates"]), rng)
    if family == "haar":
        return haar_random_unitary(n_qubits, rng)
    if family == "chaotic_ising":
        return chaotic_ising_unitary(n_qubits, float(kwargs.get("t_evolve", 1.0)))
    raise ValueError(f"unknown reservoir family {family!r}")
