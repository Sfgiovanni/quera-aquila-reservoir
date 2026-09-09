"""|psi|^2 -> noisy bitstrings, matching how the real hardware reads out: projective
measurement in the g/r basis at the end of the pulse, with per-shot atom loss and per-site
detection error -- not the exact expectation values `emulator.occupation_moments` gives.

Every noise source is independently switchable and documented, per the task spec, and all
default to 0 (an unnoised call reduces to plain multinomial sampling of `|psi|^2`).
"""
from __future__ import annotations

import torch

from quera.emulator import GeometryCache


def sample_bitstrings(psi: torch.Tensor, geom: GeometryCache, shots: int, generator=None) -> torch.Tensor:
    """`(batch, shots, n_sites)` bool, decoded with this project's own MSB-first site
    convention (`emulator.py`'s module docstring) -- self-consistent within this file and
    `features.py`, and never compared to bloqade's/braket's raw indices (see
    `docs/AQUILA_PORT.md`'s Gate 1 section on why that would be wrong)."""
    probs = (psi.conj() * psi).real
    idx = torch.multinomial(probs, shots, replacement=True, generator=generator)  # (batch, shots)
    n = geom.n_sites
    shifts = torch.arange(n - 1, -1, -1, device=idx.device)
    return ((idx.unsqueeze(-1) >> shifts) & 1).bool()  # (batch, shots, n)


def apply_readout_noise(bits: torch.Tensor, vacancy_rate: float = 0.0, fp_rate: float = 0.0,
                        fn_rate: float = 0.0, generator=None):
    """Atom loss and detection error on top of noiseless samples from `sample_bitstrings`.

    `vacancy_rate` is per-site, filtered **per cluster**: a shot is usable only if every one
    of its `n_sites` atoms survived, not if the (nonexistent, in this project) rest of a
    larger array did -- at `n_sites=12` and `vacancy_rate=0.01`, `0.99**12=88.6%` of shots
    are usable, matching the task spec's own worked number. `fp_rate`/`fn_rate` are
    independent because they act on disjoint populations (`fp` only flips true |g> sites,
    `fn` only flips true |r> sites), so applying both to the *original* bits rather than
    threading one through the other is correct, not an approximation.

    Returns `(noisy_bits, usable)`: `noisy_bits` is `(batch, shots, n_sites)` bool (readout
    error applied whether or not the shot ends up usable -- cheaper than branching, and the
    caller must mask by `usable` anyway); `usable` is `(batch, shots)` bool.
    """
    shape = bits.shape
    device = bits.device
    if vacancy_rate > 0:
        vacant = torch.rand(shape, device=device, generator=generator) < vacancy_rate
        usable = ~vacant.any(dim=-1)
    else:
        usable = torch.ones(shape[:-1], dtype=torch.bool, device=device)

    noisy = bits.clone()
    if fp_rate > 0:
        flip_fp = (~bits) & (torch.rand(shape, device=device, generator=generator) < fp_rate)
        noisy = noisy ^ flip_fp
    if fn_rate > 0:
        flip_fn = bits & (torch.rand(shape, device=device, generator=generator) < fn_rate)
        noisy = noisy ^ flip_fn
    return noisy, usable
