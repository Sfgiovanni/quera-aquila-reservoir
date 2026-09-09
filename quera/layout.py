"""Atom layout: the fixed 3x4 cluster geometry and position quantization.

The encoding spec fixes a single geometry -- a 3x4 rectangular cluster at spacing
`a=8 um` (the "rich regime is a~R_b" point) -- reused for every sample and every probe
time. `a` is a free parameter of the Gate 4 sweep, everything else about the geometry is
not.
"""
from __future__ import annotations

import numpy as np

from quera.device import POSITION_RESOLUTION_UM

N_ROWS = 3
N_COLS = 4
DEFAULT_SPACING_UM = 8.0


def quantize_positions(positions_um: np.ndarray) -> np.ndarray:
    """Round to the nearest 10 nm grid point -- required before `device.validate`."""
    pos = np.asarray(positions_um, dtype=np.float64)
    ticks = np.round(pos / POSITION_RESOLUTION_UM)
    return ticks * POSITION_RESOLUTION_UM


def cluster_3x4(spacing_um: float = DEFAULT_SPACING_UM) -> np.ndarray:
    """A 3-row by 4-column rectangular grid, `spacing_um` apart, quantized to 10 nm.

    12 sites total: 10 for the latent coordinates, 2 for the (sin, cos) timestep
    channel, in row-major order matching `encoding.py`'s site assignment.
    """
    rows, cols = np.meshgrid(np.arange(N_ROWS), np.arange(N_COLS), indexing="ij")
    positions = np.stack([cols.ravel(), rows.ravel()], axis=1).astype(np.float64) * spacing_um
    return quantize_positions(positions)
