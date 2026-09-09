"""Production-side handle on `_bloqade_worker.py`, so `rydberg_features` can get its populations
from QuEra's bloqade-analog instead of `quera/emulator.py`.

The two live in different virtualenvs on purpose (`docs/AQUILA_PORT.md`'s Environment section):
`.venv` has torch/CUDA and no bloqade, `.venv-gate1` has bloqade pinned against `numpy==2.1.3` and
no torch. That split is not an inconvenience to route around -- it is what keeps a bloqade version
bump from being able to affect the production arm at all. So this talks to a **subprocess**, over
pipes, exactly as `tests/test_emulator_vs_reference.py` already does for Gate 1.

One worker process is spawned per call site and kept alive across DDIM steps: bloqade's first solve
pays a JIT cost (2.13s measured, against 0.14s steady-state), and the internal `multiprocessing`
pool is rebuilt only when the pulse program changes, i.e. once per probe time rather than once per
batch. Throughput measured on this machine at the Gate 4 operating point: **0.139s per program on
one core**, so 16-24 cores put a full `samples=10000` generation cell at ~3h of CPU -- which is
what makes an end-to-end bloqade FID cell feasible at all.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = ROOT / '.venv-gate1' / 'bin' / 'python'
_ARRAY_KEYS = ('positions_um', 'times_us', 'omega', 'phase', 'delta_global', 'delta_local')


class BloqadePool:
    """`probs(program, h) -> (n, 2**n_sites)` populations, in this project's MSB-first index order.

    Not thread-safe and not re-entrant: one request is in flight at a time, by construction of the
    pipe protocol (a fixed-size response is read back before the next request is written).
    """

    def __init__(self, python: str | Path = DEFAULT_PYTHON, workers: int = 16):
        python = Path(python)
        if not python.exists():
            raise FileNotFoundError(
                f'{python} not found -- the bloqade backend needs the Gate 1 virtualenv '
                f'(bloqade-analog, numpy pinned to 2.1.3; see docs/AQUILA_PORT.md Environment)')
        env = dict(os.environ, PYTHONPATH=str(ROOT), BLOQADE_WORKERS=str(workers))
        self.proc = subprocess.Popen(
            [str(python), '-u', '-m', 'quera._bloqade_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def probs(self, program, h: np.ndarray) -> np.ndarray:
        h = np.ascontiguousarray(np.asarray(h, dtype=np.float64))
        n, n_sites = h.shape
        header = {'n': int(n), **{k: np.asarray(getattr(program, k)).tolist() for k in _ARRAY_KEYS}}
        self.proc.stdin.write((json.dumps(header) + '\n').encode())
        self.proc.stdin.write(h.tobytes())
        self.proc.stdin.flush()
        want = n * (1 << n_sites) * 4
        buf = bytearray()
        while len(buf) < want:
            chunk = self.proc.stdout.read(want - len(buf))
            if not chunk:
                raise RuntimeError(
                    f'bloqade worker died after {len(buf)}/{want} bytes '
                    f'(exit={self.proc.poll()}) -- check its stderr above')
            buf.extend(chunk)
        # np.frombuffer is read-only and torch warns (undefined behaviour if ever written to);
        # copy once here rather than leave a read-only tensor circulating downstream.
        return np.frombuffer(bytes(buf), dtype=np.float32).reshape(n, 1 << n_sites).copy()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"op": "stop"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_POOL = None


def shared_pool(workers: int = 16) -> BloqadePool:
    """One pool per process, reused across probe times and DDIM steps -- see the JIT note above."""
    global _POOL
    if _POOL is None:
        _POOL = BloqadePool(workers=workers)
    return _POOL
