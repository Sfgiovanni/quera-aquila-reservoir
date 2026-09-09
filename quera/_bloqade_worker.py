"""Runs under `.venv-gate1`: turns batches of `h` vectors into `|psi|^2` using bloqade-analog.

This exists so that a full generation cell can be produced with **QuEra's own emulator** doing the
physics, rather than `quera/emulator.py`. It deliberately returns raw populations and nothing else:
every downstream step (Z/ZZ observables, shot sampling, vacancy filtering, the readout, DDIM, FID)
stays in this project's already-tested code, so the only thing swapped out is the Schrodinger solve.
That is the division of labour a reviewer should want -- it makes the claim "the physics came from
bloqade" exact, instead of "we reimplemented the pipeline twice and the numbers looked similar".

**Index convention -- the one place this can silently go wrong.** bloqade's
`state_vector.space.configurations` are integers with **site 0 at the least significant bit**
(`AtomType.integer_to_string` peels site 0 off first); this project's `emulator.py` puts **site 0
at the most significant bit** (see its module docstring, and `docs/AQUILA_PORT.md`'s Gate 1
section, where all three implementations' conventions were verified empirically rather than
assumed). The reversal below is the whole reason this worker returns populations in *our* index
order rather than bloqade's -- getting it wrong would produce a plausible-looking permuted state
that fails no assertion. `tests/test_bloqade_backend.py` checks it against `quera/emulator.py` on
real production programs rather than trusting the reasoning.

bloqade's space may also be a blockade-truncated subspace smaller than `2**n_sites`, so the
populations are *scattered* into a full-length vector rather than assumed dense and ordered.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

_ARRAY_KEYS = ('positions_um', 'times_us', 'omega', 'phase', 'delta_global', 'delta_local')


def _probs_for(task):
    """`task` carries its own pulse spec rather than inheriting one from a pool initializer.

    The spec changes on **every** request from `rollout` (`rydberg_features` loops the `v_slices`
    probe times inside each batch), so a pool keyed on the spec is torn down and re-forked ~1000
    times in a full-scale cell. Passing the spec per task (under 1KB) instead keeps one pool alive.

    **This was expected to be a speedup and measured as not one.** Both designs run a full-scale
    cell at the same 3.3 cache-entries/min, i.e. ~5.0h. The earlier "~30% overhead" reasoning
    compared against a pure-solve floor computed for 22 free cores, ignoring that Gate 6 owned one
    and that IPC and pickling are not free; the real efficiency is ~78% of the solve-time floor
    either way, and the pool rebuild was never the bottleneck. The change is kept because it makes
    worker memory explicit and tunable (see `maxtasksperchild` below), not because it is faster.
    """
    from quera.device import Program
    from quera.program import to_bloqade

    spec, h = task
    program = Program(h=np.asarray(h, dtype=np.float64),
                      **{k: np.asarray(spec[k], dtype=np.float64) for k in _ARRAY_KEYS})
    emu = to_bloqade(program).bloqade.python()
    data, configs = emu.run_callback(
        lambda sv, *a: (np.asarray(sv.data), np.asarray(sv.space.configurations)))[0]

    n = program.n_sites
    # bloqade LSB-first -> this project's MSB-first, then scatter into the full 2**n space.
    idx = np.zeros(len(configs), dtype=np.int64)
    for j in range(n):
        idx |= ((configs >> j) & 1).astype(np.int64) << (n - 1 - j)
    out = np.zeros(1 << n, dtype=np.float64)
    out[idx] = np.abs(data) ** 2
    return out


def main():
    workers = int(os.environ.get('BLOQADE_WORKERS', '16'))
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    pool = None
    try:
        while True:
            header = stdin.readline()
            if not header:
                break
            msg = json.loads(header)
            if msg.get('op') == 'stop':
                break
            n_sites = len(msg['positions_um'])
            h = np.frombuffer(stdin.read(msg['n'] * n_sites * 8), dtype=np.float64)
            h = h.reshape(msg['n'], n_sites)
            spec = {k: msg[k] for k in _ARRAY_KEYS}
            # One pool for the whole process lifetime, but recycled aggressively: bloqade grows
            # roughly 250MB -> 700MB+ RSS over a few thousand solves, so at 22 workers a
            # `maxtasksperchild` of 2000 walked a 15.7GB machine down to 400MB free in four
            # minutes. 250 caps a worker near its fresh size; the re-fork plus per-process JIT
            # costs ~18 min over a full-scale cell, against the ~1.5h the previous
            # rebuild-per-request design cost. Tunable because the right value depends on how
            # much RAM the rest of the box is using.
            if pool is None:
                pool = Pool(workers, maxtasksperchild=int(os.environ.get('BLOQADE_MAXTASKS', '250')))
            probs = np.asarray(pool.map(_probs_for, [(spec, row) for row in h], chunksize=4),
                               dtype=np.float32)
            stdout.write(probs.tobytes())
            stdout.flush()
    finally:
        if pool is not None:
            pool.close(); pool.join()


if __name__ == '__main__':
    main()
