"""Content-addressed cache for Rydberg reservoir features.

Keyed by a hash of everything the physics *and* the numerics depend on -- geometry, pulse
parameters, the per-sample `h` array, probe time, shots, seed, and the emulator's own
accuracy knobs (`rk4_safety`, `dtype`). Leaving the numerics out of the key would make a
later accuracy change (e.g. tightening `rk4_safety` for a production run) silently reuse
features computed at the old, coarser precision -- the cache would report a hit when the
actual requested computation never ran. Two calls with identical inputs return the identical
cached array instead of recomputing; any change to any input produces a different key.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path("cache/quera_features")


def content_key(**parts) -> str:
    """`parts` values must be numpy arrays or JSON-primitive (str/int/float/bool/None/dict/
    list of those) -- keep them flat and primitive; hashing `repr()` of anything richer
    would collide-by-stringification instead of failing loudly."""
    h = hashlib.sha256()
    for name in sorted(parts):
        value = parts[name]
        h.update(name.encode())
        if isinstance(value, np.ndarray):
            h.update(np.ascontiguousarray(value).tobytes())
            h.update(repr((value.shape, str(value.dtype))).encode())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def load(key: str, cache_dir: Path = DEFAULT_CACHE_DIR):
    path = cache_dir / f"{key}.npy"
    return np.load(path) if path.exists() else None


def save(key: str, array: np.ndarray, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # `np.save` silently appends ".npy" to any path not already ending in it, so the tmp
    # name must end in ".npy" itself or the later rename targets a file that was never
    # written.
    tmp = cache_dir / f"{key}.tmp.npy"
    np.save(tmp, array)
    tmp.rename(cache_dir / f"{key}.npy")  # atomic rename on the same filesystem


def cached(compute, cache_dir: Path = DEFAULT_CACHE_DIR, **key_parts) -> np.ndarray:
    """`compute()` runs only on a cache miss. `key_parts` should include every argument
    `compute` closes over that affects its output -- see the module docstring."""
    key = content_key(**key_parts)
    hit = load(key, cache_dir)
    if hit is not None:
        return hit
    array = compute()
    save(key, array, cache_dir)
    return array
