"""`(x_t, t) -> h`: the affine that turns 12 raw channels (10 latent coordinates + sin/cos of
the timestep) into the `[0,1]` local-detuning pattern the Rydberg Hamiltonian's `h_j` needs.

Fit on the training split only, then reused unchanged at validation, test and rollout time
(`EncodingAffine.transform` takes no data-dependent state) -- refitting per split would leak
distributional information from val/test into the affine, and refitting per rollout step
would make `h`'s meaning drift across DDIM steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _raw_channels(x_t: np.ndarray, t, T: int = 200) -> np.ndarray:
    """`(n, 12)`: `x_t` (10) followed by `sin(2*pi*t/T)`, `cos(2*pi*t/T)` -- the same (sin,
    cos) timestep embedding `denoiser_qrc.inject_time` uses for the digital arm, so the two
    reservoirs see the timestep on comparable footing even though they consume it
    differently (memory qubit vs. detuning channel)."""
    phase = 2 * np.pi * np.asarray(t, dtype=np.float64) / T
    return np.column_stack([np.asarray(x_t, dtype=np.float64), np.sin(phase), np.cos(phase)])


@dataclass
class EncodingAffine:
    """`h = clip((raw - mean_) / (k * std_) + 0.5, 0, 1)`, fit per-channel on train data.

    `k` sets how many std devs of the training distribution span `[0, 1]`: `k=4` puts
    `+-2 sigma` at the edges, so a channel that is exactly Gaussian on the fitting data
    saturates about 4.6% of the time by construction -- a deliberately nonzero base rate
    (an affine that never saturates on its own fitting data would hide saturation instead of
    measuring it, defeating the point of logging it).
    """
    mean_: np.ndarray  # (12,)
    std_: np.ndarray   # (12,)
    k: float = 4.0
    T: int = 200
    t_scale: float = 1.0
    """Post-standardization multiplier on the two timestep channels (sin, cos) only. Scaling
    the *raw* channel before fitting would be a no-op (`fit_encoding` divides by each
    channel's own std, so any constant factor cancels) -- this has to act on `z`, after
    standardization, to actually change how much of `h`'s dynamic range the timestep gets
    relative to the 10 latent channels."""

    def transform(self, x_t: np.ndarray, t) -> np.ndarray:
        raw = _raw_channels(x_t, t, self.T)
        z = (raw - self.mean_) / (self.k * self.std_)
        z[:, -2:] *= self.t_scale
        return np.clip(z + 0.5, 0.0, 1.0)

    def saturation_rate(self, x_t: np.ndarray, t) -> dict:
        """Fraction of (sample, channel) pairs pinned at each edge -- log this at fit time
        *and* separately at rollout time: `rollout` feeds `torch.clamp(x, -1, 1)` into the
        feature map while this affine is fit on unclamped `x_t`, so the two saturation rates
        measure different things (clamp-induced vs. genuine distribution shift) and
        conflating them hides whichever one is actually biting."""
        h = self.transform(x_t, t)
        return dict(at_zero=float((h <= 0.0).mean()), at_one=float((h >= 1.0).mean()),
                   per_channel_at_zero=(h <= 0.0).mean(0).tolist(),
                   per_channel_at_one=(h >= 1.0).mean(0).tolist())

    def save(self, path: str | Path) -> None:
        np.savez(path, mean_=self.mean_, std_=self.std_, k=self.k, T=self.T, t_scale=self.t_scale)

    @staticmethod
    def load(path: str | Path) -> "EncodingAffine":
        d = np.load(path)
        return EncodingAffine(d["mean_"], d["std_"], float(d["k"]), int(d["T"]),
                              float(d["t_scale"]) if "t_scale" in d else 1.0)


def fit_encoding(x_t_train: np.ndarray, t_train, k: float = 4.0, T: int = 200,
                 t_scale: float = 1.0) -> EncodingAffine:
    raw = _raw_channels(x_t_train, t_train, T)
    mean_ = raw.mean(0)
    std_ = raw.std(0)
    std_[std_ < 1e-8] = 1.0  # constant channel (shouldn't occur for real latents/timesteps)
    return EncodingAffine(mean_, std_, k, T, t_scale)
