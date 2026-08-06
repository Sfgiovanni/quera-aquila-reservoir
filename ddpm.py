"""Classical Gaussian DDPM utilities used by Round 2."""

from __future__ import annotations

import numpy as np


def cosine_schedule(T: int = 20, s: float = 0.008):
    """Nichol-Dhariwal cosine schedule, indexed from t=1 through T."""
    steps = np.arange(T + 1, dtype=np.float64)
    abar = np.cos(((steps / T + s) / (1.0 + s)) * np.pi / 2.0) ** 2
    abar /= abar[0]
    betas = np.clip(1.0 - abar[1:] / abar[:-1], 1e-8, 0.999)
    return betas, np.cumprod(1.0 - betas)


def q_sample(x0: np.ndarray, t: np.ndarray, noise: np.ndarray, alpha_bar: np.ndarray):
    """Draw x_t and clamp it to the quadrature encoder domain [-1,1]."""
    a = alpha_bar[np.asarray(t, dtype=int) - 1, None]
    raw = np.sqrt(a) * x0 + np.sqrt(1.0 - a) * noise
    clipped = np.clip(raw, -1.0, 1.0)
    clamp_rate = float(np.mean(raw != clipped))
    return clipped, clamp_rate


def regression_design(xt: np.ndarray, t: np.ndarray, T: int) -> np.ndarray:
    """Raw latent plus a normalized scalar timestep for both classical models."""
    return np.column_stack([xt, 2.0 * np.asarray(t) / T - 1.0])
