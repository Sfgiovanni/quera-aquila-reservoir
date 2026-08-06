"""Latent standardisation for the diffusion arms.

The cosine schedule assumes unit-variance data, but the autoencoder emits `latent_scale * tanh(.)`
with SD ~0.17-0.22. Training diffusion directly on those latents leaves the true SNR at ~sigma^2 of
the nominal schedule, so for most of the trajectory the denoiser sees essentially pure noise, the
x0 prediction `(x_t - sqrt(1-abar) eps)/sqrt(abar)` diverges, and the clamp pins 25-35% of
coordinates to the latent boundary. Validated on Fashion-MNIST: classical d32 goes from FID
142.7 +/- 0.9 to ~41.5, and clamp saturation from 28% to 0.07%.

Usage at every call site is three lines:

    scale = LatentScale(z)
    model = train(seed, scale.forward(z), ...)
    images = decode_numpy(ae, scale.inverse(sample(model, noise, abar, x0_clip=scale.x0_clip)))

so the autoencoder always sees encoder-unit latents and only the diffusion operates standardised.
See SUPERVISION_breastmnist.md for the measurements behind this.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LATENT_SCALE = 0.5


class LatentScale:
    """Scalar standardisation of a latent set, with the matching x0 clamp bound.

    `clamp_mult` tunes the clamp relative to the encoder bound. The inherited 1.0 is what every
    result before 2026-08-03 used; a sweep on BreastMNIST found 0.8 brings the generated latent
    distribution to parity with a fitted Gaussian (energy 0.0030 vs 0.0029, p = 0.80). The optimum
    is arm-dependent -- readouts with weaker eps prediction need more headroom -- so it should be
    calibrated per arm rather than shared, and is left at 1.0 here to keep re-runs comparable to
    the archived results they replace.
    """

    def __init__(self, latents, latent_scale=DEFAULT_LATENT_SCALE, clamp_mult=1.0, enabled=True):
        self.enabled = enabled
        self.sigma = float(np.asarray(latents).std()) if enabled else 1.0
        self.latent_scale = latent_scale
        self.clamp_mult = clamp_mult

    @property
    def x0_clip(self):
        return self.latent_scale * self.clamp_mult / self.sigma

    def forward(self, z):
        return np.asarray(z) / self.sigma

    def inverse(self, z):
        return np.asarray(z) * self.sigma

    def describe(self):
        return {"latent_sigma": self.sigma, "x0_clip": self.x0_clip,
                "clamp_mult": self.clamp_mult, "standardised": self.enabled}
