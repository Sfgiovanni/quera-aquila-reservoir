"""Bounded convolutional Fashion-MNIST autoencoder used by every denoiser arm."""

from __future__ import annotations

import numpy as np


def _torch():
    import torch
    from torch import nn
    return torch, nn


def build_autoencoder(latent_dim: int, latent_scale: float = 0.5):
    torch, nn = _torch()

    class FashionAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = latent_dim
            self.latent_scale = latent_scale
            # De Falco Fig. 3: 28x28x1 -> 14x14x64 -> 7x7x128 -> 3x3x512.
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
                nn.Conv2d(128, 512, 3, 2, 0), nn.SiLU(),
            )
            self.encoder_dense = nn.Sequential(nn.Flatten(), nn.Linear(4608, 1024), nn.SiLU(),
                                               nn.Linear(1024, latent_dim))
            self.decoder_dense = nn.Sequential(nn.Linear(latent_dim, 1024), nn.SiLU(),
                                               nn.Linear(1024, 4608), nn.SiLU())
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(512, 128, 3, 2, 0), nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
            )

        def encode(self, x):
            return self.latent_scale * torch.tanh(self.encoder_dense(self.encoder_conv(x)))

        def decode(self, z):
            h = self.decoder_dense(z / self.latent_scale).reshape(-1, 512, 3, 3)
            return self.decoder_conv(h)

        def forward(self, x):
            return self.decode(self.encode(x))

    return FashionAutoencoder()


def load_autoencoder(path, device="cuda"):
    torch, _ = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_autoencoder(payload["latent_dim"], payload["latent_scale"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def encode_numpy(model, images: np.ndarray, batch_size=1024):
    torch, _ = _torch(); device = next(model.parameters()).device; out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.as_tensor(images[i:i + batch_size, None], dtype=torch.float32, device=device)
            out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out)


def decode_numpy(model, latents: np.ndarray, batch_size=1024):
    torch, _ = _torch(); device = next(model.parameters()).device; out = []
    with torch.inference_mode():
        for i in range(0, len(latents), batch_size):
            z = torch.as_tensor(latents[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model.decode(z).squeeze(1).cpu().numpy())
    return np.concatenate(out)
