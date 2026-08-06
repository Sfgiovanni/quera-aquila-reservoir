"""De Falco-style three-block classical latent epsilon denoiser."""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int, T: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    phase = (t.float() / T)[:, None] * freq[None] * 1000.0
    emb = torch.cat([phase.sin(), phase.cos()], 1)
    return torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))


class ClassicalDenoiser(nn.Module):
    """Three d->d affine maps mirroring the paper's x, time, and merge blocks."""
    def __init__(self, latent_dim: int, T: int):
        super().__init__(); self.latent_dim=latent_dim; self.T=T
        self.x_block=nn.Linear(latent_dim,latent_dim)
        self.t_block=nn.Linear(latent_dim,latent_dim)
        self.merge_block=nn.Linear(latent_dim,latent_dim)

    def forward(self,x,t):
        hx=torch.tanh(self.x_block(x)); ht=torch.tanh(self.t_block(sinusoidal_embedding(t,self.latent_dim,self.T)))
        return x+self.merge_block(torch.tanh(hx+ht))

    @property
    def n_trainable(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
