"""DiT1d: a transformer backbone shared by diffusion_policy.py (as a noise
predictor) and bc.py (as a direct action regressor).

Ported from CleanDiffuser's cleandiffuser/nn_diffusion/dit.py, with
BaseNNDiffusion's timestep-embedding registry inlined (this repo always uses
the Fourier embedding) and the unused DiT1Ref cross-attention variant dropped.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal embedding, used here to encode each action-horizon
    step's position (not the diffusion timestep)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.einsum("...i,j->...ij", x, emb.to(x.dtype))
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FourierEmbedding(nn.Module):
    """Learned random-Fourier-feature embedding of the diffusion timestep."""

    def __init__(self, dim: int, scale: float = 16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 8) * scale, requires_grad=False)
        self.mlp = nn.Sequential(nn.Linear(dim // 4, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], -1)
        return self.mlp(emb)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), approx_gelu(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer1d(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1d(nn.Module):
    """
    Input:
        x:          (b, horizon, in_dim) -- a (possibly noisy) action sequence
        noise:      (b,) -- diffusion timestep index (DP), or a fixed dummy
                    value such as zeros (BC, see bc.py's BCPolicy)
        condition:  (b, emb_dim) or None -- observation embedding from
                    networks/vision_condition.py's MultiImageObsCondition

    Output:
        y:          (b, horizon, in_dim)

    All conditioning (timestep + observation) is injected exclusively through
    `emb`, which drives adaLN-Zero modulation at every block; `x` only
    contributes a per-horizon-step positional term. This is why the same
    class can be reused non-iteratively for BC: feeding zeros for `x`/`noise`
    still forces the output to be derived from the real observation
    condition via the same adaLN pathway DP uses, not a degenerate shortcut.
    """

    def __init__(
        self,
        in_dim: int,
        emb_dim: int,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim, self.emb_dim = in_dim, emb_dim
        self.d_model = d_model

        self.map_noise = FourierEmbedding(emb_dim)

        self.x_proj = nn.Linear(in_dim, d_model)
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, d_model), nn.Mish(), nn.Linear(d_model, d_model), nn.Mish())

        self.pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None

        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(depth)])
        self.final_layer = FinalLayer1d(d_model, in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.map_emb[0].weight, std=0.02)
        nn.init.normal_(self.map_emb[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, noise: torch.Tensor, condition: Optional[torch.Tensor] = None):
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != x.shape[1]:
            self.pos_emb_cache = self.pos_emb(torch.arange(x.shape[1], device=x.device))

        x = self.x_proj(x) + self.pos_emb_cache[None,]
        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + torch.zeros_like(emb)
        emb = self.map_emb(emb)

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_layer(x, emb)
        return x
