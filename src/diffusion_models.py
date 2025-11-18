"""
Diffusion models for music generation.

This module implements state-of-the-art diffusion models for both:
- Discrete diffusion for symbolic music (MIDI tokens)
- Continuous diffusion for audio synthesis

Based on:
- DDPM (Denoising Diffusion Probabilistic Models)
- DDIM (Denoising Diffusion Implicit Models)
- Score-based models
- Discrete diffusion (for tokens)

Author: Claude
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import math


class NoiseSchedule(Enum):
    """Noise schedule types."""
    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model."""
    num_timesteps: int = 1000
    noise_schedule: NoiseSchedule = NoiseSchedule.COSINE
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"  # "epsilon", "x0", or "v"


class NoiseScheduler:
    """Manages noise scheduling for diffusion process."""

    def __init__(self, config: DiffusionConfig):
        """Initialize noise scheduler."""
        self.config = config
        self.betas = self._create_noise_schedule()

        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule."""
        T = self.config.num_timesteps

        if self.config.noise_schedule == NoiseSchedule.LINEAR:
            # Linear schedule
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                T,
                dtype=torch.float32
            )

        elif self.config.noise_schedule == NoiseSchedule.COSINE:
            # Cosine schedule (Improved DDPM)
            def cosine_schedule(t):
                s = 0.008  # Offset
                f_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
                f_0 = torch.cos(s / (1 + s) * math.pi / 2) ** 2
                return torch.clamp((1 - f_t / f_0), min=0, max=0.999)

            t = torch.arange(T, dtype=torch.float32)
            betas = cosine_schedule(t)

        elif self.config.noise_schedule == NoiseSchedule.SIGMOID:
            # Sigmoid schedule
            t = torch.linspace(0, 1, T, dtype=torch.float32)
            sigmoid = torch.sigmoid(10 * (t - 0.5))
            betas = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * sigmoid

        else:
            raise ValueError(f"Unknown noise schedule: {self.config.noise_schedule}")

        return betas

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).

        Args:
            x_0: Original data [batch, ...]
            t: Timesteps [batch]
            noise: Optional noise tensor

        Returns:
            Noised data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_0.ndim - 1)))
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_0.ndim - 1)))

        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def predict_x0_from_epsilon(self, x_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from epsilon prediction."""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_t.ndim - 1)))
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_t.ndim - 1)))

        return (x_t - sqrt_one_minus_alpha_t * epsilon) / sqrt_alpha_t


class UNet1D(nn.Module):
    """
    1D U-Net for sequence modeling (MIDI/music).

    Used as denoising network for diffusion models.
    """

    def __init__(self,
                 in_channels: int = 128,
                 model_channels: int = 256,
                 num_res_blocks: int = 2,
                 channel_mult: Tuple[int, ...] = (1, 2, 4),
                 num_timesteps: int = 1000):
        """
        Initialize 1D U-Net.

        Args:
            in_channels: Number of input channels (vocab size)
            model_channels: Base channel count
            num_res_blocks: Number of residual blocks per level
            channel_mult: Channel multipliers for each level
            num_timesteps: Number of diffusion timesteps
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock1D(ch, out_ch, time_embed_dim)
                )
                ch = out_ch
            if i < len(channel_mult) - 1:
                self.down_blocks.append(Downsample1D(ch))

        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResBlock1D(ch, ch, time_embed_dim),
            AttentionBlock1D(ch),
            ResBlock1D(ch, ch, time_embed_dim),
        ])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock1D(ch + out_ch, out_ch, time_embed_dim)
                )
                ch = out_ch
            if i > 0:
                self.up_blocks.append(Upsample1D(ch))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv1d(ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, channels, length]
            t: Timesteps [batch]

        Returns:
            Denoised output [batch, channels, length]
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # Input projection
        h = self.input_proj(x)

        # Downsampling
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResBlock1D):
                h = block(h, t_emb)
            else:
                h = block(h)
            hs.append(h)

        # Middle
        for block in self.middle_blocks:
            if isinstance(block, ResBlock1D):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResBlock1D):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)

        # Output
        return self.output_proj(h)

    @staticmethod
    def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    """Residual block for 1D sequences."""

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_proj(F.silu(t_emb))[:, :, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock1D(nn.Module):
    """Self-attention block for 1D sequences."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x)

        # QKV
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, L).transpose(2, 3)
        k = k.view(B, self.num_heads, head_dim, L).transpose(2, 3)
        v = v.view(B, self.num_heads, head_dim, L).transpose(2, 3)

        # Attention
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        h = attn @ v
        h = h.transpose(2, 3).contiguous().view(B, C, L)

        # Output projection
        h = self.proj_out(h)

        return x + h


class Downsample1D(nn.Module):
    """Downsampling layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsampling layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DiffusionModel(nn.Module):
    """Main diffusion model for music generation."""

    def __init__(self,
                 denoising_network: nn.Module,
                 config: DiffusionConfig):
        """
        Initialize diffusion model.

        Args:
            denoising_network: Neural network that predicts noise
            config: Diffusion configuration
        """
        super().__init__()
        self.network = denoising_network
        self.config = config
        self.scheduler = NoiseScheduler(config)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x_0: Clean data [batch, ...]

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.scheduler.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.network(x_t, t)

        # Calculate loss
        if self.config.prediction_type == "epsilon":
            loss = F.mse_loss(predicted_noise, noise)
        elif self.config.prediction_type == "x0":
            loss = F.mse_loss(predicted_noise, x_0)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")

        return loss

    @torch.no_grad()
    def sample(self,
               shape: Tuple[int, ...],
               conditioning: Optional[torch.Tensor] = None,
               num_steps: Optional[int] = None,
               eta: float = 0.0) -> torch.Tensor:
        """
        Sample from diffusion model (DDIM sampling).

        Args:
            shape: Shape of output
            conditioning: Optional conditioning tensor
            num_steps: Number of sampling steps (fewer = faster)
            eta: DDIM eta parameter (0 = deterministic, 1 = DDPM)

        Returns:
            Generated samples
        """
        device = next(self.network.parameters()).device
        batch_size = shape[0]

        if num_steps is None:
            num_steps = self.config.num_timesteps

        # Create sampling schedule
        timesteps = torch.linspace(
            self.config.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=device
        )

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Iterative denoising
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.network(x_t, t_batch)

            # Predict x_0
            x_0_pred = self.scheduler.predict_x0_from_epsilon(x_t, t_batch, predicted_noise)

            # DDIM update
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_t_next = self.scheduler.alphas_cumprod[t_next]

                sigma = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

                mean = torch.sqrt(alpha_t_next) * x_0_pred + torch.sqrt(1 - alpha_t_next - sigma ** 2) * predicted_noise

                if eta > 0:
                    noise = torch.randn_like(x_t)
                    x_t = mean + sigma * noise
                else:
                    x_t = mean
            else:
                x_t = x_0_pred

        return x_t


class DiscreteDiffusion(nn.Module):
    """
    Discrete diffusion for symbolic music (tokens).

    Based on "Structured Denoising Diffusion Models in Discrete State-Spaces".
    """

    def __init__(self,
                 denoising_network: nn.Module,
                 num_classes: int,
                 num_timesteps: int = 1000):
        """
        Initialize discrete diffusion.

        Args:
            denoising_network: Network that predicts class logits
            num_classes: Vocabulary size
            num_timesteps: Number of diffusion steps
        """
        super().__init__()
        self.network = denoising_network
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Create transition matrices
        self._create_transition_matrices()

    def _create_transition_matrices(self):
        """Create transition matrices for discrete diffusion."""
        # Uniform transition (absorbing state model)
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)

        Qt = []
        Qt_bar = []

        Q_bar = torch.eye(self.num_classes)

        for t in range(self.num_timesteps):
            beta_t = betas[t]

            # Transition matrix: gradually move to uniform
            Q_t = (1 - beta_t) * torch.eye(self.num_classes) + beta_t / self.num_classes

            Qt.append(Q_t)

            # Cumulative transition
            Q_bar = Q_bar @ Q_t
            Qt_bar.append(Q_bar)

        self.register_buffer('Qt', torch.stack(Qt))
        self.register_buffer('Qt_bar', torch.stack(Qt_bar))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion for discrete data.

        Args:
            x_0: Original tokens [batch, length]
            t: Timesteps [batch]

        Returns:
            Noised tokens
        """
        batch_size, length = x_0.shape

        # Get transition probabilities
        Q_t_bar = self.Qt_bar[t]  # [batch, num_classes, num_classes]

        # Convert tokens to one-hot
        x_0_onehot = F.one_hot(x_0, self.num_classes).float()  # [batch, length, num_classes]

        # Apply transition
        x_t_probs = torch.einsum('bln,bnm->blm', x_0_onehot, Q_t_bar)

        # Sample from categorical
        x_t = torch.multinomial(x_t_probs.view(-1, self.num_classes), 1).view(batch_size, length)

        return x_t

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Sample from discrete diffusion.

        Args:
            shape: (batch_size, length)

        Returns:
            Generated tokens
        """
        device = next(self.network.parameters()).device
        batch_size, length = shape

        # Start from uniform random
        x_t = torch.randint(0, self.num_classes, shape, device=device)

        # Iterative denoising
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict logits
            logits = self.network(x_t, t_batch)

            # Sample from predicted distribution
            probs = F.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.view(-1, self.num_classes), 1).view(batch_size, length)

        return x_t


# Example usage
if __name__ == '__main__':
    print("=== Continuous Diffusion (for audio) ===")
    config = DiffusionConfig(num_timesteps=1000, noise_schedule=NoiseSchedule.COSINE)

    # Create U-Net
    unet = UNet1D(in_channels=128, model_channels=256)

    # Create diffusion model
    diffusion = DiffusionModel(unet, config)

    # Training example
    x_0 = torch.randn(4, 128, 256)  # [batch, channels, length]
    loss = diffusion(x_0)
    print(f"Training loss: {loss.item():.4f}")

    # Sampling example
    samples = diffusion.sample((4, 128, 256), num_steps=50)
    print(f"Generated samples shape: {samples.shape}")

    print("\n=== Discrete Diffusion (for MIDI tokens) ===")
    vocab_size = 512
    discrete_unet = UNet1D(in_channels=vocab_size, model_channels=256)
    discrete_diffusion = DiscreteDiffusion(discrete_unet, num_classes=vocab_size)

    # Training example
    tokens = torch.randint(0, vocab_size, (4, 128))
    t = torch.randint(0, 1000, (4,))
    noised_tokens = discrete_diffusion.q_sample(tokens, t)
    print(f"Original tokens: {tokens[0, :10]}")
    print(f"Noised tokens: {noised_tokens[0, :10]}")

    # Sampling example
    generated_tokens = discrete_diffusion.sample((4, 128))
    print(f"Generated tokens shape: {generated_tokens.shape}")
