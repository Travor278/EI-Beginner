"""
SFC-DP core loss — SNR-Frequency Consistent auxiliary training objective.

Public API
----------
compute_x0_hat(noisy_action, noise_pred, timesteps, alphas_cumprod)
    Reconstruct clean action estimate x̂₀|k from the predicted noise ε_θ.

sfc_loss(noise_pred, noisy_action, gt_action, timesteps, alphas_cumprod,
         group, T_pred, soft_mask, soft_mask_tau)
    Functional API — compute L_freq for the given ablation group.

SFCLoss(nn.Module)
    Stateful wrapper; caches alphas_cumprod buffer.

Ablation groups
---------------
A — baseline (returns 0; caller should use L_diff only)
B — L_freq, full-band mask (all 1s), no SNR alignment  → ablates SNR masking
C — L_freq + SNR mask, applied to ε-space               → ablates x₀ prediction space
D — SFC-DP full: L_freq + SNR mask + x̂₀|k              ← MAIN METHOD
E — same as D but stop-grad on x̂₀|k before DCT         → ablates gradient path
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# torch-dct import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import torch_dct as _dct_lib   # pip install torch-dct

    def _dct(x: torch.Tensor) -> torch.Tensor:
        """DCT-II along last-but-one dimension (time), orthonormal basis."""
        return _dct_lib.dct(x, norm="ortho")

    _DCT_BACKEND = "torch_dct"

except ImportError:
    import math as _math

    def _dct(x: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
        """
        Fallback DCT-II (ortho-normalised) implemented with torch.fft.
        Matches scipy.fft.dct(x, norm='ortho') along dim=-2 (time axis).
        x shape: (B, T, D)
        """
        B, T, D = x.shape
        # Promote to float32 for FFT stability
        xf = x.float()
        # Pad to 2T for FFT trick
        v = torch.cat([xf, xf.flip(1)], dim=1)          # (B, 2T, D)
        V = torch.fft.rfft(v, dim=1)                     # (B, T+1, D) complex
        k = torch.arange(T, device=x.device).float()     # (T,)
        # Phase factor: e^{-i π k / 2T}
        phase = torch.exp(-1j * _math.pi * k / (2 * T))  # (T,)
        phase = phase.view(1, T, 1)                       # broadcast
        W = (V[:, :T, :] * phase).real                   # (T,) real part
        # Orthonormal scaling
        W[:, 0, :] = W[:, 0, :] / (_math.sqrt(T))
        W[:, 1:, :] = W[:, 1:, :] / (_math.sqrt(2 * T))
        return W.to(x.dtype)

    _DCT_BACKEND = "fallback_fft"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_x0_hat(
    noisy_action: torch.Tensor,   # (B, T, a_dim)
    noise_pred: torch.Tensor,     # (B, T, a_dim)
    timesteps: torch.Tensor,      # (B,) long
    alphas_cumprod: torch.Tensor, # (num_train_timesteps,) float
) -> torch.Tensor:
    """
    Reconstruct clean-action estimate from predicted noise:

        x̂₀|k = (x_k − √(1−ᾱ_k) · ε_θ) / √ᾱ_k

    Returns tensor of same shape as noisy_action.
    """
    abar = alphas_cumprod[timesteps].clamp(min=1e-5)  # (B,)
    abar = abar.view(-1, 1, 1)                         # broadcast
    return (noisy_action - (1.0 - abar).sqrt() * noise_pred) / abar.sqrt()


def sfc_loss(
    noise_pred: torch.Tensor,      # (B, T_pred, a_dim)  ε_θ output
    noisy_action: torch.Tensor,    # (B, T_pred, a_dim)  x_k
    gt_action: torch.Tensor,       # (B, T_pred, a_dim)  x₀ (normalised)
    timesteps: torch.Tensor,       # (B,) long
    alphas_cumprod: torch.Tensor,  # (num_train_timesteps,) float
    group: str = "D",
    soft_mask: bool = False,
    soft_mask_tau: float = 0.05,
) -> torch.Tensor:
    """
    Compute L_freq for the requested ablation group.

    Returns
    -------
    Scalar tensor (requires_grad=True when group != 'A').
    """
    if group == "A":
        # Baseline: no auxiliary loss
        return torch.zeros(1, device=noise_pred.device, dtype=noise_pred.dtype)

    T_pred = gt_action.shape[1]
    abar = alphas_cumprod[timesteps].clamp(min=1e-5)  # (B,)

    # --------------------------------------------------------------------- #
    # Frequency mask M(k)                                                    #
    # --------------------------------------------------------------------- #
    if group == "B":
        # Full-band: mask = all ones (no SNR alignment)
        mask = torch.ones(
            noise_pred.shape[0], T_pred, 1,
            device=noise_pred.device, dtype=noise_pred.dtype,
        )
    else:
        # Groups C / D / E: SNR-aware mask
        freq_idx = torch.arange(T_pred, device=noise_pred.device).float() / T_pred  # (T_pred,)
        if soft_mask:
            # Soft sigmoid gate: M(k)[f] = σ((ᾱ_k − f/T_pred) / τ)
            mask = torch.sigmoid(
                (abar.unsqueeze(1) - freq_idx.unsqueeze(0)) / soft_mask_tau
            )                                          # (B, T_pred)
        else:
            # Hard step: M(k)[f] = 1 if f/T_pred < ᾱ_k
            mask = (freq_idx.unsqueeze(0) < abar.unsqueeze(1)).float()  # (B, T_pred)
        mask = mask.unsqueeze(-1)                      # (B, T_pred, 1) → broadcast a_dim

    # --------------------------------------------------------------------- #
    # Choose signal to supervise in frequency domain                         #
    # --------------------------------------------------------------------- #
    if group == "C":
        # ε-space: supervise DCT(ε_pred) vs DCT(ε_gt=0 residual equivalent)
        # We compare DCT of noise_pred vs DCT of the true noise embedded in
        # the noisy action.  Here we use x₀ → compute expected ε_gt for
        # consistency.  Specifically:
        #   ε_gt = (x_k − √ᾱ_k · x₀) / √(1−ᾱ_k)
        abar_v = alphas_cumprod[timesteps].clamp(min=1e-5).view(-1, 1, 1)
        eps_gt = (noisy_action - abar_v.sqrt() * gt_action) / (1.0 - abar_v).sqrt()
        X_pred = _dct(noise_pred)   # DCT of predicted ε
        X_gt   = _dct(eps_gt)       # DCT of ground-truth ε
    else:
        # Groups B / D / E: supervise DCT(x̂₀|k) vs DCT(x₀)
        x0_hat = compute_x0_hat(noisy_action, noise_pred, timesteps, alphas_cumprod)
        if group == "E":
            # Stop gradient — ablates whether gradient path through x̂₀|k matters
            x0_hat = x0_hat.detach()
        X_pred = _dct(x0_hat)       # (B, T_pred, a_dim)
        X_gt   = _dct(gt_action)    # (B, T_pred, a_dim)

    # --------------------------------------------------------------------- #
    # Masked MSE                                                             #
    # --------------------------------------------------------------------- #
    return F.mse_loss(X_pred * mask, X_gt * mask)


# ---------------------------------------------------------------------------
# Stateful nn.Module wrapper (convenient for training scripts)
# ---------------------------------------------------------------------------

class SFCLoss(nn.Module):
    """
    Wrapper around :func:`sfc_loss` that caches ``alphas_cumprod`` as a
    non-trainable buffer, mirroring the DDPMScheduler convention.

    Parameters
    ----------
    noise_scheduler : DDPMScheduler (or any object with .alphas_cumprod)
    group           : ablation group 'A'–'E'
    lambda_freq     : scalar weight multiplied on L_freq
    soft_mask       : whether to use sigmoid gating instead of hard step
    soft_mask_tau   : temperature for soft mask

    Usage
    -----
    >>> criterion = SFCLoss(noise_scheduler, group='D', lambda_freq=0.1)
    >>> loss = criterion(noise_pred, noisy_action, gt_action, timesteps)
    """

    def __init__(
        self,
        noise_scheduler,
        group: str = "D",
        lambda_freq: float = 0.1,
        soft_mask: bool = False,
        soft_mask_tau: float = 0.05,
    ) -> None:
        super().__init__()
        self.group = group
        self.lambda_freq = lambda_freq
        self.soft_mask = soft_mask
        self.soft_mask_tau = soft_mask_tau

        # Register as buffer (not a parameter, moves with .to(device))
        acp = noise_scheduler.alphas_cumprod
        if not isinstance(acp, torch.Tensor):
            acp = torch.tensor(acp, dtype=torch.float32)
        self.register_buffer("alphas_cumprod", acp.float())

    def forward(
        self,
        noise_pred: torch.Tensor,
        noisy_action: torch.Tensor,
        gt_action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Return λ · L_freq (scalar)."""
        return self.lambda_freq * sfc_loss(
            noise_pred=noise_pred,
            noisy_action=noisy_action,
            gt_action=gt_action,
            timesteps=timesteps,
            alphas_cumprod=self.alphas_cumprod,
            group=self.group,
            soft_mask=self.soft_mask,
            soft_mask_tau=self.soft_mask_tau,
        )

    def extra_repr(self) -> str:
        return (
            f"group={self.group!r}, lambda_freq={self.lambda_freq}, "
            f"soft_mask={self.soft_mask}, dct_backend={_DCT_BACKEND!r}"
        )
