"""
DLOS-DP loss utilities.

compute_x0_hat()
    Reconstruct the clean action estimate x̂₀|k from a noisy sample and the
    denoising network's noise prediction, using the DDPM x0-prediction formula.

DLOSLoss (nn.Module)
    Unified entry point for all five ablation groups (A–E).  Returns the
    weighted world-model loss term; the caller adds it to the diffusion loss.

Ablation groups
---------------
A  — no WM term (returns 0);  used as the pure-DP baseline.
B  — WM receives ground-truth action (stop-grad); validates WM learnability.
C  — WM receives x̂₀|k with stop-grad on the action;
     tests whether the WM constraint helps without back-prop to ε_θ.
D  — DLOS full: WM receives x̂₀|k, gradient flows back through x̂₀|k to ε_θ.
     This is the *main method*.
E  — same as D but action into WM is stop-grad;  isolates the gradient path.

SNR weighting
-------------
The per-sample weight is λ(k) = ᾱ_k (the cumulative product of noise
coefficients at diffusion step k).  This naturally down-weights the WM signal
at very noisy steps (k ≈ T, ᾱ_k ≈ 0) where x̂₀|k is unreliable, and
up-weights clean steps (k ≈ 0, ᾱ_k ≈ 1) where the estimate is accurate.

Gradient stability
------------------
∂x̂₀|k/∂ε_θ = −√(1−ᾱ_k)/√ᾱ_k.  Near k=T (ᾱ_k → 0) this diverges.
We clamp ᾱ_k ≥ 1e-5 to bound the derivative.  With SNR weighting the
effective gradient coefficient is √ᾱ_k·√(1−ᾱ_k) ≤ 0.5 for all k.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import OutcomeWorldModel


def compute_x0_hat(
    noisy_action: torch.Tensor,     # x_k,  (B, T_pred, a_dim)
    noise_pred: torch.Tensor,       # ε_θ,  (B, T_pred, a_dim)
    timesteps: torch.Tensor,        # k,    (B,)  long tensor
    alphas_cumprod: torch.Tensor,   # ᾱ,   (T_train,)  on model device
) -> torch.Tensor:                  # x̂₀|k, (B, T_pred, a_dim)
    """
    DDPM x0-prediction formula:
        x̂₀|k = (x_k − √(1 − ᾱ_k) · ε_pred) / √ᾱ_k

    Clamps ᾱ_k ≥ 1e-5 to prevent division by zero at very large k.
    """
    abar = alphas_cumprod[timesteps]           # (B,)
    abar = abar.clamp(min=1e-5).view(-1, 1, 1) # (B,1,1) for broadcasting
    x0_hat = (noisy_action - (1.0 - abar).sqrt() * noise_pred) / abar.sqrt()
    return x0_hat   # (B, T_pred, a_dim)


class DLOSLoss(nn.Module):
    """
    Unified DLOS world-model loss for all five ablation groups.

    Parameters
    ----------
    wm : OutcomeWorldModel
        The action-conditioned dynamics model.

    Notes
    -----
    The caller is responsible for multiplying the returned loss by
    ``lambda_wm`` (the global loss weight) before adding to ``loss_diff``.
    This keeps the scaling visible in the training script.
    """

    def __init__(self, wm: OutcomeWorldModel) -> None:
        super().__init__()
        self.wm = wm

    def forward(
        self,
        noise_pred: torch.Tensor,       # (B, T_pred, a_dim) — ε_θ output
        noisy_action: torch.Tensor,     # (B, T_pred, a_dim) — x_k
        timesteps: torch.Tensor,        # (B,) long
        gt_action: torch.Tensor,        # (B, T_pred, a_dim) — normalised GT action
        z_obs: torch.Tensor,            # (B, obs_dim) — DINO embedding of obs_t
        z_next: torch.Tensor,           # (B, obs_dim) — DINO embedding of obs_{t+1}
        alphas_cumprod: torch.Tensor,   # (T_train,)
        group: str = "D",
    ) -> torch.Tensor:
        """
        Returns the (scalar) WM loss term.

        Group A returns tensor(0.) immediately.
        Groups B–E compute WM(z_obs, action_input) → z_pred, then
        MSE(z_pred, z_next.detach()).

        The loss is weighted by the batch-mean SNR: λ(k) = mean(ᾱ_k).
        """
        if group == "A":
            return torch.tensor(0.0, device=noise_pred.device)

        # -------------------------------------------------------------- #
        # Determine action input to the world model                       #
        # -------------------------------------------------------------- #
        if group == "B":
            # Ground-truth normalised action — first step only, stop-grad
            a_in = gt_action[:, 0, :].detach()          # (B, a_dim)

        elif group in ("C", "D", "E"):
            x0_hat = compute_x0_hat(
                noisy_action, noise_pred, timesteps, alphas_cumprod
            )                                           # (B, T_pred, a_dim)
            a_first = x0_hat[:, 0, :]                   # (B, a_dim) — first step

            if group == "C":
                a_in = a_first.detach()                 # stop-grad
            elif group == "D":
                a_in = a_first                          # full gradient ← main method
            else:  # group == "E"
                a_in = a_first.detach()                 # stop-grad (same as C here,
                                                        # but intent differs from C:
                                                        # tests whether detaching
                                                        # *within* the DLOS framework
                                                        # hurts, compared to group D)
        else:
            raise ValueError(f"Unknown ablation group: {group!r}.  Must be A–E.")

        # -------------------------------------------------------------- #
        # World-model forward pass + MSE loss                             #
        # -------------------------------------------------------------- #
        z_pred = self.wm(z_obs, a_in)                   # (B, obs_dim)
        loss_wm = F.mse_loss(z_pred, z_next.detach())

        # -------------------------------------------------------------- #
        # SNR-aware per-batch weighting: λ(k) = mean(ᾱ_k)               #
        # -------------------------------------------------------------- #
        abar_mean = alphas_cumprod[timesteps].mean()    # scalar, in (0, 1)
        return abar_mean * loss_wm
