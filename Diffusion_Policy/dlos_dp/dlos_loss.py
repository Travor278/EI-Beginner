"""Loss helpers for DLOS-DP."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import OutcomeWorldModel


def compute_x0_hat(
    noisy_action: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Recover x0 from an epsilon-prediction DDPM parameterization."""
    abar = alphas_cumprod[timesteps].clamp(min=1e-5).view(-1, 1, 1)
    return (noisy_action - (1.0 - abar).sqrt() * noise_pred) / abar.sqrt()


class DLOSLoss(nn.Module):
    """
    World-model term used by DLOS-DP.

    Group semantics:
      A: no world-model loss
      B: GT action
      C: final-step x0 estimate from a clean-step forward pass
      D: intermediate x0_hat at random k, gradient flows to epsilon_theta
      E: same as D, but x0_hat is detached before the WM
    """

    def __init__(self, wm: OutcomeWorldModel) -> None:
        super().__init__()
        self.wm = wm

    def forward(
        self,
        *,
        noise_pred: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        gt_action: torch.Tensor,
        z_obs: torch.Tensor,
        z_next: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        group: str,
        final_x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if group == "A":
            return torch.tensor(0.0, device=noise_pred.device)

        abar_mean = alphas_cumprod[timesteps].mean()

        if group == "B":
            action_in = gt_action[:, 0, :].detach()
        elif group == "C":
            if final_x0 is None:
                raise ValueError("group C requires final_x0.")
            action_in = final_x0[:, 0, :].detach()
        elif group == "D":
            x0_hat = compute_x0_hat(
                noisy_action=noisy_action,
                noise_pred=noise_pred,
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
            )
            action_in = x0_hat[:, 0, :]
        elif group == "E":
            x0_hat = compute_x0_hat(
                noisy_action=noisy_action,
                noise_pred=noise_pred,
                timesteps=timesteps,
                alphas_cumprod=alphas_cumprod,
            )
            action_in = x0_hat[:, 0, :].detach()
        else:
            raise ValueError(f"Unknown group: {group}")

        z_pred = self.wm(z_obs, action_in)
        loss_wm = F.mse_loss(z_pred, z_next.detach())

        if group in {"D", "E"}:
            return abar_mean * loss_wm
        return loss_wm
