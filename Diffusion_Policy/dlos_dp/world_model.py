"""
World-model modules for DLOS-DP.

OutcomeWorldModel  — lightweight MLP with LayerNorm + SiLU activations.
                     (obs_dim + action_dim) → obs_dim, ~247K params for
                     obs_dim=384, action_dim=2, hidden=256.
                     This is the primary model used in all ablation groups.

KoopmanWorldModel  — linear Koopman operator: z_{t+1} = K·z_t + B·a_t.
                     K is wrapped with spectral_norm so ||K||_2 ≤ 1, which
                     guarantees bounded forward-pass gradients.  Can be
                     swapped in as an alternative to OutcomeWorldModel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils


class OutcomeWorldModel(nn.Module):
    """
    MLP that predicts the next DINO observation embedding given the current
    embedding and the first action step.

    Architecture:
        [z_obs ‖ action]  →  Linear(obs_dim+action_dim, hidden)
                          →  LayerNorm(hidden)  →  SiLU
                          →  Linear(hidden, hidden)
                          →  LayerNorm(hidden)  →  SiLU
                          →  Linear(hidden, obs_dim)

    For obs_dim=384, action_dim=2, hidden=256 ≈ 247 K parameters.
    """

    def __init__(
        self,
        obs_dim: int = 384,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(
        self,
        z_obs: torch.Tensor,    # (B, obs_dim)
        action: torch.Tensor,   # (B, action_dim)
    ) -> torch.Tensor:          # (B, obs_dim)
        x = torch.cat([z_obs, action], dim=-1)
        return self.net(x)


class KoopmanWorldModel(nn.Module):
    """
    Linear Koopman operator: z_{t+1} = K · z_t + B · a_t.

    K is wrapped with spectral_norm so that ||K||_2 ≤ 1 throughout training,
    ensuring that forward-pass gradients do not blow up.

    spectral_penalty() returns relu(||K||_2 - 1); add this (scaled by a small
    coefficient, e.g. 1e-3) to the WM loss when training in stage 0 to keep
    the constraint tight without hard clipping.
    """

    def __init__(
        self,
        obs_dim: int = 384,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        # spectral_norm wraps the weight matrix so that the 2-norm is ≤ 1
        self.K = nn_utils.spectral_norm(
            nn.Linear(obs_dim, obs_dim, bias=False)
        )
        self.B = nn.Linear(action_dim, obs_dim, bias=False)

    def forward(
        self,
        z_obs: torch.Tensor,    # (B, obs_dim)
        action: torch.Tensor,   # (B, action_dim)
    ) -> torch.Tensor:          # (B, obs_dim)
        return self.K(z_obs) + self.B(action)

    def spectral_penalty(self) -> torch.Tensor:
        """
        Soft spectral-norm penalty: relu(sigma_1(K) - 1).
        Add to loss scaled by a small factor (e.g. 1e-3) during WM training.
        """
        # spectral_norm stores the singular vector u; the sigma estimate is
        # ||W_normalized|| * sigma, but the simpler approach is to use the
        # weight_orig attribute and compute directly.
        W = self.K.weight_orig  # raw weight before normalization
        sigma = torch.linalg.norm(W, ord=2)
        return torch.relu(sigma - 1.0)
