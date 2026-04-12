"""SFCConfig: single dataclass holding all SFC-DP hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFCConfig:
    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    zarr_path: str = (
        "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )
    dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy"
    checkpoint_path: str = ""   # optional: resume from pre-trained DP checkpoint
    outdir: str = "artifacts/sfc_runs"

    # ------------------------------------------------------------------ #
    # Push-T task settings                                                 #
    # ------------------------------------------------------------------ #
    obs_horizon: int = 2
    pred_horizon: int = 16      # T_pred — action chunk length, DCT applied here
    action_horizon: int = 8
    action_dim: int = 2         # Push-T: (x, y)
    num_train_timesteps: int = 100

    # ------------------------------------------------------------------ #
    # Ablation group                                                       #
    # ------------------------------------------------------------------ #
    # A — baseline: standard DP, no L_freq
    # B — L_freq with full-band mask (mask = all 1s, no SNR alignment)
    #     → ablates whether SNR-aware masking is the key
    # C — L_freq + SNR mask, but applied to ε-space rather than x̂₀|k
    #     → ablates whether x₀-prediction space is better than ε-space
    # D — SFC-DP full: L_freq + SNR mask + x̂₀|k, gradient flows to ε_θ
    #     ← MAIN METHOD
    # E — same as D but stop-grad on x̂₀|k
    #     → ablates whether gradient path through x̂₀|k → ε_θ matters
    group: str = "D"

    # ------------------------------------------------------------------ #
    # SFC loss hyper-parameters                                            #
    # ------------------------------------------------------------------ #
    lambda_freq: float = 0.1    # weight on L_freq
    # Soft mask: instead of hard 0/1, use sigmoid gating
    # soft_mask=True: M(k)[f] = sigmoid((ᾱ_k - f/T_pred) / tau)
    # soft_mask=False: hard step function (default)
    soft_mask: bool = False
    soft_mask_tau: float = 0.05  # temperature for soft mask sigmoid

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    seed: int = 42
    device: str = "auto"
    epochs: int = 3000
    batch_size: int = 64
    lr: float = 1e-4
    val_ratio: float = 0.1
