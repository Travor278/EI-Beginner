"""Configuration for SFC-DP experiments.

Task presets
------------
Push-T (default):
    zarr_path  = .../pusht_cchi_v7_replay.zarr
    action_dim = 2, obs_horizon = 2, pred_horizon = 16, action_horizon = 8
    lambda_freq default = 0.001  (optimal λ for Push-T = 0.01; see Section 12)

Robomimic Lift / Can  (use train_sfc_robomimic.py):
    zarr_path  = .../robomimic/lift/ph/image_abs.hdf5   (or .zarr)
    action_dim = 7, obs_horizon = 2, pred_horizon = 16, action_horizon = 8
    lambda_freq default = 0.01   (start with Push-T optimal; tune if needed)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFCConfig:
    zarr_path: str = (
        "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )
    dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy"
    checkpoint_path: str = ""
    outdir: str = "artifacts/sfc_runs"

    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    action_dim: int = 2
    num_train_timesteps: int = 100

    group: str = "D"

    # ---------------------------------------------------------------------- #
    # L_freq weight and warmup                                                 #
    # ---------------------------------------------------------------------- #
    # Empirical finding (seed=42, 100 epochs):
    #   epoch-0 L_freq ≈ 87 (untrained network), L_diff ≈ 0.22
    #   λ=0.1 → λ·L_freq ≈ 8.76, which is 39× L_diff → training collapses
    #   for the first 30 epochs before L_freq decays.
    #   Fix: reduce λ to 0.001 AND ramp it up over lambda_warmup_epochs so
    #   the effective weight starts near 0 and reaches λ_freq by warmup end.
    lambda_freq: float = 0.001          # was 0.1; reduced 100× to keep
                                        # λ·L_freq ≤ L_diff at epoch 0
    lambda_warmup_epochs: int = 20      # linearly ramp λ from 0 → lambda_freq
    soft_mask: bool = False
    soft_mask_tau: float = 0.05

    seed: int = 42
    device: str = "auto"
    epochs: int = 100   # 100 for first-pass; use 200 for extended convergence check
                        # (A42 still rising at ep90 → both A and D may not have converged)
    batch_size: int = 16
    lr: float = 1e-4
    val_ratio: float = 0.1

    crop_shape: tuple[int, int] = (84, 84)
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, int, int] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    obs_encoder_group_norm: bool = True
    eval_fixed_crop: bool = True
    num_inference_steps: int = 100
