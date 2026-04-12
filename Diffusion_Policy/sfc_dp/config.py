"""Configuration for SFC-DP experiments."""
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

    lambda_freq: float = 0.1
    soft_mask: bool = False
    soft_mask_tau: float = 0.05

    seed: int = 42
    device: str = "auto"
    epochs: int = 100
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
