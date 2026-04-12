"""Configuration for DLOS-DP experiments."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DLOSConfig:
    obs_dim: int = 384
    action_dim: int = 2
    wm_hidden: int = 256

    lambda_wm: float = 0.1

    dino_model: str = "dinov2_vits14"
    dino_img_size: int = 224

    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    num_train_timesteps: int = 100

    group: str = "D"
    dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy"
    zarr_path: str = (
        "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )
    checkpoint_path: str = ""

    seed: int = 42
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    val_ratio: float = 0.02
    max_train_episodes: int | None = 90
    pad_before: int = 1
    pad_after: int = 7
    outdir: str = "artifacts/dlos_runs"
    crop_shape: tuple[int, int] = (84, 84)
    diffusion_step_embed_dim: int = 128
    down_dims: tuple[int, int, int] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    obs_encoder_group_norm: bool = True
    eval_fixed_crop: bool = True
    num_inference_steps: int = 100

    wm_probe_epochs: int = 50
    wm_probe_lr: float = 1e-4
    dino_embed_cache: str = "artifacts/dino_embeddings.npy"
    wm_probe_outdir: str = "artifacts/wm_probe"
