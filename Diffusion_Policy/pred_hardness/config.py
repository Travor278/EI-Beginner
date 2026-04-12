"""PHRew-DP 用到的统一配置。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PHRConfig:
    # 路径
    zarr_path: str = (
        "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )
    # Stage 1 训练好的状态空间前向模型
    probe_checkpoint: str = (
        "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
        "artifacts/state_forward_probe/best_state_forward_probe.pt"
    )
    dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy"
    # Stage 2b 生成的 chunk hardness
    chunk_hardness_path: str = (
        "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
        "artifacts/phrew/chunk_hardness.npy"
    )
    outdir: str = (
        "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
        "artifacts/phrew_runs"
    )

    # Push-T 任务配置
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    action_dim: int = 2
    num_train_timesteps: int = 100

    # 重加权方式
    mode: str = "soft_hard"
    temperature: float = 0.1
    alpha: float = 1.0

    # 训练超参数
    seed: int = 42
    device: str = "auto"
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    val_ratio: float = 0.1

    # Stage 2b 打分
    score_batch_size: int = 512
