"""PHRConfig: single dataclass holding all PHRew-DP hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PHRConfig:
    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    zarr_path: str = (
        "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )
    # Trained state-space forward probe (Stage 1 output, already available)
    probe_checkpoint: str = (
        "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
        "artifacts/state_forward_probe/best_state_forward_probe.pt"
    )
    dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy"
    # Chunk hardness scores (Stage 2b output)
    chunk_hardness_path: str = (
        "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
        "artifacts/phrew/chunk_hardness.npy"
    )
    outdir: str = (
        "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
        "artifacts/phrew_runs"
    )

    # ------------------------------------------------------------------ #
    # Push-T task settings (must match zarr data)                         #
    # ------------------------------------------------------------------ #
    obs_horizon: int = 2        # observation frames per sample
    pred_horizon: int = 16      # action chunk length
    action_horizon: int = 8     # executed steps per inference step
    num_train_timesteps: int = 100  # DDPM diffusion steps

    # ------------------------------------------------------------------ #
    # Reweighting                                                          #
    # ------------------------------------------------------------------ #
    # Sampling mode:
    #   uniform   — standard DP baseline, all weights = 1
    #   hard      — weights ∝ hardness^alpha  (focus on hardest chunks)
    #   soft_hard — weights ∝ softmax(h_norm / T)  ← main method
    #   easy      — weights ∝ (1 − h_norm)^alpha   (focus on easiest, ablation)
    mode: str = "soft_hard"

    # softmax temperature for soft_hard mode (lower = more concentrated)
    temperature: float = 0.1

    # power exponent for hard / easy modes
    alpha: float = 1.0

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    seed: int = 42
    device: str = "auto"
    epochs: int = 3000
    batch_size: int = 64
    lr: float = 1e-4
    val_ratio: float = 0.1

    # ------------------------------------------------------------------ #
    # Stage-2b scoring                                                     #
    # ------------------------------------------------------------------ #
    score_batch_size: int = 512   # batch size for forward-pass during scoring
