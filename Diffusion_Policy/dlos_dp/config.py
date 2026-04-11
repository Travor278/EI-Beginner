"""DLOSConfig: single dataclass holding all DLOS-DP hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DLOSConfig:
    # ------------------------------------------------------------------ #
    # World-model / encoder dimensions                                     #
    # ------------------------------------------------------------------ #
    obs_dim: int = 384      # DINOv2-S/14 CLS token dimension
    action_dim: int = 2     # Push-T action: (x, y) end-effector position
    wm_hidden: int = 256    # OutcomeWorldModel hidden layer width

    # ------------------------------------------------------------------ #
    # Loss weighting                                                       #
    # ------------------------------------------------------------------ #
    lambda_wm: float = 0.1  # global scale on L_wm before adding to L_diff

    # ------------------------------------------------------------------ #
    # DINOv2 settings                                                      #
    # ------------------------------------------------------------------ #
    dino_model: str = "dinov2_vits14"   # torch.hub model name
    dino_img_size: int = 224            # input resize (ViT-S/14 native)

    # ------------------------------------------------------------------ #
    # Push-T task settings (must match the zarr data)                      #
    # ------------------------------------------------------------------ #
    obs_horizon: int = 2        # number of observation frames per sample
    pred_horizon: int = 16      # action chunk length
    action_horizon: int = 8     # executed action steps per inference step
    num_train_timesteps: int = 100  # DDPM diffusion steps

    # ------------------------------------------------------------------ #
    # Ablation group                                                       #
    # ------------------------------------------------------------------ #
    # A — no WM (baseline)
    # B — WM with ground-truth action (no gradient to ε_θ)
    # C — WM with x̂₀|k, action stop-gradient (no gradient to ε_θ)
    # D — DLOS full: WM with x̂₀|k, gradient flows through to ε_θ   ← main
    # E — same as D but x̂₀|k stop-gradient (tests whether grad path matters)
    group: str = "D"

    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #
    dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy"
    zarr_path: str = (
        "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    )
    # Optional: resume from a pre-trained DP checkpoint before DLOS fine-tune
    checkpoint_path: str = ""

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    seed: int = 42
    device: str = "auto"        # "auto" | "cuda" | "cpu"
    epochs: int = 3000          # training epochs (same as official DP)
    batch_size: int = 64
    lr: float = 1e-4
    val_ratio: float = 0.1
    outdir: str = "artifacts/dlos_runs"

    # ------------------------------------------------------------------ #
    # Stage-0 WM probe                                                     #
    # ------------------------------------------------------------------ #
    wm_probe_epochs: int = 50
    wm_probe_lr: float = 1e-4
    dino_embed_cache: str = "artifacts/dino_embeddings.npy"
    wm_probe_outdir: str = "artifacts/wm_probe"
