"""
DLOS-DP Stage 1 Training Script.

Trains a Diffusion Policy augmented with Denoising-Level Outcome Supervision
(DLOS).  Five ablation groups (A–E) are supported via the ``--group`` flag.

Integration strategy
--------------------
The official Diffusion Policy repository is NOT modified.  Instead, we:
  1. Append the DP repo to sys.path at runtime.
  2. Subclass DiffusionUnetHybridImagePolicy, overriding compute_loss().
  3. Replace the official dataset with PushTImageDLOSDataset (adds obs_next).
  4. Load the official hydra config for Push-T image policy and patch only the
     policy class and dataset.

Usage (from WSL, with GPU)
--------------------------
    cd /path/to/EI-Beginner/Diffusion_Policy/dlos_dp
    python train_dlos.py \\
        --group D --seed 42 \\
        --zarr-path /home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr \\
        --dp-repo   /home/Travor/workspaces/diffusion_policy \\
        --outdir    artifacts/dlos_runs/group_D_seed42

Outputs
-------
    <outdir>/checkpoints/     — epoch checkpoints (every 100 epochs)
    <outdir>/best.pt          — best validation-loss checkpoint
    <outdir>/training_history.csv
    <outdir>/config.json      — full config snapshot
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Resolve package imports whether run as a script or as a module
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dlos_dp.config import DLOSConfig
from dlos_dp.dataset import PushTImageDLOSDataset
from dlos_dp.dino_encoder import FrozenDINOv2Encoder
from dlos_dp.dlos_loss import DLOSLoss
from dlos_dp.world_model import OutcomeWorldModel


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ------------------------------------------------------------------ #
# DLOS-augmented policy (subclasses official DP)                      #
# ------------------------------------------------------------------ #

def build_dlos_policy(
    cfg: DLOSConfig,
    device: torch.device,
    dp_policy_cls,          # DiffusionUnetHybridImagePolicy class, loaded at runtime
    dp_policy_kwargs: dict, # kwargs to pass to the base policy constructor
) -> nn.Module:
    """
    Construct a DLOSPolicy by dynamically subclassing the official DP policy.
    """
    dino = FrozenDINOv2Encoder(
        model_name=cfg.dino_model,
        img_size=cfg.dino_img_size,
    ).to(device)

    wm = OutcomeWorldModel(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.wm_hidden,
    ).to(device)

    dlos_loss_fn = DLOSLoss(wm=wm)

    # ----------------------------------------------------------------
    # Dynamically define DLOSPolicy to avoid a hard import of the
    # official DP at module load time.
    # ----------------------------------------------------------------
    class DLOSPolicy(dp_policy_cls):
        """
        DiffusionUnetHybridImagePolicy with DLOS loss injected into
        compute_loss().
        """

        def __init__(self, *args, _dlos_cfg, _dino, _wm, _dlos_loss_fn, **kwargs):
            super().__init__(*args, **kwargs)
            self.dlos_cfg     = _dlos_cfg
            self.dino         = _dino
            self.wm           = _wm
            self.dlos_loss_fn = _dlos_loss_fn

        def compute_loss(self, batch: dict) -> torch.Tensor:
            # --------------------------------------------------------
            # Standard Diffusion Policy forward pass
            # --------------------------------------------------------
            # nobs: (B, obs_horizon, C, H, W) — DP-normalised observation
            nobs    = self.normalizer["obs"].normalize(batch["obs"])
            # naction: (B, pred_horizon, 2) — DP-normalised action
            naction = self.normalizer["action"].normalize(batch["action"])
            obs_next = batch["obs_next"]    # (B, 3, H, W) — raw float [0,1]

            B = naction.shape[0]
            device = naction.device

            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(naction)
            noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)

            # Encode observations with official obs encoder
            obs_feat   = self.obs_encoder(nobs)
            noise_pred = self.noise_pred_net(
                noisy_action, timesteps, global_cond=obs_feat
            )
            loss_diff = F.mse_loss(noise_pred, noise)

            if self.dlos_cfg.group == "A":
                return loss_diff

            # --------------------------------------------------------
            # DLOS world-model loss
            # --------------------------------------------------------
            # Use the raw (non-DP-normalised) last frame for DINO.
            # batch['obs'] is float [0,1], shape (B, obs_horizon, 3, H, W).
            obs_t  = batch["obs"][:, -1]           # (B, 3, H, W)
            z_obs  = self.dino(obs_t.to(device))   # (B, 384)  — no_grad inside dino
            z_next = self.dino(obs_next.to(device)) # (B, 384)

            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)

            loss_wm = self.dlos_loss_fn(
                noise_pred=noise_pred,
                noisy_action=noisy_action,
                timesteps=timesteps,
                gt_action=naction,
                z_obs=z_obs,
                z_next=z_next,
                alphas_cumprod=alphas_cumprod,
                group=self.dlos_cfg.group,
            )

            return loss_diff + self.dlos_cfg.lambda_wm * loss_wm

    # Instantiate and return
    policy = DLOSPolicy(
        **dp_policy_kwargs,
        _dlos_cfg=cfg,
        _dino=dino,
        _wm=wm,
        _dlos_loss_fn=dlos_loss_fn,
    ).to(device)
    return policy


# ------------------------------------------------------------------ #
# Training loop                                                        #
# ------------------------------------------------------------------ #

def train(cfg: DLOSConfig, args: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load official DP policy class and default kwargs
    # ----------------------------------------------------------------
    if str(cfg.dp_repo_path) not in sys.path:
        sys.path.insert(0, str(cfg.dp_repo_path))

    try:
        from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
            DiffusionUnetHybridImagePolicy,
        )
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
        from diffusion_policy.common.normalizer import LinearNormalizer
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError as e:
        raise ImportError(
            f"Failed to import official Diffusion Policy modules from "
            f"'{cfg.dp_repo_path}'.  Make sure dp_repo_path points to the "
            f"cloned diffusion_policy repository.\nOriginal error: {e}"
        )

    # ----------------------------------------------------------------
    # Datasets & DataLoaders
    # ----------------------------------------------------------------
    print("Building datasets …")
    train_dataset = PushTImageDLOSDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="train",
        seed=cfg.seed,
    )
    val_dataset = PushTImageDLOSDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="val",
        seed=cfg.seed,
    )
    print(f"  train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ----------------------------------------------------------------
    # Build the DLOS policy
    # ----------------------------------------------------------------
    # Push-T image policy hyper-parameters (mirror official config)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # obs_encoder: MultiImageObsEncoder with ResNet18 (official default)
    # We construct a minimal kwargs dict compatible with the official policy.
    # For a full hydra-based initialisation, pass --dp-cfg-override.
    obs_encoder = MultiImageObsEncoder(
        shape_meta={
            "obs": {
                "image": {
                    "shape": [3, 96, 96],
                    "type": "rgb",
                }
            }
        },
        rgb_model_name="resnet18",
        resize_shape=None,
        crop_shape=[76, 76],
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
    )

    obs_feature_dim = obs_encoder.output_shape()[0]

    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=obs_feature_dim * cfg.obs_horizon,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )

    dp_policy_kwargs = dict(
        shape_meta={
            "obs": {
                "image": {
                    "shape": [3, 96, 96],
                    "type": "rgb",
                }
            },
            "action": {
                "shape": [cfg.action_dim],
            },
        },
        noise_scheduler=noise_scheduler,
        obs_encoder=obs_encoder,
        noise_pred_net=noise_pred_net,
        horizon=cfg.pred_horizon,
        n_action_steps=cfg.action_horizon,
        n_obs_steps=cfg.obs_horizon,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=[76, 76],
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    )

    print(f"Building DLOSPolicy (group={cfg.group}) …")
    policy = build_dlos_policy(cfg, device, DiffusionUnetHybridImagePolicy, dp_policy_kwargs)

    # Fit normalizer on training data statistics
    print("Fitting normalizer …")
    all_actions = np.stack(
        [train_dataset[i]["action"].numpy() for i in range(len(train_dataset))]
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.from_numpy(all_actions)})
    policy.set_normalizer(normalizer)

    # ----------------------------------------------------------------
    # Optimisers
    # ----------------------------------------------------------------
    # Separate learning rates: WM trains faster; DINO is frozen.
    dp_params  = [p for n, p in policy.named_parameters()
                  if not n.startswith("dino.") and not n.startswith("wm.")]
    wm_params  = list(policy.wm.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": dp_params, "lr": cfg.lr},
            {"params": wm_params, "lr": cfg.lr * 10},  # WM trains faster
        ],
        weight_decay=1e-6,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # ----------------------------------------------------------------
    # Optionally resume from checkpoint
    # ----------------------------------------------------------------
    start_epoch = 0
    if cfg.checkpoint_path:
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from checkpoint at epoch {start_epoch - 1}")

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")
    best_ckpt_path = outdir / "best.pt"

    print(f"Starting training for {cfg.epochs} epochs …")
    for epoch in range(start_epoch, cfg.epochs):
        # -- train --
        policy.train()
        train_loss_sum, train_count = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = policy.compute_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            bs = batch["action"].shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        lr_scheduler.step()
        train_loss = train_loss_sum / max(train_count, 1)

        # -- val --
        policy.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = policy.compute_loss(batch)
                bs = batch["action"].shape[0]
                val_loss_sum += float(loss.item()) * bs
                val_count += bs
        val_loss = val_loss_sum / max(val_count, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch={epoch:04d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
        )

        # -- best checkpoint --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "group": cfg.group,
                    "seed": cfg.seed,
                },
                best_ckpt_path,
            )

        # -- periodic checkpoint --
        if (epoch + 1) % 100 == 0:
            periodic_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                periodic_path,
            )

    # ----------------------------------------------------------------
    # Save history + config
    # ----------------------------------------------------------------
    history_path = outdir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    config_snapshot = {
        k: v for k, v in vars(cfg).items()
    }
    config_snapshot["best_val_loss"] = best_val_loss
    (outdir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nbest_val_loss={best_val_loss:.6f}")
    print(f"saved best checkpoint → {best_ckpt_path}")
    print(f"saved history         → {history_path}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DLOS-DP Stage 1 training (groups A–E)."
    )
    parser.add_argument("--group",    type=str, default="D",
                        choices=["A", "B", "C", "D", "E"],
                        help="Ablation group (default: D = full DLOS).")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--epochs",   type=int,   default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--lambda-wm", type=float, default=0.1,
                        help="Global weight on the WM loss term.")
    parser.add_argument("--device",   type=str,   default="auto")
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr",
    )
    parser.add_argument(
        "--dp-repo",
        type=str,
        default="/home/Travor/workspaces/diffusion_policy",
        help="Path to the cloned diffusion_policy repository.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="artifacts/dlos_runs/group_D_seed42",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    args = parser.parse_args()

    cfg = DLOSConfig(
        group=args.group,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_wm=args.lambda_wm,
        device=args.device,
        zarr_path=args.zarr_path,
        dp_repo_path=args.dp_repo,
        checkpoint_path=args.checkpoint,
        val_ratio=args.val_ratio,
        outdir=args.outdir,
    )

    train(cfg, args)


if __name__ == "__main__":
    main()
