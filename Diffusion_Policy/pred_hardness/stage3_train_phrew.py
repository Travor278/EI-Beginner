"""
Stage 3 — PHRew-DP Training Script.

Trains a standard Diffusion Policy image policy with Predictive Hardness
Reweighting (PHRew).  The model and loss are UNCHANGED from the official DP;
the only modification is the DataLoader's sampler.

Weighting modes (--mode):
  uniform   — WeightedRandomSampler with all weights = 1 (reproduces standard DP)
  hard      — weights ∝ hardness^alpha   (max focus on hardest chunks)
  soft_hard — weights ∝ softmax(h/T)     ← main PHRew-DP method
  easy      — weights ∝ (1−h)^alpha      (easy-first ablation, expected to underperform)

Integration strategy (same as dlos_dp/train_dlos.py):
  - Appends official DP repo to sys.path at runtime (no repo modification)
  - Uses MultiImageObsEncoder + ConditionalUnet1D + DDPMScheduler from official repo
  - Trains with standard MSE diffusion loss

Usage (WSL with GPU)
--------------------
    python stage3_train_phrew.py \\
        --mode soft_hard --seed 42 \\
        --chunk-hardness artifacts/phrew/chunk_hardness_norm.npy \\
        --zarr-path /home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr \\
        --dp-repo   /home/Travor/workspaces/diffusion_policy \\
        --outdir    artifacts/phrew_runs/soft_hard_seed42
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
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Package path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pred_hardness.config import PHRConfig
from pred_hardness.dataset import PushTImagePHRDataset


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
# Training loop                                                        #
# ------------------------------------------------------------------ #

def train(cfg: PHRConfig, args: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load official DP modules
    # ----------------------------------------------------------------
    if str(cfg.dp_repo_path) not in sys.path:
        sys.path.insert(0, str(cfg.dp_repo_path))

    try:
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
        from diffusion_policy.common.normalizer import LinearNormalizer
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except ImportError as e:
        raise ImportError(
            f"Cannot import official DP modules from '{cfg.dp_repo_path}'. "
            f"Make sure --dp-repo points to the cloned diffusion_policy repo.\n{e}"
        )

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------
    print("Building datasets …")

    # Load hardness scores if provided
    hardness_scores: np.ndarray | None = None
    if args.chunk_hardness and Path(args.chunk_hardness).exists():
        hardness_scores = np.load(str(args.chunk_hardness)).astype(np.float32)
        print(f"  loaded hardness scores: shape={hardness_scores.shape}")
    elif args.mode != "uniform":
        raise FileNotFoundError(
            f"--chunk-hardness file not found: {args.chunk_hardness}\n"
            "Run stage2b_score_chunks.py first."
        )

    train_dataset = PushTImagePHRDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="train",
        seed=cfg.seed,
        hardness_scores=hardness_scores,
    )
    val_dataset = PushTImagePHRDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="val",
        seed=cfg.seed,
        hardness_scores=None,   # val always uses uniform (unbiased evaluation)
    )
    print(f"  train chunks={len(train_dataset)}, val chunks={len(val_dataset)}")

    # ----------------------------------------------------------------
    # Weighted sampler
    # ----------------------------------------------------------------
    weights = train_dataset.get_weights(
        mode=args.mode,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
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
    # Build DP model (same architecture as official push-T image policy)
    # ----------------------------------------------------------------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

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

    # Combine into a simple module for clean forward / parameter grouping
    class DPModel(nn.Module):
        def __init__(self, obs_encoder, noise_pred_net, noise_scheduler):
            super().__init__()
            self.obs_encoder    = obs_encoder
            self.noise_pred_net = noise_pred_net
            self.noise_scheduler = noise_scheduler
            self.normalizer: LinearNormalizer | None = None

        def compute_loss(self, batch: dict, device: torch.device) -> torch.Tensor:
            nobs    = self.normalizer["obs"].normalize(batch["obs"].to(device))
            naction = self.normalizer["action"].normalize(batch["action"].to(device))
            B = naction.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=device, dtype=torch.long,
            )
            noise = torch.randn_like(naction)
            noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
            obs_feat   = self.obs_encoder(nobs)
            noise_pred = self.noise_pred_net(
                noisy_action, timesteps, global_cond=obs_feat
            )
            return F.mse_loss(noise_pred, noise)

    model = DPModel(obs_encoder, noise_pred_net, noise_scheduler).to(device)

    # Fit normalizer on training set
    print("Fitting normalizer …")
    all_actions = np.stack(
        [train_dataset[i]["action"].numpy() for i in range(len(train_dataset))]
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.from_numpy(all_actions)})
    model.normalizer = normalizer

    # ----------------------------------------------------------------
    # Optimiser + LR schedule
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=1e-6,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # ----------------------------------------------------------------
    # Optional resume from checkpoint
    # ----------------------------------------------------------------
    start_epoch = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")
    best_path = outdir / "best.pt"

    print(f"Training for {cfg.epochs} epochs (mode={args.mode}, seed={cfg.seed}) …")
    for epoch in range(start_epoch, cfg.epochs):
        # -- train --
        model.train()
        train_sum, train_cnt = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            bs = batch["action"].shape[0]
            train_sum += float(loss.item()) * bs
            train_cnt += bs
        lr_scheduler.step()
        train_loss = train_sum / max(train_cnt, 1)

        # -- val --
        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss = model.compute_loss(batch, device)
                bs = batch["action"].shape[0]
                val_sum += float(loss.item()) * bs
                val_cnt += bs
        val_loss = val_sum / max(val_cnt, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:04d}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":         val_loss,
                    "mode":             args.mode,
                    "seed":             cfg.seed,
                },
                best_path,
            )

        # Periodic checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":         val_loss,
                },
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

    # ----------------------------------------------------------------
    # Save history + config snapshot
    # ----------------------------------------------------------------
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    config_snap = {
        "mode":            args.mode,
        "temperature":     args.temperature,
        "alpha":           args.alpha,
        "seed":            cfg.seed,
        "epochs":          cfg.epochs,
        "batch_size":      cfg.batch_size,
        "lr":              cfg.lr,
        "best_val_loss":   best_val_loss,
        "chunk_hardness":  str(args.chunk_hardness),
        "zarr_path":       str(cfg.zarr_path),
    }
    (outdir / "config.json").write_text(
        json.dumps(config_snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nbest_val_loss = {best_val_loss:.6f}")
    print(f"saved best checkpoint → {best_path}")
    print(f"saved history         → {outdir / 'training_history.csv'}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PHRew-DP Stage 3: train DP with predictive hardness reweighting."
    )
    parser.add_argument("--mode", type=str, default="soft_hard",
                        choices=["uniform", "hard", "soft_hard", "easy"],
                        help="Sampling weighting mode.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Softmax temperature for soft_hard mode.")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Power exponent for hard/easy modes.")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--epochs",      type=int,   default=3000)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument(
        "--chunk-hardness",
        type=str,
        default="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
                "artifacts/phrew/chunk_hardness_norm.npy",
        help="Path to chunk_hardness_norm.npy from Stage 2b.",
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr",
    )
    parser.add_argument(
        "--dp-repo",
        type=str,
        default="/home/Travor/workspaces/diffusion_policy",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="artifacts/phrew_runs/soft_hard_seed42",
    )
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Resume from checkpoint.")
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    args = parser.parse_args()

    cfg = PHRConfig(
        zarr_path=args.zarr_path,
        dp_repo_path=args.dp_repo,
        mode=args.mode,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_ratio=args.val_ratio,
        outdir=args.outdir,
    )

    train(cfg, args)


if __name__ == "__main__":
    main()
