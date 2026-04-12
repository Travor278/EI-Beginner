"""
SFC-DP Stage 1 — Training Script.

Trains a standard Diffusion Policy (Push-T image variant) with the
SNR-Frequency Consistent auxiliary loss (SFC-DP, v3.6).

Ablation groups (--group):
  A — baseline: standard DP, no L_freq
  B — L_freq with full-band mask (no SNR alignment)
  C — L_freq + SNR mask in ε-space
  D — SFC-DP full (main method) ← default
  E — same as D but stop-grad on x̂₀|k

Integration strategy (same as dlos_dp/train_dlos.py):
  - Appends official DP repo to sys.path at runtime (no repo modification)
  - Uses MultiImageObsEncoder + ConditionalUnet1D + DDPMScheduler from official repo
  - Adds L_freq on top of standard MSE diffusion loss

Usage (WSL with GPU)
--------------------
    python train_sfc.py \\
        --group D --seed 42 \\
        --zarr-path /home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr \\
        --dp-repo   /home/Travor/workspaces/diffusion_policy \\
        --outdir    artifacts/sfc_runs/D_seed42
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
# Package path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfc_dp.config import SFCConfig
from sfc_dp.sfc_loss import SFCLoss

# ---------------------------------------------------------------------------
# Dataset — reuse the image dataset from pred_hardness (no hardness needed)
# ---------------------------------------------------------------------------
try:
    from pred_hardness.dataset import PushTImagePHRDataset as _BaseDataset

    class PushTImageSFCDataset(_BaseDataset):
        """Thin alias; hardness_scores=None gives uniform sampling."""
        pass

except ImportError:
    # Fallback: define minimal dataset inline if pred_hardness not on path
    import zarr
    import torch
    from torch.utils.data import Dataset

    class PushTImageSFCDataset(Dataset):  # type: ignore[no-redef]
        """Minimal Push-T image dataset for SFC-DP training."""

        def __init__(
            self,
            zarr_path: str,
            obs_horizon: int = 2,
            pred_horizon: int = 16,
            val_ratio: float = 0.1,
            split: str = "train",
            seed: int = 42,
            **kwargs,
        ) -> None:
            import zarr as _zarr
            root = _zarr.open(zarr_path, mode="r")
            self.images  = root["data"]["img"][:]          # (N, H, W, C) uint8
            self.actions = root["data"]["action"][:]       # (N, 2) float32
            ep_ends = root["meta"]["episode_ends"][:]      # 1-indexed

            # Build (obs_start, action_start) index
            rng = np.random.default_rng(seed)
            all_starts: list[tuple[int, int]] = []
            prev = 0
            for end in ep_ends:
                max_start = int(end) - pred_horizon
                for s in range(prev + obs_horizon - 1, max_start + 1):
                    all_starts.append((s - obs_horizon + 1, s))
                prev = int(end)

            n = len(all_starts)
            idx = np.arange(n)
            rng.shuffle(idx)
            split_i = max(1, int(n * val_ratio))
            val_idx  = idx[:split_i]
            train_idx = idx[split_i:]

            chosen = train_idx if split == "train" else val_idx
            self._starts = [all_starts[i] for i in chosen]
            self.obs_horizon  = obs_horizon
            self.pred_horizon = pred_horizon

        def __len__(self) -> int:
            return len(self._starts)

        def __getitem__(self, idx: int) -> dict:
            obs_s, act_s = self._starts[idx]
            imgs = torch.from_numpy(
                self.images[obs_s : obs_s + self.obs_horizon]
                .astype(np.float32)
                .transpose(0, 3, 1, 2) / 255.0
            )
            action = torch.from_numpy(
                self.actions[act_s : act_s + self.pred_horizon].astype(np.float32)
            )
            return {"obs": imgs, "action": action}

        def get_weights(self, mode="uniform", **kwargs):
            return torch.ones(len(self))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: SFCConfig, args: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    outdir  = Path(args.outdir)
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
    train_dataset = PushTImageSFCDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="train",
        seed=cfg.seed,
        hardness_scores=None,   # SFC-DP uses uniform sampling
    )
    val_dataset = PushTImageSFCDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="val",
        seed=cfg.seed,
        hardness_scores=None,
    )
    print(f"  train chunks={len(train_dataset)}, val chunks={len(val_dataset)}")

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

    # SFC auxiliary loss (zero extra parameters)
    sfc_criterion = SFCLoss(
        noise_scheduler=noise_scheduler,
        group=cfg.group,
        lambda_freq=cfg.lambda_freq,
        soft_mask=cfg.soft_mask,
        soft_mask_tau=cfg.soft_mask_tau,
    )

    class DPModel(nn.Module):
        def __init__(self, obs_encoder, noise_pred_net, noise_scheduler, sfc_criterion):
            super().__init__()
            self.obs_encoder     = obs_encoder
            self.noise_pred_net  = noise_pred_net
            self.noise_scheduler = noise_scheduler
            self.sfc_criterion   = sfc_criterion
            self.normalizer: LinearNormalizer | None = None

        def compute_loss(
            self,
            batch: dict,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns (total_loss, loss_diff, loss_freq).
            loss_freq is 0 for group A.
            """
            nobs    = self.normalizer["obs"].normalize(batch["obs"].to(device))
            naction = self.normalizer["action"].normalize(batch["action"].to(device))
            B = naction.shape[0]

            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=device, dtype=torch.long,
            )
            noise        = torch.randn_like(naction)
            noisy_action = self.noise_scheduler.add_noise(naction, noise, timesteps)
            obs_feat     = self.obs_encoder(nobs)
            noise_pred   = self.noise_pred_net(
                noisy_action, timesteps, global_cond=obs_feat
            )

            loss_diff = F.mse_loss(noise_pred, noise)
            loss_freq = self.sfc_criterion(
                noise_pred=noise_pred,
                noisy_action=noisy_action,
                gt_action=naction,
                timesteps=timesteps,
            )
            return loss_diff + loss_freq, loss_diff, loss_freq

    model = DPModel(obs_encoder, noise_pred_net, noise_scheduler, sfc_criterion).to(device)
    # Move SFC buffers (alphas_cumprod) to device as well
    model.sfc_criterion = model.sfc_criterion.to(device)

    # Fit normalizer on training set
    print("Fitting normalizer …")
    all_actions = np.stack(
        [train_dataset[i]["action"].numpy() for i in range(len(train_dataset))]
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"action": torch.from_numpy(all_actions)})
    # Placeholder obs normalizer (images are already in [0,1])
    sample_obs = train_dataset[0]["obs"]
    normalizer.fit({"obs": sample_obs.unsqueeze(0), "action": torch.from_numpy(all_actions)})
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
    if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
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

    print(
        f"Training for {cfg.epochs} epochs  "
        f"(group={cfg.group}, λ_freq={cfg.lambda_freq}, seed={cfg.seed}) …"
    )
    for epoch in range(start_epoch, cfg.epochs):
        # -- train --
        model.train()
        tr_sum, tr_diff_sum, tr_freq_sum, tr_cnt = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, l_diff, l_freq = model.compute_loss(batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            bs = batch["action"].shape[0]
            tr_sum      += float(loss.item())   * bs
            tr_diff_sum += float(l_diff.item()) * bs
            tr_freq_sum += float(l_freq.item()) * bs
            tr_cnt += bs
        lr_scheduler.step()
        train_loss      = tr_sum      / max(tr_cnt, 1)
        train_loss_diff = tr_diff_sum / max(tr_cnt, 1)
        train_loss_freq = tr_freq_sum / max(tr_cnt, 1)

        # -- val (diffusion loss only, unbiased) --
        model.eval()
        val_sum, val_cnt = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                _, l_diff, _ = model.compute_loss(batch, device)
                bs = batch["action"].shape[0]
                val_sum += float(l_diff.item()) * bs
                val_cnt += bs
        val_loss = val_sum / max(val_cnt, 1)

        row = {
            "epoch":           epoch,
            "train_loss":      train_loss,
            "train_loss_diff": train_loss_diff,
            "train_loss_freq": train_loss_freq,
            "val_loss":        val_loss,
        }
        history.append(row)
        print(
            f"epoch={epoch:04d}  "
            f"train={train_loss:.6f}  "
            f"diff={train_loss_diff:.6f}  "
            f"freq={train_loss_freq:.6f}  "
            f"val={val_loss:.6f}"
        )

        # Best checkpoint (based on val diffusion loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            val_loss,
                    "group":               cfg.group,
                    "seed":                cfg.seed,
                },
                best_path,
            )

        # Periodic checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            val_loss,
                },
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

    # ----------------------------------------------------------------
    # Save history + config snapshot
    # ----------------------------------------------------------------
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    config_snap = {
        "group":         cfg.group,
        "lambda_freq":   cfg.lambda_freq,
        "soft_mask":     cfg.soft_mask,
        "soft_mask_tau": cfg.soft_mask_tau,
        "seed":          cfg.seed,
        "epochs":        cfg.epochs,
        "batch_size":    cfg.batch_size,
        "lr":            cfg.lr,
        "best_val_loss": best_val_loss,
        "zarr_path":     str(cfg.zarr_path),
    }
    (outdir / "config.json").write_text(
        json.dumps(config_snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nbest_val_loss = {best_val_loss:.6f}")
    print(f"saved best checkpoint → {best_path}")
    print(f"saved history         → {outdir / 'training_history.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFC-DP Stage 1: train DP with SNR-frequency consistent auxiliary loss."
    )
    parser.add_argument(
        "--group", type=str, default="D",
        choices=["A", "B", "C", "D", "E"],
        help="Ablation group (A=baseline, D=full SFC-DP).",
    )
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--epochs",        type=int,   default=3000)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--device",        type=str,   default="auto")
    parser.add_argument("--lambda-freq",   type=float, default=0.1,
                        help="Weight on L_freq (ignored for group A).")
    parser.add_argument("--soft-mask",     action="store_true",
                        help="Use soft sigmoid mask instead of hard step.")
    parser.add_argument("--soft-mask-tau", type=float, default=0.05,
                        help="Temperature for soft mask sigmoid.")
    parser.add_argument(
        "--zarr-path", type=str,
        default="/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr",
    )
    parser.add_argument(
        "--dp-repo", type=str,
        default="/home/Travor/workspaces/diffusion_policy",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="artifacts/sfc_runs/D_seed42",
    )
    parser.add_argument("--checkpoint",  type=str, default="",
                        help="Resume from checkpoint path.")
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    args = parser.parse_args()

    cfg = SFCConfig(
        zarr_path=args.zarr_path,
        dp_repo_path=args.dp_repo,
        checkpoint_path=args.checkpoint,
        outdir=args.outdir,
        group=args.group,
        lambda_freq=args.lambda_freq,
        soft_mask=args.soft_mask,
        soft_mask_tau=args.soft_mask_tau,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
    )

    train(cfg, args)


if __name__ == "__main__":
    main()
