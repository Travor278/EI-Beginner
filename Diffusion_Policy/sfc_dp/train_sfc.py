"""Stage 1 training entrypoint for SFC-DP."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pred_hardness.dataset import PushTImagePHRDataset as PushTImageSFCDataset
from sfc_dp.config import SFCConfig
from sfc_dp.sfc_loss import SFCLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def json_safe_metrics(metrics: dict) -> dict:
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            safe[key] = float(value)
    return safe


def lambda_with_warmup(
    epoch: int,
    lambda_freq: float,
    warmup_epochs: int,
) -> float:
    """Linearly ramp lambda from 0 → lambda_freq over warmup_epochs.

    Rationale: at epoch 0 the untrained network produces L_freq ≈ 87,
    while L_diff ≈ 0.22.  Even λ=0.001 gives λ·L_freq ≈ 0.087 at epoch 0,
    which is still ~40% of L_diff.  A short linear warmup (default 20 epochs)
    ensures the freq term starts from zero and grows gradually, preventing
    early-training collapse without sacrificing final signal strength.
    """
    if warmup_epochs <= 0:
        return lambda_freq
    return lambda_freq * min(1.0, epoch / warmup_epochs)


def build_policy_class(base_policy_cls):
    class SFCPolicy(base_policy_cls):
        def __init__(self, *args, _sfc_cfg: SFCConfig, **kwargs):
            super().__init__(*args, **kwargs)
            self.sfc_cfg = _sfc_cfg
            self.sfc_loss_fn = SFCLoss(
                noise_scheduler=self.noise_scheduler,
                group=_sfc_cfg.group,
                lambda_freq=_sfc_cfg.lambda_freq,
                soft_mask=_sfc_cfg.soft_mask,
                soft_mask_tau=_sfc_cfg.soft_mask_tau,
            )

        def compute_loss_components(
            self, batch: dict
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert "valid_mask" not in batch

            nobs = self.normalizer.normalize(batch["obs"])
            nactions = self.normalizer["action"].normalize(batch["action"])
            batch_size = nactions.shape[0]
            horizon = nactions.shape[1]

            local_cond = None
            global_cond = None
            trajectory = nactions
            cond_data = trajectory

            if self.obs_as_global_cond:
                from diffusion_policy.common.pytorch_util import dict_apply

                this_nobs = dict_apply(
                    nobs,
                    lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
                )
                nobs_features = self.obs_encoder(this_nobs)
                global_cond = nobs_features.reshape(batch_size, -1)
            else:
                from diffusion_policy.common.pytorch_util import dict_apply

                this_nobs = dict_apply(
                    nobs,
                    lambda x: x.reshape(-1, *x.shape[2:]),
                )
                nobs_features = self.obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
                trajectory = cond_data.detach()

            condition_mask = self.mask_generator(trajectory.shape)
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=trajectory.device,
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timesteps
            )

            loss_mask = ~condition_mask
            noisy_trajectory[condition_mask] = cond_data[condition_mask]

            pred = self.model(
                noisy_trajectory,
                timesteps,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            pred_type = self.noise_scheduler.config.prediction_type
            if pred_type == "epsilon":
                target = noise
            elif pred_type == "sample":
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")

            loss_diff = F.mse_loss(pred, target, reduction="none")
            loss_diff = loss_diff * loss_mask.type(loss_diff.dtype)
            loss_diff = loss_diff.reshape(batch_size, -1).mean(dim=1).mean()

            loss_freq = self.sfc_loss_fn(
                noise_pred=pred,
                noisy_action=noisy_trajectory,
                gt_action=nactions,
                timesteps=timesteps,
            )
            return loss_diff + loss_freq, loss_diff, loss_freq

        def compute_loss(self, batch: dict) -> torch.Tensor:
            loss, _, _ = self.compute_loss_components(batch)
            return loss

    return SFCPolicy


def train(cfg: SFCConfig, args: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    outdir = Path(args.outdir).resolve()
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if str(cfg.dp_repo_path) not in sys.path:
        sys.path.insert(0, str(cfg.dp_repo_path))

    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    print("Building datasets …")
    train_dataset = PushTImageSFCDataset(
        zarr_path=cfg.zarr_path,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_ratio=cfg.val_ratio,
        split="train",
        seed=cfg.seed,
        hardness_scores=None,
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
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    shape_meta = {
        "action": {"shape": [cfg.action_dim]},
        "obs": {
            "image": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
    }

    SFCPolicy = build_policy_class(DiffusionUnetHybridImagePolicy)
    policy = SFCPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=cfg.pred_horizon,
        n_action_steps=cfg.action_horizon,
        n_obs_steps=cfg.obs_horizon,
        num_inference_steps=cfg.num_inference_steps,
        obs_as_global_cond=True,
        crop_shape=cfg.crop_shape,
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        down_dims=cfg.down_dims,
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        cond_predict_scale=cfg.cond_predict_scale,
        obs_encoder_group_norm=cfg.obs_encoder_group_norm,
        eval_fixed_crop=cfg.eval_fixed_crop,
        _sfc_cfg=cfg,
    )

    print("Fitting normalizer …")
    normalizer = LinearNormalizer()
    normalizer.fit(
        data={
            "action": train_dataset.actions,
            "agent_pos": train_dataset.states[:, :2],
        },
        last_n_dims=1,
        mode="limits",
    )
    normalizer["image"] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        weight_decay=1e-6,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    env_runner = PushTImageRunner(
        output_dir=str(outdir),
        n_train=args.n_train,
        n_train_vis=args.n_train_vis,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        n_envs=args.n_envs,
        n_obs_steps=cfg.obs_horizon,
        n_action_steps=cfg.action_horizon,
        max_steps=args.max_steps,
        legacy_test=True,
        tqdm_interval_sec=1.0,
    )

    start_epoch = 0
    best_metric = -float("inf")
    best_val_loss = float("inf")
    best_path = outdir / "best.pt"
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from epoch {start_epoch - 1}")

    history: list[dict] = []
    log_path = outdir / "logs.json.txt"
    print(
        f"Training SFC-DP for {cfg.epochs} epochs "
        f"(group={cfg.group}, seed={cfg.seed}, device={device}) …"
    )
    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(start_epoch, cfg.epochs):
            policy.train()
            # λ warmup: ramp from 0 → lambda_freq over lambda_warmup_epochs
            policy.sfc_loss_fn.lambda_freq = lambda_with_warmup(
                epoch, cfg.lambda_freq, cfg.lambda_warmup_epochs
            )
            train_sum, diff_sum, freq_sum, train_cnt = 0.0, 0.0, 0.0, 0
            for batch_idx, batch in enumerate(train_loader):
                if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                    break
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                loss, loss_diff, loss_freq = policy.compute_loss_components(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()

                bs = batch["action"].shape[0]
                train_sum += float(loss.item()) * bs
                diff_sum += float(loss_diff.item()) * bs
                freq_sum += float(loss_freq.item()) * bs
                train_cnt += bs
            lr_scheduler.step()
            train_loss = train_sum / max(train_cnt, 1)
            train_loss_diff = diff_sum / max(train_cnt, 1)
            train_loss_freq = freq_sum / max(train_cnt, 1)

            policy.eval()
            val_sum, val_cnt = 0.0, 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                        break
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    _, loss_diff, _ = policy.compute_loss_components(batch)
                    bs = batch["action"].shape[0]
                    val_sum += float(loss_diff.item()) * bs
                    val_cnt += bs
            val_loss = val_sum / max(val_cnt, 1)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_loss_diff": train_loss_diff,
                "train_loss_freq": train_loss_freq,
                "val_loss": val_loss,
                "lr": lr_scheduler.get_last_lr()[0],
                "group": cfg.group,
                "seed": cfg.seed,
            }
            if (epoch % args.eval_every) == 0:
                rollout_log = json_safe_metrics(env_runner.run(policy))
                row.update(rollout_log)

            history.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"epoch={epoch:04d} train={train_loss:.6f} "
                f"diff={train_loss_diff:.6f} freq={train_loss_freq:.6f} "
                f"val={val_loss:.6f} "
                f"test={row.get('test/mean_score', float('nan')):.6f}"
            )

            metric = float(row.get("test/mean_score", -val_loss))
            if metric > best_metric:
                best_metric = metric
                best_val_loss = min(best_val_loss, val_loss)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "best_val_loss": best_val_loss,
                        "group": cfg.group,
                        "seed": cfg.seed,
                    },
                    best_path,
                )

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric": best_metric,
                        "best_val_loss": best_val_loss,
                        "group": cfg.group,
                        "seed": cfg.seed,
                    },
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                )

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    (outdir / "config.json").write_text(
        json.dumps(
            {
                "group": cfg.group,
                "seed": cfg.seed,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "lambda_freq": cfg.lambda_freq,
                "lambda_warmup_epochs": cfg.lambda_warmup_epochs,
                "soft_mask": cfg.soft_mask,
                "soft_mask_tau": cfg.soft_mask_tau,
                "eval_every": args.eval_every,
                "best_metric": best_metric,
                "best_val_loss": best_val_loss,
                "zarr_path": str(cfg.zarr_path),
                "dp_repo_path": str(cfg.dp_repo_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nbest_metric   = {best_metric:.6f}")
    print(f"best_val_loss = {best_val_loss:.6f}")
    print(f"saved best checkpoint -> {best_path}")
    print(f"saved history         -> {outdir / 'training_history.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 training for SFC-DP."
    )
    parser.add_argument("--group", type=str, default="D", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-freq", type=float, default=0.001,
                        help="Weight on L_freq; reduced from 0.1 (initial L_freq≈87 "
                             "caused 39× L_diff at epoch 0, collapsing early training)")
    parser.add_argument("--lambda-warmup-epochs", type=int, default=20,
                        help="Linear warmup: ramp λ from 0 → lambda_freq over this many epochs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--soft-mask", action="store_true")
    parser.add_argument("--soft-mask-tau", type=float, default=0.05)
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
        default="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/sfc_runs/group_D_seed42",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=2)
    parser.add_argument("--n-train-vis", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--n-test-vis", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = SFCConfig(
        group=args.group,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_freq=args.lambda_freq,
        lambda_warmup_epochs=args.lambda_warmup_epochs,
        soft_mask=args.soft_mask,
        soft_mask_tau=args.soft_mask_tau,
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
