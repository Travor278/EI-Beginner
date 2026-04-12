"""用官方 Diffusion Policy 组件训练 PHRew-DP。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pred_hardness.config import PHRConfig
from pred_hardness.dataset import PushTImagePHRDataset


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


def train(cfg: PHRConfig, args: argparse.Namespace) -> None:
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
    hardness_scores: np.ndarray | None = None
    if args.chunk_hardness and Path(args.chunk_hardness).exists():
        hardness_scores = np.load(str(args.chunk_hardness)).astype(np.float32)
        print(f"  loaded hardness scores: shape={hardness_scores.shape}")
    elif args.mode != "uniform":
        raise FileNotFoundError(
            f"--chunk-hardness file not found: {args.chunk_hardness}\n"
            "Run pred_hardness/stage2b_score_chunks.py first."
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
        hardness_scores=None,
    )
    print(f"  train chunks={len(train_dataset)}, val chunks={len(val_dataset)}")

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
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
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

    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=cfg.pred_horizon,
        n_action_steps=cfg.action_horizon,
        n_obs_steps=cfg.obs_horizon,
        num_inference_steps=cfg.num_train_timesteps,
        obs_as_global_cond=True,
        crop_shape=(84, 84),
        diffusion_step_embed_dim=128,
        down_dims=(128, 256, 512),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
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
        f"Training PHRew-DP for {cfg.epochs} epochs "
        f"(mode={args.mode}, seed={cfg.seed}, device={device}) …"
    )
    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(start_epoch, cfg.epochs):
            policy.train()
            train_sum, train_cnt = 0.0, 0
            for batch in train_loader:
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()
                bs = batch["action"].shape[0]
                train_sum += float(loss.item()) * bs
                train_cnt += bs
            lr_scheduler.step()
            train_loss = train_sum / max(train_cnt, 1)

            policy.eval()
            val_sum, val_cnt = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    loss = policy.compute_loss(batch)
                    bs = batch["action"].shape[0]
                    val_sum += float(loss.item()) * bs
                    val_cnt += bs
            val_loss = val_sum / max(val_cnt, 1)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_scheduler.get_last_lr()[0],
                "mode": args.mode,
                "seed": cfg.seed,
            }

            if (epoch % args.eval_every) == 0:
                rollout_log = json_safe_metrics(env_runner.run(policy))
                row.update(rollout_log)

            history.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"epoch={epoch:04d} "
                f"train={train_loss:.6f} val={val_loss:.6f} "
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
                        "mode": args.mode,
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
                        "mode": args.mode,
                        "seed": cfg.seed,
                    },
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                )

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    (outdir / "config.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "seed": cfg.seed,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "eval_every": args.eval_every,
                "best_metric": best_metric,
                "best_val_loss": best_val_loss,
                "chunk_hardness": str(args.chunk_hardness),
                "zarr_path": str(cfg.zarr_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nbest_metric   = {best_metric:.6f}")
    print(f"best_val_loss = {best_val_loss:.6f}")
    print(f"saved best checkpoint → {best_path}")
    print(f"saved history         → {outdir / 'training_history.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PHRew-DP Stage 3: train DP with predictive hardness reweighting."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="soft_hard",
        choices=["uniform", "hard", "soft_hard", "easy"],
        help="Sampling weighting mode.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--chunk-hardness",
        type=str,
        default="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/phrew/chunk_hardness_norm.npy",
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
        default="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/phrew_runs/soft_hard_seed42",
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
