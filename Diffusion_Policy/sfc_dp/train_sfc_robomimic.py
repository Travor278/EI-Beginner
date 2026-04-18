"""
SFC-DP Stage 2 — Robomimic Lift/Can gate test.

Purpose
-------
Verify that the frequency-supervision advantage observed on Push-T (first 100
epochs) also holds on a harder manipulation task with:
  - 7-DoF arm, action_dim=7 (pos+rot+gripper)
  - Two camera views (agentview + eye-in-hand)
  - 50-episode evaluation (far lower noise than Push-T's 10)
  - Richer action frequency structure (longer-range motions)

Usage
-----
    python sfc_dp/train_sfc_robomimic.py \
        --task lift \
        --group A \
        --seed 42 \
        --zarr-path /home/Travor/.../lift_image.zarr \
        --dp-repo   /home/Travor/workspaces/diffusion_policy \
        --outdir    /mnt/d/.../artifacts/sfc_robomimic/A_lift_seed42

Gate criterion (v3.6 final gate)
---------------------------------
    E > A at ≥ 80 epochs on *both* seeds 42 and 43.
    If criterion is met → proceed to full B/C/D/E × 2-seed ablation on Lift.
    If not → v3.6 = convergence-speed result only; pivot to v3.7.
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import copy
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

for _name in ("Iterable", "Mapping", "MutableMapping", "Sequence"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

from sfc_dp.config import SFCConfig
from sfc_dp.sfc_loss import SFCLoss
from sfc_dp.train_sfc import (   # reuse helpers from Push-T script
    build_policy_class,
    lambda_with_warmup,
    resolve_device,
    set_seed,
    json_safe_metrics,
)

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "lift": {
        "base_action_dim": 7,
        "obs_horizon": 2,
        "pred_horizon": 16,
        "action_horizon": 8,
        "shape_meta": {
            "action": {"shape": [7]},
            "obs": {
                "agentview_image":           {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eye_in_hand_image":  {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eef_pos":            {"shape": [3],         "type": "low_dim"},
                "robot0_eef_quat":           {"shape": [4],         "type": "low_dim"},
                "robot0_gripper_qpos":       {"shape": [2],         "type": "low_dim"},
            },
        },
        "crop_shape": (76, 76),
        "n_test": 50,      # Lift has 50 test episodes — much lower eval noise
        "n_test_vis": 4,
        "max_steps": 400,
    },
    "can": {
        "base_action_dim": 7,
        "obs_horizon": 2,
        "pred_horizon": 16,
        "action_horizon": 8,
        "shape_meta": {
            "action": {"shape": [7]},
            "obs": {
                "agentview_image":           {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eye_in_hand_image":  {"shape": [3, 84, 84], "type": "rgb"},
                "robot0_eef_pos":            {"shape": [3],         "type": "low_dim"},
                "robot0_eef_quat":           {"shape": [4],         "type": "low_dim"},
                "robot0_gripper_qpos":       {"shape": [2],         "type": "low_dim"},
            },
        },
        "crop_shape": (76, 76),
        "n_test": 50,
        "n_test_vis": 4,
        "max_steps": 400,
    },
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: SFCConfig, args: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    task_cfg = TASK_CONFIGS[args.task]
    shape_meta = copy.deepcopy(task_cfg["shape_meta"])
    action_dim = 10 if args.abs_action else int(task_cfg["base_action_dim"])
    shape_meta["action"]["shape"] = [action_dim]

    outdir = Path(args.outdir).resolve()
    ckpt_dir = outdir / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if str(cfg.dp_repo_path) not in sys.path:
        sys.path.insert(0, str(cfg.dp_repo_path))

    # ------------------------------------------------------------------
    # Imports from dp repo
    # ------------------------------------------------------------------
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    # Robomimic-specific
    try:
        from diffusion_policy.dataset.robomimic_replay_image_dataset import (
            RobomimicReplayImageDataset,
        )
        import diffusion_policy.env_runner.robomimic_image_runner as robomimic_runner_mod
    except ImportError as e:
        raise ImportError(
            "Could not import Robomimic dataset/runner from dp repo. "
            "Make sure the dp repo supports robomimic and robosuite is installed.\n"
            f"Original error: {e}"
        )

    original_async_vector_env = robomimic_runner_mod.AsyncVectorEnv

    def async_vector_env_no_shared(*env_args, **env_kwargs):
        env_kwargs.setdefault("shared_memory", False)
        return original_async_vector_env(*env_args, **env_kwargs)

    robomimic_runner_mod.AsyncVectorEnv = async_vector_env_no_shared
    RobomimicImageRunner = robomimic_runner_mod.RobomimicImageRunner

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print(f"Building Robomimic {args.task} datasets …")
    dataset_kwargs = dict(
        dataset_path=cfg.zarr_path,
        shape_meta=shape_meta,
        horizon=cfg.pred_horizon,
        n_obs_steps=cfg.obs_horizon,
        pad_before=cfg.obs_horizon - 1,
        pad_after=cfg.pred_horizon - cfg.action_horizon - 1,
        abs_action=args.abs_action,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=False,
        seed=cfg.seed,
        val_ratio=cfg.val_ratio,
    )
    train_dataset = RobomimicReplayImageDataset(**dataset_kwargs)
    val_dataset = train_dataset.get_validation_dataset()
    print(f"  train={len(train_dataset)}, val={len(val_dataset)}")

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

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    SFCPolicy = build_policy_class(DiffusionUnetHybridImagePolicy)
    policy = SFCPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=cfg.pred_horizon,
        n_action_steps=cfg.action_horizon,
        n_obs_steps=cfg.obs_horizon,
        num_inference_steps=cfg.num_inference_steps,
        obs_as_global_cond=True,
        crop_shape=task_cfg["crop_shape"],
        diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
        down_dims=cfg.down_dims,
        kernel_size=cfg.kernel_size,
        n_groups=cfg.n_groups,
        cond_predict_scale=cfg.cond_predict_scale,
        obs_encoder_group_norm=cfg.obs_encoder_group_norm,
        eval_fixed_crop=cfg.eval_fixed_crop,
        _sfc_cfg=cfg,
    )

    # ------------------------------------------------------------------
    # Normalizer — fit from dataset's replay buffer
    # ------------------------------------------------------------------
    print("Fitting normalizer …")
    normalizer = train_dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, weight_decay=1e-6
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # ------------------------------------------------------------------
    # Env runner
    # ------------------------------------------------------------------
    env_runner = RobomimicImageRunner(
        output_dir=str(outdir),
        dataset_path=cfg.zarr_path,
        shape_meta=shape_meta,
        n_train=args.n_train,
        n_train_vis=args.n_train_vis,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        max_steps=args.max_steps,
        n_obs_steps=cfg.obs_horizon,
        n_action_steps=cfg.action_horizon,
        abs_action=args.abs_action,
        tqdm_interval_sec=1.0,
        n_envs=args.n_envs,
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Training loop  (identical logic to train_sfc.py)
    # ------------------------------------------------------------------
    history: list[dict] = []
    log_path = outdir / "logs.json.txt"
    print(
        f"Training SFC-DP ({args.task}) for {cfg.epochs} epochs "
        f"(group={cfg.group}, seed={cfg.seed}, λ={cfg.lambda_freq}) …"
    )

    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(start_epoch, cfg.epochs):
            policy.train()
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
                train_sum  += float(loss.item()) * bs
                diff_sum   += float(loss_diff.item()) * bs
                freq_sum   += float(loss_freq.item()) * bs
                train_cnt  += bs
            lr_scheduler.step()

            train_loss      = train_sum  / max(train_cnt, 1)
            train_loss_diff = diff_sum   / max(train_cnt, 1)
            train_loss_freq = freq_sum   / max(train_cnt, 1)

            # Val (diffusion loss only — fair cross-group comparison)
            policy.eval()
            val_sum, val_cnt = 0.0, 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                        break
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    _, loss_diff, _ = policy.compute_loss_components(batch)
                    bs = batch["action"].shape[0]
                    val_sum  += float(loss_diff.item()) * bs
                    val_cnt  += bs
            val_loss = val_sum / max(val_cnt, 1)

            row: dict = {
                "epoch":            epoch,
                "train_loss":       train_loss,
                "train_loss_diff":  train_loss_diff,
                "train_loss_freq":  train_loss_freq,
                "val_loss":         val_loss,
                "lr":               lr_scheduler.get_last_lr()[0],
                "group":            cfg.group,
                "seed":             cfg.seed,
            }

            # Eval
            if (epoch + 1) % args.eval_every == 0:
                with torch.no_grad():
                    policy.eval()
                    runner_log = env_runner.run(policy)
                row.update(json_safe_metrics(runner_log))

                test_score = runner_log.get("test/mean_score", float("nan"))
                if not np.isnan(test_score) and test_score > best_metric:
                    best_metric = test_score
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
                print(
                    f"[ep {epoch:>3}] val_loss={val_loss:.4f}  "
                    f"test_score={test_score:.4f}  best={best_metric:.4f}  "
                    f"l_freq={train_loss_freq:.2e}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

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

            history.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    (outdir / "config.json").write_text(
        json.dumps(
            {
                "task":               args.task,
                "group":              cfg.group,
                "seed":               cfg.seed,
                "epochs":             cfg.epochs,
                "batch_size":         cfg.batch_size,
                "lr":                 cfg.lr,
                "lambda_freq":        cfg.lambda_freq,
                "lambda_warmup_epochs": cfg.lambda_warmup_epochs,
                "soft_mask":          cfg.soft_mask,
                "best_metric":        best_metric,
                "best_val_loss":      best_val_loss,
                "zarr_path":          str(cfg.zarr_path),
                "dp_repo_path":       str(cfg.dp_repo_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nbest_metric   = {best_metric:.6f}")
    print(f"best_val_loss = {best_val_loss:.6f}")
    print(f"saved best checkpoint → {best_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFC-DP gate test on Robomimic Lift/Can."
    )
    parser.add_argument("--task",   type=str, default="lift", choices=list(TASK_CONFIGS))
    parser.add_argument("--group",  type=str, default="A",  choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--lambda-freq",  type=float, default=0.01,
                        help="λ_freq; 0.01 is the Push-T optimal; may need tuning on Lift")
    parser.add_argument("--lambda-warmup-epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--soft-mask", action="store_true")
    parser.add_argument("--soft-mask-tau", type=float, default=0.05)
    parser.add_argument("--abs-action", action="store_true", default=True,
                        help="Use absolute actions (standard for Lift); set --no-abs-action for delta")
    parser.add_argument("--no-abs-action", dest="abs_action", action="store_false")
    parser.add_argument(
        "--zarr-path", type=str,
        default="/home/Travor/workspaces/diffusion_policy/data/robomimic/lift/ph/image_abs.hdf5",
        help="Path to robomimic dataset (hdf5 or zarr). "
             "Common locations:\n"
             "  Lift: data/robomimic/lift/ph/image_abs.hdf5\n"
             "  Can:  data/robomimic/can/ph/image_abs.hdf5",
    )
    parser.add_argument(
        "--dp-repo", type=str,
        default="/home/Travor/workspaces/diffusion_policy",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/sfc_robomimic/A_lift_seed42",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--n-train",     type=int, default=2)
    parser.add_argument("--n-train-vis", type=int, default=1)
    parser.add_argument("--n-test",      type=int, default=None)
    parser.add_argument("--n-test-vis",  type=int, default=None)
    parser.add_argument("--max-steps",   type=int, default=None)
    parser.add_argument("--n-envs",      type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches",   type=int, default=None)
    args = parser.parse_args()

    task_cfg = TASK_CONFIGS[args.task]
    if args.n_test is None:
        args.n_test = int(task_cfg["n_test"])
    if args.n_test_vis is None:
        args.n_test_vis = int(task_cfg["n_test_vis"])
    if args.max_steps is None:
        args.max_steps = int(task_cfg["max_steps"])

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
        action_dim=10 if args.abs_action else int(task_cfg["base_action_dim"]),
        obs_horizon=task_cfg["obs_horizon"],
        pred_horizon=task_cfg["pred_horizon"],
        action_horizon=task_cfg["action_horizon"],
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
