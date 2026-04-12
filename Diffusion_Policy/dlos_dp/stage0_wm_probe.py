"""Stage 0: learnability probe for the lightweight world model."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, TensorDataset

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dlos_dp.dino_encoder import FrozenDINOv2Encoder
from dlos_dp.world_model import OutcomeWorldModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def encode_all_frames(
    imgs_hwc: np.ndarray,
    encoder: FrozenDINOv2Encoder,
    device: torch.device,
    batch_size: int,
    max_batches: int | None = None,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    total = imgs_hwc.shape[0]
    for batch_idx, start in enumerate(range(0, total, batch_size)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        stop = min(start + batch_size, total)
        batch = imgs_hwc[start:stop].astype(np.float32) / 255.0
        batch = np.moveaxis(batch, -1, 1)
        batch_tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            z = encoder(batch_tensor).cpu().numpy()
        encoded.append(z)
        print(f"  encoded {stop}/{total} frames")
    return np.concatenate(encoded, axis=0).astype(np.float32)


def build_transition_dataset(
    embeddings: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_t, a_t, z_next, ep_ids = [], [], [], []
    ep_start = 0
    for ep_id, ep_end in enumerate(episode_ends.tolist()):
        ep_end = int(ep_end)
        for idx in range(ep_start, ep_end - 1):
            z_t.append(embeddings[idx])
            a_t.append(actions[idx])
            z_next.append(embeddings[idx + 1])
            ep_ids.append(ep_id)
        ep_start = ep_end

    return (
        np.asarray(z_t, dtype=np.float32),
        np.asarray(a_t, dtype=np.float32),
        np.asarray(z_next, dtype=np.float32),
        np.asarray(ep_ids, dtype=np.int64),
    )


def truncate_to_complete_episodes(
    embeddings: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only frames that belong to fully encoded episodes."""
    encoded_frames = int(embeddings.shape[0])
    valid_episode_ends = episode_ends[episode_ends <= encoded_frames]
    if len(valid_episode_ends) < 2:
        raise ValueError(
            "Stage 0 needs at least 2 fully encoded episodes. "
            "Increase --max-encode-batches or disable truncated encoding."
        )

    last_valid_frame = int(valid_episode_ends[-1])
    return (
        embeddings[:last_valid_frame].astype(np.float32, copy=False),
        actions[:last_valid_frame].astype(np.float32, copy=False),
        valid_episode_ends.astype(np.int64, copy=False),
    )


def build_split_masks(
    *,
    ep_ids: np.ndarray,
    num_episodes: int,
    val_ratio: float,
    max_train_episodes: int | None,
    seed: int,
    dp_repo_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    if dp_repo_path and dp_repo_path not in sys.path:
        sys.path.insert(0, dp_repo_path)

    from diffusion_policy.common.sampler import downsample_mask, get_val_mask

    val_episode_mask = get_val_mask(
        n_episodes=num_episodes,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_episode_mask = ~val_episode_mask
    train_episode_mask = downsample_mask(
        mask=train_episode_mask,
        max_n=max_train_episodes,
        seed=seed,
    )

    train_mask = train_episode_mask[ep_ids]
    val_mask = val_episode_mask[ep_ids]
    return train_mask, val_mask


def run_epoch(
    model: OutcomeWorldModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> float:
    criterion = torch.nn.MSELoss()
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0

    for batch_idx, (z_obs, action, z_next) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        z_obs = z_obs.to(device)
        action = action.to(device)
        z_next = z_next.to(device)

        with torch.set_grad_enabled(training):
            pred = model(z_obs, action)
            loss = criterion(pred, z_next)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = z_obs.shape[0]
        total += float(loss.item()) * bs
        count += bs

    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 WM probe.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/wm_probe"),
    )
    parser.add_argument("--embed-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-train-episodes", type=int, default=90)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dp-repo",
        type=str,
        default="/home/Travor/workspaces/diffusion_policy",
    )
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--max-encode-batches", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    root = zarr.open(str(args.zarr_path), mode="r")
    imgs = np.asarray(root["data"]["img"][:], dtype=np.uint8)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    print(f"frames={len(imgs)}, episodes={len(episode_ends)}")

    if args.embed_cache is not None:
        embed_cache = args.embed_cache
    elif args.max_encode_batches is None:
        embed_cache = args.outdir / "dino_embeddings.npy"
    else:
        embed_cache = args.outdir / f"dino_embeddings_smoke_b{args.max_encode_batches}.npy"

    expected_frames = len(imgs)
    if args.max_encode_batches is not None:
        expected_frames = min(
            len(imgs),
            args.max_encode_batches * args.encode_batch_size,
        )

    if embed_cache.exists():
        embeddings = np.load(str(embed_cache))
        if embeddings.shape[0] != expected_frames:
            print(
                f"cache shape mismatch ({embeddings.shape[0]} vs expected {expected_frames}), "
                "re-encoding."
            )
            embed_cache.unlink()
            embeddings = None
        else:
            print(f"loaded embeddings from {embed_cache}")
    else:
        embeddings = None

    if embeddings is None:
        print(f"encoding frames and writing cache -> {embed_cache}")
        encoder = FrozenDINOv2Encoder().to(device)
        embeddings = encode_all_frames(
            imgs_hwc=imgs,
            encoder=encoder,
            device=device,
            batch_size=args.encode_batch_size,
            max_batches=args.max_encode_batches,
        )
        np.save(str(embed_cache), embeddings)
        print(f"saved embeddings -> {embed_cache}")
        del encoder

    embeddings, actions, episode_ends = truncate_to_complete_episodes(
        embeddings=embeddings,
        actions=actions,
        episode_ends=episode_ends,
    )
    print(
        f"using complete episodes only: frames={len(embeddings)}, "
        f"episodes={len(episode_ends)}"
    )

    z_t, a_t, z_next, ep_ids = build_transition_dataset(embeddings, actions, episode_ends)
    train_mask, val_mask = build_split_masks(
        ep_ids=ep_ids,
        num_episodes=len(episode_ends),
        val_ratio=args.val_ratio,
        max_train_episodes=args.max_train_episodes,
        seed=args.seed,
        dp_repo_path=args.dp_repo,
    )
    print(
        f"transitions train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
        f"train_eps={int(len(np.unique(ep_ids[train_mask])))}, "
        f"val_eps={int(len(np.unique(ep_ids[val_mask])))}"
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_t[train_mask]),
            torch.from_numpy(a_t[train_mask]),
            torch.from_numpy(z_next[train_mask]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(z_t[val_mask]),
            torch.from_numpy(a_t[val_mask]),
            torch.from_numpy(z_next[val_mask]),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    baseline_mse = torch.nn.functional.mse_loss(
        torch.from_numpy(z_t[val_mask]),
        torch.from_numpy(z_next[val_mask]),
    ).item()

    model = OutcomeWorldModel(
        obs_dim=embeddings.shape[1],
        action_dim=a_t.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_path = args.outdir / "best_wm_probe.pt"
    history: list[dict] = []

    for epoch in range(args.epochs):
        train_mse = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            max_batches=args.max_train_batches,
        )
        val_mse = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_dim": embeddings.shape[1],
                    "action_dim": a_t.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_mse": val_mse,
                },
                best_path,
            )

        print(f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f} best={best_val:.6f}")

    pd.DataFrame(history).to_csv(args.outdir / "training_history.csv", index=False)
    summary = {
        "seed": args.seed,
        "baseline_mse": baseline_mse,
        "best_val_mse": best_val,
        "improvement_ratio": best_val / max(baseline_mse, 1e-12),
        "passed": bool(best_val < 0.8 * baseline_mse),
        "checkpoint": str(best_path),
        "embed_cache": str(embed_cache),
        "frames_used": int(len(embeddings)),
        "episodes_used": int(len(episode_ends)),
        "train_transitions": int(train_mask.sum()),
        "val_transitions": int(val_mask.sum()),
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
