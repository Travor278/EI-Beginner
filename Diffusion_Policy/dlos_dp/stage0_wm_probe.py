"""
Stage 0 — World-Model learnability probe.

This stand-alone script verifies that a lightweight OutcomeWorldModel can
learn meaningful dynamics in the DINOv2 embedding space *before* we attempt
the full DLOS-DP Stage 1 training.

Pipeline
--------
1. Load all Push-T image frames from zarr.
2. Run FrozenDINOv2Encoder on every frame (batched) → save as
   ``artifacts/dino_embeddings.npy`` (cached on subsequent runs).
3. Build transition dataset: (z_t, a_t) → z_{t+1}, respecting episode
   boundaries.
4. Split transitions into train / val by episode.
5. Train OutcomeWorldModel (50 epochs, AdamW, lr=1e-4).
6. Report val MSE and compare to the trivial baseline (predict z_t for z_{t+1}).
7. Save best checkpoint + training-history CSV.

Pass criterion
--------------
    val_mse < 0.8 × baseline_mse    (≥ 20 % improvement over copy-input)

If the criterion is not met the script prints a prominent warning but still
saves results for inspection.

Usage
-----
    # Run from WSL with GPU:
    python stage0_wm_probe.py \\
        --zarr-path /home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr \\
        --outdir    /mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/wm_probe

    # Use cached DINO embeddings on subsequent runs:
    python stage0_wm_probe.py --embed-cache artifacts/dino_embeddings.npy ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader, TensorDataset

# Add the dlos_dp package to the path when run as __main__ from inside the
# dlos_dp/ directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dlos_dp.dino_encoder import FrozenDINOv2Encoder
from dlos_dp.world_model import OutcomeWorldModel


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def encode_all_frames(
    imgs: np.ndarray,       # (N, 3, H, W) uint8
    encoder: FrozenDINOv2Encoder,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:            # (N, 384) float32
    """Batch-encode all frames; returns numpy array for easy caching."""
    encoder.eval()
    results: list[np.ndarray] = []
    N = imgs.shape[0]
    for start in range(0, N, batch_size):
        batch = torch.from_numpy(
            imgs[start : start + batch_size].astype(np.float32) / 255.0
        ).to(device)
        with torch.no_grad():
            z = encoder(batch).cpu().numpy()
        results.append(z)
        if (start // batch_size) % 10 == 0:
            print(f"  encoded {min(start + batch_size, N)}/{N} frames")
    return np.concatenate(results, axis=0).astype(np.float32)


def build_transition_dataset(
    embeddings: np.ndarray,  # (N, D) float32
    actions: np.ndarray,     # (N, 2) float32
    episode_ends: np.ndarray,# (E,)   int64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (z_t, a_t, z_next, episode_ids_per_transition).

    Only steps within the same episode are paired; the last frame of each
    episode is skipped.
    """
    zs, as_, z_nexts, ep_ids = [], [], [], []
    ep_start = 0
    for ep_id, ep_end in enumerate(episode_ends.tolist()):
        ep_end = int(ep_end)
        for idx in range(ep_start, ep_end - 1):
            zs.append(embeddings[idx])
            as_.append(actions[idx])
            z_nexts.append(embeddings[idx + 1])
            ep_ids.append(ep_id)
        ep_start = ep_end
    return (
        np.stack(zs).astype(np.float32),
        np.stack(as_).astype(np.float32),
        np.stack(z_nexts).astype(np.float32),
        np.array(ep_ids, dtype=np.int64),
    )


def split_by_episode(
    ep_ids: np.ndarray,
    num_episodes: int,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train_mask, val_mask) over transitions."""
    rng = np.random.default_rng(seed)
    ids = np.arange(num_episodes)
    rng.shuffle(ids)
    val_count = max(1, int(round(num_episodes * val_ratio)))
    val_eps = set(ids[:val_count].tolist())
    val_mask = np.array([e in val_eps for e in ep_ids.tolist()], dtype=bool)
    return ~val_mask, val_mask


def train_one_epoch(
    model: OutcomeWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = torch.nn.MSELoss()
    total, count = 0.0, 0
    for z_batch, a_batch, zt1_batch in loader:
        z_batch  = z_batch.to(device)
        a_batch  = a_batch.to(device)
        zt1_batch = zt1_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(z_batch, a_batch)
        loss = criterion(pred, zt1_batch)
        loss.backward()
        optimizer.step()
        bs = z_batch.size(0)
        total += float(loss.item()) * bs
        count += bs
    return total / max(count, 1)


def eval_one_epoch(
    model: OutcomeWorldModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = torch.nn.MSELoss()
    total, count = 0.0, 0
    with torch.no_grad():
        for z_batch, a_batch, zt1_batch in loader:
            pred = model(z_batch.to(device), a_batch.to(device))
            loss = criterion(pred, zt1_batch.to(device))
            bs = z_batch.size(0)
            total += float(loss.item()) * bs
            count += bs
    return total / max(count, 1)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0: train a WM probe on DINOv2 Push-T transitions."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path(
            "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/wm_probe"
        ),
    )
    parser.add_argument(
        "--embed-cache",
        type=Path,
        default=None,
        help="Path to cached dino_embeddings.npy.  If provided and file exists, "
             "skip the encoding step.",
    )
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--hidden-dim", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Batch size for DINO encoding (reduce if GPU OOM).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    # ---------------------------------------------------------------- #
    # 1. Load zarr                                                       #
    # ---------------------------------------------------------------- #
    print("Loading zarr …")
    root = zarr.open(str(args.zarr_path), mode="r")
    imgs         = np.asarray(root["data"]["img"][:],           dtype=np.uint8)    # (N,3,96,96)
    actions      = np.asarray(root["data"]["action"][:],        dtype=np.float32)  # (N,2)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:],  dtype=np.int64)    # (E,)
    N, E = imgs.shape[0], len(episode_ends)
    print(f"  frames={N}, episodes={E}")

    # ---------------------------------------------------------------- #
    # 2. DINO encoding (with cache)                                      #
    # ---------------------------------------------------------------- #
    embed_cache = args.embed_cache
    if embed_cache is None:
        embed_cache = args.outdir / "dino_embeddings.npy"

    if embed_cache.exists():
        print(f"Loading cached embeddings from {embed_cache} …")
        embeddings = np.load(str(embed_cache))
        assert embeddings.shape[0] == N, (
            f"Cache shape mismatch: {embeddings.shape[0]} vs {N} frames"
        )
    else:
        print("Encoding frames with FrozenDINOv2Encoder …")
        encoder = FrozenDINOv2Encoder().to(device)
        embeddings = encode_all_frames(imgs, encoder, device, args.encode_batch_size)
        embed_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(embed_cache), embeddings)
        print(f"  saved embeddings → {embed_cache}")
        del encoder  # free GPU memory

    obs_dim = embeddings.shape[1]   # 384 for ViT-S/14
    print(f"  embedding dim={obs_dim}")

    # ---------------------------------------------------------------- #
    # 3. Build transition arrays                                         #
    # ---------------------------------------------------------------- #
    print("Building transitions …")
    z_t, a_t, z_next, ep_ids = build_transition_dataset(embeddings, actions, episode_ends)
    print(f"  transitions={len(z_t)}")

    # ---------------------------------------------------------------- #
    # 4. Train / val split                                               #
    # ---------------------------------------------------------------- #
    train_mask, val_mask = split_by_episode(ep_ids, E, args.val_ratio, args.seed)
    print(f"  train={train_mask.sum()}, val={val_mask.sum()}")

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

    # Trivial baseline: copy z_t as prediction for z_{t+1}
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        baseline_mse = float(
            criterion(
                torch.from_numpy(z_t[val_mask]),
                torch.from_numpy(z_next[val_mask]),
            ).item()
        )
    print(f"  baseline_mse (copy z_t)={baseline_mse:.6f}")

    # ---------------------------------------------------------------- #
    # 5. Train WM probe                                                  #
    # ---------------------------------------------------------------- #
    model = OutcomeWorldModel(
        obs_dim=obs_dim, action_dim=a_t.shape[1], hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val = float("inf")
    best_path = args.outdir / "best_wm_probe.pt"

    for epoch in range(args.epochs):
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_mse   = eval_one_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_dim":    obs_dim,
                    "action_dim": a_t.shape[1],
                    "hidden_dim": args.hidden_dim,
                    "epoch":      epoch,
                    "val_mse":    val_mse,
                    "seed":       args.seed,
                },
                best_path,
            )

        print(
            f"epoch={epoch:03d}  train_mse={train_mse:.6f}  "
            f"val_mse={val_mse:.6f}  best={best_val:.6f}"
        )

    # ---------------------------------------------------------------- #
    # 6. Report                                                          #
    # ---------------------------------------------------------------- #
    threshold = 0.8 * baseline_mse
    passed = best_val < threshold
    print("\n" + "=" * 60)
    print(f"baseline_mse = {baseline_mse:.6f}")
    print(f"best_val_mse = {best_val:.6f}  (threshold < {threshold:.6f})")
    print(f"STAGE-0 PASS: {passed}")
    if not passed:
        print(
            "WARNING: WM probe did not achieve ≥20 % improvement over baseline.  "
            "Check learning rate, model capacity, or data quality before Stage 1."
        )
    print("=" * 60)

    # ---------------------------------------------------------------- #
    # 7. Save history + summary                                          #
    # ---------------------------------------------------------------- #
    history_path = args.outdir / "wm_probe_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "zarr_path":      str(args.zarr_path),
        "embed_cache":    str(embed_cache),
        "best_checkpoint": str(best_path),
        "history_csv":    str(history_path),
        "baseline_mse":   baseline_mse,
        "best_val_mse":   best_val,
        "stage0_pass":    passed,
        "epochs":         args.epochs,
        "seed":           args.seed,
        "obs_dim":        obs_dim,
        "action_dim":     int(a_t.shape[1]),
        "device":         str(device),
    }
    summary_path = args.outdir / "wm_probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nsaved checkpoint → {best_path}")
    print(f"saved history    → {history_path}")
    print(f"saved summary    → {summary_path}")


if __name__ == "__main__":
    main()
