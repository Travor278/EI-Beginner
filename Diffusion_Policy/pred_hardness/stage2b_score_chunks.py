"""给训练 chunk 计算预测难度分数。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pred_hardness.chunk_scorer import ChunkScorer
from pred_hardness.dataset import PushTImagePHRDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2b: compute chunk-level predictive hardness scores."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path(
            "/home/Travor/workspaces/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/"
            "artifacts/state_forward_probe/best_state_forward_probe.pt"
        ),
        help="Path to best_state_forward_probe.pt from Stage 1.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "/mnt/d/Code/Learning/EI/EI-learning-notes/Diffusion_Policy/artifacts/phrew"
        ),
    )
    parser.add_argument("--obs-horizon",  type=int,   default=2)
    parser.add_argument("--pred-horizon", type=int,   default=16)
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--batch-size",   type=int,   default=512,
                        help="Batch size for forward-pass during scoring.")
    parser.add_argument("--device",       type=str,   default="auto")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Building train dataset index …")
    train_dataset = PushTImagePHRDataset(
        zarr_path=args.zarr_path,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        val_ratio=args.val_ratio,
        split="train",
        seed=args.seed,
        hardness_scores=None,
    )
    obs_starts = train_dataset.obs_starts
    N = len(obs_starts)
    print(f"  train chunks = {N}")

    print("Loading state arrays from zarr …")
    import zarr as zarr_lib
    root = zarr_lib.open(str(args.zarr_path), mode="r")
    state  = np.asarray(root["data"]["state"][:],  dtype=np.float32)
    action = np.asarray(root["data"]["action"][:], dtype=np.float32)
    print(f"  state shape={state.shape}  action shape={action.shape}")

    print("Initialising ChunkScorer …")
    scorer = ChunkScorer(checkpoint_path=args.checkpoint, device=args.device)
    print(f"  probe state_dim={scorer.state_dim}  action_dim={scorer.action_dim}")

    print("Scoring chunks …")
    hardness_raw = scorer.score_chunks(
        state=state,
        action=action,
        obs_starts=obs_starts,
        pred_horizon=args.pred_horizon,
        batch_size=args.batch_size,
    )

    hardness_norm = ChunkScorer.normalise_scores(hardness_raw)

    raw_path  = args.outdir / "chunk_hardness.npy"
    norm_path = args.outdir / "chunk_hardness_norm.npy"
    np.save(str(raw_path),  hardness_raw)
    np.save(str(norm_path), hardness_norm)

    meta = {
        "zarr_path":       str(args.zarr_path),
        "checkpoint":      str(args.checkpoint),
        "obs_horizon":     args.obs_horizon,
        "pred_horizon":    args.pred_horizon,
        "val_ratio":       args.val_ratio,
        "seed":            args.seed,
        "num_chunks":      N,
        "hardness_min":    float(hardness_raw.min()),
        "hardness_max":    float(hardness_raw.max()),
        "hardness_mean":   float(hardness_raw.mean()),
        "hardness_std":    float(hardness_raw.std()),
        "raw_npy":         str(raw_path),
        "norm_npy":        str(norm_path),
    }
    meta_path = args.outdir / "chunk_hardness_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    df = pd.DataFrame({
        "chunk_idx":      np.arange(N),
        "obs_start":      np.array(obs_starts, dtype=np.int64),
        "hardness_raw":   hardness_raw,
        "hardness_norm":  hardness_norm,
    })
    summary_path = args.outdir / "chunk_hardness_summary.csv"
    pd.concat([
        df.nsmallest(20, "hardness_norm").assign(group="easiest_20"),
        df.nlargest(20,  "hardness_norm").assign(group="hardest_20"),
    ]).sort_values("hardness_norm").to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"num_chunks    = {N}")
    print(f"hardness_min  = {hardness_raw.min():.6f}")
    print(f"hardness_max  = {hardness_raw.max():.6f}")
    print(f"hardness_mean = {hardness_raw.mean():.6f}")
    print(f"hardness_std  = {hardness_raw.std():.6f}")
    print("=" * 60)
    print("\nEasiest 5 chunks (low prediction error):")
    print(df.nsmallest(5, "hardness_norm")[
        ["chunk_idx", "obs_start", "hardness_raw", "hardness_norm"]
    ].to_string(index=False))
    print("\nHardest 5 chunks (high prediction error):")
    print(df.nlargest(5, "hardness_norm")[
        ["chunk_idx", "obs_start", "hardness_raw", "hardness_norm"]
    ].to_string(index=False))

    print(f"\nsaved raw scores   → {raw_path}")
    print(f"saved norm scores  → {norm_path}")
    print(f"saved metadata     → {meta_path}")
    print(f"saved summary csv  → {summary_path}")


if __name__ == "__main__":
    main()
