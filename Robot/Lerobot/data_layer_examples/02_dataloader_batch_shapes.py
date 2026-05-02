# -*- coding: utf-8 -*-
"""
Inspect batched LeRobot data.

This is the next step after 01_load_and_inspect_dataset.py. A policy never sees
one loose frame; during training it sees a batch returned by torch DataLoader.

Based on the official LeRobotDataset loading pattern:
https://huggingface.co/docs/lerobot/lerobot-dataset-v3

Try:
    python 02_dataloader_batch_shapes.py --batch-size 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "svla_so101_pickplace"


def describe_batch_value(value) -> str:
    if hasattr(value, "shape"):
        return f"shape={tuple(value.shape)}, dtype={getattr(value, 'dtype', None)}"
    if isinstance(value, list):
        return f"list[len={len(value)}]"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--root", type=Path, default=default_dataset_root())
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    root = args.root if args.root and args.root.exists() else None
    if args.root and root is None:
        print(f"Local root not found, falling back to Hub download: {args.root}")

    dataset = LeRobotDataset(args.repo_id, root=root, episodes=args.episodes, video_backend=args.video_backend)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    batch = next(iter(dataloader))

    print("=== Dataset ===")
    print("repo_id:", args.repo_id)
    print("root:", dataset.root)
    print("num_frames:", dataset.num_frames)
    print("num_episodes:", dataset.num_episodes)

    print("=== Batch keys and shapes ===")
    for key in sorted(batch.keys()):
        print(f"{key}: {describe_batch_value(batch[key])}")

    print("\n=== How to read common shapes ===")
    print("observation.state: [B, state_dim]")
    print("action:            [B, action_dim] or [B, horizon, action_dim], depending on policy/dataset")
    print("image:             [B, C, H, W] or [B, T, C, H, W] with delta_timestamps")


if __name__ == "__main__":
    main()
