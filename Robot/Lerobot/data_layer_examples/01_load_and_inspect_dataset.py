# -*- coding: utf-8 -*-
"""
Inspect a LeRobotDataset sample.

This example follows the official LeRobotDataset v3.0 docs:
https://huggingface.co/docs/lerobot/lerobot-dataset-v3

Purpose:
    Understand what a robot-learning frame looks like before studying policies.

Try:
    python 01_load_and_inspect_dataset.py --index 0
    python 01_load_and_inspect_dataset.py --index 0 --with-history
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "svla_so101_pickplace"


def describe_value(value) -> str:
    """Return a compact description of a tensor/list/scalar."""
    if hasattr(value, "shape"):
        return f"{type(value).__name__}, shape={tuple(value.shape)}, dtype={getattr(value, 'dtype', None)}"
    if isinstance(value, (str, int, float, bool)):
        return f"{type(value).__name__}: {value}"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--root", type=Path, default=default_dataset_root())
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--history-key", default="observation.images.up")
    parser.add_argument(
        "--with-history",
        action="store_true",
        help="Load a temporal image window using delta_timestamps, as in the official docs.",
    )
    args = parser.parse_args()

    delta_timestamps = None
    if args.with_history:
        # Seconds relative to the current frame. LeRobotDataset will return a time stack:
        # sample[args.history_key].shape == [T, C, H, W] if this key exists.
        delta_timestamps = {
            args.history_key: [-0.2, -0.1, 0.0],
        }

    root = args.root if args.root and args.root.exists() else None
    if args.root and root is None:
        print(f"Local root not found, falling back to Hub download: {args.root}")

    dataset = LeRobotDataset(
        args.repo_id,
        root=root,
        episodes=args.episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    print("=== Dataset summary ===")
    print("repo_id:", args.repo_id)
    print("root:", dataset.root)
    print("num_frames:", getattr(dataset, "num_frames", None))
    print("num_episodes:", getattr(dataset, "num_episodes", None))
    print("fps:", getattr(dataset, "fps", None))

    print("\n=== Feature schema ===")
    pprint(dataset.features)

    print("\n=== Stats keys ===")
    stats = getattr(getattr(dataset, "meta", None), "stats", None)
    if stats is None:
        print("No dataset.meta.stats found.")
    else:
        print(list(stats.keys()))

    print(f"\n=== Sample[{args.index}] ===")
    sample = dataset[args.index]
    for key in sorted(sample.keys()):
        print(f"{key}: {describe_value(sample[key])}")

    print("\n=== Important reading guide ===")
    print("observation.* = what the robot/environment senses at time t")
    print("action        = what the robot should execute, usually continuous")
    print("timestamp     = time alignment across state/action/images")
    print("episode_index = which trajectory this frame belongs to")


if __name__ == "__main__":
    main()
