# -*- coding: utf-8 -*-
"""
Replay actions from a recorded LeRobot episode.

Adapted from the official LeRobot real-robot imitation learning tutorial:
https://huggingface.co/docs/lerobot/il_robots

Use this to understand that a recorded dataset is not just images: the "action"
column is directly convertible back to robot commands via dataset.features["action"]["names"].
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.utils import log_say


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "svla_so101_pickplace"


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional compatibility alias: pass a local root path or a Hub repo id.",
    )
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions; do not connect to robot.")
    parser.add_argument("--port", default="/dev/tty.usbmodem58760434471")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    args = parser.parse_args()

    repo_id = args.repo_id
    root = args.root if args.root and args.root.exists() else None
    if args.dataset:
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            root = dataset_path
        else:
            repo_id = args.dataset

    dataset = LeRobotDataset(repo_id, root=root, episodes=[args.episode_idx], video_backend=args.video_backend)
    actions = dataset.select_columns("action")

    action_names = dataset.features["action"]["names"]
    print("action names:", action_names)
    print("fps:", dataset.fps)
    print("num_frames:", dataset.num_frames)

    robot = None
    if not args.dry_run:
        robot_config = SO100FollowerConfig(port=args.port, id=args.robot_id)
        robot = SO100Follower(robot_config)
        robot.connect()

    try:
        log_say(f"Replaying episode {args.episode_idx}")
        num_steps = dataset.num_frames if args.max_steps <= 0 else min(args.max_steps, dataset.num_frames)
        for idx in range(num_steps):
            t0 = time.perf_counter()
            action = {
                name: float(actions[idx]["action"][i])
                for i, name in enumerate(action_names)
            }

            if args.dry_run:
                print(idx, action)
            else:
                robot.send_action(action)

            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
