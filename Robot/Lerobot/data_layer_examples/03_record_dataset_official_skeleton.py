# -*- coding: utf-8 -*-
"""
Official-style real-robot recording skeleton.

This file follows the Hugging Face LeRobot "record" data flow. By default it
only builds and prints the dataset schema. Add --record after changing ports,
camera ids, robot type, repo_id, and task description to record with hardware.

Official docs:
https://huggingface.co/docs/lerobot/il_robots

Core idea:
    teleop.get_action()
    robot.get_observation()
    robot.send_action(action)
    dataset.add_frame(...)
    dataset.save_episode()

In the official helper path, record_loop handles the frame loop and dataset writes.
"""

from __future__ import annotations

import argparse
from pprint import pprint

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Grab the black cube"
DATASET_REPO_ID = "YOUR_HF_USER/record-test"


def build_dataset_features(robot, teleop_action_processor, robot_observation_processor) -> dict:
    """Convert hardware feature contracts into the current LeRobot dataset schema."""
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="Actually connect to the robot/teleop devices and record episodes.",
    )
    args = parser.parse_args()

    robot_config = SO100FollowerConfig(
        id="my_awesome_follower_arm",
        port="/dev/tty.usbmodem58760434471",
        cameras={
            "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        },
    )
    teleop_config = SO100LeaderConfig(
        id="my_awesome_leader_arm",
        port="/dev/tty.usbmodem585A0077581",
    )

    robot = SO100Follower(robot_config)
    teleop = SO100Leader(teleop_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Official pattern in LeRobot 0.5.x:
    # hardware features -> processor feature transform -> dataset schema.
    dataset_features = build_dataset_features(robot, teleop_action_processor, robot_observation_processor)

    if not args.record:
        print("Dry run: dataset schema generated from robot feature contracts.")
        print("Change ports/camera ids/repo_id/task first, then run with --record to connect hardware.\n")
        pprint(dataset_features)
        return

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    try:
        for episode_idx in range(NUM_EPISODES):
            log_say(f"Recording episode {episode_idx + 1}/{NUM_EPISODES}")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if events.get("rerecord_episode"):
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()

            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )
    finally:
        robot.disconnect()
        teleop.disconnect()
        # v3 datasets should be finalized before pushing if using incremental writers.
        if hasattr(dataset, "finalize"):
            dataset.finalize()

    # Uncomment after checking the dataset locally.
    # dataset.push_to_hub()


if __name__ == "__main__":
    main()
