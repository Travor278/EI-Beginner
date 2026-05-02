# -*- coding: utf-8 -*-
"""
Minimal "Bring Your Own Hardware" contract.

This is adapted from the official LeRobot hardware integration tutorial:
https://huggingface.co/docs/lerobot/integrate_hardware

Read this file as the lowest-level data contract:

    observation_features  describes what get_observation() returns.
    action_features       describes what send_action(action) expects.

LeRobotDataset.create(...) later converts these hardware features into
dataset keys such as:

    observation.state
    observation.images.<camera_name>
    action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots import Robot, RobotConfig


@RobotConfig.register_subclass("my_cool_robot")
@dataclass
class MyCoolRobotConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )


class MyCoolRobot(Robot):
    config_class = MyCoolRobotConfig
    name = "my_cool_robot"

    def __init__(self, config: MyCoolRobotConfig):
        super().__init__(config)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
                "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_4": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_5": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def disconnect(self) -> None:
        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        raise NotImplementedError("Use the official tutorial to implement hardware-specific calibration.")

    def configure(self) -> None:
        # Set motor modes, gains, safety limits, etc. Keep this idempotent.
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}
        self.bus.sync_write("Goal_Position", goal_pos)
        return action
