"""Push-T image dataset wrapper for DLOS-DP."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import torch


class PushTImageDLOSDataset:
    """
    Aligns with the official Push-T image dataset used by Diffusion Policy and
    adds one extra field: ``obs_next_image``.

    Output structure:
        {
            "obs": {
                "image":      (T, 3, 96, 96),
                "agent_pos":  (T, 2),
            },
            "action":         (T, 2),
            "obs_next_image": (3, 96, 96),
        }
    """

    def __init__(
        self,
        zarr_path: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        pad_before: int = 1,
        pad_after: int = 7,
        val_ratio: float = 0.02,
        max_train_episodes: int | None = 90,
        split: str = "train",
        seed: int = 42,
        dp_repo_path: str = "/home/Travor/workspaces/diffusion_policy",
    ) -> None:
        if dp_repo_path and dp_repo_path not in sys.path:
            sys.path.insert(0, dp_repo_path)
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        from diffusion_policy.common.sampler import (
            SequenceSampler,
            downsample_mask,
            get_val_mask,
        )
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        self._LinearNormalizer = LinearNormalizer
        self._get_image_range_normalizer = get_image_range_normalizer

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.seed = seed

        self.replay_buffer = ReplayBuffer.copy_from_path(
            str(zarr_path),
            keys=["img", "state", "action"],
        )
        self.states = self.replay_buffer["state"]
        self.actions = self.replay_buffer["action"]

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        episode_mask = train_mask if split == "train" else ~train_mask
        self.train_mask = train_mask
        self.split = split
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=pred_horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=episode_mask,
        )

    def get_validation_dataset(self) -> "PushTImageDLOSDataset":
        val_set = copy.copy(self)
        from diffusion_policy.common.sampler import SequenceSampler

        val_set.split = "val"
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.pred_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs):
        normalizer = self._LinearNormalizer()
        normalizer.fit(
            data={
                "action": self.replay_buffer["action"],
                "agent_pos": self.replay_buffer["state"][..., :2],
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["image"] = self._get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: dict) -> dict:
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32) / 255.0
        agent_pos = sample["state"][:, :2].astype(np.float32)
        action = sample["action"].astype(np.float32)
        obs_next_image = image[self.obs_horizon].copy()

        return {
            "obs": {
                "image": image,
                "agent_pos": agent_pos,
            },
            "action": action,
            "obs_next_image": obs_next_image,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from diffusion_policy.common.pytorch_util import dict_apply

        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
