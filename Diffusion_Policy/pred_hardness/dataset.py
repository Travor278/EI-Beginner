"""给 PHRew-DP 用的 Push-T 数据集。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset


class PushTImagePHRDataset(Dataset):
    """返回官方 image policy 能直接吃的 obs/action 结构。"""

    def __init__(
        self,
        zarr_path: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        val_ratio: float = 0.1,
        split: str = "train",
        seed: int = 42,
        hardness_scores: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.obs_horizon  = obs_horizon
        self.pred_horizon = pred_horizon

        root = zarr.open(str(zarr_path), mode="r")
        self.imgs    = np.asarray(root["data"]["img"][:],    dtype=np.float32)   # (N,96,96,3)
        self.states  = np.asarray(root["data"]["state"][:],  dtype=np.float32)   # (N,state_dim)
        self.actions = np.asarray(root["data"]["action"][:], dtype=np.float32)   # (N,2)
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

        # Build all valid (obs_start,) indices, episode-boundary-safe
        all_obs_starts = self._build_obs_starts(episode_ends)

        # Episode-level train/val split
        ep_split = self._split_episodes(len(episode_ends), val_ratio, seed)
        split_eps = set(ep_split["train"] if split == "train" else ep_split["val"])

        # Filter to desired split
        self.obs_starts: list[int] = self._filter_by_split(
            all_obs_starts, episode_ends, split_eps
        )

        # Store hardness scores aligned with self.obs_starts
        if hardness_scores is not None:
            assert len(hardness_scores) == len(self.obs_starts), (
                f"hardness_scores length {len(hardness_scores)} "
                f"!= dataset size {len(self.obs_starts)}"
            )
            self._hardness = torch.from_numpy(
                hardness_scores.astype(np.float32)
            )
        else:
            self._hardness = None

    def get_weights(
        self,
        mode: str = "uniform",
        temperature: float = 0.1,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """给 WeightedRandomSampler 返回每个样本的权重。"""
        n = len(self.obs_starts)

        if mode == "uniform" or self._hardness is None:
            return torch.ones(n, dtype=torch.float32)

        h = self._hardness  # (n,) raw scores

        lo, hi = h.min(), h.max()
        if (hi - lo).item() < 1e-8:
            return torch.ones(n, dtype=torch.float32)
        h_norm = (h - lo) / (hi - lo)   # (n,) in [0, 1]

        if mode == "hard":
            w = h_norm.pow(alpha)

        elif mode == "soft_hard":
            logits = h_norm / max(temperature, 1e-6)
            w = F.softmax(logits, dim=0) * n

        elif mode == "easy":
            w = (1.0 - h_norm).pow(alpha)

        else:
            raise ValueError(
                f"Unknown weighting mode: {mode!r}. "
                "Choose from 'uniform', 'hard', 'soft_hard', 'easy'."
            )

        w = w / w.mean()
        return w.float()

    def __len__(self) -> int:
        return len(self.obs_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs_start = self.obs_starts[idx]

        image = self.imgs[obs_start : obs_start + self.obs_horizon] / 255.0
        image = np.moveaxis(image, -1, 1).astype(np.float32)
        agent_pos = self.states[
            obs_start : obs_start + self.obs_horizon, :2
        ].astype(np.float32)
        action = self.actions[obs_start : obs_start + self.pred_horizon].copy()
        return {
            "obs": {
                "image": torch.from_numpy(image),
                "agent_pos": torch.from_numpy(agent_pos),
            },
            "action": torch.from_numpy(action),
        }

    def _build_obs_starts(self, episode_ends: np.ndarray) -> list[int]:
        """构造不跨 episode 边界的有效起点。"""
        obs_starts: list[int] = []
        ep_start = 0
        for ep_end in episode_ends.tolist():
            ep_end = int(ep_end)
            max_start = ep_end - self.pred_horizon
            for s in range(ep_start, max_start):
                obs_starts.append(s)
            ep_start = ep_end
        return obs_starts

    @staticmethod
    def _split_episodes(
        num_episodes: int, val_ratio: float, seed: int
    ) -> dict[str, list[int]]:
        rng = np.random.default_rng(seed)
        ids = np.arange(num_episodes)
        rng.shuffle(ids)
        val_count = max(1, int(round(num_episodes * val_ratio)))
        return {
            "val":   np.sort(ids[:val_count]).tolist(),
            "train": np.sort(ids[val_count:]).tolist(),
        }

    @staticmethod
    def _filter_by_split(
        obs_starts: list[int],
        episode_ends: np.ndarray,
        split_eps: set[int],
    ) -> list[int]:
        ep_ends = episode_ends.tolist()

        def episode_of(idx: int) -> int:
            lo, hi = 0, len(ep_ends) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if ep_ends[mid] <= idx:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        return [s for s in obs_starts if episode_of(s) in split_eps]
