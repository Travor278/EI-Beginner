"""
PushTImageDLOSDataset — zarr-based Push-T dataset for DLOS-DP training.

Each sample contains:
    obs      : (obs_horizon, 3, H, W)  float32 [0, 1]
    action   : (pred_horizon, 2)       float32
    obs_next : (3, H, W)               float32 [0, 1]  — first frame after obs window

The key addition over the vanilla Push-T dataset is ``obs_next``, which is
the frame immediately following the observation window (i.e. the frame at
index obs_start + obs_horizon).  This is the supervision target for the
world-model branch in DLOS.

Episode boundary handling
--------------------------
For an episode spanning [ep_start, ep_end), the valid range for obs_start is:
    [ep_start,  ep_end - obs_horizon - pred_horizon]

This guarantees:
  • obs frames    : [obs_start,              obs_start + obs_horizon)   — all in episode
  • action frames : [obs_start,              obs_start + pred_horizon)  — all in episode
  • obs_next frame: obs_start + obs_horizon                              — still in episode

Note: obs_start == action_start (we use a aligned sliding window).

zarr data format
-----------------
    root['data']['img']          (N, 3, 96, 96)  uint8   [0, 255]
    root['data']['action']       (N, 2)           float32
    root['meta']['episode_ends'] (E,)             int64   exclusive end indices
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class PushTImageDLOSDataset(Dataset):
    """
    Parameters
    ----------
    zarr_path   : path to pusht_cchi_v7_replay.zarr
    obs_horizon : number of consecutive observation frames per sample (default 2)
    pred_horizon: action chunk length (default 16)
    val_ratio   : fraction of episodes reserved for validation (default 0.1)
    split       : 'train' or 'val'
    seed        : RNG seed for episode split
    """

    def __init__(
        self,
        zarr_path: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        val_ratio: float = 0.1,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        root = zarr.open(str(zarr_path), mode="r")
        # Load into memory (Push-T is small enough, ~200 MB uncompressed)
        self.imgs = np.asarray(root["data"]["img"][:], dtype=np.uint8)      # (N,3,96,96)
        self.actions = np.asarray(root["data"]["action"][:], dtype=np.float32)  # (N,2)
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)

        # Build (obs_start, obs_next_idx) index list
        all_indices = self._build_indices(episode_ends)

        # Episode-level train/val split
        episode_ids = self._split_episodes(len(episode_ends), val_ratio, seed)
        split_ep_set = set(episode_ids["train"] if split == "train" else episode_ids["val"])

        # Filter indices to only include episodes in the desired split
        # We need to know which episode each obs_start belongs to
        self.indices = self._filter_by_split(all_indices, episode_ends, split_ep_set)

    # ------------------------------------------------------------------ #
    # Index construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_indices(
        self,
        episode_ends: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Returns list of (obs_start, obs_next_idx) tuples.

        obs_start    — index of the first observation frame in the window
        obs_next_idx — index of the frame immediately after the obs window
                       (= obs_start + obs_horizon), used as WM target
        """
        indices: list[tuple[int, int]] = []
        ep_start = 0
        for ep_end in episode_ends.tolist():
            ep_end = int(ep_end)
            # Last valid obs_start so that both the obs window AND the action
            # chunk plus obs_next are within [ep_start, ep_end)
            # Need: obs_start + obs_horizon + pred_horizon - 1 < ep_end
            # i.e.  obs_start <= ep_end - obs_horizon - pred_horizon
            max_obs_start = ep_end - self.obs_horizon - self.pred_horizon
            for obs_start in range(ep_start, max_obs_start + 1):
                obs_next_idx = obs_start + self.obs_horizon  # always < ep_end
                indices.append((obs_start, obs_next_idx))
            ep_start = ep_end
        return indices

    @staticmethod
    def _split_episodes(
        num_episodes: int,
        val_ratio: float,
        seed: int,
    ) -> dict[str, list[int]]:
        rng = np.random.default_rng(seed)
        ids = np.arange(num_episodes)
        rng.shuffle(ids)
        val_count = max(1, int(round(num_episodes * val_ratio)))
        return {
            "val": np.sort(ids[:val_count]).tolist(),
            "train": np.sort(ids[val_count:]).tolist(),
        }

    @staticmethod
    def _filter_by_split(
        indices: list[tuple[int, int]],
        episode_ends: np.ndarray,
        split_ep_set: set[int],
    ) -> list[tuple[int, int]]:
        """Keep only indices whose obs_start falls within an episode in split_ep_set."""
        # Build a lookup: for each global frame index, which episode it belongs to
        # Use binary search on episode_ends for efficiency
        ep_ends = episode_ends.tolist()

        def episode_of(idx: int) -> int:
            # episode e contains frames [episode_ends[e-1], episode_ends[e])
            # (with episode_ends[-1] = 0 implicitly)
            lo, hi = 0, len(ep_ends) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if ep_ends[mid] <= idx:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        return [
            (obs_start, obs_next_idx)
            for obs_start, obs_next_idx in indices
            if episode_of(obs_start) in split_ep_set
        ]

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs_start, obs_next_idx = self.indices[idx]

        # Observation window: (obs_horizon, 3, H, W)
        obs = (
            self.imgs[obs_start : obs_start + self.obs_horizon]
            .astype(np.float32) / 255.0
        )

        # Action chunk: (pred_horizon, 2)
        action = self.actions[obs_start : obs_start + self.pred_horizon].copy()

        # Next observation frame: (3, H, W)
        obs_next = self.imgs[obs_next_idx].astype(np.float32) / 255.0

        return {
            "obs": torch.from_numpy(obs),           # (obs_horizon, 3, 96, 96)
            "action": torch.from_numpy(action),     # (pred_horizon, 2)
            "obs_next": torch.from_numpy(obs_next), # (3, 96, 96)
        }
