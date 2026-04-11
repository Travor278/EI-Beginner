"""
PushTImagePHRDataset — zarr-based Push-T image dataset for PHRew-DP.

Each sample:
    obs    : (obs_horizon, 3, H, W)  float32 [0, 1]
    action : (pred_horizon, 2)       float32

This matches the official Diffusion Policy image dataset format exactly
(no obs_next — the world-model constraint is NOT used in v3.1).

Key addition: get_weights() returns per-sample sampling weights that can
be passed directly to torch.utils.data.WeightedRandomSampler, enabling
soft reweighting based on chunk-level predictive hardness scores.

Weighting modes
---------------
uniform   — all weights = 1.0   (standard DP baseline)
hard      — weights ∝ h^alpha   (focus on hard chunks)
soft_hard — weights ∝ softmax(h_norm / T)   ← main PHRew-DP method
easy      — weights ∝ (1−h_norm)^alpha      (easy-first ablation)

All modes return weights with mean = 1.0 so that the effective number of
gradient steps per epoch is comparable across modes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset


class PushTImagePHRDataset(Dataset):
    """
    Parameters
    ----------
    zarr_path       : path to pusht_cchi_v7_replay.zarr
    obs_horizon     : consecutive observation frames (default 2)
    pred_horizon    : action chunk length (default 16)
    val_ratio       : fraction of episodes for validation (default 0.1)
    split           : 'train' or 'val'
    seed            : RNG seed for episode split
    hardness_scores : (num_train_chunks,) float32 array, or None for uniform
    """

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
        self.imgs    = np.asarray(root["data"]["img"][:],    dtype=np.uint8)     # (N,3,96,96)
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

    # ------------------------------------------------------------------ #
    # Weighting interface                                                  #
    # ------------------------------------------------------------------ #

    def get_weights(
        self,
        mode: str = "uniform",
        temperature: float = 0.1,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Return per-sample sampling weights for WeightedRandomSampler.

        All modes normalise the returned tensor so that its mean = 1.0,
        keeping the expected number of samples per epoch constant.

        Parameters
        ----------
        mode        : 'uniform' | 'hard' | 'soft_hard' | 'easy'
        temperature : softmax temperature for 'soft_hard' (lower → sharper)
        alpha       : power exponent for 'hard' / 'easy'
        """
        n = len(self.obs_starts)

        if mode == "uniform" or self._hardness is None:
            return torch.ones(n, dtype=torch.float32)

        h = self._hardness  # (n,) raw scores

        # Min-max normalise to [0, 1] for stable arithmetic
        lo, hi = h.min(), h.max()
        if (hi - lo).item() < 1e-8:
            return torch.ones(n, dtype=torch.float32)
        h_norm = (h - lo) / (hi - lo)   # (n,) in [0, 1]

        if mode == "hard":
            w = h_norm.pow(alpha)

        elif mode == "soft_hard":
            # Temperature-scaled softmax: T→0 = argmax (one chunk only),
            # T→∞ = uniform.  Multiply by n so mean ≈ 1.
            logits = h_norm / max(temperature, 1e-6)
            w = F.softmax(logits, dim=0) * n

        elif mode == "easy":
            w = (1.0 - h_norm).pow(alpha)

        else:
            raise ValueError(
                f"Unknown weighting mode: {mode!r}. "
                "Choose from 'uniform', 'hard', 'soft_hard', 'easy'."
            )

        # Normalise to mean = 1 so epoch length is consistent
        w = w / w.mean()
        return w.float()

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.obs_starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs_start = self.obs_starts[idx]

        obs = (
            self.imgs[obs_start : obs_start + self.obs_horizon]
            .astype(np.float32) / 255.0
        )                                                        # (obs_h, 3, 96, 96)
        action = self.actions[obs_start : obs_start + self.pred_horizon].copy()
                                                                 # (pred_h, 2)
        return {
            "obs":    torch.from_numpy(obs),
            "action": torch.from_numpy(action),
        }

    # ------------------------------------------------------------------ #
    # Index construction helpers                                           #
    # ------------------------------------------------------------------ #

    def _build_obs_starts(self, episode_ends: np.ndarray) -> list[int]:
        """Return all valid obs_start indices (episode-boundary-safe)."""
        obs_starts: list[int] = []
        ep_start = 0
        for ep_end in episode_ends.tolist():
            ep_end = int(ep_end)
            # Need: obs + pred horizon both within [ep_start, ep_end)
            # obs window: [obs_start, obs_start + obs_horizon)
            # act window: [obs_start, obs_start + pred_horizon)
            # → max obs_start = ep_end - pred_horizon (pred_horizon > obs_horizon)
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
