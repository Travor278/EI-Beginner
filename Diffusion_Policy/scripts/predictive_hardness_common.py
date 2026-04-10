from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr


@dataclass(frozen=True)
class EpisodeSpan:
    episode_id: int
    start_idx: int
    end_idx: int

    @property
    def num_steps(self) -> int:
        return self.end_idx - self.start_idx


class ForwardDynamicsMLP(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_pusht_arrays(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    state = np.asarray(root["data"]["state"][:], dtype=np.float32)
    action = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return state, action, episode_ends


def build_episode_spans(episode_ends: np.ndarray) -> list[EpisodeSpan]:
    spans: list[EpisodeSpan] = []
    start = 0
    for episode_id, end in enumerate(episode_ends.tolist()):
        end = int(end)
        spans.append(EpisodeSpan(episode_id=episode_id, start_idx=start, end_idx=end))
        start = end
    return spans


def split_episode_ids(num_episodes: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = np.arange(num_episodes)
    rng.shuffle(ids)
    val_count = max(1, int(round(num_episodes * val_ratio)))
    val_ids = np.sort(ids[:val_count]).tolist()
    train_ids = np.sort(ids[val_count:]).tolist()
    return train_ids, val_ids


def build_transition_arrays(
    state: np.ndarray,
    action: np.ndarray,
    spans: list[EpisodeSpan],
    episode_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for span in spans:
        if span.episode_id not in episode_ids:
            continue
        for idx in range(span.start_idx, span.end_idx - 1):
            xs.append(np.concatenate([state[idx], action[idx]], axis=0))
            ys.append(state[idx + 1])
    if not xs:
        raise ValueError("No transitions were built. Check the episode split.")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def one_step_predict(
    model: ForwardDynamicsMLP,
    state_t: np.ndarray,
    action_t: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    state_norm = apply_standardizer(state_t[None, :], state_mean, state_std)
    action_norm = apply_standardizer(action_t[None, :], action_mean, action_std)
    with torch.no_grad():
        pred_norm = model(
            torch.from_numpy(state_norm).to(device),
            torch.from_numpy(action_norm).to(device),
        )
    pred = pred_norm.cpu().numpy()[0] * target_std + target_mean
    return pred.astype(np.float32)

