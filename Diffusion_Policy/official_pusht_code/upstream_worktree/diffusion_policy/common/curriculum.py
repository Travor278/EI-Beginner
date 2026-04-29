from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class OrderedIndexSampler(Sampler[int]):
    def __init__(self, indices):
        self.indices = [int(idx) for idx in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_curriculum_index_order(
    sampler_indices: np.ndarray,
    episode_lookup: np.ndarray,
    episode_mask: np.ndarray,
    curriculum_csv_path: Optional[str],
    curriculum_mode: str = "random",
    curriculum_metric: str = "difficulty_score_mean",
) -> Optional[np.ndarray]:
    if curriculum_mode == "random" or curriculum_csv_path is None:
        return None

    if curriculum_mode not in {"easy_to_hard", "hard_to_easy"}:
        raise ValueError(f"Unsupported curriculum_mode={curriculum_mode}")

    curriculum_path = Path(curriculum_csv_path).expanduser()
    if not curriculum_path.is_file():
        raise FileNotFoundError(f"Curriculum CSV not found: {curriculum_path}")

    table = pd.read_csv(curriculum_path)
    if curriculum_metric not in table.columns:
        raise KeyError(f"Curriculum metric {curriculum_metric} not in CSV columns")

    train_episode_ids = np.flatnonzero(episode_mask).tolist()
    table = table[table["episode_id"].isin(train_episode_ids)].copy()
    ascending = curriculum_mode == "easy_to_hard"
    table = table.sort_values(
        by=[curriculum_metric, "episode_id"],
        ascending=[ascending, True],
    )

    sample_episode_ids = episode_lookup[sampler_indices[:, 0].astype(np.int64)]
    ordered_dataset_indices = []
    for episode_id in table["episode_id"].tolist():
        episode_dataset_indices = np.flatnonzero(sample_episode_ids == episode_id)
        ordered_dataset_indices.extend(episode_dataset_indices.tolist())

    return np.asarray(ordered_dataset_indices, dtype=np.int64)
