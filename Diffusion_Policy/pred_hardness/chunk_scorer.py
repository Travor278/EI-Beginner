"""用 Stage 1 的 forward probe 给每个 chunk 打分。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from predictive_hardness_common import ForwardDynamicsMLP, apply_standardizer


class ChunkScorer:
    """加载 probe checkpoint，并计算每个 chunk 的预测误差。"""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        ckpt = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,   # checkpoint stores numpy arrays for normalisation stats
        )

        self._model = ForwardDynamicsMLP(
            state_dim=int(ckpt["state_dim"]),
            action_dim=int(ckpt["action_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
        ).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        self._state_mean   = ckpt["state_mean"]    # (state_dim,)
        self._state_std    = ckpt["state_std"]
        self._action_mean  = ckpt["action_mean"]   # (action_dim,)
        self._action_std   = ckpt["action_std"]
        self._target_mean  = ckpt["target_mean"]
        self._target_std   = ckpt["target_std"]

        self.state_dim  = int(ckpt["state_dim"])
        self.action_dim = int(ckpt["action_dim"])

    def score_chunks(
        self,
        state: np.ndarray,           # (N_frames, state_dim) float32
        action: np.ndarray,          # (N_frames, action_dim) float32
        obs_starts: list[int] | np.ndarray,  # chunk start indices
        pred_horizon: int,
        batch_size: int = 512,
    ) -> np.ndarray:                 # (num_chunks,) float32 hardness scores
        """返回每个 chunk 的平均一步预测误差。"""
        obs_starts = np.asarray(obs_starts, dtype=np.int64)
        num_chunks = len(obs_starts)

        flat_states:  list[np.ndarray] = []
        flat_actions: list[np.ndarray] = []
        flat_targets: list[np.ndarray] = []

        for obs_start in obs_starts.tolist():
            for t in range(obs_start, obs_start + pred_horizon):
                flat_states.append(state[t])
                flat_actions.append(action[t])
                flat_targets.append(state[t + 1])

        flat_states  = np.stack(flat_states,  axis=0).astype(np.float32)
        flat_actions = np.stack(flat_actions, axis=0).astype(np.float32)
        flat_targets = np.stack(flat_targets, axis=0).astype(np.float32)

        flat_states_n  = apply_standardizer(flat_states,  self._state_mean,  self._state_std)
        flat_actions_n = apply_standardizer(flat_actions, self._action_mean, self._action_std)

        preds_norm: list[np.ndarray] = []
        M = flat_states_n.shape[0]
        with torch.no_grad():
            for start in range(0, M, batch_size):
                s_b = torch.from_numpy(flat_states_n[start : start + batch_size]).to(self._device)
                a_b = torch.from_numpy(flat_actions_n[start : start + batch_size]).to(self._device)
                pred_n = self._model(s_b, a_b).cpu().numpy()
                preds_norm.append(pred_n)
        preds_norm = np.concatenate(preds_norm, axis=0)

        preds = preds_norm * self._target_std + self._target_mean

        errors = np.linalg.norm(preds - flat_targets, axis=-1)

        errors_chunked = errors.reshape(num_chunks, pred_horizon)
        hardness = errors_chunked.mean(axis=-1).astype(np.float32)
        return hardness

    @staticmethod
    def normalise_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """把分数压到 [0, 1]。"""
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < eps:
            return np.zeros_like(scores)
        return ((scores - lo) / (hi - lo)).astype(np.float32)
