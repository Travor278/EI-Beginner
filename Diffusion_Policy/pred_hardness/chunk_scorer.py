"""
ChunkScorer — probe-based chunk-level hardness scorer.

Given the state-space forward probe trained in Stage 1
(scripts/stage1_train_state_forward_probe.py) and a list of chunk start
indices, this module computes a **per-chunk** predictive hardness score:

    hardness(obs_start) = mean_{t=obs_start}^{obs_start + pred_horizon - 1}
                              || probe(s_t, a_t) − s_{t+1} ||₂

This is finer-grained than the existing stage2 script which computes a single
score per episode.  Each DP training sample (chunk) gets its own score,
enabling WeightedRandomSampler to focus on harder chunks at the transition
level rather than the demo level.

The scoring uses batched inference for speed: all (state_t, action_t) pairs
across all chunks are stacked into a single matrix and run through the probe
in one forward pass (or in mini-batches for GPU memory efficiency).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Allow import from scripts/ when run directly
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from predictive_hardness_common import ForwardDynamicsMLP, apply_standardizer


class ChunkScorer:
    """
    Loads a trained ForwardDynamicsMLP checkpoint and scores training chunks.

    Parameters
    ----------
    checkpoint_path : path to best_state_forward_probe.pt
    device          : 'auto' | 'cuda' | 'cpu'
    """

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

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def score_chunks(
        self,
        state: np.ndarray,           # (N_frames, state_dim) float32
        action: np.ndarray,          # (N_frames, action_dim) float32
        obs_starts: list[int] | np.ndarray,  # chunk start indices
        pred_horizon: int,
        batch_size: int = 512,
    ) -> np.ndarray:                 # (num_chunks,) float32 hardness scores
        """
        Compute mean one-step L2 prediction error for every chunk.

        For each chunk index i and window step t in [obs_start, obs_start + pred_horizon):
            error_t = || probe(s_t, a_t) - s_{t+1} ||_2

        hardness_i = mean(error_t for t in window)

        Returns
        -------
        np.ndarray of shape (len(obs_starts),) with float32 hardness values.
        Values are NOT normalised here; call `normalise_scores()` afterwards
        if you need [0, 1] range.
        """
        obs_starts = np.asarray(obs_starts, dtype=np.int64)
        num_chunks = len(obs_starts)

        # Build flat (state, action) pairs for all (chunk, step) combinations
        # Shape: (num_chunks * pred_horizon, state_dim + action_dim)
        flat_states:  list[np.ndarray] = []
        flat_actions: list[np.ndarray] = []
        flat_targets: list[np.ndarray] = []

        for obs_start in obs_starts.tolist():
            for t in range(obs_start, obs_start + pred_horizon):
                flat_states.append(state[t])
                flat_actions.append(action[t])
                flat_targets.append(state[t + 1])

        flat_states  = np.stack(flat_states,  axis=0).astype(np.float32)  # (M, sd)
        flat_actions = np.stack(flat_actions, axis=0).astype(np.float32)  # (M, ad)
        flat_targets = np.stack(flat_targets, axis=0).astype(np.float32)  # (M, sd)

        # Normalise inputs
        flat_states_n  = apply_standardizer(flat_states,  self._state_mean,  self._state_std)
        flat_actions_n = apply_standardizer(flat_actions, self._action_mean, self._action_std)

        # Batched inference
        preds_norm: list[np.ndarray] = []
        M = flat_states_n.shape[0]
        with torch.no_grad():
            for start in range(0, M, batch_size):
                s_b = torch.from_numpy(flat_states_n[start : start + batch_size]).to(self._device)
                a_b = torch.from_numpy(flat_actions_n[start : start + batch_size]).to(self._device)
                pred_n = self._model(s_b, a_b).cpu().numpy()
                preds_norm.append(pred_n)
        preds_norm = np.concatenate(preds_norm, axis=0)  # (M, sd)

        # Denormalise predictions back to original state space
        preds = preds_norm * self._target_std + self._target_mean  # (M, sd)

        # L2 error per step
        errors = np.linalg.norm(preds - flat_targets, axis=-1)  # (M,)

        # Reshape to (num_chunks, pred_horizon) and take mean over horizon
        errors_chunked = errors.reshape(num_chunks, pred_horizon)  # (C, H)
        hardness = errors_chunked.mean(axis=-1).astype(np.float32) # (C,)
        return hardness

    @staticmethod
    def normalise_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Min-max normalise scores to [0, 1]."""
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < eps:
            return np.zeros_like(scores)
        return ((scores - lo) / (hi - lo)).astype(np.float32)
