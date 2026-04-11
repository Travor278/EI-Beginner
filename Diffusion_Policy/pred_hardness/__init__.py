"""
PHRew-DP: Predictive Hardness Reweighting for Diffusion Policy (v3.1).

Core idea: a lightweight state-space forward predictor scores each training
*chunk* (obs_start + pred_horizon window) by its mean one-step prediction
error.  These scores are used as continuous sampling weights via
WeightedRandomSampler — harder chunks are seen more often, giving the policy
a richer learning signal without changing the model or the loss.

Key distinction from existing scripts/stage2-4:
  - stage2 (existing): episode-level scoring (one score per demo)
  - stage3 (existing): Hydra-based curriculum ordering only
  - pred_hardness/  :  chunk-level scoring + WeightedRandomSampler

Package layout
--------------
config.py              -- PHRConfig dataclass (all hyper-parameters)
chunk_scorer.py        -- ChunkScorer: probe → per-chunk hardness array
dataset.py             -- PushTImagePHRDataset (image obs, weight interface)
stage2b_score_chunks.py -- CLI: chunk-level scoring → npy
stage3_train_phrew.py   -- CLI: DP training with WeightedRandomSampler
stage4_generate_cmds.py -- CLI: generate all comparison runs
"""

from .chunk_scorer import ChunkScorer
from .config import PHRConfig
from .dataset import PushTImagePHRDataset

__all__ = [
    "PHRConfig",
    "ChunkScorer",
    "PushTImagePHRDataset",
]
