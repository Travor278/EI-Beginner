"""
DLOS-DP: Denoising-Level Outcome Supervision for Diffusion Policy.

Core innovation: at each noise level k during training, the intermediate
clean-action estimate x̂₀|k is fed into a lightweight outcome world model to
enforce action-conditioned observation consistency.  Gradients flow back
through the denoising network ε_θ, providing an additional physics-grounded
learning signal beyond the standard diffusion objective.

Package layout
--------------
config.py            -- DLOSConfig dataclass (all hyper-parameters)
world_model.py       -- OutcomeWorldModel + KoopmanWorldModel
dino_encoder.py      -- FrozenDINOv2Encoder (frozen ViT-S/14)
dlos_loss.py         -- compute_x0_hat() + DLOSLoss (5 ablation groups)
dataset.py           -- PushTImageDLOSDataset (zarr, with obs_next)
stage0_wm_probe.py   -- Stage 0: stand-alone WM learnability check
train_dlos.py        -- Stage 1 training script (groups A/B/C/D/E + seed)
stage1_generate_cmds.py -- Generate 5×2=10 training commands
"""

from .config import DLOSConfig
from .dataset import PushTImageDLOSDataset
from .dino_encoder import FrozenDINOv2Encoder
from .dlos_loss import DLOSLoss, compute_x0_hat
from .world_model import KoopmanWorldModel, OutcomeWorldModel

__all__ = [
    "DLOSConfig",
    "PushTImageDLOSDataset",
    "FrozenDINOv2Encoder",
    "DLOSLoss",
    "compute_x0_hat",
    "OutcomeWorldModel",
    "KoopmanWorldModel",
]
