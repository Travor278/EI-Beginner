"""
SFC-DP: SNR-Frequency Consistent Diffusion Policy (v3.6).

Core innovation
---------------
Standard Diffusion Policy implicitly relies on the fact that at high noise
levels (k ≈ T, ᾱ_k → 0) the denoising network can only reliably recover the
low-frequency components of the action sequence, while at low noise levels
(k ≈ 0, ᾱ_k → 1) all frequency bands are recoverable.  SFC-DP makes this
relationship *explicit* by adding a frequency-selective auxiliary loss:

    L_total = L_diff + λ · L_freq

    L_freq = E[||(DCT(x̂₀|k) − DCT(x₀)) · M(k)||²]

    M(k)[f] = 1  if  f / T_pred < ᾱ_k   (supervise only the bands the
             = 0  otherwise                model can resolve at step k)

Key properties
--------------
- Zero extra parameters (DCT has no trainable weights)
- Drop-in training loss addition — no architecture change
- Works with both image and state observations
- Fully compatible with the official DP codebase via sys.path import

Distinction from prior work
---------------------------
- FreqPolicy  : changes the *generation paradigm* (AR in freq domain)
- Wavelet Pol.: replaces the action *representation* + adds decoders
- DLOS-DP     : constrains x̂₀|k via a *world model* (obs consistency)
- Blurring Diff.: changes the *forward process*, not the loss
- Min-SNR     : scalar timestep weighting, not frequency-band selection

Package layout
--------------
config.py               -- SFCConfig dataclass
sfc_loss.py             -- compute_x0_hat(), sfc_loss(), DCT utilities
train_sfc.py            -- training script (groups A–E, official DP integration)
stage1_generate_cmds.py -- generate 5×2=10 ablation run commands
"""

from .config import SFCConfig
from .sfc_loss import SFCLoss, compute_x0_hat, sfc_loss

__all__ = [
    "SFCConfig",
    "SFCLoss",
    "compute_x0_hat",
    "sfc_loss",
]
