"""
FrozenDINOv2Encoder — wraps a frozen DINOv2 ViT-S/14 from torch.hub.

Usage
-----
    encoder = FrozenDINOv2Encoder()          # loads dinov2_vits14
    z = encoder(img_batch)                   # (B, 384)

Input contract
--------------
- img: torch.Tensor of shape (B, 3, H, W)
- dtype: float32 with values in [0, 1]  OR  uint8 with values in [0, 255].
  Both are handled automatically.
- Any spatial resolution is accepted; the transform resizes to 224×224.

Output
------
- (B, 384) CLS token embedding (no gradient, model is frozen).

First-run note
--------------
torch.hub.load downloads ~300 MB the first time; weights are cached under
~/.cache/torch/hub.  Use torch.hub.set_dir() to override the cache location.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T


class FrozenDINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 image encoder.

    All parameters are frozen at construction time.  The forward pass is
    decorated with @torch.no_grad() so it never accumulates gradients or
    affects the autograd graph — safe to call inside compute_loss.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        img_size: int = 224,
    ) -> None:
        super().__init__()
        # Load DINOv2 from facebookresearch hub
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            verbose=False,
        )
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Spatial resize + ImageNet normalisation (DINOv2 was trained with these)
        self.transform = T.Compose([
            T.Resize(
                (img_size, img_size),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img : (B, 3, H, W) — float32 [0,1] or uint8 [0,255]

        Returns
        -------
        (B, 384) L2-normalised CLS token, float32, on the same device as img.
        """
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = self.transform(img)
        # Use forward_features() which returns a dict; extract the L2-normalised
        # CLS token under 'x_norm_clstoken'.  This is the canonical way to get
        # features from the facebookresearch/dinov2 torch.hub models and avoids
        # any ambiguity about the plain __call__ return type.
        feat = self.model.forward_features(img)
        return feat["x_norm_clstoken"]  # (B, 384)
