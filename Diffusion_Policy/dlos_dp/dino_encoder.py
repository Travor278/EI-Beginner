"""Frozen DINOv2 encoder used by DLOS-DP."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T


class FrozenDINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 image encoder."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        img_size: int = 224,
        repo_path: str | None = None,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()
        default_repo = os.environ.get("DLOS_DINO_REPO", "/mnt/d/Code/Learning/EI/_cache/dinov2")
        default_weights = os.environ.get(
            "DLOS_DINO_WEIGHTS",
            "/mnt/d/Code/Learning/EI/_cache/dinov2_vits14_pretrain.pth",
        )
        repo_path = repo_path or default_repo
        weights_path = weights_path or default_weights

        repo = Path(repo_path)
        weights = Path(weights_path)
        if repo.exists():
            hub_kwargs = {
                "repo_or_dir": str(repo),
                "model": model_name,
                "source": "local",
                "verbose": False,
            }
            if weights.exists():
                hub_kwargs["weights"] = str(weights)
                hub_kwargs["pretrained"] = True
            self.model = torch.hub.load(**hub_kwargs)
        else:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                verbose=False,
            )
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

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
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        img = self.transform(img)
        feat = self.model.forward_features(img)
        return feat["x_norm_clstoken"]
