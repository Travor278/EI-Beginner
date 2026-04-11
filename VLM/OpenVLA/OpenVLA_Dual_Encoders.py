import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNormalize(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("Expected image tensor shape: (B, C, H, W).")
        return (images - self.mean) / self.std


class ImageResize(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = image_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("Expected image tensor shape: (B, C, H, W).")

        if images.shape[-2:] == (self.image_size, self.image_size):
            return images

        return F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


class DINOImageTransform(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.resize = ImageResize(image_size=image_size)
        self.normalize = ImageNormalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.resize(images)
        return self.normalize(images)


class SigLIPImageTransform(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.resize = ImageResize(image_size=image_size)
        self.normalize = ImageNormalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.resize(images)
        return self.normalize(images)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        d_model: int = 1024,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Conv2d with kernel=stride=patch_size works as patchify + linear projection.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Expected image tensor shape: (B, C, H, W).")

        _, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected image size ({self.image_size}, {self.image_size}), "
                f"but got ({height}, {width})."
            )

        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard encoder block: self-attention -> residual -> FFN -> residual.
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


class VisionEncoder(nn.Module):
    """
    A single ViT-like vision branch.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        d_ff: int = 4096,
        dropout: float = 0.1,
        use_cls_token: bool = False,
        feature_layer: str = "last",
        return_attn_weights: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.feature_layer = feature_layer
        self.return_attn_weights = return_attn_weights

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )
        self.num_patches = self.patch_embed.num_patches
        self.seq_len = self.num_patches + (1 if use_cls_token else 0)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                VisionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        fan_in = (
            self.patch_embed.proj.in_channels
            * self.patch_embed.proj.kernel_size[0]
            * self.patch_embed.proj.kernel_size[1]
        )
        nn.init.normal_(self.patch_embed.proj.weight, std=math.sqrt(1.0 / fan_in))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Turn image into patch tokens for one vision branch.
        x = self.patch_embed(images)
        batch_size = x.size(0)

        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls, x), dim=1)

        x = x + self.pos_embedding
        x = self.embed_dropout(x)

        all_attn_weights = []
        selected_x = None
        for layer_idx, layer in enumerate(self.layers):
            x, attn_weights = layer(x)
            if self.return_attn_weights:
                all_attn_weights.append(attn_weights)
            if self.feature_layer == "second_last" and layer_idx == len(self.layers) - 2:
                selected_x = x

        if self.feature_layer == "second_last" and selected_x is not None:
            x = selected_x

        x = self.final_norm(x)
        return x, all_attn_weights


class DINOEncoder(nn.Module):
    """
    DINOv2-like visual branch.
    Tends to preserve spatial structure, geometry, and local visual details.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        d_ff: int = 4096,
        dropout: float = 0.1,
        feature_layer: str = "last",
        return_attn_weights: bool = True,
    ) -> None:
        super().__init__()
        self.branch_name = "dino"

        # DINO-like branch: stronger on spatial structure, geometry,
        # and local visual detail modeling.
        # DINOv2-style branch usually uses a branch-specific image transform.
        self.preprocess = DINOImageTransform(image_size=image_size)
        self.backbone = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_cls_token=False,
            feature_layer=feature_layer,
            return_attn_weights=return_attn_weights,
        )

    @property
    def num_patches(self) -> int:
        return self.backbone.num_patches

    @property
    def d_model(self) -> int:
        return self.backbone.d_model

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # DINO branch: normalize -> ViT backbone.
        images = self.preprocess(images)
        return self.backbone(images)


class SigLIPEncoder(nn.Module):
    """
    SigLIP-like visual branch.
    Tends to be stronger at vision-language semantic alignment and object semantics.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        d_ff: int = 4096,
        dropout: float = 0.1,
        feature_layer: str = "last",
        return_attn_weights: bool = True,
    ) -> None:
        super().__init__()
        self.branch_name = "siglip"

        # SigLIP-like branch: stronger on vision-language semantic alignment and object-level semantics.
        # This is per-channel image normalization, not LayerNorm.
        self.preprocess = SigLIPImageTransform(image_size=image_size)
        self.backbone = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_cls_token=False,
            feature_layer=feature_layer,
            return_attn_weights=return_attn_weights,
        )

    @property
    def num_patches(self) -> int:
        return self.backbone.num_patches

    @property
    def d_model(self) -> int:
        return self.backbone.d_model

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # SigLIP branch: normalize -> ViT backbone.
        images = self.preprocess(images)
        return self.backbone(images)


class FusedProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        hidden_dim = vision_dim * 2
        # Map concatenated visual features into the LLM hidden space.
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class OpenVLADualEncoder(nn.Module):
    """
    A small structural demo of:
    DINOv2 branch + SigLIP branch + feature concat + projector.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dino_dim: int = 768,
        siglip_dim: int = 768,
        llm_dim: int = 4096,
        dino_num_heads: int = 12,
        dino_num_layers: int = 6,
        dino_d_ff: int = 3072,
        siglip_num_heads: int = 12,
        siglip_num_layers: int = 6,
        siglip_d_ff: int = 3072,
        dropout: float = 0.1,
        feature_layer: str = "last",
        return_attn_weights: bool = True,
    ) -> None:
        super().__init__()

        # In real OpenVLA, these two branches come from different pretrained backbones.
        self.dino_encoder = DINOEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=dino_dim,
            num_heads=dino_num_heads,
            num_layers=dino_num_layers,
            d_ff=dino_d_ff,
            dropout=dropout,
            feature_layer=feature_layer,
            return_attn_weights=return_attn_weights,
        )
        self.siglip_encoder = SigLIPEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            d_model=siglip_dim,
            num_heads=siglip_num_heads,
            num_layers=siglip_num_layers,
            d_ff=siglip_d_ff,
            dropout=dropout,
            feature_layer=feature_layer,
            return_attn_weights=return_attn_weights,
        )
        self.projector = FusedProjector(
            vision_dim=dino_dim + siglip_dim,
            llm_dim=llm_dim,
        )

    def freeze_backbones(self) -> None:
        self.dino_encoder.requires_grad_(False)
        self.siglip_encoder.requires_grad_(False)

    def unfreeze_backbones(self) -> None:
        self.dino_encoder.requires_grad_(True)
        self.siglip_encoder.requires_grad_(True)

    def freeze_projector(self) -> None:
        self.projector.requires_grad_(False)

    def unfreeze_projector(self) -> None:
        self.projector.requires_grad_(True)

    def forward(
        self,
        dino_images: torch.Tensor,
        siglip_images: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        # Run the same scene through two different visual branches.
        # Each branch keeps its own normalization and backbone.
        dino_tokens, dino_attn = self.dino_encoder(dino_images)
        siglip_tokens, siglip_attn = self.siglip_encoder(siglip_images)

        if dino_tokens.shape[:2] != siglip_tokens.shape[:2]:
            # In a more realistic setup, we could align token grids here:
            # 1. pool one branch to the other's patch count
            # 2. add a learnable resampler
            # 3. interpolate 2D token maps before flattening back to 1D tokens
            raise ValueError("The two encoders must produce the same patch grid.")

        # Concatenate along the channel dimension, then project to LLM width.
        fused_tokens = torch.cat([dino_tokens, siglip_tokens], dim=-1)
        llm_ready_tokens = self.projector(fused_tokens)

        return {
            "dino_tokens": dino_tokens,
            "siglip_tokens": siglip_tokens,
            "fused_tokens": fused_tokens,
            "llm_ready_tokens": llm_ready_tokens,
            "dino_attn": dino_attn,
            "siglip_attn": siglip_attn,
        }

    def forward_same_image(self, images: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        # Handy helper for demos where both branches read the same image tensor.
        return self.forward(dino_images=images, siglip_images=images)


def demo_dual_encoder_shapes() -> None:
    batch_size = 2
    image_size = 224
    patch_size = 14
    num_patches = (image_size // patch_size) ** 2

    images = torch.randn(batch_size, 3, image_size, image_size)

    model = OpenVLADualEncoder(
        image_size=image_size,
        patch_size=patch_size,
        dino_dim=512,
        siglip_dim=512,
        llm_dim=1024,
        dino_num_heads=8,
        dino_num_layers=3,
        dino_d_ff=2048,
        siglip_num_heads=8,
        siglip_num_layers=4,
        siglip_d_ff=2048,
        dropout=0.0,
        feature_layer="second_last",
        return_attn_weights=True,
    )

    # Demo helper: here both branches read the same tensor.
    outputs = model.forward_same_image(images)

    print("images shape            :", images.shape)
    print("num patches             :", num_patches)
    print("dino tokens shape       :", outputs["dino_tokens"].shape)
    print("siglip tokens shape     :", outputs["siglip_tokens"].shape)
    print("fused tokens shape      :", outputs["fused_tokens"].shape)
    print("llm-ready tokens shape  :", outputs["llm_ready_tokens"].shape)
    print("dino layer1 attn shape  :", outputs["dino_attn"][0].shape)
    print("siglip layer1 attn shape:", outputs["siglip_attn"][0].shape)
    print()
    print("shape flow:")
    print(f"(B, 3, {image_size}, {image_size})")
    print(f"-> (B, {num_patches}, 512)  [DINO]")
    print(f"-> (B, {num_patches}, 512)  [SigLIP]")
    print(f"-> (B, {num_patches}, 1024) [concat]")
    print(f"-> (B, {num_patches}, 1024) [projector -> LLM width]")


if __name__ == "__main__":
    demo_dual_encoder_shapes()
