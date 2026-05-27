"""
Wrapper for Clay embedding model
"""

import torch
from utils import *
import torch.nn.functional as F


class ClayClassifier(nn.Module):
    def __init__(self, backbone, img_channels, num_classes, waves, pool="mean"):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.pool = pool
        self.image_size = 256

        # Store waves as list to avoid tensor copy warning
        if isinstance(waves, torch.Tensor):
            self._waves_list = waves.tolist()
        elif isinstance(waves, (list, tuple)):
            self._waves_list = list(waves)
        else:
            raise TypeError(f"waves must be list, tuple, or tensor, got {type(waves)}")

        # Embedding dimension
        self.embed_dim = 768

        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _get_waves_tensor(self, device):
        """Convert waves list to tensor on specified device"""
        return torch.tensor(self._waves_list, dtype=torch.float32, device=device)


    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D] or [B, D]
        """
        if x.dim() == 3:
            if self.pool == "cls":
                return x[:, 0, :]
            elif self.pool == "mean":
                return x.mean(dim=1)
            else:
                raise ValueError(f"Invalid pool mode: {self.pool}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        waves_tensor = self._get_waves_tensor(x.device)
        datacube = self.backbone.datacuber(x, waves=waves_tensor)
        outs = self.backbone.clay_encoder(datacube)
        last = outs[-1]
        feats = self._pool(last)
        return self.classifier(feats)


class ClaySegmentation(nn.Module):
    def __init__(self, backbone, img_channels, num_classes, waves, image_size=256):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.image_size = image_size

        # Store waves as list to avoid tensor copy warning
        if isinstance(waves, torch.Tensor):
            self._waves_list = waves.tolist()
        elif isinstance(waves, (list, tuple)):
            self._waves_list = list(waves)
        else:
            raise TypeError(f"waves must be list, tuple, or tensor, got {type(waves)}")

        # Validate waves matches img_channels
        if len(self._waves_list) != img_channels:
            raise ValueError(
                f"Length of waves ({len(self._waves_list)}) must match img_channels ({img_channels})"
            )

        # Embedding and grid size from input (static)
        self.embed_dim = 768
        self.grid_size = (32,32)

        # Segmentation decoder head
        self.decode_head = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)

    def _get_waves_tensor(self, device):
        """Convert waves list to tensor on specified device"""
        return torch.tensor(self._waves_list, dtype=torch.float32, device=device)

    def _extract_spatial_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Convert Clay token output [B, N, D] to spatial [B, D, H', W']
        """
        if feats.dim() != 3:
            raise RuntimeError(f"Expected [B, N, D], got {feats.shape}")

        B, N, D = feats.shape
        num_patches = N - 1
        gh, gw = self.grid_size
        expected = gh * gw

        feats = feats[:, 1:, :]

        if feats.shape[1] != expected:
            raise RuntimeError(
                f"Token count {feats.shape[1]} doesn't match grid {gh}x{gw}={expected}"
            )

        feats = feats.transpose(1, 2).contiguous().view(B, D, gh, gw)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation
        """
        waves_tensor = self._get_waves_tensor(x.device)
        datacube = self.backbone.datacuber(x, waves=waves_tensor)

        outs = self.backbone.clay_encoder(datacube)
        if not outs:
            raise RuntimeError("Clay encoder returned empty output")
        last = outs[-1]

        feats = self._extract_spatial_features(last)
        logits = self.decode_head(feats)

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return logits