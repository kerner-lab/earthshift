"""
Prithvi wrappers for finetuning
"""

import torch
from utils import *

class PrithviClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        Prithvi classifier wrapper

        Args:
            backbone: prtihvi backbone loaded from terratorch
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels

        # Channel adapter if not taking 6-channel input that prithvi expects
        self.channel_adapter = channel_adapter(6, self.img_channels)

        self.embed_dim = 1024        # hardcoded for prithvi 300M

        # Classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Class logits [B, num_classes]
        """
        # Adapt channels
        x = self.channel_adapter(x)

        # Get features from backbone
        features = self.backbone(x)

        # Get last layers features
        features = features[-1]

        # Extract CLS token (first token)
        cls_token = features[:, 0]

        # Classify
        logits = self.classifier(cls_token)  # [B, num_classes]

        return logits


class PrithviSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        Prithvi segmentation wrapper
        Args:
            backbone: prtihvi backbone loaded from terratorch
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels

        # Channel adapter if not taking 6-channel input that prithvi expects
        self.input_adapter = channel_adapter(6, self.img_channels)

        self.embed_dim = 1024        # hardcoded for prithvi 300M

        # Segmentation decoder head
        self.decoder = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # Adapt channels if needed from input
        x = self.input_adapter(x)

        # Get features from backbone
        features = self.backbone(x)

        # Take last layer features
        features = features[-1]

        # Remove CLS token (keep only patch tokens)
        patch_tokens = features[:, 1:]

        # Reshape to spatial format
        # 196 patches = 14×14 grid (for 224×224 input with 16×16 patches)
        grid_size = int(patch_tokens.shape[1] ** 0.5)  # 14

        # Reshape: [B, 196, 192] → [B, 192, 14, 14]
        spatial_features = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, grid_size, grid_size
        )

        # Decode
        logits = self.decoder(spatial_features)  # [B, num_classes, 14, 14]

        # Upsample to original size
        logits = torch.nn.functional.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return logits
