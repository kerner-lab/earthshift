"""
TerraFM wrappers for finetuning
"""

import torch
from utils import *
import torch.nn.functional as F


class TerrafmClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        TerraFM classifier wrapper

        Args:
            backbone: TerraFM loaded from huggingface
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels

        # Channel adapter if not taking 12-channel input that terrafm expects
        self.input_adapter = channel_adapter(12, self.img_channels)

        self.embed_dim = 768        # hardcoded for terrafm model

        # Classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, C, H, W] (C is adapted if not 12 channels)
        Returns:
            Class logits [B, num_classes]
        """
        # Adapt the input channels if needed
        x = self.input_adapter(x)

        # TerraFM forward returns CLS token features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)
        return logits


class TerrafmSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        TerraFM segmentation wrapper

        Args:
            backbone: TerraFM loaded from huggingface
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels

        # Channel adapter if not taking 12-channel input that terrafm expects
        self.input_adapter = channel_adapter(12, self.img_channels)

        self.embed_dim = 768  # hardcoded for terrafm model

        # Segmentation decoder
        self.decoder = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Adapt channels if needed
        x = self.input_adapter(x)

        # Extract spatial features from last layer
        features = self.backbone.extract_feature(
            x,
            return_h_w=True,
            out_indices=[11]  # Last transformer block
        )

        # Get the feature map
        features = features[-1]  # [B, 768, 14, 14]

        # Decode to segmentation map
        logits = self.decoder(features)  # [B, 768, 14, 14] → [B, num_classes, 14, 14]

        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # [B, num_classes, H, W]

        return logits







































