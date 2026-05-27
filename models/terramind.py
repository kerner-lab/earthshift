"""
Wrapper for terramind finetuning
"""

from utils import *
import torch.nn.functional as F

class TerramindClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int,
                 modality: str = "S2L2A"):
        """
        TerraMind classifier wrapper

        Args:
            backbone: TerraMind loaded from huggingface
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.modality = modality

        # Terramind embed dim
        self.embed_dim = 768

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
        # Terramind expects dict
        inputs = {self.modality: x}

        outputs = self.backbone(inputs)

        # using features from last block
        features = outputs[-1]

        # Global average pooling over patches
        features = features.mean(dim=1)

        # Classify
        logits = self.classifier(features)

        return logits

class TerramindSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int,
                 modality: str = "S2L2A"):
        """
        Terramind segmentation wrapper

        Args:
            backbone: Terramind loaded from huggingface
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.modality = modality

        self.embed_dim = 768

        # Segmentation decoder
        self.decoder = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation

        Args:
            x: Input image [B, C, H, W]
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape

        # Input dict expected by Terramind
        inputs = {self.modality: x}

        # Forward through backbone
        outputs = self.backbone(inputs)

        # Use last block
        features = outputs[-1]

        # Reshape to spatial: [B, 196, 768] -> [B, 768, 14, 14]
        B, N, D = features.shape
        H_p = W_p = int(N ** 0.5)

        features = features.transpose(1, 2).contiguous()
        features = features.view(B, D, H_p, W_p)

        # Decode
        logits = self.decoder(features)

        # Upsample to original size
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # [B, num_classes, H, W]

        return logits