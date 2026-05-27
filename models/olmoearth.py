"""
Wrapper classes for olmo-earth finetuning
"""
from torch import nn
from utils import *
from rslearn.train.model_context import ModelContext

class OlmoEarthClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_channels = img_channels

        # Channel adapter if not taking 12-channel input that olmo-earth expects
        self.channel_adapter = channel_adapter(12, self.img_channels)

        self.embed_dim = 192        # hardcoded for olmoearth tiny

        # Classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _create_model_context(self, x):
        """Create ModelContext from tensor"""
        B, C, H, W = x.shape

        # Adapt channels
        x = self.channel_adapter(x)  # [B, 12, H, W]

        # Create inputs
        inputs = {'sentinel2_l2a': x}

        # Create dummy metadata (one per sample)
        metadatas = [
            {
                'time': 0,
                'bounds': (0, 0, W, H),
                'crs': None,
            }
            for _ in range(B)
        ]

        # Create ModelContext
        context = ModelContext(inputs=inputs, metadatas=metadatas)

        return context



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