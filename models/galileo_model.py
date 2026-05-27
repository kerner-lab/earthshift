"""
Wrapper for galileo encoder finetuning
"""

from utils import *
import torch.nn.functional as F


class GalileoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.embed_dim = backbone.embedding_size

        # Hardcoding to use the S2_RGB embedding group
        self.embed_group_name = "S2_RGB"
        self.embed_layer = backbone.space_time_embed[self.embed_group_name]

        # Channel adapter if not taking 3-channel that S2_RGB embedding group
        self.input_adapter = channel_adapter(3, self.img_channels)

        # Classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)


    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, C, H, W] (C is adapted if not 4 channels)
        Returns:
            Class logits [B, num_classes]
        """
        # Adapt the input channels if needed
        x = self.input_adapter(x)

        # FlexiPatchEmbed expects channels last so need to re-arrange
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C_group]

        # Forward through the embedding layer
        tokens = self.embed_layer(x)

        # Flatten spatial tokens and average to get a single embedding
        tokens_flat = tokens.flatten(1, 2)
        embeddings = tokens_flat.mean(dim=1)  # [B, embedding_dim]

        # Classify on embeddings
        logits = self.classifier(embeddings)

        return logits


class GalileoSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.embed_dim = backbone.embedding_size

        # Hardcoding to use the S2_RGB embedding group
        self.embed_group_name = "S2_RGB"
        self.embed_layer = backbone.space_time_embed[self.embed_group_name]

        # Channel adapter if not taking 3-channel that S2_RGB embedding group
        self.input_adapter = channel_adapter(3, self.img_channels)

        # Segmentation head
        self.decoder = SimpleDecoder(self.embed_dim, nc=num_classes)

    def forward(self, x):
        """
        Forward pass
        :param x:
        :return: logits
        """

        B, C, H, W = x.shape

        # Adapt if needed
        x = self.input_adapter(x)

        # FlexiPatchEmbed expects channels last so need to re-arrange
        x = x.permute(0, 2, 3, 1)

        # Forward through the embedding layer
        tokens = self.embed_layer(x)

        # Convert tokens back to [B, embedding_dim, H_p, W_p] for decoder
        features = tokens.permute(0, 3, 1, 2)

        # Decode to segmentation map
        logits = self.decoder(features)

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return logits











