"""
Wrapper for CLIP classification and segmentation
"""
import torch
import torch.nn.functional as F
from utils import *


class CLIPClassifier(nn.Module):
    def __init__(self, backbone, img_channels: int, num_classes: int):
        """
        CLIP classification wrapper with channel adapter for multi-spectral input.
        :param backbone: CLIP model (from clip.load())
        :param img_channels: Number of input channels in the dataset
        :param num_classes: Number of output classes
        """
        super().__init__()
        self.img_channels = img_channels

        self.channel_adapter = channel_adapter(3, img_channels)

        embed_dim = backbone.visual.output_dim
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.channel_adapter(x)
        features = self.backbone.encode_image(x)
        return self.classifier(features.float())


class CLIPSegmentation(nn.Module):
    def __init__(self, backbone, img_channels: int, num_classes: int):
        """
        CLIP segmentation wrapper using ViT patch tokens as spatial features.
        :param backbone: CLIP model (from clip.load())
        :param img_channels: Number of input channels in the dataset
        :param num_classes: Number of output classes
        """
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes

        self.channel_adapter = channel_adapter(3, img_channels)

        self.visual = backbone.visual
        self.patch_size = self.visual.conv1.kernel_size[0]
        self.embed_dim = self.visual.conv1.out_channels

        self.decoder = SimpleDecoder(self.embed_dim, nc=num_classes)

    def _extract_patch_features(self, x):
        v = self.visual

        # Patch embedding: [B, width, grid_h, grid_w]
        x = v.conv1(x)
        B, width, grid_h, grid_w = x.shape

        # Flatten spatial dims: [B, grid^2, width]
        x = x.reshape(B, width, -1).permute(0, 2, 1)

        # Prepend CLS token
        cls = v.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+grid^2, width]

        # Positional embedding and pre-norm
        x = x + v.positional_embedding
        x = v.ln_pre(x)

        # Transformer expects [L, N, D]
        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 1+grid^2, width]

        # Drop CLS token, reshape to spatial grid
        patch_tokens = x[:, 1:, :].permute(0, 2, 1)  # [B, width, grid^2]
        patch_tokens = patch_tokens.reshape(B, width, grid_h, grid_w)

        return patch_tokens

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.channel_adapter(x)
        features = self._extract_patch_features(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits
