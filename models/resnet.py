"""
Wrapper for resnet classification and segmentation
"""
import torch
import torch.nn.functional as F
from utils import *


class ResNetClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        ResNet classification wrapper with a channel adapter for multi-spectral input.
        :param backbone: ResNet model with the original fc layer intact
        :param img_channels: Number of input channels in the dataset
        :param num_classes: Number of output classes
        """
        super().__init__()
        self.img_channels = img_channels

        # Adapt dataset channels → 3 (ResNet backbone expects 3-channel input)
        self.channel_adapter = channel_adapter(3, img_channels)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        x = self.channel_adapter(x)
        return self.backbone(x)


class ResenetSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        Resenet segmentation wrapper for semseg tasks
        :param backbone:
        :param num_classes:
        :param img_channels:
        """
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes

        # adapt channels if needed
        self.channel_adapter = channel_adapter(3, self.img_channels)

        # Get backbone output channels
        if hasattr(backbone, 'fc'):
            self.backbone_out_channels = backbone.fc.in_features
        else:
            # Fallback: dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.backbone(dummy)
                self.backbone_out_channels = out.shape[1]

        # Remove avg pooling and fc layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Decoder head
        self.decoder = SimpleDecoder(self.backbone_out_channels, nc=num_classes)


    def forward(self, x):
        """
        Run forward pass for semseg tasks
        :param self:
        :param x:
        :return:
        """

        # Get shape from input
        B, C, H, W = x.shape

        # adapt channels if needed
        x = self.channel_adapter(x)

        # Get features from backbone model
        features = self.backbone(x)

        # Decode to seg map
        logits = self.decoder(features)

        # Upsample
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return logits

