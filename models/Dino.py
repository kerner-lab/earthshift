"""
Dino class wrappers
"""
from utils import *
import torch.nn.functional as F


class Dinov3Classifier(nn.Module):
    def __init__(self, backbone_model, num_classes, img_channels):
        """
        DINOv2 classification wrapper with channel adapter for multi-spectral input.
        :param backbone_model: timm DINOv2 model (no features_only) with num_classes=0 to expose embeddings
        :param num_classes: Number of output classes
        :param img_channels: Number of input channels in the dataset
        """
        super(Dinov3Classifier, self).__init__()
        self.backbone_model = backbone_model
        self.img_channels = img_channels

        self.channel_adapter = channel_adapter(3, self.img_channels)

        if hasattr(self.backbone_model, "num_features"):
            embed_dim = self.backbone_model.num_features
        else:
            raise ValueError("Expected a timm backbone with num_features attribute.")

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.channel_adapter(x)
        features = self.backbone_model(x)
        return self.classifier(features)


class Dinov3Segmentation(nn.Module):
    def __init__(self, backbone_model, num_classes, img_channels):
        super(Dinov3Segmentation, self).__init__()
        self.backbone_model = backbone_model
        self.num_classes = num_classes
        self.img_channels = img_channels

        # Channel adapter if not taking 3-channel input that dino expects
        self.channel_adapter = channel_adapter(3, self.img_channels)

        # infer channels of last feature map
        if hasattr(self.backbone_model, "feature_info"):
            in_channels = self.backbone_model.feature_info.channels()[-1]
        else:
            raise ValueError("Expected a timm features_only backbone (missing feature_info).")

        self.decode_head = SimpleDecoder(in_channels=in_channels, nc=num_classes)

    def forward(self, x):
        # Adapt channels
        x = self.channel_adapter(x)
        features = self.backbone_model(x)
        f = features[-1]
        logits = self.decode_head(f)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits