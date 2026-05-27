"""
Heads for VIT models
"""

import torch.nn.functional as F
from utils import *


class VITClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        Torchvision ViT classification wrapper with channel adapter for multi-spectral input.
        Expects the backbone's classification head to already be replaced before passing in.
        :param backbone: ViT model with final head already set to num_classes outputs
        :param img_channels: Number of input channels in the dataset
        :param num_classes: Number of output classes
        """
        super().__init__()
        self.img_channels = img_channels
        self.channel_adapter = channel_adapter(3, img_channels)
        self.backbone = backbone

    def forward(self, x):
        x = self.channel_adapter(x)
        return self.backbone(x)


class VITSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        Vision Transformer segmentation wrapper for semseg tasks (torchvision)
        :param backbone: ViT model (e.g., vit_b_16)
        :param num_classes: Number of segmentation classes
        :param img_channels: Input image channels
        """
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes

        # Adapt channels if needed
        self.channel_adapter = channel_adapter(3, self.img_channels)

        # Get ViT configuration
        self.embed_dim = backbone.hidden_dim
        self.patch_size = backbone.patch_size
        self.image_size = backbone.image_size
        self.grid_size = self.image_size // self.patch_size

        # Store backbone components
        self.conv_proj = backbone.conv_proj
        self.encoder = backbone.encoder

        # Store class token and positional embeddings
        self.class_token = backbone.class_token
        self.encoder_pos_embedding = backbone.encoder.pos_embedding

        # Decoder head for segmentation
        self.decoder = SimpleDecoder(self.embed_dim, nc=num_classes)

    def forward(self, x):
        """
        Run forward pass for semseg tasks
        :param x: Input tensor [B, C, H, W]
        :return: Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape

        # Adapt channels if needed
        x = self.channel_adapter(x)

        # Patch embedding: [B, 3, H, W] -> [B, 768, 14, 14]
        x = self.conv_proj(x)

        # Reshape: [B, 768, 14, 14] -> [B, 196, 768]
        n, c, gh, gw = x.shape
        x = x.reshape(n, c, gh * gw).permute(0, 2, 1)

        # Add class token: [B, 196, 768] -> [B, 197, 768]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Encoder: [B, 197, 768] -> [B, 197, 768]
        x = self.encoder(x)

        # Remove CLS token, keep only patch tokens: [B, 197, 768] -> [B, 196, 768]
        patch_tokens = x[:, 1:, :]  # Skip first token (CLS)

        # Reshape to spatial format: [B, 196, 768] -> [B, 768, 14, 14]
        spatial_features = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, self.grid_size, self.grid_size
        )

        # Decode to segmentation map
        logits = self.decoder(spatial_features)

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return logits

class TimmVITSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int):
        """
        Timm Vision Transformer segmentation wrapper for semseg tasks
        :param backbone: Timm ViT model
        :param num_classes: Number of segmentation classes
        :param img_channels: Input image channels
        """
        super().__init__()
        self.img_channels = img_channels
        self.num_classes = num_classes

        # Adapt channels if needed
        self.channel_adapter = channel_adapter(3, self.img_channels)

        # Get ViT configuration
        self.embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_embed.patch_size[0]
        self.num_patches = backbone.patch_embed.num_patches
        self.grid_size = int(self.num_patches ** 0.5)

        # Store components
        self.patch_embed = backbone.patch_embed
        self.cls_token = backbone.cls_token
        self.pos_embed = backbone.pos_embed
        self.pos_drop = backbone.pos_drop
        self.norm_pre = backbone.norm_pre
        self.blocks = backbone.blocks
        self.norm = backbone.norm

        # Decoder head for segmentation
        self.decoder = SimpleDecoder(self.embed_dim, nc=num_classes)

    def forward(self, x):
        """
        Run forward pass for semseg tasks
        :param x: Input tensor [B, C, H, W]
        :return: Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape

        # Adapt channels if needed
        x = self.channel_adapter(x)

        # Patch embedding: [B, 3, H, W] -> [B, N, embed_dim]
        x = self.patch_embed(x)

        # Add class token: [B, N, embed_dim] -> [B, N+1, embed_dim]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Pre-norm (if exists)
        x = self.norm_pre(x)

        # Transformer blocks
        x = self.blocks(x)

        # Final norm
        x = self.norm(x)

        # Remove class token: [B, N+1, embed_dim] -> [B, N, embed_dim]
        x = x[:, 1:, :]

        # Reshape to spatial: [B, N, embed_dim] -> [B, embed_dim, grid_h, grid_w]
        x = x.transpose(1, 2).contiguous()  # [B, embed_dim, N]
        x = x.view(B, self.embed_dim, self.grid_size, self.grid_size)

        # Decode to segmentation map
        logits = self.decoder(x)

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return logits
