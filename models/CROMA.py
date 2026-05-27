"""
CROMA wrappers for finetuning
"""
import torch
from utils import *
import torch.nn.functional as F


class CROMAClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int,
                 modality: str = "optical", pool: str = "mean"):
        """
        CROMA Classifier for TorchGeo's CROMA

        Args:
            backbone: TorchGeo CROMA model initialized with specific modality
            img_channels: Number of input channels in your data
            num_classes: Number of classification classes
            modality: "sar" or "optical" (must match backbone initialization)
            pool: Pooling strategy - "mean" or "cls"
        """
        super().__init__()
        self.backbone = backbone
        self.modality = modality.lower()
        self.pool = pool
        self.img_channels = img_channels

        # Verify backbone has the right modality
        if self.modality == "sar" and not hasattr(self.backbone, 's1_encoder'):
            raise ValueError("Backbone doesn't have SAR encoder. Initialize with modalities=['sar']")
        if self.modality == "optical" and not hasattr(self.backbone, 's2_encoder'):
            raise ValueError("Backbone doesn't have optical encoder. Initialize with modalities=['optical']")

        # Infer expected backbone input channels
        backbone_in_chans = self._infer_backbone_in_chans()

        # Channel adapter: map dataset channels → backbone's expected channels
        self.input_adapter = channel_adapter(backbone_in_chans, self.img_channels)

        # Embedding dimension (CROMA uses 768)
        self.embed_dim = 768

        # Classifier head
        self.classifier = nn.Linear(self.embed_dim, num_classes)


    def _infer_backbone_in_chans(self) -> int:
        """
        TorchGeo CROMA expects:
        - SAR: 2 channels
        - Optical: 12 channels
        """
        if self.modality == "sar":
            return 2
        elif self.modality == "optical":
            return 12
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def _pool_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Pool token features into [B, D]

        CROMA outputs [B, N, D] where N = num_patches
        """
        if feats.dim() == 3:  # [B, N, D]
            if self.pool == "cls":
                # Use first token if it's a CLS token
                return feats[:, 0]
            else:
                # Mean pool over all patches
                return feats.mean(dim=1)
        elif feats.dim() == 2:  # [B, D] - already pooled
            return feats
        else:
            raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification

        Args:
            x: Input image [B, C, H, W]

        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # CRITICAL: Move backbone to input device before calling
        device = x.device
        self.backbone = self.backbone.to(device)

        # Force all buffers to input device
        for buffer in self.backbone.buffers():
            buffer.data = buffer.data.to(device)

        # Adapt channels
        x = self.input_adapter(x)

        # Call CROMA's forward with the appropriate modality
        if self.modality == "sar":
            outputs = self.backbone(x_sar=x)
            # Try different possible key names
            if 'sar_encodings' in outputs:
                feats = outputs['sar_encodings']
            elif 'SAR_encodings' in outputs:
                feats = outputs['SAR_encodings']
            else:
                raise RuntimeError(f"Could not find SAR encodings in output keys: {outputs.keys()}")

        elif self.modality == "optical":
            outputs = self.backbone(x_optical=x)
            # Try different possible key names
            if 'optical_encodings' in outputs:
                feats = outputs['optical_encodings']
            elif 'optical_GAP' in outputs:
                feats = outputs['optical_GAP']
            else:
                raise RuntimeError(f"Could not find optical encodings in output keys: {outputs.keys()}")
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        # Pool features
        feats = self._pool_feats(feats)  # [B, D]

        # Classify
        return self.classifier(feats)  # [B, num_classes]


class CROMASegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int,
                 modality: str = "optical", grid_size=(15, 15)):
        """
        CROMA Segmentation Head for TorchGeo's CROMA

        Args:
            backbone: TorchGeo CROMA model (from croma_base or croma_large)
            img_channels: Number of input channels in your data
            num_classes: Number of segmentation classes
            modality: "sar" or "optical"
            grid_size: Spatial grid (15x15 for 120x120 images with patch_size=8)
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.modality = modality.lower()
        self.img_channels = img_channels

        # Get expected channels
        backbone_in_chans = self._infer_backbone_in_chans()

        # Channel adapter: map dataset channels → backbone's expected channels
        self.input_adapter = channel_adapter(backbone_in_chans, self.img_channels)

        # Embedding dimension (CROMA base uses 768)
        self.embed_dim = 768

        # Segmentation decoder
        self.decode_head = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)


    def _infer_backbone_in_chans(self) -> int:
        """
        TorchGeo CROMA expects:
        - SAR (S1): 2 channels
        - Optical (S2): 12 channels
        """
        if self.modality == "sar":
            return 2
        elif self.modality == "optical":
            return 12
        else:
            raise ValueError(f"Unknown modality: {self.modality}. Choose 'sar' or 'optical'")

    def _extract_spatial_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Convert CROMA token output [B, N, D] to spatial [B, D, H', W']
        """
        if feats.dim() == 4:
            return feats

        if feats.dim() == 3:
            B, N, D = feats.shape
            gh, gw = self.grid_size
            expected = gh * gw

            if N != expected:
                raise RuntimeError(
                    f"Token count {N} doesn't match grid {gh}x{gw}={expected}"
                )

            # Reshape: [B, N, D] -> [B, D, H', W']
            feats = feats.transpose(1, 2).contiguous().view(B, D, gh, gw)
            return feats

        raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation

        Args:
            x: Input image [B, C, H, W]

        Returns:
            torch.Tensor: Segmentation logits [B, num_classes, H, W]
        """

        # CRITICAL: Move backbone to input device before calling
        device = x.device
        self.backbone = self.backbone.to(device)

        # Force all buffers to input device
        for buffer in self.backbone.buffers():
            buffer.data = buffer.data.to(device)

        # Adapt channels
        x = self.input_adapter(x)

        # Call CROMA's forward with the appropriate modality
        if self.modality == "sar":
            outputs = self.backbone(x_sar=x)
            # Try different possible key names
            if 'sar_encodings' in outputs:
                feats = outputs['sar_encodings']
            elif 'SAR_encodings' in outputs:
                feats = outputs['SAR_encodings']
            else:
                raise RuntimeError(f"Could not find SAR encodings in output keys: {outputs.keys()}")

        elif self.modality == "optical":
            outputs = self.backbone(x_optical=x)
            # Try different possible key names
            if 'optical_encodings' in outputs:
                feats = outputs['optical_encodings']
            elif 'optical_GAP' in outputs:
                feats = outputs['optical_GAP']
            else:
                raise RuntimeError(f"Could not find optical encodings in output keys: {outputs.keys()}")
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        # Convert to spatial features
        feats = self._extract_spatial_features(feats)

        # Apply decoder
        logits = self.decode_head(feats)

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return logits