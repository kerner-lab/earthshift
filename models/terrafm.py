"""
TerraFM wrappers for finetuning
"""

import torch
from utils import *
import torch.nn.functional as F

# TerraFM native input handling for the Sentinel sensor-shift pairs.
# TerraFM expects raw values (S2 digital numbers, S1 in dB) normalised with its own pretraining
# statistics, S2 bands placed in its 12-band order (B01..B12 without B10), the L2A stem for L2A data,
# and S1 [VV, VH] through its dedicated S1 stem. See https://github.com/mbzuai-oryx/TerraFM#input-preprocessing
TERRAFM_NATIVE_PAIRS = {'EuroSatRGB-EuroSatNS1S2', 'EuroSatRGB-EuroSatRGE1', 'EuroSatRGB-EuroSatRE1E2',
                        'BenV2-S2-S1', 'Sen1Floods11-S2-S1'}
WAVELENGTH_BANDS = {
    'rgb': ['B04', 'B03', 'B02'],
    'rge1': ['B04', 'B03', 'B05'],
    'ns1s2': ['B8A', 'B11', 'B12'],
    're1e2': ['B05', 'B06', 'B07'],
    's2': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
    's2_13': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
}
L2A_PAIRS = {'BenV2-S2-S1'}  # BigEarthNet v2 is L2A; m-EuroSat and Sen1Floods11 are L1C


class TerrafmInput(nn.Module):
    """Places bands into TerraFM's 12 S2 slots (missing bands zero), normalises with TerraFM stats.

    The slot assignment is fixed by the fine-tuning bands, so an EuroSat model trained on RGB sees
    R, G, B05 in the same slots at test time, as with the 1x1 adapter used for other models.
    """
    def __init__(self, terrafm_module, wavelength, dataset_pair):
        super().__init__()
        self.tf = terrafm_module
        self.is_s1 = wavelength == 's1'
        self.is_l2a = dataset_pair in L2A_PAIRS
        if not self.is_s1:
            order = self.tf.S2_BANDS
            self.slots = [order.index(b) if b in order else -1 for b in WAVELENGTH_BANDS[wavelength]]  # B10 -> dropped
            self.level = 's2_l2a' if self.is_l2a else 's2_l1c'

    def forward(self, x):
        if self.is_s1:
            return torch.nan_to_num(self.tf.normalize(x, 's1'), nan=0.5)
        out = x.new_zeros(x.shape[0], 12, *x.shape[2:])
        for c, s in enumerate(self.slots):
            if s >= 0:
                out[:, s] = x[:, c]
        z = self.tf.normalize(out, self.level)
        present = torch.zeros(12, dtype=torch.bool, device=x.device)
        present[[s for s in self.slots if s >= 0]] = True
        return torch.where(present[None, :, None, None], z, torch.zeros_like(z))  # missing bands stay 0


class TerrafmClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int, native_input: nn.Module = None):
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

        # TerraFM native preprocessing (sensor-shift pairs); otherwise a channel adapter
        self.native_input = native_input
        self.is_l2a = bool(native_input is not None and native_input.is_l2a)
        self.input_adapter = None if native_input is not None else channel_adapter(12, self.img_channels)

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
        x = self.native_input(x) if self.native_input is not None else self.input_adapter(x)

        # TerraFM forward returns CLS token features (2-channel input uses the S1 stem)
        features = self.backbone(x, is_l2a=self.is_l2a)

        # Classify
        logits = self.classifier(features)
        return logits


class TerrafmSegmentation(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_classes: int, native_input: nn.Module = None):
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

        # TerraFM native preprocessing (sensor-shift pairs); otherwise a channel adapter
        self.native_input = native_input
        self.is_l2a = bool(native_input is not None and native_input.is_l2a)
        self.input_adapter = None if native_input is not None else channel_adapter(12, self.img_channels)

        self.embed_dim = 768  # hardcoded for terrafm model

        # Segmentation decoder
        self.decoder = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.native_input(x) if self.native_input is not None else self.input_adapter(x)

        # Extract spatial features from last layer (2-channel input uses the S1 stem)
        features = self.backbone.extract_feature(
            x,
            return_h_w=True,
            out_indices=[11],  # Last transformer block
            is_l2a=self.is_l2a
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







































