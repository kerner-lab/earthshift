"""
Wrapper classes for olmo-earth finetuning
"""
from utils import *
import torch
from torch import nn
import torch.nn.functional as F
from olmoearth_pretrain_minimal import NormalizerV1_3
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample


class OlmoEarthEncoder(nn.Module):
    def __init__(self, encoder, bands, raw_s2=False, patch_size=8, input_res=10):
        """
        Build OlmoEarth input from [B, C, H, W] image and return
        [B, 768, H/p, W/p] features
        """
        super().__init__()
        self.encoder = encoder
        self.bands = bands
        self.raw_s2 = raw_s2
        self.patch_size = patch_size
        self.input_res = input_res
        self.stats = NormalizerV1_3().norm_config

    def forward(self, x, bands=None):
        modality, names = bands if bands is not None else self.bands
        order = Modality.get(modality).band_order
        # Drop input bands OlmoEarth does not take and find location of each remaining band
        keep = [i for i, b in enumerate(names) if b in order]
        idx = [order.index(names[i]) for i in keep]
        x = x[:, keep].permute(0, 2, 3, 1)  # [B, H, W, C]
        B, H, W, _ = x.shape

        if self.raw_s2 and modality == 'sentinel2_l2a':
            # OlmoEarth normalisation: (x - (mean - 2 std)) / (4 std)
            mean = x.new_tensor([self.stats[modality][order[i]]['mean'] for i in idx])
            std = x.new_tensor([self.stats[modality][order[i]]['std'] for i in idx])
            x = (x * 10000.0 - (mean - 2 * std)) / (4 * std)
        else:
            x = (x + 2) / 4

        data = x.new_zeros(B, H, W, len(order))
        data[..., idx] = x
        data = data.unsqueeze(3)  # one timestep: [B, H, W, 1, bands]

        n_groups = self.encoder.patch_embeddings.tokenization_config.get_num_bandsets(modality)
        mask = torch.zeros(B, H, W, 1, n_groups, dtype=torch.long, device=x.device)  # all visible
        timestamps = torch.zeros(B, 1, 3, dtype=torch.long, device=x.device)
        timestamps[..., 0], timestamps[..., 1], timestamps[..., 2] = 1, 5, 2020  # day, month (0 index), year

        sample = MaskedOlmoEarthSample(timestamps=timestamps, **{modality: data, f'{modality}_mask': mask})
        out = self.encoder(sample, patch_size=self.patch_size, input_res=self.input_res, fast_pass=True)

        features = out['location_tokens']  # [B, H/p, W/p, 768]
        return features.permute(0, 3, 1, 2).contiguous()


class OlmoEarthClassifier(nn.Module):
    def __init__(self, backbone, bands, num_classes, raw_s2=False):
        super().__init__()
        self.backbone = OlmoEarthEncoder(backbone, bands, raw_s2=raw_s2)
        self.embed_dim = 768
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, bands=None):
        features = self.backbone(x, bands)
        return self.classifier(features.mean(dim=(2, 3)))


class OlmoEarthSegmentation(nn.Module):
  def __init__(self, backbone, bands, num_classes, raw_s2=False):
      super().__init__()
      self.backbone = OlmoEarthEncoder(backbone, bands, raw_s2=raw_s2)
      self.embed_dim = 768
      self.decoder = SimpleDecoder(in_channels=self.embed_dim, nc=num_classes)

  def forward(self, x, bands=None):
      B, C, H, W = x.shape
      features = self.backbone(x, bands)
      logits = self.decoder(features)
      return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)