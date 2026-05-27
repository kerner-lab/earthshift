"""
Script for DOFA wrapper for segmentation and other tasks
"""

from utils import *
import torch.nn.functional as F


class DOFASegmentation(nn.Module):
    def __init__(self, backbone_model, num_classes, waves):
        super(DOFASegmentation, self).__init__()
        self.backbone_model = backbone_model
        self.num_classes = num_classes
        self.waves = waves

        # Load simple decoder head for now
        self.decode_head = SimpleDecoder(in_channels=768, nc=num_classes)

        # Grid size for reshaping - harcoding
        self.grid_size = (14, 14)

    def get_tokens(self, x):
        tokens = {}

        def hook_fn(module, inp, out):
            tokens["x"] = out

        h = self.backbone_model.blocks[-1].register_forward_hook(hook_fn)       # Get tokens from last transformer block which contains most semantic patch embeddings

        try:
            _ = self.backbone_model(x, self.waves)  # classification forward, but hook grabs tokens
        finally:
            h.remove()

        if "x" not in tokens:
            raise RuntimeError("Failed to capture tokens from DOFA last block. Hook didn't fire.")
        return tokens["x"]


    def forward(self, x, waves):
        tok = self.get_tokens(x)  # [B, N, D]

        # Drop CLS token if present
        if tok.dim() != 3:
            raise RuntimeError(f"Expected tokens [B,N,D], got {tok.shape}")

        # If N == 1 + H'*W', remove cls token if present
        B, N, D = tok.shape
        gh, gw = self.grid_size # setting grid height and width
        expected = gh * gw
        if N == expected + 1:
            tok = tok[:, 1:, :]  # [B, H'*W', D]
            N = expected

        if N != expected:
            raise RuntimeError(f"Token count {N} doesn't match grid {gh}x{gw}={expected}")

        # [B, N, D] -> [B, D, gh, gw] reconstruct spatial feature map from transformer patch tokens
        feat = tok.transpose(1, 2).contiguous().view(B, D, gh, gw)

        logits = self.decode_head(feat)  # [B, C, gh, gw]
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits