"""
Utils for different model needs
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import os


class HPCPatches:
    @staticmethod
    def patch_clay_posemb():
        import torch

        def posemb_sincos_1d_hpc(pos, dim):
            device = pos.device
            dtype = pos.dtype

            omega = torch.arange(dim // 2, device=device, dtype=dtype)
            omega = omega / (dim // 2 - 1)

            pos_ = pos.unsqueeze(-1)
            out = pos_ * omega
            return torch.cat([out.sin(), out.cos()], dim=-1)

        # Patch utils
        import terratorch.models.backbones.clay_v1.utils as clay_utils
        clay_utils.posemb_sincos_1d = posemb_sincos_1d_hpc

        # Patch modules (CRITICAL)
        import terratorch.models.backbones.clay_v1.modules as clay_modules
        clay_modules.posemb_sincos_1d = posemb_sincos_1d_hpc

    @classmethod
    def apply_all(cls):
        cls.patch_clay_posemb()


class CROMAPatcher:
    """
    Monkey patches TorchGeo CROMA to fix device mismatch issues on HPC.

    Usage:
        CROMAPatcher.apply()  # Call before loading CROMA models
    """

    _patched = False

    @classmethod
    def apply(cls):
        """Apply monkey patches to CROMA attention mechanisms"""
        if cls._patched:
            print("⚠ CROMA patch already applied, skipping")
            return

        try:
            import torchgeo.models.croma as croma_module

            # Patch Attention.forward
            original_attn = croma_module.Attention.forward

            def patched_attn(self, x, relative_position_bias=None):
                if relative_position_bias is not None:
                    relative_position_bias = relative_position_bias.to(x.device)
                if hasattr(self, 'relative_position_bias_table'):
                    self.relative_position_bias_table = \
                        self.relative_position_bias_table.to(x.device)
                return original_attn(self, x, relative_position_bias)

            croma_module.Attention.forward = patched_attn

            # Patch SwinTransformerBlock.forward (only exists in older torchgeo versions)
            if hasattr(croma_module, 'SwinTransformerBlock'):
                original_block = croma_module.SwinTransformerBlock.forward

                def patched_block(self, x, relative_position_bias):
                    if relative_position_bias is not None:
                        relative_position_bias = relative_position_bias.to(x.device)
                    return original_block(self, x, relative_position_bias)

                croma_module.SwinTransformerBlock.forward = patched_block

            cls._patched = True
            print("✓ CROMA HPC device patch applied successfully")

        except ImportError:
            print("⚠ TorchGeo not installed, skipping CROMA patch")
        except Exception as e:
            print(f"⚠ Failed to apply CROMA patch: {e}")

    @classmethod
    def is_patched(cls):
        """Check if patch has been applied"""
        return cls._patched


# Simple decoder class
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, nc=1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nc, kernel_size=1)
        )
    def forward(self, x):
        return self.decode(x)

def channel_adapter(expected_channels, img_channels):
    """
    Tool to adapt channels from expected for backbone model to
    the number of channels provided by the dataset in finetuning.
    learnable 1×1 convolution that transforms input from one number of
    channels to another
    :return:
    """
    if img_channels != expected_channels:
        input_adapter = nn.Conv2d(
            img_channels,
            expected_channels,
            kernel_size=1,
            bias=False,
        )
    else:
        input_adapter = nn.Identity()

    return input_adapter


def dual_sensor_adapter(expected_channels, rgb_channels=3, s1_channels=2):
    """
    Creates adapters for both RGB and S1 sensors to test BigEarthNetV2 channel shift
    Returns a ModuleDict with both adapters.
    """
    return nn.ModuleDict({
        'rgb': channel_adapter(expected_channels, rgb_channels),
        's1': channel_adapter(expected_channels, s1_channels),
        's2': channel_adapter(expected_channels, rgb_channels)
    })


class FeatureHolder(nn.Module):
    """
    Replaces the decoder of a segmentation wrapper so the backbone features can be
    grabbed. Saves the features and returns one channel so the wrapper forward pass still runs
    """
    def __init__(self):
        super().__init__()
        self.features = None

    def forward(self, x):
        self.features = x
        return x[:, :1]


class SiameseLateFusion(nn.Module):
    def __init__(self, seg_model, num_classes, pre_channels=3, post_channels=1, pre_waves=None, post_waves=None):
        """
        Siamese late fusion wrapper for pre and post event semseg tasks (e.g. BRIGHT pre event optical,
        post event SAR). The same backbone is used for both images, features are concatenated and decoded.
        Expects the pre and post event images stacked on the channel dim [B, pre_channels + post_channels, H, W]
        :param seg_model: segmentation wrapper set up for pre_channels, with a SimpleDecoder as decoder or decode_head
        :param num_classes: Number of segmentation classes
        :param pre_channels: Number of channels in the pre event image
        :param post_channels: Number of channels in the post event image
        :param pre_waves: Wavelengths for the pre event image, only for models that take wavelengths (DOFA, Clay)
        :param post_waves: Wavelengths for the post event image, only for models that take wavelengths (DOFA, Clay).
        If given, the post event image is passed as is instead of repeated to pre_channels
        """
        super().__init__()
        self.pre_channels = pre_channels
        self.post_channels = post_channels
        self.num_classes = num_classes
        self.pre_waves = pre_waves
        self.post_waves = post_waves

        # Wrappers name their decoder either decoder or decode_head
        decoder_name = 'decoder' if hasattr(seg_model, 'decoder') else 'decode_head'

        # Get number of feature channels going into the wrapper decoder
        feat_channels = getattr(seg_model, decoder_name).decode[0].in_channels

        # Swap decoder for feature holder so the wrapper returns backbone features
        self.feature_holder = FeatureHolder()
        setattr(seg_model, decoder_name, self.feature_holder)
        self.encoder = seg_model

        # Decoder head on the concatenated pre and post event features
        self.head = SimpleDecoder(2 * feat_channels, nc=num_classes)

    def encode(self, x, *args):
        """
        Get backbone features for one image
        :param x: Input tensor [B, pre_channels, H, W]
        :return: Feature map from the backbone
        """
        self.encoder(x, *args)
        return self.feature_holder.features

    def set_waves(self, waves):
        """
        Set wavelengths on the wrapper before a forward pass, DOFA stores them in waves and Clay in _waves_list
        :param waves: Wavelengths for the next image
        :return:
        """
        if hasattr(self.encoder, '_waves_list'):
            self.encoder._waves_list = list(waves)
        else:
            self.encoder.waves = waves

    def forward(self, x, *args):
        """
        Run forward pass for semseg tasks
        :param x: Input tensor [B, pre_channels + post_channels, H, W]
        :return: Segmentation logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape

        # Split into pre and post event images
        pre = x[:, :self.pre_channels]
        post = x[:, self.pre_channels:]

        if self.post_waves is None:
            # Repeat single channel post event image to match pre event channels
            if self.post_channels == 1:
                post = post.expand(-1, self.pre_channels, -1, -1)

            # Get features for both images from the same backbone
            pre_features = self.encode(pre, *args)
            post_features = self.encode(post, *args)
        else:
            # Models that take wavelengths get each image as is, with its own wavelengths. The wavelengths
            # passed in are not forwarded as they would override the ones set here
            self.set_waves(self.pre_waves)
            pre_features = self.encode(pre)
            self.set_waves(self.post_waves)
            post_features = self.encode(post)

        # Decode concatenated features to seg map
        logits = self.head(torch.cat([pre_features, post_features], dim=1))

        # Upsample
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        return logits


def set_seed(seed):
    """
    Set random seed for reproducibility across PyTorch, NumPy, and Python's random.
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # PyTorch backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed}")