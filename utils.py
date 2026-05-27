"""
Utils for different model needs
"""
import torch.nn as nn
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