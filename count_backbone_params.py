"""
Count backbone parameters for each model before any task head is added.
Run from the earthshift directory: python count_backbone_params.py
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import timm
import torchvision.models as tv_models
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def fmt(n):
    return f"{n:,} ({n/1e6:.1f}M)"


results = {}


def load(name, fn):
    print(f"Loading {name}...", flush=True)
    try:
        model = fn()
        n = count_params(model)
        results[name] = n
        print(f"  {name}: {fmt(n)}")
    except Exception as e:
        results[name] = f"ERROR: {e}"
        print(f"  {name}: ERROR - {e}")


# ── ResNets ──────────────────────────────────────────────────────────────────
load("resnet_random",   lambda: tv_models.resnet50(weights=None))
load("resnet50_imgnet", lambda: tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT))

# ── ViTs ─────────────────────────────────────────────────────────────────────
load("vit_random",  lambda: tv_models.vit_b_16(weights=None))
load("vit_imgnet",  lambda: tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT))
load("clip",        lambda: timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True))
load("dinov3_large_sat", lambda: timm.create_model('vit_large_patch16_dinov3.sat493m', pretrained=True, num_classes=0))

# ── CROMA ────────────────────────────────────────────────────────────────────
def load_croma():
    from model_manager import CROMAPatcher
    CROMAPatcher.apply()
    from torchgeo.models import croma_base, CROMABase_Weights
    return croma_base(weights=CROMABase_Weights.CROMA_VIT, modalities=['optical'])

load("croma", load_croma)

# ── Clay ─────────────────────────────────────────────────────────────────────
load("clay", lambda: timm.create_model('clay_v1_base', pretrained=True, in_chans=10, num_classes=0))

# ── DOFA ─────────────────────────────────────────────────────────────────────
def load_dofa():
    from models.DOFA import dofa_base_patch16_224
    model = dofa_base_patch16_224()
    url = "https://huggingface.co/torchgeo/dofa/resolve/main/dofa_base_patch16_224-a0275954.pth"
    state = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model

load("dofa", load_dofa)

# ── Galileo ───────────────────────────────────────────────────────────────────
def load_galileo():
    from galileo.single_file_galileo import SingleFileEncoder
    return SingleFileEncoder.load_from_folder(Path("galileo/data/models/base"), device=torch.device("cpu"))

load("galileo", load_galileo)

# ── Prithvi ───────────────────────────────────────────────────────────────────
def load_prithvi():
    from terratorch.registry import BACKBONE_REGISTRY
    return BACKBONE_REGISTRY.build("prithvi_eo_v2_300", pretrained=True)

load("prithvi", load_prithvi)

# ── TerraFM ───────────────────────────────────────────────────────────────────
def load_terrafm():
    import importlib.util
    from huggingface_hub import hf_hub_download
    repo_id = "MBZUAI/TerraFM"
    code_path = hf_hub_download(repo_id=repo_id, filename="terrafm.py")
    spec = importlib.util.spec_from_file_location("terrafm", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.terrafm_base(num_classes=0)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="TerraFM-B.pth")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'head' not in k}
    model.load_state_dict(state_dict, strict=False)
    return model

load("terrafm", load_terrafm)

# ── TerraMind ─────────────────────────────────────────────────────────────────
def load_terramind():
    from terratorch.registry import BACKBONE_REGISTRY
    return BACKBONE_REGISTRY.build(
        'terramind_v1_base', pretrained=True,
        modalities=['S2L2A'],
        bands={'S2L2A': ['BLUE', 'GREEN', 'RED', 'RED_EDGE_1', 'RED_EDGE_2',
                         'RED_EDGE_3', 'NIR', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
    )

load("terramind", load_terramind)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"{'Model':<20} {'Parameters':>20}")
print("="*50)
for name, val in results.items():
    if isinstance(val, int):
        print(f"{name:<20} {fmt(val):>20}")
    else:
        print(f"{name:<20} {val}")
