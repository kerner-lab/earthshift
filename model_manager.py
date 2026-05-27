"""
Model loader class
"""
from pathlib import Path

# Import ONLY the patch definition
from utils import HPCPatches
from utils import CROMAPatcher
# Apply patch BEFORE any terratorch / torchgeo / model imports
HPCPatches.apply_all()
import terratorch.models.backbones.clay_v1.modules as m
import terratorch.models.backbones.clay_v1.utils as u
print("[HPC verify modules]", m.posemb_sincos_1d.__name__)
print("[HPC verify utils]  ", u.posemb_sincos_1d.__name__)
from torchgeo.models import dofa_base_patch16_224, resnet18, resnet50, resnet152, ResNet18_Weights, ResNet50_Weights, ResNet152_Weights
import torchvision.models as tv_models
from terratorch.registry import BACKBONE_REGISTRY
import timm
from models.DOFA import *
from models.CROMA import *
from models.Clay import *
from models.Dino import *
from models.terrafm import *
from models.prithvi import *
from models.resnet import *
from models.vits import *
from models.clip_model import *
from models.terramind import *
#from models.olmoearth import *
from models.galileo_model import *
from utils import *
from huggingface_hub import hf_hub_download
import importlib.util
from galileo.single_file_galileo import Encoder as SingleFileEncoder

class ModelManager:
    """
    Model Manager class to load pre-trained models and manage specific details
    """

    def __init__(self, model_name, task, num_classes, img_channels, wavelength, device):
        self.model_name = model_name
        self.task = task
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.wavelength = wavelength
        self.device = device

        if not model_name:
            raise ValueError("model_name cannot be empty")

        # Update model name to be lower case
        self.model_name = self.model_name.lower()


    def get_waves(self, wavelength=None):
        """
        Get waves for model if needed based on the dataset type
        Sentinel-2 band format
        S2 bands from Planetary Computer
        :return:
        """
        # ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        # ["VV", "VH"]

        wave_dict = {
            'rgb': [0.665, 0.56, 0.49],
            'rgbn': [0.665, 0.56, 0.49, 0.865],
            'ns1s2': [0.865, 1.610, 2.190],
            'rge1': [0.665, 0.560, 0.705],
            're1e2': [0.705, 0.740, 0.783],
            's2': [0.49, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19],
            's2_13': [0.443, 0.49, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 0.945, 1.375, 1.61, 2.19],
            's1': [55465.7, 55465.7]
        }

        wave = wavelength if wavelength is not None else self.wavelength

        match self.model_name:
            case "dofa":
                return torch.tensor(wave_dict[wave], dtype=torch.float32).to(self.device)
            case "clay":
                return wave_dict[wave]
            case "dinov3_small" | "resnet18" | "resnet50" | "resnet152"| "CROMA" | "resnet18_imgnet" | \
                 "resnet50_imgnet" | "resnet152_imgnet" | "dinov3_large_sat":
                return None
            case _:
                return None


    def modify_model(self, model):
        """
        To update; currently a lot of copy-paste for task update
        Modify model for task
        :param model: loaded geotorch/hugging model
        :return:
        """
        match self.model_name:
            case "dofa":
                if self.task == 'class':
                    model.head = torch.nn.Linear(model.head.in_features, self.num_classes)
                    return model
                elif self.task == 'semseg':
                    return DOFASegmentation(model, self.num_classes, self.get_waves())
                else:
                    return None
            case "croma":
                modality = 'sar' if self.wavelength == 's1' else 'optical'
                if self.task == 'class':
                    return CROMAClassifier(model, self.img_channels, num_classes=self.num_classes, modality=modality)
                else:
                    return CROMASegmentation(model, self.img_channels, num_classes=self.num_classes, modality=modality)
            case "clay":
                if self.task == 'class':
                    return ClayClassifier(model, self.img_channels, self.num_classes,
                                          self.get_waves(), pool="mean")
                elif self.task == 'semseg':
                    return ClaySegmentation(model, self.img_channels, num_classes=self.num_classes,
                                            waves=self.get_waves())
                else:
                    return None
            case "dinov3_large_sat" | "dinov3_large_nosat":
                if self.task == 'class':
                    return Dinov3Classifier(model, num_classes=self.num_classes, img_channels=self.img_channels)
                elif self.task == 'semseg':
                    return Dinov3Segmentation(model, num_classes=self.num_classes, img_channels=self.img_channels)
                else:
                    return None
            case "terrafm":
                if self.task == 'class':
                    return TerrafmClassifier(model, self.img_channels, num_classes=self.num_classes)
                elif self.task == 'semseg':
                    return TerrafmSegmentation(model, self.img_channels, num_classes=self.num_classes)
                else:
                    return None
            case "prithvi":
                if self.task == 'class':
                    return PrithviClassifier(model, self.img_channels, num_classes=self.num_classes)
                elif self.task == 'semseg':
                    return PrithviSegmentation(model, self.img_channels, num_classes=self.num_classes)
                else:
                    return None
            case "galileo":
                if self.task == 'class':
                    return GalileoClassifier(model, self.img_channels, num_classes=self.num_classes)
                elif self.task == 'semseg':
                    return GalileoSegmentation(model, self.img_channels, num_classes=self.num_classes)
            case "olmo-earth":
                if self.task == 'class':
                    return OlmoEarthClassifier(model, self.img_channels, num_classes=self.num_classes)
                else:
                    return None
            case "terramind":
                modality = 'S1GRD' if self.wavelength == 's1' else 'S2L2A'
                if self.task == 'class':
                    return TerramindClassifier(model, self.img_channels, num_classes=self.num_classes, modality=modality)
                elif self.task == 'semseg':
                    return TerramindSegmentation(model, self.img_channels, num_classes=self.num_classes, modality=modality)
            case "resnet18" | "resnet50" | "resnet152" | "resnet18_imgnet" | "resnet50_imgnet" | "resnet152_imgnet" | \
                 "resnet_random":
                if self.task == 'class':
                    return ResNetClassifier(model, self.img_channels, num_classes=self.num_classes)
                elif self.task == 'semseg':
                    return ResenetSegmentation(model, self.img_channels, num_classes=self.num_classes)
                else:
                    return None
            case "vit_imgnet" | "vit_random":
                if self.task == 'class':
                    in_features = model.heads.head.in_features
                    model.heads.head = torch.nn.Linear(in_features, self.num_classes)
                    return VITClassifier(model, self.img_channels, num_classes=self.num_classes)
                elif self.task == 'semseg':
                    return VITSegmentation(model, self.img_channels, num_classes=self.num_classes)
                else:
                    return None
            case "clip":
                if self.task == 'class':
                    in_features = model.head.in_features
                    model.head = torch.nn.Linear(in_features, self.num_classes)
                    return VITClassifier(model, self.img_channels, num_classes=self.num_classes)
                elif self.task == 'semseg':
                    return TimmVITSegmentation(model, self.img_channels, num_classes=self.num_classes)
            case _:
                return 'Invalid model query'


    def get_model(self):
        """
        Load model from user query
        :param model_name: Model name queried by user
        :param task: Task type - one of class, semseg, od
        :param num_classes: Number of classes
        :return:
        """
        match self.model_name:
            case "dofa":
                model = dofa_base_patch16_224()
                url = "https://huggingface.co/torchgeo/dofa/resolve/main/dofa_base_patch16_224-a0275954.pth"
                state = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
                model.load_state_dict(state, strict=False)
                model = model.to(self.device)
                return self.modify_model(model)
            case "croma":
                # Apply patch before loading model
                CROMAPatcher.apply()
                from torchgeo.models import croma_base, CROMABase_Weights
                weights = CROMABase_Weights.CROMA_VIT
                modality = 'sar' if self.wavelength == 's1' else 'optical'
                model = croma_base(weights=weights, modalities=[modality])
                model = model.to(self.device)
                return self.modify_model(model)
            case "clay":
                model = timm.create_model('clay_v1_base',
                                          pretrained=True,
                                          in_chans=self.img_channels,
                                          num_classes=self.num_classes)
                return self.modify_model(model)
            case "dinov3_small":
                model = timm.create_model(
                    "vit_small_patch16_dinov3",
                    pretrained=True,  # loads RGB ImageNet-pretrained DINOv3 weights
                    num_classes=self.num_classes  # your number of classes
                )
                return self.modify_model(model)
            case "dinov3_large_sat":
                if self.task == 'class':
                    model = timm.create_model(
                        "vit_large_patch16_dinov3.sat493m",
                        pretrained=True,
                        num_classes=0,
                    )
                else:
                    model = timm.create_model(
                        "vit_large_patch16_dinov3.sat493m",
                        pretrained=True,  # loads dinov3 trained SAT-493M dataset
                        features_only=True
                    )
                return self.modify_model(model)
            case "terrafm":
                repo_id = "MBZUAI/TerraFM"
                # Get TerraFM from Huggingface
                terrafm_code_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="terrafm.py"
                )
                # Load TerraFM module
                spec = importlib.util.spec_from_file_location("terrafm", terrafm_code_path)
                terrafm_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(terrafm_module)
                # Get TerraFM backbone
                model = terrafm_module.terrafm_base(num_classes=0)

                # Load the pretrained weights
                checkpoint_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="TerraFM-B.pth"
                )

                state_dict = torch.load(checkpoint_path)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}

                # Load weights into model
                model.load_state_dict(state_dict, strict=False)
                return self.modify_model(model)
            case "prithvi":
                model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300", pretrained=True)
                return self.modify_model(model)
            case "olmo-earth":
                # hard coding to use the tiny model for now
                model = OlmoEarthBase(
                    model_id=ModelID.OLMOEARTH_V1_TINY,
                    patch_size=16,
                )
                return self.modify_model(model)
            case "galileo":
                DATA_FOLDER = Path("galileo/data")
                model = SingleFileEncoder.load_from_folder(DATA_FOLDER / "models/base",
                                                           device=torch.device("cpu"))
                return self.modify_model(model)
            case "ssl4eo":
                model = BACKBONE_REGISTRY.build("ssl4eos12_vit_small_patch16_224_sentinel2_all_moco",
                                                pretrained=True,
                                                model_bands=3)
            case "terramind":
                if self.wavelength == 'rgb':
                    modalities = ['S2L2A']
                    bands = {'S2L2A': ['BLUE', 'GREEN', 'RED']}
                elif self.wavelength == 'rgbn':
                    modalities = ['S2L2A']
                    bands = {'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW']}
                elif self.wavelength == 's2':
                    modalities = ['S2L2A']
                    bands = {'S2L2A': ['BLUE', 'GREEN', 'RED', 'RED_EDGE_1', 'RED_EDGE_2', 'RED_EDGE_3', 'NIR', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}
                elif self.wavelength == 's2_13':
                    modalities = ['S2L2A']
                    bands = {'S2L2A': ['COASTAL_AEROSOL', 'BLUE', 'GREEN', 'RED', 'RED_EDGE_1', 'RED_EDGE_2', 'RED_EDGE_3', 'NIR_BROAD', 'NIR_NARROW', 'WATER_VAPOR', 'CIRRUS', 'SWIR_1', 'SWIR_2']}
                elif self.wavelength == 's1':
                    modalities = ['S1GRD']
                    bands = {'S1GRD': ['VV', 'VH']}
                else:
                    raise ValueError(f"Unsupported wavelength '{self.wavelength}' for terramind")
                model = BACKBONE_REGISTRY.build(
                    'terramind_v1_base',
                    pretrained=True,
                    modalities=modalities,
                    bands=bands
                )
                return self.modify_model(model)
            case "resnet18":
                weights = ResNet18_Weights.SENTINEL2_RGB_SECO
                model = resnet18(weights=weights)
                return self.modify_model(model)
            case "resnet50":
                weights = ResNet50_Weights.SENTINEL2_RGB_SECO
                model = resnet50(weights=weights)
                return self.modify_model(model)
            case "resnet152":
                weights = ResNet152_Weights.SENTINEL2_MI_RGB_SATLAS
                model = resnet152(weights=weights)
                return self.modify_model(model)
            case "resnet18_imgnet":
                weights = tv_models.ResNet18_Weights.DEFAULT        # imgnet weights IMAGENET1K_V2
                model = tv_models.resnet18(weights=weights)
                return self.modify_model(model)
            case "resnet50_imgnet":
                weights = tv_models.ResNet50_Weights.DEFAULT        # imgnet weights IMAGENET1K_V2
                model = tv_models.resnet50(weights=weights)
                return self.modify_model(model)
            case "resnet152_imgnet":
                weights = tv_models.ResNet152_Weights.DEFAULT       # imgnet weights IMAGENET1K_V2
                model = tv_models.resnet152(weights=weights)
                return self.modify_model(model)
            case "resnet_random":
                model = tv_models.resnet50(weights=False)
                return self.modify_model(model)
            case "vit_random":
                model = tv_models.vit_b_16(weights=False)
                return self.modify_model(model)
            case "vit_imgnet":
                weights = tv_models.ViT_B_16_Weights.DEFAULT  # imgnet weights IMAGENET1K_V1
                model = tv_models.vit_b_16(weights=weights)
                return self.modify_model(model)
            case "clip":
                model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True)
                return self.modify_model(model)
            case "dinov3_large_nosat":
                if self.task == 'class':
                    model = timm.create_model(
                        "vit_large_patch16_dinov3.lvd1689m",
                        pretrained=True,
                        num_classes=0,
                    )
                else:
                    model = timm.create_model(
                        "vit_large_patch16_dinov3.lvd1689m",
                        pretrained=True,  # loads dinov3 trained SAT-493M dataset
                        features_only=True
                    )
                return self.modify_model(model)
            case _:
                return 'Invalid model query'

