# Contributing to EarthShift

EarthShift is an open benchmarking testbed for evaluating geospatial foundation models under distribution shift. Contributions that add new models or new datasets are especially welcome.

---

## Table of Contents

- [Adding a New Model](#adding-a-new-model)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Updating the Config](#updating-the-config)
- [Pull Request Checklist](#pull-request-checklist)

---

## Adding a New Model

Models live in `models/` as task-specific wrapper classes around a pretrained backbone. Each model file defines at minimum a `*Classifier` class, and optionally a `*Segmentation` class.

### Step 1 — Create a model file

Create `models/mymodel.py`. The wrapper must subclass `torch.nn.Module` and follow this pattern:

```python
import torch
import torch.nn as nn
from utils import channel_adapter, SimpleDecoder  # re-use existing utilities

class MyModelClassifier(nn.Module):
    def __init__(self, backbone, img_channels, num_classes, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.channel_adapt = channel_adapter(img_channels, BACKBONE_IN_CHANNELS)
        embed_dim = 768  # set to your backbone's output dimension
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapt(x)
        feats = self.backbone(x)          # shape: [B, embed_dim]
        return self.classifier(feats)


class MyModelSegmentation(nn.Module):
    def __init__(self, backbone, img_channels, num_classes, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.channel_adapt = channel_adapter(img_channels, BACKBONE_IN_CHANNELS)
        embed_dim = 768
        self.decoder = SimpleDecoder(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapt(x)
        feats = self.backbone.forward_features(x)   # shape: [B, embed_dim, H', W']
        return self.decoder(feats)
```

Backbone weights should be loaded from a public source (HuggingFace Hub, `timm`, or `torchgeo`) so that anyone can reproduce results without private data. Document the source in a comment at the top of the file.

### Step 2 — Register the model in `model_manager.py`

`ModelManager.get_model()` uses a `match/case` block to instantiate backbones. Add a new case:

```python
# model_manager.py  –  inside ModelManager.get_model()
case "mymodel":
    from models.mymodel import MyModelClassifier, MyModelSegmentation
    backbone = ...  # load pretrained backbone here
    return backbone
```

Then in `ModelManager.modify_model()`, add the wrapper assignment:

```python
case "mymodel":
    if self.task == "class":
        return MyModelClassifier(model, self.img_channels, self.num_classes)
    elif self.task == "semseg":
        return MyModelSegmentation(model, self.img_channels, self.num_classes)
```

If your model requires wavelength/band information, handle it in `ModelManager.get_waves()` following the pattern used by DOFA, Clay, or CROMA.

### Step 3 — Add model-specific transforms

`DataManager.generate_transform()` applies model-specific resizing and normalization. Add a case for your model, specifying its expected input resolution and normalization statistics:

```python
# data_manager.py  –  inside generate_transform()
case "mymodel":
    resize = (224, 224)   # adjust to your backbone's expected size
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
```

Use ImageNet statistics for RGB-pretrained models, or compute and provide dataset-specific statistics for multispectral models.

### Step 4 — Update the config

Add `"mymodel"` to the `"models"` list for each relevant experiment config (see [Updating the Config](#updating-the-config)).

---

## Adding a New Dataset

Datasets are accessed through `DataManager.get_dataset()`. For datasets available in [torchgeo](https://torchgeo.readthedocs.io/), no new dataset class is needed — torchgeo handles downloading and loading.

### Step 1 — Verify torchgeo availability

Check whether the dataset exists in torchgeo:

```python
from torchgeo.datasets import MyNewDataset  # confirm this import works
```

If it does not exist in torchgeo, you will need to write a custom `torch.utils.data.Dataset` subclass in `data_manager.py`, following the pattern of `GeospatialDataset` (line 24) or `MEuroSAT` (line 446).

### Step 2 — Add a case in `DataManager.get_dataset()`

`get_dataset()` uses a `match/case` on `self.dataset_pair`. Add a case that returns the torchgeo dataset instance for each split:

```python
# data_manager.py  –  inside DataManager.get_dataset()
case "SourceDataset-TargetDataset":
    from torchgeo.datasets import SourceDataset, TargetDataset
    if split == "train" or split == "val":
        return SourceDataset(root=self.root_dir, split=split, download=False)
    else:
        return TargetDataset(root=self.root_dir, split=split, download=False)
```

Torchgeo datasets accept a `transforms` argument; pass `self.transform` so model-specific preprocessing is applied automatically.

### Step 3 — Choose or define a dataset pair name

Pick a descriptive hyphenated name that encodes the shift being tested, e.g. `"EuroSAT-PatternNet"` for a data-domain shift or `"BigEarthNet-S2-S1"` for a sensor shift. This name is used as the key in the config and as the `--dataset_pair` CLI argument.

### Step 4 — Add normalization statistics

If your dataset uses a sensor or spectral configuration not already covered, add mean/std constants near the top of `data_manager.py` and reference them in `generate_transform()`.

### Step 5 — Update the config

Add the dataset pair to the appropriate config file (see below).

---

## Updating the Config

Config files in `configs/` define which dataset pairs and models are active for each shift experiment. Each file corresponds to one shift type:

| File | Shift type |
|---|---|
| `configs/data-shift-exp.json` | Dataset / domain shift |
| `configs/sensor-shift-exp.json` | Sensor modality shift |
| `configs/temporal-shift-exp.json` | Temporal / location shift |
| `configs/scale-shift-exp.json` | Spatial scale shift |
| `configs/geo-shift-exp.json` | Geographic shift |

### Schema

```json
{
  "task": {
    "class": {
      "data-pairs": {
        "SourceDataset-TargetDataset": {
          "finetune":            "SourceDataset",
          "finetune_classes":    ["class_a", "class_b"],
          "test":                "TargetDataset",
          "test_classes":        ["class_a", "class_b"],
          "img_channels":        3,
          "finetune_wavelength": "rgb",
          "test_wavelength":     "rgb",
          "filter":              true
        }
      },
      "models": ["resnet18", "DOFA", "CROMA", "mymodel"]
    }
  }
}
```

**Field reference:**

| Field | Description |
|---|---|
| `finetune` | Dataset name used for fine-tuning (must match a case in `get_dataset()`) |
| `finetune_classes` | Class labels available in the training split |
| `test` | Dataset name used for out-of-distribution evaluation |
| `test_classes` | Class labels in the test split (may differ from `finetune_classes`) |
| `img_channels` | Number of input channels (e.g. 3 for RGB, 12 for Sentinel-2 all-bands) |
| `finetune_wavelength` | Modality key for training data; passed to `get_waves()` (e.g. `"rgb"`, `"s2"`, `"s1"`) |
| `test_wavelength` | Modality key for test data; set differently from `finetune_wavelength` for sensor-shift experiments |
| `filter` | Whether to filter samples to only the specified classes |

---

## Pull Request Checklist

Before opening a PR, confirm the following:

**For new models:**
- [ ] Model file added to `models/` with `*Classifier` (and `*Segmentation` if applicable)
- [ ] Backbone source documented (URL / HuggingFace repo ID) in the model file
- [ ] Case added to `ModelManager.get_model()` and `ModelManager.modify_model()` in `model_manager.py`
- [ ] Transform case added to `DataManager.generate_transform()` in `data_manager.py`
- [ ] Model name added to `"models"` list in at least one config file
- [ ] Smoke-tested locally: `python run_pipeline.py --model mymodel --task class --shift data --dataset_pair <pair> --finetune_type head ...`

**For new datasets:**
- [ ] Dataset pair case added to `DataManager.get_dataset()` in `data_manager.py`
- [ ] Dataset pair name and metadata added to the appropriate `configs/*.json` file
- [ ] Normalization statistics added or referenced in `generate_transform()`
- [ ] Data source and download instructions documented here or in `README.md`
- [ ] Smoke-tested locally with at least one existing model

**General:**
- [ ] No hard-coded absolute paths (use `root_dir` / `save_dir` arguments)
- [ ] Reproducibility: any new random operations use the seeded generator from `set_seed()`
- [ ] PR description includes: what shift scenario is addressed, where weights/data can be obtained, and any known limitations
