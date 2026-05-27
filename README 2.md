# 🌍 EarthShift Testbed

**EarthShift** is the first public testbed for benchmarking the robustness of geospatial foundation models (GFMs) across multiple realistic distribution shifts encountered in remote sensing.

> Current Earth observation benchmarks focus on measuring performance on diverse tasks and applications, typically measuring generalization *in-distribution*. But when models are deployed, they must generalize to myriad *out-of-distribution* scenarios — new time periods, geographies, scales, and sensors. **EarthShift** is designed to close this gap.
---

## 🔍 What is EarthShift?

EarthShift enables users to measure **distributional robustness** by comparing model performance in- and out-of-distribution using datasets from paired:

- 🗺️ **Geographic locations** — does the model generalize to unseen regions?
- 📅 **Temporal windows** — does performance hold across different time periods?
- 🛰️ **Sensors** — how robust is the model when the input sensor changes?
- 📡 **Data sources** — can the model handle shifts between different data providers?
- 🔭 **Spatial scales** — does the model transfer across different spatial resolutions?

Our experiments across **8 geospatial foundation models** and **11 tasks** covering all 5 shift types reveal that GFMs consistently perform around **20% worse out-of-distribution**, regardless of model architecture, size, pre-training strategy, or fine-tuning approach. Strikingly, GFM robustness is similar to that of generic vision foundation models — and even fully-supervised models — highlighting that distributional robustness remains an open and critical challenge for the field.

---

## 📦 Shift Types

EarthShift measures distribution shifts across five categories:

| Shift Type | Description |
|---|---|
| 🔭 **Spatial Scale** | Shifts in ground sampling distance / spatial resolution |
| 📅 **Temporal** | Shifts across different time periods or seasons |
| 🗺️ **Geographic** | Shifts across different geographic regions |
| 🛰️ **Sensor** | Shifts between different remote sensing sensors |
| 📡 **Data Source** | Shifts between different data collection sources or modalities |

---

## 🚀 Running the EarthShift Pipeline

The EarthShift pipeline can be run from the command line via:

```bash
python run_pipeline.py
```

### ⚙️ Configurable Parameters

| Parameter | Description |
|---|---|
| `--root_dir` | Root directory of datasets for fine-tuning and inference |
| `--task` | Model task: `class` (classification), `semseg` (semantic segmentation), or `od` (object detection) |
| `--model` | GFM or baseline model to evaluate |
| `--shift` | Shift experiment type: one of `data`, `sensor`, `location`, `temporal`, `scale` |
| `--dataset_pair` | Dataset pair for testing distribution shift (use `--help` for full list) |
| `--finetune_type` | Fine-tuning strategy: `head` (frozen backbone) or `full` (full fine-tuning) |
| `--save_dir` | Output directory for results |

---

## 💡 Why EarthShift?

EarthShift is motivated by a critical gap in how we evaluate remote sensing models: **high benchmark performance does not imply robust real-world deployment**. By providing a standardized testbed with paired in- and out-of-distribution datasets, EarthShift enables the community to:

- 📊 Quantify the robustness gap for any model
- 🔬 Diagnose which shift types are hardest for a given architecture
- 🏆 Drive progress toward models that are not just accurate, but **reliable**

We release our code and datasets to provide a testbed to guide future work toward foundation models that are robust and reliable in real-world remote sensing applications.

## Dataset License Details

| Dataset | License |
|---|---|
  RESISC45     |    CC BY-NC-SA 4.0
  UCMerced     |  Public Domain
  DeepGlobe    |    Non-commercial research and educational use only
  DFC2022      |    IGN's "licence ouverte\"
  FTW          |    CC BY-SA 4.0
  Sen1Floods11 |   Open Access
  BigEarthNet v2 |  Community Data License Agreement -- Permissive -- Version 1.0
  m-EuroSat  |      MIT License
