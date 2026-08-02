"""
Dataset class/module to load and subset
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchgeo.datasets import RESISC45, UCMerced, NLCD
from geobench_v2.datasets.benv2 import GeoBenchBENV2
import os
import rasterio
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import geopandas as gpd
from PIL import Image
import h5py
from torch.utils.data import Dataset
import json
import re
import collections
from pathlib import Path
import numpy as np


class GeospatialDataset(Dataset):
    def __init__(self, data_dir, split, task='semseg', filter_empty_labels=False, target_classes=None):
        self.data_dir = data_dir
        self.split = split
        self.task = task
        self.filter_empty_labels = filter_empty_labels
        self.target_classes = target_classes

        self.image_dir = '{}/{}/images'.format(data_dir, split)
        self.label_dir = '{}/{}/labels'.format(data_dir, split)

        # Filter out hidden files and .DS_Store
        self.image_fns = sorted([f for f in os.listdir(self.image_dir) if not f.startswith('.')])
        self.label_fns = sorted([f for f in os.listdir(self.label_dir) if not f.startswith('.')])

        assert len(self.image_fns) == len(self.label_fns)

        # Only filter for semantic segmentation tasks
        if filter_empty_labels and task == 'semseg':
            self._filter_samples()
        elif filter_empty_labels and task == 'class':
            print(f"Warning: filter_empty_labels=True but task='class'. Skipping filtering for classification.")

    def _filter_samples(self):
        """
        Filter out samples that don't contain any of the target classes.
        Only applicable for semantic segmentation tasks.
        """
        print(f"Filtering semseg samples without target classes {self.target_classes}...")
        valid_indices = []
        filtered_filenames = []

        for idx, label_fn in enumerate(self.label_fns):
            label_fp = os.path.join(self.label_dir, label_fn)

            # Check if label contains target classes
            if self._has_target_classes(label_fp):
                valid_indices.append(idx)
            else:
                filtered_filenames.append(label_fn)

        # Update file lists
        original_count = len(self.image_fns)
        self.image_fns = [self.image_fns[i] for i in valid_indices]
        self.label_fns = [self.label_fns[i] for i in valid_indices]

        filtered_count = original_count - len(self.image_fns)
        print(f"Filtered out {filtered_count}/{original_count} samples without target classes")
        print(f"Remaining samples: {len(self.image_fns)}")

        # Print filtered filenames
        if filtered_filenames:
            print(f"\nFiltered files ({len(filtered_filenames)}):")
            for fn in filtered_filenames:
                print(f"  - {fn}")

    def _has_target_classes(self, label_path):
        """
        Check if a label file contains any of the target classes (for semseg).
        Handles both regular arrays and paletted/RGB images.
        """
        ext = os.path.splitext(label_path)[1].lower()

        if ext == '.png':
            img = Image.open(label_path)

            if img.mode == 'P':  # Palette mode
                label = np.array(img)
            elif img.mode in ['RGB', 'RGBA']:
                # Check if this needs RGB-to-class conversion (DeepGlobe/DFC2022)
                needs_conversion = any(dataset in label_path for dataset in ["DeepGlobe", "DFC2022"])

                if needs_conversion:
                    # Convert RGB to class indices
                    label_rgb = np.array(img)
                    r = label_rgb[:, :, 0].astype(np.int32)
                    g = label_rgb[:, :, 1].astype(np.int32)
                    b = label_rgb[:, :, 2].astype(np.int32)

                    # Pack RGB into single int
                    packed = (r << 16) | (g << 8) | b

                    # DeepGlobe/DFC2022 class mappings
                    color_to_id = {
                        0x000000: 0,  # Unknown - black
                        0x00FFFF: 1,  # Urban - cyan
                        0xFFFF00: 2,  # Agriculture - yellow
                        0xFF00FF: 3,  # Rangeland - magenta
                        0x00FF00: 4,  # Forest - green
                        0x0000FF: 5,  # Water - blue
                        0xFFFFFF: 6,  # Barren - white
                    }

                    label = np.zeros((label_rgb.shape[0], label_rgb.shape[1]), dtype=np.int32)
                    for packed_color, cls_id in color_to_id.items():
                        label[packed == packed_color] = cls_id
                else:
                    # For other RGB images, take first channel
                    label = np.array(img)[:, :, 0]
            else:
                label = np.array(img)
        elif ext in ['.tif', '.tiff']:
            with rasterio.open(label_path) as src:
                label = src.read(1)  # Read first band
        else:
            # Fallback: use torch read_image
            label_tensor = read_image(label_path)
            if label_tensor.dim() == 3:
                label = label_tensor[0].numpy()
            else:
                label = label_tensor.numpy()

        # Check for target classes
        unique_values = np.unique(label)
        return any(val in self.target_classes for val in unique_values)

    def __len__(self):
        return len(self.image_fns)

    def _load_img(self, path):
        """
        Load img based on type
        :param path:
        :return:
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in [".tif", ".tiff"]:
            with rasterio.open(path) as src:
                arr = src.read()
            return torch.from_numpy(arr)

        return read_image(path)

    def deepglobe_rgb_mask_to_class(self, mask_rgb):
        """
        Function to deal with deepglobe mask
        """
        if mask_rgb.dim() != 3 or mask_rgb.size(0) != 3:
            raise ValueError(f"Expected mask [3,H,W], got {tuple(mask_rgb.shape)}")

        r = mask_rgb[0].to(torch.int32)
        g = mask_rgb[1].to(torch.int32)
        b = mask_rgb[2].to(torch.int32)

        # Pack RGB into a single int: 0xRRGGBB
        packed = (r << 16) | (g << 8) | b

        # DeepGlobe class mappings
        color_to_id = {
            0x000000: 0,  # Unknown - black
            0x00FFFF: 1,  # Urban - cyan
            0xFFFF00: 2,  # Agriculture - yellow
            0xFF00FF: 3,  # Rangeland - magenta
            0x00FF00: 4,  # Forest - green
            0x0000FF: 5,  # Water - blue
            0xFFFFFF: 6,  # Barren - white
        }

        out = torch.full((mask_rgb.size(1), mask_rgb.size(2)), 0, dtype=torch.long, device=mask_rgb.device)
        for packed_color, cls_id in color_to_id.items():
            out[packed == packed_color] = cls_id

        return out

    def __getitem__(self, index):
        image_fp = os.path.join(self.image_dir, self.image_fns[index])
        label_fp = os.path.join(self.label_dir, self.label_fns[index])

        image = self._load_img(image_fp)
        label = self._load_img(label_fp)

        if "DeepGlobe" in image_fp:
            label = self.deepglobe_rgb_mask_to_class(label)

        return image, label


class FTWDataset(Dataset):
    """
    Class for loading FTW dataset splits
    """
    def __init__(self, config_task, data_pair, data_dir, split, finetune_state):
        self.config_task = config_task
        self.data_pair = data_pair
        self.data_dir = data_dir
        self.split = split
        self.finetune_state = finetune_state

        # Get country
        if finetune_state == "in":
            country = self.config_task["finetune"]
        else:
            country = self.config_task["test"]

        # Set paths for labels and images
        self.label_dir = '{}/{}/label_masks/semantic_3class'.format(self.data_dir, country)
        self.image_dir_a = '{}/{}/s2_images/window_a'.format(self.data_dir, country)
        self.image_dir_b = '{}/{}/s2_images/window_b'.format(self.data_dir, country)

        gdf = gpd.read_parquet('{}/{}/chips_{}.parquet'.format(self.data_dir, country, country))

        def filter_split(gdf, split=self.split):
            # filter based on split
            gdf = gdf[gdf['split'] == split]
            # Get field ids
            if 'field_ids' in gdf.columns:
                field_ids = gdf['field_ids'].tolist()
            elif 'aoi_id' in gdf.columns:
                field_ids = gdf['aoi_id'].tolist()
            else:
                raise ValueError(f"Neither 'field_ids' nor 'aoi_ids' found in parquet columns: {gdf.columns.tolist()}")
            return field_ids

        # Handle ftw-germany-year special case
        if 'ftw-germany-year' in data_pair:
            self.field_ids = filter_split(gdf)
            filt_samples = [(fid, 'a') for fid in self.field_ids] + \
                           [(fid, 'b') for fid in self.field_ids]

        # Handle ftw-window case
        elif 'window' in data_pair:
            self.field_ids = filter_split(gdf)
            if finetune_state == "in":
                filt_samples = [(fid, 'a') for fid in self.field_ids]
            else:
                # testing out of distribution on the window b samples
                filt_samples = [(fid, 'b') for fid in self.field_ids]

        # Default case: use both windows
        else:
            self.field_ids = filter_split(gdf)
            filt_samples = [(fid, 'a') for fid in self.field_ids] + \
                           [(fid, 'b') for fid in self.field_ids]

        # Only keep samples where files exist
        self.samples = []
        for fid, window in filt_samples:
            filename = f"{fid}.tif"
            # Get paths based on window
            if window == 'a':
                image_path = f'{self.image_dir_a}/{filename}'
            else:
                image_path = f'{self.image_dir_b}/{filename}'

            label_path = f'{self.label_dir}/{filename}'

            if os.path.exists(image_path) and os.path.exists(label_path):
                self.samples.append((fid, window))

        print(f"Country: {country}, Split: {self.split}, Finetune: {finetune_state}")
        print(f"Total field_ids: {len(self.field_ids)}, Total samples: {len(self.samples)}")
        print(
            f"Windows used: {'a only' if finetune_state == 'in' and 'window' in data_pair else 'b only' if 'window' in data_pair else 'a and b'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Get field ID and which window to use
        field_id, window = self.samples[index]

        filename = f"{field_id}.tif"

        # Select the correct image directory based on window
        if window == 'a':
            image_path = '{}/{}'.format(self.image_dir_a,filename)
        else:  # window == 'b'
            image_path ='{}/{}'.format(self.image_dir_b,filename)

        # Label path (same for both windows)
        label_path = '{}/{}'.format(self.label_dir, filename)

        # Load image
        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))

        with rasterio.open(label_path) as src:
            label = src.read(1)

        # Convert uint16 to float32 and normalize to 0-1 range
        image = image.astype(np.float32) / 10000.0

        # Convert to tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.from_numpy(label).long()

        return image, label


class FTWAllCountriesDataset(Dataset):
    """
    FTW dataset that loads all available countries with a hard-coded deterministic split.

    Field IDs are partitioned per-country using a fixed seed, independently of the
    parquet's existing split column. This guarantees that the ID test set (window_a)
    and the OOD test set (window_b) cover exactly the same field IDs, so the two
    test sets are directly comparable.

    finetune_state='in'  → window_a  (train / val / ID test)
    finetune_state='out' → window_b  (OOD test; only meaningful for split='test')

    Split fractions: 70 % train / 10 % val / 20 % test.
    Per-country seeds are derived from the global SEED + country name so that
    adding or removing a country does not perturb any other country's split.
    """

    SEED       = 42
    TRAIN_FRAC = 0.70
    VAL_FRAC   = 0.10
    # TEST_FRAC  = remaining 0.20

    @staticmethod
    def _country_rng(country: str, seed: int) -> np.random.Generator:
        import hashlib
        h = int(hashlib.md5(f"{seed}_{country}".encode()).hexdigest(), 16) % (2 ** 32)
        return np.random.default_rng(h)

    def __init__(self, data_dir: str, split: str, finetune_state: str, debug: bool = False):
        assert split in ('train', 'val', 'test'), f"Unknown split '{split}'"
        self.data_dir = data_dir
        self.split = split
        self.finetune_state = finetune_state

        window = 'a' if finetune_state == 'in' else 'b'

        self.samples = []  # list of (image_path, label_path)
        countries_loaded = []

        for country_dir in sorted(Path(data_dir).iterdir()):
            if not country_dir.is_dir():
                continue
            country = country_dir.name
            parquet_path = country_dir / f'chips_{country}.parquet'
            if not parquet_path.exists():
                continue

            print(f"Loading {country}...", flush=True)

            gdf = gpd.read_parquet(parquet_path)

            if 'field_ids' in gdf.columns:
                all_ids = sorted(gdf['field_ids'].tolist())
            elif 'aoi_id' in gdf.columns:
                all_ids = sorted(gdf['aoi_id'].tolist())
            else:
                continue

            # Deterministic per-country shuffle, independent of other countries
            rng = self._country_rng(country, self.SEED)
            shuffled = rng.permutation(all_ids)

            n = len(shuffled)
            n_train = int(n * self.TRAIN_FRAC)
            n_val   = int(n * self.VAL_FRAC)

            if split == 'train':
                field_ids = shuffled[:n_train].tolist()
            elif split == 'val':
                field_ids = shuffled[n_train:n_train + n_val].tolist()
            else:  # test — same IDs used for both window_a (ID) and window_b (OOD)
                field_ids = shuffled[n_train + n_val:].tolist()

            label_dir = country_dir / 'label_masks' / 'semantic_3class'
            image_dir = country_dir / 's2_images' / f'window_{window}'

            n_before = len(self.samples)
            country_label_paths = []
            for fid in field_ids:
                filename = f'{fid}.tif'
                img_path = image_dir / filename
                lbl_path = label_dir / filename
                if img_path.exists() and lbl_path.exists():
                    self.samples.append((str(img_path), str(lbl_path)))
                    country_label_paths.append(lbl_path)

            n_added = len(self.samples) - n_before
            if n_added > 0:
                countries_loaded.append(country)

            # Debug: scan all label files for this country and report any out-of-range values.
            # Only runs when FTW_DEBUG=1. Scans every label file so can be slow — run once
            # to identify problem countries, then disable.
            if debug and country_label_paths:
                bad_values = set()
                for lbl_path in country_label_paths:
                    with rasterio.open(lbl_path) as src:
                        label = src.read(1)
                    bad_values.update(int(v) for v in np.unique(label) if v not in (0, 1, 2))
                if bad_values:
                    print(f"  [debug] {country} ({n_added} files): out-of-range values {sorted(bad_values)}", flush=True)
                else:
                    print(f"  [debug] {country} ({n_added} files): clean", flush=True)

        print(f"FTWAllCountries split={split}, window={window}: "
              f"{len(self.samples)} samples across {len(countries_loaded)} countries "
              f"({', '.join(countries_loaded)})", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]

        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))

        with rasterio.open(label_path) as src:
            label = src.read(1)

        image = image.astype(np.float32) / 10000.0
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.from_numpy(label).long()

        # Remap any value outside {0, 1, 2} to 255 (ignore_index).
        # Some countries encode nodata differently; this keeps the loss safe.
        label = torch.where((label >= 0) & (label <= 2), label, torch.tensor(255, dtype=torch.long))

        return image, label


class MEuroSAT(Dataset):
    """
    Dataset loader for m-EuroSAT (geobench)
    """
    # Band name mapping (Sentinel-2 band names to indices)
    BAND_SETS = {
        '01': '01 - Coastal aerosol',
        '02': '02 - Blue',
        'B01': '01 - Coastal aerosol',
        'B02': '02 - Blue',
        'B03': '03 - Green',
        'B04': '04 - Red',
        'B05': '05 - Vegetation Red Edge',
        'B06': '06 - Vegetation Red Edge',
        'B07': '07 - Vegetation Red Edge',
        'B08': '08 - NIR',
        'B8A': '08A - Vegetation Red Edge',
        'B09': '09 - Water vapour',
        'B10': '10 - SWIR - Cirrus',
        'B11': '11 - SWIR',
        'B12': '12 - SWIR',
    }

    def __init__(self, root, split='train', bands=('B04', 'B03', 'B02')):
        """
        Args:
            root: Path to m-eurosat directory
            split: 'train', 'val', or 'test'
            bands: Tuple of band names to load (e.g., ('B04', 'B03', 'B02') for RGB)
        """
        self.root = Path(root)
        self.split = split
        self.bands = bands

        # Map requested bands to full band names
        self.selected_band_names = []
        for band in bands:
            if band in self.BAND_SETS:
                self.selected_band_names.append(self.BAND_SETS[band])
            else:
                raise ValueError(f"Unknown band: {band}. Available bands: {list(self.BAND_SETS.keys())}")

        # Load label mapping
        label_map_path = self.root / 'label_map.json'
        if not label_map_path.exists():
            raise FileNotFoundError(f"label_map.json not found in {self.root}")

        with open(label_map_path, 'r') as f:
            label_map = json.load(f)

        # Create sample_id -> label mapping
        self.sample_to_label = {}
        for label, sample_ids in label_map.items():
            for sample_id in sample_ids:
                self.sample_to_label[sample_id] = int(label)

        # Load split information
        partition_path = self.root / '0.20x_train_partition.json'
        if not partition_path.exists():
            raise FileNotFoundError(f"0.20x_train_partition.json not found in {self.root}")

        with open(partition_path, 'r') as f:
            partition = json.load(f)

        # Map 'val' to 'valid' for partition
        split_map = {'train': 'train', 'val': 'valid', 'test': 'test'}
        mapped_split = split_map.get(split, split)
        self.sample_ids = partition[mapped_split]

        # Filter to only samples that have labels
        self.sample_ids = [sid for sid in self.sample_ids if sid in self.sample_to_label]
        print(f"m-EuroSAT {split}: Loaded {len(self.sample_ids)} samples with bands {bands}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        hdf5_path = self.root / f"{sample_id}.hdf5"

        # Load selected bands from HDF5
        with h5py.File(hdf5_path, 'r') as f:
            # Stack selected bands into [C, H, W] array
            bands = []
            for band_name in self.selected_band_names:
                band_data = f[band_name][:]
                bands.append(band_data)

            image = np.stack(bands, axis=0)  # Shape: [C, 64, 64]

        # Convert to torch tensor
        image = torch.from_numpy(image).float()

        # Get label
        label = self.sample_to_label[sample_id]
        label = torch.tensor(label, dtype=torch.long)

        # Return in TorchGeo format (dict)
        return {
            "image": image,
            "label": label
        }

class Sen1Floods11(Dataset):
    """
    Dataloader for Sen1Floods11 flood mapping dataset.

    Loads all 13 S2 bands (in-distribution) or 2-channel S1 VV/VH (out-of-distribution).
    Labels: 0 = non-flood, 1 = flood, -1 = invalid (remapped to 255 for ignore_index).
    """

    # Maps (finetune_state, split) → CSV filename
    _CSV_MAP = {
        ('in',  'train'): 'flood_train_data.csv',
        ('in',  'val'):   'flood_valid_data.csv',
        ('in',  'test'):  'flood_test_data.csv',
        ('out', 'test'):  'flood_test_data.csv',
    }

    def __init__(self, root, split='train', finetune_state='in'):
        self.root = Path(root)
        self.split = split
        self.finetune_state = finetune_state

        base = self.root / 'data' / 'flood_events' / 'HandLabeled'
        self.label_dir = base / 'LabelHand'
        self.image_dir = base / ('S2Hand' if finetune_state == 'in' else 'S1Hand')
        self.img_suffix = '_S2Hand.tif' if finetune_state == 'in' else '_S1Hand.tif'

        csv_name = self._CSV_MAP.get((finetune_state, split))
        if csv_name is None:
            raise ValueError(f"No split CSV defined for finetune_state='{finetune_state}', split='{split}'")

        csv_path = self.root / 'splits' / 'flood_handlabeled' / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        # CSV: col 0 = S1 filename, col 1 = label filename (no header)
        # Derive the stem from the label filename and keep only samples where both files exist
        self.samples = []
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_fn = line.split(',')[1].strip()
                stem = label_fn.replace('_LabelHand.tif', '')
                if (self.image_dir / f'{stem}{self.img_suffix}').exists() and \
                   (self.label_dir / label_fn).exists():
                    self.samples.append(stem)

        print(f'Sen1Floods11 {split} ({finetune_state}): {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        image_path = self.image_dir / f'{stem}{self.img_suffix}'
        label_path = self.label_dir / f'{stem}_LabelHand.tif'

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int16)

        if self.finetune_state == 'in':
            image = image / 10000.0  # S2 reflectance normalisation

        # Remap -1 (invalid/no-data) to 255 for ignore_index compatibility
        label = np.where(label == -1, 255, label).astype(np.int64)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return image, label


def ucmerced_split_size(root_dir, split, keep_classes):
    """
    Number of UCMerced samples in `split`, restricted to `keep_classes`.

    Read straight from the torchgeo split file rather than by instantiating the
    dataset, which would decode every image just to count labels. Used to size
    the RESISC45-UCMerced-sub finetuning set so the budget is derived from the
    data rather than hardcoded.
    """
    path = os.path.join(root_dir, 'UCMerced', 'uc_merced-{}.txt'.format(split))
    keep = set(keep_classes)
    n = 0
    with open(path) as fh:
        for line in fh:
            name = line.strip().split('/')[-1]
            if not name:
                continue
            if re.sub(r'\d+\.tif$', '', name) in keep:
                n += 1
    return n


class DataManager:
    """
    Data Manager class
    """

    def __init__(self, root_dir, model_name, task, dataset_pair, config_task, data_pair, transform=None, mask=False):
        self.root_dir = root_dir
        self.model_name = model_name
        self.task = task
        self.dataset_pair = dataset_pair
        self.config_task = config_task
        self.data_pair = data_pair
        self.transform = transform
        self.mask = mask
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.ftw_mean = [0.1552, 0.1355, 0.1105, 0.2743] # R G B NIR
        self.ftw_std = [0.1888, 0.1757, 0.1809, 0.1742] # R G B NIR
        self.s2_mean = [0.1552, 0.1355, 0.1105] # from sentinel-2-l2a Clay
        self.s2_std = [0.1888, 0.1757, 0.1809] # from sentinel-2-l2a Clay

        # Update model name to be lower case
        self.model_name = self.model_name.lower()


    def get_dataset(self, finetune_state, split):
        """
        Get the dataset
        :param finetune_state: Specify if the finetune state is the in-distribution data (in)
        or out-of-distribution data (out)
        :param split: split of the dataset
        :return:
        """
        dataset = None
        match self.dataset_pair:
            case "RESISC45-UCMerced" | "RESISC45-UCMerced-sub":
                # -sub is the same pair; it only differs by the finetuning-set
                # subsample applied in get_filtered_dataset.
                if finetune_state == "in":
                    dataset = RESISC45(root=self.root_dir + '/' + 'RESISC45', split=split, download=False)
                elif finetune_state == "out":
                    dataset = UCMerced(root=self.root_dir + '/' + 'UCMerced', split=split, download=False)
            case "UCMerced-RESISC45":
                if finetune_state == "in":
                    dataset = UCMerced(root=self.root_dir + '/' + 'UCMerced', split=split, download=False)
                elif finetune_state == "out":
                    dataset = RESISC45(root=self.root_dir + '/' + 'RESISC45', split=split, download=False)
            case "DeepGlobe-DFC2022":
                if finetune_state == "in":
                    dataset = GeospatialDataset(data_dir=self.root_dir + '/' + 'DeepGlobe', split=split,
                                                task=self.task, filter_empty_labels=True,
                                                target_classes=self.config_task['finetune_classes'])
                elif finetune_state == "out":
                    dataset = GeospatialDataset(data_dir=self.root_dir + '/' + 'DFC2022', split=split,
                                                task=self.task, filter_empty_labels=True,
                                                target_classes=self.config_task['test_classes'])
            case "EuroSatRGB-EuroSatNS1S2":
                if finetune_state == "in":
                    dataset = MEuroSAT(root=self.root_dir + '/' + 'm-eurosat', split=split, bands=('B04', 'B03', 'B02'))
                elif finetune_state == "out":
                    dataset = MEuroSAT(root=self.root_dir + '/' + 'm-eurosat', split=split, bands=('B8A', 'B11', 'B12'))
            case "EuroSatRGB-EuroSatRGE1":
                if finetune_state == "in":
                    dataset = MEuroSAT(root=self.root_dir + '/' + 'm-eurosat', split=split, bands=('B04', 'B03', 'B02'))
                elif finetune_state == "out":
                    dataset = MEuroSAT(root=self.root_dir + '/' + 'm-eurosat', split=split, bands=('B04', 'B03', 'B05'))
            case "EuroSatRGB-EuroSatRE1E2":
                if finetune_state == "in":
                    dataset = MEuroSAT(root=self.root_dir + '/' + 'm-eurosat', split=split, bands=('B04', 'B03', 'B02'))
                elif finetune_state == "out":
                    dataset = MEuroSAT(root=self.root_dir + '/' + 'm-eurosat', split=split, bands=('B05', 'B06', 'B07'))
            case "Sen1Floods11-S2-S1":
                if finetune_state == "in":
                    dataset = Sen1Floods11(root=self.root_dir + '/' + 'sen1floods11/v1.1', split=split,
                                           finetune_state='in')
                elif finetune_state == "out":
                    dataset = Sen1Floods11(root=self.root_dir + '/' + 'sen1floods11/v1.1', split=split,
                                           finetune_state='out')
            case _ if "ftw-all-window" in self.dataset_pair.lower():
                dataset = FTWAllCountriesDataset(
                    data_dir=self.root_dir + '/ftw',
                    split=split,
                    finetune_state=finetune_state,
                    debug=os.environ.get('FTW_DEBUG', '0') == '1',
                )
            case _ if "ftw" in self.dataset_pair.lower():
                dataset = FTWDataset(config_task=self.config_task, data_pair=self.data_pair,
                                     data_dir=self.root_dir + '/' + 'ftw', split=split,
                                     finetune_state=finetune_state)
            case "BenV2-S2-S1":
                s2_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
                s1_bands = ['VV', 'VH']
                if finetune_state == "in":
                    dataset = GeoBenchBENV2(root=self.root_dir + '/' + 'benv2', split=split,
                                            band_order={'s2': s2_bands}, return_stacked_image=False)
                elif finetune_state == "out":
                    dataset = GeoBenchBENV2(root=self.root_dir + '/' + 'benv2', split=split,
                                            band_order={'s1': s1_bands}, return_stacked_image=False)
            case _:
                raise ValueError("Invalid dataset pair")

        return dataset

    def generate_transform(self):
        """
        Return the img transformation needed for model
        """

        t = []
        match self.model_name:
            case "croma":
                t.append(transforms.Resize((120, 120)))
                if "EuroSat" in self.dataset_pair.lower():
                    t.append(transforms.Lambda(lambda x: x / 10000.0))
                    t.append(transforms.Normalize(
                        mean=self.s2_mean ,
                        std=self.s2_std))
                elif "ftw" in self.dataset_pair.lower():
                    t.append(transforms.Normalize(
                        mean=self.ftw_mean,
                        std=self.ftw_std))
                elif "BenV2" in self.dataset_pair or "Sen1Floods11" in self.dataset_pair:
                    # Already normalised in dataset __getitem__
                    pass
                else:
                    t.append(transforms.ConvertImageDtype(torch.float32))
                return transforms.Compose(t)
            case 'clay':
                t.append(transforms.Resize((256, 256)))
                if "EuroSat" in self.dataset_pair.lower():
                    t.append(transforms.Lambda(lambda x: x / 10000.0))
                    t.append(transforms.Normalize(
                        mean=self.s2_mean,
                        std=self.s2_std))
                elif "ftw" in self.dataset_pair.lower():
                    t.append(transforms.Normalize(
                        mean=self.ftw_mean,
                        std=self.ftw_std))
                elif "BenV2" in self.dataset_pair or "Sen1Floods11" in self.dataset_pair:
                    # Already normalised in dataset __getitem__
                    pass
                else:
                    t.append(transforms.ConvertImageDtype(torch.float32))
                return transforms.Compose(t)
            case "terrafm" | "prithvi" | "olmo-earth" | "terramind" | "dofa" | "galileo":
                t.append(transforms.Resize((224, 224)))
                if "EuroSat" in self.dataset_pair.lower():
                    t.append(transforms.Lambda(lambda x: x / 10000.0))
                    t.append(transforms.Normalize(
                        mean=self.s2_mean,
                        std=self.s2_std))
                elif "ftw" in self.dataset_pair.lower():
                    t.append(transforms.Normalize(
                        mean=self.ftw_mean,
                        std=self.ftw_std))
                elif "BenV2" in self.dataset_pair or "Sen1Floods11" in self.dataset_pair:
                    # Already normalised in dataset __getitem__
                    pass
                else:
                    t.append(transforms.ConvertImageDtype(torch.float32))
                return transforms.Compose(t)
            case "resnet18" | "resnet50" | "resnet152" | "resnet18_imgnet" | "resnet50_imgnet" | "resnet152_imgnet" |\
                 "dinov3_small" | "dinov3_large_sat" | "dinov3_large_nosat" | "resnet_random" | "clip" | "vit_imgnet" \
                 | "vit_random":
                t.append(transforms.Resize((224, 224)))
                t.append(transforms.ConvertImageDtype(torch.float32))
                if "BenV2" in self.dataset_pair or "Sen1Floods11" in self.dataset_pair:
                    # Already normalised in dataset __getitem__
                    pass
                elif "ftw" in self.data_pair.lower():         # ftw has 4 channels
                    t.append(transforms.Normalize(
                        mean=self.ftw_mean,
                        std=self.ftw_std,
                    ))
                elif "EuroSat" in self.dataset_pair.lower():
                    t.append(transforms.Lambda(lambda x: x / 10000.0))
                    t.append(transforms.Normalize(
                        mean=self.s2_mean,
                        std=self.s2_std))
                else:
                    t.append(transforms.Normalize(
                        mean=self.imagenet_mean,
                        std=self.imagenet_std,
                    ))
                return transforms.Compose(t)
            case _:
                raise ValueError(f"Invalid model query: {self.model_name}")


    class _FilteredDataset(Dataset):
        """
        Internal Dataset wrapper to filter and transform
        """

        def __init__(self, dataset, task, keep_classes=None, ignore_px=None, transform=None,
                     mask=False, model_name=None, subsample_total=None, subsample_seed=42):
            self.dataset = dataset
            self.task = task
            self.keep_classes = keep_classes
            self.ignore_px = ignore_px
            self.transform = transform
            self.mask = mask
            self.model_name = model_name
            self.subsample_total = subsample_total
            self.subsample_seed = subsample_seed

            if not keep_classes:
                raise ValueError("You must provide at least one class to keep.")

            print('Filtering dataset {}'.format(dataset))
            if self.task == 'class':
                self.indices, self.class_to_new_index = self.filter_classification()
            else:
                self.indices = None


        def filter_classification(self):
            """
            Filter function for classification datasets
            :return:
            """
            # Sort classes alphabetically to define label mapping
            keep_classes_sorted = sorted(self.keep_classes)

            # New mapping (alphabetical): class_name -> new index
            class_to_new_index = {cls: i for i, cls in enumerate(keep_classes_sorted)}

            # Filter indices, grouped by class so they can be subsampled per class
            by_class = collections.defaultdict(list)
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                label = sample["label"]
                label_name = self.dataset.classes[label.item()]

                if label_name in keep_classes_sorted:
                    by_class[label_name].append(idx)

            if self.subsample_total:
                by_class = self.stratified_subsample(by_class, self.subsample_total,
                                                     self.subsample_seed)

            indices = sorted(i for idxs in by_class.values() for i in idxs)
            print(f"Filtered classification dataset: {len(indices)} samples for classes {keep_classes_sorted}")
            return indices, class_to_new_index


        @staticmethod
        def stratified_subsample(by_class, total, seed):
            """
            Cut a class-indexed dict of sample indices down to `total` samples.

            Allocation is proportional to each class's share of the full dataset,
            not matched to the per-class counts of the paired dataset. The point of
            a -sub variant is to isolate finetuning-set size, so the class prior has
            to stay that of the source dataset -- copying the target's prior would
            change two things at once and make the comparison unattributable.

            Largest-remainder rounding hits `total` exactly, and every class keeps
            at least one sample so a small class is never emptied.
            """
            classes = sorted(by_class)
            sizes = np.array([len(by_class[c]) for c in classes], dtype=float)
            n = sizes.sum()
            if total >= n:
                print(f"Subsample target {total} >= available {int(n)}, keeping all")
                return by_class

            exact = sizes / n * total
            take = np.floor(exact).astype(int)
            rem = int(total - take.sum())
            if rem > 0:
                take[np.argsort(-(exact - take))[:rem]] += 1

            rng = np.random.default_rng(seed)
            out = {}
            for cls, k in zip(classes, take):
                idxs = by_class[cls]
                k = int(min(max(k, 1), len(idxs)))
                out[cls] = sorted(rng.choice(idxs, size=k, replace=False).tolist())

            kept = sum(len(v) for v in out.values())
            per = [len(out[c]) for c in classes]
            print(f"Stratified subsample: {int(n)} -> {kept} samples "
                  f"({len(classes)} classes, per-class min={min(per)} max={max(per)}, seed={seed})")
            return out


        def filter_semseg(self, mask):
            """
            Filter function for semantic segmentation datasets
            """
            orig_to_new = {orig: new for new, orig in enumerate(self.keep_classes)}
            out = torch.full_like(mask, fill_value=self.ignore_px)
            for orig, new in orig_to_new.items():
                out[mask == orig] = new
            return out.long()


        def __len__(self):
            return len(self.dataset) if self.indices is None else len(self.indices)

        def __getitem__(self, idx):
            real_idx = idx if self.indices is None else self.indices[idx]
            sample = self.dataset[real_idx]

            if type(sample) is dict:
                # Torchgeo datasets are normally stored as dictionaries
                image = sample["image"]
                label = sample["label"].item()
                label_name = self.dataset.classes[label]
                label = torch.tensor(self.class_to_new_index[label_name])
            else:
                # otherwise its key 0: image, key 1: label
                image = sample[0]
                label = sample[1]

            if self.transform is not None:
                image = self.transform(image)

            if self.task == "semseg":
                # get target size from transformed image
                H, W = image.shape[-2], image.shape[-1]

                # ensure label is [H,W]
                if isinstance(label, torch.Tensor) and label.dim() == 3 and label.size(0) == 1:
                    label = label.squeeze(0)

                # if label came in as [C,H,W] for some reason, make it [H,W]
                if isinstance(label, torch.Tensor) and label.dim() == 3 and label.size(0) in (3, 4):
                    label = label[0]

                # resize mask with NEAREST (preserve class ids)
                label = TF.resize(
                    label.unsqueeze(0).float(),  # [1,H,W]
                    size=[H, W],
                    interpolation=InterpolationMode.NEAREST,
                ).squeeze(0).long()  # [H,W]

                label = self.filter_semseg(label)

            return image, label

    class _UnfilteredDataset(Dataset):
        """
        Apply transform but don't filter if not needed
        """

        def __init__(self, dataset, task, transform=None, model_name=None):
            self.dataset = dataset
            self.transform = transform
            self.model_name = model_name
            self.task = task

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]

            if type(sample) is dict:
                # Handle different dictionary formats
                if 'image_s2' in sample:
                    # BigEarthNet-v2 S2 data
                    image = sample['image_s2']
                elif 'image_s1' in sample:
                    # BigEarthNet-v2 S1 data
                    image = sample['image_s1']
                elif 'image' in sample:
                    # Standard TorchGeo datasets
                    image = sample['image']
                else:
                    raise KeyError(f"Unknown image key in sample. Available keys: {sample.keys()}")

                label = sample["label"]
            else:
                # Tuple format
                image = sample[0]
                label = sample[1]

            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)

            # Handle segmentation labels - resize to match transformed image if needed
            if self.task == "semseg" and isinstance(label, torch.Tensor):
                # Get target size from transformed image
                H, W = image.shape[-2], image.shape[-1]

                # Ensure label is 2D [H, W]
                if label.dim() == 3:
                    if label.size(0) == 1:
                        label = label.squeeze(0)
                    elif label.size(0) in (3, 4):
                        label = label[0]

                # Resize label to match image size
                if label.shape[-2:] != (H, W):
                    label = TF.resize(
                        label.unsqueeze(0).float(),
                        size=[H, W],
                        interpolation=InterpolationMode.NEAREST,
                    ).squeeze(0).long()

            if isinstance(label, torch.Tensor):
                if label.dim() == 0:
                    label = label.item()

            return image, label


    def get_filtered_dataset(self, finetune_state, split, keep_classes, filter=True):
        """
        Build filtered pytorch dataset
        :param finetune_state: Specify if the finetune state is the in-distribution data or out-of-distribution data
        :param split: split of the dataset
        :param keep_classes: Keep classes sorted by label
        :return:
        """
        # If no transform was passed, generate based on model
        if self.transform is None:
            self.transform = self.generate_transform()

        # Grab dataset
        dataset = self.get_dataset(finetune_state, split)

        # Check if filtering is needed, else return non-filtered dataset
        if not filter:
            return DataManager._UnfilteredDataset(dataset=dataset,
                                                  task=self.task,
                                                  transform=self.transform,
                                                  model_name=self.model_name)

        if not keep_classes:
            raise ValueError("You must provide at least one class to keep.")

        # RESISC45-UCMerced-sub: cut the RESISC45 finetuning set down to the size
        # of the UCMerced one, so that pair can be compared against
        # UCMerced-RESISC45 without a 6.9x difference in finetuning data
        # confounding the result. Applied to the splits that are actually fitted
        # on (train, and val for model selection); the ID test split is left at
        # full size so it stays identical to the un-subsampled pair and gives a
        # lower-variance ID estimate.
        subsample_total = None
        if self.dataset_pair == 'RESISC45-UCMerced-sub' and finetune_state == 'in' \
                and split in ('train', 'val'):
            subsample_total = ucmerced_split_size(
                self.root_dir, split, self.config_task['test_classes'])
            print(f"[{self.dataset_pair}] {split}: targeting UCMerced size {subsample_total}")

        # Filter for class or semseg, hardcoded ignore_px to 255 for testing
        filtered = DataManager._FilteredDataset(
            dataset=dataset,
            task= self.task,
            keep_classes=keep_classes,
            ignore_px=255,
            transform=self.transform,
            mask=self.mask,
            model_name=self.model_name,
            subsample_total=subsample_total,
        )

        return filtered