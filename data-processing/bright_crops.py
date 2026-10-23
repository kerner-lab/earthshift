"""
Cut BRIGHT (DFC25 track 2) tiles listed in split txt files (from bright_geo_splits.py) into
non-overlapping crops, keeping only crops that contain at least one building pixel
(label > 0: 1 intact, 2 damaged, 3 destroyed; 0 is background).

Crops keep their georeferencing and follow the BRIGHT layout and naming:
  <out>/pre-event/<crop>_pre_disaster.tif      (3 bands, optical)
  <out>/post-event/<crop>_post_disaster.tif    (1 band, SAR)
  <out>/target/<crop>_building_damage.tif      (1 band, damage labels)
where <crop> = <tile>_r<row>_c<col> with row/col the crop's pixel offset in the tile.

Also writes <out>/<split name>.txt (one crop id per line, same names as the input split files)
and <out>/crop_summary.txt (crop counts and label pixel counts per split).

With --match_test <id split> <ood split>, the ID test crops are randomly subsampled (--seed) to
the number of OOD test crops. <id split>.txt then holds the subsample and <id split>_all.txt
every kept crop; crop files outside the subsample stay on disk but are not listed.

Usage:
    python data-processing/bright_crops.py --root <BRIGHT dir> --splits <split dir> --out <crop dir>
"""
import argparse
import glob
import os
import random

import numpy as np
import rasterio
from rasterio.windows import Window

SOURCES = {
    # DFC25 split: (image dir, label dir), relative to the BRIGHT root
    "train": ("dfc25_track2_trainval/train", "dfc25_track2_trainval/train/target"),
    "val": ("dfc25_track2_trainval/val", "dfc25_track2_val_labels/target"),
}
FILES = {"pre-event": "_pre_disaster.tif", "post-event": "_post_disaster.tif", "target": "_building_damage.tif"}


def find_tile(root, tile):
    """Paths of the pre, post and label rasters for a tile, from whichever DFC25 split holds it."""
    for img_dir, label_dir in SOURCES.values():
        paths = {"pre-event": f"{root}/{img_dir}/pre-event/{tile}{FILES['pre-event']}",
                 "post-event": f"{root}/{img_dir}/post-event/{tile}{FILES['post-event']}",
                 "target": f"{root}/{label_dir}/{tile}{FILES['target']}"}
        if all(os.path.exists(p) for p in paths.values()):
            return paths
    raise FileNotFoundError(f"No pre/post/label set found for {tile}")


def write_crop(src, window, path):
    profile = src.profile.copy()
    profile.update(width=int(window.width), height=int(window.height),
                   transform=src.window_transform(window), compress="deflate")
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(src.read(window=window))


def crop_tile(paths, tile, out, size):
    """Write the building-containing crops of one tile; return their ids and label pixel counts."""
    with rasterio.open(paths["target"]) as lbl:
        label = lbl.read(1)
    H, W = label.shape
    kept, counts = [], np.zeros(4, dtype=np.int64)
    srcs = {k: rasterio.open(p) for k, p in paths.items()}
    try:
        for r in range(0, H - size + 1, size):
            for c in range(0, W - size + 1, size):
                patch = label[r:r + size, c:c + size]
                if not (patch > 0).any():
                    continue
                crop = f"{tile}_r{r:04d}_c{c:04d}"
                window = Window(c, r, size, size)
                for kind, src in srcs.items():
                    write_crop(src, window, f"{out}/{kind}/{crop}{FILES[kind]}")
                kept.append(crop)
                counts += np.bincount(patch.ravel(), minlength=4)[:4]
    finally:
        for src in srcs.values():
            src.close()
    return kept, counts, (H // size) * (W // size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BRIGHT dir containing the dfc25_track2_* folders")
    p.add_argument("--splits", required=True, help="Dir with split txt files of tile ids (bright_geo_splits.py output)")
    p.add_argument("--out", required=True, help="Directory to write the crops to")
    p.add_argument("--size", type=int, default=256, help="Crop size in pixels")
    p.add_argument("--match_test", nargs=2, metavar=("ID_SPLIT", "OOD_SPLIT"), default=None,
                   help="Subsample ID_SPLIT crops to the number of OOD_SPLIT crops, e.g. "
                        "turkey-earthquake_test morocco-earthquake_test")
    p.add_argument("--seed", type=int, default=42, help="Seed for --match_test subsampling")
    args = p.parse_args()
    for kind in FILES:
        os.makedirs(f"{args.out}/{kind}", exist_ok=True)

    lines = [f"size={args.size} kept crops contain >= 1 building pixel (label > 0)",
             "split: tiles, crops kept / total, label pixels [background, intact, damaged, destroyed]"]
    crops_by_split = {}
    for split_file in sorted(glob.glob(f"{args.splits}/*.txt")):
        name = os.path.basename(split_file)[:-len(".txt")]
        if name == "split_summary":
            continue
        with open(split_file) as f:
            tiles = [t.strip() for t in f if t.strip()]
        crops, counts, total = [], np.zeros(4, dtype=np.int64), 0
        for tile in tiles:
            kept, c, n = crop_tile(find_tile(args.root, tile), tile, args.out, args.size)
            crops += kept
            counts += c
            total += n
        with open(f"{args.out}/{name}.txt", "w") as f:
            f.write("\n".join(sorted(crops)) + "\n")
        crops_by_split[name] = sorted(crops)
        lines.append(f"{name}: {len(tiles)} tiles, {len(crops)} / {total} crops, {counts.tolist()}")
        print(lines[-1])

    if args.match_test:
        id_split, ood_split = args.match_test
        id_crops, n_ood = crops_by_split[id_split], len(crops_by_split[ood_split])
        sub = random.Random(args.seed).sample(id_crops, min(n_ood, len(id_crops)))
        os.replace(f"{args.out}/{id_split}.txt", f"{args.out}/{id_split}_all.txt")
        with open(f"{args.out}/{id_split}.txt", "w") as f:
            f.write("\n".join(sorted(sub)) + "\n")
        counts = np.zeros(4, dtype=np.int64)
        for crop in sub:
            with rasterio.open(f"{args.out}/target/{crop}{FILES['target']}") as src:
                counts += np.bincount(src.read(1).ravel(), minlength=4)[:4]
        lines.append(f"{id_split} subsampled to {len(sub)} crops (seed={args.seed}) to match {ood_split}, "
                     f"{counts.tolist()}; all {len(id_crops)} crops listed in {id_split}_all.txt")
        print(lines[-1])

    with open(f"{args.out}/crop_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
