"""
Build BRIGHT (DFC25 track 2) tile-id splits for the geo shift experiment:

  ID  (default turkey-earthquake):  train / val / test
  OOD (default morocco-earthquake): test

ID tiles are restricted to the given ground resolutions (the 0.35 m Turkey tiles are left out).
The DFC25 val tiles stay as val; the DFC25 train tiles are split into train and an ID test set,
stratified by resolution so the test set keeps the same resolution mix as train. The OOD test
set is a random subset of all OOD tiles (DFC25 train + val), the same size as the ID test set,
or all OOD tiles with --ood_all (e.g. to size-match ID and OOD later at crop level).

Writes one tile id per line (e.g. turkey-earthquake_00000000) to:
  <out>/{id_aoi}_train.txt, {id_aoi}_val.txt, {id_aoi}_test.txt, {ood_aoi}_test.txt
  <out>/split_summary.txt

Usage:
    python data-processing/bright_geo_splits.py --root <BRIGHT dir> --out <split dir>
"""
import argparse
import glob
import math
import os
import random
from collections import defaultdict

import rasterio

POST_SUFFIX = "_post_disaster.tif"
LABEL_SUFFIX = "_building_damage.tif"


def ground_res(path):
    """Pixel width in metres (converts degree-based CRSs at the tile's latitude)."""
    with rasterio.open(path) as src:
        res = abs(src.res[0])
        if src.crs is not None and src.crs.is_geographic:
            lat = (src.bounds.top + src.bounds.bottom) / 2
            res *= 111320 * math.cos(math.radians(lat))
    return round(res, 3)


def list_tiles(root, aoi, split):
    """Tile ids for an AOI in a DFC25 trainval split that have both a post-event image and a label."""
    label_dir = (f"{root}/dfc25_track2_trainval/train/target" if split == "train"
                 else f"{root}/dfc25_track2_val_labels/target")
    tiles = {}
    for post in sorted(glob.glob(f"{root}/dfc25_track2_trainval/{split}/post-event/{aoi}_*{POST_SUFFIX}")):
        tile = os.path.basename(post)[:-len(POST_SUFFIX)]
        if os.path.exists(f"{label_dir}/{tile}{LABEL_SUFFIX}"):
            tiles[tile] = post
    return tiles


def stratified_test(tiles_by_res, n_test, rng):
    """Draw n_test tiles across resolution groups in proportion to group size (largest remainder)."""
    total = sum(len(v) for v in tiles_by_res.values())
    quota = {r: n_test * len(v) / total for r, v in tiles_by_res.items()}
    alloc = {r: int(q) for r, q in quota.items()}
    for r in sorted(quota, key=lambda r: quota[r] - alloc[r], reverse=True)[:n_test - sum(alloc.values())]:
        alloc[r] += 1
    test = []
    for r in sorted(tiles_by_res):
        test += rng.sample(sorted(tiles_by_res[r]), alloc[r])
    return set(test)


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(sorted(ids)) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BRIGHT dir containing the dfc25_track2_* folders")
    p.add_argument("--out", required=True, help="Directory to write the split txt files to")
    p.add_argument("--id_aoi", default="turkey-earthquake")
    p.add_argument("--ood_aoi", default="morocco-earthquake")
    p.add_argument("--resolutions", type=float, nargs="+", default=[0.305, 0.307, 0.313],
                   help="ID ground resolutions (m) to keep")
    p.add_argument("--test_frac", type=float, default=0.2, help="ID test size as a fraction of all kept ID tiles")
    p.add_argument("--ood_all", action="store_true", help="Keep all OOD tiles instead of matching the ID test size")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    keep = {round(r, 3) for r in args.resolutions}

    # ID: keep only the requested resolutions, grouped by resolution
    id_by_split = {}
    for split in ["train", "val"]:
        by_res = defaultdict(list)
        for tile, post in list_tiles(args.root, args.id_aoi, split).items():
            res = ground_res(post)
            if res in keep:
                by_res[res].append(tile)
        id_by_split[split] = by_res
    res_of = {t: r for by_res in id_by_split.values() for r, ts in by_res.items() for t in ts}

    n_kept = len(res_of)
    n_test = round(args.test_frac * n_kept)
    val = {t for ts in id_by_split["val"].values() for t in ts}
    test = stratified_test(id_by_split["train"], n_test, rng)
    train = {t for ts in id_by_split["train"].values() for t in ts} - test

    # OOD: all tiles from DFC25 train + val, subsampled to the ID test size unless --ood_all
    ood_all = sorted(set(list_tiles(args.root, args.ood_aoi, "train")) | set(list_tiles(args.root, args.ood_aoi, "val")))
    ood_test = set(ood_all) if args.ood_all else set(rng.sample(ood_all, min(n_test, len(ood_all))))

    splits = {f"{args.id_aoi}_train": train, f"{args.id_aoi}_val": val,
              f"{args.id_aoi}_test": test, f"{args.ood_aoi}_test": ood_test}
    for name, ids in splits.items():
        write_ids(f"{args.out}/{name}.txt", ids)

    lines = [f"seed={args.seed} test_frac={args.test_frac} resolutions={sorted(keep)} ood_all={args.ood_all}",
             f"{args.id_aoi}: {n_kept} tiles kept; {args.ood_aoi}: {len(ood_all)} tiles available"]
    for name, ids in splits.items():
        mix = defaultdict(int)
        for t in ids:
            mix[res_of.get(t, "ood")] += 1
        mix_str = ", ".join(f"{r}: {n} ({100 * n / len(ids):.1f}%)" for r, n in sorted(mix.items(), key=str))
        lines.append(f"{name}: {len(ids)}  [{mix_str}]")
    with open(f"{args.out}/split_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
