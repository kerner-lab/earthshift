"""
Download Chesapeake Land Cover and export two chip datasets to disk:

  <out>/1m/{train,val,test}/images/*.npy   NAIP, 4 bands (R,G,B,NIR), 256x256 @ 1 m
  <out>/1m/{train,val,test}/masks/*.npy    land-cover labels @ 1 m
  <out>/30m/{train,val,test}/images/*.npy  Landsat, 4 bands (R,G,B,NIR), 64x64 @ 30 m
  <out>/30m/{train,val,test}/masks/*.npy   land-cover labels @ 30 m (majority vote)

Usage:
    python export_chesapeake.py --root  --state md
"""
import argparse
import glob
import os

import numpy as np
import rasterio
from torchgeo.datasets import ChesapeakeCVPR

NODATA = 0
LABELS = [1, 2, 3, 4, 5, 6]          # valid land-cover values; check with np.unique
NAIP_BANDS = [0, 1, 2, 3]            # R, G, B, NIR
LANDSAT_BANDS = [3, 2, 1, 4]         # VERIFY: assumes Landsat 8 B1..B7 order -> B4,B3,B2,B5
F = 30                               # 1 m -> 30 m


def read(path, bands=None):
    with rasterio.open(path) as src:
        return src.read([b + 1 for b in bands]) if bands else src.read(1)


def blocks(a, f):
    """(..., H, W) -> (..., H/f, W/f, f*f) non-overlapping blocks."""
    *lead, h, w = a.shape
    h, w = h // f * f, w // f * f
    a = a[..., :h, :w].reshape(*lead, h // f, f, w // f, f)
    return np.moveaxis(a, -3, -2).reshape(*lead, h // f, w // f, f * f)


def to_30m(img, lc):
    img30 = blocks(img.astype(np.float32), F).mean(-1)
    b = blocks(lc, F)
    counts = np.stack([(b == c).sum(-1) for c in LABELS])
    lc30 = np.array(LABELS)[counts.argmax(0)].astype(np.uint8)
    lc30[counts.sum(0) < (F * F) / 2] = NODATA
    return img30, lc30


def save_chips(img, lc, size, out_dir, prefix):
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    n = 0
    for i in range(0, lc.shape[0] - size + 1, size):
        for j in range(0, lc.shape[1] - size + 1, size):
            m = lc[i:i + size, j:j + size]
            if (m == NODATA).mean() > 0.5:
                continue
            np.save(f"{out_dir}/images/{prefix}_{i}_{j}.npy", img[:, i:i + size, j:j + size])
            np.save(f"{out_dir}/masks/{prefix}_{i}_{j}.npy", m)
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--state", default="md", choices=["de", "md", "ny", "pa", "va", "wv"])
    args = p.parse_args()
    out = args.out or os.path.join(args.root, f"chips_{args.state}")

    ChesapeakeCVPR(root=args.root, splits=[f"{args.state}-train"], download=True)

    for split in ["train", "val", "test"]:
        tiles = sorted(glob.glob(
            f"{args.root}/**/{args.state}_*{split}_tiles/*_naip-new.tif", recursive=True))
        print(f"{split}: {len(tiles)} tiles")
        n1 = n30 = 0
        for naip_path in tiles:
            prefix = os.path.basename(naip_path).replace("_naip-new.tif", "")
            lc = read(naip_path.replace("_naip-new.tif", "_lc.tif"))

            naip = read(naip_path, NAIP_BANDS)
            n1 += save_chips(naip, lc, 256, f"{out}/1m/{split}", prefix)

            landsat = read(naip_path.replace("_naip-new.tif", "_landsat-leaf-on.tif"), LANDSAT_BANDS)
            ls30, lc30 = to_30m(landsat, lc)
            n30 += save_chips(ls30, lc30, 64, f"{out}/30m/{split}", prefix)
        print(f"  1m chips: {n1}   30m chips: {n30}")


if __name__ == "__main__":
    main()