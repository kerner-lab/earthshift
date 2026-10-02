#!/usr/bin/env python
"""
Extract Landsat-8 summer composites onto the NLCD grid for the EarthShift
temporal-shift task (ID year -> OOD year, identical footprints).

A tile is written only if BOTH years succeed, so the ID and OOD sets always
cover exactly the same geography.

Usage
-----
  python download-landsat.py --limit 20 --plot          # pilot + alignment check
  python download-landsat.py --limit 500                # timing run
  caffeinate -i python download-landsat.py 2>&1 | tee extract.log   # full run
"""

import argparse
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

BANDS = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]

# Landsat Collection 2 QA_PIXEL bits to reject
QA_BITS = {0: "fill", 1: "dilated_cloud", 2: "cirrus", 3: "cloud", 4: "cloud_shadow"}
FILL_BIT = 0

TILES_DEFAULT = "/Users/kdoerkse/earthshift_data/nlcd/nlcd_tiles.gpkg"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default=TILES_DEFAULT)
    p.add_argument("--nlcd-dir", default="/Users/kdoerkse/earthshift_data/nlcd")
    p.add_argument("--out", default="/Users/kdoerkse/earthshift_data/nlcd/chips")
    p.add_argument("--id-year", type=int, default=2013)
    p.add_argument("--ood-year", type=int, default=2021)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--season", default="06-01/08-31", help="MM-DD/MM-DD")
    p.add_argument("--max-cloud", type=int, default=80, help="scene-level %%")
    p.add_argument("--max-nan", type=float, default=0.05,
                   help="max masked/NaN pixel frac in the composite")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="stop after N new tiles")
    p.add_argument("--plot", action="store_true", help="save RGB+label PNGs")
    return p.parse_args()


# rasterio/GDAL dataset objects are not thread-safe, so each worker thread gets
# its own handle rather than sharing one across the pool.
_tls = threading.local()


def nlcd_src(year, path):
    if not hasattr(_tls, "src"):
        _tls.src = {}
    if year not in _tls.src:
        _tls.src[year] = rasterio.open(path)
    return _tls.src[year]


def save_npy(path, arr):
    """Atomic write: a crash mid-save must not leave a truncated done-marker."""
    tmp = f"{path}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def permanent(why):
    """no_items / cloudy_* are properties of the tile; exceptions are usually transient."""
    return "exception" not in why


def build_catalog():
    import planetary_computer as pc
    import pystac_client

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )


def make_geobox(nlcd_transform, nlcd_crs, row, col, patch):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    tf = nlcd_transform * Affine.translation(int(col), int(row))
    return GeoBox((patch, patch), tf, nlcd_crs)


def composite(cat, geobox, bbox4326, year, args):
    """Cloud-masked median composite on the NLCD grid, or None if unusable."""
    import odc.stac

    start, end = args.season.split("/")
    items = cat.search(
        collections=["landsat-c2-l2"],
        bbox=bbox4326,
        datetime=f"{year}-{start}/{year}-{end}",
        query={
            "platform": {"in": ["landsat-8"]},
            "eo:cloud_cover": {"lt": args.max_cloud},
        },
    ).item_collection()
    if len(items) == 0:
        return None, "no_items"

    ds = odc.stac.load(
        items,
        bands=BANDS + ["qa_pixel"],
        geobox=geobox,
        resampling={"qa_pixel": "nearest", "*": "bilinear"},
        chunks=None,  # eager: nested dask schedulers inside the thread pool oversubscribe
    )

    qa = ds.qa_pixel.values  # materialise once, not once per bit
    bad = np.zeros(qa.shape, dtype=bool)
    for bit in QA_BITS:
        bad |= ((qa >> bit) & 1).astype(bool)

    # QA resamples nearest but the bands resample bilinear, so scene-edge fill
    # bleeds one pixel into neighbours that QA still calls clean. Dilate the
    # fill mask in (y, x) only, never across time.
    fill = ((qa >> FILL_BIT) & 1).astype(bool)
    bad |= binary_dilation(fill, structure=np.ones((1, 3, 3), dtype=bool))

    med = ds[BANDS].where(~bad).median("time", skipna=True)
    arr = med.to_array().values  # (bands, patch, patch)

    nan_frac = float(np.isnan(arr).mean())
    if nan_frac > args.max_nan:
        return None, f"cloudy_{nan_frac:.2f}"

    return np.nan_to_num(arr).astype("uint16"), "ok"


def save_plot(out, idx, rgb13, rgb21, lab13, lab21):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stretch(a):
        a = a.astype("float32")
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    ax[0, 0].imshow(stretch(rgb13)); ax[0, 0].set_title("Landsat ID year")
    ax[0, 1].imshow(lab13, interpolation="nearest"); ax[0, 1].set_title("NLCD ID year")
    ax[1, 0].imshow(stretch(rgb21)); ax[1, 0].set_title("Landsat OOD year")
    ax[1, 1].imshow(lab21, interpolation="nearest"); ax[1, 1].set_title("NLCD OOD year")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}/plots/{idx}.png", dpi=110)
    plt.close(fig)


def main():
    args = parse_args()

    for sub in ("failed", "plots"):
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)
    for y in (args.id_year, args.ood_year):
        os.makedirs(f"{args.out}/{y}", exist_ok=True)
        os.makedirs(f"{args.out}/labels_{y}", exist_ok=True)

    tiles = gpd.read_file(args.tiles)
    tiles_ll = tiles.to_crs(4326)

    nlcd_paths = {
        y: f"{args.nlcd_dir}/Annual_NLCD_LndCov_{y}_CU_C1V1.tif"
        for y in (args.id_year, args.ood_year)
    }
    for p in nlcd_paths.values():
        if not os.path.exists(p):
            sys.exit(f"missing NLCD raster: {p}")

    # One footprint list serves both years only if the grids are identical --
    # the same (row, col) window is applied to each year's raster below.
    grid = {}
    for y, p in nlcd_paths.items():
        with rasterio.open(p) as s:
            grid[y] = (s.crs, s.transform, s.width, s.height)
    if grid[args.id_year] != grid[args.ood_year]:
        sys.exit(
            f"NLCD grids differ between {args.id_year} and {args.ood_year}; "
            "a shared footprint list is not valid"
        )
    nlcd_crs, nlcd_transform = grid[args.id_year][0], grid[args.id_year][1]

    done_marker = lambda i: f"{args.out}/{args.ood_year}/{i}.npy"
    fail_marker = lambda i: f"{args.out}/failed/{i}"

    todo = [
        (i, t)
        for i, t in tiles.iterrows()
        if not os.path.exists(done_marker(i)) and not os.path.exists(fail_marker(i))
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(tiles)} tiles total, {len(todo)} to process", flush=True)

    cat = build_catalog()

    def do_tile(i, t):
        try:
            gb = make_geobox(nlcd_transform, nlcd_crs, t.row, t.col, args.patch)
            b = tiles_ll.loc[i, "geometry"].bounds  # (minx, miny, maxx, maxy)
            pad = 0.02
            bbox = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]

            arrs, labs = {}, {}
            for y in (args.id_year, args.ood_year):
                a, why = composite(cat, gb, bbox, y, args)
                if a is None:
                    return i, None, f"{y}:{why}"
                arrs[y] = a
                labs[y] = nlcd_src(y, nlcd_paths[y]).read(
                    1,
                    window=rasterio.windows.Window(
                        int(t.col), int(t.row), args.patch, args.patch
                    ),
                )

            save_npy(f"{args.out}/labels_{args.id_year}/{i}.npy", labs[args.id_year])
            save_npy(f"{args.out}/labels_{args.ood_year}/{i}.npy", labs[args.ood_year])
            save_npy(f"{args.out}/{args.id_year}/{i}.npy", arrs[args.id_year])
            save_npy(done_marker(i), arrs[args.ood_year])  # written last = complete

            if args.plot:
                rgb = lambda a: np.dstack([a[3], a[2], a[1]])
                save_plot(
                    args.out, i,
                    rgb(arrs[args.id_year]), rgb(arrs[args.ood_year]),
                    labs[args.id_year], labs[args.ood_year],
                )
            return i, True, "ok"
        except Exception as e:
            traceback.print_exc()
            return i, None, f"exception:{type(e).__name__}"

    t0, ok, bad = time.time(), [], {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(do_tile, i, t) for i, t in todo]
        for n, f in enumerate(as_completed(futs), 1):
            i, good, why = f.result()
            if good:
                ok.append(i)
            else:
                bad[i] = why
                # Only mark permanent failures. Exceptions are usually transient
                # (STAC/network), and blocking them forever would drop tiles
                # non-randomly.
                if permanent(why):
                    with open(fail_marker(i), "w") as fh:
                        fh.write(why)
            if n % 25 == 0 or n == len(futs):
                el = time.time() - t0
                print(
                    f"{n}/{len(futs)}  ok={len(ok)}  fail={len(bad)}  "
                    f"{el/n:.2f}s/tile  eta={(len(futs)-n)*el/n/60:.1f}min",
                    flush=True,
                )

    if bad:
        from collections import Counter
        print("\nfailure reasons:", Counter(bad.values()).most_common(), flush=True)
        retryable = sum(1 for w in bad.values() if not permanent(w))
        if retryable:
            print(f"{retryable} transient failures left unmarked - rerun to retry them",
                  flush=True)

    complete = [i for i in tiles.index if os.path.exists(done_marker(i))]
    final = args.tiles.replace(".gpkg", "_final.gpkg")
    tiles.loc[complete].to_file(final, driver="GPKG")
    print(f"\n{len(complete)} paired tiles -> {final}")


if __name__ == "__main__":
    main()