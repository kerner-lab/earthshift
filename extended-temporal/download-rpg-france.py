#!/usr/bin/env python
"""
Build a long-baseline temporal-shift task from the French RPG (LPIS) field
boundaries: ID = 2017, OOD = 2022.

Task formulation is deliberately identical to FTW so results drop straight into
the existing benchmark: 256x256 chips, 4-band S2 (R, G, B, NIR) uint16 at 10 m,
3-class semantic segmentation (0 = background, 1 = field interior,
2 = field boundary), mIoU.

Why this pair
-------------
  * 2015 is the first RPG edition distributing parcels (2007-2014 are ilots only).
  * Sentinel-2B joins in March 2017, so 2017 is the first year with the full
    constellation revisit.
  * The CAP period ran 2014-2022, so 2017 -> 2022 is a 5-year gap inside a single
    policy regime. 2017 -> 2024 is longer but crosses the 2023 CAP reform, which
    injects annotation drift into the temporal gap.

Two things this script handles that are easy to get wrong
---------------------------------------------------------
  * Processing-baseline offset. S2 baseline 04.00 (Jan 2022) introduced a
    -1000 radiometric offset. Items are partitioned by `s2:processing_baseline`
    and the offset is removed per item, so 2017 and 2022 are on one scale.
    This is done per item, not per year, because ESA reprocessing means some
    2017 acquisitions are also served at baseline >= 04.00.
  * Parcel churn. Parcels split, merge and leave the scheme between editions.
    Every chip stores the 2017/2022 field-mask IoU and a `stable` flag so the
    task can be reported both unrestricted and restricted to stable geometry.

Format note: contrary to what the IGN docs imply, whole-France GeoPackage only
exists from the 2019/2020 editions onward. The 2017 edition is distributed as a
national Shapefile, so it is converted to GPKG in the `prep` stage to get a
spatial index (bbox queries against a 2.9 GB unindexed shapefile are hopeless).

Usage
-----
  python download-rpg-france.py --stage fetch                 # ~5.6 GB of archives
  python download-rpg-france.py --stage prep                  # shp -> indexed gpkg
  python download-rpg-france.py --stage chips --n-chips 5000
  python download-rpg-france.py --stage labels
  python download-rpg-france.py --stage images --limit 20 --plot   # pilot
  caffeinate -i python download-rpg-france.py --stage images 2>&1 | tee rpg.log
"""

import argparse
import hashlib
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# GDAL/COG tuning, set before rasterio initialises its GDAL environment.
# Each composite is a few hundred serial range reads against Azure blob storage,
# so this stage is latency-bound rather than bandwidth-bound: HTTP/2 multiplexing
# and skipping the per-open directory listing matter far more than throughput.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.jp2")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "67108864")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

# --------------------------------------------------------------------------
# RPG editions. Names verified against the GeoPF download feed
# (https://data.geopf.fr/telechargement/resource/RPG); both are single-volume
# ~2.8 GB 7z archives.
# --------------------------------------------------------------------------
RPG = {
    2017: ("RPG_2-0__SHP_LAMB93_FR-2017_2017-01-01", ".7z"),
    2022: ("RPG_2-0__GPKG_LAMB93_FXX_2022-01-01", ".7z.001"),
}
RPG_URL = "https://data.geopf.fr/telechargement/download/RPG/{name}/{name}{ext}"

LAMB93 = "EPSG:2154"
# Metropolitan France extent in Lambert-93, used to seed candidate chips.
FRANCE_BBOX = (99000.0, 6046000.0, 1242000.0, 7111000.0)

# S2 L2A: red, green, blue, NIR -- order matches data_manager.ftw_mean/ftw_std.
S2_BANDS = ["B04", "B03", "B02", "B08"]
# SCL classes to reject: nodata, saturated, cloud shadow, cloud medium/high,
# thin cirrus, snow.
SCL_REJECT = (0, 1, 3, 8, 9, 10, 11)
BASELINE_OFFSET = 1000  # baseline >= 04.00 carries BOA_ADD_OFFSET = -1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["all", "fetch", "prep", "chips", "labels", "images"])
    p.add_argument("--raw", default="/Users/kdoerkse/earthshift_data/rpg_raw",
                   help="where the RPG archives are downloaded and extracted")
    p.add_argument("--out", default="/Users/kdoerkse/earthshift_data/rpg_france")
    p.add_argument("--id-year", type=int, default=2017)
    p.add_argument("--ood-year", type=int, default=2022)
    p.add_argument("--patch", type=int, default=256, help="chip size in px (FTW = 256)")
    p.add_argument("--res", type=float, default=10.0, help="ground sampling in m")
    p.add_argument("--n-chips", type=int, default=5000, help="target kept chips")
    p.add_argument("--n-candidates", type=int, default=120_000,
                   help="random grid cells to test before giving up")
    p.add_argument("--min-field-frac", type=float, default=0.20,
                   help="min fraction of chip covered by parcels, required in BOTH years")
    p.add_argument("--stable-iou", type=float, default=0.90,
                   help="parcel-BOUNDARY IoU above which a chip is flagged stable "
                        "(footprint IoU is also recorded but is a poor proxy)")
    p.add_argument("--boundary-width", type=float, default=20.0,
                   help="parcel boundary thickness in m (20 m = 2 px at 10 m)")
    p.add_argument("--block-km", type=float, default=100.0,
                   help="spatial block size for the train/val/test split")
    p.add_argument("--buffer-blocks", action="store_true",
                   help="drop chips within one chip-width of a block edge")
    p.add_argument("--window-a", default="04-01/06-15", help="early-season window MM-DD/MM-DD")
    p.add_argument("--window-b", default="07-15/09-15", help="late-season window MM-DD/MM-DD")
    p.add_argument("--max-cloud", type=int, default=60, help="scene-level %%")
    p.add_argument("--max-items", type=int, default=12,
                   help="scenes per composite, stratified over the window. Equalises "
                        "the observation count between ID and OOD years; 0 = use all")
    p.add_argument("--max-nan", type=float, default=0.05,
                   help="max masked/NaN pixel frac in the composite")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="stop after N new chips")
    p.add_argument("--plot", action="store_true", help="save RGB+label PNGs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_tls = threading.local()


def s2_catalog():
    """One pystac client per thread; the client is not documented as thread-safe."""
    if not hasattr(_tls, "cat"):
        import planetary_computer as pc
        import pystac_client
        _tls.cat = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
    return _tls.cat


def save_npy_atomic(path, arr):
    tmp = f"{path}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_tif(path, arr, transform, crs, nodata=None):
    """arr: (bands, H, W) or (H, W). Written atomically."""
    if arr.ndim == 2:
        arr = arr[None]
    tmp = f"{path}.tmp"
    with rasterio.open(
        tmp, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=arr.dtype, crs=crs, transform=transform,
        nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(arr)
    os.replace(tmp, path)


def permanent(why):
    """no_items / cloudy_* describe the chip; exceptions are usually transient."""
    return "exception" not in why


def sevenzip_extract(archive, dest):
    """py7zr if present, else a system 7z binary. Both handle .7z and .7z.001."""
    try:
        import py7zr
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(path=dest)
        return
    except ImportError:
        pass
    for exe in ("7zz", "7z", "7za"):
        if shutil.which(exe):
            subprocess.run([exe, "x", "-y", f"-o{dest}", archive], check=True)
            return
    sys.exit(
        "No 7z extractor found. Install one of:\n"
        "  pip install py7zr          (no admin needed)\n"
        "  brew install sevenzip      (provides 7zz)"
    )


def find_parcel_layer(root, year):
    """Locate the parcel layer for an extracted edition. Returns (path, layer|None)."""
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            low = f.lower()
            if low.endswith(".shp") and "parcelle" in low:
                hits.append((os.path.join(dirpath, f), None))
            elif low.endswith(".gpkg"):
                path = os.path.join(dirpath, f)
                for lyr in pyogrio.list_layers(path)[:, 0]:
                    if "parcelle" in lyr.lower():
                        hits.append((path, lyr))
    if not hits:
        sys.exit(
            f"No parcel layer found under {root} for {year}. Editions before 2015 "
            "ship ilots (blocks) only and cannot be used for parcel boundaries."
        )
    # Prefer the largest file: RPG ships small departmental extras alongside.
    hits.sort(key=lambda h: os.path.getsize(h[0]), reverse=True)
    return hits[0]


def parcels_gpkg(args, year):
    return f"{args.raw}/parcelles_{year}.gpkg"


# --------------------------------------------------------------------------
# stage: fetch
# --------------------------------------------------------------------------

def stage_fetch(args):
    os.makedirs(args.raw, exist_ok=True)
    for year in (args.id_year, args.ood_year):
        if year not in RPG:
            sys.exit(f"No RPG edition registered for {year}; known: {sorted(RPG)}")
        name, ext = RPG[year]
        url = RPG_URL.format(name=name, ext=ext)
        archive = f"{args.raw}/{name}{ext}"
        extract_dir = f"{args.raw}/{name}"

        if os.path.isdir(extract_dir) and os.listdir(extract_dir):
            print(f"{year}: already extracted -> {extract_dir}", flush=True)
            continue

        if not os.path.exists(archive):
            print(f"{year}: downloading {url}", flush=True)
            # curl -C - resumes a partial file, which matters at ~2.8 GB.
            subprocess.run(["curl", "-L", "-C", "-", "--fail", "-o", archive, url],
                           check=True)

        md5_url = RPG_URL.format(name=name, ext=".md5")
        try:
            md5_txt = subprocess.run(["curl", "-sL", "--fail", md5_url],
                                     capture_output=True, text=True, check=True).stdout
            want = md5_txt.split()[0].strip()
            h = hashlib.md5()
            with open(archive, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(blk)
            if h.hexdigest() != want:
                sys.exit(f"{year}: md5 mismatch on {archive}; delete it and refetch")
            print(f"{year}: md5 ok", flush=True)
        except subprocess.CalledProcessError:
            print(f"{year}: no md5 published, skipping checksum", flush=True)

        print(f"{year}: extracting", flush=True)
        os.makedirs(extract_dir, exist_ok=True)
        sevenzip_extract(archive, extract_dir)
        print(f"{year}: extracted -> {extract_dir}", flush=True)


# --------------------------------------------------------------------------
# stage: prep  (normalise both years to spatially indexed GPKG)
# --------------------------------------------------------------------------

def stage_prep(args):
    for year in (args.id_year, args.ood_year):
        dst = parcels_gpkg(args, year)
        if os.path.exists(dst):
            print(f"{year}: {dst} exists", flush=True)
            continue
        name, _ = RPG[year]
        src, layer = find_parcel_layer(f"{args.raw}/{name}", year)
        print(f"{year}: {src}" + (f" [{layer}]" if layer else ""), flush=True)

        cmd = ["ogr2ogr", "-f", "GPKG", "-t_srs", LAMB93, "-nln", "parcelles",
               "-nlt", "MULTIPOLYGON", "-skipfailures", dst, src]
        if layer:
            cmd.append(layer)
        subprocess.run(cmd, check=True)
        # ogr2ogr builds an rtree on the GPKG geometry column automatically, which
        # is what makes the per-chip bbox queries in later stages tractable.
        n = pyogrio.read_info(dst, layer="parcelles")["features"]
        print(f"{year}: wrote {dst}  ({n:,} parcels)", flush=True)


# --------------------------------------------------------------------------
# stage: chips
# --------------------------------------------------------------------------

def chip_transform(x0, y1, args):
    return from_origin(x0, y1, args.res, args.res)


def field_masks(args, x0, y0, x1, y1):
    """Interior/boundary masks per year for one chip, or None if a year is empty."""
    out = {}
    for year in (args.id_year, args.ood_year):
        gdf = pyogrio.read_dataframe(
            parcels_gpkg(args, year), layer="parcelles",
            bbox=(x0, y0, x1, y1), columns=[],
        )
        if len(gdf) == 0:
            return None
        geoms = gdf.geometry.values
        tf = chip_transform(x0, y1, args)
        shape = (args.patch, args.patch)

        interior = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=tf,
                             fill=0, all_touched=False, dtype="uint8")
        # Boundary drawn as a buffered ring so it survives rasterisation at 10 m.
        rings = [g.boundary.buffer(args.boundary_width / 2.0) for g in geoms]
        edge = rasterize([(g, 1) for g in rings], out_shape=shape, transform=tf,
                         fill=0, all_touched=True, dtype="uint8")

        lab = np.zeros(shape, dtype="uint8")
        lab[interior == 1] = 1
        lab[edge == 1] = 2  # boundary wins over interior, as in FTW
        out[year] = lab
    return out


def stage_chips(args):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cell = args.patch * args.res  # 2560 m at FTW settings
    x0b, y0b, x1b, y1b = FRANCE_BBOX
    ni = int((x1b - x0b) // cell)
    nj = int((y1b - y0b) // cell)

    # Sample distinct grid cells rather than scanning the country: most of France
    # is not cropland, and a bbox query per candidate is cheap against the rtree.
    cand = rng.choice(ni * nj, size=min(args.n_candidates, ni * nj), replace=False)
    print(f"grid {ni} x {nj} cells of {cell:.0f} m; testing {len(cand):,} candidates",
          flush=True)

    recs = []
    t0 = time.time()
    for n, c in enumerate(cand, 1):
        if len(recs) >= args.n_chips:
            break
        i, j = int(c % ni), int(c // ni)
        x0 = x0b + i * cell
        y0 = y0b + j * cell
        x1, y1 = x0 + cell, y0 + cell

        try:
            masks = field_masks(args, x0, y0, x1, y1)
        except Exception:
            continue
        if masks is None:
            continue

        a = masks[args.id_year]
        b = masks[args.ood_year]
        fa = float((a > 0).mean())
        fb = float((b > 0).mean())
        if fa < args.min_field_frac or fb < args.min_field_frac:
            continue

        # Two different notions of "unchanged", and they disagree a lot.
        # Footprint IoU asks whether the land is still farmed; it sits near 1.0
        # almost everywhere because cropland stays cropland. Boundary IoU asks
        # whether the parcelisation is the same, which is what the 3-class task
        # actually predicts -- a field splitting in two barely moves the
        # footprint but rewrites the boundaries. `stable` keys on the boundary.
        def iou(m1, m2):
            u = float((m1 | m2).sum())
            return float((m1 & m2).sum()) / u if u else 0.0

        iou_foot = iou(a > 0, b > 0)
        iou_bnd = iou(a == 2, b == 2)

        recs.append({
            "aoi_id": f"fr_{i:07d}-{j:07d}",
            "i": i, "j": j,
            "field_frac_id": fa, "field_frac_ood": fb,
            "iou_footprint": iou_foot, "iou_boundary": iou_bnd,
            "stable": bool(iou_bnd >= args.stable_iou),
            "geometry": box(x0, y0, x1, y1),
        })
        if n % 500 == 0 or len(recs) >= args.n_chips:
            el = time.time() - t0
            print(f"  {n:,} tested  {len(recs):,} kept  {el/n*1000:.0f} ms/candidate",
                  flush=True)

    if not recs:
        sys.exit("No chips met the coverage thresholds; lower --min-field-frac")

    chips = gpd.GeoDataFrame(recs, crs=LAMB93)

    # Spatially blocked split, matching the protocol used elsewhere in the paper.
    blk = args.block_km * 1000.0
    cx = chips.geometry.centroid.x
    cy = chips.geometry.centroid.y
    chips["blk"] = ((cx // blk).astype(int).astype(str) + "_" +
                    (cy // blk).astype(int).astype(str))

    if args.buffer_blocks:
        # Block assignment alone allows two chips in different splits to sit a
        # chip-width apart across a block edge; drop the boundary band.
        edge = (np.minimum(cx % blk, blk - cx % blk) < cell) | \
               (np.minimum(cy % blk, blk - cy % blk) < cell)
        print(f"buffer-blocks: dropping {int(edge.sum())} edge chips", flush=True)
        chips = chips[~edge].reset_index(drop=True)

    blocks = rng.permutation(chips.blk.unique())
    nb = len(blocks)
    tr, va = blocks[: int(0.7 * nb)], blocks[int(0.7 * nb): int(0.8 * nb)]
    chips["split"] = np.where(chips.blk.isin(tr), "train",
                              np.where(chips.blk.isin(va), "val", "test"))

    dst = f"{args.out}/chips_france.parquet"
    chips.to_parquet(dst)
    print(f"\n{len(chips):,} chips -> {dst}", flush=True)
    print(chips.split.value_counts().to_string(), flush=True)
    print(f"median IoU  footprint {chips.iou_footprint.median():.3f}   "
          f"boundary {chips.iou_boundary.median():.3f}", flush=True)
    print(f"stable (boundary IoU >= {args.stable_iou}): {int(chips.stable.sum()):,} "
          f"({100*chips.stable.mean():.1f}%)", flush=True)


# --------------------------------------------------------------------------
# stage: labels
# --------------------------------------------------------------------------

def stage_labels(args):
    chips = gpd.read_parquet(f"{args.out}/chips_france.parquet")
    for year in (args.id_year, args.ood_year):
        os.makedirs(f"{args.out}/{year}/label_masks/semantic_3class", exist_ok=True)

    cell = args.patch * args.res
    done = 0
    for _, r in chips.iterrows():
        paths = {y: f"{args.out}/{y}/label_masks/semantic_3class/{r.aoi_id}.tif"
                 for y in (args.id_year, args.ood_year)}
        if all(os.path.exists(p) for p in paths.values()):
            continue
        x0, y0, _, y1 = r.geometry.bounds
        masks = field_masks(args, x0, y0, x0 + cell, y0 + cell)
        if masks is None:
            continue
        tf = chip_transform(x0, y1, args)
        for y in (args.id_year, args.ood_year):
            write_tif(paths[y], masks[y], tf, LAMB93, nodata=0)
        done += 1
        if done % 200 == 0:
            print(f"  {done} chips labelled", flush=True)
    print(f"labels written for {done} chips", flush=True)


# --------------------------------------------------------------------------
# stage: images
# --------------------------------------------------------------------------

# Planetary Computer intermittently returns 502/OriginConnectionAborted from
# Azure Front Door. It is server-side flakiness rather than rate limiting, and it
# clears on a retry -- without this, one blip discards all four composites of a
# chip that is otherwise fine.
TRANSIENT = ("APIError", "502", "503", "504", "Timeout", "timed out",
             "Connection", "RasterioIO", "CURL", "ServerDisconnected")


def with_retry(fn, tries=4, base=2.0):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            sig = f"{type(e).__name__} {e}"[:400]
            if k == tries - 1 or not any(s in sig for s in TRANSIENT):
                raise
            time.sleep(base * (2 ** k) + random.random())


def select_items(items, n):
    """Least-cloudy scene from each of n equal sub-periods of the search window.

    Two reasons this is not just a speed knob. Sentinel-2 revisit is not constant
    across the study period -- April-June 2017 is Sentinel-2A only (2B is not
    operational until mid-2017), so an uncapped 2017 composite is built from
    ~40 scenes against ~100 for 2022. Taking a fixed count equalises the
    observation depth between ID and OOD, so composite quality is not itself a
    source of the temporal shift. Stratifying rather than taking the globally
    least-cloudy n keeps the scenes spread across the season, which stops the
    median from drifting to a different phenological stage in one year.
    """
    if not n or len(items) <= n:
        return list(items)

    def cloud(it):
        return it.properties.get("eo:cloud_cover", 100.0)

    ts = sorted(items, key=lambda it: it.datetime)
    lo, hi = ts[0].datetime, ts[-1].datetime
    span = (hi - lo).total_seconds() or 1.0

    bins = {}
    for it in ts:
        k = min(int((it.datetime - lo).total_seconds() / span * n), n - 1)
        if k not in bins or cloud(it) < cloud(bins[k]):
            bins[k] = it

    chosen = list(bins.values())
    if len(chosen) < n:  # empty sub-periods: backfill with the clearest leftovers
        rest = sorted((it for it in ts if it not in chosen), key=cloud)
        chosen += rest[: n - len(chosen)]
    return sorted(chosen, key=lambda it: it.datetime)


def composite(geobox, bbox4326, year, window, args):
    """Baseline-harmonised, cloud-masked median S2 composite, or None."""
    import odc.stac

    start, end = window.split("/")
    items = with_retry(lambda: s2_catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=bbox4326,
        datetime=f"{year}-{start}/{year}-{end}",
        query={"eo:cloud_cover": {"lt": args.max_cloud}},
    ).item_collection())
    if len(items) == 0:
        return None, "no_items"

    items = select_items(items, args.max_items)

    # Split by processing baseline: >= 04.00 carries BOA_ADD_OFFSET = -1000.
    # Done per item because ESA reprocessing puts some pre-2022 acquisitions on
    # the new baseline too, so a per-year rule would be wrong.
    def baseline(it):
        try:
            return float(it.properties.get("s2:processing_baseline", "0"))
        except (TypeError, ValueError):
            return 0.0

    groups = {"old": [it for it in items if baseline(it) < 4.0],
              "new": [it for it in items if baseline(it) >= 4.0]}

    stacks = []
    for tag, its in groups.items():
        if not its:
            continue
        ds = with_retry(lambda its=its: odc.stac.load(
            its, bands=S2_BANDS + ["SCL"], geobox=geobox,
            resampling={"SCL": "nearest", "*": "bilinear"},
            chunks=None,  # eager: nested dask inside the thread pool oversubscribes
            dtype="float32",
        ))
        scl = ds.SCL.values
        bad = np.isin(scl, SCL_REJECT)
        arr = ds[S2_BANDS].to_array().values  # (band, time, y, x)
        if tag == "new":
            arr = arr - BASELINE_OFFSET
        arr = np.where(bad[None], np.nan, arr)
        stacks.append(arr)

    arr = np.concatenate(stacks, axis=1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(arr, axis=1)  # (band, y, x)

    nan_frac = float(np.isnan(med).mean())
    if nan_frac > args.max_nan:
        return None, f"cloudy_{nan_frac:.2f}"

    med = np.nan_to_num(med, nan=0.0)
    # Offset removal can push dark pixels below zero; clip before the uint16 cast.
    return np.clip(med, 0, 65535).astype("uint16"), "ok"


def save_plot(out, aoi, imgs, labs, years):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def stretch(a):
        a = a.astype("float32")
        lo, hi = np.percentile(a, [2, 98])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    for k, y in enumerate(years):
        rgb = np.dstack([imgs[y][0], imgs[y][1], imgs[y][2]])  # already R,G,B
        ax[k, 0].imshow(stretch(rgb)); ax[k, 0].set_title(f"S2 {y}")
        ax[k, 1].imshow(labs[y], interpolation="nearest", vmin=0, vmax=2)
        ax[k, 1].set_title(f"RPG {y}")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}/plots/{aoi}.png", dpi=110)
    plt.close(fig)


def stage_images(args):
    from odc.geo.geobox import GeoBox

    chips = gpd.read_parquet(f"{args.out}/chips_france.parquet")
    chips_ll = chips.to_crs(4326)
    windows = {"window_a": args.window_a, "window_b": args.window_b}
    years = (args.id_year, args.ood_year)

    for sub in ("failed", "plots"):
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)
    for y in years:
        for w in windows:
            os.makedirs(f"{args.out}/{y}/s2_images/{w}", exist_ok=True)

    # A chip counts as done only when every (year, window) exists, so ID and OOD
    # always cover exactly the same geography.
    def paths_for(aoi):
        return {(y, w): f"{args.out}/{y}/s2_images/{w}/{aoi}.tif"
                for y in years for w in windows}

    todo = [r for _, r in chips.iterrows()
            if not all(os.path.exists(p) for p in paths_for(r.aoi_id).values())
            and not os.path.exists(f"{args.out}/failed/{r.aoi_id}")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(chips)} chips total, {len(todo)} to process", flush=True)

    cell = args.patch * args.res

    def do_chip(r):
        try:
            x0, y0 = r.geometry.bounds[0], r.geometry.bounds[1]
            y1 = y0 + cell
            tf = chip_transform(x0, y1, args)
            gb = GeoBox((args.patch, args.patch), tf, LAMB93)
            b = chips_ll.loc[r.name, "geometry"].bounds
            pad = 0.02
            bbox = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]

            imgs = {}
            for y in years:
                for w, span in windows.items():
                    a, why = composite(gb, bbox, y, span, args)
                    if a is None:
                        return r.aoi_id, None, f"{y}:{w}:{why}"
                    imgs[(y, w)] = a

            for (y, w), a in imgs.items():
                write_tif(f"{args.out}/{y}/s2_images/{w}/{r.aoi_id}.tif",
                          a, tf, LAMB93)

            if args.plot:
                labs = {}
                for y in years:
                    lp = f"{args.out}/{y}/label_masks/semantic_3class/{r.aoi_id}.tif"
                    if os.path.exists(lp):
                        with rasterio.open(lp) as s:
                            labs[y] = s.read(1)
                if len(labs) == len(years):
                    save_plot(args.out, r.aoi_id,
                              {y: imgs[(y, "window_a")] for y in years}, labs, years)
            return r.aoi_id, True, "ok"
        except Exception as e:
            traceback.print_exc()
            return r.aoi_id, None, f"exception:{type(e).__name__}"

    t0, ok, bad = time.time(), [], {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(do_chip, r) for r in todo]
        for n, f in enumerate(as_completed(futs), 1):
            aoi, good, why = f.result()
            if good:
                ok.append(aoi)
            else:
                bad[aoi] = why
                if permanent(why):
                    with open(f"{args.out}/failed/{aoi}", "w") as fh:
                        fh.write(why)
            if n % 25 == 0 or n == len(futs):
                el = time.time() - t0
                print(f"{n}/{len(futs)}  ok={len(ok)}  fail={len(bad)}  "
                      f"{el/n:.2f}s/chip  eta={(len(futs)-n)*el/n/60:.1f}min", flush=True)

    if bad:
        from collections import Counter
        print("\nfailure reasons:", Counter(bad.values()).most_common(), flush=True)
        retry = sum(1 for w in bad.values() if not permanent(w))
        if retry:
            print(f"{retry} transient failures left unmarked - rerun to retry", flush=True)

    complete = [r.aoi_id for _, r in chips.iterrows()
                if all(os.path.exists(p) for p in paths_for(r.aoi_id).values())]
    final = chips[chips.aoi_id.isin(complete)]
    dst = f"{args.out}/chips_france_final.parquet"
    final.to_parquet(dst)
    print(f"\n{len(final)} complete chips -> {dst}", flush=True)
    print(f"of which stable: {int(final.stable.sum())}", flush=True)

    # Per-year copies under the FTW naming convention. FTWDataset looks for
    # {data_dir}/{country}/chips_{country}.parquet, and the year takes the
    # country slot here, so the loader picks these up with no special-casing.
    for y in years:
        yd = f"{args.out}/{y}/chips_{y}.parquet"
        final.to_parquet(yd)
        print(f"  -> {yd}", flush=True)


def main():
    args = parse_args()
    stages = ["fetch", "prep", "chips", "labels", "images"] \
        if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"\n=== stage: {s} ===", flush=True)
        globals()[f"stage_{s}"](args)


if __name__ == "__main__":
    main()
