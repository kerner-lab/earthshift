#!/usr/bin/env python
"""
Build a processing-level shift for BigEarthNet v2: ID = the L2A test split
already in EarthShift, OOD = the same test patches re-chipped from the matching
Sentinel-2 L1C (top-of-atmosphere) granules.

Every BenV2 patch ID embeds its source product, e.g.
    S2A_MSIL2A_20171002T112111_N9999_R037_T29SNC_62_45
    sat   level  sensing time   base orbit tile  col row
so the L1C counterpart is found by searching on (sat, sensing time, orbit, tile)
rather than by constructing the full product name.

Things this script handles that are easy to get wrong
-----------------------------------------------------
  * GEO-Bench geotransform. The local copy is GEO-Bench-2's
    geobench_benv2.tortilla, not the raw BenV2 tree. Its S2 GeoTIFFs stack all
    12 bands at 120x120 but carry B01's 60 m transform, so `src.bounds` spans
    7.2 km instead of 1.2 km. Patch bounds are therefore taken from the S2
    top-left corner + 1200 m and cross-checked against the (correct) S1 bounds.
  * Matching GEO-Bench's resampling. GEO-Bench upsampled every native BenV2
    band to 120x120 with rasterio bilinear (generate_benchmark/benv2.py). L1C
    bands are first window-read at native resolution (120/60/20 px, integer
    offsets, no resampling), written in BenV2's per-band layout, then put
    through the identical bilinear read. The resampling is thus the same on
    both sides and the only remaining difference is processing level.
  * Collection-1 reprocessing. The public GCS bucket serves both the original
    L1C (e.g. N0205) and the Collection-1 reprocessing (N0500) for the same
    datatake. The lowest baseline is used, since that is what Sen2Cor was run
    on for BenV2. Baseline >= 04.00 carries RADIO_ADD_OFFSET = -1000, which is
    removed if such a product is ever the only one available.

Output layout (root = --out)
----------------------------
  products.csv                      one row per test product: chosen SAFE + alternatives
  raw/<SAFE>/<band>.jp2             12 L1C band files per product (B10 dropped)
  BigEarthNet-S2-L1C/<product>/<patch_id>/<patch_id>_<band>.tif   native-res chips
  S1/<s1>.tif, S2/<patch_id>.tif    GEO-Bench-format files (S1 copied byte-for-byte)
  chips.csv                         per-patch status, drop reason, alignment checks
  qa/                               alignment / radiometry / completeness reports

Usage
-----
  python benv2-l1c.py --stage index                     # map all test patches -> L1C
  python benv2-l1c.py --stage download --tile 29SNC     # pilot on one tile
  python benv2-l1c.py --stage chip --tile 29SNC
  python benv2-l1c.py --stage qa --tile 29SNC
  python benv2-l1c.py --stage all                       # everything
"""

import argparse
import base64
import hashlib
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import requests
import tacoreader
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

# Band order and native resolutions, matching GeoBenchBENV2.band_default_order.
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B11", "B12"]
BAND_RES = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
            "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20}
PATCH_M = 1200            # BenV2 patches are 1.2 km on a side
GB_SIZE = 120             # GEO-Bench stacks every band at 120x120
BASELINE_OFFSET = 1000    # L1C baseline >= 04.00 carries RADIO_ADD_OFFSET = -1000

PATCH_RE = re.compile(r"^(S2[AB])_MSIL2A_(\d{8}T\d{6})_N\d{4}_R(\d{3})_T(\d{2}[A-Z]{3})_(\d+)_(\d+)$")
SAFE_RE = re.compile(r"^(S2[AB])_MSIL1C_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})\.SAFE$")

GCS_BUCKET = "gcp-public-data-sentinel-2"
GCS_LIST = f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
GCS_GET = f"https://storage.googleapis.com/{GCS_BUCKET}/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["all", "index", "download", "chip", "qa"])
    p.add_argument("--benv2", default="/Users/kdoerkse/earthshift_data/benv2",
                   help="dir holding the GEO-Bench geobench_benv2.tortilla (ID set)")
    p.add_argument("--out", default="/Users/kdoerkse/earthshift_data/benv2_l1c")
    p.add_argument("--split", default="test")
    p.add_argument("--tile", nargs="*", default=None,
                   help="restrict download/chip/qa to these MGRS tiles, e.g. 29SNC")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--keep-raw", action="store_true",
                   help="keep the L1C JP2s after chipping (default deletes them)")
    return p.parse_args()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def load_split(args):
    """Tortilla handle + metadata of the requested split, with parsed product fields."""
    tor = tacoreader.load(os.path.join(args.benv2, "geobench_benv2.tortilla"))
    meta = pd.DataFrame(tor.drop(columns=["internal:subfile"]))
    meta = meta[meta["tortilla:data_split"] == args.split].copy()
    parts = meta["patch_id"].str.extract(PATCH_RE.pattern)
    bad = parts[0].isna()
    if bad.any():
        raise ValueError(f"{bad.sum()} patch IDs do not parse, e.g. {meta.patch_id[bad].iloc[0]}")
    meta[["sat", "sensing", "orbit", "tile", "pcol", "prow"]] = parts.values
    meta["product_key"] = meta.sat + "_" + meta.sensing + "_R" + meta.orbit + "_T" + meta.tile
    return tor, meta


def gcs_list(prefix, delimiter=None):
    items, prefixes, token = [], [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["pageToken"] = token
        r = requests.get(GCS_LIST, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        items += j.get("items", [])
        prefixes += j.get("prefixes", [])
        token = j.get("nextPageToken")
        if not token:
            return items, prefixes


def tile_prefix(tile):
    return f"tiles/{int(tile[:2])}/{tile[2]}/{tile[3:]}/"


def select_tiles(df, args):
    if args.tile:
        df = df[df["tile"].isin(args.tile)]
        if df.empty:
            sys.exit(f"no {args.split} patches on tiles {args.tile}")
    return df


# --------------------------------------------------------------------------
# Stage 1: patch -> L1C product mapping
# --------------------------------------------------------------------------
def find_product(key):
    sat, sensing, orbit, tile = re.match(r"(S2[AB])_(\d{8}T\d{6})_R(\d{3})_T(\w{5})", key).groups()
    _, prefixes = gcs_list(f"{tile_prefix(tile)}{sat}_MSIL1C_{sensing}_", delimiter="/")
    cands = []
    for pre in prefixes:
        m = SAFE_RE.match(pre.rstrip("/").split("/")[-1])
        if m and m.group(4) == orbit and m.group(5) == tile:
            cands.append((int(m.group(3)), m.group(6), pre.rstrip("/").split("/")[-1]))
    cands.sort()  # lowest baseline first, then earliest generation
    return {"product_key": key,
            "safe": cands[0][2] if cands else None,
            "baseline": cands[0][0] if cands else None,
            "n_candidates": len(cands),
            "alternatives": ";".join(c[2] for c in cands[1:])}


def stage_index(args):
    _, meta = load_split(args)
    counts = meta.groupby(["product_key", "tile"]).size().rename("n_patches").reset_index()
    print(f"{len(meta)} {args.split} patches -> {len(counts)} unique products "
          f"on {meta.tile.nunique()} tiles")
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(find_product, k): k for k in counts.product_key}
        for f in as_completed(futs):
            rows.append(f.result())
    prods = counts.merge(pd.DataFrame(rows), on="product_key").sort_values("product_key")
    os.makedirs(args.out, exist_ok=True)
    prods.to_csv(os.path.join(args.out, "products.csv"), index=False)
    missing = prods[prods.safe.isna()]
    print(f"matched {len(prods) - len(missing)}/{len(prods)} products; "
          f"baselines used: {prods.baseline.value_counts().to_dict()}")
    if len(missing):
        print(f"NO L1C for {len(missing)} products ({missing.n_patches.sum()} patches):")
        print(missing[["product_key", "n_patches"]].to_string(index=False))
    multi = (prods.n_candidates > 1).sum()
    print(f"{multi} products had >1 L1C candidate (lowest baseline chosen; see products.csv)")


def load_products(args):
    path = os.path.join(args.out, "products.csv")
    if not os.path.exists(path):
        sys.exit("run --stage index first")
    prods = pd.read_csv(path, dtype={"orbit": str})
    return select_tiles(prods, args).dropna(subset=["safe"])


# --------------------------------------------------------------------------
# Stage 2: download the 12 band JP2s per product
# --------------------------------------------------------------------------
def gcs_md5_ok(path, md5_b64):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode() == md5_b64


def download_file(name, dst, md5_b64):
    if os.path.exists(dst) and gcs_md5_ok(dst, md5_b64):
        return "cached"
    tmp = dst + ".part"
    with requests.get(GCS_GET + name, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if not gcs_md5_ok(tmp, md5_b64):
        os.remove(tmp)
        raise IOError(f"md5 mismatch for {name}")
    os.replace(tmp, dst)
    return "downloaded"


def band_objects(safe, tile):
    items, _ = gcs_list(f"{tile_prefix(tile)}{safe}/GRANULE/")
    jp2 = {}
    for it in items:
        m = re.search(r"/IMG_DATA/T\w{5}_\d{8}T\d{6}_(B\w{2})\.jp2$", it["name"])
        if m and m.group(1) in BAND_RES:  # drops B10 and TCI
            if m.group(1) in jp2:
                raise ValueError(f"{safe}: more than one granule for {m.group(1)}")
            jp2[m.group(1)] = it
    missing = set(S2_BANDS) - set(jp2)
    if missing:
        raise ValueError(f"{safe}: missing bands {sorted(missing)}")
    return jp2


def raw_band_path(args, safe, band):
    return os.path.join(args.out, "raw", safe, f"{band}.jp2")


def stage_download(args):
    prods = load_products(args)
    jobs = []
    for p in prods.itertuples():
        os.makedirs(os.path.join(args.out, "raw", p.safe), exist_ok=True)
        for band, it in band_objects(p.safe, p.tile).items():
            jobs.append((it["name"], raw_band_path(args, p.safe, band), it["md5Hash"], int(it["size"])))
    total = sum(j[3] for j in jobs)
    print(f"{len(prods)} products, {len(jobs)} band files, {total / 1e9:.2f} GB")
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_file, *j[:3]): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                status = f.result()
            except Exception as e:
                print(f"FAILED {j[0]}: {e}")
                continue
            done += j[3]
            print(f"[{done / total:6.1%}] {status:10s} {os.path.relpath(j[1], args.out)}")


# --------------------------------------------------------------------------
# Stage 3: re-chip L1C onto the BenV2 grid
# --------------------------------------------------------------------------
def subfile_bytes(subfile):
    """Raw bytes of a /vsisubfile/<offset>_<length>,<path> entry."""
    m = re.match(r"/vsisubfile/(\d+)_(\d+),(.+)$", subfile)
    off, length, path = int(m.group(1)), int(m.group(2)), m.group(3)
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(length)


def exact_window(src, bounds, band):
    """Window for `bounds` in src; must land on integer pixels with native size."""
    w = from_bounds(*bounds, transform=src.transform)
    vals = np.array([w.col_off, w.row_off, w.width, w.height])
    if np.abs(vals - np.round(vals)).max() > 1e-6:
        raise ValueError(f"{band}: non-integer window {w}")
    col, row, width, height = np.round(vals).astype(int)
    n = PATCH_M // BAND_RES[band]
    if (width, height) != (n, n):
        raise ValueError(f"{band}: window {width}x{height}, expected {n}x{n}")
    if col < 0 or row < 0 or col + n > src.width or row + n > src.height:
        raise ValueError(f"{band}: window outside tile")
    return rasterio.windows.Window(col, row, n, n)


def chip_product(args, prod, patches, tor):
    """Re-chip every patch of one product. Returns one record per patch."""
    srcs = {b: rasterio.open(raw_band_path(args, prod.safe, b)) for b in S2_BANDS}
    offset = BASELINE_OFFSET if prod.baseline >= 400 else 0
    pdir = prod.product_key  # one dir per product, like BenV2's patch_dir
    recs = []
    try:
        for idx, row in patches.iterrows():
            rec = {"patch_id": row.patch_id, "product_key": prod.product_key,
                   "safe": prod.safe, "status": "ok", "reason": ""}
            try:
                sample = tor.read(idx)
                s1_sub, s2_sub = sample.read(0), sample.read(1)
                with rasterio.open(s2_sub) as s2, rasterio.open(s1_sub) as s1:
                    gb_profile = s2.profile.copy()
                    x0, y0 = s2.transform.c, s2.transform.f
                    crs, s1_bounds = s2.crs, tuple(s1.bounds)
                bounds = (x0, y0 - PATCH_M, x0 + PATCH_M, y0)
                if not np.allclose(bounds, s1_bounds):
                    raise ValueError(f"S2 top-left bounds {bounds} != S1 bounds {s1_bounds}")
                if srcs["B02"].crs != crs:
                    raise ValueError(f"CRS mismatch {srcs['B02'].crs} vs {crs}")
                # Patch-ID grid indices vs tile origin (record only; BenV2's own convention).
                tx, ty = srcs["B02"].transform.c, srcs["B02"].transform.f
                rec["grid_col"] = (x0 - tx) / PATCH_M
                rec["grid_row"] = (ty - y0) / PATCH_M
                rec["id_col"], rec["id_row"] = int(row.pcol), int(row.prow)

                ndir = os.path.join(args.out, "BigEarthNet-S2-L1C", pdir, row.patch_id)
                os.makedirs(ndir, exist_ok=True)
                stack = []
                for b in S2_BANDS:
                    src = srcs[b]
                    win = exact_window(src, bounds, b)
                    arr = src.read(1, window=win)
                    if (arr == 0).any():
                        raise ValueError(f"{b}: {(arr == 0).sum()} nodata px")
                    if offset:
                        arr = np.clip(arr.astype(np.int32) - offset, 0, None).astype(np.uint16)
                    res = BAND_RES[b]
                    npath = os.path.join(ndir, f"{row.patch_id}_{b}.tif")
                    with rasterio.open(npath, "w", driver="GTiff", height=arr.shape[0],
                                       width=arr.shape[1], count=1, dtype="uint16", crs=crs,
                                       transform=from_origin(x0, y0, res, res)) as dst:
                        dst.write(arr, 1)
                    # Identical call to geobench_v2/generate_benchmark/benv2.py.
                    with rasterio.open(npath) as nsrc:
                        stack.append(nsrc.read(indexes=1, out_shape=(GB_SIZE, GB_SIZE),
                                               out_dtype="int32",
                                               resampling=Resampling.bilinear))
                # Same profile as the ID file (incl. GEO-Bench's B01 transform), so the
                # two sets are interchangeable byte-layout-wise.
                s2_out = os.path.join(args.out, "S2", f"{row.patch_id}.tif")
                with rasterio.open(s2_out, "w", **gb_profile) as dst:
                    dst.write(np.stack(stack))
                s1_out = os.path.join(args.out, "S1", f"{row.patch_id}.tif")
                with open(s1_out, "wb") as f:
                    f.write(subfile_bytes(s1_sub))
            except Exception as e:
                rec["status"], rec["reason"] = "dropped", str(e)
            recs.append(rec)
    finally:
        for s in srcs.values():
            s.close()
    return recs


def stage_chip(args):
    tor, meta = load_split(args)
    prods = load_products(args)
    meta = select_tiles(meta, args)
    for d in ("S1", "S2"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    recs = []
    # Patches whose product has no L1C at all are dropped up front.
    nomatch = meta[~meta.product_key.isin(prods.product_key)]
    for pid in nomatch.patch_id:
        recs.append({"patch_id": pid, "status": "dropped", "reason": "no L1C product"})

    def run(prod):
        if not all(os.path.exists(raw_band_path(args, prod.safe, b)) for b in S2_BANDS):
            return [{"patch_id": pid, "product_key": prod.product_key, "status": "dropped",
                     "reason": "L1C not downloaded"}
                    for pid in meta[meta.product_key == prod.product_key].patch_id]
        out = chip_product(args, prod, meta[meta.product_key == prod.product_key], tor)
        if not args.keep_raw and all(r["status"] == "ok" for r in out):
            shutil.rmtree(os.path.join(args.out, "raw", prod.safe), ignore_errors=True)
        return out

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run, p): p for p in prods.itertuples()}
        for f in as_completed(futs):
            p = futs[f]
            try:
                out = f.result()
            except Exception:
                traceback.print_exc()
                continue
            nok = sum(r["status"] == "ok" for r in out)
            print(f"{p.product_key}: {nok}/{len(out)} ok")
            recs += out

    new = pd.DataFrame(recs)
    path = os.path.join(args.out, "chips.csv")
    if os.path.exists(path):  # merge with earlier (per-tile) runs
        old = pd.read_csv(path)
        new = pd.concat([old[~old.patch_id.isin(new.patch_id)], new])
    new.sort_values("patch_id").to_csv(path, index=False)
    cur = new[new.patch_id.isin(meta.patch_id)]
    print(f"\n{(cur.status == 'ok').sum()}/{len(cur)} patches chipped this run")
    if (cur.status != "ok").any():
        print(cur[cur.status != "ok"].reason.value_counts().to_string())


# --------------------------------------------------------------------------
# Stage 4: QA - alignment, radiometry, completeness
# --------------------------------------------------------------------------
def stage_qa(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.registration import phase_cross_correlation

    tor, meta = load_split(args)
    meta = select_tiles(meta, args)
    chips = pd.read_csv(os.path.join(args.out, "chips.csv"))
    chips = chips[chips.patch_id.isin(meta.patch_id)]
    qdir = os.path.join(args.out, "qa")
    os.makedirs(qdir, exist_ok=True)

    # --- Completeness ------------------------------------------------------
    ok_ids = set(chips[chips.status == "ok"].patch_id)
    unaccounted = set(meta.patch_id) - set(chips.patch_id)
    dropped = chips[chips.status != "ok"]
    dropped.to_csv(os.path.join(qdir, "dropped.csv"), index=False)
    print(f"[completeness] {len(ok_ids)}/{len(meta)} patches have a valid L1C chip; "
          f"{len(dropped)} dropped, {len(unaccounted)} never processed")
    if len(dropped):
        print(dropped.reason.str.split(":").str[0].value_counts().to_string())
    grid_ok = ((chips.grid_col == chips.id_col) & (chips.grid_row == chips.id_row))[chips.status == "ok"]
    print(f"[grid] patch-ID (col,row) matches tile-grid position for {grid_ok.mean():.1%} of patches")

    # --- Alignment + radiometry -------------------------------------------
    sub = meta[meta.patch_id.isin(ok_ids)]
    l2a_px = {b: [] for b in S2_BANDS}
    l1c_px = {b: [] for b in S2_BANDS}
    rows = []
    rng = np.random.default_rng(0)
    for idx, row in sub.iterrows():
        with rasterio.open(tor.read(idx).read(1)) as s:
            l2a = s.read().astype(np.float64)
        with rasterio.open(os.path.join(args.out, "S2", f"{row.patch_id}.tif")) as s:
            l1c = s.read().astype(np.float64)
        rec = {"patch_id": row.patch_id, "tile": row.tile}
        # B08 (10 m, so not resampled) and B11 (20 m) as stable reference bands.
        for b in ("B08", "B11"):
            i = S2_BANDS.index(b)
            shift, _, _ = phase_cross_correlation(l2a[i], l1c[i], upsample_factor=100)
            rec[f"{b}_dy"], rec[f"{b}_dx"] = shift
            rec[f"{b}_l2a_std"] = l2a[i].std()
        for i, b in enumerate(S2_BANDS):
            with np.errstate(invalid="ignore", divide="ignore"):
                rec[f"{b}_r"] = np.corrcoef(l2a[i].ravel(), l1c[i].ravel())[0, 1]
            rec[f"{b}_l2a_mean"], rec[f"{b}_l1c_mean"] = l2a[i].mean(), l1c[i].mean()
            pick = rng.choice(l2a[i].size, 400, replace=False)
            l2a_px[b].append(l2a[i].ravel()[pick])
            l1c_px[b].append(l1c[i].ravel()[pick])
        rows.append(rec)
    qa = pd.DataFrame(rows)
    qa.to_csv(os.path.join(qdir, "per_patch.csv"), index=False)

    # Phase correlation is meaningless on near-uniform patches (open water, where
    # Sen2Cor's L2A sits at a few DN), so alignment is summarised on textured ones.
    for b in ("B08", "B11"):
        t = qa[qa[f"{b}_l2a_std"] >= 20]
        mag = np.hypot(t[f"{b}_dy"], t[f"{b}_dx"])
        print(f"[alignment] {b} phase-corr shift (px), {len(t)}/{len(qa)} textured patches: "
              f"mean dy={t[f'{b}_dy'].mean():+.3f} dx={t[f'{b}_dx'].mean():+.3f} | "
              f"|shift| median={mag.median():.3f} p95={mag.quantile(.95):.3f} max={mag.max():.3f}")

    summ = pd.DataFrame({
        "band": S2_BANDS,
        "l2a_mean": [qa[f"{b}_l2a_mean"].mean() for b in S2_BANDS],
        "l1c_mean": [qa[f"{b}_l1c_mean"].mean() for b in S2_BANDS],
        "median_r": [qa[f"{b}_r"].median() for b in S2_BANDS],  # NaN r = constant L2A band
    })
    summ["l1c_minus_l2a"] = summ.l1c_mean - summ.l2a_mean
    summ["ratio"] = summ.l1c_mean / summ.l2a_mean
    summ.to_csv(os.path.join(qdir, "radiometry.csv"), index=False)
    print("[radiometry] per-band means (DN = reflectance x 1e4)")
    print(summ.round(3).to_string(index=False))

    # Per-band distributions: L2A (ID) vs L1C (OOD).
    l2a_c, l1c_c, ink, mute = "#2a78d6", "#eb6834", "#2b2b2b", "#8a8a85"
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5))
    for ax, b in zip(axes.ravel(), S2_BANDS):
        a, c = np.concatenate(l2a_px[b]), np.concatenate(l1c_px[b])
        hi = np.percentile(np.concatenate([a, c]), 99.5)
        bins = np.linspace(0, hi, 60)
        ax.hist(a, bins=bins, histtype="step", lw=2, color=l2a_c, label="L2A (ID)")
        ax.hist(c, bins=bins, histtype="step", lw=2, color=l1c_c, label="L1C (OOD)")
        ax.set_title(f"{b} ({BAND_RES[b]} m)", color=ink, fontsize=11, loc="left")
        ax.set_yticks([])
        ax.tick_params(colors=mute, labelsize=8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(mute)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
               ncol=2, frameon=False, fontsize=10)
    fig.supxlabel("Reflectance x 10,000", color=ink)
    tiles = ",".join(args.tile) if args.tile else "all tiles"
    fig.suptitle(f"BenV2 {args.split}: L2A vs L1C per-band distributions "
                 f"({len(qa)} patches, {tiles})", color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(qdir, "band_distributions.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(S2_BANDS, summ.l1c_minus_l2a, color=l1c_c, width=0.6)
    ax.axhline(0, color=mute, lw=1)
    ax.set_ylabel("Mean L1C - L2A (DN)", color=ink)
    ax.set_title("Mean L1C - L2A per band (positive = brighter at TOA)", color=ink, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=mute)
    fig.tight_layout()
    fig.savefig(os.path.join(qdir, "band_mean_delta.png"), dpi=150)
    print(f"[qa] wrote {qdir}")


def main():
    args = parse_args()
    stages = ["index", "download", "chip", "qa"] if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"===== {s} =====")
        globals()[f"stage_{s}"](args)


if __name__ == "__main__":
    main()
