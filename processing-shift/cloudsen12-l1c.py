#!/usr/bin/env python
"""
Build a processing-level shift for CloudSEN12: ID = the GEO-Bench-2 L2A samples
(geobench_cloudsen12.tortilla), OOD = the same samples at L1C (top-of-atmosphere).

GEO-Bench-2 only contains L2A. Its samples are taken unchanged from CloudSEN12+
(tacofoundation/cloudsen12, byte-identical L2A), which also contains an L1C version
of every sample with the same tortilla:id, footprint (509x509 padded to 512),
geotransform and target. The L1C sample is therefore read straight from the
CloudSEN12+ L1C taco files (only the byte ranges of the requested samples are
downloaded) and no re-chipping or resampling is needed.

Things this script handles
--------------------------
  * Band order. CloudSEN12+ L1C has 13 bands (B01-B12 incl. B10). B10 is dropped
    so the chips follow GEO-Bench's 12 band L2A order
    (B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12). The L2A AOT/WVP layers
    (bands 13-14 of the GEO-Bench file) have no L1C counterpart.
  * Radiometric offset. All samples are processing baseline N0206-N0209
    (< 04.00), so neither product carries the -1000 offset. A sample with
    baseline >= 04.00 is flagged in chips.csv instead of silently written.
  * Alignment. Each L1C sample is checked against its GEO-Bench L2A sample:
    same CRS and geotransform, identical target, same zero-padded pixels.

Output layout (root = --out)
----------------------------
  S2/<tortilla:id>.tif    12 band uint16 L1C chip, GEO-Bench band order
  chips.csv               per-sample status and alignment checks

Usage
-----
  python processing-shift/cloudsen12-l1c.py --cloudsen12 <dir with geobench_cloudsen12.tortilla> --out <dir>
  python processing-shift/cloudsen12-l1c.py --cloudsen12 <dir> --out <dir> --limit 10    # pilot
"""

import argparse
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import tacoreader

HF_URL = "https://huggingface.co/datasets/tacofoundation/cloudsen12/resolve/main/"
L1C_PARTS = [f"cloudsen12-l1c.000{i}.part.taco" for i in range(5)]
L1C_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
BANDS = [b for b in L1C_BANDS if b != "B10"]
FIELDS = ["tortilla_id", "s2_id", "status", "reason", "transform_match", "target_match", "pad_match"]


def read_child(sample, name):
    """Read one child raster (e.g. s2l1c, target) of a tortilla sample."""
    kids = pd.DataFrame(sample)
    k = int(np.flatnonzero(kids["tortilla:id"].values == name)[0])
    with rasterio.open(sample.read(k)) as src:
        return src.read(), src.transform, src.crs, src.profile


def chip_sample(gb, l1c, gb_pos, l1c_pos, tid, s2_id, out):
    """Write the L1C chip for one sample and return its chips.csv row."""
    row = dict(tortilla_id=tid, s2_id=s2_id, status="ok", reason="",
               transform_match="", target_match="", pad_match="")
    path = f"{out}/S2/{tid}.tif"

    baseline = int(re.search(r"_N(\d{4})_", s2_id).group(1))
    if baseline >= 400:
        row.update(status="skip", reason=f"baseline N{baseline:04d} carries -1000 offset")
        return row

    l2a, l2a_transform, l2a_crs, _ = read_child(gb.read(gb_pos), "s2l2a")
    l2a_target = read_child(gb.read(gb_pos), "target")[0]
    sample = l1c.read(l1c_pos)
    img, transform, crs, profile = read_child(sample, "s2l1c")
    target = read_child(sample, "target")[0]

    if img.shape[0] != len(L1C_BANDS):
        row.update(status="fail", reason=f"expected {len(L1C_BANDS)} L1C bands, got {img.shape[0]}")
        return row
    img = img[[L1C_BANDS.index(b) for b in BANDS]]

    row["transform_match"] = transform == l2a_transform and crs == l2a_crs
    row["target_match"] = bool(np.array_equal(target, l2a_target))
    row["pad_match"] = bool(np.array_equal(img.sum(0) == 0, l2a[:len(BANDS)].sum(0) == 0))
    if not (row["transform_match"] and row["target_match"]):
        row.update(status="fail", reason="L1C sample does not align with GEO-Bench L2A sample")
        return row

    profile.update(count=len(BANDS), dtype="uint16", compress="deflate", driver="GTiff")
    tmp = f"{path}.{os.getpid()}.tmp"
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(img.astype(np.uint16))
        dst.descriptions = tuple(BANDS)
    os.replace(tmp, path)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloudsen12", required=True, help="Dir containing geobench_cloudsen12.tortilla")
    p.add_argument("--out", required=True, help="Output dir for the L1C chips")
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Only process the first N samples (pilot)")
    args = p.parse_args()
    os.makedirs(f"{args.out}/S2", exist_ok=True)

    gb = tacoreader.load(f"{args.cloudsen12}/geobench_cloudsen12.tortilla")
    l1c = tacoreader.load([HF_URL + part for part in L1C_PARTS])
    gb_df, l1c_df = pd.DataFrame(gb), pd.DataFrame(l1c)
    l1c_pos = {t: i for i, t in enumerate(l1c_df["tortilla:id"])}

    todo = [(i, t, s) for i, (t, s, split) in enumerate(zip(gb_df["tortilla:id"], gb_df["s2_id"],
                                                            gb_df["tortilla:data_split"])) if split == args.split]
    if args.limit:
        todo = todo[:args.limit]

    # Resume: keep rows of chips already written
    csv_path = f"{args.out}/chips.csv"
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            done = {r["tortilla_id"]: r for r in csv.DictReader(f)
                    if r["status"] == "ok" and os.path.exists(f"{args.out}/S2/{r['tortilla_id']}.tif")}
    missing = [(i, t, s) for i, t, s in todo if t not in l1c_pos]
    todo = [(i, t, s) for i, t, s in todo if t in l1c_pos and t not in done]
    print(f"{args.split}: {len(done)} chips already done, {len(todo)} to fetch, {len(missing)} without an L1C sample")

    rows = list(done.values()) + [dict(tortilla_id=t, s2_id=s, status="fail", reason="no CloudSEN12+ L1C sample")
                                  for _, t, s in missing]
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(chip_sample, gb, l1c, i, l1c_pos[t], t, s, args.out): (t, s) for i, t, s in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            t, s = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append(dict(tortilla_id=t, s2_id=s, status="fail", reason=f"{type(e).__name__}: {e}"))
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")

    rows.sort(key=lambda r: r["tortilla_id"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    status = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    print(f"Wrote {csv_path}: {status}")


if __name__ == "__main__":
    main()
