"""
Processing to get NLCD dataset
"""

import numpy as np, rasterio, geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import box

PATCH   = 128          # 128 px @ 30 m = 3.84 km  (comparable to your other tasks)
N_TRY   = 40000
rng     = np.random.default_rng(42)

src = rasterio.open('/Users/kdoerkse/earthshift_data/nlcd/Annual_NLCD_LndCov_2021_CU_C1V1.tif')
H, W, T, nd = src.height, src.width, src.transform, src.nodata

rows = rng.integers(0, H - PATCH, N_TRY)
cols = rng.integers(0, W - PATCH, N_TRY)

recs = []
for r, c in zip(rows, cols):
    a = src.read(1, window=Window(int(c), int(r), PATCH, PATCH))
    if nd is not None and (a == nd).any():        # any CONUS edge/ocean -> drop
        continue
    if (a == 11).mean() > 0.9 or np.unique(a).size < 2:   # all-water / single-class
        continue
    x0, y0 = T * (c, r)
    x1, y1 = T * (c + PATCH, r + PATCH)
    recs.append({'row': int(r), 'col': int(c), 'geometry': box(min(x0,x1), min(y0,y1),
                                                               max(x0,x1), max(y0,y1))})

tiles = gpd.GeoDataFrame(recs, crs=src.crs)
print(len(tiles))

BLOCK = 100_000  # 100 km blocks
tiles['blk'] = (tiles.geometry.centroid.x // BLOCK).astype(int).astype(str) + '_' + \
               (tiles.geometry.centroid.y // BLOCK).astype(int).astype(str)

blks = rng.permutation(tiles.blk.unique())
n = len(blks); tr, va = blks[:int(.7*n)], blks[int(.7*n):int(.8*n)]
tiles['split'] = np.where(tiles.blk.isin(tr), 'train',
                  np.where(tiles.blk.isin(va), 'val', 'test'))

tiles.to_file('/Users/kdoerkse/earthshift_data/nlcd/nlcd_tiles_2021.gpkg', driver='GPKG')
print(tiles.split.value_counts())

import collections, numpy as np
hist = collections.Counter()
for _, t in tiles.sample(500, random_state=0).iterrows():
    a = src.read(1, window=Window(t.col, t.row, PATCH, PATCH))
    hist.update(np.unique(a).tolist())
print(sorted(hist.items(), key=lambda x: -x[1]))

'''
for year in (2013, 2021):
    s = rasterio.open(f'/Users/kdoerkse/earthshift_data/nlcd/Annual_NLCD_LndCov_{year}_CU_C1V1.tif')
    for i, t in tiles.iterrows():
        lab = s.read(1, window=Window(t.col, t.row, PATCH, PATCH))
        # save lab; pull matching Landsat composite for `year` over t.geometry
'''