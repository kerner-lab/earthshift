#!/bin/bash
# Download + chip the BenV2 test L1C set in tile batches to bound disk use.
PY=/opt/anaconda3/envs/earthshift/bin/python
OUT=/Users/kdoerkse/earthshift_data/benv2_l1c
TILES=$(tail -n +2 $OUT/products.csv | cut -d, -f2 | sort -u | grep -v '^29SNC$')
set -- $TILES
n=0
while [ $# -gt 0 ]; do
  batch="${@:1:5}"; shift $(( $# < 5 ? $# : 5 )); n=$((n+1))
  echo "=== batch $n: $batch ($(date +%H:%M:%S)) ==="
  $PY benv2-l1c.py --stage download --tile $batch | grep -v '^\[' || exit 1
  $PY benv2-l1c.py --stage chip --tile $batch || exit 1
  echo "raw on disk: $(du -sh $OUT/raw | cut -f1), free: $(df -h $OUT | tail -1 | awk '{print $4}')"
done
echo "=== done $(date +%H:%M:%S) ==="
