#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export SCALE=${1:?Use scale 2 or 4.}
case "$SCALE" in
    2|4) ;;
    *) exit 2 ;;
esac
export OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="$PWD/RPMIN-e79d2b3${PYTHONPATH:+:$PYTHONPATH}"
export STRIDE_X=128 STRIDE_Y=128
rpmin=/root/autodl-tmp/rpmin-project/RP-MIN-fresh
cache=/root/autodl-tmp/rpmin-project/RP-MIN/.torchhub
source="$rpmin/runs/full/data/mitos-14/x$SCALE/pairs.csv"
out=${RUN_ROOT:-runs/mitos14-h10-train}/x$SCALE
export REFERENCES="$out/references"
mkdir -p "$out"

torch-env/bin/python - "$source" "$out/input.csv" "$out/split.csv" <<'PY'
import csv
import random
import sys
from dataclasses import replace
from pathlib import Path

from rpmin.data import read_manifest, write_manifest
from rpmin.engine import manifest_digest

source, target, report = map(Path, sys.argv[1:])
pairs = read_manifest(source, require_split=False)
if len(pairs) != 5 or {pair.group for pair in pairs} != {'h10'}:
    raise ValueError('Expected the five H10 fields in the project data.')
for kind in ('lr', 'hr'):
    if len({manifest_digest(getattr(pair, kind)) for pair in pairs}) != 5:
        raise ValueError(f'Duplicate {kind} image across fields.')
random.Random(42).shuffle(pairs)
assigned = [replace(pair, split=split, group=pair.id)
            for pair, split in zip(pairs, ('train', 'train', 'train', 'val', 'test'), strict=True)]
write_manifest(target, assigned)
with report.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['field', 'split', 'source_group'])
    writer.writerows((pair.lr.stem, pair.split, 'H10') for pair in assigned)
PY

rp() { torch-env/bin/python -m rpmin --device cuda "$@"; }
rp audit --manifest "$out/input.csv"
for partition in train-val test; do
    splits=(test)
    if [[ "$partition" == train-val ]]; then
        splits=(train val)
    fi
    if [[ ! -f "$REFERENCES/$partition/aligned.csv" ]]; then
        rp align-references --manifest "$out/input.csv" --scale "$SCALE" --out "$REFERENCES/$partition" \
            --splits "${splits[@]}" --margin 12 --refine --weights-cache "$cache"
    fi
done

train="$REFERENCES/train-val/aligned.csv"
test="$REFERENCES/test/aligned.csv"
grid=(--stride-x "$STRIDE_X" --stride-y "$STRIDE_Y")
rp train --manifest "$train" --out "$out/rpmin-train" --scale "$SCALE" \
    --epochs 68 --patch 128 --batch 4 --warmup 5 --val-patch-stride 128 --validate-every 8 \
    --phase energy-weighted --mse-weight 0.5 --ssim-weight 0.1 --patches-per-field 16 --cache-train
rp evaluate --manifest "$test" --split test --checkpoint "$out/rpmin-train/best.pt" \
    --lr-patch 128 "${grid[@]}" --origin 1 --batch 4 --seeds 101 --out "$out/rpmin"
rp evaluate --manifest "$test" --split test --bicubic --scale "$SCALE" \
    --lr-patch 128 "${grid[@]}" --origin 1 --batch 4 --seeds 101 --out "$out/bicubic"

for model in AESOP SeD; do
    torch-env/bin/python compare_x4.py "$model" --scale "$SCALE" --manifest "$train" \
        --split val --out "$out/$model" --fit-transfer --fit-epochs 17 --patches-per-field 32
    torch-env/bin/python compare_x4.py "$model" --scale "$SCALE" --manifest "$test" --split test \
        "${grid[@]}" --out "$out/$model" --adapted "$out/$model/adapted.pt"
done

export RUN="$out/next" PRETRAIN=204 GAN=100 INIT_RRDB=weights/rrdb-official-psnr.pth
torch-env/bin/python compare_x4.py prepare --scale "$SCALE" --manifest "$train" --split val \
    "${grid[@]}" --out "$RUN/inputs"
torch-env/bin/python compare_x4.py prepare --scale "$SCALE" --manifest "$test" --split test \
    "${grid[@]}" --out "$RUN/inputs"
bash run_next_x4.sh train
bash run_next_x4.sh evaluate
bash next_interpolate_x4.sh
