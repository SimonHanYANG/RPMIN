#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2 PYTHONPATH=RPMIN-e79d2b3
root=runs/mitos14-rpmin-h07
scale=${1:?Use 2 or 4.}
case "$scale" in
    2|4) ;;
    *) exit 2 ;;
esac
original="runs/mitos14-h10-train/x$scale/references"
source="$root/mosaics/x$scale/input.csv"
target="$root/x$scale"
for ((attempt=0; attempt<120; attempt++)); do
    [[ ! -f "$source" ]] || break
    sleep 5
done
test -f "$source"

torch-env/bin/python -m rpmin --device cuda align-references --manifest "$source" --out "$target/h07-references" \
    --scale "$scale" --splits train --margin 12 --refine \
    --weights-cache /root/autodl-tmp/rpmin-project/RP-MIN/.torchhub
torch-env/bin/python - "$original" "$target" <<'PY'
import sys
from pathlib import Path

from rpmin.data import read_manifest, write_manifest

original, target = map(Path, sys.argv[1:])
h10 = read_manifest(original / 'train-val/aligned.csv')
h07 = read_manifest(target / 'h07-references/aligned.csv')
test = read_manifest(original / 'test/aligned.csv')
if {row.id for row in h07} != {'h07_00', 'h07_01'} or any(p.group != 'h07' or p.split != 'train' for p in h07):
    raise ValueError('Unexpected additional source.')
if {p.group for p in h10 + h07} & {p.group for p in test}:
    raise ValueError('Test field occurs in the training/validation manifest.')
write_manifest(target / 'augmented.csv', h10 + h07)
PY
torch-env/bin/python -m rpmin --device cuda audit --manifest "$target/augmented.csv"
torch-env/bin/python train_mitos_camera.py --luminance --native-grid --context 16 --scale "$scale" \
    --train "$target/augmented.csv" --real-steps 3600 --real-lr 0.00006 \
    --init "runs/mitos14-rpmin-consistency/x$scale/context-trained/best.pt" --out "$target/transferred"
