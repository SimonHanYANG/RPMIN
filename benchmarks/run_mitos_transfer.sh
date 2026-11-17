#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

scale=${1:?Use scale 2 or 4.}
case "$scale" in
    2|4) ;;
    *) exit 2 ;;
esac
export OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="$PWD/RPMIN-e79d2b3${PYTHONPATH:+:$PYTHONPATH}"
rpmin=/root/autodl-tmp/rpmin-project/RP-MIN-fresh
source="$rpmin/runs/full/data/mitos-14/x$scale/pairs.csv"
out=${RUN:-runs/mitos14-h10-transfer}/x$scale
mkdir -p "$out"

torch-env/bin/python - "$source" "$out/input.csv" <<'PY'
import sys
from dataclasses import replace
from pathlib import Path

from rpmin.data import read_manifest, write_manifest

source, target = map(Path, sys.argv[1:])
pairs = read_manifest(source, require_split=False)
if len(pairs) != 5 or {pair.group for pair in pairs} != {'h10'}:
    raise ValueError('Expected the five H10 fields from the project data.')
write_manifest(target, [replace(pair, split='test') for pair in pairs])
PY

if [[ ! -f "$out/references/aligned.csv" ]]; then
    torch-env/bin/python -m rpmin --device cuda align-references --manifest "$out/input.csv" \
        --scale "$scale" --out "$out/references" --splits test --margin 12 --refine \
        --weights-cache /root/autodl-tmp/rpmin-project/RP-MIN/.torchhub
fi

manifest="$out/references/aligned.csv"
grid=(--stride-x 128 --stride-y 128)
run="runs/x$scale-pilot"
if [[ "$scale" == 2 ]]; then
    checkpoint="$rpmin/runs/full/models/x2/base/best.pt"
    next_run=runs/x2-pilot
else
    checkpoint="$rpmin/runs/x4-polish/pixel-refine/best.pt"
    next_run=runs/x4-next-transfer
fi

torch-env/bin/python -m rpmin --device cuda evaluate --manifest "$manifest" --split test \
    --lr-patch 128 "${grid[@]}" --origin 1 --batch 4 --seeds 101 --checkpoint "$checkpoint" --out "$out/rpmin"
torch-env/bin/python -m rpmin --device cuda evaluate --manifest "$manifest" --split test \
    --lr-patch 128 "${grid[@]}" --origin 1 --batch 4 --seeds 101 --bicubic --scale "$scale" --out "$out/bicubic"
for model in AESOP SeD; do
    torch-env/bin/python compare_x4.py "$model" --scale "$scale" --manifest "$manifest" --split test \
        "${grid[@]}" --out "$out/$model" --adapted "$run/$model-transfer/adapted.pt"
done

next_checkpoint=$(torch-env/bin/python - "$next_run/interpolated/selection.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(selection['checkpoint'])
with checkpoint.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != selection['checkpoint_sha256']:
        raise ValueError('Unexpected NeXtSRGAN checkpoint fingerprint.')
print(checkpoint)
PY
)
torch-env/bin/python compare_x4.py prepare --scale "$scale" --manifest "$manifest" --split test \
    "${grid[@]}" --out "$out/inputs"
next-env/bin/python next_x4.py infer --scale "$scale" --weights "$next_checkpoint" \
    --inputs "$out/inputs/test_lr.npy" --out "$out/next/predictions.npy"
torch-env/bin/python compare_x4.py NeXtSRGAN --scale "$scale" --manifest "$manifest" --split test \
    "${grid[@]}" --predictions "$out/next/predictions.npy" --out "$out/next"
