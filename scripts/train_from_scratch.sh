#!/usr/bin/env bash
set -Eeuo pipefail

dataset=$(cd -- "${1:?Usage: train_from_scratch.sh DATASET OUTPUT_DIR [TORCH_HUB_CACHE]}" && pwd -P)
mkdir -p -- "${2:?Provide a dedicated output directory}"
out=$(cd -- "$2" && pwd -P)
cache=()
if [[ -n ${3:-} ]]; then cache=(--weights-cache "$(cd -- "$3" && pwd -P)"); fi
case "$out/" in "$dataset/"*) echo 'Output must be outside the source dataset.' >&2; exit 1;; esac
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
py=${RP_PYTHON:-python}
device=${RP_DEVICE:-cuda:0}
epochs=${RP_EPOCHS:-200}
detail_epochs=${RP_DETAIL_EPOCHS:-20}
samples=${RP_PATCHES_PER_FIELD:-16}

exec 9>"$out/pipeline.lock"
flock -n 9 || { echo 'This run is already active.' >&2; exit 1; }
status() { printf '%s\t%s\n' "$(date -Iseconds)" "$*" | tee -a "$out/status.tsv"; }
trap 'code=$?; status "failed ($code) at line $LINENO"; exit "$code"' ERR
rp() { "$py" -u -m rpmin --device "$device" --threads 4 "$@"; }

snapshot() {
    "$py" - "$dataset" "$out" "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

from rpmin.engine import dump_json

dataset, out = map(Path, sys.argv[1:3])
mode = sys.argv[3]
path = out / 'source_before.json'
hashes = {}
for image in sorted(dataset.rglob('*')):
    if image.is_file():
        with image.open('rb') as f:
            hashes[image.relative_to(dataset).as_posix()] = hashlib.file_digest(f, 'sha256').hexdigest()
if path.exists():
    import json
    if json.loads(path.read_text()) != hashes:
        raise ValueError('Source dataset differs from the first snapshot.')
elif mode == 'before':
    dump_json(path, hashes)
else:
    raise FileNotFoundError(path)
if mode == 'after':
    dump_json(out / 'source_integrity.json', {'verified_files': len(hashes)})
PY
}

evaluate() {
    local manifest=$1 split=$2 dest=$3
    shift 3
    if [[ ! -f "$dest/summary.json" ]]; then
        rp evaluate --manifest "$manifest" --split "$split" --out "$dest" \
            --lr-patch 128 --stride-x 32 --stride-y 25 --batch 4 "$@"
    fi
}

train_variant() {
    local ref=$1 scale=$2 dest=$3
    shift 3
    if [[ ! -f "$dest/selection.json" ]]; then
        mkdir -p "$dest"
        rp train --manifest "$ref" --scale "$scale" --out "$dest" \
            --patch 128 --batch 4 --val-patch-stride 64 --validate-every 4 --cache-train \
            --phase energy-weighted "$@" | tee -a "$dest/train.log"
    fi
}

status 'verify code and source data'
"$py" -m unittest discover -s tests -q
snapshot before
if [[ ! -f "$out/data/dataset_report.json" ]]; then
    rp prepare-dataset --root "$dataset" --out "$out/data" --seed 42
fi

for scale in 2 4; do
    tag=x$scale
    manifest="$out/data/mfa-cell/$tag/manifest.csv"
    refs="$out/references/$tag/train-val"
    if [[ ! -f "$refs/aligned.csv" ]]; then
        status "$tag register training and validation pairs"
        rp align-references --manifest "$manifest" --out "$refs" --scale "$scale" --splits train val --refine "${cache[@]}"
    fi

    status "$tag train from random initialization"
    model="$out/models/$tag/base"
    detail="$out/models/$tag/detail"
    start=()
    if [[ -f "$model/latest.pt" ]]; then start=(--resume); fi
    train_variant "$refs/aligned.csv" "$scale" "$model" --epochs "$epochs" --warmup 5 \
        --lr 0.0001 --min-lr 0.000005 --patches-per-field "$samples" --mse-weight 0.5 --ssim-weight 0.1 "${start[@]}"

    status "$tag fine-tune phase-specific detail branch from new base weights"
    start=(--init-checkpoint "$model/best.pt")
    if [[ -f "$detail/latest.pt" ]]; then start=(--resume); fi
    train_variant "$refs/aligned.csv" "$scale" "$detail" --epochs "$detail_epochs" --warmup 0 \
        --lr 0.000075 --min-lr 0.000005 --patches-per-field 12 --detail-head \
        --pixel-weight 0.45 --mse-weight 1 --ssim-weight 0.15 "${start[@]}"

    status "$tag complete grid validation and freeze configuration"
    val="$out/validation/$tag"
    evaluate "$refs/aligned.csv" val "$val/bicubic" --bicubic --scale "$scale"
    for variant in base detail; do
        checkpoint="$out/models/$tag/$variant/best.pt"
        evaluate "$refs/aligned.csv" val "$val/$variant" --checkpoint "$checkpoint"
        evaluate "$refs/aligned.csv" val "$val/$variant-d4" --checkpoint "$checkpoint" --self-ensemble
    done
    "$py" - "$out" "$tag" <<'PY'
import json
import sys
from pathlib import Path

from rpmin.engine import dump_json

out, tag = Path(sys.argv[1]), sys.argv[2]
val = out / 'validation' / tag
candidates = {}
for variant in ('base', 'detail'):
    for suffix in ('', '-d4'):
        name = variant + suffix
        report = json.loads((val / name / 'summary.json').read_text())
        candidates[name] = {key: report[key]['mean'] for key in ('psnr', 'ssim')}
        candidates[name].update(checkpoint=str(out / 'models' / tag / variant / 'best.pt'), self_ensemble=bool(suffix))
selected = max((name for name, scores in candidates.items() if scores['ssim'] >= candidates['base']['ssim']),
               key=lambda name: (candidates[name]['psnr'], candidates[name]['ssim']))
dump_json(val / 'frozen_choice.json', {'selected': selected, 'candidates': candidates})
PY
done

for scale in 2 4; do
    tag=x$scale
    status "$tag fixed holdout test"
    manifest="$out/data/mfa-cell/$tag/manifest.csv"
    refs="$out/references/$tag/test"
    if [[ ! -f "$refs/aligned.csv" ]]; then
        rp align-references --manifest "$manifest" --out "$refs" --scale "$scale" --splits test --refine "${cache[@]}"
    fi
    readarray -t choice < <("$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d["candidates"][d["selected"]]; print(c["checkpoint"]); print(int(c["self_ensemble"]))' "$out/validation/$tag/frozen_choice.json")
    opts=(--checkpoint "${choice[0]}" --seeds 101 102 103)
    if [[ ${choice[1]} == 1 ]]; then opts+=(--self-ensemble); fi
    for protocol in aligned raw original; do
        ref="$refs/$protocol.csv"
        if [[ $protocol == original ]]; then ref="$manifest"; fi
        evaluate "$ref" test "$out/test/$tag/$protocol/bicubic" --bicubic --scale "$scale"
        evaluate "$ref" test "$out/test/$tag/$protocol/selected" "${opts[@]}"
    done
done

snapshot after
"$py" - "$out" <<'PY'
import json
import sys
from pathlib import Path

from rpmin.engine import dump_csv

out = Path(sys.argv[1])
rows = []
for path in sorted((out / 'test').glob('*/*/*/summary.json')):
    scale, reference, method = path.relative_to(out / 'test').parts[:3]
    report = json.loads(path.read_text())
    rows.append({'scale': scale, 'reference': reference, 'method': method,
                 **{key: report[key]['mean'] for key in ('psnr', 'ssim')},
                 'fields': report['patch_input']['fields'], 'patches_per_seed': report['runs'][0]['pairs'],
                 'seeds': len(report['runs']), 'self_ensemble': report.get('self_ensemble', '')})
if len(rows) != 12 or any(row['fields'] != 4 for row in rows):
    raise ValueError('Expected complete results from four fixed holdout fields at each scale.')
dump_csv(out / 'comparison.csv', rows)
PY
status complete
touch "$out/COMPLETE"
