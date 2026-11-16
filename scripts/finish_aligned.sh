#!/usr/bin/env bash
set -Eeuo pipefail

base=$(cd -- "${1:?Usage: finish_aligned.sh BASE_PROJECT PREVIOUS_COMPARISON OUTPUT_DIR}" && pwd -P)
prior=$(cd -- "${2:?Provide the previous comparison project}" && pwd -P)
out=$(cd -- "${3:?Provide the training output directory}" && pwd -P)
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ -f "$out/TRAINED" ]] || { echo 'Run train_aligned.sh to completion first.' >&2; exit 1; }
py=${RP_PYTHON:-python}
device=${RP_DEVICE:-cuda:0}
epochs=${RP_DETAIL_EPOCHS:-20}
samples=${RP_DETAIL_PATCHES_PER_FIELD:-12}
cache=()
if [[ -d "$base/.torchhub/checkpoints" ]]; then cache=(--weights-cache "$base/.torchhub"); fi
exec 9>"$out/pipeline.lock"
flock -w 15 9 || { echo 'The aligned pipeline is already running.' >&2; exit 1; }

status() { printf '%s\t%s\n' "$(date -Iseconds)" "$*" | tee -a "$out/status.tsv"; }
trap 'code=$?; status "failed ($code) at line $LINENO"; exit "$code"' ERR
rp() { "$py" -u -m rpmin --device "$device" --threads 4 "$@"; }
evaluate() {
    local manifest=$1 split=$2 dest=$3
    shift 3
    if [[ ! -f "$dest/summary.json" ]]; then
        rp evaluate --manifest "$manifest" --split "$split" --out "$dest" \
            --lr-patch 128 --stride-x 32 --stride-y 25 --batch 4 "$@"
    fi
}

for scale in 2 4; do
    tag=x$scale
    refs="$out/references/$tag/train-val/aligned.csv"
    trained="$out/models/$tag/aligned/best.pt"
    detail="$out/models/$tag/detail"
    val="$out/validation/$tag/refined"
    status "$tag validate phase-specific reconstruction and self-ensemble"
    if [[ ! -f "$detail/selection.json" ]]; then
        mkdir -p "$detail"
        start=(--init-checkpoint "$trained")
        if [[ -f "$detail/latest.pt" ]]; then start=(--resume); fi
        rp train --manifest "$refs" --scale "$scale" --out "$detail" --detail-head \
            --patch 128 --batch 4 --epochs "$epochs" --warmup 0 --lr 0.000075 --min-lr 0.000005 \
            --patches-per-field "$samples" --validate-every 2 --val-patch-stride 64 --cache-train \
            --phase energy-weighted --pixel-weight 0.45 --mse-weight 1.0 --ssim-weight 0.15 \
            "${start[@]}" | tee -a "$detail/train.log"
    fi
    evaluate "$refs" val "$val/detail" --checkpoint "$detail/best.pt"
    evaluate "$refs" val "$val/trained-d4" --checkpoint "$trained" --self-ensemble
    evaluate "$refs" val "$val/detail-d4" --checkpoint "$detail/best.pt" --self-ensemble

    "$py" - "$val" "$trained" "$detail/best.pt" <<'PY'
import json
import sys
from pathlib import Path

from rpmin.engine import dump_json

val, trained, detail = map(Path, sys.argv[1:])
choice = val / 'frozen_choice.json'
if not choice.exists():
    candidates = {}
    for name, checkpoint in (('trained', trained), ('detail', detail)):
        for suffix in ('', '-d4'):
            key = name + suffix
            report = json.loads((val / key / 'summary.json').read_text())
            candidates[key] = {metric: report[metric]['mean'] for metric in ('psnr', 'ssim')}
            candidates[key].update(checkpoint=str(checkpoint), self_ensemble=bool(suffix))
    baseline = candidates['trained']
    selected = max((key for key, scores in candidates.items() if scores['ssim'] >= baseline['ssim']),
                   key=lambda key: (candidates[key]['psnr'], candidates[key]['ssim']))
    dump_json(choice, {'selected': selected, 'candidates': candidates})
PY
done

for scale in 2 4; do
    tag=x$scale
    status "$tag fixed test comparison"
    manifest="$base/runs/paper_retest/data/mfa-cell/$tag/manifest.csv"
    initial="$base/runs/server_cuda/models/angle/$tag/best.pt"
    if [[ "$scale" == 4 ]]; then
        initial="$prior/runs/controlled/models/x4/structural-raw/best.pt"
    fi
    choice="$out/validation/$tag/refined/frozen_choice.json"
    readarray -t selection < <("$py" -c 'import json,sys; a=json.load(open(sys.argv[1])); best=a["candidates"][a["selected"]]; print(best["checkpoint"]); print(int(best["self_ensemble"]))' "$choice")
    options=(--checkpoint "${selection[0]}" --seeds 101 102 103)
    if [[ "${selection[1]}" == 1 ]]; then options+=(--self-ensemble); fi
    refs="$out/references/$tag/test"
    if [[ ! -f "$refs/aligned.csv" ]]; then
        rp align-references --manifest "$manifest" --scale "$scale" --out "$refs" --splits test --refine "${cache[@]}"
    fi
    for protocol in raw aligned; do
        ref="$refs/$protocol.csv"
        test="$out/test/$tag/$protocol"
        evaluate "$ref" test "$test/bicubic" --bicubic --scale "$scale"
        evaluate "$ref" test "$test/initial" --checkpoint "$initial" --seeds 101 102 103
        evaluate "$ref" test "$test/selected" "${options[@]}"
    done
    evaluate "$manifest" test "$out/test/$tag/original/selected" "${options[@]}"
done

"$py" - "$base" "$out" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from rpmin.engine import dump_csv, dump_json

base, out = map(Path, sys.argv[1:])
dataset = base.parent / 'SuperResolutionDataset'
files = json.loads((base / 'runs/paper_retest/source_integrity.json').read_text())['before_and_after']
if not all(hashlib.sha256((dataset / key).read_bytes()).hexdigest() == value for key, value in files.items()):
    raise ValueError('Source image bytes differ from the original audit.')
dump_json(out / 'source_integrity.json', {'verified_files': len(files)})

rows = []
for summary in sorted((out / 'test').glob('*/*/*/summary.json')):
    scale, protocol, method = summary.relative_to(out / 'test').parts[:3]
    report = json.loads(summary.read_text())
    rows.append({'scale': scale, 'reference': protocol, 'method': method,
                 **{metric: report[metric]['mean'] for metric in ('psnr', 'ssim')},
                 'fields': report['patch_input']['fields'], 'patches_per_seed': report['runs'][0]['pairs'],
                 'seeds': len(report['runs']), 'self_ensemble': report.get('self_ensemble', '')})
dump_csv(out / 'comparison.csv', rows)
PY
status 'complete'
touch "$out/COMPLETE"
