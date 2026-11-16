#!/usr/bin/env bash
set -Eeuo pipefail

base=$(cd -- "${1:?Usage: train_aligned.sh BASE_PROJECT PREVIOUS_COMPARISON OUTPUT_DIR}" && pwd -P)
prior=$(cd -- "${2:?Provide the previous comparison project}" && pwd -P)
mkdir -p -- "${3:?Provide an output directory}"
out=$(cd -- "$3" && pwd -P)
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
py=${RP_PYTHON:-python}
device=${RP_DEVICE:-cuda:0}
epochs=${RP_EPOCHS:-48}
samples=${RP_PATCHES_PER_FIELD:-16}
cache=()
if [[ -d "$base/.torchhub/checkpoints" ]]; then cache=(--weights-cache "$base/.torchhub"); fi
exec 9>"$out/pipeline.lock"
flock -n 9 || { echo 'The aligned pipeline is already running.' >&2; exit 1; }

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
    manifest="$base/runs/paper_retest/data/mfa-cell/$tag/manifest.csv"
    initial="$base/runs/server_cuda/models/angle/$tag/best.pt"
    if [[ "$scale" == 4 ]]; then
        initial="$prior/runs/controlled/models/x4/structural-raw/best.pt"
    fi
    refs="$out/references/$tag/train-val"
    coarse="$out/references/$tag/lightglue-val"
    status "$tag registration validation"
    if [[ ! -f "$coarse/aligned.csv" ]]; then
        rp align-references --manifest "$manifest" --scale "$scale" --out "$coarse" --splits val "${cache[@]}"
    fi
    if [[ ! -f "$refs/aligned.csv" ]]; then
        rp align-references --manifest "$manifest" --scale "$scale" --out "$refs" --splits train val --refine "${cache[@]}"
    fi
    for protocol in raw lightglue refined; do
        case "$protocol" in
            raw) ref="$refs/raw.csv" ;;
            lightglue) ref="$coarse/aligned.csv" ;;
            refined) ref="$refs/aligned.csv" ;;
        esac
        evaluate "$ref" val "$out/validation/$tag/$protocol/bicubic" --bicubic --scale "$scale"
        evaluate "$ref" val "$out/validation/$tag/$protocol/initial" --checkpoint "$initial"
    done

    status "$tag train on LR-coordinate references"
    model="$out/models/$tag/aligned"
    if [[ ! -f "$model/selection.json" ]]; then
        mkdir -p "$model"
        start=(--init-checkpoint "$initial")
        if [[ -f "$model/latest.pt" ]]; then start=(--resume); fi
        rp train --manifest "$refs/aligned.csv" --scale "$scale" --out "$model" \
            --patch 128 --batch 4 --epochs "$epochs" --warmup 2 --lr 0.0001 --min-lr 0.000005 \
            --patches-per-field "$samples" --validate-every 2 --val-patch-stride 64 --cache-train \
            --phase energy-weighted --mse-weight 0.5 --ssim-weight 0.1 "${start[@]}" | tee -a "$model/train.log"
    fi
    evaluate "$refs/aligned.csv" val "$out/validation/$tag/refined/trained" --checkpoint "$model/best.pt"
    status "$tag validation comparison complete"
done

status 'training and validation complete'
touch "$out/TRAINED"
