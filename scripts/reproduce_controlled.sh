#!/usr/bin/env bash
set -Eeuo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
previous=$(cd -- "${1:?Usage: reproduce_controlled.sh PREVIOUS_PROJECT OUTPUT_DIR}" && pwd -P)
mkdir -p -- "${2:?Provide an output directory}"
run=$(cd -- "$2" && pwd -P)
py=${RP_PYTHON:-python}
device=${RP_DEVICE:-cuda:0}
epochs=${RP_EPOCHS:-16}
samples=${RP_PATCHES_PER_FIELD:-8}
batch=${RP_BATCH:-4}
width=${RP_WIDTH:-32}
cd "$project"

exec 9>"$run/pipeline.lock"
flock -n 9 || { echo "This comparison is already running" >&2; exit 1; }
status() {
    printf '%s\t%s\n' "$(date -Iseconds)" "$*" | tee -a "$run/status.tsv"
}
trap 'code=$?; status "failed ($code) at line $LINENO"; exit "$code"' ERR

status "verify environment and prepare matched training manifests"
"$py" -m unittest discover -s tests -q
"$py" - "$previous" "$run" <<'PY'
import sys
from pathlib import Path
from rpmin.data import read_manifest, write_manifest

base, run = map(Path, sys.argv[1:])
for scale in (2, 4):
    tag = f"x{scale}"
    raw = read_manifest(base / "runs/paper_retest/data/mfa-cell" / tag / "manifest.csv")
    registered = read_manifest(base / "runs/paper_retest/training_data" / tag / "manifest.csv")
    raw_train = [p for p in raw if p.split == "train"]
    reg_train = [p for p in registered if p.split == "train"]
    if {(p.id, p.group) for p in raw_train} != {(p.id, p.group) for p in reg_train}:
        raise ValueError("Raw and registered experiments must use the same training fields.")
    holdout = [p for p in registered if p.split != "train"]
    write_manifest(run / "training_inputs" / tag / "raw.csv", raw_train + holdout)
PY

patch_eval() {
    local manifest=$1 split=$2 out=$3
    shift 3
    if [[ ! -f "$out/summary.json" ]]; then
        "$py" -m rpmin --device "$device" --threads 4 evaluate --manifest "$manifest" \
            --split "$split" --out "$out" --lr-patch 128 --stride-x 32 --stride-y 25 \
            --origin 1 --batch "$batch" "$@"
    fi
}

for scale in 4 2; do
    tag=x$scale
    raw="$previous/runs/paper_retest/data/mfa-cell/$tag/manifest.csv"
    registered="$previous/runs/paper_retest/training_data/$tag/manifest.csv"
    initial="$previous/runs/server_cuda/models/angle/$tag/best.pt"
    status "$tag validation-only evaluation protocol check"
    patch_eval "$raw" val "$run/validation/$tag/paper" --checkpoint "$initial" --seeds 101
    patch_eval "$raw" val "$run/validation/$tag/bicubic" --bicubic --scale "$scale"
    patch_eval "$raw" val "$run/validation/$tag/bicubic_legacy_origin" --bicubic --scale "$scale" --hr-origin 1

    for variant in robust-registered structural-registered structural-raw; do
        out="$run/models/$tag/$variant"
        train_manifest=$registered
        loss=()
        if [[ "$variant" == structural-* ]]; then loss=(--ssim-weight 0.05 --mse-weight 0.1); fi
        if [[ "$variant" == structural-raw ]]; then train_manifest="$run/training_inputs/$tag/raw.csv"; fi
        if [[ ! -f "$out/selection.json" ]]; then
            status "$tag fine-tuning $variant"
            start=(--init-checkpoint "$initial")
            if [[ -f "$out/latest.pt" ]]; then start=(--resume); fi
            mkdir -p "$out"
            "$py" -u -m rpmin --device "$device" --threads 4 train \
                --manifest "$train_manifest" --scale "$scale" --out "$out" --width "$width" \
                --patch 128 --batch "$batch" --epochs "$epochs" --warmup 0 --lr 0.0001 --min-lr 0.00001 \
                --patches-per-field "$samples" --validate-every 4 --cache-train \
                --phase energy-weighted "${loss[@]}" "${start[@]}" | tee -a "$out/train.log"
        fi
        patch_eval "$raw" val "$run/validation/$tag/$variant" --checkpoint "$out/best.pt" --seeds 101
    done

    status "$tag freeze the choice using validation before opening holdout scores"
    "$py" - "$run" "$tag" "$initial" <<'PY'
import json
import sys
from pathlib import Path
from rpmin.engine import dump_json

root, tag, initial = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
val = root / "validation" / tag
candidates = {}
for name in ("paper", "robust-registered", "structural-registered", "structural-raw"):
    report = json.loads((val / name / "summary.json").read_text())
    checkpoint = initial if name == "paper" else root / "models" / tag / name / "best.pt"
    candidates[name] = {"psnr": report["psnr"]["mean"], "ssim": report["ssim"]["mean"], "checkpoint": str(checkpoint)}
baseline = candidates["paper"]
eligible = {name for name, scores in candidates.items()
            if scores["psnr"] >= baseline["psnr"] and scores["ssim"] >= baseline["ssim"]}
winner = max(eligible, key=lambda name: (candidates[name]["psnr"], candidates[name]["ssim"], name == "paper"))
dump_json(val / "frozen_choice.json", {"selected": winner, "candidates": candidates})
print(winner, candidates[winner])
PY
    chosen=$("$py" -c 'import json,sys; a=json.load(open(sys.argv[1])); print(a["candidates"][a["selected"]]["checkpoint"])' \
        "$run/validation/$tag/frozen_choice.json")

    status "$tag fixed holdout evaluation (128-patch inference)"
    patch_eval "$raw" test "$run/test/$tag/bicubic" --bicubic --scale "$scale"
    patch_eval "$raw" test "$run/test/$tag/paper" --checkpoint "$initial" --seeds 101 102 103 104 105
    patch_eval "$raw" test "$run/test/$tag/selected" --checkpoint "$chosen" --seeds 101 102 103 104 105

    mitos="$previous/runs/server_cuda/mitos_external/$tag/center_crops_test_only.csv"
    if [[ -f "$mitos" && ! -f "$run/mitos_external/$tag/summary.json" ]]; then
        "$py" -m rpmin --device "$device" --threads 4 evaluate --manifest "$mitos" \
            --checkpoint "$chosen" --seeds 101 --out "$run/mitos_external/$tag"
    fi
    status "$tag comparison complete"
done
status "complete"
touch "$run/COMPLETE"
