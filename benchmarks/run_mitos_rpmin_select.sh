#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2
scale=${1:?Use 2 or 4.}
case "$scale" in
    2|4) ;;
    *) exit 2 ;;
esac
original="runs/mitos14-h10-train/x$scale"
out="${RUN_ROOT:-runs/mitos14-rpmin-opt}/x$scale"
train="$original/references/train-val/aligned.csv"
test="$original/references/test/aligned.csv"
for ((attempt=0; attempt<180; attempt++)); do
    if [[ -f "$out/detail/selection.json" && -f "$out/width48/selection.json" ]]; then break; fi
    sleep 5
done
[[ -f "$out/detail/selection.json" && -f "$out/width48/selection.json" ]]
torch-env/bin/python optimize_mitos_rpmin.py select --scale "$scale" --train "$train" --out "$out/locked" \
    --checkpoint "base=$original/rpmin-train/best.pt" "detail=$out/detail/best.pt" "width48=$out/width48/best.pt"
torch-env/bin/python optimize_mitos_rpmin.py test --scale "$scale" --train "$train" --test "$test" --out "$out/locked"
