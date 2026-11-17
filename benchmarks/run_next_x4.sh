#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
scale=${SCALE:-4}
references=${REFERENCES:-/root/autodl-tmp/rpmin-project/RP-MIN-fresh/runs/full/references/x$scale}
run=${RUN:-runs/x4-pilot}
grid=(--stride-x "${STRIDE_X:-32}" --stride-y "${STRIDE_Y:-25}")
export OMP_NUM_THREADS=2
next_python() {
    next-env/bin/python "$@"
}
torch_python() {
    torch-env/bin/python "$@"
}

case "${1:-}" in
    smoke)
        next_python next_x4.py train --scale "$scale" --manifest "$references/train-val/aligned.csv" \
            --out "$run/next-smoke" --pretrain 1 --gan 2 --batch 2
        ;;
    train)
        init=()
        if [[ -n ${INIT_RRDB:-} ]]; then
            init=(--init-rrdb "$INIT_RRDB")
        fi
        next_python next_x4.py train --scale "$scale" --manifest "$references/train-val/aligned.csv" --out "$run/next-train" \
            --pretrain "${PRETRAIN:-800}" --gan "${GAN:-200}" "${init[@]}"
        ;;
    evaluate)
        final_step=$(torch_python -c 'import json, sys; print(json.load(open(sys.argv[1]))["gan"])' "$run/next-train/config.json")
        for step in "$((final_step / 2))" "$final_step"; do
            next_python next_x4.py infer --scale "$scale" --weights "$run/next-train/gan-$step.weights.h5" \
                --inputs "$run/inputs/val_lr.npy" --out "$run/next-val-$step/predictions.npy"
            torch_python compare_x4.py NeXtSRGAN --scale "$scale" --manifest "$references/train-val/aligned.csv" \
                --split val "${grid[@]}" --predictions "$run/next-val-$step/predictions.npy" --out "$run/next-val-$step"
        done
        selected=$(torch_python - "$run" "$final_step" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

run = Path(sys.argv[1])
final_step = int(sys.argv[2])
scores = {step: json.loads((run / f'next-val-{step}' / 'summary.json').read_text())['psnr']
          for step in (final_step // 2, final_step)}
step = max(scores, key=scores.get)
target = run / 'next' / 'selection.json'
target.parent.mkdir(parents=True, exist_ok=True)
checkpoint = run / 'next-train' / f'gan-{step}.weights.h5'
with checkpoint.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
target.write_text(json.dumps({'selection_metric': 'validation RGB PSNR', 'selected_gan_step': step,
                              'val_psnr_by_gan_step': scores, 'checkpoint': str(checkpoint),
                              'checkpoint_sha256': digest}, indent=2) + '\n')
print(step)
PY
)
        next_python next_x4.py infer --scale "$scale" --weights "$run/next-train/gan-$selected.weights.h5" \
            --inputs "$run/inputs/test_lr.npy" --out "$run/next/predictions.npy"
        torch_python compare_x4.py NeXtSRGAN --scale "$scale" --manifest "$references/test/aligned.csv" \
            --split test "${grid[@]}" --predictions "$run/next/predictions.npy" --out "$run/next"
        ;;
    *)
        echo 'Use smoke, train or evaluate' >&2
        exit 2
        ;;
esac
