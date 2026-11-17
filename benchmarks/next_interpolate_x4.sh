#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
run=${RUN:-runs/x4-next-transfer}
scale=${SCALE:-4}
refs=${REFERENCES:-/root/autodl-tmp/rpmin-project/RP-MIN-fresh/runs/full/references/x$scale}
grid=(--stride-x "${STRIDE_X:-32}" --stride-y "${STRIDE_Y:-25}")
export OMP_NUM_THREADS=2
gan_step=$(torch-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_gan_step"])' "$run/next/selection.json")
pretrain_step=$(torch-env/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["pretrain"])' "$run/next-train/config.json")
psnr="$run/next-train/pretrain-$pretrain_step.weights.h5"
gan="$run/next-train/gan-$gan_step.weights.h5"
for alpha in 0.0 0.2 0.4 0.6 0.8; do
    target="$run/interp-val-$alpha"
    next-env/bin/python next_x4.py interpolate --scale "$scale" --psnr "$psnr" --gan "$gan" --alpha "$alpha" --out "$target/model.weights.h5"
    next-env/bin/python next_x4.py infer --scale "$scale" --weights "$target/model.weights.h5" --inputs "$run/inputs/val_lr.npy" --out "$target/predictions.npy"
    torch-env/bin/python compare_x4.py NeXtSRGAN --scale "$scale" --manifest "$refs/train-val/aligned.csv" --split val "${grid[@]}" --predictions "$target/predictions.npy" --out "$target"
done
alpha=$(torch-env/bin/python - "$run" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
scores = {alpha: json.loads((run / f'interp-val-{alpha}' / 'summary.json').read_text())['psnr']
          for alpha in ('0.0', '0.2', '0.4', '0.6', '0.8')}
gan = json.loads((run / 'next' / 'selection.json').read_text())
scores['1.0'] = gan['val_psnr_by_gan_step'][str(gan['selected_gan_step'])]
selected = max((alpha for alpha in scores if float(alpha) > 0), key=scores.get)
checkpoint = (Path(gan['checkpoint']) if selected == '1.0'
              else run / f'interp-val-{selected}' / 'model.weights.h5')
target = run / 'interpolated'
target.mkdir(parents=True, exist_ok=True)
with checkpoint.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
(target / 'selection.json').write_text(json.dumps({'val_psnr_by_alpha': scores, 'selected_alpha': selected,
                                                   'checkpoint': str(checkpoint), 'checkpoint_sha256': digest},
                                                  indent=2) + '\n')
print(selected)
PY
)
if [[ "$alpha" == 1.0 ]]; then
    checkpoint="$gan"
else
    checkpoint="$run/interp-val-$alpha/model.weights.h5"
fi
next-env/bin/python next_x4.py infer --scale "$scale" --weights "$checkpoint" --inputs "$run/inputs/test_lr.npy" --out "$run/interpolated/predictions.npy"
torch-env/bin/python compare_x4.py NeXtSRGAN --scale "$scale" --manifest "$refs/test/aligned.csv" --split test "${grid[@]}" --predictions "$run/interpolated/predictions.npy" --out "$run/interpolated"
