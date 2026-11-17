#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export SCALE=2
export RUN=${RUN:-runs/x2-pilot}
export OMP_NUM_THREADS=2
rpmin=/root/autodl-tmp/rpmin-project/RP-MIN-fresh
refs="$rpmin/runs/full/references/x2"

case "${1:-}" in
    torch)
        for model in AESOP SeD; do
            target="$RUN/$model-transfer"
            torch-env/bin/python compare_x4.py "$model" --scale 2 --manifest "$refs/train-val/aligned.csv" \
                --split val --out "$target" --fit-transfer
            torch-env/bin/python compare_x4.py "$model" --scale 2 --manifest "$refs/test/aligned.csv" \
                --split test --out "$target" --adapted "$target/adapted.pt"
        done
        PYTHONPATH=RPMIN-e79d2b3 torch-env/bin/python -m rpmin --device cuda evaluate \
            --manifest "$refs/test/aligned.csv" --split test --checkpoint "$rpmin/runs/full/models/x2/base/best.pt" \
            --lr-patch 128 --stride-x 32 --stride-y 25 --origin 1 --batch 4 --seeds 101 --out "$RUN/rpmin"
        ;;
    next)
        export PRETRAIN=204 GAN=100 INIT_RRDB=weights/rrdb-official-psnr.pth
        export TF_CPP_MIN_LOG_LEVEL=2 PYTHONWARNINGS=ignore::UserWarning
        torch-env/bin/python compare_x4.py prepare --scale 2 --manifest "$refs/train-val/aligned.csv" \
            --split val --out "$RUN/inputs"
        torch-env/bin/python compare_x4.py prepare --scale 2 --manifest "$refs/test/aligned.csv" \
            --split test --out "$RUN/inputs"
        bash run_next_x4.sh train
        bash run_next_x4.sh evaluate
        bash next_interpolate_x4.sh
        ;;
    *)
        echo 'Use torch or next.' >&2
        exit 2
        ;;
esac
