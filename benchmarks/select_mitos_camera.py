import argparse
import json
from pathlib import Path

import torch

from cross_scale_mitos_rpmin import BASE, GRID, SEED, paired_predictions, transfer_scale
from rpmin.engine import dump_csv, dump_json, load_model, manifest_digest, predict, summarize
from rpmin.objective import image_metrics
from train_mitos_camera import luma


SCALES = (2, 4)
PRIOR = Path('runs/mitos14-rpmin-cross-scale')
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)


@torch.inference_mode()
def evaluate(root, split, blends, metric):
    models, choices = {}, {}
    device = torch.device('cuda')
    for scale in SCALES:
        folder = root / f'x{scale}/camera'
        fit = json.loads((folder / 'selection.json').read_text())
        checkpoint = folder / 'best.pt'
        train = BASE / f'x{scale}/references/train-val/aligned.csv'
        if fit['train_sha256'] != manifest_digest(train) or fit['checkpoint_sha256'] != manifest_digest(checkpoint):
            raise ValueError(f'Unexpected camera-adapted weights for x{scale}.')
        models[scale] = load_model(checkpoint, device)
        choices[scale] = json.loads((PRIOR / f'x{scale}/selection.json').read_text())['selected']

    rng = torch.Generator()
    outputs = {scale: [[] for _ in blends[scale]] for scale in SCALES}
    for lr, ids, targets, originals, state in paired_predictions(split):
        for scale in SCALES:
            choice = choices[scale]
            previous = transfer_scale(originals, scale, choice['strength'], choice['centered'])
            rng.set_state(state)
            candidate = predict(models[scale], lr, rng) if any(blends[scale]) else previous
            for index, weight in enumerate(blends[scale]):
                sr = torch.lerp(previous, candidate, weight)
                if metric == 'y':
                    p, s = image_metrics(luma(sr.clamp(0, 1)), luma(targets[scale]))
                else:
                    p, s = image_metrics(sr, targets[scale])
                outputs[scale][index].extend(dict(id=key, seed=SEED, psnr=pv, ssim=sv)
                                              for key, pv, sv in zip(ids, p.tolist(), s.tolist()))
    return outputs


def source_digests(root, scale):
    return {
        'prior': manifest_digest(PRIOR / f'x{scale}/selection.json'),
        'training': manifest_digest(root / f'x{scale}/camera/selection.json'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--metric', choices=('rgb', 'y'), default='rgb')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'select':
        rows = evaluate(args.root, 'val', {scale: BLENDS for scale in SCALES}, args.metric)
        for scale, tracks in rows.items():
            candidates = [dict(blend=blend, **summarize(track)) for blend, track in zip(BLENDS, tracks, strict=True)]
            baseline = candidates[0]
            chosen = max((row for row in candidates if row['ssim'] >= baseline['ssim']), key=lambda row: row['psnr'])
            dump_json(args.root / f'x{scale}/locked/selection.json', dict(baseline=baseline, selected=chosen,
                                                                          candidates=candidates, metric=args.metric,
                                                                          sources=source_digests(args.root, scale)))
            print(scale, chosen, flush=True)
        return

    chosen = {}
    for scale in SCALES:
        record = json.loads((args.root / f'x{scale}/locked/selection.json').read_text())
        if record['sources'] != source_digests(args.root, scale) or record['metric'] != args.metric:
            raise ValueError(f'A locked source changed for x{scale}.')
        chosen[scale] = record['selected']
    rows = evaluate(args.root, 'test', {scale: (0.0, config['blend']) for scale, config in chosen.items()}, args.metric)
    for scale, tracks in rows.items():
        out = args.root / f'x{scale}/locked/test'
        test = BASE / f'x{scale}/references/test/aligned.csv'
        metrics = summarize(tracks[1])
        dump_csv(out / 'per_image.csv', tracks[1])
        dump_json(out / 'summary.json', dict(scale=scale, split='test', border_hr=0,
                                             manifest_sha256=manifest_digest(test),
                                             patch_input=dict(GRID, batch=4, fields=1),
                                             selected=chosen[scale], metric=args.metric,
                                             prior_test=summarize(tracks[0]), **metrics))
        print(scale, metrics, 'prior', summarize(tracks[0]), flush=True)


if __name__ == '__main__':
    main()
