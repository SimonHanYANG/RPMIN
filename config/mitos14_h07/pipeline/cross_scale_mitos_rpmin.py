import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from refine_mitos_rpmin import GRID, SEED, correct, data, load_sources, predict_sources
from rpmin.data import read_manifest
from rpmin.engine import dump_csv, dump_json, manifest_digest, summarize
from rpmin.objective import image_metrics


BASE = Path('runs/mitos14-h10-train')
PRIOR = Path('runs/mitos14-rpmin-opt')
CANDIDATES = [(0.0, False)] + [(strength, centered) for centered in (False, True)
                                for strength in (0.25, 0.5, 0.75, 1.0)]


def paired_predictions(split):
    device = torch.device('cuda')
    sources = {}
    batches = {}
    for scale in (2, 4):
        refs = BASE / f'x{scale}/references'
        training = refs / 'train-val/aligned.csv'
        source = training if split == 'val' else refs / 'test/aligned.csv'
        if split == 'test':
            if {pair.group for pair in read_manifest(training)} & {pair.group for pair in read_manifest(source)}:
                raise ValueError('Test field was used for training or validation.')
        sources[scale] = load_sources(PRIOR / f'x{scale}/locked', training, device)
        batches[scale] = data(source, split, scale, device)

    generator = torch.Generator().manual_seed(SEED)
    for batch2, batch4 in zip(batches[2], batches[4], strict=True):
        lr, gt2, ids = batch2
        other_lr, gt4, other_ids = batch4
        if ids != other_ids or not torch.equal(lr, other_lr):
            raise ValueError('Expected identical LR inputs and patch IDs across scales.')
        lr, gt2, gt4 = lr.to(device), gt2.to(device), gt4.to(device)
        state = generator.get_state()
        originals = {}
        for scale in (2, 4):
            generator.set_state(state)
            models, correction = sources[scale]
            a, b = predict_sources(models, lr, generator)
            originals[scale] = correct((a + b) * 0.5, lr, ids, correction)
        yield lr, ids, {2: gt2, 4: gt4}, originals, state


def transfer_scale(predictions, scale, strength, centered):
    if not strength:
        return predictions[scale]
    difference = predictions[2] - F.avg_pool2d(predictions[4], 2)
    if centered:
        difference = difference - difference.mean((-1, -2), keepdim=True)
    if scale == 4:
        delta = F.interpolate(difference, scale_factor=2, mode='bicubic', align_corners=False)
        return predictions[scale] + strength * delta
    return predictions[scale] - strength * difference


@torch.inference_mode()
def evaluate(split, choices):
    rows = {scale: [[] for _ in choices[scale]] for scale in (2, 4)}
    for _, ids, targets, originals, _ in paired_predictions(split):
        for scale, target in targets.items():
            for i, (strength, centered) in enumerate(choices[scale]):
                psnr, ssim = image_metrics(transfer_scale(originals, scale, strength, centered), target)
                rows[scale][i].extend(dict(id=key, seed=SEED, psnr=p, ssim=s)
                                      for key, p, s in zip(ids, psnr.tolist(), ssim.tolist()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'select':
        results = evaluate('val', {2: CANDIDATES, 4: CANDIDATES})
        for scale, tracks in results.items():
            candidates = [dict(strength=strength, centered=centered, **summarize(rows))
                          for (strength, centered), rows in zip(CANDIDATES, tracks, strict=True)]
            reference = candidates[0]
            selected = max((row for row in candidates if row['ssim'] >= reference['ssim']),
                           key=lambda row: row['psnr'])
            dump_json(args.out / f'x{scale}/selection.json', dict(baseline=reference, selected=selected,
                                                                 candidates=candidates))
            print(scale, selected, flush=True)
        return

    selected = {scale: json.loads((args.out / f'x{scale}/selection.json').read_text())['selected'] for scale in (2, 4)}
    rows = evaluate('test', {scale: [(choice['strength'], choice['centered'])] for scale, choice in selected.items()})
    for scale, tracks in rows.items():
        out = args.out / f'x{scale}/locked/test'
        ref = BASE / f'x{scale}/references/test/aligned.csv'
        dump_csv(out / 'per_image.csv', tracks[0])
        metrics = summarize(tracks[0])
        dump_json(out / 'summary.json', dict(scale=scale, split='test', border_hr=0,
                                             manifest_sha256=manifest_digest(ref),
                                             patch_input=dict(GRID, batch=4, fields=1),
                                             selected=selected[scale], **metrics))
        print(scale, metrics, flush=True)


if __name__ == '__main__':
    main()
