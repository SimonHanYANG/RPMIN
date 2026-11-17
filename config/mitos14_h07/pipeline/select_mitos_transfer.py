import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from select_mitos_consistency import BASE, GRID, SCALES, SEED, Candidate, center_crop, luma
from rpmin.data import read_manifest
from rpmin.engine import dump_csv, dump_json, load_model, manifest_digest, predict, summarize
from rpmin.objective import image_metrics


PRIOR = Path('runs/mitos14-rpmin-consistency')
OPTIONS = list(itertools.product((0.0, 0.25, 0.5), repeat=3)) + [(1.0, 1.0, 1.0)]


def blurred(image, radius):
    return F.avg_pool2d(F.pad(image, (radius,) * 4, mode='reflect'), 2 * radius + 1, stride=1)


def frequency_bands(difference, scale):
    low = blurred(difference, scale)
    coarse = blurred(low, 2 * scale)
    return coarse, low - coarse, difference - low


class Comparison:
    def __init__(self, root, h07=False, checkpoints=None):
        self.anchor = Candidate(PRIOR)
        self.choices, self.models, self.digests = {}, {}, {}
        for scale in SCALES:
            tag = f'x{scale}'
            previous = PRIOR / tag / 'selection.json'
            baseline = json.loads(previous.read_text())
            if baseline['sources'] != self.anchor.sources:
                raise ValueError('A source of the previously locked configuration changed.')
            self.choices[scale] = baseline['selected']
            run = root / tag / 'transferred'
            record = run / 'selection.json'
            checkpoint = checkpoints[scale] if checkpoints is not None else run / 'best.pt'
            fitted = json.loads(record.read_text())
            train = BASE / tag / 'references/train-val/aligned.csv'
            if h07:
                reference = set(read_manifest(train))
                train = root / tag / 'augmented.csv'
                augmented = set(read_manifest(train))
                added = augmented - reference
                if (not reference <= augmented or {row.id for row in added} != {'h07_00', 'h07_01'}
                        or any(row.group != 'h07' or row.split != 'train' for row in added)):
                    raise ValueError('Expected exactly two new H07 training fields.')
            digest = manifest_digest(checkpoint)
            if (fitted['train_sha256'] != manifest_digest(train) or fitted['checkpoint_sha256'] != digest
                    or fitted['context'] != 16 or fitted['metric'] != 'BT.601 Y'):
                raise ValueError(f'Unexpected transfer training artifacts for {tag}.')
            source = dict(prior=manifest_digest(previous), run=manifest_digest(record), weights=digest)
            if h07:
                source['train'] = manifest_digest(train)
            self.digests[str(scale)] = source
            self.models[scale] = load_model(checkpoint, self.anchor.device).eval()

    @torch.inference_mode()
    def score(self, split, weights):
        radius = 16
        generator = torch.Generator().manual_seed(SEED)
        shared = torch.Generator()
        size = GRID['patch'] + radius * 2
        scores = {scale: [[] for _ in weights[scale]] for scale in SCALES}
        for lr, hr, ids, fields in self.anchor.predictor.make_batch(split, radius):
            expanded = torch.stack([field[:, y + 2:y + 2 + size, x + 2:x + 2 + size] for field, x, y in fields])
            state = generator.get_state()
            originals = self.anchor.predictor.predict_batch(lr, ids, fields, radius, 'd4', generator, as_luma=False)
            for scale in SCALES:
                rgb = originals[scale]
                y = self.anchor.current(rgb, lr, scale)
                config = self.choices[scale]
                if config['blend']:
                    shared.set_state(state)
                    out = predict(self.anchor.trained[scale], expanded, shared, self_ensemble=True)
                    refined = center_crop(out, radius * scale).clamp(0, 1)
                    rgb = torch.lerp(rgb, refined, config['blend'])
                    y = torch.lerp(y, luma(refined), config['blend'])
                if config['update']:
                    correction = self.anchor.propose(rgb, y, lr, scale)[config['band']]
                    y = (y + config['update'] * correction).clamp(16 / 255, 235 / 255)
                if any(any(option) for option in weights[scale]):
                    shared.set_state(state)
                    out = predict(self.models[scale], expanded, shared, self_ensemble=True)
                    adapted = luma(center_crop(out, radius * scale).clamp(0, 1))
                    bands = frequency_bands(adapted - y, scale)
                expected = luma(hr[scale])
                for index, option in enumerate(weights[scale]):
                    result = y
                    if any(option):
                        adjustment = sum(weight * band for weight, band in zip(option, bands, strict=True))
                        result = (y + adjustment).clamp(16 / 255, 235 / 255)
                    p, s = image_metrics(result, expected)
                    scores[scale][index].extend(dict(id=key, seed=SEED, psnr=pv, ssim=sv)
                                                for key, pv, sv in zip(ids, p.tolist(), s.tolist()))
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--h07', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    comparison = Comparison(args.root, args.h07)
    if args.stage == 'select':
        reports = comparison.score('val', {scale: OPTIONS for scale in SCALES})
        for scale, tracks in reports.items():
            rows = [dict(weights=weights, **summarize(entries))
                    for weights, entries in zip(OPTIONS, tracks, strict=True)]
            old = rows[0]
            selected = max((row for row in rows if row['ssim'] >= old['ssim']), key=lambda row: row['psnr'])
            dump_json(args.root / f'x{scale}/selection.json', dict(selected=selected, baseline=old, candidates=rows,
                                                                    sources=comparison.digests))
            print(scale, 'baseline', old, 'selected', selected, flush=True)
        return

    choices = {scale: json.loads((args.root / f'x{scale}/selection.json').read_text()) for scale in SCALES}
    if any(entry['sources'] != comparison.digests for entry in choices.values()):
        raise ValueError('A locked source changed.')
    reports = comparison.score('test', {scale: [choice['selected']['weights']] for scale, choice in choices.items()})
    for scale, tracks in reports.items():
        out = args.root / f'x{scale}/test'
        dump_csv(out / 'per_image.csv', tracks[0])
        metrics = summarize(tracks[0])
        manifest = BASE / f'x{scale}/references/test/aligned.csv'
        dump_json(out / 'summary.json', dict(scale=scale, split='test', metric='BT.601 Y', border_hr=0,
                                             manifest_sha256=manifest_digest(manifest), selected=choices[scale]['selected'],
                                             patch_input=dict(GRID, batch=4, fields=1), **metrics))
        print(scale, metrics, flush=True)


if __name__ == '__main__':
    main()
