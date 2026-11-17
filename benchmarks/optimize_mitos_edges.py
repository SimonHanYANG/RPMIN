import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from optimize_mitos_context import BASE, GRID, SCALES, SEED, Predictor, luma, phase_features
from rpmin.engine import dump_csv, dump_json, manifest_digest, save_checkpoint, summarize
from rpmin.objective import image_metrics


CONTEXT = Path('runs/mitos14-rpmin-context')
OPTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
GROUPS = 9


def orientation(lr):
    image = F.pad(luma(lr), (1,) * 4, mode='reflect')
    x = (image[..., 1:-1, 2:] - image[..., 1:-1, :-2]) * 0.5
    y = (image[..., 2:, 1:-1] - image[..., :-2, 1:-1]) * 0.5
    xx, yy, xy = [F.avg_pool2d(r, 3, stride=1, padding=1)[:, 0] for r in (x.square(), y.square(), x * y)]
    theta = (0.5 * torch.atan2(2 * xy, xx - yy)).remainder(math.pi)
    angle = (theta * ((GROUPS - 1) / math.pi)).long().clamp(max=GROUPS - 2) + 1
    return torch.where(xx + yy > 0.018**2, angle, 0)


class Run:
    def __init__(self, scale):
        self.scale = scale
        self.predictor = Predictor()
        self.device = self.predictor.device
        prior = CONTEXT / f'x{scale}/selection.json'
        frozen = json.loads(prior.read_text())
        if frozen['sources'] != self.predictor.sources:
            raise ValueError('The context source selection changed.')
        self.context = frozen['selected']
        self.sources = {'prior': manifest_digest(prior), 'weights': frozen['residual_sha256']}
        weights = CONTEXT / 'luma-residual.pt'
        if manifest_digest(weights) != self.sources['weights']:
            raise ValueError('Context weights changed.')
        self.calibration = torch.load(weights, map_location=self.device, weights_only=True)[str(scale)]

    def batches(self, split):
        radius, mode, strength = [self.context[key] for key in ('radius', 'mode', 'strength')]
        rng = torch.Generator().manual_seed(SEED)
        for lr, gt, ids, fields in self.predictor.make_batch(split, radius):
            predicted = self.predictor.predict_batch(lr, ids, fields, radius, mode, rng)[self.scale]
            if strength:
                basis = phase_features(lr, predicted, self.scale)
                residual = F.pixel_shuffle((basis @ self.calibration).permute(0, 3, 1, 2), self.scale)
                predicted += strength * residual
            yield lr, predicted.clamp(16 / 255, 235 / 255), luma(gt[self.scale]), ids

    @torch.inference_mode()
    def fit(self):
        gram, cross = [0] * GROUPS, [0] * GROUPS
        counts = [0] * GROUPS
        for index, (lr, predicted, target, _) in enumerate(self.batches('train')):
            i, j = divmod(index % 16, 4)
            z = phase_features(lr, predicted, self.scale)[:, i::4, j::4].flatten(0, 2)
            residual = F.pixel_unshuffle(target - predicted, self.scale).permute(0, 2, 3, 1)
            residual = residual[:, i::4, j::4].flatten(0, 2)
            groups = orientation(lr)[:, i::4, j::4].flatten()
            for group in range(GROUPS):
                keep = groups == group
                a, b = z[keep], residual[keep]
                gram[group] = gram[group] + a.T @ a
                cross[group] = cross[group] + a.T @ b
                counts[group] += len(a)
        if min(counts) < 512:
            raise ValueError(f'Orientation groups have insufficient training support: {counts}')
        prior_gram = sum(gram) / sum(counts)
        prior_cross = sum(cross) / sum(counts)
        penalty = torch.eye(len(prior_gram), device=self.device) * 0.00015
        penalty[0, 0] = 0
        coefficients = [torch.linalg.solve((g / n) * 0.9 + prior_gram * 0.1 + penalty,
                                           (c / n) * 0.9 + prior_cross * 0.1)
                        for g, c, n in zip(gram, cross, counts, strict=True)]
        return torch.stack(coefficients), counts

    @torch.inference_mode()
    def score(self, split, parameters, options):
        rows = {weight: [] for weight in options}
        for lr, predicted, target, ids in self.batches(split):
            if any(options):
                features = phase_features(lr, predicted, self.scale).flatten(0, 2)
                groups = orientation(lr).flatten()
                delta = predicted.new_zeros(len(groups), self.scale**2)
                for group, matrix in enumerate(parameters):
                    active = groups == group
                    delta[active] = features[active] @ matrix
                delta = F.pixel_shuffle(delta.reshape(len(lr), *lr.shape[-2:], -1).permute(0, 3, 1, 2), self.scale)
            for weight in options:
                sr = (predicted + weight * delta).clamp(16 / 255, 235 / 255) if weight else predicted
                p, s = image_metrics(sr, target)
                rows[weight].extend(dict(id=key, seed=SEED, psnr=pv, ssim=sv)
                                    for key, pv, sv in zip(ids, p.tolist(), s.tolist()))
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--scale', type=int, choices=SCALES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = Run(args.scale)
    out = args.out / f'x{args.scale}'
    weights = out / 'edge-residual.pt'
    if args.stage == 'select':
        out.mkdir(parents=True, exist_ok=True)
        coefficients, counts = run.fit()
        candidates = [dict(weight=w, **summarize(rows)) for w, rows in run.score('val', coefficients, OPTIONS).items()]
        selected = max((row for row in candidates if row['ssim'] >= candidates[0]['ssim']), key=lambda row: row['psnr'])
        save_checkpoint(weights, {'coefficients': coefficients.cpu()})
        dump_json(out / 'selection.json', dict(candidates=candidates, selected=selected, sources=run.sources,
                                               training_counts=counts, checkpoint_sha256=manifest_digest(weights)))
        print(args.scale, selected, flush=True)
        return

    config = json.loads((out / 'selection.json').read_text())
    if config['sources'] != run.sources or config['checkpoint_sha256'] != manifest_digest(weights):
        raise ValueError('A locked source changed.')
    selected = config['selected']
    params = torch.load(weights, map_location=run.device, weights_only=True)['coefficients']
    weight = selected['weight']
    rows = run.score('test', params, (weight,))[weight]
    path = BASE / f'x{args.scale}/references/test/aligned.csv'
    dump_csv(out / 'test/per_image.csv', rows)
    metrics = summarize(rows)
    dump_json(out / 'test/summary.json', dict(scale=args.scale, split='test', metric='BT.601 Y', border_hr=0,
                                              manifest_sha256=manifest_digest(path), selected=selected,
                                              patch_input=dict(GRID, batch=4, fields=1), **metrics))
    print(args.scale, metrics, flush=True)


if __name__ == '__main__':
    main()
