import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from cross_scale_mitos_rpmin import BASE, GRID, PRIOR, SEED, transfer_scale
from refine_mitos_rpmin import correct, load_sources
from rpmin.data import PairedGrid, load_rgb, read_manifest
from rpmin.engine import dump_csv, dump_json, load_model, manifest_digest, predict, save_checkpoint, summarize
from rpmin.objective import image_metrics
from train_mitos_camera import luma


SCALES = (2, 4)
CROSS = Path('runs/mitos14-rpmin-cross-scale')
LUMA = Path('runs/mitos14-rpmin-luma')
CONTEXTS = ((0, 'single'), (16, 'single'), (32, 'single'), (16, 'phase'), (16, 'd4'))


class Predictor:
    def __init__(self):
        self.device = torch.device('cuda')
        self.models, self.calibration, self.cross, self.mix, self.tuned, self.sources = ({} for _ in range(6))
        for scale in SCALES:
            train = BASE / f'x{scale}/references/train-val/aligned.csv'
            prior = PRIOR / f'x{scale}/locked'
            cross = CROSS / f'x{scale}/selection.json'
            luma_choice = LUMA / f'x{scale}/locked/selection.json'
            self.models[scale], self.calibration[scale] = load_sources(prior, train, self.device)
            self.cross[scale] = json.loads(cross.read_text())['selected']
            self.mix[scale] = json.loads(luma_choice.read_text())['selected']['blend']
            self.sources[str(scale)] = {key: manifest_digest(path) for key, path in (
                ('prior', prior / 'selection.json'), ('cross', cross), ('luma', luma_choice), ('train', train),
            )}
            if self.mix[scale]:
                tuned = LUMA / f'x{scale}/camera'
                checkpoint = tuned / 'best.pt'
                fitted = json.loads((tuned / 'selection.json').read_text())
                if fitted['checkpoint_sha256'] != manifest_digest(checkpoint) or fitted['train_sha256'] != manifest_digest(train):
                    raise ValueError('Unexpected luminance checkpoint or training manifest.')
                self.tuned[scale] = load_model(checkpoint, self.device).eval()
                self.sources[str(scale)]['tuned'] = fitted['checkpoint_sha256']

    def make_batch(self, split, context):
        files = {scale: BASE / f'x{scale}/references' for scale in SCALES}
        datasets = {}
        for scale in SCALES:
            train = read_manifest(files[scale] / 'train-val/aligned.csv')
            pairs = read_manifest(files[scale] / 'test/aligned.csv') if split == 'test' else train
            if split == 'test' and {p.group for p in train} & {p.group for p in pairs}:
                raise ValueError('The test field occurs in the training manifest.')
            datasets[scale] = PairedGrid(pairs, split, scale, **GRID)
        low = datasets[2]
        high = datasets[4]
        if [(p.id, x, y) for p, x, y in low.crops] != [(p.id, x, y) for p, x, y in high.crops]:
            raise ValueError('The two scales use different scoring grids.')
        full = {}
        for pair in low.pairs:
            matched = next(p for p in high.pairs if p.id == pair.id)
            if manifest_digest(pair.lr) != manifest_digest(matched.lr):
                raise ValueError('The two scales use different LR imagery.')
            field = load_rgb(pair.lr).to(self.device)
            full[pair.id] = F.pad(field[None], (context + 2,) * 4, mode='reflect')[0]

        for start in range(0, len(low), 4):
            window = range(start, min(start + 4, len(low)))
            tiles = [low[index] for index in window]
            ids = [record[2] for record in tiles]
            gt = {2: torch.stack([record[1] for record in tiles]).to(self.device),
                  4: torch.stack([high[index][1] for index in window]).to(self.device)}
            positions = low.crops[start:start + len(tiles)]
            lr = torch.stack([record[0] for record in tiles]).to(self.device)
            yield lr, gt, ids, [(full[pair.id], x, y) for pair, x, y in positions]

    def predict_batch(self, lr, ids, fields, radius, mode, generator, *, as_luma=True):
        shifts = ((0, 0), (2, 0), (0, 2), (2, 2)) if mode == 'phase' else ((0, 0),)
        initial = generator.get_state()
        side = GRID['patch'] + radius * 2
        originals, heads = {}, {}
        for scale in SCALES:
            averaged, last = [], []
            for dx, dy in shifts:
                source = torch.stack([field[:, y + 2 - dy:y + 2 - dy + side, x + 2 - dx:x + 2 - dx + side]
                                      for field, x, y in fields]) if radius else lr
                start_x, start_y = (radius + dx) * scale, (radius + dy) * scale
                predictions = []
                models = [*self.models[scale]]
                if scale in self.tuned:
                    models.append(self.tuned[scale])
                for model in models:
                    generator.set_state(initial)
                    sr = predict(model, source, generator, self_ensemble=mode == 'd4')
                    width = GRID['patch'] * scale
                    predictions.append(sr[:, :, start_y:start_y + width, start_x:start_x + width])
                averaged.append((predictions[0] + predictions[1]) * 0.5)
                if scale in self.tuned:
                    last.append(predictions[-1])
            base = torch.stack(averaged).mean(0)
            originals[scale] = correct(base, lr, ids, self.calibration[scale])
            if last:
                heads[scale] = torch.stack(last).mean(0)

        output = {}
        for scale in SCALES:
            prior = self.cross[scale]
            image = transfer_scale(originals, scale, prior['strength'], prior['centered'])
            if scale in heads:
                image = torch.lerp(image, heads[scale], self.mix[scale])
            image = image.clamp(0, 1)
            output[scale] = luma(image) if as_luma else image
        return output


def phase_features(lr, predictions, scale):
    center = lr - 0.5
    taps = F.unfold(F.pad(center, (2,) * 4, mode='reflect'), kernel_size=5).reshape(len(lr), 75, *lr.shape[-2:])
    phases = F.pixel_unshuffle(predictions - 0.5, scale)
    return torch.cat((torch.ones_like(center[:, :1]), taps, center.square(), center * center.roll(1, 1), phases),
                     dim=1).permute(0, 2, 3, 1)


@torch.inference_mode()
def fit_residual(predictor, modes):
    totals = {}
    for radius, mode in set(modes.values()):
        scales = [scale for scale in SCALES if modes[scale] == (radius, mode)]
        generator = torch.Generator().manual_seed(SEED)
        for i, (lr, gt, ids, fields) in enumerate(predictor.make_batch('train', radius)):
            output = predictor.predict_batch(lr, ids, fields, radius, mode, generator)
            y, x = divmod(i % 16, 4)
            for scale in scales:
                design = phase_features(lr, output[scale], scale)[:, y::4, x::4].flatten(0, 2)
                target = F.pixel_unshuffle(luma(gt[scale]) - output[scale], scale).permute(0, 2, 3, 1)
                target = target[:, y::4, x::4].flatten(0, 2)
                gram, cross, count = totals.get(scale, (0, 0, 0))
                totals[scale] = (gram + design.T @ design, cross + design.T @ target, count + len(target))
    result = {}
    for scale, (gram, cross, count) in totals.items():
        penalty = torch.eye(len(gram), device=gram.device) * 0.00012
        penalty[0, 0] = 0
        result[scale] = torch.linalg.solve(gram / count + penalty, cross / count)
    return result


@torch.inference_mode()
def score(predictor, split, radius, mode, strengths, coefficients):
    rows = {scale: {strength: [] for strength in weights} for scale, weights in strengths.items()}
    generator = torch.Generator().manual_seed(SEED)
    for lr, gt, ids, fields in predictor.make_batch(split, radius):
        output = predictor.predict_batch(lr, ids, fields, radius, mode, generator)
        for scale, options in rows.items():
            expected = luma(gt[scale])
            if any(options):
                design = phase_features(lr, output[scale], scale)
                residual = F.pixel_shuffle((design @ coefficients[scale]).permute(0, 3, 1, 2), scale)
            for strength, records in options.items():
                prediction = output[scale] + strength * residual if strength else output[scale]
                p, s = image_metrics(prediction.clamp(16 / 255, 235 / 255), expected)
                records.extend(dict(id=key, seed=SEED, psnr=pv, ssim=sv)
                               for key, pv, sv in zip(ids, p.tolist(), s.tolist()))
    return rows


def choose(rows, reference):
    return max((r for r in rows if r['ssim'] >= reference['ssim']), key=lambda r: r['psnr'])


def select(predictor, out):
    out.mkdir(parents=True, exist_ok=True)
    candidates = {scale: [] for scale in SCALES}
    for radius, mode in CONTEXTS:
        scores = score(predictor, 'val', radius, mode, {scale: (0.0,) for scale in SCALES}, {})
        for scale, report in scores.items():
            row = dict(radius=radius, mode=mode, strength=0.0, **summarize(report[0.0]))
            candidates[scale].append(row)
            print(scale, row, flush=True)
    references = {scale: rows[0] for scale, rows in candidates.items()}
    context = {scale: (item['radius'], item['mode']) for scale, rows in candidates.items()
               for item in (choose(rows, references[scale]),)}
    params = fit_residual(predictor, context)
    for radius, mode in set(context.values()):
        strengths = {scale: (0.25, 0.5, 0.75, 1.0) for scale in SCALES if context[scale] == (radius, mode)}
        for scale, report in score(predictor, 'val', radius, mode, strengths, params).items():
            candidates[scale].extend(dict(radius=radius, mode=mode, strength=strength, **summarize(rows))
                                     for strength, rows in report.items())
    weights = out / 'luma-residual.pt'
    save_checkpoint(weights, {str(scale): value.cpu() for scale, value in params.items()})
    for scale in SCALES:
        selected = choose(candidates[scale], references[scale])
        dump_json(out / f'x{scale}/selection.json', dict(selected=selected, baseline=references[scale],
                   candidates=candidates[scale], sources=predictor.sources, residual_sha256=manifest_digest(weights)))
        print(scale, 'selected', selected, flush=True)


def test(predictor, out):
    weights = out / 'luma-residual.pt'
    selected = {}
    for scale in SCALES:
        report = json.loads((out / f'x{scale}/selection.json').read_text())
        if report['sources'] != predictor.sources or report['residual_sha256'] != manifest_digest(weights):
            raise ValueError('A locked source changed.')
        selected[scale] = report['selected']
    params = {int(scale): value for scale, value in torch.load(weights, map_location=predictor.device,
                                                               weights_only=True).items()}
    modes = {(item['radius'], item['mode']) for item in selected.values()}
    for radius, mode in modes:
        strengths = {scale: (item['strength'],) for scale, item in selected.items()
                     if (item['radius'], item['mode']) == (radius, mode)}
        outcomes = score(predictor, 'test', radius, mode, strengths, params)
        for scale, report in outcomes.items():
            rows = report[selected[scale]['strength']]
            dest = out / f'x{scale}/test'
            source = BASE / f'x{scale}/references/test/aligned.csv'
            dump_csv(dest / 'per_image.csv', rows)
            metrics = summarize(rows)
            dump_json(dest / 'summary.json', dict(scale=scale, split='test', metric='BT.601 Y', border_hr=0,
                       manifest_sha256=manifest_digest(source), patch_input=dict(GRID, batch=4, fields=1),
                       selected=selected[scale], **metrics))
            print(scale, metrics, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictor = Predictor()
    (select if args.stage == 'select' else test)(predictor, args.out)


if __name__ == '__main__':
    main()
