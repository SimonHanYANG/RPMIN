import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from optimize_mitos_rpmin import GRID, SEED, correct, data
from rpmin.data import PairedGrid, PairedImages, read_manifest
from rpmin.engine import dump_csv, dump_json, load_model, loader, manifest_digest, predict, save_checkpoint, summarize
from rpmin.objective import image_metrics, ssim_map


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation), nn.GELU(),
                                    nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + 0.2 * self.layers(x)


class DetailRefinement(nn.Module):
    def __init__(self, scale, width=64, with_position=False):
        super().__init__()
        self.scale = scale
        self.phase_channels = 3 * scale**2
        self.stem = nn.Conv2d(3 + 2 * self.phase_channels + 2 * with_position, width, 3, padding=1)
        self.layers = nn.Sequential(*(ResidualBlock(width, rate) for rate in (1, 2, 3, 1, 2, 3)))
        self.output = nn.Conv2d(width, self.phase_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lr, outputs, base=None, position=None):
        if base is None:
            base = (outputs[0] + outputs[1]) * 0.5
        feature = [F.pixel_unshuffle(v, self.scale) for v in (base, outputs[0] - outputs[1])]
        if position is not None:
            feature.append(position)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            stem = self.stem(torch.cat([lr, *feature], dim=1))
            residual = self.output(self.layers(stem))
        return base + F.pixel_shuffle(residual.float(), self.scale)


def load_sources(path, train, device):
    previous = json.loads((path / 'selection.json').read_text())
    selected = previous['selected']
    if selected['checkpoints'] != ['detail', 'width48'] or selected['pad'] or selected['d4']:
        raise ValueError('This refinement requires the locked two-model patch ensemble.')
    if previous['manifest_sha256'] != manifest_digest(train):
        raise ValueError('The training manifest changed.')
    models = []
    for name in selected['checkpoints']:
        source = previous['weights'][name]
        if manifest_digest(source['path']) != source['sha256']:
            raise ValueError('A source checkpoint changed.')
        models.append(load_model(source['path'], device).eval())
    correction = None
    if selected['strength']:
        parameters = path / 'camera-residual.pt'
        if manifest_digest(parameters) != previous['correction_sha256']:
            raise ValueError('The source camera correction changed.')
        correction = torch.load(parameters, map_location=device, weights_only=True)
        correction['strength'] = selected['strength']
    return models, correction


@torch.no_grad()
def predict_sources(models, lr, generator):
    seed_state = generator.get_state()
    outputs = []
    for model in models:
        generator.set_state(seed_state)
        outputs.append(predict(model, lr, generator))
    return outputs


def positions(image, ids, field_size):
    _, _, height, width = image.shape
    offsets = [[int(value) for value in key.rsplit('-x', 1)[1].split('-y')] for key in ids]
    x0, y0 = image.new_tensor(offsets).unbind(1)
    x = (x0[:, None, None] + torch.arange(width, device=image.device)[None, None, :]) / field_size[0]
    y = (y0[:, None, None] + torch.arange(height, device=image.device)[None, :, None]) / field_size[1]
    return torch.stack(torch.broadcast_tensors(x, y), dim=1) * 2 - 1


@torch.inference_mode()
def evaluate(batches, models, model, device, correction, blends, calibrated=False):
    generator = torch.Generator().manual_seed(SEED)
    rows = [[] for _ in blends]
    for lr, hr, ids in batches:
        lr, hr = lr.to(device), hr.to(device)
        outputs = predict_sources(models, lr, generator)
        base = (outputs[0] + outputs[1]) * 0.5
        old = correct(base, lr, ids, correction) if correction is not None else base
        refined = (model(lr, outputs, old, positions(lr, ids, correction['image_size']))
                   if calibrated else model(lr, outputs))
        for index, blend in enumerate(blends):
            p, s = image_metrics(torch.lerp(old, refined, blend), hr)
            rows[index].extend(dict(id=key, seed=SEED, psnr=pv, ssim=sv)
                               for key, pv, sv in zip(ids, p.tolist(), s.tolist()))
    return [summarize(entries) for entries in rows], rows


def fit(args):
    if (args.out / 'selection.json').exists():
        raise FileExistsError(args.out / 'selection.json')
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    random.seed(193)
    torch.manual_seed(193)
    models, correction = load_sources(args.prior, args.train, device)
    if args.calibrated and correction is None:
        raise ValueError('A fitted source correction is required for calibrated refinement.')
    model = DetailRefinement(args.scale, with_position=args.calibrated).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    pairs = read_manifest(args.train)
    if args.calibrated:
        train = PairedGrid(pairs, 'train', args.scale, patch=80, stride_x=58, stride_y=61, origin=1)
    else:
        train = PairedImages(pairs, 'train', args.scale, patch=80, samples_per_pair=64, cache=True)
    batches = loader(train, device, 0, batch=4, seed=193)
    validation = data(args.train, 'val', args.scale, device)
    scores, _ = evaluate(validation, models, model, device, correction, (0.0,), args.calibrated)
    baseline = scores[0]
    selected = dict(step=0, blend=0.0, **baseline)
    history = []
    shift_rng = torch.Generator().manual_seed(194)
    iterator = iter(batches)
    for step in range(1, args.steps + 1):
        batch = next(iterator, None)
        if batch is None:
            iterator = iter(batches)
            batch = next(iterator)
        lr, hr = (v.to(device, non_blocking=True) for v in batch[:2])
        multiplier = 0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2
        optimizer.param_groups[0]['lr'] = 3e-4 * multiplier
        outputs = predict_sources(models, lr, shift_rng)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.calibrated:
            base = correct((outputs[0] + outputs[1]) * 0.5, lr, batch[2], correction)
            sr = model(lr, outputs, base, positions(lr, batch[2], correction['image_size']))
        else:
            sr = model(lr, outputs)
        error = sr - hr
        similarity, _ = ssim_map(sr, hr)
        loss = error.square().mean() + 0.15 * error.abs().mean() + 0.10 * (1 - similarity.mean())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 400 == 0 or step == args.steps:
            model.eval()
            blends = (0.5, 0.75, 1.0)
            scores, _ = evaluate(validation, models, model, device, correction, blends, args.calibrated)
            for blend, stats in zip(blends, scores, strict=True):
                row = dict(step=step, blend=blend, **stats)
                history.append(row)
                if row['psnr'] > selected['psnr'] and row['ssim'] >= baseline['ssim']:
                    selected = row
                    save_checkpoint(args.out / 'best.pt', {'model': model.state_dict()})
            dump_csv(args.out / 'validation.csv', history)
            print(step, float(loss.detach()), scores[-1], 'best', selected, flush=True)
    dump_json(args.out / 'selection.json', dict(baseline=baseline, selected=selected, steps=args.steps,
                                               calibrated=args.calibrated,
                                               train_sha256=manifest_digest(args.train),
                                               prior_sha256=manifest_digest(args.prior / 'selection.json')))


@torch.inference_mode()
def test(args):
    selection = json.loads((args.out / 'selection.json').read_text())
    if (selection['train_sha256'] != manifest_digest(args.train)
            or selection['prior_sha256'] != manifest_digest(args.prior / 'selection.json')):
        raise ValueError('The locked manifests or prior checkpoint selection changed.')
    if {r.group for r in read_manifest(args.train)} & {r.group for r in read_manifest(args.test)}:
        raise ValueError('Test field occurs in the training or validation data.')
    device = torch.device('cuda')
    models, correction = load_sources(args.prior, args.train, device)
    calibrated = selection.get('calibrated', False)
    model = DetailRefinement(args.scale, with_position=calibrated).to(device).eval()
    selected = selection['selected']
    if selected['step']:
        state = torch.load(args.out / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
    scores, rows = evaluate(data(args.test, 'test', args.scale, device), models, model, device, correction,
                            (selected['blend'],), calibrated)
    dump_csv(args.out / 'test/per_image.csv', rows[0])
    dump_json(args.out / 'test/summary.json', dict(scale=args.scale, split='test', border_hr=0,
                                                  manifest_sha256=manifest_digest(args.test),
                                                  patch_input=dict(GRID, batch=4, fields=1),
                                                  selected=selected, **scores[0]))
    print(scores[0], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('fit', 'test'))
    parser.add_argument('--scale', type=int, choices=(2, 4), required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--calibrated', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'fit':
        if args.steps < 1:
            parser.error('Use a positive training budget.')
        fit(args)
    else:
        if args.test is None:
            parser.error('Provide the locked test manifest.')
        test(args)


if __name__ == '__main__':
    main()
