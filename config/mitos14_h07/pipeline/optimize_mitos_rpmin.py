import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'RPMIN-e79d2b3'))

from rpmin.data import PairedGrid, read_manifest
from rpmin.engine import dump_csv, dump_json, load_model, loader, manifest_digest, predict, save_checkpoint, summarize
from rpmin.objective import image_metrics


GRID = dict(patch=128, stride_x=128, stride_y=128, origin=1)
INFERENCE = ((0, False), (16, False), (16, True))
SEED = 101


def data(manifest, split, scale, device):
    dataset = PairedGrid(read_manifest(manifest), split, scale, **GRID)
    return loader(dataset, device, 0, batch=4)


def forward(models, lr, generator, pad, d4):
    inputs = F.pad(lr, (pad,) * 4, mode='reflect') if pad else lr
    state = generator.get_state()
    predictions = []
    for model in models:
        generator.set_state(state)
        prediction = predict(model, inputs, generator, self_ensemble=d4)
        edge = pad * model.config.scale
        predictions.append(prediction[..., edge:-edge, edge:-edge] if edge else prediction)
    return torch.stack(predictions).mean(0)


def basis(lr, ids, image_size):
    batch, _, height, width = lr.shape
    padded = F.pad(lr - 0.5, (2,) * 4, mode='reflect')
    taps = F.unfold(padded, kernel_size=5).reshape(batch, -1, height, width)
    center = lr - 0.5
    localization = []
    pixel_offsets = [[int(part) for part in value.rsplit('-x', 1)[1].split('-y')] for value in ids]
    left, top = lr.new_tensor(pixel_offsets).unbind(1)
    x = (2 * (left[:, None, None] + torch.arange(width, device=lr.device)[None, None, :]) / image_size[0] - 1)
    y = (2 * (top[:, None, None] + torch.arange(height, device=lr.device)[None, :, None]) / image_size[1] - 1)
    dx = (padded[..., 2:-2, 3:-1] - padded[..., 2:-2, 1:-3]) / 2
    dy = (padded[..., 3:-1, 2:-2] - padded[..., 1:-3, 2:-2]) / 2
    for pos in (x, y, x * y, x.square(), y.square()):
        localization.extend([dx * pos[:, None], dy * pos[:, None]])
    high = center - F.avg_pool2d(padded, 5, stride=1)
    return torch.cat((torch.ones_like(center[:, :1]), taps, center.square(), center * center.roll(1, 1),
                      high * center, *localization), dim=1).permute(0, 2, 3, 1)


def correct(sr, lr, ids, parameters):
    design = basis(lr, ids, parameters['image_size'])
    residual = F.pixel_shuffle((design @ parameters['coefficients']).permute(0, 3, 1, 2), parameters['scale'])
    return sr + parameters['strength'] * residual


def fit_correction(data_loader, models, device, pad, d4):
    pairs = data_loader.dataset.pairs
    sizes = set()
    for pair in pairs:
        with Image.open(pair.lr) as image:
            sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError('The camera correction assumes a consistent LR field geometry.')
    size = sizes.pop()
    scale = models[0].config.scale
    generator = torch.Generator().manual_seed(SEED)
    gram, target, pixels = None, None, 0
    for step, (lr, hr, ids) in enumerate(data_loader):
        lr, hr = lr.to(device), hr.to(device)
        sr = forward(models, lr, generator, pad, d4)
        residual = F.pixel_unshuffle(hr - sr, scale).permute(0, 2, 3, 1)
        features = basis(lr, ids, size)
        y, x = divmod(step % 16, 4)
        features = features[:, y::4, x::4].flatten(0, 2)
        residual = residual[:, y::4, x::4].flatten(0, 2)
        cross, right = features.T @ features, features.T @ residual
        gram = cross if gram is None else gram + cross
        target = right if target is None else target + right
        pixels += len(features)
    ridge = torch.eye(len(gram), device=device) * 0.00005
    ridge[0, 0] = 0
    return dict(coefficients=torch.linalg.solve(gram / pixels + ridge, target / pixels),
                image_size=size, scale=scale)


def evaluate(data_loader, models, device, pad, d4, correction=None):
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for lr, hr, ids in data_loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = forward(models, lr, generator, pad, d4)
        if correction is not None:
            sr = correct(sr, lr, ids, correction)
        psnr, ssim = image_metrics(sr, hr, 0)
        rows.extend(dict(id=key, seed=SEED, psnr=p, ssim=s)
                    for key, p, s in zip(ids, psnr.tolist(), ssim.tolist()))
    return summarize(rows), rows


def qualify(candidates, baseline):
    eligible = [row for row in candidates if row['psnr'] >= baseline['psnr'] and row['ssim'] >= baseline['ssim']]
    return max(eligible, key=lambda row: (row['psnr'], row['ssim']))


@torch.inference_mode()
def select(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    validation = data(args.train, 'val', args.scale, args.device)
    models = {name: load_model(path, args.device) for name, path in args.checkpoint.items()}
    variants = {name: [name] for name in models}
    if len(models) > 1:
        variants['ensemble'] = list(models)[-2:]
    candidates = []
    for name, keys in variants.items():
        for pad, d4 in INFERENCE:
            scores, _ = evaluate(validation, [models[key] for key in keys], args.device, pad, d4)
            row = dict(variant=name, checkpoints=keys, pad=pad, d4=d4, strength=0, **scores)
            candidates.append(row)
            print(row, flush=True)
    baseline = candidates[0]
    choice = qualify(candidates, baseline)
    selected_models = [models[key] for key in choice['checkpoints']]
    train = data(args.train, 'train', args.scale, args.device)
    correction = fit_correction(train, selected_models, args.device, choice['pad'], choice['d4'])
    for strength in (0.5, 1.0):
        scores, _ = evaluate(validation, selected_models, args.device, choice['pad'], choice['d4'],
                             dict(correction, strength=strength))
        row = {**choice, **scores, 'strength': strength}
        candidates.append(row)
        print(row, flush=True)
    selected = qualify(candidates, baseline)
    weights = {name: dict(path=str(path), sha256=manifest_digest(path)) for name, path in args.checkpoint.items()}
    correction_file = out / 'camera-residual.pt'
    save_checkpoint(correction_file, dict(correction, coefficients=correction['coefficients'].cpu()))
    dump_json(out / 'selection.json', dict(selected=selected, candidates=candidates, weights=weights,
                                           correction_sha256=manifest_digest(correction_file),
                                           manifest_sha256=manifest_digest(args.train), seed=SEED))


@torch.inference_mode()
def test(args):
    out = args.out
    record = json.loads((out / 'selection.json').read_text())
    if record['manifest_sha256'] != manifest_digest(args.train):
        raise ValueError('Training manifest differs from the locked validation run.')
    if {pair.group for pair in read_manifest(args.train)} & {pair.group for pair in read_manifest(args.test)}:
        raise ValueError('Test field occurs in the training or validation manifest.')
    selected = record['selected']
    models = []
    for key in selected['checkpoints']:
        source = record['weights'][key]
        if manifest_digest(source['path']) != source['sha256']:
            raise ValueError('A selected checkpoint has changed.')
        models.append(load_model(source['path'], args.device))
    correction = None
    if selected['strength']:
        correction_file = out / 'camera-residual.pt'
        if manifest_digest(correction_file) != record['correction_sha256']:
            raise ValueError('The fitted correction has changed.')
        correction = torch.load(correction_file, map_location=args.device, weights_only=True)
        correction['strength'] = selected['strength']
    scores, rows = evaluate(data(args.test, 'test', args.scale, args.device), models, args.device,
                            selected['pad'], selected['d4'], correction)
    dump_csv(out / 'test/per_image.csv', rows)
    dump_json(out / 'test/summary.json', dict(scale=args.scale, split='test', border_hr=0, seed=SEED,
                                              manifest_sha256=manifest_digest(args.test),
                                              patch_input={**GRID, 'fields': 1, 'batch': 4},
                                              selected=selected, **scores))
    print(scores, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['select', 'test'])
    parser.add_argument('--scale', type=int, choices=[2, 4], required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', metavar='NAME=PATH', nargs='+')
    args = parser.parse_args()
    args.device = torch.device('cuda')
    torch.set_num_threads(2)
    if args.stage == 'select':
        entries = [value.split('=', 1) for value in args.checkpoint or []]
        if not entries or any(len(entry) != 2 or not all(entry) for entry in entries):
            parser.error('Provide named checkpoints, listing the original baseline first.')
        args.checkpoint = {name: Path(path) for name, path in entries}
        if len(args.checkpoint) != len(entries):
            parser.error('Checkpoint names must be distinct.')
        select(args)
    else:
        if args.test is None:
            parser.error('Provide the locked test manifest.')
        test(args)


if __name__ == '__main__':
    main()
