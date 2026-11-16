"""Validate and test ×4 checkpoint ensembles and inference context on fixed grids."""

import argparse
import hashlib
from pathlib import Path
from statistics import fmean, stdev

import torch

from rpmin.data import PairedGrid, PairedImages, read_manifest
from rpmin.engine import dump_csv, dump_json, load_model, loader, predict
from rpmin.objective import grid_metrics, image_metrics


def predict_ensemble(models, lr, generator):
    initial_state = generator.get_state()
    total = torch.zeros((), device=lr.device, dtype=lr.dtype)
    for model in models:
        generator.set_state(initial_state)
        total = total + predict(model, lr, generator, self_ensemble=True)
    return total / len(models)


@torch.inference_mode()
def score(pairs, split, models, context, device, seeds):
    scale = 4
    grid = dict(patch=128, stride_x=32, stride_y=25)
    full = context == 'whole'
    if full:
        data = loader(PairedImages(pairs, split, scale), device, 0)
    else:
        data = loader(PairedGrid(pairs, split, scale, **grid), device, 0, batch=4)

    rows, runs = [], []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        start = len(rows)
        for lr, hr, identifiers in data:
            lr, hr = lr.to(device), hr.to(device)
            sr = predict_ensemble(models, lr, generator)
            if full:
                positions, psnr, ssim = grid_metrics(sr, hr, scale, **grid)
                identifiers = [f'{identifiers[0]}-x{x}-y{y}' for x, y in positions]
            else:
                psnr, ssim = image_metrics(sr, hr)
            rows.extend({'id': identifier, 'seed': seed, 'psnr': p, 'ssim': s}
                        for identifier, p, s in zip(identifiers, psnr.tolist(), ssim.tolist()))
        batch_rows = rows[start:]
        runs.append({'seed': seed, 'patches': len(batch_rows),
                     **{key: fmean(row[key] for row in batch_rows) for key in ('psnr', 'ssim')}})
    result = {'context': context, 'runs': runs}
    for key in ('psnr', 'ssim'):
        values = [run[key] for run in runs]
        result[key] = {'mean': fmean(values), 'std': stdev(values) if len(values) > 1 else None}
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--refined', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--include-single', action='store_true', help='Also compare the second checkpoint alone.')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    paths = {'base': args.base, 'refined': args.refined}
    base = load_model(args.base, device)
    refined = load_model(args.refined, device)
    if {base.config.scale, refined.config.scale} != {4}:
        raise ValueError('Both checkpoints must be ×4 models.')
    models = {'base': [base], 'ensemble': [base, refined]}
    if args.include_single:
        models['refined'] = [refined]
    val_pairs = read_manifest(args.validation)
    validation, choices = {}, {}
    for context in ('patch', 'whole'):
        validation[context] = {
            method: score(val_pairs, 'val', group, context, device, [101])[0]
            for method, group in models.items()
        }
        results = validation[context]
        floor = results['base']['ssim']['mean']
        choices[context] = max((key for key, metrics in results.items() if metrics['ssim']['mean'] >= floor),
                               key=lambda key: results[key]['psnr']['mean'])
    sources = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for key, path in paths.items()}
    dump_json(args.output / 'frozen_choice.json', {'selected': choices, 'validation': validation, 'checkpoints': sources})

    test_pairs = read_manifest(args.test)
    if {pair.group for pair in val_pairs} & {pair.group for pair in test_pairs}:
        raise ValueError('Validation or training groups overlap the holdout manifest.')
    scores = {}
    for context, winner in choices.items():
        scores[context], rows = score(test_pairs, 'test', models[winner], context, device, [101, 102, 103])
        scores[context]['selected'] = winner
        dump_csv(args.output / f'{context}_per_patch.csv', rows)
    dump_json(args.output / 'comparison.json', scores)
    print(scores)


if __name__ == '__main__':
    main()
