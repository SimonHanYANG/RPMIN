import argparse
import importlib
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'RPMIN-e79d2b3'))

from rpmin.data import PairedGrid, PairedImages, read_manifest
from rpmin.engine import dump_csv, dump_json, manifest_digest, save_checkpoint, summarize
from rpmin.objective import image_metrics


def load_network(method, adapted=None, scale=4):
    if method == 'SeD':
        sys.path.insert(0, str(ROOT / 'SeD'))
        from models.model_rrdb import RRDBNet

        model = RRDBNet(in_nc=3, out_nc=3)
        checkpoint = ROOT / 'weights' / 'sed-rrdb.pt'
        key = 'model'
    else:
        # Load the official generator without the training and IQA plugins
        # eagerly imported by the vendored BasicSR package initializers.
        basicsr = ROOT / 'AESOP-SR' / 'AESOP' / 'basicsr'
        for name, folder in [('basicsr', basicsr), ('basicsr.archs', basicsr / 'archs')]:
            package = types.ModuleType(name)
            package.__path__ = [str(folder)]
            sys.modules[name] = package
        arch = importlib.import_module('basicsr.archs.rrdbnet_arch')
        model = arch.RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        checkpoint = ROOT / 'weights' / 'aesop-rrdb128.pth'
        key = 'params_ema'
    model.upsample_second = scale == 4
    if adapted is not None:
        checkpoint, key = adapted, 'model'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state[key], strict=True)
    return model.cuda().eval(), checkpoint


def train_transfer(method, manifest, out, scale, epochs, samples):
    seed, batch = 42, 8
    random.seed(seed)
    torch.manual_seed(seed)
    model, checkpoint = load_network(method, scale=scale)
    model.train()
    dataset = PairedImages(read_manifest(manifest), 'train', scale, patch=32, samples_per_pair=samples, cache=True)
    data = DataLoader(dataset, batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.99))
    dump_json(out / 'config.json', {
        'method': method, 'scale': scale, 'seed': seed, 'epochs': epochs, 'batch': batch, 'lr_patch': 32,
        'steps': epochs * len(data), 'train_fields': len(dataset.pairs), 'patches_per_field': samples, 'loss': 'L1',
        'optimizer': 'Adam', 'lr': 2e-5, 'betas': [0.9, 0.99],
        'manifest_sha256': manifest_digest(manifest), 'initial_checkpoint': str(checkpoint),
        'initial_checkpoint_sha256': manifest_digest(checkpoint),
    })
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for lr, hr, _ in data:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(lr.cuda()), hr.cuda())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if not np.isfinite(losses).all():
            raise FloatingPointError(f'Non-finite loss in epoch {epoch}')
        history.append({'epoch': epoch, 'train_l1': float(np.mean(losses))})
        print(method, history[-1], flush=True)
    save_checkpoint(out / 'adapted.pt', {'model': model.state_dict()})
    dump_csv(out / 'history.csv', history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['prepare', 'AESOP', 'SeD', 'NeXtSRGAN'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scale', type=int, choices=[2, 4], default=4)
    parser.add_argument('--stride-x', type=int, default=32)
    parser.add_argument('--stride-y', type=int, default=25)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--adapted', type=Path)
    parser.add_argument('--fit-transfer', action='store_true')
    parser.add_argument('--fit-epochs', type=int, default=3)
    parser.add_argument('--patches-per-field', type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fit_transfer:
        if args.method not in {'AESOP', 'SeD'} or args.fit_epochs < 1 or args.patches_per_field < 1:
            parser.error('Fit transfer requires AESOP or SeD and positive training counts')
        train_transfer(args.method, args.manifest, args.out, args.scale, args.fit_epochs, args.patches_per_field)
        return
    grid = dict(patch=128, stride_x=args.stride_x, stride_y=args.stride_y, origin=1)
    dataset = PairedGrid(read_manifest(args.manifest), args.split, args.scale, **grid)
    batches = DataLoader(dataset, batch_size=4)
    if args.method == 'prepare':
        images = [lr.mul(255).round().byte().permute(0, 2, 3, 1).numpy() for lr, _, _ in batches]
        np.save(args.out / f'{args.split}_lr.npy', np.concatenate(images))
        return

    metadata = {}
    if args.method == 'NeXtSRGAN':
        predictions = np.load(args.predictions, mmap_mode='r')
        hr_patch = grid['patch'] * args.scale
        if predictions.shape != (len(dataset), hr_patch, hr_patch, 3):
            raise ValueError(f'Unexpected prediction shape {predictions.shape}')
        metadata['prediction_path'] = str(args.predictions)
    else:
        model, checkpoint = load_network(args.method, args.adapted, args.scale)
        metadata['checkpoint'] = str(checkpoint)
        metadata['checkpoint_sha256'] = manifest_digest(checkpoint)
    rows = []
    with torch.inference_mode():
        for lr, hr, ids in batches:
            if args.method == 'NeXtSRGAN':
                data = np.array(predictions[len(rows):len(rows) + len(ids)])
                sr = torch.from_numpy(data).permute(0, 3, 1, 2).cuda()
            else:
                sr = model(lr.cuda())
            if sr.shape != hr.shape or not torch.isfinite(sr).all():
                raise ValueError(f'Invalid model output for {ids}')
            psnr, ssim = image_metrics(sr, hr.cuda(), 0)
            rows.extend({'id': identifier, 'psnr': p, 'ssim': s}
                        for identifier, p, s in zip(ids, psnr.tolist(), ssim.tolist()))

    dump_csv(args.out / 'per_image.csv', rows)
    metrics = summarize(rows)
    dump_json(args.out / 'summary.json', {
        'method': args.method, 'split': args.split, 'scale': args.scale, 'border_hr': 0,
        'manifest_sha256': manifest_digest(args.manifest),
        'patch_input': {**grid, 'fields': len(dataset.pairs), 'batch': 4},
        'self_ensemble': False, **metadata, **metrics,
    })
    print(args.method, metrics, flush=True)


if __name__ == '__main__':
    main()
