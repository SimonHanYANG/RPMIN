import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from optimize_mitos_rpmin import SEED, data
from rpmin.data import PairedGrid, PairedImages, load_rgb, read_manifest, rgb_tensor
from rpmin.engine import dump_csv, dump_json, evaluate, load_model, loader, manifest_digest, predict, save_checkpoint, summarize
from rpmin.objective import ReconstructionLoss, image_metrics, ssim_map


def luma(x):
    weights = x.new_tensor([65.481, 128.553, 24.966])[None, :, None, None] / 255
    return (x * weights).sum(1, keepdim=True) + 16 / 255


@torch.inference_mode()
def validate_luma(batches, model, device, context=0):
    model.eval()
    rng = torch.Generator().manual_seed(SEED)
    rows = []
    for lr, hr, _ in batches:
        sr = center_crop(predict(model, lr.to(device), rng), context * model.config.scale).clamp(0, 1)
        p, s = image_metrics(luma(sr), luma(hr.to(device)))
        rows.extend(dict(psnr=pv, ssim=sv) for pv, sv in zip(p.tolist(), s.tolist()))
    return summarize(rows)


def center_crop(x, edge):
    return x[..., edge:-edge, edge:-edge] if edge else x


class ContextGrid(PairedGrid):
    def __init__(self, *args, context, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context

    def __getitem__(self, index):
        pair, x, y = self.crops[index]
        side, edge, scale = self.crop_size, self.context, self.scale
        with self.open_image(pair.lr) as low, self.open_image(pair.hr) as high:
            x0, y0 = max(0, x - edge), max(0, y - edge)
            x1, y1 = min(low.width, x + side + edge), min(low.height, y + side + edge)
            lr = rgb_tensor(low.crop((x0, y0, x1, y1)), pair.lr)
            padding = (x0 - x + edge, x + side + edge - x1,
                       y0 - y + edge, y + side + edge - y1)
            lr = F.pad(lr, padding, mode='reflect')
            hr = rgb_tensor(high.crop((scale * x, scale * y, scale * (x + side), scale * (y + side))), pair.hr)
        return lr, hr, f'{pair.id}-x{x}-y{y}'


def fit_camera(pairs, scale, device):
    gram = torch.zeros((76, 76), device=device)
    rhs = torch.zeros((76, 3), device=device)
    count, energy = 0, 0.0
    for pair in pairs:
        lr = load_rgb(pair.lr)[None].to(device)
        hr = load_rgb(pair.hr)[None].to(device)
        reduced = F.avg_pool2d(hr, scale)
        if reduced.shape != lr.shape:
            raise ValueError(f'Unexpected scale for {pair.id}')
        residual = (lr - reduced)[0, :, 2:-2:8, 2:-2:8].permute(1, 2, 0).flatten(0, 1)
        patches = (reduced - 0.5).unfold(2, 5, 8).unfold(3, 5, 8)
        samples = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 75)
        design = torch.cat((samples, torch.ones_like(samples[:, :1])), dim=1)
        gram.add_(design.T @ design)
        rhs.add_(design.T @ residual)
        count += len(residual)
        energy += residual.square().sum().item()

    regularizer = torch.eye(76, device=device) * 0.00008
    regularizer[-1, -1] = 0
    coefficients = torch.linalg.solve(gram / count + regularizer, rhs / count)
    corrected_energy = energy - 2 * (coefficients * rhs).sum() + (coefficients * (gram @ coefficients)).sum()
    rmse = (max(float(corrected_energy), 0) / (3 * count)) ** 0.5
    return coefficients[:-1].T.reshape(3, 3, 5, 5), coefficients[-1], {
        'sampled_pixels': count, 'area_rmse': (energy / (3 * count)) ** 0.5, 'fitted_camera_rmse': rmse,
    }


def synthetic_lr(hr, scale, kernel, bias, noise):
    low = F.avg_pool2d(hr, scale)
    corrected = F.conv2d(F.pad(low - 0.5, (2,) * 4, mode='reflect'), kernel, bias)
    strength = torch.empty((len(hr), 1, 1, 1), device=hr.device).uniform_(0.75, 1.15)
    return (low + corrected * strength + torch.randn_like(low) * noise).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scale', type=int, choices=(2, 4), required=True)
    parser.add_argument('--luminance', action='store_true')
    parser.add_argument('--native-grid', action='store_true')
    parser.add_argument('--context', default=0, type=int, choices=(0, 16))
    parser.add_argument('--real-steps', type=int)
    parser.add_argument('--real-lr', type=float)
    args = parser.parse_args()
    if args.native_grid and not args.luminance:
        parser.error('Native-grid fine-tuning requires the luminance objective.')
    if args.context and not (args.native_grid and args.luminance):
        parser.error('Context fine-tuning requires the native grid and luminance objective.')
    if args.real_steps is not None and args.real_steps < 1:
        parser.error('Real-only training steps must be positive.')
    if args.real_lr is not None and (not math.isfinite(args.real_lr) or args.real_lr <= 0):
        parser.error('Real-only learning rate must be finite and positive.')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(207)
    random.seed(207)
    device = torch.device('cuda')
    pairs = read_manifest(args.train)
    model = load_model(args.init, device)
    if model.config.scale != args.scale or any(pair.split == 'test' for pair in pairs):
        raise ValueError('Invalid scale or test data in the training manifest.')
    manifest_sha = manifest_digest(args.train)
    init_sha = manifest_digest(args.init)
    if not args.luminance:
        kernel, bias, camera = fit_camera([pair for pair in pairs if pair.split == 'train'], args.scale, device)
        noise = min(0.0075, camera['fitted_camera_rmse'] * 0.2)
        camera.update(kernel=kernel.cpu().tolist(), bias=bias.cpu().tolist(), synthetic_noise_std=noise,
                      manifest_sha256=manifest_sha, init_sha256=init_sha)
        dump_json(args.out / 'camera.json', camera)
        print({key: camera[key] for key in ('area_rmse', 'fitted_camera_rmse', 'synthetic_noise_std')}, flush=True)

    if args.context:
        train = ContextGrid(pairs, 'train', args.scale, patch=96, stride_x=61, stride_y=67, origin=1,
                            context=args.context)
    elif args.native_grid:
        train = PairedGrid(pairs, 'train', args.scale, patch=96, stride_x=61, stride_y=67, origin=1)
    else:
        train = PairedImages(pairs, 'train', args.scale, patch=96, samples_per_pair=64, cache=True)
    batches = loader(train, device, 0, batch=4, seed=207)
    if args.context:
        val_tiles = ContextGrid(pairs, 'val', args.scale, patch=128, stride_x=128, stride_y=128, origin=1,
                                context=args.context)
        validation = loader(val_tiles, device, 0, batch=4)
    else:
        validation = data(args.train, 'val', args.scale, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, fused=True)
    criterion = ReconstructionLoss(0.2, 0.025, 'energy-weighted', 1.0, 0.12)
    shift_rng = torch.Generator().manual_seed(208)
    validation_args = dict(device=device, scale=args.scale, seed=SEED, border=0)
    def validate():
        if args.luminance:
            return validate_luma(validation, model, device, args.context)
        return summarize(evaluate(validation, model, **validation_args))

    before = validate()
    best = dict(step=0, **before)
    history = [best]
    mixed_steps, real_steps = (0, 2400 if args.native_grid else 1600) if args.luminance else (1600, 2800)
    if args.context:
        real_steps = 2000
    real_steps = args.real_steps or real_steps
    real_lr = args.real_lr or (4e-5 if args.native_grid else 6e-5)
    steps = mixed_steps + real_steps
    iterator = iter(batches)
    for step in range(1, steps + 1):
        batch = next(iterator, None)
        if batch is None:
            iterator = iter(batches)
            batch = next(iterator)
        lr, hr = (image.to(device, non_blocking=True) for image in batch[:2])
        if step <= mixed_steps and step % 2:
            lr = synthetic_lr(hr, args.scale, kernel, bias, noise)
        progress = step / mixed_steps if step <= mixed_steps else (step - mixed_steps) / real_steps
        initial_lr = 8e-5 if step <= mixed_steps else real_lr
        optimizer.param_groups[0]['lr'] = initial_lr * (0.08 + 0.92 * (1 + math.cos(math.pi * progress)) / 2)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sr = center_crop(model(lr, generator=shift_rng), args.context * args.scale)
        if args.luminance:
            y_sr, y_hr = luma(sr), luma(hr)
            difference = y_sr - y_hr
            local, _ = ssim_map(y_sr, y_hr)
            loss = (difference.square().mean() + 0.15 * difference.abs().mean()
                    + 0.12 * (1 - local.mean()) + 0.25 * F.mse_loss(sr, hr))
        else:
            loss = criterion(sr, hr)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {step}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 400 == 0 or step == steps:
            scores = dict(step=step, **validate())
            history.append(scores)
            if step >= mixed_steps and (not best['step'] or scores['psnr'] > best['psnr']):
                best = scores
                save_checkpoint(args.out / 'best.pt', {'model_config': asdict(model.config), 'model_state': model.state_dict(),
                                                       'epoch': step})
            dump_csv(args.out / 'validation.csv', history)
            print(scores, 'best', best, flush=True)
    dump_json(args.out / 'selection.json', dict(best=best, initial=before, mixed_steps=mixed_steps,
                                               real_steps=real_steps, train_sha256=manifest_sha, init_sha256=init_sha,
                                               metric='BT.601 Y' if args.luminance else 'RGB', native_grid=args.native_grid,
                                               context=args.context, real_lr=real_lr,
                                               checkpoint_sha256=manifest_digest(args.out / 'best.pt')))


if __name__ == '__main__':
    main()
