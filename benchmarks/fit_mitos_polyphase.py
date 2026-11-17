import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / 'RPMIN-e79d2b3'))
from rpmin.data import load_rgb, read_manifest
from rpmin.engine import dump_json, manifest_digest


def fit(manifest, scale, device):
    width = 3 * scale**2 * 9 + 1
    covariance = torch.zeros((width, width), device=device)
    rhs = torch.zeros((width, 3), device=device)
    count, energy = 0, 0.0
    for pair in read_manifest(manifest):
        if pair.split != 'train':
            continue
        hr = load_rgb(pair.hr)[None].to(device)
        lr = load_rgb(pair.lr)[None].to(device)
        detail = F.pixel_unshuffle(hr - 0.5, scale).unfold(2, 3, 7).unfold(3, 3, 7)
        features = detail.permute(0, 2, 3, 1, 4, 5).reshape(-1, width - 1)
        features = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
        residual = (lr - F.avg_pool2d(hr, scale))[0, :, 1:-1:7, 1:-1:7].permute(1, 2, 0).flatten(0, 1)
        covariance.add_(features.T @ features)
        rhs.add_(features.T @ residual)
        count += len(residual)
        energy += residual.square().sum().item()
    if not count:
        raise ValueError('Empty training support.')
    ridge = torch.eye(width, device=device) * 0.00004
    ridge[-1, -1] = 0
    coefficients = torch.linalg.solve(covariance / count + ridge, rhs / count)
    err = energy - 2 * (coefficients * rhs).sum() + (coefficients * (covariance @ coefficients)).sum()
    return dict(kernel=coefficients[:-1].T.reshape(3, -1, 3, 3).cpu().tolist(),
                bias=coefficients[-1].cpu().tolist(), scale=scale, sampled_pixels=count,
                area_rmse=(energy / (3 * count))**0.5,
                polyphase_rmse=(max(float(err), 0) / (3 * count))**0.5,
                manifest_sha256=manifest_digest(manifest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--scale', type=int, choices=(2, 4), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = fit(args.train, args.scale, torch.device('cuda'))
    dump_json(args.out, result)
    print(args.scale, {key: result[key] for key in ('area_rmse', 'polyphase_rmse', 'sampled_pixels')}, flush=True)


if __name__ == '__main__':
    main()
