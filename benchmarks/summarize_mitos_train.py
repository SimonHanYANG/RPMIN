import argparse
import csv
import json
from pathlib import Path


METHODS = {
    'RP-MIN': 'rpmin',
    'AESOP (RRDB128)': 'AESOP',
    'SeD (RRDB P+SeD)': 'SeD',
    'NeXtSRGAN': 'next/interpolated',
    'Bicubic': 'bicubic',
}


def scores(folder, scale, rpmin_override=None):
    reference = None
    results = {}
    for method, path in METHODS.items():
        path = rpmin_override if method == 'RP-MIN' and rpmin_override is not None else folder / path
        summary = json.loads((path / 'summary.json').read_text())
        grid = {key: value for key, value in summary['patch_input'].items() if value is not None}
        with (path / 'per_image.csv').open(newline='') as stream:
            ids = tuple(row['id'] for row in csv.DictReader(stream))
        signature = (summary['manifest_sha256'], grid, ids)
        if reference is None:
            reference = signature
        if (reference != signature or summary['scale'] != scale or summary['split'] != 'test'
                or summary['border_hr'] != 0 or grid['fields'] != 1 or len(set(ids)) != 132):
            raise ValueError(f'{path}: expected the same 132 test patches from a single held-out field')
        results[method] = [summary[key]['mean'] if isinstance(summary[key], dict) else summary[key]
                           for key in ('psnr', 'ssim')]
    return results, reference[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--rpmin-opt', type=Path)
    args = parser.parse_args()
    x2, x4 = (args.run / f'x{scale}' for scale in (2, 4))
    if (x2 / 'split.csv').read_bytes() != (x4 / 'split.csv').read_bytes():
        raise ValueError('Both scales must use the same held-out field and training split.')
    optimized = args.rpmin_opt
    scores2, ids2 = scores(x2, 2, optimized / 'x2/locked/test' if optimized else None)
    scores4, ids4 = scores(x4, 4, optimized / 'x4/locked/test' if optimized else None)
    if ids2 != ids4:
        raise ValueError('Both scales must score the same LR grid.')
    out = (optimized or args.run) / 'comparison.csv'
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'x2_psnr_db', 'x2_ssim', 'x4_psnr_db', 'x4_ssim', 'test_fields', 'patches_per_scale'])
        writer.writerows([method, *scores2[method], *scores4[method], 1, len(ids2)] for method in METHODS)
    print(out)


if __name__ == '__main__':
    main()
