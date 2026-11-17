import argparse
import csv
import json
from pathlib import Path


METHODS = [
    ('rpmin', 'RP-MIN', 'Width-32 pixel-refine checkpoint, independent patches, no ensemble, seed 101'),
    ('bicubic', 'Bicubic', 'Bicubic interpolation'),
    ('aesop', 'AESOP RRDB128', 'Official zero-shot checkpoint'),
    ('sed', 'RRDB P+SeD', 'Official zero-shot checkpoint'),
    ('AESOP-transfer', 'AESOP RRDB128', 'Official checkpoint + 204 in-domain L1 updates'),
    ('SeD-transfer', 'RRDB P+SeD', 'Official checkpoint + 204 in-domain L1 updates'),
    ('next', 'NeXtSRGAN', 'In-domain from scratch, 800 L1 and 200 GAN updates'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--next-transfer', type=Path)
    args = parser.parse_args()
    results = []
    reference = None
    methods = [(args.run / folder, method, protocol) for folder, method, protocol in METHODS]
    if args.next_transfer:
        methods.extend([
            (args.next_transfer / 'next', 'NeXtSRGAN',
             'ESRGAN PSNR initialization, 204 in-domain L1 and 100 GAN updates'),
            (args.next_transfer / 'interpolated', 'NeXtSRGAN',
             'ESRGAN PSNR initialization, validation-selected PSNR/GAN weight interpolation'),
        ])
    for target, method, protocol in methods:
        summary_path = target / 'summary.json'
        if not summary_path.exists() and args.allow_incomplete:
            continue
        summary = json.loads(summary_path.read_text())
        with (target / 'per_image.csv').open(newline='') as stream:
            ids = tuple(row['id'] for row in csv.DictReader(stream))
        grid = {key: value for key, value in summary['patch_input'].items() if value is not None}
        spec = (summary['manifest_sha256'], grid, ids)
        if reference is None:
            reference = spec
        if reference != spec or summary['split'] != 'test' or summary['border_hr'] or len(set(ids)) != 308:
            raise ValueError(f'{target} did not use the fixed evaluation protocol')
        scores = {metric: (summary[metric]['mean'] if isinstance(summary[metric], dict) else summary[metric])
                  for metric in ('psnr', 'ssim')}
        results.append({'method': method, 'protocol': protocol, **scores, 'patches': len(ids),
                        'fields': summary['patch_input']['fields']})

    output = args.run / 'comparison.csv'
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(output)


if __name__ == '__main__':
    main()
