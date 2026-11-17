import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
RPMIN = ROOT.parent / 'rpmin-project' / 'RP-MIN-fresh' / 'runs'


def read_result(folder, scale, reference, records='per_image.csv'):
    summary = json.loads((folder / 'summary.json').read_text())
    if summary.get('scale', scale) != scale or summary.get('split', 'test') != 'test':
        raise ValueError(f'Unexpected scale or split in {folder}')
    with (folder / records).open(newline='') as stream:
        ids = Counter(row['id'] for row in csv.DictReader(stream))
    if (summary['manifest_sha256'] != reference[0] or ids.keys() != reference[1]
            or len(set(ids.values())) != 1):
        raise ValueError(f'Unexpected test fields or patches in {folder}')
    return [summary[key]['mean'] if isinstance(summary[key], dict) else summary[key]
            for key in ('psnr', 'ssim')]


def main():
    x2 = RUNS / 'x2-pilot'
    x4 = RUNS / 'x4-pilot'
    sources = [
        ('RP-MIN', RPMIN / 'full/test/x2/aligned/selected', RUNS / 'rpmin-latest-audit/x4-whole',
         'Width 32, patch input, D4, three seeds', 'Width 32+48, whole-field input, D4, three seeds'),
        ('NeXtSRGAN', x2 / 'interpolated', RUNS / 'x4-next-transfer/interpolated',
         'Scale-adapted RRDB initialization and PSNR/GAN interpolation', 'RRDB initialization and PSNR/GAN interpolation'),
        ('AESOP (RRDB128)', x2 / 'AESOP-transfer', x4 / 'AESOP-transfer',
         'Official checkpoint, scale adaptation and 204 L1 updates', 'Official checkpoint and 204 L1 updates'),
        ('SeD (RRDB P+SeD)', x2 / 'SeD-transfer', x4 / 'SeD-transfer',
         'Official checkpoint, scale adaptation and 204 L1 updates', 'Official checkpoint and 204 L1 updates'),
    ]
    references = []
    for scale, run in ((2, x2), (4, x4)):
        folder = run / 'rpmin'
        summary = json.loads((folder / 'summary.json').read_text())
        with (folder / 'per_image.csv').open(newline='') as stream:
            ids = {row['id'] for row in csv.DictReader(stream)}
        if len(ids) != 308 or summary['patch_input']['fields'] != 4:
            raise ValueError(f'Unexpected test grid in {folder}')
        reference = (summary['manifest_sha256'], ids)
        references.append(reference)
        print(f'RP-MIN x{scale} matched single-pass:', read_result(folder, scale, reference))

    out = RUNS / 'comparison_x2_x4.csv'
    rows = []
    for method, folder2, folder4, protocol2, protocol4 in sources:
        scores2 = read_result(folder2, 2, references[0])
        records4 = 'per_patch.csv' if method == 'RP-MIN' else 'per_image.csv'
        scores4 = read_result(folder4, 4, references[1], records=records4)
        rows.append([method, *scores2, *scores4, protocol2, protocol4])
    with out.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'x2_psnr_db', 'x2_ssim', 'x4_psnr_db', 'x4_ssim', 'x2_protocol', 'x4_protocol'])
        writer.writerows(rows)
    print(out)


if __name__ == '__main__':
    main()
