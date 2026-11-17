import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent / 'RPMIN-e79d2b3'))
from rpmin.data import Pair, write_manifest
from rpmin.engine import dump_json


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def structure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return gray - cv2.GaussianBlur(gray, (0, 0), 5)


def mosaic(parent_path, paths, scale):
    if len(paths) != scale**2:
        raise ValueError(f'{parent_path.stem} needs {scale**2} x{scale} tiles, found {len(paths)}')
    parent = rgb(parent_path)
    height, width = parent.shape[:2]
    reference = structure(parent)
    reduced_size = (width // scale, height // scale)
    dw, dh = reduced_size
    locations = [(iy, ix) for iy in range(scale) for ix in range(scale)]
    tiles = [rgb(path) for path in paths]
    quality = np.zeros((len(paths), len(locations)))
    offsets = {}
    trim, search = 24, 12
    for line, tile in enumerate(tiles):
        if tile.shape != parent.shape:
            raise ValueError(f'Unexpected tile size {paths[line]}')
        preview = cv2.resize(tile, reduced_size, interpolation=cv2.INTER_AREA)
        features = structure(preview)[trim:-trim, trim:-trim]
        for column, (iy, ix) in enumerate(locations):
            x0, y0 = round(ix * width / scale) + trim, round(iy * height / scale) + trim
            crop = reference[y0-search:y0+dh-2*trim+search, x0-search:x0+dw-2*trim+search]
            scores = cv2.matchTemplate(crop, features, cv2.TM_CCOEFF_NORMED)
            _, score, _, best = cv2.minMaxLoc(scores)
            quality[line, column] = score
            offsets[line, column] = [best[0] - search, best[1] - search]

    rows, assigned = linear_sum_assignment(-quality)
    output = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    records = []
    for line, column in zip(rows, assigned, strict=True):
        iy, ix = locations[column]
        score = quality[line, column]
        offset = offsets[line, column]
        if score < 0.80 or max(map(abs, offset)) >= search:
            raise ValueError(f'Unverified correspondence for {paths[line].name}: {score:.3f}, {offset}')
        output[iy*height:(iy+1)*height, ix*width:(ix+1)*width] = tiles[line]
        records.append(dict(frame=paths[line].name, row=iy, column=ix, correlation=float(score), offset_lr=offset))
    return output, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    parent_paths = sorted((args.frames / 'x10').glob('*.tiff'))
    if [p.stem for p in parent_paths] != ['H07_00', 'H07_01']:
        raise ValueError('Expected the two independently downloaded H07 fields.')
    for scale, directory in ((2, 'x20'), (4, 'x40')):
        dest = args.out / f'x{scale}'
        dest.mkdir(parents=True, exist_ok=True)
        pairs, audit = [], []
        for parent in parent_paths:
            paths = sorted((args.frames / directory).glob(parent.stem + '*.tiff'))
            hr, tiles = mosaic(parent, paths, scale)
            target = dest / f'{parent.stem}.png'
            Image.fromarray(hr).save(target, compress_level=1)
            pairs.append(Pair(parent.stem.lower(), 'train', 'h07', parent.resolve(), target.resolve()))
            audit.append({'id': parent.stem, 'tiles': tiles})
            print(scale, parent.stem, 'min correlation', min(row['correlation'] for row in tiles), flush=True)
        dump_json(dest / 'mosaic.json', audit)
        write_manifest(dest / 'input.csv', pairs)


if __name__ == '__main__':
    main()
