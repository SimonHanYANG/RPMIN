"""Assemble the Figure 5 PNG from the preserved original and server ROIs."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'server_results'
OLD = ('HR overview', 'Ground truth', 'RP-MIN', 'Bicubic', 'PathSRGAN', 'SwinIR', 'SR3', 'RepRFN', 'SeemoRe')
NEW = ('NeXtSRGAN', 'AESOP', 'SeD')
LEFTS = (1, 152, 303, 454, 605, 757, 909, 1060, 1211)
TOPS = (0, 151, 303, 454, 605, 755)
TILE, GAP = 188, 5
X0, Y0 = 20, 60
SECTION, GROUP_GAP = 44, 17
TEXT = '#171c23'
MUTED = '#414b57'


def font(size, bold=False):
    names = ['arialbd.ttf', 'DejaVuSans-Bold.ttf'] if bold else ['arial.ttf', 'DejaVuSans.ttf']
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def x_for(index):
    return X0 + index * (TILE + GAP) + (11 if index >= len(OLD) else 0)


def y_for(index):
    section = index // 3
    return Y0 + section * (SECTION + 3 * (TILE + GAP) + GROUP_GAP) + SECTION + (index % 3) * (TILE + GAP)


def on_white(rgba):
    base = Image.new('RGB', rgba.size, 'white')
    base.paste(rgba, mask=rgba.getchannel('A'))
    return base


def main():
    original = Image.open(ROOT / 'Figure5_original.png').convert('RGBA')
    rows = json.loads((ASSETS / 'manifest.json').read_text(encoding='utf-8'))['rows']
    assert [item['row'] for item in rows] == list(range(1, 7))
    width, height = x_for(len(OLD + NEW) - 1) + TILE + 20, y_for(5) + TILE + 20
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    for idx, label in enumerate(OLD + NEW):
        draw.text((x_for(idx) + TILE / 2, 33), label, font=font(22, bold=idx>=len(OLD)),
                  fill=TEXT, anchor='mm')

    for index, row in enumerate(rows):
        row_top = y_for(index)
        if index % 3 == 0:
            y = row_top - SECTION + 17
            draw.text((X0, y), row['dataset'], font=font(25, bold=True), fill=TEXT, anchor='lm')
            draw.text((width - 20, y), '10× to 40×', font=font(20), fill=MUTED, anchor='rm')
            draw.line((X0, y + 16, width - 20, y + 16), fill='#cfd5dc', width=2)

        splitx = x_for(len(OLD)) - 8
        draw.line((splitx, row_top, splitx, row_top + TILE - 1), fill='#dee2e8', width=2)

        dest = ROOT / 'original_panels' / row['folder']
        dest.mkdir(parents=True, exist_ok=True)
        for j, label in enumerate(OLD):
            x, y = LEFTS[j], TOPS[index]
            panel = original.crop((x, y, x + 147, y + 147))
            panel.save(dest / (label.replace(' ', '_') + '.png'))
            image.paste(on_white(panel).resize((TILE, TILE), Image.Resampling.BICUBIC), (x_for(j), row_top))
        for j, label in enumerate(NEW, len(OLD)):
            with Image.open(ASSETS / row['folder'] / f'{label}_roi.png') as panel:
                image.paste(panel.resize((TILE, TILE), Image.Resampling.BICUBIC), (x_for(j), row_top))

    image.save(ROOT / 'Figure5_extended.png', dpi=(300, 300))


if __name__ == '__main__':
    main()
