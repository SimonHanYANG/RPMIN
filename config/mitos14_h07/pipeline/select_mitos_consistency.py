import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from optimize_mitos_context import BASE, GRID, SCALES, SEED, Predictor, luma, phase_features
from rpmin.engine import dump_csv, dump_json, load_model, manifest_digest, predict, summarize
from rpmin.objective import image_metrics
from train_mitos_camera import center_crop


CONTEXT = Path('runs/mitos14-rpmin-context')
CAMERA = Path('runs/mitos14-rpmin-camera')
UPDATES = [(0.0, 'none')] + [(weight, band) for weight in (0.15, 0.35, 0.6)
                            for band in ('full', 'smooth', 'detail')]
OPTIONS = [(blend, update, band) for blend in (0.0, 0.25, 0.5) for update, band in UPDATES]


class Candidate:
    def __init__(self, root):
        self.predictor = Predictor()
        self.device = self.predictor.device
        self.configs, self.sources, self.trained, self.camera = ({} for _ in range(4))
        residual = CONTEXT / 'luma-residual.pt'
        self.params = torch.load(residual, map_location=self.device, weights_only=True)
        for scale in SCALES:
            key = f'x{scale}'
            prior = CONTEXT / key / 'selection.json'
            coefficients = CAMERA / key / 'camera/camera.json'
            context_model = root / key / 'context-trained'
            fitted = context_model / 'selection.json'
            model = context_model / 'best.pt'
            train = BASE / key / 'references/train-val/aligned.csv'
            frozen = json.loads(prior.read_text())
            if frozen['sources'] != self.predictor.sources or frozen['residual_sha256'] != manifest_digest(residual):
                raise ValueError('The previously selected context source changed.')
            training = json.loads(fitted.read_text())
            camera = json.loads(coefficients.read_text())
            if (training['train_sha256'] != manifest_digest(train) or training['context'] != 16
                    or training['checkpoint_sha256'] != manifest_digest(model)
                    or camera['manifest_sha256'] != manifest_digest(train)):
                raise ValueError('The camera or model was fitted on a different manifest.')
            self.configs[scale] = frozen['selected']
            self.sources[str(scale)] = {name: manifest_digest(path) for name, path in (
                ('prior', prior), ('camera', coefficients), ('training', fitted), ('model', model),
            )}
            self.trained[scale] = load_model(model, self.device).eval()
            self.camera[scale] = tuple(torch.tensor(camera[key], device=self.device) for key in ('kernel', 'bias'))
        if any(c['radius'] != 16 or c['mode'] != 'd4' for c in self.configs.values()):
            raise ValueError('This comparison expects the locked 16-pixel context and D4 settings.')

    def current(self, rgb, lr, scale):
        prediction = luma(rgb)
        strength = self.configs[scale]['strength']
        if strength:
            design = phase_features(lr, prediction, scale)
            residual = F.pixel_shuffle((design @ self.params[str(scale)]).permute(0, 3, 1, 2), scale)
            prediction = (prediction + strength * residual).clamp(16 / 255, 235 / 255)
        return prediction

    def propose(self, rgb, y, lr, scale):
        kernel, bias = self.camera[scale]
        reconstructed = (rgb + (y - luma(rgb)) * (255 / 219)).clamp(0, 1)
        low = F.avg_pool2d(reconstructed, scale)
        expected = low + F.conv2d(F.pad(low - 0.5, (2,) * 4, mode='reflect'), kernel, bias)
        error = (lr - expected).clamp(-0.08, 0.08)
        adjoint = error + F.conv_transpose2d(error, kernel, padding=2)
        update = luma(adjoint) - 16 / 255
        smooth = F.avg_pool2d(F.pad(update, (1,) * 4, mode='reflect'), 3, stride=1)
        return {key: F.interpolate(value, scale_factor=scale, mode='bicubic', align_corners=False)
                for key, value in (('full', update), ('smooth', smooth), ('detail', update - smooth))}

    @torch.inference_mode()
    def score(self, split, options):
        pad = 16
        rows = {scale: [[] for _ in options[scale]] for scale in SCALES}
        generator = torch.Generator().manual_seed(SEED)
        tuned_generator = torch.Generator()
        for lr, gt, ids, fields in self.predictor.make_batch(split, pad):
            state = generator.get_state()
            base = self.predictor.predict_batch(lr, ids, fields, pad, 'd4', generator, as_luma=False)
            for scale in SCALES:
                y = self.current(base[scale], lr, scale)
                tuned = None
                if any(blend for blend, _, _ in options[scale]):
                    size = GRID['patch'] + pad * 2
                    expanded = torch.stack([field[:, py + 2:py + 2 + size, px + 2:px + 2 + size]
                                            for field, px, py in fields])
                    tuned_generator.set_state(state)
                    tuned = center_crop(predict(self.trained[scale], expanded, tuned_generator, self_ensemble=True),
                                        pad * scale).clamp(0, 1)
                expected = luma(gt[scale])
                for blend in dict.fromkeys(blend for blend, _, _ in options[scale]):
                    prior = torch.lerp(y, luma(tuned), blend) if blend else y
                    rgb = torch.lerp(base[scale], tuned, blend) if blend else base[scale]
                    if any(w for b, w, _ in options[scale] if b == blend):
                        proposals = self.propose(rgb, prior, lr, scale)
                    for i, (b, strength, filter_name) in enumerate(options[scale]):
                        if b != blend:
                            continue
                        sr = prior if not strength else (prior + strength * proposals[filter_name]).clamp(16 / 255, 235 / 255)
                        p, s = image_metrics(sr, expected)
                        rows[scale][i].extend(dict(id=key, seed=SEED, psnr=pv, ssim=sv)
                                              for key, pv, sv in zip(ids, p.tolist(), s.tolist()))
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('select', 'test'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = Candidate(args.root)
    if args.stage == 'select':
        results = run.score('val', {scale: OPTIONS for scale in SCALES})
        for scale, tracks in results.items():
            candidates = [dict(blend=b, update=u, band=band, **summarize(rows))
                          for (b, u, band), rows in zip(OPTIONS, tracks, strict=True)]
            baseline = candidates[0]
            selected = max((x for x in candidates if x['ssim'] >= baseline['ssim']), key=lambda x: x['psnr'])
            dump_json(args.root / f'x{scale}/selection.json', dict(selected=selected, candidates=candidates,
                                                                    baseline=baseline, sources=run.sources))
            print(scale, selected, flush=True)
        return

    frozen = {scale: json.loads((args.root / f'x{scale}/selection.json').read_text()) for scale in SCALES}
    if any(record['sources'] != run.sources for record in frozen.values()):
        raise ValueError('A locked source changed.')
    choices = {scale: [tuple(report['selected'][key] for key in ('blend', 'update', 'band'))]
               for scale, report in frozen.items()}
    outcomes = run.score('test', choices)
    for scale, tracks in outcomes.items():
        out = args.root / f'x{scale}/test'
        metrics = summarize(tracks[0])
        dump_csv(out / 'per_image.csv', tracks[0])
        path = BASE / f'x{scale}/references/test/aligned.csv'
        dump_json(out / 'summary.json', dict(scale=scale, split='test', metric='BT.601 Y', border_hr=0,
                                             manifest_sha256=manifest_digest(path), patch_input=dict(GRID, batch=4, fields=1),
                                             selected=frozen[scale]['selected'], **metrics))
        print(scale, metrics, flush=True)


if __name__ == '__main__':
    main()
