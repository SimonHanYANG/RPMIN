"""Build a fixed, LR-coordinate benchmark from real paired fields."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_rgb, read_manifest, save_rgb, write_manifest
from .engine import dump_json
from .registration import RegistrationFailure, estimate_homography


def lift_transform(matrix, scale):
    """Convert an LR pixel-center transform to the corresponding HR grid."""
    offset = (scale - 1) / 2
    change = np.array([[scale, 0, offset], [0, scale, offset], [0, 0, 1]])
    return change @ matrix @ np.linalg.inv(change)


def _structure(image):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray - cv2.GaussianBlur(gray, (0, 0), 6)


def _alignment_score(matrix, src, dst, margin):
    import cv2

    height, width = src.shape
    corners = np.float64([[margin, margin], [width - 1 - margin, margin],
                          [margin, height - 1 - margin], [width - 1 - margin, height - 1 - margin]])
    denominator = corners @ matrix[2, :2] + matrix[2, 2]
    if not (np.all(denominator > 1e-8) or np.all(denominator < -1e-8)):
        return None
    warped = cv2.perspectiveTransform(corners[None], matrix)[0]
    # Reserve an interpolation halo around the common region.
    if (not np.isfinite(warped).all() or
            np.any(np.abs(warped - corners) > margin - 2)):
        return None
    aligned = cv2.warpPerspective(dst, matrix, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    a, b = (image[margin:-margin, margin:-margin].ravel() for image in (src, aligned))
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


@torch.inference_mode()
def prepare_references(manifest, out, scale, device, margin, refine, splits, seed=42):
    import cv2
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    out = Path(out)
    if (out / 'aligned.csv').exists() or margin < 3:
        raise ValueError('Use a fresh output directory and margin ≥ 3 LR pixels.')
    cv2.setNumThreads(2)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    aligned, raw, reports = [], [], []
    for pair in read_manifest(manifest):
        if pair.split not in splits:
            continue
        lr, hr = (load_rgb(path) for path in (pair.lr, pair.hr))
        height, width = lr.shape[-2:]
        if hr.shape[-2:] != (height * scale, width * scale) or min(height, width) - 2 * margin < 128:
            raise ValueError(f'{pair.id}: invalid scale or insufficient common LR support.')
        reduced = F.interpolate(hr[None], size=(height, width), mode='area')[0]
        f0, f1 = (extractor.extract(x.to(device), resize=None) for x in (lr, reduced))
        matches = rbd(matcher({'image0': f0, 'image1': f1}))['matches']
        src = rbd(f0)['keypoints'][matches[:, 0]].cpu().numpy()
        dst = rbd(f1)['keypoints'][matches[:, 1]].cpu().numpy()
        low, high = (_structure(image.permute(1, 2, 0).numpy()) for image in (lr, reduced))
        identity = np.eye(3)
        best = identity
        baseline = best_score = _alignment_score(identity, low, high, margin)
        report = {'id': pair.id, 'split': pair.split, 'identity_correlation': baseline, 'method': 'identity'}
        try:
            matrix, stats = estimate_homography(src, dst, 2.0, seed)
            report.update(stats)
            # Homography is the paper's reference registration candidate. Keep identity
            # if a fit is unsupported or disagrees with the image structure.
            if stats['fit_inlier_ratio'] >= 0.45 and stats['holdout_after_median_lr_px'] <= 2.0:
                score = _alignment_score(matrix, low, high, margin)
                report['lightglue_correlation'] = score
                if score is not None and score > best_score:
                    best, best_score = matrix, score
                    report['method'] = 'lightglue'
                if refine:
                    # Fit subpixel affine motion from a LightGlue initialization.
                    affine, _ = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=2.0)
                    if affine is not None:
                        _, refined = cv2.findTransformECC(
                            low, high, affine.astype(np.float32), cv2.MOTION_AFFINE,
                            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6), None, 3,
                        )
                        refined = np.vstack((refined, (0, 0, 1))).astype(np.float64)
                        score = _alignment_score(refined, low, high, margin)
                        report['refined_correlation'] = score
                        if score is not None and score > best_score:
                            best, best_score = refined, score
                            report['method'] = 'lightglue-ecc'
        except (RegistrationFailure, cv2.error) as error:
            report['estimation_note'] = str(error)

        report.update({'correlation': best_score, 'lr_to_reduced_hr': best.tolist()})
        reports.append(report)
        edge = margin * scale
        lr_path = out / 'lr' / f'{pair.id}.png'
        hr_path = out / 'hr' / f'{pair.id}.png'
        raw_path = out / 'hr_raw' / f'{pair.id}.png'
        save_rgb(lr[:, margin:-margin, margin:-margin], lr_path)
        save_rgb(hr[:, edge:-edge, edge:-edge], raw_path)
        warped = cv2.warpPerspective(
            hr.permute(1, 2, 0).numpy(), lift_transform(best, scale), (width * scale, height * scale),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        )
        save_rgb(torch.from_numpy(warped[edge:-edge, edge:-edge]).permute(2, 0, 1), hr_path)
        aligned.append(replace(pair, lr=lr_path.resolve(), hr=hr_path.resolve()))
        raw.append(replace(pair, lr=lr_path.resolve(), hr=raw_path.resolve()))

    if not aligned:
        raise ValueError('No fields selected for registration.')
    dump_json(out / 'registration.json', {'scale': scale, 'margin_lr': margin, 'refine': refine, 'pairs': reports})
    write_manifest(out / 'raw.csv', raw)
    write_manifest(out / 'aligned.csv', aligned)
