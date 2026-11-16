from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_rgb, read_manifest, save_rgb, write_manifest
from .engine import dump_json


class RegistrationFailure(ValueError):
    pass


def estimate_homography(src, dst, threshold, seed):
    import cv2

    if len(src) < 20:
        raise RegistrationFailure("Fewer than 20 matched keypoints.")
    perm = np.random.default_rng(seed).permutation(len(src))
    fit, holdout = np.split(perm, [len(src) * 3 // 4])
    cv2.setRNGSeed(seed % (2**31))
    matrix, inliers = cv2.findHomography(src[fit], dst[fit], cv2.RANSAC, threshold)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        raise RegistrationFailure("Homography estimation failed.")
    count = int(inliers.sum())
    ratio = count / len(fit)
    warped = cv2.perspectiveTransform(src[holdout][None], matrix)[0]
    before = np.linalg.norm(src[holdout] - dst[holdout], axis=-1)
    after = np.linalg.norm(warped - dst[holdout], axis=-1)
    if not np.isfinite(after).all():
        raise RegistrationFailure("Invalid holdout projection.")
    return matrix, {
        "matches": len(src), "fit_inliers": count, "fit_inlier_ratio": ratio, "holdout_matches": len(holdout),
        "holdout_before_median_lr_px": float(np.median(before)),
        "holdout_after_median_lr_px": float(np.median(after)),
        "holdout_after_p95_lr_px": float(np.quantile(after, 0.95)),
    }


def valid_crop(mask, size):
    import cv2

    height, width = mask.shape
    if size > min(height, width):
        raise RegistrationFailure("Requested LR crop exceeds the image dimensions.")
    invalid = cv2.integral((mask == 0).astype(np.uint8))
    counts = invalid[size:, size:] - invalid[size:, :-size] - invalid[:-size, size:] + invalid[:-size, :-size]
    coordinates = np.argwhere(counts == 0)
    if not len(coordinates):
        raise RegistrationFailure("Insufficient valid overlap for requested LR crop.")
    center = np.array(((height - size) / 2, (width - size) / 2))
    return coordinates[np.argmin(np.sum((coordinates - center)**2, axis=1))]


def overlap_rectangle(mask, min_size):
    """Largest fully valid rectangle meeting the minimum LR crop dimensions."""
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best, area = None, 0
    for y, valid in enumerate(mask):
        heights = np.where(valid, heights + 1, 0)
        stack = []
        for x, height in enumerate((*heights.tolist(), 0)):
            start = x
            while stack and height < stack[-1][1]:
                start, previous = stack.pop()
                width = x - start
                if min(width, previous) >= min_size and width * previous > area:
                    area = width * previous
                    best = (start, y - previous + 1, width, previous)
            if height and (not stack or height > stack[-1][1]):
                stack.append((start, height))
    if best is None:
        raise RegistrationFailure("Insufficient valid overlap for the minimum LR crop.")
    return best


@torch.inference_mode()
def register_training(manifest, out, scale, patch, device, threshold, min_ratio, max_holdout, seed, keep_overlap=False):
    import cv2
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    pairs = read_manifest(manifest)
    if not any(pair.split == "train" for pair in pairs):
        raise ValueError("Manifest has no training pairs.")
    if patch < 4 or threshold <= 0 or not 0 <= min_ratio <= 1 or max_holdout <= 0:
        raise ValueError("Invalid crop size or registration threshold.")
    out = Path(out)
    if (out / "registered.csv").exists():
        raise ValueError(f"Output already exists: {out / 'registered.csv'}")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    finished, reports = [], []
    for pair in pairs:
        if pair.split != "train":
            finished.append(pair)
            continue
        stats = {"id": pair.id, "group": pair.group}
        try:
            lr, hr = load_rgb(pair.lr), load_rgb(pair.hr)
            height, width = lr.shape[-2:]
            if hr.shape[-2:] != (height * scale, width * scale):
                raise RegistrationFailure(f"HR shape must be exactly {scale}× LR with approximately the same field of view.")
            reference = F.interpolate(hr[None], size=(height, width), mode="area")[0]
            f0 = extractor.extract(lr.to(device), resize=None)
            f1 = extractor.extract(reference.to(device), resize=None)
            matches = rbd(matcher({"image0": f0, "image1": f1}))["matches"]
            src = rbd(f0)["keypoints"][matches[:, 0]].cpu().numpy()
            dst = rbd(f1)["keypoints"][matches[:, 1]].cpu().numpy()
            matrix, errors = estimate_homography(src, dst, threshold, seed)
            stats.update(errors)
            if errors["fit_inliers"] < 8 or errors["fit_inlier_ratio"] < min_ratio:
                raise RegistrationFailure("Insufficient RANSAC support.")
            if errors["holdout_after_median_lr_px"] > max_holdout:
                raise RegistrationFailure("Holdout median reprojection error exceeds the threshold.")
            warped = cv2.warpPerspective(lr.permute(1, 2, 0).numpy(), matrix, (width, height))
            support = cv2.warpPerspective(np.ones((height, width), np.float32), matrix, (width, height))
            valid = support >= (1 - 1e-6)
            stats["valid_fraction"] = float(valid.mean())
            if keep_overlap:
                x, y, crop_w, crop_h = overlap_rectangle(valid, patch)
            else:
                y, x = map(int, valid_crop(valid, patch))
                crop_w = crop_h = patch
            lr = torch.from_numpy(warped[y:y + crop_h, x:x + crop_w]).permute(2, 0, 1)
            hr = hr[:, y * scale:(y + crop_h) * scale, x * scale:(x + crop_w) * scale]
        except (RegistrationFailure, cv2.error) as error:
            reports.append({**stats, "status": "rejected", "reason": str(error)})
            continue
        lr_out, hr_out = out / "lr" / f"{pair.id}.png", out / "hr" / f"{pair.id}.png"
        save_rgb(lr, lr_out)
        save_rgb(hr, hr_out)
        finished.append(replace(pair, lr=lr_out.resolve(), hr=hr_out.resolve()))
        reports.append({**stats, "status": "accepted", "crop_lr_xywh": [x, y, crop_w, crop_h]})
    accepted = [row for row in reports if row["status"] == "accepted"]
    report = {
        "scale": scale, "patch_lr": patch, "keep_overlap": keep_overlap, "reprojection_threshold_lr_px": threshold,
        "min_inlier_ratio": min_ratio, "max_holdout_median_lr_px": max_holdout,
        "seed": seed, "accepted": len(accepted), "rejected": len(reports) - len(accepted),
        "pairs": reports,
    }
    if accepted:
        report["accepted_pair_holdout_median_lr_px"] = float(np.median(
            [row["holdout_after_median_lr_px"] for row in accepted]
        ))
    dump_json(out / "registration.json", report)
    if not accepted:
        raise ValueError(f"No accepted training pairs. Inspect {out / 'registration.json'}.")
    write_manifest(out / "registered.csv", finished)
    return {key: value for key, value in report.items() if key != "pairs"}
