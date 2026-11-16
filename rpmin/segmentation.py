from pathlib import Path
from statistics import fmean

import numpy as np
from PIL import Image

from .data import read_manifest
from .engine import dump_csv, dump_json


def read_instances(folder, identifier):
    from scipy.io import loadmat

    path = Path(folder) / f"{identifier}.npy"
    if path.exists():
        result = np.load(path, allow_pickle=False)
    else:
        path = path.with_suffix(".mat")
        if not path.exists():
            raise FileNotFoundError(f"Missing instance map {identifier}.npy or {identifier}.mat in {folder}.")
        result = loadmat(path)["inst_map"]
    if result.ndim != 2 or not np.issubdtype(result.dtype, np.integer) or (result < 0).any():
        raise ValueError(f"{path}: expected 2D nonnegative integer instance labels with 0 for background.")
    return result


def instance_metrics(truth, prediction):
    from scipy.optimize import linear_sum_assignment

    if truth.shape != prediction.shape:
        raise ValueError(f"Reference and prediction must share the HR grid. Got {truth.shape}, {prediction.shape}.")
    truth_fg, pred_fg = truth > 0, prediction > 0
    both = truth_fg & pred_fg
    foreground = np.count_nonzero(truth_fg) + np.count_nonzero(pred_fg)
    dice = 2 * np.count_nonzero(both) / foreground if foreground else 1.0
    gt_ids, gt_inverse, gt_area = np.unique(truth[truth_fg], return_inverse=True, return_counts=True)
    sr_ids, sr_inverse, sr_area = np.unique(prediction[pred_fg], return_inverse=True, return_counts=True)
    n_gt, n_sr = len(gt_ids), len(sr_ids)
    count_error = abs(n_gt - n_sr)
    if not n_gt or not n_sr:
        empty_score = float(n_gt == n_sr)
        return {"dice": dice, "aji": empty_score, "aji_plus": empty_score, "pq": empty_score,
                "reference_count": n_gt, "predicted_count": n_sr, "count_abs_error": count_error}

    indices = gt_inverse[both[truth_fg]] * n_sr + sr_inverse[both[pred_fg]]
    intersection = np.bincount(indices, minlength=n_gt * n_sr).reshape(n_gt, n_sr)
    union = gt_area[:, None] + sr_area[None, :] - intersection
    iou = intersection / union

    def aggregated_jaccard(g, p):
        keep = intersection[g, p] > 0
        g, p = g[keep], p[keep]
        missing_g = ~np.isin(np.arange(n_gt), g)
        missing_p = ~np.isin(np.arange(n_sr), p)
        denominator = union[g, p].sum() + gt_area[missing_g].sum() + sr_area[missing_p].sum()
        return float(intersection[g, p].sum() / denominator)

    aji = aggregated_jaccard(np.arange(n_gt), iou.argmax(1))
    aji_plus = aggregated_jaccard(*linear_sum_assignment(iou, maximize=True))
    # At IoU > 0.5, disjoint instance labels guarantee unique pairs.
    pq = float(iou[iou > 0.5].sum() / (0.5 * (n_gt + n_sr)))
    return {"dice": dice, "aji": aji, "aji_plus": aji_plus, "pq": pq,
            "reference_count": n_gt, "predicted_count": n_sr, "count_abs_error": count_error}


def evaluate_instances(manifest, split, reference, methods, kind, out):
    samples = [p for p in read_manifest(manifest) if p.split == split]
    if not samples or not methods:
        raise ValueError("At least one sample and method are required.")
    rows_by_method = {name: [] for name in methods}
    for pair in samples:
        truth = read_instances(reference, pair.id)
        with Image.open(pair.hr) as image:
            if truth.shape != (image.height, image.width):
                raise ValueError(f"{pair.id}: the reference instance map must have the same shape as the HR image.")
        for name, folder in methods.items():
            metrics = instance_metrics(truth, read_instances(folder, pair.id))
            rows_by_method[name].append({"method": name, "id": pair.id, **metrics})
    summaries = [
        {"method": name, "pairs": len(rows), **{
            metric: fmean(row[metric] for row in rows)
            for metric in ("dice", "aji", "aji_plus", "pq", "count_abs_error")
        }}
        for name, rows in rows_by_method.items()
    ]
    out = Path(out)
    dump_csv(out / "per_image.csv", [row for rows in rows_by_method.values() for row in rows])
    summary = {"reference_kind": kind, "split": split, "methods": summaries}
    dump_json(out / "summary.json", summary)
    return summary
