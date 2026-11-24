"""Evaluate the archived RP-MIN checkpoints on the fixed MFA-Cell test split."""

import argparse
import hashlib
from itertools import groupby
import json
from pathlib import Path
from statistics import fmean, stdev
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpmin.data import Pair, PairedGrid, PairedImages, load_rgb, read_manifest, save_rgb, write_manifest
from rpmin.engine import dump_csv, dump_json, evaluate, get_device, load_model, loader, manifest_digest, predict
from rpmin.model import ModelConfig
from rpmin.objective import grid_metrics
from rpmin.paired_reference import lift_transform


def prepare_references(config, scale, dataset, manifest):
    import cv2

    collection = (dataset / config["collection"]).resolve()
    if not collection.is_dir():
        raise FileNotFoundError(collection)
    if manifest.resolve().is_relative_to(collection):
        raise ValueError("Write aligned references outside the source collection.")
    margin = config["margin_lr"]
    pairs = []
    for field in config["fields"]:
        identifier, group = field["id"], field["group"]
        magnifications = (10, 10 * scale)
        paths = [collection / f"x{mag}/{group}_x{mag}.png" for mag in magnifications]
        for path, mag in zip(paths, magnifications, strict=True):
            if manifest_digest(path) != field["source_sha256"][f"x{mag}"]:
                raise ValueError(f"Unexpected source image: {path}")
        lr, hr = (load_rgb(path) for path in paths)
        height, width = lr.shape[-2:]
        if hr.shape[-2:] != (height * scale, width * scale):
            raise ValueError(f"Unexpected image dimensions for {identifier}.")
        matrix = lift_transform(np.array(field["lr_to_reduced_hr"][f"x{scale}"]), scale)
        warped = cv2.warpPerspective(
            hr.permute(1, 2, 0).numpy(), matrix, (width * scale, height * scale),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        )
        edge = margin * scale
        lr_path = manifest.parent / "lr" / f"{identifier}.png"
        hr_path = manifest.parent / "hr" / f"{identifier}.png"
        save_rgb(lr[:, margin:-margin, margin:-margin], lr_path)
        save_rgb(torch.from_numpy(warped[edge:-edge, edge:-edge]).permute(2, 0, 1), hr_path)
        pairs.append(Pair(identifier, "test", group, lr_path, hr_path))
    temporary = manifest.with_suffix(".tmp")
    write_manifest(temporary, pairs)
    check_references(config, scale, temporary)
    temporary.replace(manifest)


def check_references(config, scale, manifest):
    pairs = read_manifest(manifest)
    expected = sorted((field["id"], "test", field["group"]) for field in config["fields"])
    if [(pair.id, pair.split, pair.group) for pair in pairs] != expected:
        raise ValueError(f"Unexpected MFA-Cell test fields in {manifest}.")
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(f"{pair.id},{pair.split},{pair.group}\n".encode())
        for path in (pair.lr, pair.hr):
            with Image.open(path) as image:
                digest.update(image.convert("RGB").tobytes())
    if digest.hexdigest() != config["scales"][f"x{scale}"]["reference_pixels_sha256"]:
        raise ValueError(f"Aligned RGB pixels differ in {manifest}. Use the documented OpenCV version.")
    return pairs


def save_triplet(folder, identifier, lr, sr, hr):
    for name, image in (("lr", lr), ("sr", sr), ("hr", hr)):
        save_rgb(image, folder / identifier / f"{name}.png")


def score_patches(pairs, model, config, device, example_dir):
    grid = PairedGrid(pairs, "test", 2, **config["grid"])
    data = loader(grid, device, workers=0, batch=config["scales"]["x2"]["batch"])
    example_ids = set()
    for identifier, crops in groupby(grid.crops, key=lambda crop: crop[0].id):
        crops = list(crops)
        _, x, y = crops[len(crops) // 2]
        example_ids.add(f"{identifier}-x{x}-y{y}")

    def save_example(identifier, lr, sr, hr):
        if identifier in example_ids:
            save_triplet(example_dir, identifier, lr, sr, hr)

    for seed in config["seeds"]:
        yield evaluate(
            data, model, device=device, seed=seed, border=config["border_hr"], self_ensemble=True,
            save_prediction=save_example if seed == config["seeds"][0] else None,
        )


@torch.inference_mode()
def score_fields(pairs, model, config, device, example_dir):
    scale = 4
    data = loader(PairedImages(pairs, "test", scale), device, workers=0)
    for seed in config["seeds"]:
        generator = torch.Generator().manual_seed(seed)
        rows = []
        for lr, hr, (identifier,) in data:
            lr, hr = lr.to(device), hr.to(device)
            sr = predict(model, lr, generator, self_ensemble=True)
            positions, psnr, ssim = grid_metrics(sr, hr, scale, **config["grid"], border=config["border_hr"])
            for (x, y), p, s in zip(positions, psnr.tolist(), ssim.tolist(), strict=True):
                rows.append({"id": f"{identifier}-x{x}-y{y}", "seed": seed, "psnr": p, "ssim": s})
            if seed == config["seeds"][0]:
                x, y = positions[len(positions) // 2]
                patch = config["grid"]["patch"]
                lr_crop = lr[0, :, y:y + patch, x:x + patch]
                y_hr = slice(y * scale, (y + patch) * scale)
                x_hr = slice(x * scale, (x + patch) * scale)
                save_triplet(example_dir, f"{identifier}-x{x}-y{y}", lr_crop, sr[0, :, y_hr, x_hr], hr[0, :, y_hr, x_hr])
                save_rgb(sr[0], example_dir / "full_fields" / f"{identifier}.png")
        yield rows


def load_checkpoint(config, tag, weights, device):
    entry = config["scales"][tag]["checkpoint"]
    path = weights / entry["file"]
    if manifest_digest(path) != entry["sha256"]:
        raise ValueError(f"Unexpected checkpoint SHA-256: {path}")
    model = load_model(path, device)
    expected = ModelConfig(**(config["model_defaults"] | entry["model"]))
    if model.config != expected or model.checkpoint_epoch != entry["epoch"]:
        raise ValueError(f"Unexpected model configuration or epoch: {path}")
    return model


def write_metrics(path, runs):
    per_seed = [
        {"seed": rows[0]["seed"], "patches": len(rows), **{
            key: fmean(row[key] for row in rows) for key in ("psnr", "ssim")
        }}
        for rows in runs
    ]
    summary = {"runs": per_seed}
    for key in ("psnr", "ssim"):
        values = [run[key] for run in per_seed]
        summary[key] = {"mean": fmean(values), "std": stdev(values)}
    dump_csv(path / "per_patch.csv", [row for run in runs for row in run])
    dump_json(path / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale", choices=("x2", "x4", "both"), default="both")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/mfa_cell")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mfa_cell")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path, help="SuperResolutionDataset root for regenerating aligned references.")
    source.add_argument("--prepared", type=Path, help="Aligned reference root containing x2/test and x4/test.")
    parser.add_argument("--metrics-out", type=Path, help="Optional directory for computed per-patch and aggregate metrics.")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("mfa_cell_test.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    device = get_device(args.device)
    references = args.prepared or args.out / "references"
    tags = ("x2", "x4") if args.scale == "both" else (args.scale,)
    for tag in tags:
        scale = int(tag[1:])
        manifest = references / tag / "test/aligned.csv"
        if not manifest.is_file():
            if args.dataset is None:
                parser.error(f"Missing {manifest}. Provide --prepared or --dataset.")
            prepare_references(config, scale, args.dataset, manifest)
        pairs = check_references(config, scale, manifest)
        model = load_checkpoint(config, tag, args.weights, device)
        examples = args.out / "predictions" / tag
        if scale == 2:
            runs = list(score_patches(pairs, model, config, device, examples))
        else:
            runs = list(score_fields(pairs, model, config, device, examples))
        if any(len(rows) != config["patches_per_seed"] for rows in runs):
            raise ValueError(f"Unexpected number of scored {tag} windows.")
        if args.metrics_out is not None:
            write_metrics(args.metrics_out / tag, runs)
        print(f"{tag}: evaluated {len(pairs)} test fields across {len(runs)} seeds. Images: {examples}", flush=True)


if __name__ == "__main__":
    main()
