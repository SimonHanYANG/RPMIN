"""Prepare, train, and evaluate the fixed MFA-Cell experiment."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image
import torch

from rpmin.data import Pair, load_rgb, read_manifest, save_rgb, write_manifest
from rpmin.engine import TrainConfig, dump_csv, dump_json, evaluate_manifest, get_device, load_model, manifest_digest, train
from rpmin.model import ModelConfig
from rpmin.paired_reference import lift_transform
from scripts.refine_x4_inference import score


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_digest(manifest):
    digest = hashlib.sha256()
    for pair in read_manifest(manifest):
        digest.update(f"{pair.id},{pair.split},{pair.group}\n".encode())
        for path in (pair.lr, pair.hr):
            with Image.open(path) as image:
                digest.update(image.convert("RGB").tobytes())
    return digest.hexdigest()


def prepare_scale(config, dataset, run, scale, part):
    import cv2

    tag = f"x{scale}"
    out = run / "references" / tag / part
    target = out / "aligned.csv"
    expected = config["registration"]["reference_pixels_sha256"][tag][part]
    if target.is_file():
        if reference_digest(target) != expected:
            raise ValueError(f"Prepared reference pixels do not match the recorded event: {target}")
        return target

    dataset = (dataset / config["dataset"]["collection"]).resolve()
    if not dataset.is_dir() or run.is_relative_to(dataset):
        raise ValueError("Provide the dataset root and a run directory outside the source collection.")
    with (HERE / config["dataset"]["split"]).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if Counter(row["split"] for row in rows) != config["dataset"]["groups"]:
        raise ValueError("MFA-Cell field counts differ from the recorded split.")
    selected = [r for r in rows if (r["split"] == "test") == (part == "test")]
    registration = read_json(HERE / "registration" / tag / part / "registration.json")
    margin = config["registration"]["margin_lr"]
    transforms = {(p["id"], p["split"]): p["lr_to_reduced_hr"] for p in registration["pairs"]}
    if (registration["scale"] != scale or registration["margin_lr"] != margin
            or transforms.keys() != {(r["id"], r["split"]) for r in selected}):
        raise ValueError(f"Registration settings do not match the fixed split for {tag}/{part}.")
    expected_sources = read_json(HERE / config["dataset"]["source_hashes"])
    pairs = []
    for row in selected:
        identifier, split, group = row["id"], row["split"], row["group"]
        sources = [f"x{m}/{group}_x{m}.png" for m in (10, 10 * scale)]
        for path in sources:
            if manifest_digest(dataset / path) != expected_sources[path]:
                raise ValueError(f"Source image SHA-256 mismatch: {path}")
        lr, hr = (load_rgb(dataset / path) for path in sources)
        height, width = lr.shape[-2:]
        if hr.shape[-2:] != (height * scale, width * scale):
            raise ValueError(f"Unexpected image dimensions for {tag}/{identifier}.")
        matrix = lift_transform(np.array(transforms[identifier, split]), scale)
        warped = cv2.warpPerspective(hr.permute(1, 2, 0).numpy(), matrix, (width * scale, height * scale),
                                     flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        edge = margin * scale
        lr_out = out / "lr" / f"{identifier}.png"
        hr_out = out / "hr" / f"{identifier}.png"
        save_rgb(lr[:, margin:-margin, margin:-margin], lr_out)
        save_rgb(torch.from_numpy(warped[edge:-edge, edge:-edge]).permute(2, 0, 1), hr_out)
        pairs.append(Pair(identifier, split, group, lr_out, hr_out))
    temporary = target.with_suffix(".tmp")
    write_manifest(temporary, sorted(pairs, key=lambda pair: pair.id))
    if reference_digest(temporary) != expected:
        raise ValueError(f"RGB reference pixels differ for {tag}/{part}. Use the OpenCV version in experiment.json.")
    temporary.replace(target)
    return target


def pick_weights(config, run, trained):
    root = PROJECT / config["archive"]
    weights = {}
    for name, entry in config["models"].items():
        path = run / "models" / name / "best.pt" if trained else root / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = manifest_digest(path)
        if not trained and digest != entry["sha256"]:
            raise ValueError(f"Archived weight SHA-256 mismatch: {path}")
        weights[name] = (path, digest)
    return weights


def run_test(config, dataset, run, output, device, trained):
    test = config["test"]
    weights = pick_weights(config, run, trained)
    rows = []
    for tag, target in test["scales"].items():
        scale = int(tag[1:])
        manifest = prepare_scale(config, dataset, run, scale, "test")
        models = []
        for name in target["models"]:
            model = load_model(weights[name][0], device)
            entry = config["models"][name]
            if model.config != ModelConfig(**(config["model_defaults"] | entry["model"])):
                raise ValueError(f"Checkpoint model settings do not match {name}.")
            if not trained and model.checkpoint_epoch != entry["best_epoch"]:
                raise ValueError(f"Unexpected archived epoch for {name}.")
            models.append(model)
        out = output / tag
        if target["context"] == "patch":
            summary = evaluate_manifest(manifest, "test", models[0], scale, test["seeds"], out, device,
                                        workers=config["environment"]["workers"], border=test["border_hr"],
                                        images=False, maps=False, patch_grid=test["grid"], batch=target["batch"],
                                        self_ensemble=True)
        else:
            if test["grid"] != {"patch": 128, "stride_x": 32, "stride_y": 25, "origin": 1} or test["border_hr"] != 0:
                raise ValueError("The fixed ×4 scorer requires its original window grid and border.")
            summary, patches = score(read_manifest(manifest), "test", models, "whole", device, test["seeds"])
            dump_csv(out / "per_image.csv", patches)
        if any(r.get("patches", r.get("pairs")) != test["patches_per_seed"] for r in summary["runs"]):
            raise ValueError(f"Unexpected number of windows in {tag}.")
        metrics = {key: summary[key]["mean"] for key in ("psnr", "ssim")}
        okay = all(math.isclose(metrics[key], target[key], rel_tol=0, abs_tol=test["tolerance"][key]) for key in metrics)
        summary.update(context=target["context"], grid=test["grid"], self_ensemble=test["self_ensemble"],
                       manifest_sha256=manifest_digest(manifest),
                       checkpoints={name: {"file": str(weights[name][0]), "sha256": weights[name][1]}
                                    for name in target["models"]},
                       expected={key: target[key] for key in metrics}, within_archive_tolerance=okay)
        dump_json(out / "summary.json", summary)
        rows.append({"scale": tag, "context": target["context"], **metrics, "within_archive_tolerance": okay})
        print(f"{tag}: {metrics['psnr']:.8f} dB / {metrics['ssim']:.8f}", flush=True)
    dump_csv(output / "comparison.csv", rows)
    if not trained and not all(row["within_archive_tolerance"] for row in rows):
        raise RuntimeError(f"Archived scores differ from the recorded benchmark. Check {output / 'comparison.csv'}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "prepare", "train", "test"))
    parser.add_argument("--dataset", type=Path, default=PROJECT.parent / "SuperResolutionDataset")
    parser.add_argument("--run", type=Path, default=PROJECT / "runs/mfa_cell_repro")
    parser.add_argument("--out", type=Path, help="Test output. Defaults to a test directory inside --run.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trained", action="store_true", help="Test checkpoints trained in --run.")
    args = parser.parse_args()
    config = read_json(HERE / "experiment.json")
    run = args.run.resolve()
    torch.set_num_threads(config["environment"]["threads"])
    device = get_device(args.device)
    if args.mode == "verify":
        for name, (path, digest) in pick_weights(config, run, False).items():
            print(f"{name}: {path.name} {digest}")
    elif args.mode == "prepare":
        for scale in (2, 4):
            for part in ("train-val", "test"):
                print(prepare_scale(config, args.dataset, run, scale, part))
    elif args.mode == "train":
        for name, entry in config["models"].items():
            model = ModelConfig(**(config["model_defaults"] | entry["model"]))
            settings = TrainConfig(**(config["train_defaults"] | entry["train"]))
            manifest = prepare_scale(config, args.dataset, run, model.scale, "train-val")
            output = run / "models" / name
            train(manifest, output, model, settings, device, workers=config["environment"]["workers"],
                  resume=(output / "latest.pt").is_file())
    else:
        output = args.out or run / ("test-trained" if args.trained else "test-archived")
        run_test(config, args.dataset, run, output, device, args.trained)


if __name__ == "__main__":
    main()
