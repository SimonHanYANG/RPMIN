import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from .data import SPLITS, load_rgb, manifest_summary, read_manifest, save_rgb, split_manifest
from .engine import TrainConfig, evaluate_manifest, get_device, load_model, predict, train
from .model import ModelConfig


def add_model_options(parser):
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--radius", type=float, help="Fixed subpixel radius. Default draws a random radius.")
    parser.add_argument("--detail-head", action="store_true", help="Add a phase-specific reconstruction residual.")


def add_train_options(parser):
    defaults = TrainConfig()
    for field in fields(TrainConfig):
        value = getattr(defaults, field.name)
        option = "--" + field.name.replace("_", "-")
        if field.name == "phase":
            parser.add_argument(option, choices=("angle", "circular", "energy-weighted"), default=value)
        elif isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(option, type=type(value), default=value)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def config_from_args(kind, args):
    return kind(**{field.name: getattr(args, field.name, field.default) for field in fields(kind)})


def add_manifest_options(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)


def add_scoring_options(parser):
    add_manifest_options(parser)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--border", type=int, default=0, help="HR pixels removed on each side for metrics.")


def add_patch_grid_options(parser, patch_help):
    parser.add_argument("--lr-patch", type=int, help=patch_help)
    parser.add_argument("--stride-x", type=int, help="Horizontal LR step. Defaults to the patch size.")
    parser.add_argument("--stride-y", type=int, help="Vertical LR step. Defaults to the patch size.")
    parser.add_argument("--origin", type=int, default=0, help="LR grid origin in both axes.")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="RP-MIN microscopy single-image super-resolution.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--threads", type=int, help="Override number of PyTorch CPU threads.")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="Check split leakage by group and paths; print counts.")
    audit.add_argument("--manifest", type=Path, required=True)
    split = commands.add_parser("split", help="Assign whole specimen or source groups to 80/10/10 splits.")
    add_manifest_options(split)
    split.add_argument("--seed", type=int, default=42)

    prepare = commands.add_parser("prepare-dataset", help="Index x10/x20/x40 collections without changing source images.")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--groups", type=Path, help="CSV with dataset, field_id, group columns.")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--demo-tile", type=int, help="Optional LR tile size for a one-field spatial demo.")
    prepare.add_argument("--demo-gap", type=int, default=8, help="LR guard band between demo train/val/test regions.")

    training = commands.add_parser("train", help="Train one ×2 or ×4 model and save checkpoints.")
    add_manifest_options(training)
    add_model_options(training)
    add_train_options(training)
    training.add_argument("--init-checkpoint", type=Path, help="Initialize a new run from model weights.")

    evaluation = commands.add_parser("evaluate", help="Evaluate one RP-MIN checkpoint on a split.")
    add_scoring_options(evaluation)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--scale", type=int, choices=(2, 4), help="Optional checkpoint scale check.")
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[101])
    evaluation.add_argument("--workers", type=int, default=0)
    evaluation.add_argument("--save-images", action="store_true")
    evaluation.add_argument("--attention", action="store_true")
    evaluation.add_argument("--self-ensemble", action="store_true", help="Average eight flip/transpose predictions.")
    evaluation.add_argument("--calibration", type=Path, help="Optional training-fitted camera calibration JSON.")
    add_patch_grid_options(evaluation, "Run inference on individual LR grid patches.")
    evaluation.add_argument("--hr-origin", type=int, help="HR grid origin, default is LR origin multiplied by scale.")
    evaluation.add_argument("--batch", type=int, default=1, help="Inference batch size for fixed-size patches.")

    inference = commands.add_parser("infer", help="Super-resolve one RGB or grayscale LR image.")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)
    inference.add_argument("--seed", type=int, default=101)
    inference.add_argument("--self-ensemble", action="store_true")
    inference.add_argument("--calibration", type=Path, help="Optional training-fitted camera calibration JSON.")

    registration = commands.add_parser("register", help="Register training pairs using SuperPoint and LightGlue.")
    add_manifest_options(registration)
    registration.add_argument("--scale", type=int, choices=(2, 4), required=True)
    registration.add_argument("--patch", type=int, default=128)
    registration.add_argument("--keep-overlap", action="store_true", help="Keep the largest valid rectangle; --patch is its minimum side.")
    registration.add_argument("--threshold", type=float, default=2.0, help="RANSAC fit threshold in LR pixels.")
    registration.add_argument("--min-inlier-ratio", type=float, default=0.45)
    registration.add_argument("--max-holdout", type=float, default=2.0, help="Maximum holdout median error in LR pixels.")
    registration.add_argument("--seed", type=int, default=42)
    registration.add_argument("--weights-cache", type=Path, help="Torch Hub cache directory containing checkpoints/.")

    references = commands.add_parser("align-references", help="Prepare paired HR references on the original LR grid.")
    add_manifest_options(references)
    references.add_argument("--scale", type=int, choices=(2, 4), required=True)
    references.add_argument("--margin", type=int, default=12, help="Common border removed in LR pixels.")
    references.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    references.add_argument("--refine", action="store_true", help="Allow affine ECC refinement initialized by LightGlue.")
    references.add_argument("--seed", type=int, default=42)
    references.add_argument("--weights-cache", type=Path)

    return parser.parse_args(argv)


def run(args):
    command = args.command
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = get_device(args.device)
    if command == "audit":
        return manifest_summary(read_manifest(args.manifest))
    if command == "split":
        return split_manifest(args.manifest, args.out, args.seed)
    if command == "prepare-dataset":
        from .prepare import prepare_dataset
        return prepare_dataset(args.root, args.out, args.seed, args.groups, args.demo_tile, args.demo_gap)
    if command == "train":
        model_config = config_from_args(ModelConfig, args)
        train_config = config_from_args(TrainConfig, args)
        path = train(
            args.manifest, args.out, model_config, train_config, device, args.workers, args.resume,
            init_checkpoint=args.init_checkpoint,
        )
        return {"checkpoint": str(path)}
    if command == "evaluate":
        from .calibration import CameraCalibration
        model = load_model(args.checkpoint, device)
        scale = model.config.scale
        if args.scale is not None and args.scale != scale:
            raise ValueError("--scale must match the checkpoint scale.")
        if args.hr_origin is not None and args.lr_patch is None:
            raise ValueError("--hr-origin requires --lr-patch.")
        grid = None if args.lr_patch is None else {
            "patch": args.lr_patch,
            "stride_x": args.lr_patch if args.stride_x is None else args.stride_x,
            "stride_y": args.lr_patch if args.stride_y is None else args.stride_y,
            "origin": args.origin, "hr_origin": args.hr_origin,
        }
        return evaluate_manifest(
            args.manifest, args.split, model, scale, args.seeds, args.out, device, args.workers,
            args.border, args.save_images, args.attention,
            CameraCalibration.load(args.calibration) if args.calibration else None,
            patch_grid=grid, batch=args.batch, self_ensemble=args.self_ensemble,
        )
    if command == "infer":
        from .calibration import CameraCalibration
        model = load_model(args.checkpoint, device)
        generator = torch.Generator().manual_seed(args.seed)
        calibration = CameraCalibration.load(args.calibration) if args.calibration else None
        with torch.inference_mode():
            lr = load_rgb(args.input)[None].to(device)
            if calibration is not None:
                lr = calibration.align(lr)
            output = predict(model, lr, generator, args.self_ensemble)
            if calibration is not None:
                output = calibration.correct_tone(output, lr)
        save_rgb(output[0], args.out)
        return {"image": str(args.out)}
    if command == "register":
        from .registration import register_training
        if args.weights_cache is not None:
            torch.hub.set_dir(str(args.weights_cache.resolve()))
        return register_training(
            args.manifest, args.out, args.scale, args.patch, device, args.threshold,
            args.min_inlier_ratio, args.max_holdout, args.seed, args.keep_overlap,
        )
    if command == "align-references":
        from .paired_reference import prepare_references
        if args.weights_cache is not None:
            torch.hub.set_dir(str(args.weights_cache.resolve()))
        prepare_references(args.manifest, args.out, args.scale, device, args.margin, args.refine, args.splits, args.seed)
        return {"aligned": str(args.out / "aligned.csv"), "raw": str(args.out / "raw.csv")}
    raise ValueError(f"Unknown command {command}")


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
