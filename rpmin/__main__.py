import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from .data import SPLITS, load_rgb, manifest_summary, read_manifest, save_rgb, split_manifest
from .engine import TrainConfig, dump_json, evaluate_manifest, get_device, load_model, predict, train
from .model import ModelConfig
from .views import VIEW_MODES


def add_model_options(parser, *, suite=False):
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--radius", type=float, help="Fixed subpixel radius. Default draws a random radius.")
    if suite:
        return
    parser.add_argument("--directions", type=int, choices=(0, 2, 4, 8), default=8)
    parser.add_argument("--view-mode", choices=VIEW_MODES, default="difference")
    parser.add_argument("--amfm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rpum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deformable", action=argparse.BooleanOptionalAction, default=True)
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


def method_paths(entries):
    methods = {}
    for entry in entries:
        name, separator, path = entry.partition("=")
        if not separator or not name.strip() or not path.strip() or name in methods:
            raise ValueError("Use --method NAME=PATH with a distinct nonempty NAME for each method.")
        methods[name] = Path(path)
    return methods


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

    evaluation = commands.add_parser("evaluate", help="Evaluate one checkpoint or bicubic on a split.")
    add_scoring_options(evaluation)
    method = evaluation.add_mutually_exclusive_group(required=True)
    method.add_argument("--checkpoint", type=Path)
    method.add_argument("--bicubic", action="store_true")
    evaluation.add_argument("--scale", type=int, choices=(2, 4), help="Required for bicubic.")
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

    suite = commands.add_parser("ablate", help="Retrain controlled variants and score the same test split.")
    add_manifest_options(suite)
    add_model_options(suite, suite=True)
    add_train_options(suite)
    suite.add_argument("--suite", required=True, choices=("components", "directions", "input-controls", "deformable", "frequency"))
    suite.add_argument("--seeds", type=int, nargs="+", default=[101])

    benchmark = commands.add_parser("benchmark", help="Convolution FLOPs, parameters and end-to-end forward latency.")
    add_model_options(benchmark)
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--factory", help="External baseline builder as module:callable, accepting a checkpoint path.")
    benchmark.add_argument("--out", type=Path, required=True)
    benchmark.add_argument("--height", type=int, default=128)
    benchmark.add_argument("--image-width", type=int, default=128)
    benchmark.add_argument("--batch", type=int, default=1)
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--repeats", type=int, default=100)
    benchmark.add_argument("--seed", type=int, default=42)

    gradient = commands.add_parser("gradient", help="Run the analytical displacement-gradient experiment.")
    gradient.add_argument("--out", type=Path, required=True)
    gradient.add_argument("--seed", type=int, default=42)

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
    references.add_argument("--margin", type=int, default=12, help="Common LR border removed in both comparison sets.")
    references.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    references.add_argument("--refine", action="store_true", help="Allow affine ECC refinement initialized by LightGlue.")
    references.add_argument("--seed", type=int, default=42)
    references.add_argument("--weights-cache", type=Path)

    scores = commands.add_parser("score-images", help="Score all methods from saved PNGs using the same protocol.")
    add_scoring_options(scores)
    scores.add_argument("--method", action="append", required=True, metavar="NAME=FOLDER")
    add_patch_grid_options(scores, "Score grid patches cut from the saved full-field predictions.")

    segmentation = commands.add_parser("segment-score", help="Score precomputed Hover-Net instance maps.")
    add_manifest_options(segmentation)
    segmentation.add_argument("--split", choices=SPLITS, default="test")
    segmentation.add_argument("--reference", type=Path, required=True)
    segmentation.add_argument("--reference-kind", choices=("manual", "hovernet-hr-pseudo"), required=True)
    segmentation.add_argument("--method", action="append", required=True, metavar="NAME=FOLDER")
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
    if command in {"train", "ablate"}:
        model_config = config_from_args(ModelConfig, args)
        train_config = config_from_args(TrainConfig, args)
        if command == "train":
            path = train(
                args.manifest, args.out, model_config, train_config, device, args.workers, args.resume,
                init_checkpoint=args.init_checkpoint,
            )
            return {"checkpoint": str(path)}
        from .experiments import run_suite
        return run_suite(
            args.suite, args.manifest, args.out, model_config, train_config, device, args.workers, args.seeds, args.resume
        )
    if command == "evaluate":
        from .calibration import CameraCalibration
        model = load_model(args.checkpoint, device) if args.checkpoint else None
        scale = model.config.scale if model is not None else args.scale
        if scale is None or (args.scale is not None and args.scale != scale):
            raise ValueError("Specify --scale for bicubic and match the checkpoint scale for model evaluation.")
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
    if command == "benchmark":
        from .experiments import profile_model
        summary = profile_model(
            config_from_args(ModelConfig, args), args.checkpoint, device, args.height,
            args.image_width, args.batch, args.warmup, args.repeats, args.seed, args.factory,
        )
        dump_json(args.out, summary)
        return summary
    if command == "gradient":
        from .experiments import gradient_experiment
        return gradient_experiment(args.out, args.seed)
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
    if command == "score-images":
        from .experiments import compare_images
        return compare_images(
            args.manifest, args.split, method_paths(args.method), args.out, args.border, device,
            args.lr_patch, args.stride_x, args.stride_y, args.origin,
        )
    if command == "segment-score":
        from .segmentation import evaluate_instances
        return evaluate_instances(
            args.manifest, args.split, args.reference, method_paths(args.method), args.reference_kind, args.out
        )
    raise ValueError(f"Unknown command {command}")


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
