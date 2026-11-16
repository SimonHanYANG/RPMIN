import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import platform
import random
from statistics import fmean, stdev
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .data import PairedGrid, PairedImages, manifest_summary, read_manifest, save_rgb
from .model import ModelConfig, RPMIN
from .objective import ReconstructionLoss, image_metrics


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch: int = 4
    patch: int = 128
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup: int = 5
    weight_decay: float = 0.01
    pixel_weight: float = 0.9
    frequency_weight: float = 0.1
    phase: str = "angle"
    seed: int = 42
    val_seed: int = 101
    border: int = 0
    validate_every: int = 1
    patches_per_field: int = 1
    cache_train: bool = False
    mse_weight: float = 0.0
    ssim_weight: float = 0.0
    val_patch_stride: int = 0

    def __post_init__(self):
        if self.epochs < 1 or self.batch < 1 or self.patch < 4 or not 0 <= self.warmup < self.epochs:
            raise ValueError("Require positive epochs and batch size, patch ≥ 4, and 0 ≤ warmup < epochs.")
        if not 0 <= self.min_lr <= self.lr or self.lr <= 0 or self.weight_decay < 0 or self.border < 0:
            raise ValueError("Invalid learning rate, weight decay, or metric border.")
        weights = (self.pixel_weight, self.frequency_weight, self.mse_weight, self.ssim_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
            raise ValueError("Use nonnegative weights and at least one nonzero loss term.")
        if self.validate_every < 1:
            raise ValueError("Validation interval must be at least one epoch.")
        if self.patches_per_field < 1:
            raise ValueError("Patches per training field must be positive.")
        if self.val_patch_stride < 0:
            raise ValueError("Validation patch stride cannot be negative.")


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def dump_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RPMIN(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model_state"])
    model.checkpoint_epoch = state.get("epoch")
    return model.eval()


def loader(dataset, device, workers, *, batch=1, seed=None):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset, batch_size=batch, shuffle=seed is not None, num_workers=workers,
        pin_memory=device.type == "cuda", generator=generator,
    )


def learning_rate(config, epoch):
    if epoch <= config.warmup:
        return config.lr * epoch / config.warmup
    decay = config.epochs - config.warmup
    ratio = (epoch - config.warmup - 1) / max(1, decay - 1)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * ratio))


def predict(model, image, generator, self_ensemble=False):
    if not self_ensemble:
        return model(image, generator=generator)
    total = None
    # Reuse displacement draws so the ensemble changes geometry only.
    initial_state = generator.get_state()
    for transpose in (False, True):
        for axes in ((), (-1,), (-2,), (-1, -2)):
            sample = image.transpose(-2, -1) if transpose else image
            if axes:
                sample = sample.flip(axes)
            generator.set_state(initial_state)
            sr = model(sample, generator=generator)
            if axes:
                sr = sr.flip(axes)
            if transpose:
                sr = sr.transpose(-2, -1)
            total = sr if total is None else total + sr
    return total / 8


@torch.inference_mode()
def evaluate(data, model, *, device, scale, seed, border, images=None, maps_out=None, calibration=None,
             self_ensemble=False):
    if model is not None:
        model.eval()
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for lr, hr, identifiers in data:
        lr, hr = lr.to(device), hr.to(device)
        if calibration is not None:
            lr = calibration.align(lr)
        if model is None:
            sr = F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)
            maps = {}
        elif maps_out is None:
            sr = predict(model, lr, generator, self_ensemble)
            maps = {}
        else:
            sr, maps = model(lr, generator=generator, return_maps=True)
        if calibration is not None:
            sr = calibration.correct_tone(sr, lr)
        psnr, ssim = image_metrics(sr, hr, border)
        for index, (identifier, p, s) in enumerate(zip(identifiers, psnr.tolist(), ssim.tolist())):
            rows.append({"id": identifier, "seed": seed, "psnr": p, "ssim": s})
            if images is not None:
                save_rgb(sr[index], images / f"{identifier}.png")
            for name, weights in maps.items():
                # Preserve numeric weights as well as viewable grayscale maps.
                values = weights[index].detach().cpu().numpy()
                maps_out.mkdir(parents=True, exist_ok=True)
                np.save(maps_out / f"{identifier}_{name}.npy", values)
                save_rgb(weights[index].mean(0, keepdim=True).expand(3, -1, -1), maps_out / f"{identifier}_{name}.png")
    return rows


def summarize(rows):
    return {"pairs": len(rows), **{metric: fmean(row[metric] for row in rows) for metric in ("psnr", "ssim")}}


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_manifest(
    manifest, split, model, scale, seeds, out, device, workers, border, images, maps, calibration=None,
    patch_grid=None, batch=1, self_ensemble=False,
):
    pairs = read_manifest(manifest)
    if patch_grid is None and batch != 1:
        raise ValueError("Use batch size one for evaluating full fields.")
    if patch_grid is not None and calibration is not None:
        raise ValueError("Camera calibration requires whole-field LR context.")
    if patch_grid is None:
        dataset = PairedImages(pairs, split, scale)
    else:
        dataset = PairedGrid(pairs, split, scale, **patch_grid)
    data = loader(dataset, device, workers, batch=batch)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be distinct.")
    if maps and (model is None or not model.config.amfm):
        raise ValueError("Attention export requires an AMFM checkpoint.")
    if self_ensemble and (model is None or maps):
        raise ValueError("Self-ensemble requires a model and cannot export individual attention maps.")
    out = Path(out)
    all_rows = []
    runs = []
    for seed in seeds:
        target = out / f"seed_{seed}"
        rows = evaluate(
            data, model, device=device, scale=scale, seed=seed, border=border,
            images=target / "sr" if images else None, maps_out=target / "attention" if maps else None,
            calibration=calibration, self_ensemble=self_ensemble,
        )
        all_rows.extend(rows)
        runs.append({"seed": seed, **summarize(rows)})
    summary = {
        "manifest_sha256": manifest_digest(manifest), "split": split, "scale": scale, "border_hr": border,
        "model": asdict(model.config) if model is not None else "bicubic",
        "runs": runs,
    }
    if model is not None and model.checkpoint_epoch is not None:
        summary["checkpoint_epoch"] = model.checkpoint_epoch
    if self_ensemble:
        summary["self_ensemble"] = "d4"
    if patch_grid is not None:
        summary["patch_input"] = {**patch_grid, "fields": len(dataset.pairs), "batch": batch}
    if calibration is not None:
        summary["calibration"] = asdict(calibration)
    for metric in ("psnr", "ssim"):
        values = [run[metric] for run in runs]
        summary[metric] = {"mean": fmean(values), "std": stdev(values) if len(values) > 1 else None}
    dump_csv(out / "per_image.csv", all_rows)
    dump_json(out / "summary.json", summary)
    return summary


def train(manifest, out, model_config, config, device, workers=0, resume=False, init_checkpoint=None):
    if resume and init_checkpoint is not None:
        raise ValueError("Use --init-checkpoint for a new run or --resume for an existing run.")
    pairs = read_manifest(manifest)
    train_data = PairedImages(
        pairs, "train", model_config.scale, config.patch, config.patches_per_field, config.cache_train,
    )
    val_dataset = (PairedGrid(pairs, "val", model_config.scale, config.patch,
                              config.val_patch_stride, config.val_patch_stride)
                   if config.val_patch_stride else PairedImages(pairs, "val", model_config.scale))
    val_data = loader(val_dataset, device, workers, batch=config.batch if config.val_patch_stride else 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    latest, best = out / "latest.pt", out / "best.pt"
    metadata = {
        "model_config": asdict(model_config), "train_config": asdict(config),
        "manifest_sha256": manifest_digest(manifest), "split_counts": manifest_summary(pairs),
    }
    if latest.exists() != resume:
        raise ValueError("A checkpoint already exists. Use --resume or a new output directory." if latest.exists()
                         else "--resume requires an existing latest.pt.")
    torch.manual_seed(config.seed)
    model = RPMIN(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = ReconstructionLoss(
        config.pixel_weight, config.frequency_weight, config.phase, config.mse_weight, config.ssim_weight,
    )
    validation_options = {"device": device, "scale": model_config.scale, "seed": config.val_seed,
                          "border": config.border}
    history = []
    best_psnr, best_epoch = -math.inf, 0
    initial_validation = None
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=True)
        state["train_config"] = asdict(TrainConfig(**state["train_config"]))
        state["model_config"] = asdict(ModelConfig(**state["model_config"]))
        if any(state[key] != value for key, value in metadata.items()):
            raise ValueError("Resume requires the same model, training settings and manifest content.")
        if "initial_checkpoint_sha256" in state:
            metadata["initial_checkpoint_sha256"] = state["initial_checkpoint_sha256"]
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state["history"]
        best_psnr = state["best_psnr"]
        initial_validation = state.get("initial_validation")
        best_epoch = state.get("best_epoch")
        if best_epoch is None:
            best_epoch = max((row["epoch"] for row in history if row["val_psnr"] == best_psnr), default=0)
        if best_epoch == len(history):
            save_checkpoint(best, {"model_config": metadata["model_config"], "model_state": state["model_state"],
                                   "epoch": best_epoch})
        elif not best.is_file():
            raise FileNotFoundError(f"Missing best checkpoint from an earlier epoch: {best}")
    else:
        if init_checkpoint is not None:
            initial = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
            initial_config = ModelConfig(**initial["model_config"])
            if initial_config == model_config:
                model.load_state_dict(initial["model_state"])
            elif model_config.detail_head and initial_config == replace(model_config, detail_head=False):
                missing, unexpected = model.load_state_dict(initial["model_state"], strict=False)
                expected = {f"detail.{key}" for key in model.detail.state_dict()}
                if set(missing) != expected or unexpected:
                    raise ValueError("Initialization must match all parameters outside the new detail head.")
            else:
                raise ValueError("Initialization checkpoint must match the requested model configuration.")
            metadata["initial_checkpoint_sha256"] = manifest_digest(init_checkpoint)
        initial_validation = summarize(evaluate(val_data, model, **validation_options))

    if initial_validation is not None:
        dump_json(out / "initial_validation.json", initial_validation)

    dump_json(out / "config.json", metadata)
    dump_json(out / "runtime.json", {
        "device": str(device), "platform": platform.platform(), "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(), "loader_workers": workers, "precision": "float32",
    })
    for epoch in range(len(history) + 1, config.epochs + 1):
        epoch_start = perf_counter()
        epoch_seed = config.seed + epoch
        random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        shift_rng = torch.Generator().manual_seed(epoch_seed + 1_000_000)
        train_loader = loader(train_data, device, workers, batch=config.batch, seed=epoch_seed)
        lr_value = learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr_value
        model.train()
        loss_sum = 0.0
        for lr, hr, _ in train_loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(lr, generator=shift_rng), hr)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}; checkpoint remains at the last completed epoch.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * lr.shape[0]
        should_validate = epoch == 1 or epoch % config.validate_every == 0 or epoch == config.epochs
        validation = summarize(evaluate(val_data, model, **validation_options)) if should_validate else None
        history.append({
            "epoch": epoch, "lr": lr_value, "loss": loss_sum / len(train_data),
            "val_psnr": validation["psnr"] if validation else None,
            "val_ssim": validation["ssim"] if validation else None,
            "steps": len(train_loader), "seconds_including_validation": perf_counter() - epoch_start,
        })
        improved = validation is not None and validation["psnr"] >= best_psnr
        if improved:
            best_psnr, best_epoch = validation["psnr"], epoch
        state = {
            **metadata, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "history": history, "best_psnr": best_psnr, "best_epoch": best_epoch,
            "initial_validation": initial_validation,
        }
        save_checkpoint(latest, state)
        if improved:
            save_checkpoint(best, {"model_config": metadata["model_config"], "model_state": state["model_state"],
                                   "epoch": best_epoch})
        dump_csv(out / "history.csv", history)
        print(json.dumps(history[-1]), flush=True)
    if history:
        dump_csv(out / "history.csv", history)
    initial_psnr = initial_validation["psnr"] if initial_validation is not None else None
    dump_json(out / "selection.json", {
        "best_epoch": best_epoch, "val_psnr": best_psnr, "initial_val_psnr": initial_psnr,
        "gain_vs_initial_db": best_psnr - initial_psnr if initial_psnr is not None else None,
    })
    return best
