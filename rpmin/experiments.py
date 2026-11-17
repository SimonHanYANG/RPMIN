from dataclasses import asdict, replace
from importlib import import_module
from itertools import product
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torchvision.ops import DeformConv2d

from .data import load_rgb, read_manifest
from .engine import dump_csv, dump_json, evaluate_manifest, load_model, manifest_digest, summarize, train
from .model import RPMIN
from .objective import grid_metrics, image_metrics
from PIL import Image
from .views import build_views, displacements, pixel_grid


def variants(suite, model):
    full = replace(model, directions=8, view_mode="difference", amfm=True, rpum=True, deformable=True)
    if suite == "components":
        return {
            f"displacement{d}_amfm{a}_rpum{r}": (
                replace(full, directions=8 if d else 0, amfm=bool(a), rpum=bool(r)), {}
            )
            for d, a, r in product((0, 1), repeat=3)
        }
    if suite == "directions":
        return {f"views{d + 1}": (replace(full, directions=d), {}) for d in (2, 4, 8)}
    if suite == "input-controls":
        return {
            mode: (replace(full, view_mode=mode), {})
            for mode in ("difference", "sobel", "scharr", "random-direction", "shifted", "replicated")
        }
    if suite == "deformable":
        return {"dconv": (full, {}), "conv": (replace(full, deformable=False), {})}
    if suite == "frequency":
        return {"full_loss": (full, {}), "pixel_only": (full, {"pixel_weight": 1.0, "frequency_weight": 0.0})}
    raise ValueError(f"Unknown experiment suite {suite}")


def run_suite(suite, manifest, out, model, training, device, workers, seeds, resume=False):
    out = Path(out)
    table = []
    for name, (variant, changes) in variants(suite, model).items():
        print(f"[{name}]", flush=True)
        run = out / name
        checkpoint = train(
            manifest, run, variant, replace(training, **changes), device, workers,
            resume=resume and (run / "latest.pt").exists(),
        )
        trained = load_model(checkpoint, device)
        summary = evaluate_manifest(
            manifest, "test", trained, variant.scale, seeds, run / "test", device, workers,
            training.border, images=False, maps=False,
        )
        table.append({
            "variant": name, "views": variant.directions + 1,
            "parameters": sum(parameter.numel() for parameter in trained.parameters()),
            "psnr_mean": summary["psnr"]["mean"], "psnr_shift_std": summary["psnr"]["std"],
            "ssim_mean": summary["ssim"]["mean"], "ssim_shift_std": summary["ssim"]["std"],
        })
    dump_csv(out / "comparison.csv", table)
    return table


@torch.inference_mode()
def compare_images(manifest, split, methods, out, border, device, lr_patch=None, stride_x=None, stride_y=None, origin=0):
    pairs = [pair for pair in read_manifest(manifest) if pair.split == split]
    if not pairs or not methods:
        raise ValueError("At least one sample and method are required.")
    stride_x = lr_patch if stride_x is None else stride_x
    stride_y = lr_patch if stride_y is None else stride_y
    rows_by_method = {method: [] for method in methods}
    for pair in pairs:
        hr = load_rgb(pair.hr)[None].to(device)
        if lr_patch is not None:
            with Image.open(pair.lr) as low:
                scale = hr.shape[-1] // low.width
                if hr.shape[-2:] != (low.height * scale, low.width * scale):
                    raise ValueError(f"{pair.id}: LR and HR sizes do not match an integer scale.")
        for method, folder in methods.items():
            sr = load_rgb(Path(folder) / f"{pair.id}.png")[None].to(device)
            if lr_patch is None:
                psnr, ssim = image_metrics(sr, hr, border)
                rows_by_method[method].append({
                    "method": method, "id": pair.id, "psnr": psnr.item(), "ssim": ssim.item(),
                })
            else:
                points, psnr, ssim = grid_metrics(sr, hr, scale, lr_patch, stride_x, stride_y, origin, border)
                rows_by_method[method].extend(
                    {"method": method, "id": pair.id, "lr_x": x, "lr_y": y, "psnr": p, "ssim": s}
                    for (x, y), p, s in zip(points, psnr.tolist(), ssim.tolist())
                )
    summary = {
        "split": split, "manifest_sha256": manifest_digest(manifest), "border_hr": border,
        "methods": [{"method": method, **summarize(rows)} for method, rows in rows_by_method.items()],
    }
    if lr_patch is not None:
        summary["patch_grid"] = {"lr_patch": lr_patch, "stride_x": stride_x, "stride_y": stride_y, "origin": origin,
                                 "fields": len(pairs), "protocol": "crop saved full-field predictions"}
    out = Path(out)
    dump_csv(out / "per_image.csv", [row for rows in rows_by_method.values() for row in rows])
    dump_json(out / "summary.json", summary)
    return summary


def convolution_macs(model, forward):
    """Count kernel multiply-accumulates and explicitly exclude spatial sampling."""
    macs = 0

    def record(module, _, output):
        nonlocal macs
        macs += output.numel() * module.weight[0].numel()

    hooks = [
        module.register_forward_hook(record) for module in model.modules()
        if isinstance(module, (nn.Conv2d, DeformConv2d))
    ]
    try:
        forward()
    finally:
        for hook in hooks:
            hook.remove()
    return macs


@torch.inference_mode()
def profile_model(config, checkpoint, device, height, width, batch, warmup, repeats, seed, factory=None):
    if min(height, width) < 4 or min(batch, warmup, repeats) < 1:
        raise ValueError("Benchmark requires dimensions ≥ 4 and positive batch, warmup and repeats.")
    torch.manual_seed(seed)
    if factory:
        module, separator, builder = factory.partition(":")
        if not separator or not module or not builder:
            raise ValueError("Use --factory module:callable.")
        model = getattr(import_module(module), builder)(checkpoint).to(device).eval()
    else:
        model = load_model(checkpoint, device) if checkpoint else RPMIN(config).to(device).eval()
    image = torch.rand(batch, 3, height, width, device=device)
    generator = torch.Generator().manual_seed(seed)

    def forward():
        return model(image) if factory else model(image, generator=generator)

    output = forward()
    scale = model.config.scale if not factory else config.scale
    if not isinstance(output, torch.Tensor) or output.shape != (batch, 3, height * scale, width * scale):
        raise ValueError(f"Benchmark expects a single RGB tensor with output scale ×{scale}.")
    del output
    macs = convolution_macs(model, forward)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        forward()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        forward()
        synchronize()
        times.append((perf_counter() - start) * 1000)
    return {
        "model": factory if factory else asdict(model.config), "scale": scale,
        "shape_bchw": list(image.shape), "device": str(device), "precision": str(image.dtype),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "platform": platform.platform(),
        "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "convolution_macs_per_image": macs // batch,
        "convolution_flops_per_image_at_2_per_mac": 2 * macs // batch,
        "flops_scope": "Conv2d weights and deformable kernels only, excluding interpolation, functional ops and bias.",
        "latency_ms_per_batch_median": median(times), "latency_ms_per_batch_p95": float(np.quantile(times, 0.95)),
        "peak_cuda_bytes_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "warmup": warmup, "repeats": repeats,
    }


def analytical_image():
    y, x = np.mgrid[:256, :256].astype(np.float64)
    image = np.full_like(x, 0.75)
    dx, dy = np.zeros_like(x), np.zeros_like(y)
    components = (
        (53, 68, 17, 25, -0.34, 0.0, 0.0),
        (111, 76, 18, 14, -0.23, 0.0, 0.0),
        (161, 160, 27, 17, -0.29, 0.0, 0.0),
        (197, 108, 19, 23, -0.27, 0.0, 0.0),
        (70, 175, 28, 21, -0.28, 0.0, 0.0),
        (84, 174, 24, 26, 0.055, 0.16, 0.12),
        (160, 163, 28, 25, 0.063, -0.17, 0.08),
        (102, 116, 55, 52, 0.09, 0.04, -0.06),
    )
    for cx, cy, sx, sy, amplitude, wx, wy in components:
        u, v = x - cx, y - cy
        envelope = amplitude * np.exp(-0.5 * ((u / sx)**2 + (v / sy)**2))
        phase = wx * u + wy * v
        signal = envelope * np.cos(phase)
        image += signal
        dx += -u / sx**2 * signal - wx * envelope * np.sin(phase)
        dy += -v / sy**2 * signal - wy * envelope * np.sin(phase)
    return image, dx, dy


def recover_gradient(observations, offsets):
    x, y = np.moveaxis(offsets.numpy().astype(np.float64), -1, 0)
    design = np.stack((x, y, x*x / 2, x*y, y*y / 2), axis=-1)
    responses = observations[:, 1:, 0].flatten(2).numpy()
    derivatives = np.linalg.pinv(design) @ responses
    magnitude = np.hypot(derivatives[:, 0], derivatives[:, 1])
    return magnitude.reshape(len(design), *observations.shape[-2:]), design


def gradient_noise_sweep(image, truth, seed, trials=24):
    rows = []
    truth = truth[None, 2:-2, 2:-2]
    noise = torch.randn((trials, 1, *image.shape[-2:]), generator=torch.Generator().manual_seed(seed + 1))
    for directions, mode in ((2, "difference"), (4, "difference"), (8, "difference"), (8, "random-direction")):
        offsets = displacements(trials, directions, mode, 0.5, torch.Generator().manual_seed(seed))
        for noise_std in (0.0, 0.005, 0.01, 0.02):
            observations = build_views(image + noise_std * noise, offsets, mode)
            recovered, design = recover_gradient(observations, offsets)
            error = recovered[:, 2:-2, 2:-2] - truth
            rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
            ranks = np.linalg.matrix_rank(design)
            rows.append({
                "directions": directions, "geometry": "compass" if mode == "difference" else "random",
                "input_noise_std": noise_std, "radius_px": 0.5, "trials": trials,
                "min_design_rank": int(ranks.min()),
                "median_design_condition_if_full_rank": float(np.median(np.linalg.cond(design))) if ranks.min() == 5 else None,
                "gradient_rmse_mean": float(rmse.mean()), "gradient_rmse_std": float(rmse.std(ddof=1)),
            })
    return rows


def gradient_experiment(out, seed):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hr, dx, dy = analytical_image()

    def average_pool(array):
        return array.reshape(64, 4, 64, 4).mean(axis=(1, 3))

    lr = average_pool(hr)
    truth = np.hypot(average_pool(dx), average_pool(dy)) * 4
    truth_roi = truth[2:-2, 2:-2]
    tensor = torch.from_numpy(lr)[None, None]
    rows = []
    for radius in (0.25, 0.5, 0.75, 1.0, None):
        generator = torch.Generator().manual_seed(seed)
        offsets = displacements(1, 8, "difference", radius, generator)
        observations = build_views(tensor, offsets)
        views = observations[0, 1:, 0].numpy()
        magnitudes, designs = recover_gradient(observations, offsets)
        recovered, design = magnitudes[0], designs[0]
        roi = recovered[2:-2, 2:-2]
        rows.append({
            "radius_mode": "random" if radius is None else "fixed",
            "radius_px": float(offsets[0, 0].norm()), "design_rank": int(np.linalg.matrix_rank(design)),
            "design_condition": float(np.linalg.cond(design)),
            "pearson_r": float(np.corrcoef(roi.ravel(), truth_roi.ravel())[0, 1]),
            "gradient_rmse_per_lr_pixel": float(np.sqrt(np.mean((roi - truth_roi)**2))),
        })
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), layout="constrained")
    upper = np.max(np.abs(views[:3]))
    for axis, array, title in zip(axes[0], (lr, *views[:3]), ("LR input", "Right", "Down-right", "Down")):
        if title == "LR input":
            axis.imshow(array, cmap="gray")
        else:
            axis.imshow(array, cmap="RdBu_r", vmin=-upper, vmax=upper)
        axis.set_title(title)
        axis.set_axis_off()
    for axis, array, title in zip(axes[1], (truth, recovered), ("True magnitude", "Recovered magnitude")):
        axis.imshow(array, cmap="magma", vmin=0, vmax=truth.max())
        axis.set_title(title)
        axis.set_axis_off()
    error_axis = axes[1, 2]
    error = np.abs(truth - recovered)
    heatmap = error_axis.imshow(error, cmap="magma", vmin=0, vmax=error[2:-2, 2:-2].max())
    error_axis.set_title("Absolute error")
    error_axis.set_axis_off()
    fig.colorbar(heatmap, ax=error_axis, fraction=0.045, pad=0.025)
    scatter = axes[1, 3]
    scatter.scatter(truth_roi, roi, s=2, alpha=0.25)
    scatter.plot([0, truth.max()], [0, truth.max()], color="black", linewidth=0.7)
    scatter.set_title(f"Pearson r = {rows[-1]['pearson_r']:.5f}")
    scatter.set(xlabel="True magnitude", ylabel="Recovered magnitude")
    fig.savefig(out / "gradient.png", dpi=180)
    plt.close(fig)
    dump_csv(out / "gradient.csv", rows)
    dump_csv(out / "noise_sweep.csv", gradient_noise_sweep(tensor, truth, seed))
    dump_json(out / "settings.json", {
        "seed": seed, "hr_size": 256, "lr_size": 64, "ignored_lr_border": 2,
        "noise": "Gaussian noise added once to the LR input per trial, before building all views.",
    })
    return rows
