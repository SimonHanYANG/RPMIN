import csv
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test")
FIELDS = ("id", "split", "group", "lr", "hr")


@dataclass(frozen=True)
class Pair:
    id: str
    split: str
    group: str
    lr: Path
    hr: Path


def read_manifest(path, *, require_split=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"id", "lr", "hr"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: required columns are id, lr, hr.")
        pairs = []
        for line, row in enumerate(reader, start=2):
            identifier = (row["id"] or "").strip()
            split = (row.get("split") or "").strip()
            group = (row.get("group") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
                raise ValueError(f"{path}:{line}: id must be a filename-safe ASCII identifier.")
            if (require_split or split) and split not in SPLITS:
                raise ValueError(f"{path}:{line}: split must be train, val, or test.")
            if not row["lr"] or not row["hr"]:
                raise ValueError(f"{path}:{line}: missing lr or hr path.")
            lr = (path.parent / row["lr"].strip()).resolve()
            hr = (path.parent / row["hr"].strip()).resolve()
            if not lr.is_file() or not hr.is_file():
                raise ValueError(f"{path}:{line}: image file missing ({lr}, {hr}).")
            pairs.append(Pair(identifier, split, group, lr, hr))
    if not pairs:
        raise ValueError(f"No pairs in {path}.")
    if len({pair.id.casefold() for pair in pairs}) != len(pairs):
        raise ValueError(f"Repeated sample id in {path}.")
    assigned = defaultdict(set)
    for pair in pairs:
        if pair.split:
            for key in ("lr", "hr", "group"):
                value = getattr(pair, key)
                if value:
                    namespace = key if key == "group" else "image"
                    assigned[(namespace, value)].add(pair.split)
    leaks = [(key, value) for (key, value), splits in assigned.items() if len(splits) > 1]
    if leaks:
        raise ValueError(f"Image or group occurs in multiple splits: {leaks[:5]}")
    return sorted(pairs, key=lambda pair: pair.id)


def write_manifest(path, pairs):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def relative(image):
        try:
            return Path(os.path.relpath(image, path.parent)).as_posix()
        except ValueError:
            return image.as_posix()

    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(FIELDS)
        writer.writerows((p.id, p.split, p.group, relative(p.lr), relative(p.hr)) for p in pairs)


def assign_group_splits(group_sizes, seed):
    if len(group_sizes) < 3:
        raise ValueError("At least three independent groups are required.")
    names = sorted(group_sizes)
    random.Random(seed).shuffle(names)
    names.sort(key=group_sizes.__getitem__, reverse=True)
    targets = dict(zip(SPLITS, (0.8, 0.1, 0.1)))
    counts = Counter()
    assignments = {}
    total = sum(group_sizes.values())
    for index, name in enumerate(names):
        empty_splits = [split for split in SPLITS if counts[split] == 0]
        eligible = empty_splits if len(names) - index == len(empty_splits) else SPLITS
        # Minimize squared deviation from target patch counts without splitting a group.
        split = max(eligible, key=lambda s: targets[s] * total - counts[s])
        assignments[name] = split
        counts[split] += group_sizes[name]
    return assignments


def split_manifest(source, target, seed):
    pairs = read_manifest(source, require_split=False)
    if any(not pair.group or pair.split for pair in pairs):
        raise ValueError("Group splitting requires a group on every row and an empty split column.")
    assignments = assign_group_splits(Counter(pair.group for pair in pairs), seed)
    assigned = [Pair(p.id, assignments[p.group], p.group, p.lr, p.hr) for p in pairs]
    images = defaultdict(set)
    for pair in assigned:
        for image in (pair.lr, pair.hr):
            images[image].add(pair.split)
    if any(len(splits) > 1 for splits in images.values()):
        raise ValueError("The same image was assigned across splits. Fix inconsistent group IDs.")
    write_manifest(target, assigned)
    return manifest_summary(assigned)


def manifest_summary(pairs):
    return {
        split: {
            "pairs": sum(pair.split == split for pair in pairs),
            "groups": len({pair.group for pair in pairs if pair.split == split and pair.group}),
            "missing_group": sum(pair.split == split and not pair.group for pair in pairs),
        }
        for split in SPLITS
    }


def load_rgb(path):
    with Image.open(path) as image:
        return rgb_tensor(image, path)


def rgb_tensor(image, path):
    if image.mode.startswith("I") or image.mode == "F":
        raise ValueError(f"{path}: supply a consistently normalized 8-bit RGB or grayscale image.")
    array = np.array(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 255


def save_rgb(image, path):
    array = (image.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.permute(1, 2, 0).numpy()).save(path)


class PairedImages(Dataset):
    def __init__(self, pairs, split, scale, patch=None, samples_per_pair=1, cache=False):
        self.pairs = [pair for pair in pairs if pair.split == split]
        self.scale = scale
        self.patch = patch
        self.samples_per_pair = samples_per_pair
        self.cache = {} if cache else None
        if not self.pairs:
            raise ValueError(f"Manifest has no {split} samples.")
        if patch is not None and patch < 4:
            raise ValueError("LR patch size must be at least 4.")
        if samples_per_pair < 1 or (samples_per_pair > 1 and patch is None):
            raise ValueError("Multiple samples per pair require a positive count and an LR crop size.")

    def __len__(self):
        return len(self.pairs) * self.samples_per_pair

    def open_image(self, path):
        if self.cache is None:
            return Image.open(path)
        if path not in self.cache:
            with Image.open(path) as image:
                self.cache[path] = image.copy()
        return nullcontext(self.cache[path])

    def __getitem__(self, index):
        pair = self.pairs[index // self.samples_per_pair]
        scale = self.scale
        with self.open_image(pair.lr) as low, self.open_image(pair.hr) as high:
            width, height = low.size
            if min(height, width) < 4 or high.size != (width * scale, height * scale):
                raise ValueError(f"{pair.id}: HR must be exactly {scale}× LR in both axes, LR size ≥ 4.")
            if self.patch is not None:
                patch = self.patch
                if min(height, width) < patch:
                    raise ValueError(f"{pair.id}: LR {height}×{width} is smaller than patch {patch}.")
                y = random.randint(0, height - patch)
                x = random.randint(0, width - patch)
                box = (x, y, x + patch, y + patch)
                low = low.crop(box)
                high = high.crop(tuple(value * scale for value in box))
            lr, hr = rgb_tensor(low, pair.lr), rgb_tensor(high, pair.hr)
        if self.patch is not None:
            for axis in (-2, -1):
                if random.getrandbits(1):
                    lr, hr = lr.flip(axis), hr.flip(axis)
            if random.getrandbits(1):
                lr, hr = lr.transpose(-2, -1), hr.transpose(-2, -1)
        return lr, hr, pair.id


class PairedGrid(PairedImages):
    """Evaluate reconstruction from individual LR patches on a fixed paired grid."""

    def __init__(self, pairs, split, scale, patch, stride_x, stride_y, origin=1, hr_origin=None):
        super().__init__(pairs, split, scale, cache=True)
        if patch < 4 or min(stride_x, stride_y) < 1 or origin < 0 or (hr_origin is not None and hr_origin < 0):
            raise ValueError("Patch evaluation requires patch ≥ 4, positive strides and nonnegative origins.")
        self.crop_size = patch
        self.hr_offset = 0 if hr_origin is None else hr_origin - origin * scale
        self.crops = []
        for pair in self.pairs:
            with Image.open(pair.lr) as low, Image.open(pair.hr) as high:
                width, height = low.size
                if high.size != (scale * width, scale * height):
                    raise ValueError(f"{pair.id}: HR must be exactly {scale}× LR in both axes.")
                start = len(self.crops)
                for y in range(origin, height - origin - patch + 1, stride_y):
                    for x in range(origin, width - origin - patch + 1, stride_x):
                        hx, hy = x * scale + self.hr_offset, y * scale + self.hr_offset
                        if min(hx, hy) >= 0 and hx + patch * scale <= high.width and hy + patch * scale <= high.height:
                            self.crops.append((pair, x, y))
                if len(self.crops) == start:
                    raise ValueError(f"{pair.id}: no valid evaluation patches for the specified grid.")

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        pair, x, y = self.crops[index]
        patch = self.crop_size
        low_box = (x, y, x + patch, y + patch)
        high_box = tuple(coordinate * self.scale + self.hr_offset for coordinate in low_box)
        with self.open_image(pair.lr) as low, self.open_image(pair.hr) as high:
            lr = rgb_tensor(low.crop(low_box), pair.lr)
            hr = rgb_tensor(high.crop(high_box), pair.hr)
        return lr, hr, f"{pair.id}-x{x}-y{y}"
