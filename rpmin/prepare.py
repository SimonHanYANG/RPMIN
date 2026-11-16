from collections import Counter, defaultdict
import csv
from dataclasses import replace
import hashlib
from pathlib import Path
import re

from PIL import Image

from .data import Pair, SPLITS, assign_group_splits, manifest_summary, write_manifest
from .engine import dump_json


MAGNIFICATIONS = (10, 20, 40)
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
COLLECTION_NAMES = {
    "ciliated epithelial cell for super resolution": "mfa-cell",
    "mitos-14 for super resolution": "mitos-14",
}


def file_id(value):
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:55]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug or 'field'}-{digest}"


def read_groups(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"dataset", "field_id", "group"}.issubset(reader.fieldnames or ()):
            raise ValueError("Group CSV requires dataset, field_id, and group columns.")
        groups = {}
        for row in reader:
            key = ((row["dataset"] or "").strip().casefold(), (row["field_id"] or "").strip().casefold())
            group = (row["group"] or "").strip().casefold()
            if not all(key) or not group or key in groups:
                raise ValueError(f"Missing or duplicate group entry: {key}")
            groups[key] = group
    return groups


def inventory(collection, root):
    magnifications = defaultdict(dict)
    report = {}
    subfolders = {}
    mitos = COLLECTION_NAMES.get(collection.name.casefold()) == "mitos-14"
    for directory in sorted(collection.iterdir()):
        key = directory.name.lower()
        if key in {"x10", "x20", "x40"} and directory.is_dir():
            if key in subfolders:
                raise ValueError(f"Multiple {key} folders in {collection}.")
            subfolders[key] = directory
    for mag in MAGNIFICATIONS:
        key = f"x{mag}"
        directory = subfolders.get(key)
        matches = defaultdict(list)
        invalid = []
        files = [] if directory is None else sorted(
            p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in files:
            stem = re.fullmatch(rf"(.+)_x{mag}", path.stem, re.IGNORECASE)
            if stem is not None:
                name = stem[1]
            elif mitos and re.fullmatch(r"[A-Za-z0-9]+_[0-9]+", path.stem):
                name = path.stem
            else:
                invalid.append({"path": path.relative_to(root).as_posix(), "reason": "expected <field>_x<mag>"})
                continue
            relative = path.relative_to(directory)
            field = (relative.parent / name).as_posix().casefold()
            matches[field].append(path)
        for key_id, paths in matches.items():
            if len(paths) != 1:
                invalid.extend({"path": p.relative_to(root).as_posix(), "reason": "duplicate field ID"} for p in paths)
                continue
            path = paths[0]
            try:
                with Image.open(path) as image:
                    mode, size = image.mode, image.size
                if mode.startswith("I") or mode == "F":
                    raise ValueError(f"unsupported image mode {mode}")
            except (OSError, ValueError) as error:
                invalid.append({"path": path.relative_to(root).as_posix(), "reason": str(error)})
                continue
            magnifications[mag][key_id] = (path, size)
        report[key] = {
            "directory_present": directory is not None, "image_files": len(files),
            "recognized_fields": len(magnifications[mag]), "rejected": invalid,
        }
    return magnifications, report


def match_collection(images):
    result = {}
    for scale in (2, 4):
        lr, hr = images[10], images[10 * scale]
        pairs, dimension_mismatch = {}, []
        for field in sorted(lr.keys() & hr.keys()):
            low, lr_size = lr[field]
            high, hr_size = hr[field]
            if min(lr_size) < 4 or hr_size != tuple(x * scale for x in lr_size):
                dimension_mismatch.append({"field_id": field, "lr_wh": lr_size, "hr_wh": hr_size})
                continue
            pairs[field] = Pair(file_id(field), "", "", low, high)
        result[scale] = (pairs, {
            "missing_hr": sorted(lr.keys() - hr.keys()),
            "missing_lr": sorted(hr.keys() - lr.keys()),
            "invalid_dimensions": dimension_mismatch,
        })
    return result


def spatial_demo(pair, out, scale, tile, gap):
    """Tile one field in three spatial strips with a guard band at split boundaries."""
    with Image.open(pair.lr) as image:
        lr = image.convert("RGB")
    with Image.open(pair.hr) as image:
        hr = image.convert("RGB")
    horizontal = lr.width >= lr.height
    main, across = (lr.width, lr.height) if horizontal else (lr.height, lr.width)
    strips = (main - 2 * gap) // tile
    cross_tiles = across // tile
    if strips < 3 or not cross_tiles:
        return {"status": "too_small", "lr_wh": lr.size, "tile_lr": tile, "gap_lr": gap}

    pairs, regions = [], []
    along = (main - strips * tile - 2 * gap) // 2
    start = (across - cross_tiles * tile) // 2
    for split, count in zip(SPLITS, (strips - 2, 1, 1)):
        for column in range(count):
            for row in range(cross_tiles):
                major, minor = along + column * tile, start + row * tile
                x, y = (major, minor) if horizontal else (minor, major)
                box = (x, y, x + tile, y + tile)
                identifier = f"{pair.id}-{split}-{column}-{row}"
                lr_path, hr_path = out / "lr" / f"{identifier}.png", out / "hr" / f"{identifier}.png"
                for image, path, factor in ((lr, lr_path, 1), (hr, hr_path, scale)):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(tuple(factor * value for value in box)).save(path)
                pairs.append(Pair(identifier, split, "", lr_path, hr_path))
                regions.append({"id": identifier, "split": split, "lr_xyxy": box})
        along += count * tile + gap
    manifest = out / "spatial-demo.csv"
    write_manifest(manifest, pairs)
    return {
        "status": "single_field_spatial_demo", "manifest": manifest.as_posix(),
        "tile_lr": tile, "gap_lr": gap, "split_counts": manifest_summary(pairs), "tiles": regions,
    }


def prepare_dataset(root, out, seed=42, groups_path=None, demo_tile=None, demo_gap=8):
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder does not exist: {root}")
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Output must be separate from the source dataset.")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Choose a new or empty output folder: {out}")
    if demo_tile is not None and (demo_tile < 4 or demo_gap < 1):
        raise ValueError("Spatial demo requires tile size ≥ 4 and a positive guard band.")
    supplied_groups = read_groups(groups_path) if groups_path else None
    report = {"dataset_root": root.as_posix(), "seed": seed, "collections": {}}
    for collection in sorted(p for p in root.iterdir() if p.is_dir()):
        images, counts = inventory(collection, root)
        if not any(info["directory_present"] for info in counts.values()):
            continue
        name = COLLECTION_NAMES.get(collection.name.casefold(), file_id(collection.name))
        tasks = match_collection(images)
        fields = set().union(*(pairs for pairs, _ in tasks.values()))
        groups = {}
        grouping = "filename_field_ids"
        if name == "mitos-14":
            grouping = "mitos_filename_source_prefix"
        if supplied_groups is not None:
            grouping = "provided_source_groups"
        for field in sorted(fields):
            key = (collection.name.casefold(), field)
            if supplied_groups is not None and key not in supplied_groups:
                raise ValueError(f"Missing group for dataset={collection.name!r}, field_id={field!r}.")
            if supplied_groups is not None:
                groups[field] = supplied_groups[key]
            elif name == "mitos-14" and re.fullmatch(r"[a-z0-9]+_[0-9]+", Path(field).name):
                groups[field] = Path(field).name.rsplit("_", 1)[0]
            else:
                groups[field] = field
        sizes = Counter(groups.values())
        assignments = assign_group_splits(sizes, seed) if len(sizes) >= 3 else {}
        collection_report = {
            "directory": collection.name, "magnifications": counts, "fields_with_pairs": sorted(fields),
            "grouping": grouping,
            "source_groups_with_pairs": len(sizes), "tasks": {},
        }
        report["collections"][name] = collection_report
        for scale, (pairs, issues) in tasks.items():
            task_name = f"x{scale}"
            task_out = out / name / task_name
            details = {"valid_pairs": len(pairs), **issues, "status": "no_usable_pairs"}
            collection_report["tasks"][task_name] = details
            if not pairs:
                continue
            indexed = [replace(pair, group=groups[field]) for field, pair in pairs.items()]
            raw = task_out / "pairs.csv"
            write_manifest(raw, indexed)
            details["pair_index"] = raw.relative_to(out).as_posix()
            details["status"] = "insufficient_source_groups_or_split_coverage"
            if assignments:
                assigned = [replace(pair, split=assignments[pair.group]) for pair in indexed]
                summary = manifest_summary(assigned)
                details["split_counts"] = summary
                if all(summary[s]["pairs"] for s in SPLITS):
                    manifest = task_out / "manifest.csv"
                    write_manifest(manifest, assigned)
                    details.update(status="ready", manifest=manifest.relative_to(out).as_posix())
            if len(indexed) == 1 and demo_tile is not None:
                demo = spatial_demo(indexed[0], task_out / "demo", scale, demo_tile, demo_gap)
                if "manifest" in demo:
                    demo["manifest"] = Path(demo["manifest"]).relative_to(out).as_posix()
                details["demo"] = demo
    dump_json(out / "dataset_report.json", report)
    return report
