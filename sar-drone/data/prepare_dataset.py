"""Convert SAR source annotations into a train/val YOLO dataset."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Callable
import yaml

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["person", "fire", "debris", "water"]


def _normalise_class(value: str) -> int:
    name = value.strip().lower()
    if name not in CLASSES:
        raise ValueError(f"unsupported class {value!r}; expected one of {CLASSES}")
    return CLASSES.index(name)


def _csv_annotations(root: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    annotations: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for csv_path in root.rglob("annotations.csv"):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image = row.get("image") or row.get("filename")
                if not image:
                    raise ValueError(f"{csv_path} needs an image or filename column")
                annotations[Path(image).name].append((_normalise_class(row["class"]), float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])))
    return annotations


def _visdrone_annotations(source: Path) -> list[tuple[int, float, float, float, float]]:
    """Convert VisDrone's comma-separated annotations to normalized YOLO boxes.

    VisDrone categories 1 and 2 are pedestrian/person. Other categories are
    intentionally ignored because they are outside this SAR model's taxonomy.
    """
    annotation = source.with_suffix(".txt")
    if not annotation.exists():
        return []
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"unable to read image dimensions for {source}")
    height, width = image.shape[:2]
    boxes: list[tuple[int, float, float, float, float]] = []
    for raw_line in annotation.read_text(encoding="utf-8").splitlines():
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) < 6:
            continue
        left, top, box_width, box_height, score, category = map(float, values[:6])
        if int(category) not in (1, 2) or score <= 0:
            continue
        center_x = (left + box_width / 2) / width
        center_y = (top + box_height / 2) / height
        boxes.append((0, center_x, center_y, box_width / width, box_height / height))
    return boxes


def _write_label(path: Path, source: Path, csv_boxes: dict[str, list[tuple[int, float, float, float, float]]], label_loader: Callable[[Path], list[str]]) -> None:
    existing = source.with_suffix(".txt")
    if existing.exists():
        raw_lines = label_loader(source)
        lines = [line.strip() for line in raw_lines if line.strip()]
    else:
        lines = []
        for class_id, x1, y1, x2, y2 in csv_boxes.get(source.name, []):
            width, height = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
            lines.append(f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {width:.6f} {height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_dataset(raw_root: str | Path = "data/raw", output_root: str | Path = "data/processed", val_fraction: float = 0.2, seed: int = 42, visdrone: bool = False) -> Path:
    """Copy images and YOLO labels into deterministic train/val directories.

    Source data may already contain sibling YOLO ``.txt`` labels or one or more
    ``annotations.csv`` files with image, class, x1, y1, x2, y2 columns.
    Bounding boxes in CSV files are pixel coordinates and are converted to the
    normalized YOLO center/width/height representation.
    """
    source_root, destination = Path(raw_root), Path(output_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    images = sorted(path for path in source_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"no images found under {source_root}")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    csv_boxes = _csv_annotations(source_root)
    label_loader: Callable[[Path], list[str]] = lambda source: source.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    if visdrone:
        label_loader = lambda source: [f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}" for class_id, center_x, center_y, box_width, box_height in _visdrone_annotations(source)]
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_fraction)
    val_set = {path.resolve() for path in shuffled[:val_count]}
    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
    for image in images:
        split = "val" if image.resolve() in val_set else "train"
        target_image = destination / "images" / split / image.name
        target_label = destination / "labels" / split / f"{image.stem}.txt"
        shutil.copy2(image, target_image)
        _write_label(target_label, image, csv_boxes, label_loader)
    data_yaml = destination / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(destination.resolve()), "train": "images/train", "val": "images/val", "names": CLASSES, "nc": len(CLASSES)}, sort_keys=False), encoding="utf-8")
    return data_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visdrone", action="store_true", help="interpret sibling annotations as VisDrone CSV rows")
    args = parser.parse_args()
    print(f"Created {prepare_dataset(args.raw, args.output, args.val_fraction, args.seed, args.visdrone)}")
