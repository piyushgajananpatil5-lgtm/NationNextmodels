"""Evaluate trained YOLO weights and write a JSON accuracy report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def evaluate(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Run model.val(), print metrics, and save ``results.json``."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    from ultralytics import YOLO

    model = YOLO(config["paths"]["model_weights"])
    metrics = model.val(data=config["paths"]["data_yaml"], imgsz=config["model"]["image_size"], device=config["model"]["device"], verbose=False)
    box = metrics.box
    names = config["model"]["classes"]
    per_class = {}
    for index, name in enumerate(names):
        precision = float(box.p[index]) if index < len(box.p) else 0.0
        recall = float(box.r[index]) if index < len(box.r) else 0.0
        per_class[name] = {"precision": precision, "recall": recall}
    report: dict[str, Any] = {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
    }
    output_dir = Path(config["paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"mAP50: {report['map50']:.4f} | mAP50-95: {report['map50_95']:.4f}")
    print(f"Precision: {report['precision']:.4f} | Recall: {report['recall']:.4f}")
    for name, values in per_class.items():
        print(f"{name}: precision={values['precision']:.4f}, recall={values['recall']:.4f}")
    print(f"Saved accuracy report to {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    evaluate(parser.parse_args().config)
