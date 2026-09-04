"""Export YOLO weights to an FP16 TensorRT engine for Jetson."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def export_engine(config_path: str | Path = "config/config.yaml") -> Any:
    """Build a TensorRT FP16 engine using Ultralytics' Jetson-compatible exporter."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    from ultralytics import YOLO

    weights = config["paths"]["model_weights"]
    model = YOLO(weights)
    result = model.export(
        format="engine",
        half=True,
        imgsz=config["model"]["image_size"],
        device=config["model"]["device"],
        workspace=4,
    )
    print(f"TensorRT FP16 engine exported from {weights}: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    export_engine(parser.parse_args().config)
