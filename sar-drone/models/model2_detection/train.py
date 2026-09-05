"""Fine-tune the configured YOLO model on the prepared SAR dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def train_model(config_path: str | Path = "config/config.yaml") -> Any:
	"""Train YOLO and write ``best.pt`` at the configured model path."""
	with Path(config_path).open(encoding="utf-8") as handle:
		config = yaml.safe_load(handle)

	from ultralytics import YOLO

	model_config = config["model"]
	output_weights = Path(config["paths"]["model_weights"])
	output_dir = output_weights.parent.parent
	model = YOLO(model_config["checkpoint"])
	results = model.train(
		data=config["paths"]["data_yaml"],
		epochs=model_config["train"]["epochs"],
		batch=model_config["train"]["batch"],
		imgsz=model_config["image_size"],
		workers=model_config["train"]["workers"],
		patience=model_config["train"]["patience"],
		project=str(output_dir.parent),
		name=output_dir.name,
		exist_ok=True,
		device=model_config["device"],
	)
	expected_weights = output_dir / "weights" / "best.pt"
	if not expected_weights.exists():
		raise FileNotFoundError(f"training completed without producing {expected_weights}")
	print(f"Training complete. Best weights: {expected_weights}")
	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config/config.yaml")
	train_model(parser.parse_args().config)
