"""Run camera capture, detection, periodic stitching, A*, and LoRa telemetry."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml

from comms.lora_send import LoRaSender
from models.model1_stitching.stitch import Stitcher
from models.model2_detection.infer import Detection, Detector
from models.model3_pathfinding.astar import detections_to_cost_grid, find_path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def detection_payload(detections: list[Detection]) -> list[dict[str, Any]]:
    return [detection.as_dict() for detection in detections]


def run(config_path: str | Path = "config/config.yaml") -> None:
    """Run until the camera ends or the user presses Q."""
    config = load_config(config_path)
    model = config["model"]
    pipeline = config["pipeline"]
    Path(config["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    detector = Detector(config["paths"]["exported_engine"], model["confidence"], model["iou"], model["device"])
    stitcher = Stitcher()
    source: str | int = config["paths"]["input_source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise OSError(f"unable to open camera/video source {source!r}")
    sender = LoRaSender(**config["serial"])
    frame_buffer: list[Any] = []
    frame_index = 0
    try:
        with sender:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                frame_buffer.append(frame)
                detections = detector.predict(frame)
                cost_grid = detections_to_cost_grid(detections, (32, 32))
                cells = find_path(cost_grid, (0, 0), (31, 31))
                waypoints = [[pipeline["gps_origin"][0], pipeline["gps_origin"][1], cell[0], cell[1]] for cell in cells]
                if frame_index % config["pipeline"]["send_every_n_frames"] == 0:
                    sender.send_event("detections", pipeline["gps_origin"], detection_payload(detections), waypoints)
                interval = config["stitching"]["interval_frames"]
                if frame_index % interval == 0 and len(frame_buffer) >= config["stitching"]["min_frames"]:
                    try:
                        panorama = stitcher(frame_buffer)
                        cv2.imwrite("results/latest_panorama.jpg", panorama)
                    except RuntimeError:
                        pass
                    frame_buffer.clear()
                cv2.imshow("SAR drone", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    run(parser.parse_args().config)
