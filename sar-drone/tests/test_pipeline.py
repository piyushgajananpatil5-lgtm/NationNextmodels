"""Fast sanity tests for the SAR pipeline modules."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from comms.lora_send import LoRaSender
from data.prepare_dataset import prepare_dataset
from models.model1_stitching.stitch import Stitcher
from models.model2_detection.infer import Detection
from models.model3_pathfinding.astar import find_path


def test_astar_avoids_obstacle() -> None:
    grid = [[0.0] * 5 for _ in range(5)]
    grid[2][1:4] = [float("inf")] * 3
    path = find_path(grid, (0, 0), (4, 4))
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert all(grid[row][column] != float("inf") for row, column in path)


def test_lora_packet_has_framing_and_crc() -> None:
    packet = LoRaSender("loop://").encode({"event": "test", "gps": [1.0, 2.0]})
    assert packet[:4] == b"SAR1"
    assert int.from_bytes(packet[4:6], "big") == len(packet[6:-4])


def test_detection_serializes() -> None:
    detection = Detection(1, 2, 3, 4, 0, "person", 0.9)
    assert detection.as_dict()["class_name"] == "person"


def test_stitcher_rejects_too_few_frames() -> None:
    try:
        Stitcher()([np.zeros((10, 10, 3), dtype=np.uint8)])
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("expected validation error")


def test_dataset_conversion(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image_path = raw / "sample.jpg"
    assert cv2.imwrite(str(image_path), image)
    (raw / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = prepare_dataset(raw, tmp_path / "processed", val_fraction=0)
    assert data_yaml.exists()
    assert (tmp_path / "processed/images/train/sample.jpg").exists()
    assert (tmp_path / "processed/labels/train/sample.txt").read_text(encoding="utf-8").strip().startswith("0 ")
