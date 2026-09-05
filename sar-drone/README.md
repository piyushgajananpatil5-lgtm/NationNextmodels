# SAR Drone Pipeline

A modular search-and-rescue drone pipeline for NVIDIA Jetson Orin Nano running JetPack 6. It combines OpenCV frame stitching, YOLO11 detection, hazard-aware A*, and compact LoRa telemetry.

## Jetson installation

JetPack 6 supplies CUDA, TensorRT, and the Jetson-specific PyTorch build. Flash the board with NVIDIA SDK Manager first, then from this directory run:

```bash
chmod +x setup_jetson.sh
export JETSON_TORCH_WHEEL_URL='https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl'
./setup_jetson.sh
source .venv/bin/activate
```

Set `JETSON_TORCH_WHEEL_URL` to the NVIDIA wheel matching the board's JetPack release. Set `JETSON_TORCHVISION_WHEEL_URL` as well when a matching torchvision wheel is required. The setup script installs these wheels before the pinned application dependencies, uses system site packages for the apt-provided TensorRT bindings, and fails if CUDA is not available through PyTorch.

## Dataset

Read [data/datasets.md](data/datasets.md), place licensed HERIDAL, SARD, and VisDrone files under `data/raw`, and convert them:

```bash
python data/prepare_dataset.py --raw data/raw --output data/processed
```

The converter writes `data/processed/data.yaml`, images, and YOLO labels with the four classes `person`, `fire`, `debris`, and `water`.

## Train and evaluate

```bash
python models/model2_detection/train.py --config config/config.yaml
python models/model2_detection/evaluate.py --config config/config.yaml
```

Training starts from `yolo11n.pt` and writes `runs/detect/sar_yolo11n/weights/best.pt`. Validation prints mAP50, mAP50-95, precision, recall, and per-class precision/recall. The JSON report is saved to `results/results.json`.

## TensorRT export

On the Jetson, after training:

```bash
python models/model2_detection/export_tensorrt.py --config config/config.yaml
```

The Ultralytics exporter builds an FP16 `.engine` using the Jetson TensorRT runtime. The configured output is `runs/detect/sar_yolo11n/weights/best.engine`; TensorRT engines are hardware-specific and should be rebuilt on the target Jetson.

## Full pipeline

Edit `config/config.yaml` for the camera/video source, GPS origin, serial port, thresholds, paths, and frame intervals. Then run:

```bash
python pipeline/run_pipeline.py --config config/config.yaml
```

Press `q` to stop. The process detects every frame, periodically writes `results/latest_panorama.jpg`, computes a path over the integration cost grid, and sends detections, GPS, and waypoints through the configured LoRa serial device. `detections_to_cost_grid` in `models/model3_pathfinding/astar.py` is the single intended integration point for your camera/GPS hazard projection.

## Tests

Run the lightweight module checks from the project root:

```bash
python -m pytest -q
```

The tests cover A*, packet framing, dataset conversion, detection record serialization, and stitching input validation without requiring a camera, LoRa device, or trained weights.
