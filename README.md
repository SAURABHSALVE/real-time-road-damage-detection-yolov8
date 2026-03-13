# Real-Time Road Damage Detection using YOLOv8

A deep learning pipeline for detecting road damage (potholes and cracks) in images and videos using YOLOv8.

## Features

- COCO to YOLO annotation format conversion
- YOLOv8 model training on custom road damage dataset
- Real-time inference on images and videos
- Bounding box visualization with class labels and confidence scores

## Classes

| ID | Class   | Color  |
|----|---------|--------|
| 0  | Pothole | Red    |
| 1  | Crack   | Orange |

## Project Structure

```
real-time-road-detection/
├── main.py                  # CLI entry point
├── train.py                 # Model training script
├── detect_image.py          # Image inference
├── detect_video.py          # Video inference
├── convert_coco_to_yolo.py  # COCO → YOLO label converter
├── data.yaml                # Dataset configuration
├── requirements.txt         # Dependencies
├── images/
│   ├── train/               # Training images
│   └── val/                 # Validation images
└── labels/
    ├── train/               # YOLO format labels (generated)
    └── val/
```

## Setup

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
pip install -r requirements.txt
```

## Usage

All commands can be run through `main.py` or the individual scripts.

### 1. Convert Annotations

Convert COCO JSON annotations to YOLO format:

```bash
python main.py convert
```

### 2. Train the Model

```bash
python main.py train --epochs 50 --batch 16 --imgsz 640
```

The best model weights will be saved to `runs/road_damage/weights/best.pt`.

### 3. Detect on an Image

```bash
python main.py detect-image --model runs/road_damage/weights/best.pt --source path/to/image.jpg
```

### 4. Detect on a Video

```bash
python main.py detect-video --model runs/road_damage/weights/best.pt --source path/to/video.mp4 --show
```

Options:
- `--output` — output directory (default: `output/`)
- `--conf` — confidence threshold (default: `0.25`)
- `--show` — display video in real-time while processing

## Training on Google Colab (GPU)

1. Zip and upload `images/` and `labels/` folders to Google Drive
2. In a Colab notebook (Runtime → GPU):

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics
!unzip -q /content/drive/MyDrive/road_damage_data.zip -d /content/dataset
```

3. Create `data.yaml` with Colab paths and train:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
)
```

4. Copy `runs/road_damage/weights/best.pt` back to your local project for inference.

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Python
