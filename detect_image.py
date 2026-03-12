"""Detect road damage in images using a trained YOLOv8 model."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# Colors for each class (BGR)
COLORS = {
    0: (0, 0, 255),    # Pothole - Red
    1: (0, 165, 255),  # Crack - Orange
}
CLASS_NAMES = {0: "Pothole", 1: "Crack"}


def detect_image(model_path: str, image_path: str, output_dir: str = "output",
                 conf: float = 0.25):
    """Run detection on a single image and save the annotated result."""
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    results = model(img, conf=conf)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        color = COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, cls_id)} {confidence:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / Path(image_path).name)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path} ({len(results.boxes)} detections)")

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect road damage in an image")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    detect_image(args.model, args.source, args.output, args.conf)
