"""
Real-Time Road Damage Detection using YOLOv8
=============================================
Main entry point — convert data, train, and run inference.

Usage:
    python main.py convert              # Convert COCO annotations to YOLO format
    python main.py train                # Train YOLOv8 on the dataset
    python main.py detect-image ...     # Detect damage in an image
    python main.py detect-video ...     # Detect damage in a video
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Road Damage Detection with YOLOv8"
    )
    sub = parser.add_subparsers(dest="command")

    # Convert
    sub.add_parser("convert", help="Convert COCO annotations to YOLO format")

    # Train
    train_p = sub.add_parser("train", help="Train YOLOv8 model")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch", type=int, default=16)
    train_p.add_argument("--imgsz", type=int, default=640)

    # Detect image
    img_p = sub.add_parser("detect-image", help="Detect damage in an image")
    img_p.add_argument("--model", required=True, help="Path to .pt model")
    img_p.add_argument("--source", required=True, help="Path to image")
    img_p.add_argument("--output", default="output")
    img_p.add_argument("--conf", type=float, default=0.25)

    # Detect video
    vid_p = sub.add_parser("detect-video", help="Detect damage in a video")
    vid_p.add_argument("--model", required=True, help="Path to .pt model")
    vid_p.add_argument("--source", required=True, help="Path to video")
    vid_p.add_argument("--output", default="output")
    vid_p.add_argument("--conf", type=float, default=0.25)
    vid_p.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if args.command == "convert":
        from convert_coco_to_yolo import convert_coco_to_yolo
        from pathlib import Path
        base = Path(__file__).parent
        print("Converting train annotations...")
        convert_coco_to_yolo(str(base / "train.json"), str(base / "train"),
                             str(base / "labels" / "train"))
        print("Converting val annotations...")
        convert_coco_to_yolo(str(base / "val.json"), str(base / "val"),
                             str(base / "labels" / "val"))
        print("Done!")

    elif args.command == "train":
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        model.train(
            data="data.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name="road_damage",
            project="runs",
            patience=10,
            save=True,
            plots=True,
        )
        metrics = model.val()
        print(f"\nmAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

    elif args.command == "detect-image":
        from detect_image import detect_image
        detect_image(args.model, args.source, args.output, args.conf)

    elif args.command == "detect-video":
        from detect_video import detect_video
        detect_video(args.model, args.source, args.output, args.conf, args.show)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
