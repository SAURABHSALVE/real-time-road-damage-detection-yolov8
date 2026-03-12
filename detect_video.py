"""Detect road damage in videos using a trained YOLOv8 model."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

COLORS = {
    0: (0, 0, 255),    # Pothole - Red
    1: (0, 165, 255),  # Crack - Orange
}
CLASS_NAMES = {0: "Pothole", 1: "Crack"}


def detect_video(model_path: str, video_path: str, output_dir: str = "output",
                 conf: float = 0.25, show: bool = False):
    """Run detection on a video, annotate frames, and save the output video."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / Path(video_path).name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_num = 0
    print(f"Processing video: {video_path} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model(frame, conf=conf, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            color = COLORS.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES.get(cls_id, cls_id)} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Frame counter overlay
        cv2.putText(frame, f"Frame {frame_num}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(frame)

        if show:
            cv2.imshow("Road Damage Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Saved: {out_path} ({frame_num} frames processed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect road damage in a video")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display video while processing")
    args = parser.parse_args()

    detect_video(args.model, args.source, args.output, args.conf, args.show)
