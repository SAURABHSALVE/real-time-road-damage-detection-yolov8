"""Train YOLOv8 on the Road Damage Detection dataset."""

from ultralytics import YOLO


def train():
    # Load pretrained YOLOv8n (nano) — lightweight and fast
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="road_damage",
        project="runs",
        patience=10,
        save=True,
        plots=True,
    )

    # Validate on the val set
    metrics = model.val()
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return model


if __name__ == "__main__":
    train()
