"""Convert COCO JSON annotations to YOLO format (one .txt per image)."""

import json
import os
from pathlib import Path


def convert_coco_to_yolo(json_path: str, images_dir: str, labels_dir: str):
    """Convert COCO annotations to YOLO format.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    COCO category IDs (1-based) are mapped to 0-based class indices.
    """
    with open(json_path) as f:
        coco = json.load(f)

    # Build category id -> 0-based index mapping
    cat_id_to_idx = {}
    for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"])):
        cat_id_to_idx[cat["id"]] = i
        print(f"  Category '{cat['name']}' (id={cat['id']}) -> class {i}")

    # Build image id -> image info mapping
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(labels_dir, exist_ok=True)

    written = 0
    for img_id, img_info in img_map.items():
        w, h = img_info["width"], img_info["height"]
        fname = Path(img_info["file_name"]).stem + ".txt"
        label_path = os.path.join(labels_dir, fname)

        lines = []
        for ann in ann_by_img.get(img_id, []):
            cls_idx = cat_id_to_idx[ann["category_id"]]
            bx, by, bw, bh = ann["bbox"]  # COCO: top-left x, y, w, h
            # Convert to YOLO: center x, center y, w, h (normalized)
            x_center = (bx + bw / 2) / w
            y_center = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        written += 1

    print(f"  Wrote {written} label files to {labels_dir}")


if __name__ == "__main__":
    base = Path(__file__).parent

    print("Converting train annotations...")
    convert_coco_to_yolo(
        json_path=str(base / "train.json"),
        images_dir=str(base / "train"),
        labels_dir=str(base / "labels" / "train"),
    )

    print("Converting val annotations...")
    convert_coco_to_yolo(
        json_path=str(base / "val.json"),
        images_dir=str(base / "val"),
        labels_dir=str(base / "labels" / "val"),
    )

    print("Done! YOLO labels created.")
