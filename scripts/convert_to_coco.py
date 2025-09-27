import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import typer


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_class_name(name: str) -> str:
    return name.strip()


def split_train_val(files: List[str], val_ratio: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_ratio)
    return files[n_val:], files[:n_val]


def coco_anns(
    rows: List[Dict[str, str]],
    image_set: List[str],
    categories: List[Dict],
    start_ann_id: int = 1,
):
    class_to_id = {c["name"]: c["id"] for c in categories}
    images = []
    annotations = []
    img_id_map = {}
    next_img_id = 1
    ann_id = start_ann_id

    grouped = defaultdict(list)
    for r in rows:
        if r["filename"] in image_set:
            grouped[r["filename"]].append(r)

    for fname, rs in grouped.items():
        width = int(rs[0]["width"]) if rs[0]["width"] else 0
        height = int(rs[0]["height"]) if rs[0]["height"] else 0
        img = {"id": next_img_id, "file_name": fname, "width": width, "height": height}
        img_id_map[fname] = next_img_id
        images.append(img)
        next_img_id += 1

        for r in rs:
            xmin = int(float(r["xmin"]))
            ymin = int(float(r["ymin"]))
            xmax = int(float(r["xmax"]))
            ymax = int(float(r["ymax"]))
            w = xmax - xmin
            h = ymax - ymin
            ann = {
                "id": ann_id,
                "image_id": img["id"],
                "category_id": class_to_id[normalize_class_name(r["class"])],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            annotations.append(ann)
            ann_id += 1

    return images, annotations, ann_id


def main(
    root: Path = typer.Option(..., exists=True),
    train_csv: Path = typer.Option(..., exists=True),
    test_csv: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    val_ratio: float = typer.Option(0.1, min=0.0, max=0.9),
):
    out.mkdir(parents=True, exist_ok=True)

    train_rows = read_rows(train_csv)
    test_rows = read_rows(test_csv)
    rows = train_rows + test_rows

    classes = sorted({normalize_class_name(r["class"]) for r in rows})
    categories = [{"id": i + 1, "name": c, "supercategory": "plant"} for i, c in enumerate(classes)]

    train_files = sorted({r["filename"] for r in train_rows})
    train_split, val_split = split_train_val(train_files, val_ratio)
    test_files = sorted({r["filename"] for r in test_rows})

    datasets = {
        "train": train_split,
        "val": val_split,
        "test": test_files,
    }

    next_ann_id = 1
    for split_name, image_set in datasets.items():
        images, annotations, next_ann_id = coco_anns(rows, image_set, categories, start_ann_id=next_ann_id)
        coco = {"images": images, "annotations": annotations, "categories": categories}
        (out / f"instances_{split_name}.json").write_text(json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Conversion COCO terminée.")


if __name__ == "__main__":
    typer.run(main)




