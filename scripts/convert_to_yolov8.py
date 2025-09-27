import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import typer
from PIL import Image
import re


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_class_name(name: str) -> str:
    return name.strip()


def split_train_val(files: List[str], val_ratio: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(files)
    n_val = int(len(files) * val_ratio)
    return files[n_val:], files[:n_val]


def yolo_line(xmin: int, ymin: int, xmax: int, ymax: int, w: int, h: int, cls_id: int) -> str:
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def sanitize_fs_name(name: str, max_stem_len: int = 80) -> str:
    """Sanitize a filename for cross-platform FS compatibility (Windows reserved chars).
    Keeps extension, cleans base, enforces uniqueness length.
    """
    p = Path(name)
    # Replace Windows-invalid chars <>:"/\|?* and any non-alnum._-
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", p.stem)
    stem = re.sub(r"_+", "_", stem).strip("_.")
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len]
    return stem + p.suffix.lower()


def main(
    root: Path = typer.Option(..., exists=True),
    train_dir: Path = typer.Option(..., exists=True),
    test_dir: Path = typer.Option(..., exists=True),
    train_csv: Path = typer.Option(..., exists=True),
    test_csv: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    val_ratio: float = typer.Option(0.1, min=0.0, max=0.9),
    clean_out: bool = typer.Option(True, help="Purge les dossiers images/labels de sortie avant conversion"),
):
    out = out.resolve()
    images_train = out / "images" / "train"
    images_val = out / "images" / "val"
    images_test = out / "images" / "test"
    labels_train = out / "labels" / "train"
    labels_val = out / "labels" / "val"
    labels_test = out / "labels" / "test"
    # Optionally clean previous outputs to avoid stale files
    if clean_out and out.exists():
        for p in [images_train, images_val, images_test, labels_train, labels_val, labels_test]:
            if p.exists():
                for child in p.iterdir():
                    try:
                        if child.is_file() or child.is_symlink():
                            child.unlink(missing_ok=True)  # type: ignore[arg-type]
                        elif child.is_dir():
                            shutil.rmtree(child)
                    except Exception:
                        pass
    for p in [images_train, images_val, images_test, labels_train, labels_val, labels_test]:
        p.mkdir(parents=True, exist_ok=True)

    rows = read_rows(train_csv) + read_rows(test_csv)

    # Collect class names
    classes = sorted({normalize_class_name(r["class"]) for r in rows})
    class_to_id = {c: i for i, c in enumerate(classes)}

    # Group boxes per image
    img_to_boxes: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        img_to_boxes[r["filename"]].append(r)

    # Split train/val using TRAIN only
    train_files = sorted({r["filename"] for r in read_rows(train_csv)})
    train_split, val_split = split_train_val(train_files, val_ratio)
    test_files = sorted({r["filename"] for r in read_rows(test_csv)})

    # Helper to copy image and write label
    def process_split(file_list: List[str], src_dir: Path, dst_images: Path, dst_labels: Path):
        for fname in file_list:
            # Try direct path, then sanitized, then case-insensitive glob by stem
            src_img = (src_dir / fname).resolve()
            if not src_img.exists():
                safe_name = sanitize_fs_name(fname)
                alt = (src_dir / safe_name).resolve()
                if alt.exists():
                    src_img = alt
                else:
                    # fallback: match by stem ignoring case
                    target_stem = Path(sanitize_fs_name(fname)).stem.lower()
                    cand = [p for p in src_dir.iterdir() if p.is_file() and p.stem.lower() == target_stem]
                    if not cand:
                        continue
                    src_img = cand[0]
            # Destination filename should be sanitized to avoid invalid chars
            dst_base = sanitize_fs_name(Path(fname).name)
            dst_img = dst_images / dst_base
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if src_img != dst_img:
                dst_img.write_bytes(src_img.read_bytes())
            # Labels
            label_path = dst_labels / (Path(dst_base).stem + ".txt")
            label_lines: List[str] = []
            # Cache image size once per image
            img_size: Tuple[int, int] = Image.open(src_img).size
            for r in img_to_boxes.get(fname, []):
                # Parse width/height; fallback to actual image size if missing or zero
                try:
                    w_val = int(float(r["width"])) if r["width"] else 0
                except Exception:
                    w_val = 0
                try:
                    h_val = int(float(r["height"])) if r["height"] else 0
                except Exception:
                    h_val = 0
                w = w_val if w_val > 0 else img_size[0]
                h = h_val if h_val > 0 else img_size[1]
                xmin = int(float(r["xmin"]))
                ymin = int(float(r["ymin"]))
                xmax = int(float(r["xmax"]))
                ymax = int(float(r["ymax"]))
                cls_id = class_to_id[normalize_class_name(r["class"])]
                label_lines.append(yolo_line(xmin, ymin, xmax, ymax, w, h, cls_id))
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    # Execute
    process_split(train_split, train_dir, images_train, labels_train)
    process_split(val_split, train_dir, images_val, labels_val)
    process_split(test_files, test_dir, images_test, labels_test)

    # data.yaml
    data_yaml = (
        f"path: {out.as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "\n".join([f"  {i}: {c}" for i, c in enumerate(classes)]) + "\n"
    )
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print("Conversion YOLOv8 terminée.")


if __name__ == "__main__":
    typer.run(main)



