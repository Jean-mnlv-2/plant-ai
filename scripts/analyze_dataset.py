import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def read_csv_labels(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def validate_bbox(r: Dict[str, str]) -> bool:
    try:
        w = int(r["width"]) if r["width"] else 0
        h = int(r["height"]) if r["height"] else 0
        xmin = int(float(r["xmin"]))
        ymin = int(float(r["ymin"]))
        xmax = int(float(r["xmax"]))
        ymax = int(float(r["ymax"]))
        if w <= 0 or h <= 0:
            return False
        if not (0 <= xmin < xmax <= w):
            return False
        if not (0 <= ymin < ymax <= h):
            return False
        return True
    except Exception:
        return False


def main(
    root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    train_csv: Path = typer.Option(..., exists=True),
    test_csv: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(Path("reports")),
):
    out.mkdir(parents=True, exist_ok=True)

    console.rule("Chargement des annotations")
    train_rows = read_csv_labels(train_csv)
    test_rows = read_csv_labels(test_csv)

    all_rows = train_rows + test_rows

    console.print(f"Lignes train: {len(train_rows)} | test: {len(test_rows)} | total: {len(all_rows)}")

    class_counter: Counter = Counter(r["class"] for r in all_rows)
    img_to_boxes: Dict[str, int] = defaultdict(int)
    invalid_rows: List[Dict[str, str]] = []

    for r in all_rows:
        img_to_boxes[r["filename"]] += 1
        if not validate_bbox(r):
            invalid_rows.append(r)

    # Table classes
    table = Table(title="Fréquences par classe")
    table.add_column("Classe")
    table.add_column("Instances", justify="right")
    for cls, cnt in class_counter.most_common():
        table.add_row(cls, str(cnt))
    console.print(table)

    # Statistiques multibox
    num_images = len(set(r["filename"] for r in all_rows))
    avg_boxes = sum(img_to_boxes.values()) / max(1, num_images)
    console.print(f"Images uniques: {num_images} | Boxes totales: {sum(img_to_boxes.values())} | Boxes/img moy.: {avg_boxes:.2f}")

    # Invalides
    console.print(f"Annotations invalides: {len(invalid_rows)}")
    (out / "invalid_annotations.json").write_text(json.dumps(invalid_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sauvegarde stats
    stats = {
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "num_images": num_images,
        "class_frequencies": class_counter,
        "avg_boxes_per_image": avg_boxes,
    }
    # Convert Counter to dict
    stats["class_frequencies"] = dict(stats["class_frequencies"])  # type: ignore
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    console.rule("Terminé")


if __name__ == "__main__":
    typer.run(main)




