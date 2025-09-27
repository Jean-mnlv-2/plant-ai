from pathlib import Path
from typing import List

import typer


def read_label_file(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(
    dataset_dir: Path = typer.Option(Path("datasets/yolov8"), exists=True, file_okay=False, dir_okay=True),
):
    problems = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            problems.append(f"Split manquant: {split}")
            continue
        for img in img_dir.rglob("*.jpg"):
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                problems.append(f"Label manquant pour {split}: {img.name}")
                continue
            lines = read_label_file(lbl)
            for i, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"Format invalide {lbl} ligne {i}: '{line}'")
                    continue
                try:
                    cls = int(parts[0])
                    vals = list(map(float, parts[1:]))
                    if not (0 <= vals[0] <= 1 and 0 <= vals[1] <= 1 and 0 < vals[2] <= 1 and 0 < vals[3] <= 1):
                        problems.append(f"Valeurs hors bornes {lbl} ligne {i}: {vals}")
                except Exception:
                    problems.append(f"Valeurs non numériques {lbl} ligne {i}: '{line}'")

    if problems:
        (dataset_dir / "verification_report.txt").write_text("\n".join(problems), encoding="utf-8")
        print(f"Problèmes détectés: {len(problems)}. Voir {dataset_dir / 'verification_report.txt'}")
    else:
        print("Dataset YOLOv8 vérifié: aucun problème détecté.")


if __name__ == "__main__":
    typer.run(main)


