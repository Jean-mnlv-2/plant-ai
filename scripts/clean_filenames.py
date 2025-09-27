import csv
import re
from pathlib import Path
from typing import Dict

import typer


def sanitize(name: str, max_stem_len: int = 80) -> str:
    # Conserver extension, nettoyer base et limiter la longueur (Windows MAX_PATH)
    p = Path(name)
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", p.stem)
    stem = re.sub(r"_+", "_", stem).strip("_.")
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len]
    return stem + p.suffix.lower()


def rewrite_csv(csv_in: Path, mapping: Dict[str, str], csv_out: Path):
    rows = list(csv.DictReader(csv_in.open("r", encoding="utf-8")))
    fieldnames = rows[0].keys() if rows else ["filename","width","height","class","xmin","ymin","xmax","ymax"]
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            fn = r["filename"]
            if fn in mapping:
                r["filename"] = mapping[fn]
            w.writerow(r)


def main(
    train_dir: Path = typer.Option(..., exists=True),
    test_dir: Path = typer.Option(..., exists=True),
    train_csv: Path = typer.Option(..., exists=True),
    test_csv: Path = typer.Option(..., exists=True),
    out_dir: Path = typer.Option(Path("data/cleaned")),
):
    out_images_train = out_dir / "TRAIN"
    out_images_test = out_dir / "TEST"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_train.mkdir(parents=True, exist_ok=True)
    out_images_test.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}

    def process_folder(src: Path, dst: Path):
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                # Générer un nom propre et unique
                base_name = sanitize(p.name)
                candidate = base_name
                suffix_idx = 1
                while (dst / candidate).exists():
                    # Ajouter un suffixe numérique si collision
                    stem, ext = Path(base_name).stem, Path(base_name).suffix
                    candidate = f"{stem}_{suffix_idx}{ext}"
                    suffix_idx += 1
                mapping[p.name] = candidate
                (dst / candidate).write_bytes(p.read_bytes())

    process_folder(train_dir, out_images_train)
    process_folder(test_dir, out_images_test)

    rewrite_csv(train_csv, mapping, out_dir / "train_labels.csv")
    rewrite_csv(test_csv, mapping, out_dir / "test_labels.csv")

    print("Nettoyage terminé. Fichiers mis à jour dans", out_dir)


if __name__ == "__main__":
    typer.run(main)


