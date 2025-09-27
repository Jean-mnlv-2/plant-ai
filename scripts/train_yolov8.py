from pathlib import Path
from typing import Optional

import typer


def main(
    data_yaml: Path = typer.Option(Path("datasets/yolov8/data.yaml"), exists=True),
    model: str = typer.Option("yolov8n.pt"),
    imgsz: int = typer.Option(640),
    epochs: int = typer.Option(50),
    batch: int = typer.Option(16),
    workers: int = typer.Option(4),
    device: str = typer.Option("auto"),
    project: Path = typer.Option(Path("runs/train")),
    name: str = typer.Option("plant_ai"),
    save_dir: Optional[Path] = typer.Option(Path("models")),
):
    from ultralytics import YOLO
    import torch

    project.mkdir(parents=True, exist_ok=True)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Normaliser device: sur CPU-only, 'auto' peut échouer côté Ultralytics
    resolved_device = device
    if device == "auto" and not torch.cuda.is_available():
        resolved_device = "cpu"

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=resolved_device,
        project=str(project),
        name=name,
        pretrained=True,
        exist_ok=True,
        verbose=True,
    )

    # Export du meilleur poids
    try:
        best = Path(project) / name / "weights" / "best.pt"
        if best.exists() and save_dir:
            (save_dir / "yolov8_best.pt").write_bytes(best.read_bytes())
            print(f"Modèle exporté vers {save_dir / 'yolov8_best.pt'}")
    except Exception as e:
        print(f"Avertissement export: {e}")


if __name__ == "__main__":
    typer.run(main)


