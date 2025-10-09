"""
Gestionnaire de dataset pour Plant-AI.
Permet la gestion complète du dataset depuis l'interface admin.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import zipfile
import csv
import subprocess
import sys
import io

from fastapi import HTTPException, UploadFile
from PIL import Image
import pandas as pd

from .settings import settings
from .database import db

logger = logging.getLogger("plant_ai.dataset_manager")


class DatasetManager:
    """Gestionnaire de dataset pour Plant-AI."""
    
    def __init__(self):
        self.dataset_dir = Path("datasets")
        self.raw_data_dir = Path("data/raw")
        self.cleaned_data_dir = Path("data/cleaned")
        self.yolo_dataset_dir = Path("datasets/yolov8")
        self.coco_dataset_dir = Path("datasets/coco")
        self.reports_dir = Path("reports")
        
        # Créer les répertoires nécessaires
        for dir_path in [self.dataset_dir, self.raw_data_dir, self.cleaned_data_dir, 
                        self.yolo_dataset_dir, self.coco_dataset_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def reset_dataset(self) -> Dict[str, Any]:
        """Réinitialise complètement le dataset."""
        try:
            logger.info("Réinitialisation du dataset...")
            
            # Supprimer tous les datasets existants
            for dir_path in [self.raw_data_dir, self.cleaned_data_dir, 
                           self.yolo_dataset_dir, self.coco_dataset_dir]:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            # Supprimer les rapports existants
            if self.reports_dir.exists():
                shutil.rmtree(self.reports_dir)
                self.reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Nettoyer la base de données des prédictions
            db.cleanup_old_data(days_to_keep=0)
            
            logger.info("Dataset réinitialisé avec succès")
            
            return {
                "status": "success",
                "message": "Dataset réinitialisé avec succès",
                "timestamp": datetime.utcnow().isoformat(),
                "directories_cleaned": [
                    str(self.raw_data_dir),
                    str(self.cleaned_data_dir),
                    str(self.yolo_dataset_dir),
                    str(self.coco_dataset_dir),
                    str(self.reports_dir)
                ]
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation du dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la réinitialisation: {str(e)}")
    
    async def upload_dataset(self, 
                           train_images: List[UploadFile],
                           test_images: List[UploadFile],
                           train_annotations: UploadFile,
                           test_annotations: UploadFile) -> Dict[str, Any]:
        """Upload et traitement d'un nouveau dataset."""
        try:
            logger.info("Upload d'un nouveau dataset...")
            
            # Créer les répertoires temporaires
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                train_dir = temp_path / "TRAIN"
                test_dir = temp_path / "TEST"
                train_dir.mkdir()
                test_dir.mkdir()
                
                # Sauvegarder les images d'entraînement
                train_files = []
                for img_file in train_images:
                    if img_file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                        raise HTTPException(status_code=400, detail=f"Type de fichier non supporté: {img_file.content_type}")
                    
                    # Valider l'image
                    img_data = await img_file.read()
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        img.verify()
                    except Exception:
                        raise HTTPException(status_code=400, detail=f"Image invalide: {img_file.filename}")
                    
                    # Sauvegarder l'image
                    img_path = train_dir / img_file.filename
                    img_path.write_bytes(img_data)
                    train_files.append(img_file.filename)
                
                # Sauvegarder les images de test
                test_files = []
                for img_file in test_images:
                    if img_file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                        raise HTTPException(status_code=400, detail=f"Type de fichier non supporté: {img_file.content_type}")
                    
                    # Valider l'image
                    img_data = await img_file.read()
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        img.verify()
                    except Exception:
                        raise HTTPException(status_code=400, detail=f"Image invalide: {img_file.filename}")
                    
                    # Sauvegarder l'image
                    img_path = test_dir / img_file.filename
                    img_path.write_bytes(img_data)
                    test_files.append(img_file.filename)
                
                # Sauvegarder les annotations
                train_annotations_data = await train_annotations.read()
                test_annotations_data = await test_annotations.read()
                
                train_csv_path = temp_path / "train_labels.csv"
                test_csv_path = temp_path / "test_labels.csv"
                
                train_csv_path.write_bytes(train_annotations_data)
                test_csv_path.write_bytes(test_annotations_data)
                
                # Valider les annotations
                await self._validate_annotations(train_csv_path, test_csv_path)
                
                # Copier vers les répertoires de données
                shutil.copytree(train_dir, self.raw_data_dir / "TRAIN", dirs_exist_ok=True)
                shutil.copytree(test_dir, self.raw_data_dir / "TEST", dirs_exist_ok=True)
                shutil.copy2(train_csv_path, self.raw_data_dir / "train_labels.csv")
                shutil.copy2(test_csv_path, self.raw_data_dir / "test_labels.csv")
                
                logger.info(f"Dataset uploadé: {len(train_files)} images train, {len(test_files)} images test")
                
                return {
                    "status": "success",
                    "message": "Dataset uploadé avec succès",
                    "timestamp": datetime.utcnow().isoformat(),
                    "train_images": len(train_files),
                    "test_images": len(test_files),
                    "train_files": train_files[:10],  # Limiter à 10 pour la réponse
                    "test_files": test_files[:10]
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'upload du dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(e)}")
    
    async def process_dataset(self) -> Dict[str, Any]:
        """Traite le dataset uploadé (nettoyage, conversion, validation)."""
        try:
            logger.info("Traitement du dataset...")
            
            # Étape 1: Nettoyage des noms de fichiers
            await self._clean_filenames()
            
            # Étape 2: Analyse du dataset
            analysis_results = await self._analyze_dataset()
            
            # Étape 3: Conversion vers YOLO
            yolo_results = await self._convert_to_yolo()
            
            # Étape 4: Conversion vers COCO
            coco_results = await self._convert_to_coco()
            
            # Étape 5: Vérification du dataset YOLO
            verification_results = await self._verify_yolo_dataset()
            
            logger.info("Dataset traité avec succès")
            
            return {
                "status": "success",
                "message": "Dataset traité avec succès",
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis_results,
                "yolo_conversion": yolo_results,
                "coco_conversion": coco_results,
                "verification": verification_results
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")
    
    async def train_model(self, 
                         epochs: int = 50,
                         batch_size: int = 16,
                         image_size: int = 640) -> Dict[str, Any]:
        """Entraîne un nouveau modèle avec le dataset traité."""
        try:
            logger.info(f"Entraînement du modèle: {epochs} époques, batch={batch_size}")
            
            # Vérifier que le dataset YOLO existe
            data_yaml = self.yolo_dataset_dir / "data.yaml"
            if not data_yaml.exists():
                raise HTTPException(status_code=400, detail="Dataset YOLO non trouvé. Traitez d'abord le dataset.")
            
            # Lancer l'entraînement
            result = await self._run_training(
                data_yaml=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size
            )
            
            logger.info("Entraînement terminé avec succès")
            
            return {
                "status": "success",
                "message": "Modèle entraîné avec succès",
                "timestamp": datetime.utcnow().isoformat(),
                "training_results": result
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement: {str(e)}")
    
    async def get_dataset_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du dataset."""
        try:
            status = {
                "raw_data": {
                    "exists": self.raw_data_dir.exists(),
                    "train_images": len(list((self.raw_data_dir / "TRAIN").glob("*.jpg"))) if (self.raw_data_dir / "TRAIN").exists() else 0,
                    "test_images": len(list((self.raw_data_dir / "TEST").glob("*.jpg"))) if (self.raw_data_dir / "TEST").exists() else 0,
                    "train_csv": (self.raw_data_dir / "train_labels.csv").exists(),
                    "test_csv": (self.raw_data_dir / "test_labels.csv").exists()
                },
                "cleaned_data": {
                    "exists": self.cleaned_data_dir.exists(),
                    "train_images": len(list((self.cleaned_data_dir / "TRAIN").glob("*.jpg"))) if (self.cleaned_data_dir / "TRAIN").exists() else 0,
                    "test_images": len(list((self.cleaned_data_dir / "TEST").glob("*.jpg"))) if (self.cleaned_data_dir / "TEST").exists() else 0
                },
                "yolo_dataset": {
                    "exists": self.yolo_dataset_dir.exists(),
                    "data_yaml": (self.yolo_dataset_dir / "data.yaml").exists(),
                    "train_images": len(list((self.yolo_dataset_dir / "images" / "train").glob("*.jpg"))) if (self.yolo_dataset_dir / "images" / "train").exists() else 0,
                    "val_images": len(list((self.yolo_dataset_dir / "images" / "val").glob("*.jpg"))) if (self.yolo_dataset_dir / "images" / "val").exists() else 0,
                    "test_images": len(list((self.yolo_dataset_dir / "images" / "test").glob("*.jpg"))) if (self.yolo_dataset_dir / "images" / "test").exists() else 0
                },
                "coco_dataset": {
                    "exists": self.coco_dataset_dir.exists(),
                    "train_json": (self.coco_dataset_dir / "instances_train.json").exists(),
                    "val_json": (self.coco_dataset_dir / "instances_val.json").exists(),
                    "test_json": (self.coco_dataset_dir / "instances_test.json").exists()
                },
                "model": {
                    "exists": settings.model_file.exists(),
                    "path": str(settings.model_file)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du statut: {str(e)}")
    
    async def _validate_annotations(self, train_csv: Path, test_csv: Path) -> None:
        """Valide les fichiers d'annotations CSV."""
        try:
            # Valider le CSV d'entraînement
            train_df = pd.read_csv(train_csv)
            required_columns = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
            
            if not all(col in train_df.columns for col in required_columns):
                raise HTTPException(status_code=400, detail="Colonnes manquantes dans train_labels.csv")
            
            # Valider le CSV de test
            test_df = pd.read_csv(test_csv)
            if not all(col in test_df.columns for col in required_columns):
                raise HTTPException(status_code=400, detail="Colonnes manquantes dans test_labels.csv")
            
            # Valider les coordonnées
            for df, name in [(train_df, "train"), (test_df, "test")]:
                for idx, row in df.iterrows():
                    try:
                        xmin, ymin, xmax, ymax = float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])
                        width, height = float(row["width"]), float(row["height"])
                        
                        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                            raise HTTPException(status_code=400, detail=f"Coordonnées invalides dans {name}_labels.csv ligne {idx+1}")
                    except (ValueError, TypeError):
                        raise HTTPException(status_code=400, detail=f"Valeurs non numériques dans {name}_labels.csv ligne {idx+1}")
            
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Fichier CSV vide")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Format CSV invalide")
    
    async def _clean_filenames(self) -> None:
        """Nettoie les noms de fichiers."""
        try:
            # Exécuter le script de nettoyage
            result = subprocess.run([
                sys.executable, "scripts/clean_filenames.py",
                "--train_dir", str(self.raw_data_dir / "TRAIN"),
                "--test_dir", str(self.raw_data_dir / "TEST"),
                "--train_csv", str(self.raw_data_dir / "train_labels.csv"),
                "--test_csv", str(self.raw_data_dir / "test_labels.csv"),
                "--out_dir", str(self.cleaned_data_dir)
            ], capture_output=True, text=True, check=True)
            
            logger.info("Nettoyage des noms de fichiers terminé")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors du nettoyage: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du nettoyage: {e.stderr}")
    
    async def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyse le dataset nettoyé."""
        try:
            # Exécuter l'analyse
            result = subprocess.run([
                sys.executable, "scripts/analyze_dataset.py",
                "--root", ".",
                "--train_csv", str(self.cleaned_data_dir / "train_labels.csv"),
                "--test_csv", str(self.cleaned_data_dir / "test_labels.csv"),
                "--out", str(self.reports_dir)
            ], capture_output=True, text=True, check=True)
            
            # Lire les résultats
            stats_file = self.reports_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
            else:
                stats = {"error": "Fichier de statistiques non généré"}
            
            logger.info("Analyse du dataset terminée")
            return stats
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de l'analyse: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {e.stderr}")
    
    async def _convert_to_yolo(self) -> Dict[str, Any]:
        """Convertit le dataset vers format YOLO."""
        try:
            # Exécuter la conversion YOLO
            result = subprocess.run([
                sys.executable, "scripts/convert_to_yolov8.py",
                "--root", ".",
                "--train_dir", str(self.cleaned_data_dir / "TRAIN"),
                "--test_dir", str(self.cleaned_data_dir / "TEST"),
                "--train_csv", str(self.cleaned_data_dir / "train_labels.csv"),
                "--test_csv", str(self.cleaned_data_dir / "test_labels.csv"),
                "--out", str(self.yolo_dataset_dir),
                "--val_ratio", "0.1"
            ], capture_output=True, text=True, check=True)
            
            logger.info("Conversion YOLO terminée")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de la conversion YOLO: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la conversion YOLO: {e.stderr}")
    
    async def _convert_to_coco(self) -> Dict[str, Any]:
        """Convertit le dataset vers format COCO."""
        try:
            # Exécuter la conversion COCO
            result = subprocess.run([
                sys.executable, "scripts/convert_to_coco.py",
                "--root", ".",
                "--train_csv", str(self.cleaned_data_dir / "train_labels.csv"),
                "--test_csv", str(self.cleaned_data_dir / "test_labels.csv"),
                "--out", str(self.coco_dataset_dir),
                "--val_ratio", "0.1"
            ], capture_output=True, text=True, check=True)
            
            logger.info("Conversion COCO terminée")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de la conversion COCO: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la conversion COCO: {e.stderr}")
    
    async def _verify_yolo_dataset(self) -> Dict[str, Any]:
        """Vérifie l'intégrité du dataset YOLO."""
        try:
            # Exécuter la vérification
            result = subprocess.run([
                sys.executable, "scripts/verify_yolo_dataset.py",
                "--dataset_dir", str(self.yolo_dataset_dir)
            ], capture_output=True, text=True, check=True)
            
            logger.info("Vérification YOLO terminée")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de la vérification YOLO: {e.stderr}")
            return {"status": "error", "output": e.stderr}
    
    async def _run_training(self, 
                           data_yaml: Path,
                           epochs: int,
                           batch_size: int,
                           image_size: int) -> Dict[str, Any]:
        """Lance l'entraînement du modèle."""
        try:
            # Exécuter l'entraînement
            result = subprocess.run([
                sys.executable, "scripts/train_yolov8.py",
                "--data-yaml", str(data_yaml),
                "--model", "yolov8n.pt",
                "--imgsz", str(image_size),
                "--epochs", str(epochs),
                "--batch", str(batch_size),
                "--device", "auto",
                "--save_dir", str(settings.models_dir)
            ], capture_output=True, text=True, check=True)
            
            logger.info("Entraînement terminé")
            return {"status": "success", "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de l'entraînement: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement: {e.stderr}")


# Instance globale du gestionnaire de dataset
dataset_manager = DatasetManager()
