from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Plant-AI Backend Settings."""
    
    # Application
    app_name: str = "Plant-AI Backend"
    app_version: str = "0.1.0"

    # Paths - Models and data
    models_dir: Path = Field(default=Path("models"))
    model_file: Path = Field(default=Path("models/yolov8_best.pt"))
    jwt_secret_file: Path = Field(default=Path("models/.jwt_secret"))
    db_path: Path = Field(default=Path("models/plant_ai.db"))
    uploads_dir: Path = Field(default=Path("data/uploads"))
    reports_dir: Path = Field(default=Path("reports"))

    # CORS for production - adjust these origins for production deployment
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",   # React dev server
            "http://localhost:5173",   # Vite dev server  
            "http://localhost:8080",  # Vue dev server
            "https://yourdomain.com"   # Replace with your production domains
        ]
    )

    # Upload constraints
    max_upload_bytes: int = Field(default=10 * 1024 * 1024)  # 10 MB
    allowed_content_types: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    )

    # Inference thresholds & quality
    confidence_threshold_low: float = Field(default=0.35, ge=0.0, le=1.0)
    confidence_threshold_high: float = Field(default=0.7, ge=0.0, le=1.0)
    blur_variance_threshold: float = Field(default=100.0, ge=0.0)
    enable_quality_checks: bool = Field(default=True)
    enable_uncertainty_routing: bool = Field(default=True)

    # JWT Authentication
    jwt_ttl_seconds: int = Field(default=8 * 3600)  # 8 hours

    # Domain knowledge - mapping classes to recommendations
    class_to_recommendations: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "rust": [
                "Appliquer un fongicide adapté", 
                "Rotation culturale", 
                "Retirer feuilles infectées"
            ],
            "blight": [
                "Éliminer débris végétaux", 
                "Fongicide préventif", 
                "Améliorer l'aération"
            ],
            "leaf_spot": [
                "Éviter excès d'arrosage", 
                "Fungicide si nécessaire", 
                "Espacer les plants"
            ]
        }
    )

    # File saving options (optional)
    save_uploaded_images: bool = Field(default=False)
    save_reports: bool = Field(default=False)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log_date_format: str = Field(default="%Y-%m-%d %H:%M:%S")

    model_config = {
        "env_prefix": "PLANT_AI_",
        "env_file": ".env",
        "protected_namespaces": ("settings_",)
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True) 
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(__name__)
        # Ensure model path exists or check if we need to ask user to train it
        if not self.model_file.exists():
            logger.warning(f"Model file {self.model_file} not found. Train a model first.")
        # Ensure models dir exists for placing trained models
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)


# Create settings instance
settings = Settings()


