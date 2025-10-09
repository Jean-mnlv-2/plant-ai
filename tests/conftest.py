"""
Configuration globale pour les tests pytest.
"""
import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Ajouter le répertoire parent au path pour les imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.main import app
from backend.app.database import Database
from backend.app.settings import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Créer un event loop pour toute la session de tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Configuration de test avec base de données temporaire."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        settings = Settings(
            db_path=temp_path / "test.db",
            model_file=temp_path / "test_model.pt",
            models_dir=temp_path / "models",
            uploads_dir=temp_path / "uploads",
            reports_dir=temp_path / "reports",
            log_level="DEBUG"
        )
        yield settings


@pytest.fixture
def test_db(test_settings):
    """Base de données de test."""
    db = Database(test_settings.db_path)
    yield db


@pytest.fixture
def test_client(test_settings):
    """Client de test FastAPI."""
    with patch('backend.app.main.settings', test_settings):
        with TestClient(app) as client:
            yield client


@pytest.fixture
def sample_image():
    """Image de test générée programmatiquement."""
    # Créer une image RGB 640x640 avec du bruit
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Sauvegarder dans un BytesIO
    import io
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


@pytest.fixture
def sample_image_file(sample_image):
    """Fichier image de test."""
    import io
    return io.BytesIO(sample_image)


@pytest.fixture
def mock_model():
    """Mock du modèle YOLO pour les tests."""
    mock = Mock()
    mock.names = {0: "test_disease", 1: "healthy_plant"}
    
    # Mock des résultats de prédiction
    mock_result = Mock()
    mock_result.boxes = Mock()
    mock_result.boxes.xyxy = Mock()
    mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]])
    mock_result.boxes.conf = Mock()
    mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.85])
    mock_result.boxes.cls = Mock()
    mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    
    mock.predict.return_value = [mock_result]
    
    return mock


@pytest.fixture
def sample_predictions():
    """Prédictions de test."""
    return [
        {
            "class_name": "test_disease",
            "confidence": 0.85,
            "bbox": [100, 100, 200, 200],
            "recommendations": ["Test recommendation"]
        }
    ]


@pytest.fixture
def sample_user_data():
    """Données utilisateur de test."""
    return {
        "user_id": "test_user",
        "image_filename": "test_image.jpg",
        "image_width": 640,
        "image_height": 640,
        "predictions": [
            {
                "class_name": "test_disease",
                "confidence": 0.85,
                "bbox": [100, 100, 200, 200],
                "recommendations": ["Test recommendation"]
            }
        ],
        "processing_time_ms": 150,
        "confidence_avg": 0.85,
        "num_detections": 1
    }


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Nettoyage automatique des fichiers de test."""
    yield
    # Nettoyage après chaque test
    pass


# Marqueurs personnalisés
pytestmark = pytest.mark.asyncio



