"""
Script pour créer un modèle YOLO de test simple.
"""
import sys
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def create_test_model():
    """Créer un modèle YOLO de test."""
    print("Creating test YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        # Créer un modèle YOLOv8n simple
        model = YOLO('yolov8n.pt')
        
        # Sauvegarder le modèle
        model_path = Path("models/yolov8_best.pt")
        model_path.parent.mkdir(exist_ok=True)
        
        # Copier le modèle pré-entraîné
        import shutil
        shutil.copy("yolov8n.pt", str(model_path))
        
        print(f"Test model created at: {model_path}")
        return True
        
    except Exception as e:
        print(f"Error creating test model: {e}")
        return False

if __name__ == "__main__":
    create_test_model()


