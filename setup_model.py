#!/usr/bin/env python3
"""
Script de configuration du modèle pour Plant-AI.
Copie le modèle entraîné vers l'emplacement attendu par l'API.
"""

import shutil
from pathlib import Path


def setup_model():
    """Configure le modèle pour l'API."""
    print("🔧 Configuration du modèle Plant-AI")
    print("=" * 40)
    
    # Chemins
    source_model = Path("runs/train/plant_ai/weights/best.pt")
    target_model = Path("models/yolov8_best.pt")
    models_dir = Path("models")
    
    # Créer le dossier models s'il n'existe pas
    models_dir.mkdir(exist_ok=True)
    print(f"✅ Dossier models créé/vérifié: {models_dir}")
    
    # Vérifier si le modèle source existe
    if not source_model.exists():
        print(f"❌ Modèle source non trouvé: {source_model}")
        print("   Assurez-vous d'avoir entraîné le modèle avec:")
        print("   python scripts/train_yolov8.py --epochs 50")
        return False
    
    # Copier le modèle
    try:
        shutil.copy2(source_model, target_model)
        print(f"✅ Modèle copié: {source_model} → {target_model}")
        
        # Vérifier la taille du fichier
        size_mb = target_model.stat().st_size / (1024 * 1024)
        print(f"   Taille: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la copie: {e}")
        return False


def verify_setup():
    """Vérifie que la configuration est correcte."""
    print("\n🔍 Vérification de la configuration")
    print("-" * 30)
    
    # Vérifier le modèle
    model_path = Path("models/yolov8_best.pt")
    if model_path.exists():
        print(f"✅ Modèle trouvé: {model_path}")
    else:
        print(f"❌ Modèle manquant: {model_path}")
        return False
    
    # Vérifier la base de données
    db_path = Path("models/plant_ai.db")
    if db_path.exists():
        print(f"✅ Base de données trouvée: {db_path}")
    else:
        print(f"ℹ️  Base de données sera créée au premier démarrage")
    
    # Vérifier les dépendances
    try:
        import fastapi
        import ultralytics
        import sqlite3
        print("✅ Dépendances principales OK")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        return False
    
    return True


def main():
    """Fonction principale."""
    print("🌱 Configuration du système Plant-AI amélioré")
    print("=" * 50)
    
    # Configuration du modèle
    if setup_model():
        print("\n✅ Configuration du modèle réussie")
    else:
        print("\n❌ Échec de la configuration du modèle")
        return
    
    # Vérification
    if verify_setup():
        print("\n🎉 Configuration terminée avec succès !")
        print("\nPour démarrer l'API:")
        print("  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000")
        print("\nPour tester les améliorations:")
        print("  python test_improvements.py")
        print("  python demo_improvements.py")
    else:
        print("\n⚠️  Configuration incomplète, vérifiez les erreurs ci-dessus")


if __name__ == "__main__":
    main()
