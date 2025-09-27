# Commandes CMD pour Plant-AI Backend - Production

## 1. Installation et mise en place initial

```cmd
:: Activer l'environnement virtuel
call venv\Scripts\activate.bat

:: Installer toutes les dépendances
pip install -r requirements.txt

:: Vérifier la structure modèles
mkdir models 2>nul || echo Dossier models existe
```

## 2. Validation de la configuration backend

```cmd
:: Tester le chargement des settings
python -c "from backend.app.settings import settings; print('✅ Settings loaded')"

:: Démarrage API backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3. Check santé API
```cmd
:: Test des endpoints health check
curl http://localhost:8000/health
```

## 4. Placer le modèle formé
```cmd
:: Copier le modèle final du training vers models/
copy yolov8n.pt models\yolov8_best.pt

:: Vérifier le placement du modèle
dir models\yolov8_best.pt
```

## 5. Test complet backend
```cmd
:: Lancer tests unitaires
pytest -q tests\

:: Test maquette API health
call curl http://localhost:8000/health
```

## Configuration CMD spécifique

Les variables d'environnement pour personnaliser peuvent être setter via :
```cmd
:: Override par variable (optionnel)
set PLANT_AI_ALLOWED_ORIGINS=["*.yourdomain.com"]
set PLANT_AI_MAX_UPLOAD_BYTES=20971520
```

Le backend utilise maintenant `cmd` de façon native avec toute config depuis `settings.py`.

---
*Commands pour production deployment; l'environnement virtuel est nécessaire déjà en place.*
