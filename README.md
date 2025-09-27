## Plant-AI

Plant-AI fournit un diagnostic intelligent des maladies/carences/ravageurs des plantes via détection d’images et deep learning.

### Vision
- Réduire les pertes agricoles et promouvoir une agriculture durable et accessible.

### Architecture technique
- Dataset: PlantDoc (VOC XML + CSV), enrichissable avec des images locales. Scripts pour conversion YOLOv8 et COCO.
- Modèle: YOLOv8 (Ultralytics).
- Pipeline: acquisition → prétraitement → détection/classification → fusion résultats → diagnostic textuel + visuel.
- Backend: API FastAPI pour l’inférence.

### Arborescence
```
Plant-AI/
  README.md
  requirements.txt
  models/
  data/
    raw/           # copie du dataset original
    cleaned/       # fichiers/images nettoyés
    TRAIN, TEST    # si original déplacé ici
  datasets/
    yolov8/        # images/labels split + data.yaml
    coco/          # instances_train/val/test.json
  reports/         # stats et anomalies
  scripts/
    analyze_dataset.py
    clean_filenames.py
    convert_to_yolov8.py
    convert_to_coco.py
    train_yolov8.py
  backend/
    app/
      main.py
```

### Pré-requis
- Python 3.9+
- GPU optionnel (CUDA) pour l’entraînement/inférence plus rapide

### Installation
```bash
# Option A (recommandé): utiliser l'environnement existant
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Option B: créer un nouvel environnement dédié
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

### Mise en conformité de l'arborescence (Windows PowerShell)
Exécutez ces commandes depuis le dossier `Plant-AI/` pour aligner votre structure avec les scripts:
```bash
# 1) Créer les dossiers attendus
New-Item -ItemType Directory -Force -Path "models","data/raw","data/cleaned","datasets/yolov8/images","datasets/yolov8/labels","datasets/coco","reports" | Out-Null

# 2) Déplacer le dataset PlantDoc brut dans data/raw
if (Test-Path "TRAIN") { Move-Item -Force TRAIN data/raw/ }
if (Test-Path "TEST")  { Move-Item -Force TEST  data/raw/ }
if (Test-Path "train_labels.csv") { Move-Item -Force train_labels.csv data/raw/ }
if (Test-Path "test_labels.csv")  { Move-Item -Force test_labels.csv  data/raw/ }

# 3) (Optionnel) Vérifier l'arborescence
Get-ChildItem -Recurse -Depth 2 data, datasets, models, reports | Select-Object FullName
```

### Mise en conformité (Invite de commandes Windows - cmd)
Si PowerShell ne fonctionne pas, ouvrez `cmd.exe` dans `Plant-AI/` et exécutez:
```bat
:: 1) Créer les dossiers attendus
mkdir models
mkdir data\raw
mkdir data\cleaned
mkdir datasets\yolov8\images
mkdir datasets\yolov8\labels
mkdir datasets\coco
mkdir reports

:: 2) Déplacer le dataset PlantDoc brut dans data\raw
if exist TRAIN move /Y TRAIN data\raw\
if exist TEST  move /Y TEST  data\raw\
if exist train_labels.csv move /Y train_labels.csv data\raw\
if exist test_labels.csv  move /Y test_labels.csv  data\raw\

:: 3) (Optionnel) Vérifier l'arborescence
dir /b /s data datasets models reports
```

### 1) Ingestion / Ajout de nouvelles données
Placez vos nouvelles images dans un dossier (ex. `data/new_images/`) et fournissez leurs annotations sous forme CSV (mêmes colonnes que `train_labels.csv`). Puis:
```bash
# Nettoyeur de noms de fichiers et mise à jour CSV
python scripts/clean_filenames.py --train_dir data/raw/TRAIN --test_dir data/raw/TEST --train_csv data/raw/train_labels.csv --test_csv data/raw/test_labels.csv --out_dir data/cleaned
```
Pour fusionner vos nouveaux CSV aux existants, concaténez-les puis relancez `clean_filenames.py` sur l’ensemble.

### 2) Analyse de qualité et statistiques
```bash
python scripts/analyze_dataset.py --root . --train_csv data/cleaned/train_labels.csv --test_csv data/cleaned/test_labels.csv --out reports
```
Génère `reports/stats.json` et `reports/invalid_annotations.json`.

### 3) Conversion en YOLOv8 (splits train/val/test)
```bash
python scripts/convert_to_yolov8.py --root . --train_dir data/cleaned/TRAIN --test_dir data/cleaned/TEST --train_csv data/cleaned/train_labels.csv --test_csv data/cleaned/test_labels.csv --out datasets/yolov8 --val_ratio 0.1
```
Produit `datasets/yolov8/images/{train,val,test}`, `datasets/yolov8/labels/{train,val,test}`, et `datasets/yolov8/data.yaml`.

### 4) Conversion en COCO
```bash
python scripts/convert_to_coco.py --root . --train_csv data/cleaned/train_labels.csv --test_csv data/cleaned/test_labels.csv --out datasets/coco --val_ratio 0.1
```
Produit `datasets/coco/instances_{train,val,test}.json`.

### 5) Entraînement YOLOv8
```bash
python scripts/train_yolov8.py --data-yaml datasets/yolov8/data.yaml --model yolov8n.pt --imgsz 640 --epochs 50 --batch 16 --device auto
```

⚠️ **Important pour la production:** Vérifiez que l'entraînement a bien placé le modèle entraîné dans `models/yolov8_best.pt` ou renommez/faites la copie du meilleur fichier final généré vers ce chemin.

### 6) Lancer le backend (inférence)
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Requête d’exemple (PowerShell):
```bash
Invoke-WebRequest -Uri http://localhost:8000/predict -Method Post -InFile path\to\image.jpg -ContentType 'application/octet-stream'
```

### Bonnes pratiques
- Vérifier `reports/invalid_annotations.json` et corriger les labels si nécessaire.
- Équilibrer les classes (ponderation/oversampling) en cas de déséquilibre marqué.
- Conserver un split `val` fixe pour comparer les métriques entre runs.




