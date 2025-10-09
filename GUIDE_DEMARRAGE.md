# 🚀 Guide de Démarrage - Plant-AI

## 📋 Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)

## 🔧 Installation et Configuration

### 1. **Installation des Dépendances**
```bash
pip install -r requirements.txt
```

### 2. **Configuration de l'Environnement**
```bash
# Copier le fichier d'exemple
cp env.example .env

# Éditer les paramètres si nécessaire
# (Les valeurs par défaut fonctionnent pour le développement local)
```

### 3. **Démarrage du Serveur**
```bash
# Option 1: Script de démarrage (recommandé)
python start_api.py

# Option 2: Commande directe
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Le serveur démarre sur : **http://localhost:8000**

---

## 👤 Création d'un Administrateur Local

### **Script Automatique (Recommandé)**
```bash
python create_admin.py
```

### **Création Manuelle via API**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Admin Plant-AI",
    "email": "admin@plant-ai.local",
    "password": "Admin123!",
    "country": "France",
    "role": "admin"
  }'
```

### **Identifiants Admin par Défaut**
- **Email** : `admin@plant-ai.local`
- **Mot de passe** : `Admin123!`
- **Rôle** : `admin`

---

## 📚 Accès à Swagger UI

### **URLs d'Accès**
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI JSON** : http://localhost:8000/openapi.json

### **Authentification dans Swagger**
1. Ouvrir http://localhost:8000/docs
2. Cliquer sur le bouton **"Authorize"** 🔒
3. Utiliser l'endpoint `/api/v1/auth/login` pour obtenir un token
4. Entrer le token dans le format : `Bearer <votre_token>`

### **Test Rapide dans Swagger**
1. **POST /api/v1/auth/login** - Se connecter
2. **GET /api/v1/users/profile** - Voir son profil
3. **GET /api/v1/weather** - Tester la météo
4. **GET /api/v1/diseases** - Rechercher des maladies

---

## 🧪 Tests et Validation

### **Test Automatique Complet**
```bash
python quick_test.py
```

### **Test Manuel des Endpoints**
```bash
# 1. Test de santé
curl http://localhost:8000/health

# 2. Connexion admin
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@plant-ai.local", "password": "Admin123!"}'

# 3. Test météo
curl -X GET "http://localhost:8000/api/v1/weather?lat=48.8566&lng=2.3522" \
  -H "Authorization: Bearer <votre_token>"

# 4. Recherche de maladies
curl -X GET "http://localhost:8000/api/v1/diseases?search=mildiou" \
  -H "Authorization: Bearer <votre_token>"
```

---

## 🎯 Utilisation des APIs

### **1. Authentification**
```bash
# Inscription
POST /api/v1/auth/register

# Connexion
POST /api/v1/auth/login

# Renouvellement de token
POST /api/v1/auth/refresh
```

### **2. Diagnostic IA**
```bash
# Diagnostic d'images
POST /api/v1/diagnose
# Body: images (base64), location, plantType, additionalInfo
```

### **3. Gestion des Diagnostics**
```bash
# Historique
GET /api/v1/diagnostics?limit=10&offset=0

# Diagnostic spécifique
GET /api/v1/diagnostics/{id}

# Synchronisation
POST /api/v1/diagnostics/sync
```

### **4. Météo Agricole**
```bash
# Données météo
GET /api/v1/weather?lat=48.8566&lng=2.3522&includeForecast=true
```

### **5. Base de Connaissances**
```bash
# Recherche de maladies
GET /api/v1/diseases?search=mildiou&category=diseases

# Détails d'une maladie
GET /api/v1/diseases/mildiou-tomate
```

### **6. Gestion Utilisateurs**
```bash
# Profil utilisateur
GET /api/v1/users/profile
```

---

## 🔧 Administration

### **Dashboard Admin**
- **URL** : http://localhost:8000/admin/
- **Fonctionnalités** :
  - Statistiques de performance
  - Gestion des utilisateurs
  - Monitoring des diagnostics
  - Gestion du dataset

### **Endpoints d'Administration**
```bash
# Santé détaillée
GET /admin/health-detailed

# Nettoyage des données
POST /admin/cleanup

# Gestion du dataset
POST /admin/dataset/upload
POST /admin/dataset/process
POST /admin/dataset/train
```

---

## 🐛 Dépannage

### **Problèmes Courants**

#### **1. Port déjà utilisé**
```bash
# Changer le port dans .env
PLANT_AI_PORT=8001

# Ou tuer le processus
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

#### **2. Erreur de dépendances**
```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

#### **3. Erreur de base de données**
```bash
# Supprimer la base de données pour recommencer
rm data/plant_ai.db
# Le fichier sera recréé automatiquement
```

#### **4. Erreur de modèle**
```bash
# Vérifier que le modèle existe
ls models/
# Si absent, télécharger un modèle YOLOv8
```

### **Logs et Debug**
```bash
# Activer les logs détaillés
export PLANT_AI_LOG_LEVEL=DEBUG

# Voir les logs en temps réel
tail -f logs/plant_ai.log
```

---

## 📊 Monitoring

### **Métriques Disponibles**
- **Performance** : Temps de réponse, taux d'erreur
- **Utilisation** : Nombre de requêtes, utilisateurs actifs
- **Modèle IA** : Prédictions, confiance moyenne
- **Système** : CPU, mémoire, espace disque

### **Endpoints de Monitoring**
```bash
# Santé générale
GET /health

# Santé détaillée
GET /admin/health-detailed

# Statistiques de performance
GET /stats/performance

# Statistiques d'usage du modèle
GET /stats/model-usage
```

---

## 🚀 Déploiement

### **Variables d'Environnement Production**
```bash
# Sécurité
JWT_SECRET=your-super-secure-secret-key
ENCRYPTION_KEY=your-32-character-encryption-key

# Base de données
PLANT_AI_DB_PATH=/var/lib/plant-ai/plant_ai.db

# Performance
PLANT_AI_MAX_UPLOAD_BYTES=52428800  # 50MB
PLANT_AI_LOG_LEVEL=INFO

# CORS
PLANT_AI_ALLOWED_ORIGINS=["https://yourdomain.com"]
```

### **Démarrage en Production**
```bash
# Avec Gunicorn
gunicorn backend.app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Avec Docker
docker build -t plant-ai .
docker run -p 8000:8000 plant-ai
```

---

## 📞 Support

### **Documentation Complète**
- **API Documentation** : `API_DOCUMENTATION.md`
- **Implementation Summary** : `IMPLEMENTATION_SUMMARY.md`
- **README** : `README.md`

### **Tests et Validation**
- **Test Complet** : `python test_new_api.py`
- **Test Rapide** : `python quick_test.py`
- **Création Admin** : `python create_admin.py`

### **URLs Utiles**
- **API** : http://localhost:8000
- **Swagger** : http://localhost:8000/docs
- **Admin** : http://localhost:8000/admin/
- **Health** : http://localhost:8000/health

---

## ✅ Checklist de Démarrage

- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Fichier `.env` configuré
- [ ] Serveur démarré (`python start_api.py`)
- [ ] Admin créé (`python create_admin.py`)
- [ ] Tests passés (`python quick_test.py`)
- [ ] Swagger accessible (http://localhost:8000/docs)
- [ ] Dashboard admin accessible (http://localhost:8000/admin/)

**🎉 Votre API Plant-AI est maintenant prête à l'emploi !**


