# 🚀 Configuration Production - Plant-AI

## ⚠️ Avertissement Tailwind CSS

Le projet utilise actuellement le CDN Tailwind CSS pour le développement. **En production, vous devez remplacer le CDN par une version locale.**

## 🔧 Configuration Production

### 1. **Remplacer Tailwind CSS CDN**

#### Option A: Installation locale de Tailwind CSS

```bash
# 1. Installer Tailwind CSS
npm install -D tailwindcss

# 2. Générer le fichier de configuration
npx tailwindcss init

# 3. Créer le fichier d'entrée CSS
mkdir -p src
echo '@tailwind base; @tailwind components; @tailwind utilities;' > src/input.css

# 4. Compiler le CSS
npx tailwindcss -i ./src/input.css -o ./static/css/tailwind.css --watch
```

#### Option B: Utiliser le fichier CSS pré-compilé

```bash
# Télécharger Tailwind CSS compilé
curl -o static/css/tailwind.css https://cdn.tailwindcss.com/3.3.0/tailwind.min.css
```

### 2. **Modifier le template de base**

Dans `templates/admin/base.html`, remplacer :

```html
<!-- Remplacer cette ligne -->
<script src="https://cdn.tailwindcss.com"></script>

<!-- Par cette ligne -->
<link rel="stylesheet" href="/static/css/tailwind.css">
```

### 3. **Configuration PostgreSQL**

```bash
# Créer la base de données
createdb plant_ai

# Créer l'utilisateur
createuser plant_ai

# Configurer les permissions
psql -c "ALTER USER plant_ai PASSWORD 'votre_mot_de_passe_securise';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE plant_ai TO plant_ai;"
```

### 4. **Variables d'environnement**

Créer un fichier `.env` :

```bash
# Production
PLANT_AI_ENV=production
PLANT_AI_POSTGRES_HOST=localhost
PLANT_AI_POSTGRES_PORT=5432
PLANT_AI_POSTGRES_DATABASE=plant_ai
PLANT_AI_POSTGRES_USER=plant_ai
PLANT_AI_POSTGRES_PASSWORD=votre_mot_de_passe_securise

# Sécurité
PLANT_AI_JWT_SECRET=votre_secret_jwt_tres_securise
PLANT_AI_LOG_LEVEL=WARNING

# Performance
PLANT_AI_MAX_UPLOAD_BYTES=52428800
```

### 5. **Installation des dépendances**

```bash
# Python
pip install -r requirements.txt

# Node.js (pour Tailwind CSS)
npm install -D tailwindcss
```

### 6. **Démarrage en production**

```bash
# Initialiser la base de données
python scripts/init_postgres.py

# Démarrer l'API
python start_api.py
```

## 🎯 **Résultat Final**

Après ces modifications, vous aurez :

✅ **Tailwind CSS local** (plus d'avertissement CDN)  
✅ **Base de données PostgreSQL** configurée  
✅ **Base de connaissances agronomique** opérationnelle  
✅ **Interface d'administration** complète  
✅ **Support de 50,000+ fiches maladies**  
✅ **Recherche full-text optimisée**  
✅ **Support multilingue**  
✅ **Sécurité robuste**  

## 🔗 **URLs d'accès**

- **API Documentation** : http://localhost:8000/docs
- **Admin Dashboard** : http://localhost:8000/admin/
- **Base de Connaissances** : http://localhost:8000/admin/knowledge/
- **Gestion Fiches** : http://localhost:8000/admin/knowledge/diseases
- **Gestion Cultures** : http://localhost:8000/admin/knowledge/cultures
- **Gestion Pathogènes** : http://localhost:8000/admin/knowledge/pathogens
- **Statistiques** : http://localhost:8000/admin/knowledge/statistics

## 📚 **Documentation**

- **Base de Connaissances** : `KNOWLEDGE_BASE_DOCUMENTATION.md`
- **Tests** : `python test_knowledge_base.py`
- **Démonstration** : `python demo_knowledge_base.py`

Le système est maintenant **prêt pour la production** ! 🚀
