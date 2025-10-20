# 📚 Base de Connaissances Agronomique Plant-AI

## 🎯 Vue d'ensemble

La Base de Connaissances Agronomique Plant-AI est un système complet de gestion des fiches maladies végétales, conçu pour supporter jusqu'à **50,000 fiches maladies** avec des performances optimales.

## 🏗️ Architecture Technique

### **Base de Données PostgreSQL**
- **Moteur** : PostgreSQL 12+ avec optimisations
- **Tables** : 12 tables spécialisées pour la gestion des fiches
- **Index** : Index full-text et optimisés pour la recherche
- **Performance** : Support de 50,000+ fiches sans dégradation

### **Structure des Données**

#### **Tables Principales :**
1. **`disease_entries`** - Fiches maladies principales
2. **`cultures`** - Types de plantes (Tomate, Maïs, etc.)
3. **`pathogens`** - Pathogènes (Champignons, Bactéries, Virus, Parasites)
4. **`disease_symptoms`** - Symptômes détaillés
5. **`disease_images`** - Images de référence
6. **`disease_conditions`** - Conditions favorables
7. **`disease_control_methods`** - Méthodes de lutte
8. **`disease_products`** - Produits recommandés
9. **`disease_precautions`** - Précautions d'usage
10. **`disease_prevention`** - Mesures de prévention
11. **`disease_regions`** - Zones géographiques
12. **`disease_translations`** - Support multilingue

## 🔧 Installation et Configuration

### **Prérequis**
- Python 3.9+
- PostgreSQL 12+
- 4GB RAM minimum (8GB recommandé pour 50,000 fiches)

### **Installation**

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Configurer PostgreSQL
createdb plant_ai
createuser plant_ai
psql -c "ALTER USER plant_ai PASSWORD 'plant_ai_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE plant_ai TO plant_ai;"

# 3. Initialiser la base de données
python scripts/init_postgres.py

# 4. Créer des données d'exemple (optionnel)
python scripts/init_postgres.py --sample

# 5. Tester le système
python test_knowledge_base.py
```

### **Configuration**

Créer un fichier `.env` :

```bash
# PostgreSQL Configuration
PLANT_AI_POSTGRES_HOST=localhost
PLANT_AI_POSTGRES_PORT=5432
PLANT_AI_POSTGRES_DATABASE=plant_ai
PLANT_AI_POSTGRES_USER=plant_ai
PLANT_AI_POSTGRES_PASSWORD=plant_ai_password

# Performance
PLANT_AI_MAX_UPLOAD_BYTES=52428800  # 50MB
PLANT_AI_LOG_LEVEL=INFO
```

## 🚀 Utilisation

### **Démarrage du Système**

```bash
# Démarrer l'API avec la base de connaissances
python start_api.py

# Ou démarrer uniquement la base de connaissances
python start_knowledge_base.py
```

### **URLs d'Accès**

- **API Documentation** : http://localhost:8000/docs
- **Admin Dashboard** : http://localhost:8000/admin/knowledge/
- **Gestion Fiches** : http://localhost:8000/admin/knowledge/diseases
- **Gestion Cultures** : http://localhost:8000/admin/knowledge/cultures
- **Gestion Pathogènes** : http://localhost:8000/admin/knowledge/pathogens
- **Statistiques** : http://localhost:8000/admin/knowledge/statistics

## 📊 APIs Disponibles

### **Gestion des Fiches Maladies**

#### **Créer une Fiche**
```bash
POST /api/v1/knowledge/diseases
Content-Type: application/json
Authorization: Bearer <token>

{
  "culture_id": 1,
  "pathogen_id": 1,
  "scientific_name": "Phytophthora infestans",
  "common_name": "Mildiou de la tomate",
  "severity_level": "high",
  "treatment_urgency": "immediate",
  "symptoms": [
    {
      "description": "Taches brunes sur les feuilles",
      "severity": "high",
      "affected_part": "feuilles"
    }
  ],
  "images": [
    {
      "image_url": "https://example.com/mildiou.jpg",
      "image_type": "symptom",
      "description": "Taches sur les feuilles",
      "is_primary": true
    }
  ],
  "conditions": {
    "temperature_min": 15.0,
    "temperature_max": 25.0,
    "humidity_min": 80,
    "humidity_max": 100,
    "soil_type": "Humide",
    "seasonality": "Printemps, Été"
  },
  "control_methods": [
    {
      "method_type": "cultural",
      "description": "Rotation des cultures",
      "effectiveness": "high",
      "cost_level": "low"
    }
  ],
  "products": [
    {
      "product_name": "Bouillie bordelaise",
      "active_ingredient": "Sulfate de cuivre",
      "dosage": "2-3 kg/ha",
      "application_method": "Pulvérisation foliaire"
    }
  ],
  "precautions": {
    "risk_level": "medium",
    "safety_period_days": 7,
    "dosage_instructions": "Respecter les doses recommandées"
  },
  "prevention": [
    {
      "practice_type": "cultural",
      "description": "Éviter l'humidité excessive",
      "timing": "Toute la saison",
      "effectiveness": "high"
    }
  ],
  "regions": [
    {
      "region_name": "Europe",
      "country": "France",
      "climate_zone": "Tempéré"
    }
  ],
  "translations": [
    {
      "language_code": "en",
      "field_name": "common_name",
      "translated_text": "Late blight"
    }
  ]
}
```

#### **Récupérer les Fiches**
```bash
GET /api/v1/knowledge/diseases?limit=20&offset=0&search=mildiou&culture_id=1&severity=high
Authorization: Bearer <token>
```

#### **Récupérer une Fiche Spécifique**
```bash
GET /api/v1/knowledge/diseases/123?language=fr
Authorization: Bearer <token>
```

#### **Modifier une Fiche**
```bash
PUT /api/v1/knowledge/diseases/123
Content-Type: application/json
Authorization: Bearer <token>

{
  "scientific_name": "Phytophthora infestans Updated",
  "severity_level": "critical"
}
```

#### **Supprimer une Fiche**
```bash
DELETE /api/v1/knowledge/diseases/123?confirm=true
Authorization: Bearer <token>
```

### **Gestion des Cultures**

#### **Créer une Culture**
```bash
POST /api/v1/knowledge/cultures
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Tomate",
  "scientific_name": "Solanum lycopersicum",
  "category": "Légume",
  "description": "Plante potagère très cultivée"
}
```

#### **Lister les Cultures**
```bash
GET /api/v1/knowledge/cultures?search=tomate
Authorization: Bearer <token>
```

### **Gestion des Pathogènes**

#### **Créer un Pathogène**
```bash
POST /api/v1/knowledge/pathogens
Content-Type: application/json
Authorization: Bearer <token>

{
  "scientific_name": "Phytophthora infestans",
  "common_name": "Mildiou",
  "type": "fungus",
  "description": "Champignon responsable du mildiou"
}
```

#### **Lister les Pathogènes**
```bash
GET /api/v1/knowledge/pathogens?pathogen_type=fungus
Authorization: Bearer <token>
```

### **Statistiques**

#### **Récupérer les Statistiques**
```bash
GET /api/v1/knowledge/statistics
Authorization: Bearer <token>
```

## 🛠️ Interface d'Administration

### **Fonctionnalités Administratives**

1. **Dashboard Principal**
   - Statistiques en temps réel
   - Dernières fiches créées
   - Navigation rapide

2. **Gestion des Fiches Maladies**
   - Création/Modification/Suppression
   - Recherche et filtrage avancés
   - Pagination optimisée
   - Confirmation de suppression

3. **Gestion des Cultures**
   - CRUD complet
   - Recherche textuelle
   - Catégorisation

4. **Gestion des Pathogènes**
   - CRUD complet
   - Filtrage par type
   - Classification scientifique

5. **Statistiques Avancées**
   - Répartition par culture
   - Répartition par pathogène
   - Évolution temporelle
   - Métriques de performance

### **Sécurité**

- **Authentification JWT** requise
- **Rôle Admin** pour la modification
- **Validation** des données d'entrée
- **Confirmation** pour les suppressions
- **Audit** des modifications

## 🔍 Recherche et Performance

### **Recherche Full-Text**

La base de connaissances utilise PostgreSQL full-text search avec :

- **Index GIN** pour la recherche rapide
- **Support multilingue** (français, anglais, etc.)
- **Recherche par synonymes** et variantes
- **Filtrage avancé** par critères multiples

### **Optimisations Performance**

1. **Index Spécialisés**
   - Index sur les noms scientifiques
   - Index sur les symptômes
   - Index géographiques
   - Index temporels

2. **Pagination Intelligente**
   - Limite configurable (1-100)
   - Offset optimisé
   - Comptage efficace

3. **Cache de Requêtes**
   - Mise en cache des cultures
   - Mise en cache des pathogènes
   - Invalidation intelligente

### **Tests de Performance**

```bash
# Test complet
python test_knowledge_base.py

# Test de performance avec 1000 fiches
python test_knowledge_base.py --performance
```

## 🌍 Support Multilingue

### **Langues Supportées**

- **Français** (fr) - Langue par défaut
- **Anglais** (en)
- **Espagnol** (es)
- **Allemand** (de)
- **Italien** (it)

### **Champs Traduisibles**

- Nom commun de la maladie
- Description des symptômes
- Méthodes de lutte
- Produits recommandés
- Mesures de prévention

### **Exemple de Traduction**

```json
{
  "translations": [
    {
      "language_code": "en",
      "field_name": "common_name",
      "translated_text": "Late blight"
    },
    {
      "language_code": "es",
      "field_name": "common_name",
      "translated_text": "Tizón tardío"
    }
  ]
}
```

## 📈 Monitoring et Statistiques

### **Métriques Disponibles**

1. **Statistiques Générales**
   - Nombre total de fiches
   - Fiches actives/inactives
   - Répartition par culture
   - Répartition par pathogène

2. **Statistiques Temporelles**
   - Fiches créées par mois
   - Évolution des ajouts
   - Activité des utilisateurs

3. **Statistiques de Performance**
   - Temps de réponse des requêtes
   - Utilisation des index
   - Cache hit ratio

### **Dashboard Administratif**

- **Graphiques interactifs** avec Chart.js
- **Filtres temporels** (7j, 30j, 90j, 1an)
- **Export des données** (CSV, JSON)
- **Alertes de performance**

## 🚀 Déploiement Production

### **Configuration Production**

```bash
# Variables d'environnement
PLANT_AI_ENV=production
PLANT_AI_POSTGRES_HOST=postgres-server
PLANT_AI_POSTGRES_PASSWORD=secure_password
PLANT_AI_LOG_LEVEL=WARNING
```

### **Optimisations PostgreSQL**

```sql
-- Configuration pour 50,000+ fiches
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
```

### **Monitoring Production**

- **Prometheus** pour les métriques
- **Grafana** pour les tableaux de bord
- **Alertmanager** pour les alertes
- **Logs centralisés** avec ELK Stack

## 🔧 Maintenance

### **Nettoyage des Données**

```bash
# Nettoyage des fiches inactives
curl -X POST http://localhost:8000/admin/knowledge/cleanup

# Sauvegarde de la base
pg_dump plant_ai > backup_$(date +%Y%m%d).sql
```

### **Mise à Jour**

```bash
# Mise à jour du schéma
python scripts/migrate_postgres.py

# Vérification de l'intégrité
python test_knowledge_base.py
```

## 📚 Exemples d'Utilisation

### **Créer une Fiche Complète**

```python
import requests

# Données de la fiche
disease_data = {
    "culture_id": 1,
    "pathogen_id": 1,
    "scientific_name": "Phytophthora infestans",
    "common_name": "Mildiou de la tomate",
    "severity_level": "high",
    "treatment_urgency": "immediate",
    "symptoms": [
        {
            "description": "Taches brunes sur les feuilles",
            "severity": "high",
            "affected_part": "feuilles"
        }
    ],
    "images": [
        {
            "image_url": "https://example.com/mildiou.jpg",
            "image_type": "symptom",
            "description": "Taches sur les feuilles",
            "is_primary": True
        }
    ],
    "conditions": {
        "temperature_min": 15.0,
        "temperature_max": 25.0,
        "humidity_min": 80,
        "humidity_max": 100,
        "soil_type": "Humide",
        "seasonality": "Printemps, Été"
    },
    "control_methods": [
        {
            "method_type": "cultural",
            "description": "Rotation des cultures",
            "effectiveness": "high",
            "cost_level": "low"
        }
    ],
    "products": [
        {
            "product_name": "Bouillie bordelaise",
            "active_ingredient": "Sulfate de cuivre",
            "dosage": "2-3 kg/ha",
            "application_method": "Pulvérisation foliaire"
        }
    ],
    "precautions": {
        "risk_level": "medium",
        "safety_period_days": 7,
        "dosage_instructions": "Respecter les doses recommandées"
    },
    "prevention": [
        {
            "practice_type": "cultural",
            "description": "Éviter l'humidité excessive",
            "timing": "Toute la saison",
            "effectiveness": "high"
        }
    ],
    "regions": [
        {
            "region_name": "Europe",
            "country": "France",
            "climate_zone": "Tempéré"
        }
    ]
}

# Créer la fiche
response = requests.post(
    "http://localhost:8000/api/v1/knowledge/diseases",
    json=disease_data,
    headers={"Authorization": "Bearer <token>"}
)

print(response.json())
```

### **Recherche Avancée**

```python
# Recherche avec filtres
params = {
    "search": "mildiou",
    "culture_id": 1,
    "severity": "high",
    "limit": 20,
    "offset": 0
}

response = requests.get(
    "http://localhost:8000/api/v1/knowledge/diseases",
    params=params,
    headers={"Authorization": "Bearer <token>"}
)

results = response.json()
print(f"Trouvé {results['pagination']['total']} fiches")
```

## 🎯 Conclusion

La Base de Connaissances Agronomique Plant-AI est un système **professionnel et complet** qui offre :

✅ **Support de 50,000+ fiches maladies** sans perte de performance  
✅ **Interface d'administration intuitive** avec CRUD complet  
✅ **Recherche full-text optimisée** avec filtres avancés  
✅ **Support multilingue** pour l'internationalisation  
✅ **Sécurité robuste** avec authentification et validation  
✅ **Monitoring complet** avec statistiques et métriques  
✅ **Architecture scalable** prête pour la production  

Le système est maintenant **opérationnel** et prêt pour la gestion professionnelle de la base de connaissances agronomique ! 🌱📚
