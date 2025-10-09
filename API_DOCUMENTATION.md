# 📚 Documentation API Plant-AI

## 🚀 Vue d'ensemble

L'API Plant-AI fournit un ensemble complet d'endpoints pour le diagnostic intelligent des maladies des plantes, l'authentification utilisateur, la gestion des diagnostics, les données météo agricoles, et une base de connaissances des maladies.

## 🔗 Base URL

```
http://localhost:8000/api/v1
```

## 🔐 Authentification

Tous les endpoints (sauf l'authentification) nécessitent un token JWT dans l'en-tête :

```
Authorization: Bearer <your_jwt_token>
```

---

## 📋 Endpoints Disponibles

### 🔑 Authentification

#### `POST /api/v1/auth/register`
Créer un nouveau compte utilisateur.

**Body:**
```json
{
  "name": "Jean Dupont",
  "email": "jean.dupont@example.com",
  "password": "MotDePasse123!",
  "country": "France",
  "role": "farmer"
}
```

**Réponse:**
```json
{
  "success": true,
  "user": {
    "id": "user_123",
    "name": "Jean Dupont",
    "email": "jean.dupont@example.com",
    "country": "France",
    "role": "farmer",
    "createdAt": "2024-01-15T14:30:00Z"
  },
  "tokens": {
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expiresIn": 3600
  }
}
```

#### `POST /api/v1/auth/login`
Connexion utilisateur.

**Body:**
```json
{
  "email": "jean.dupont@example.com",
  "password": "MotDePasse123!"
}
```

#### `POST /api/v1/auth/refresh`
Renouveler le token d'accès.

**Body:**
```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

---

### 🔬 Diagnostic IA

#### `POST /api/v1/diagnose`
Analyser des images de plantes pour diagnostiquer les maladies.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Body:**
```json
{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
  ],
  "location": {
    "latitude": 48.8566,
    "longitude": 2.3522,
    "country": "France",
    "region": "Île-de-France"
  },
  "plantType": "tomato",
  "additionalInfo": {
    "symptoms": ["Taches brunes sur les feuilles"],
    "environment": "serre"
  }
}
```

**Réponse:**
```json
{
  "success": true,
  "diagnosticId": "diag_123456",
  "results": [
    {
      "imageIndex": 0,
      "disease": {
        "id": "mildiou-tomate",
        "name": "Mildiou de la tomate",
        "confidence": 87.5,
        "severity": "high",
        "category": "diseases"
      },
      "symptoms": [
        "Taches brunes sur les feuilles",
        "Pourriture des fruits",
        "Feuilles qui se dessèchent"
      ],
      "solutions": [
        "Traitement au cuivre (bouillie bordelaise)",
        "Améliorer l'aération des plants",
        "Éviter l'arrosage sur les feuilles"
      ],
      "prevention": [
        "Rotation des cultures",
        "Espacement adéquat entre les plants",
        "Paillage pour éviter les éclaboussures"
      ],
      "treatmentUrgency": "immediate",
      "affectedArea": 0.75
    }
  ],
  "overallConfidence": 85.2,
  "processingTime": 2.3,
  "timestamp": "2024-01-15T14:30:00Z",
  "recommendations": {
    "priority": "high",
    "nextSteps": [
      "Traiter immédiatement avec de la bouillie bordelaise",
      "Améliorer l'aération de la serre",
      "Planifier un apport d'engrais azoté"
    ]
  }
}
```

---

### 📊 Gestion des Diagnostics

#### `GET /api/v1/diagnostics`
Récupérer l'historique des diagnostics d'un utilisateur.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `limit` (int, optional): Nombre de résultats par page (1-100, défaut: 10)
- `offset` (int, optional): Décalage pour la pagination (défaut: 0)
- `sortBy` (string, optional): Champ de tri ("created_at" ou "updated_at", défaut: "created_at")
- `order` (string, optional): Ordre de tri ("asc" ou "desc", défaut: "desc")

**Exemple:**
```
GET /api/v1/diagnostics?limit=10&offset=0&sortBy=created_at&order=desc
```

**Réponse:**
```json
{
  "success": true,
  "diagnostics": [
    {
      "id": "diag_123456",
      "userId": "user_123",
      "images": [
        "https://api.plant-ai.com/images/diag_123456_0.jpg"
      ],
      "results": {
        "diseases": [
          {
            "id": "mildiou-tomate",
            "name": "Mildiou de la tomate",
            "confidence": 87.5,
            "severity": "high"
          }
        ],
        "overallConfidence": 85.2
      },
      "location": {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "country": "France"
      },
      "plantType": "tomato",
      "status": "completed",
      "createdAt": "2024-01-15T14:30:00Z",
      "updatedAt": "2024-01-15T14:32:30Z"
    }
  ],
  "pagination": {
    "total": 45,
    "limit": 10,
    "offset": 0,
    "hasMore": true
  }
}
```

#### `GET /api/v1/diagnostics/{id}`
Récupérer un diagnostic spécifique.

**Headers:**
```
Authorization: Bearer <token>
```

#### `POST /api/v1/diagnostics/sync`
Synchroniser les diagnostics créés hors ligne.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Body:**
```json
{
  "diagnostics": [
    {
      "id": "local_diag_1",
      "images": ["data:image/jpeg;base64,..."],
      "results": {
        "diseases": [...],
        "overallConfidence": 85.2
      },
      "location": {
        "latitude": 48.8566,
        "longitude": 2.3522
      },
      "timestamp": "2024-01-15T14:30:00Z"
    }
  ]
}
```

---

### 🌤️ Météo Agricole

#### `GET /api/v1/weather`
Récupérer les données météo agricoles.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `lat` (float, required): Latitude (-90 à 90)
- `lng` (float, required): Longitude (-180 à 180)
- `includeForecast` (bool, optional): Inclure les prévisions (défaut: true)

**Exemple:**
```
GET /api/v1/weather?lat=48.8566&lng=2.3522&includeForecast=true
```

**Réponse:**
```json
{
  "success": true,
  "location": {
    "name": "Paris",
    "country": "France",
    "coordinates": {
      "latitude": 48.8566,
      "longitude": 2.3522
    }
  },
  "current": {
    "temperature": 25,
    "humidity": 70,
    "windSpeed": 12,
    "windDirection": "NW",
    "visibility": 10,
    "pressure": 1013,
    "uvIndex": 6,
    "condition": "sunny",
    "description": "Ensoleillé - Idéal pour l'agriculture",
    "timestamp": "2024-01-15T14:30:00Z"
  },
  "forecast": [
    {
      "date": "2024-01-16",
      "temperature": {
        "min": 18,
        "max": 26
      },
      "humidity": 65,
      "windSpeed": 8,
      "condition": "cloudy",
      "precipitation": {
        "probability": 20,
        "amount": 0
      },
      "agriculturalAdvice": "Conditions favorables pour les traitements foliaires"
    }
  ],
  "alerts": [
    {
      "type": "frost_warning",
      "severity": "medium",
      "title": "Risque de gelée nocturne",
      "description": "Températures prévues entre 0°C et 2°C dans la nuit",
      "validFrom": "2024-01-16T02:00:00Z",
      "validTo": "2024-01-16T08:00:00Z",
      "recommendations": [
        "Protéger les cultures sensibles",
        "Arroser légèrement avant le coucher du soleil"
      ]
    }
  ],
  "agriculturalConditions": {
    "irrigation": "not_needed",
    "spraying": "favorable",
    "harvesting": "favorable",
    "planting": "favorable"
  }
}
```

---

### 📚 Base de Connaissances

#### `GET /api/v1/diseases`
Rechercher des maladies dans la base de connaissances.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `search` (string, optional): Terme de recherche
- `category` (string, optional): Catégorie de maladie
- `limit` (int, optional): Nombre de résultats (1-100, défaut: 10)
- `offset` (int, optional): Décalage pour la pagination (défaut: 0)

**Exemple:**
```
GET /api/v1/diseases?search=mildiou&category=diseases&limit=10
```

**Réponse:**
```json
{
  "success": true,
  "diseases": [
    {
      "id": "mildiou-tomate",
      "name": "Mildiou de la tomate",
      "category": "diseases",
      "symptoms": [
        "Taches brunes sur les feuilles",
        "Pourriture des fruits",
        "Feuilles qui se dessèchent",
        "Taches huileuses sur les feuilles"
      ],
      "solutions": [
        "Traitement au cuivre (bouillie bordelaise)",
        "Améliorer l'aération des plants",
        "Éviter l'arrosage sur les feuilles",
        "Supprimer les parties atteintes"
      ],
      "prevention": [
        "Rotation des cultures",
        "Espacement adéquat entre les plants",
        "Paillage pour éviter les éclaboussures",
        "Choisir des variétés résistantes"
      ],
      "images": [
        "https://api.plant-ai.com/diseases/mildiou-tomate-1.jpg",
        "https://api.plant-ai.com/diseases/mildiou-tomate-2.jpg"
      ],
      "severity": "high",
      "affectedPlants": ["tomato", "potato", "pepper"],
      "seasonality": ["spring", "summer"],
      "geographicDistribution": ["europe", "north_america"],
      "treatmentUrgency": "immediate",
      "lastUpdated": "2024-01-10T10:00:00Z"
    }
  ],
  "pagination": {
    "total": 15,
    "limit": 10,
    "offset": 0,
    "hasMore": true
  }
}
```

#### `GET /api/v1/diseases/{id}`
Récupérer les détails d'une maladie spécifique.

**Headers:**
```
Authorization: Bearer <token>
```

---

### 👤 Gestion des Utilisateurs

#### `GET /api/v1/users/profile`
Récupérer le profil de l'utilisateur connecté.

**Headers:**
```
Authorization: Bearer <token>
```

**Réponse:**
```json
{
  "success": true,
  "user": {
    "id": "user_123",
    "name": "Jean Dupont",
    "email": "jean.dupont@example.com",
    "country": "France",
    "role": "farmer",
    "preferences": {
      "language": "fr",
      "notifications": true,
      "units": "metric",
      "theme": "light"
    },
    "stats": {
      "totalPredictions": 45,
      "averageConfidence": 87.2,
      "activeDays": 12,
      "totalDiagnostics": 23
    }
  }
}
```

---

## 🔧 Endpoints Legacy (Compatibilité)

### `GET /health`
Vérifier l'état du service.

### `GET /classes`
Lister les classes du modèle.

### `POST /predict`
Prédire sur une image unique (API legacy).

---

## 📝 Codes d'Erreur

| Code | Description |
|------|-------------|
| 200 | Succès |
| 400 | Requête invalide |
| 401 | Non authentifié |
| 403 | Accès refusé |
| 404 | Ressource non trouvée |
| 413 | Fichier trop volumineux |
| 415 | Type de média non supporté |
| 500 | Erreur serveur |

---

## 🚀 Exemples d'Utilisation

### cURL - Inscription
```bash
curl -X POST https://api.plant-ai.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jean Dupont",
    "email": "jean.dupont@example.com",
    "password": "MotDePasse123!",
    "country": "France",
    "role": "farmer"
  }'
```

### cURL - Diagnostic
```bash
curl -X POST https://api.plant-ai.com/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -d '{
    "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    "location": {
      "latitude": 48.8566,
      "longitude": 2.3522,
      "country": "France",
      "region": "Île-de-France"
    },
    "plantType": "tomato",
    "additionalInfo": {
      "symptoms": ["Taches brunes sur les feuilles"],
      "environment": "serre"
    }
  }'
```

### cURL - Météo
```bash
curl -X GET "https://api.plant-ai.com/api/v1/weather?lat=48.8566&lng=2.3522&includeForecast=true" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

## 🔒 Sécurité

- Tous les endpoints (sauf l'authentification) nécessitent un token JWT valide
- Les mots de passe sont hachés avec PBKDF2
- Les tokens d'accès expirent après 1 heure
- Les tokens de rafraîchissement expirent après 30 jours
- Limitation de taille des uploads (20MB par défaut)

---

## 📊 Monitoring

L'API inclut un système de monitoring automatique qui track :
- Temps de réponse des endpoints
- Taux d'erreur
- Utilisation du modèle
- Métriques de performance

Accédez au dashboard admin à : `http://localhost:8000/admin/`


