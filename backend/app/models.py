"""
Modèles Pydantic pour l'API Plant-AI.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class UserRole(str, Enum):
    FARMER = "farmer"
    AGRONOMIST = "agronomist"
    RESEARCHER = "researcher"
    ADMIN = "admin"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TreatmentUrgency(str, Enum):
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    PLANNED = "planned"
    MONITORING = "monitoring"


class WeatherCondition(str, Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    FOGGY = "foggy"
    SNOWY = "snowy"


# -----------------------------------------------------------------------------
# Modèles d'Authentification
# -----------------------------------------------------------------------------
class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    country: str = Field(..., min_length=2, max_length=100)
    role: UserRole = Field(default=UserRole.FARMER)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenRefresh(BaseModel):
    refreshToken: str


class User(BaseModel):
    id: str
    name: str
    email: str
    country: str
    role: UserRole
    createdAt: datetime
    preferences: Optional[Dict[str, Any]] = None


class Tokens(BaseModel):
    accessToken: str
    refreshToken: str
    expiresIn: int


class AuthResponse(BaseModel):
    success: bool
    user: User
    tokens: Tokens


# -----------------------------------------------------------------------------
# Modèles de Diagnostic
# -----------------------------------------------------------------------------
class Location(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str
    region: Optional[str] = None


class AdditionalInfo(BaseModel):
    symptoms: Optional[List[str]] = None
    environment: Optional[str] = None
    plantAge: Optional[str] = None
    soilType: Optional[str] = None


class DiagnoseRequest(BaseModel):
    images: List[str] = Field(..., description="Images en base64")
    location: Location
    plantType: str = Field(..., description="Type de plante (tomato, potato, etc.)")
    additionalInfo: Optional[AdditionalInfo] = None


class Disease(BaseModel):
    id: str
    name: str
    confidence: float = Field(..., ge=0.0, le=100.0)
    severity: SeverityLevel
    category: str = Field(default="diseases")


class ImageResult(BaseModel):
    imageIndex: int
    disease: Disease
    symptoms: List[str]
    solutions: List[str]
    prevention: List[str]
    treatmentUrgency: TreatmentUrgency
    affectedArea: float = Field(..., ge=0.0, le=1.0)


class Recommendations(BaseModel):
    priority: SeverityLevel
    nextSteps: List[str]


class DiagnoseResponse(BaseModel):
    success: bool
    diagnosticId: str
    results: List[ImageResult]
    overallConfidence: float = Field(..., ge=0.0, le=100.0)
    processingTime: float
    timestamp: datetime
    recommendations: Recommendations


# -----------------------------------------------------------------------------
# Modèles de Gestion des Diagnostics
# -----------------------------------------------------------------------------
class DiagnosticSummary(BaseModel):
    id: str
    userId: str
    images: List[str]
    results: Dict[str, Any]
    location: Location
    plantType: str
    status: str = Field(default="completed")
    createdAt: datetime
    updatedAt: datetime


class DiagnosticsListResponse(BaseModel):
    success: bool
    diagnostics: List[DiagnosticSummary]
    pagination: Dict[str, Any]


class SyncDiagnostic(BaseModel):
    id: str
    images: List[str]
    results: Dict[str, Any]
    location: Location
    timestamp: datetime


class SyncRequest(BaseModel):
    diagnostics: List[SyncDiagnostic]


# -----------------------------------------------------------------------------
# Modèles Météo
# -----------------------------------------------------------------------------
class Coordinates(BaseModel):
    latitude: float
    longitude: float


class WeatherLocation(BaseModel):
    name: str
    country: str
    coordinates: Coordinates


class Temperature(BaseModel):
    min: float
    max: float


class Precipitation(BaseModel):
    probability: int = Field(..., ge=0, le=100)
    amount: float


class WeatherForecast(BaseModel):
    date: str
    temperature: Temperature
    humidity: int = Field(..., ge=0, le=100)
    windSpeed: float
    condition: WeatherCondition
    precipitation: Precipitation
    agriculturalAdvice: str


class WeatherAlert(BaseModel):
    type: str
    severity: SeverityLevel
    title: str
    description: str
    validFrom: datetime
    validTo: datetime
    recommendations: List[str]


class AgriculturalConditions(BaseModel):
    irrigation: str
    spraying: str
    harvesting: str
    planting: str


class CurrentWeather(BaseModel):
    temperature: float
    humidity: int = Field(..., ge=0, le=100)
    windSpeed: float
    windDirection: str
    visibility: float
    pressure: float
    uvIndex: int = Field(..., ge=0, le=11)
    condition: WeatherCondition
    description: str
    timestamp: datetime


class WeatherResponse(BaseModel):
    success: bool
    location: WeatherLocation
    current: CurrentWeather
    forecast: List[WeatherForecast]
    alerts: List[WeatherAlert]
    agriculturalConditions: AgriculturalConditions


# -----------------------------------------------------------------------------
# Modèles Base de Connaissances
# -----------------------------------------------------------------------------
class DiseaseDetail(BaseModel):
    id: str
    name: str
    category: str
    symptoms: List[str]
    solutions: List[str]
    prevention: List[str]
    images: List[str]
    severity: SeverityLevel
    affectedPlants: List[str]
    seasonality: List[str]
    geographicDistribution: List[str]
    treatmentUrgency: TreatmentUrgency
    lastUpdated: datetime


class DiseasesListResponse(BaseModel):
    success: bool
    diseases: List[DiseaseDetail]
    pagination: Dict[str, Any]


# -----------------------------------------------------------------------------
# Modèles Gestion Utilisateurs
# -----------------------------------------------------------------------------
class UserPreferences(BaseModel):
    language: str = Field(default="fr")
    notifications: bool = Field(default=True)
    units: str = Field(default="metric")
    theme: str = Field(default="light")


class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    country: str
    role: UserRole
    preferences: UserPreferences
    stats: Optional[Dict[str, Any]] = None


class UserProfileResponse(BaseModel):
    success: bool
    user: UserProfile


# -----------------------------------------------------------------------------
# Modèles existants (pour compatibilité)
# -----------------------------------------------------------------------------
class BBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class Prediction(BaseModel):
    class_name: str = Field(..., description="Nom de la classe détectée")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="[xmin, ymin, xmax, ymax]")
    recommendations: List[str]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    image_width: int
    image_height: int
    detected_classes: List[str]
    uncertain: bool = False
    quality: Dict[str, float] = {}


class ClassItem(BaseModel):
    name: str
    recommendations: List[str]


class HistoryItem(BaseModel):
    user_id: str
    timestamp: datetime
    predictions: List[Prediction]


class HistoryPost(BaseModel):
    user_id: str
    predictions: List[Prediction]


class FeedbackItem(BaseModel):
    user_id: str
    prediction_id: str
    is_correct: bool
    comment: Optional[str] = None
    timestamp: datetime


