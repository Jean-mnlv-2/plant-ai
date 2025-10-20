"""
Modèles Pydantic pour la base de connaissances agronomique.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, EmailStr, validator
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

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


class PathogenType(str, Enum):
    FUNGUS = "fungus"
    BACTERIA = "bacteria"
    VIRUS = "virus"
    PARASITE = "parasite"


class MethodType(str, Enum):
    CULTURAL = "cultural"
    BIOLOGICAL = "biological"
    CHEMICAL = "chemical"


class PracticeType(str, Enum):
    CULTURAL = "cultural"
    BIOLOGICAL = "biological"
    MONITORING = "monitoring"


class ImageType(str, Enum):
    SYMPTOM = "symptom"
    PATHOGEN = "pathogen"
    DAMAGE = "damage"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EffectivenessLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CostLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# MODÈLES DE BASE
# =============================================================================

class CultureBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Nom de la culture")
    scientific_name: Optional[str] = Field(None, max_length=255, description="Nom scientifique")
    category: Optional[str] = Field(None, max_length=100, description="Catégorie de la culture")
    description: Optional[str] = Field(None, description="Description de la culture")


class CultureCreate(CultureBase):
    pass


class CultureUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    scientific_name: Optional[str] = Field(None, max_length=255)
    category: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = None


class Culture(CultureBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PathogenBase(BaseModel):
    scientific_name: str = Field(..., min_length=1, max_length=255, description="Nom scientifique du pathogène")
    common_name: Optional[str] = Field(None, max_length=255, description="Nom commun")
    type: PathogenType = Field(..., description="Type de pathogène")
    description: Optional[str] = Field(None, description="Description du pathogène")


class PathogenCreate(PathogenBase):
    pass


class PathogenUpdate(BaseModel):
    scientific_name: Optional[str] = Field(None, min_length=1, max_length=255)
    common_name: Optional[str] = Field(None, max_length=255)
    type: Optional[PathogenType] = None
    description: Optional[str] = None


class Pathogen(PathogenBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# MODÈLES POUR LES FICHES MALADIES
# =============================================================================

class DiseaseSymptom(BaseModel):
    description: str = Field(..., min_length=1, description="Description du symptôme")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Sévérité du symptôme")
    affected_part: Optional[str] = Field(None, max_length=100, description="Partie affectée (feuilles, fruits, etc.)")


class DiseaseImage(BaseModel):
    image_url: str = Field(..., min_length=1, max_length=500, description="URL de l'image")
    image_type: ImageType = Field(default=ImageType.SYMPTOM, description="Type d'image")
    description: Optional[str] = Field(None, description="Description de l'image")
    is_primary: bool = Field(default=False, description="Image principale")


class DiseaseConditions(BaseModel):
    temperature_min: Optional[float] = Field(None, description="Température minimale favorable")
    temperature_max: Optional[float] = Field(None, description="Température maximale favorable")
    humidity_min: Optional[int] = Field(None, ge=0, le=100, description="Humidité minimale (%)")
    humidity_max: Optional[int] = Field(None, ge=0, le=100, description="Humidité maximale (%)")
    soil_type: Optional[str] = Field(None, max_length=100, description="Type de sol favorable")
    seasonality: Optional[str] = Field(None, max_length=100, description="Saisonnalité")
    climate_conditions: Optional[str] = Field(None, description="Conditions climatiques")


class DiseaseControlMethod(BaseModel):
    method_type: MethodType = Field(..., description="Type de méthode de lutte")
    description: str = Field(..., min_length=1, description="Description de la méthode")
    effectiveness: EffectivenessLevel = Field(default=EffectivenessLevel.MEDIUM, description="Efficacité")
    cost_level: CostLevel = Field(default=CostLevel.MEDIUM, description="Niveau de coût")


class DiseaseProduct(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=255, description="Nom du produit")
    active_ingredient: Optional[str] = Field(None, max_length=255, description="Ingrédient actif")
    dosage: Optional[str] = Field(None, max_length=100, description="Dosage recommandé")
    application_method: Optional[str] = Field(None, max_length=100, description="Méthode d'application")
    safety_class: Optional[str] = Field(None, max_length=50, description="Classe de sécurité")
    registration_number: Optional[str] = Field(None, max_length=100, description="Numéro d'homologation")


class DiseasePrecautions(BaseModel):
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Niveau de risque")
    safety_period_days: Optional[int] = Field(None, ge=0, description="Délai de sécurité en jours")
    dosage_instructions: Optional[str] = Field(None, description="Instructions de dosage")
    contraindications: Optional[str] = Field(None, description="Contre-indications")
    protective_equipment: Optional[str] = Field(None, description="Équipement de protection")


class DiseasePrevention(BaseModel):
    practice_type: PracticeType = Field(..., description="Type de pratique")
    description: str = Field(..., min_length=1, description="Description de la pratique")
    timing: Optional[str] = Field(None, max_length=100, description="Timing d'application")
    effectiveness: EffectivenessLevel = Field(default=EffectivenessLevel.MEDIUM, description="Efficacité")


class DiseaseRegion(BaseModel):
    region_name: str = Field(..., min_length=1, max_length=255, description="Nom de la région")
    country: Optional[str] = Field(None, max_length=100, description="Pays")
    climate_zone: Optional[str] = Field(None, max_length=100, description="Zone climatique")
    latitude_min: Optional[float] = Field(None, ge=-90, le=90, description="Latitude minimale")
    latitude_max: Optional[float] = Field(None, ge=-90, le=90, description="Latitude maximale")
    longitude_min: Optional[float] = Field(None, ge=-180, le=180, description="Longitude minimale")
    longitude_max: Optional[float] = Field(None, ge=-180, le=180, description="Longitude maximale")


class DiseaseTranslation(BaseModel):
    language_code: str = Field(..., min_length=2, max_length=5, description="Code de langue (fr, en, es, etc.)")
    field_name: str = Field(..., min_length=1, max_length=50, description="Nom du champ traduit")
    translated_text: str = Field(..., min_length=1, description="Texte traduit")


# =============================================================================
# MODÈLES PRINCIPAUX POUR LES FICHES MALADIES
# =============================================================================

class DiseaseEntryBase(BaseModel):
    culture_id: int = Field(..., description="ID de la culture")
    pathogen_id: int = Field(..., description="ID du pathogène")
    scientific_name: str = Field(..., min_length=1, max_length=255, description="Nom scientifique de la maladie")
    common_name: str = Field(..., min_length=1, max_length=255, description="Nom commun de la maladie")
    severity_level: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Niveau de sévérité")
    treatment_urgency: TreatmentUrgency = Field(default=TreatmentUrgency.PLANNED, description="Urgence de traitement")


class DiseaseEntryCreate(DiseaseEntryBase):
    symptoms: List[DiseaseSymptom] = Field(default_factory=list, description="Liste des symptômes")
    images: List[DiseaseImage] = Field(default_factory=list, description="Liste des images")
    conditions: Optional[DiseaseConditions] = Field(None, description="Conditions favorables")
    control_methods: List[DiseaseControlMethod] = Field(default_factory=list, description="Méthodes de lutte")
    products: List[DiseaseProduct] = Field(default_factory=list, description="Produits recommandés")
    precautions: Optional[DiseasePrecautions] = Field(None, description="Précautions d'usage")
    prevention: List[DiseasePrevention] = Field(default_factory=list, description="Mesures de prévention")
    regions: List[DiseaseRegion] = Field(default_factory=list, description="Régions affectées")
    translations: List[DiseaseTranslation] = Field(default_factory=list, description="Traductions")


class DiseaseEntryUpdate(BaseModel):
    culture_id: Optional[int] = None
    pathogen_id: Optional[int] = None
    scientific_name: Optional[str] = Field(None, min_length=1, max_length=255)
    common_name: Optional[str] = Field(None, min_length=1, max_length=255)
    severity_level: Optional[SeverityLevel] = None
    treatment_urgency: Optional[TreatmentUrgency] = None
    symptoms: Optional[List[DiseaseSymptom]] = None
    images: Optional[List[DiseaseImage]] = None
    conditions: Optional[DiseaseConditions] = None
    control_methods: Optional[List[DiseaseControlMethod]] = None
    products: Optional[List[DiseaseProduct]] = None
    precautions: Optional[DiseasePrecautions] = None
    prevention: Optional[List[DiseasePrevention]] = None
    regions: Optional[List[DiseaseRegion]] = None
    translations: Optional[List[DiseaseTranslation]] = None


class DiseaseEntry(DiseaseEntryBase):
    id: int
    is_active: bool = True
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    # Données associées
    symptoms: List[DiseaseSymptom] = Field(default_factory=list)
    images: List[DiseaseImage] = Field(default_factory=list)
    conditions: Optional[DiseaseConditions] = None
    control_methods: List[DiseaseControlMethod] = Field(default_factory=list)
    products: List[DiseaseProduct] = Field(default_factory=list)
    precautions: Optional[DiseasePrecautions] = None
    prevention: List[DiseasePrevention] = Field(default_factory=list)
    regions: List[DiseaseRegion] = Field(default_factory=list)
    translations: Dict[str, str] = Field(default_factory=dict)
    
    # Données de référence
    culture_name: Optional[str] = None
    pathogen_name: Optional[str] = None
    pathogen_type: Optional[PathogenType] = None

    class Config:
        from_attributes = True


# =============================================================================
# MODÈLES POUR LES RÉPONSES API
# =============================================================================

class DiseaseEntryListResponse(BaseModel):
    success: bool = True
    entries: List[DiseaseEntry]
    pagination: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = None


class DiseaseEntryResponse(BaseModel):
    success: bool = True
    entry: DiseaseEntry


class DiseaseEntryCreateResponse(BaseModel):
    success: bool = True
    entry_id: int
    message: str = "Fiche maladie créée avec succès"


class DiseaseEntryUpdateResponse(BaseModel):
    success: bool = True
    message: str = "Fiche maladie mise à jour avec succès"


class DiseaseEntryDeleteResponse(BaseModel):
    success: bool = True
    message: str = "Fiche maladie supprimée avec succès"


class CultureListResponse(BaseModel):
    success: bool = True
    cultures: List[Culture]


class PathogenListResponse(BaseModel):
    success: bool = True
    pathogens: List[Pathogen]


# =============================================================================
# MODÈLES POUR LES FILTRES ET RECHERCHE
# =============================================================================

class DiseaseEntryFilters(BaseModel):
    search: Optional[str] = Field(None, description="Recherche textuelle")
    culture_id: Optional[int] = Field(None, description="Filtrer par culture")
    pathogen_type: Optional[PathogenType] = Field(None, description="Filtrer par type de pathogène")
    severity: Optional[SeverityLevel] = Field(None, description="Filtrer par sévérité")
    urgency: Optional[TreatmentUrgency] = Field(None, description="Filtrer par urgence")
    language: str = Field(default="fr", description="Langue pour les traductions")


class DiseaseEntrySearchRequest(BaseModel):
    filters: DiseaseEntryFilters
    limit: int = Field(default=20, ge=1, le=100, description="Nombre d'éléments par page")
    offset: int = Field(default=0, ge=0, description="Décalage pour la pagination")


# =============================================================================
# MODÈLES POUR LES STATISTIQUES
# =============================================================================

class DiseaseStatistics(BaseModel):
    total_entries: int
    active_entries: int
    entries_by_culture: Dict[str, int]
    entries_by_pathogen_type: Dict[str, int]
    entries_by_severity: Dict[str, int]
    recent_entries: int  # Entrées créées dans les 30 derniers jours


class KnowledgeBaseStats(BaseModel):
    success: bool = True
    statistics: DiseaseStatistics
    last_updated: datetime


# =============================================================================
# MODÈLES POUR LA CONFIRMATION DE SUPPRESSION
# =============================================================================

class DeleteConfirmation(BaseModel):
    disease_id: int
    confirm: bool = Field(..., description="Confirmation de suppression")
    reason: Optional[str] = Field(None, max_length=500, description="Raison de la suppression")


class BulkDeleteRequest(BaseModel):
    disease_ids: List[int] = Field(..., min_items=1, description="IDs des fiches à supprimer")
    confirm: bool = Field(..., description="Confirmation de suppression en masse")
    reason: Optional[str] = Field(None, max_length=500, description="Raison de la suppression")


# =============================================================================
# VALIDATEURS PERSONNALISÉS
# =============================================================================

@validator('temperature_min', 'temperature_max')
def validate_temperature(cls, v):
    if v is not None and (v < -50 or v > 60):
        raise ValueError('La température doit être entre -50°C et 60°C')
    return v


@validator('humidity_min', 'humidity_max')
def validate_humidity(cls, v):
    if v is not None and (v < 0 or v > 100):
        raise ValueError('L\'humidité doit être entre 0% et 100%')
    return v


@validator('latitude_min', 'latitude_max')
def validate_latitude(cls, v):
    if v is not None and (v < -90 or v > 90):
        raise ValueError('La latitude doit être entre -90 et 90')
    return v


@validator('longitude_min', 'longitude_max')
def validate_longitude(cls, v):
    if v is not None and (v < -180 or v > 180):
        raise ValueError('La longitude doit être entre -180 et 180')
    return v


@validator('language_code')
def validate_language_code(cls, v):
    if v and len(v) < 2:
        raise ValueError('Le code de langue doit avoir au moins 2 caractères')
    return v


# =============================================================================
# MODÈLES POUR L'IMPORT/EXPORT
# =============================================================================

class DiseaseEntryImport(BaseModel):
    """Modèle pour l'import de fiches maladies depuis CSV/Excel."""
    culture_name: str
    pathogen_scientific_name: str
    pathogen_type: PathogenType
    scientific_name: str
    common_name: str
    severity_level: SeverityLevel
    treatment_urgency: TreatmentUrgency
    symptoms: List[str] = Field(default_factory=list)
    control_methods: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    prevention: List[str] = Field(default_factory=list)
    regions: List[str] = Field(default_factory=list)


class ImportResult(BaseModel):
    success: bool
    imported_count: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    format: str = Field(default="json", description="Format d'export (json, csv, excel)")
    filters: Optional[DiseaseEntryFilters] = None
    include_images: bool = Field(default=False, description="Inclure les URLs des images")
    include_translations: bool = Field(default=False, description="Inclure les traductions")
