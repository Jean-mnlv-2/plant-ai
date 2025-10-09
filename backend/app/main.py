"""
API Plant-AI - Backend principal avec toutes les nouvelles fonctionnalités.
"""
from __future__ import annotations

import io
import logging
import threading
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2

# Imports locaux
from .settings import settings
from .admin import router as admin_router
from .database import db
from .metrics import performance_monitor, monitor_performance, calculate_prediction_metrics
from .dataset_manager import dataset_manager
from .models import *
from .auth import (
    create_user, authenticate_user, generate_tokens, get_current_user,
    get_current_user_id, require_role, UserRole
)
from .weather_service import weather_service
from .diseases_service import diseases_service
from fastapi import Form
from typing import Optional

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("plant_ai")
logger.setLevel(getattr(logging, settings.log_level))
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt=settings.log_format,
    datefmt=settings.log_date_format,
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API Plant-AI - Diagnostic intelligent des maladies des plantes",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Basic health and root endpoints for quick connectivity tests
# -----------------------------------------------------------------------------

@app.get("/healthz", tags=["system"]) 
def healthz():
    return {"status": "ok"}


@app.get("/", tags=["system"]) 
def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "admin": "/admin/"
    }

# Templates & static (admin UI)
templates = Jinja2Templates(directory="templates")
# Mount static only if directory exists to avoid startup crash in dev
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routers
app.include_router(admin_router)

# -----------------------------------------------------------------------------
# Upload size limiting middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Limit upload size to prevent abuse."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            size = int(content_length)
            if size > settings.max_upload_bytes:
                logger.warning(f"Upload size {size} exceeds limit {settings.max_upload_bytes}")
                raise HTTPException(status_code=413, detail="Payload too large")
        except ValueError:
            pass
    return await call_next(request)

# -----------------------------------------------------------------------------
# Model loading (lazy, thread-safe)
# -----------------------------------------------------------------------------
_model_lock = threading.Lock()
_model = None
_model_path = settings.model_file

def get_model():
    """Get or load the YOLO model (thread-safe, lazy loading)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    # Vérifier que le fichier modèle existe
                    if not _model_path.exists():
                        logger.error(f"Model file not found: {_model_path}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Model file not found: {_model_path}. Please train a model first."
                        )
                    
                    from ultralytics import YOLO
                    _model = YOLO(str(_model_path))
                    try:
                        device = settings.model_device
                        if hasattr(_model, 'to') and device:
                            _model = _model.to(device)
                        # Apply half precision if requested and supported
                        if settings.model_half_precision:
                            try:
                                import torch  # type: ignore
                                if torch.cuda.is_available() and str(device).lower().startswith("cuda"):
                                    if hasattr(_model, 'model') and hasattr(_model.model, 'half'):
                                        _model.model.half()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    logger.info(f"Model loaded successfully from {_model_path}")
                    
                    # Vérifier que le modèle est valide
                    if not hasattr(_model, 'names') or not _model.names:
                        logger.error("Model loaded but has no classes")
                        raise HTTPException(
                            status_code=500,
                            detail="Model loaded but has no classes. Please check the model file."
                        )
                    
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model loading failed: {str(e)}"
                    )
    return _model

# =============================================================================
# ADMIN DATASET ENDPOINTS (ported from legacy)
# =============================================================================

@app.get("/admin/dataset/status", tags=["admin", "dataset"]) 
@monitor_performance("dataset_status")
async def get_dataset_status() -> Dict[str, Any]:
    return await dataset_manager.get_dataset_status()


@app.post("/admin/dataset/reset", tags=["admin", "dataset"]) 
@monitor_performance("dataset_reset")
async def reset_dataset_admin() -> Dict[str, Any]:
    return await dataset_manager.reset_dataset()


@app.post("/admin/dataset/process", tags=["admin", "dataset"]) 
@monitor_performance("dataset_process")
async def process_dataset_admin() -> Dict[str, Any]:
    return await dataset_manager.process_dataset()


@app.post("/admin/dataset/train", tags=["admin", "dataset"]) 
@monitor_performance("dataset_train")
async def train_model_admin(
    epochs: int = Query(50, ge=1, le=1000),
    batch_size: int = Query(16, ge=1, le=128),
    image_size: int = Query(640, ge=320, le=1280)
) -> Dict[str, Any]:
    return await dataset_manager.train_model(epochs=epochs, batch_size=batch_size, image_size=image_size)


@app.post("/admin/dataset/upload", tags=["admin", "dataset"], response_model=None) 
@monitor_performance("dataset_upload")
async def upload_dataset_admin(
    files: List[UploadFile] = File(..., description="Fichiers du dataset")
) -> Dict[str, Any]:
    # Séparer les fichiers par type
    train_images = []
    test_images = []
    train_annotations = None
    test_annotations = None
    
    for file in files:
        if file.filename.endswith('.jpg') or file.filename.endswith('.png'):
            if 'train' in file.filename.lower():
                train_images.append(file)
            elif 'test' in file.filename.lower():
                test_images.append(file)
        elif file.filename.endswith('.csv'):
            if 'train' in file.filename.lower():
                train_annotations = file
            elif 'test' in file.filename.lower():
                test_annotations = file
    
    return await dataset_manager.upload_dataset(
        train_images=train_images,
        test_images=test_images,
        train_annotations=train_annotations,
        test_annotations=test_annotations,
    )

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _read_image_from_upload(file: UploadFile) -> Image.Image:
    """Read and validate uploaded image file."""
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}. Allowed: {settings.allowed_content_types}",
        )
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def _pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)

def _process_image_for_yolo(image: Image.Image) -> np.ndarray:
    """Process image for YOLO inference."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = _pil_to_numpy(image)
    
    return img_array

def _generate_diagnostic_id() -> str:
    """Generate unique diagnostic ID."""
    return f"diag_{int(time.time() * 1000)}"

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

# =============================================================================
# AUTHENTICATION APIs
# =============================================================================

@app.post("/api/v1/auth/register", response_model=AuthResponse, tags=["auth"])
@monitor_performance
def register_user(user_data: UserRegister):
    """Créer un nouveau compte utilisateur."""
    try:
        # Créer l'utilisateur
        user = create_user(
            name=user_data.name,
            email=user_data.email,
            password=user_data.password,
            country=user_data.country,
            role=user_data.role
        )
        
        # Générer les tokens
        tokens = generate_tokens(user.id, user.role)
        
        return AuthResponse(
            success=True,
            user=user,
            tokens=tokens
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/v1/auth/login", response_model=AuthResponse, tags=["auth"])
@monitor_performance
def login_user(login_data: UserLogin):
    """Connexion utilisateur."""
    try:
        # Authentifier l'utilisateur
        user = authenticate_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Générer les tokens
        tokens = generate_tokens(user.id, user.role)
        
        return AuthResponse(
            success=True,
            user=user,
            tokens=tokens
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# Endpoint legacy pour compatibilité avec les tests
@app.post("/auth/login", tags=["auth", "legacy"])
@monitor_performance
def login_user_legacy(login_data: UserLogin):
    """Connexion utilisateur (endpoint legacy)."""
    return login_user(login_data)

@app.post("/api/v1/auth/refresh", response_model=Tokens, tags=["auth"])
@monitor_performance
def refresh_token(refresh_data: TokenRefresh):
    """Renouveler le token d'accès."""
    try:
        from .auth import jwt_decode
        payload = jwt_decode(refresh_data.refreshToken)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        user_data = db.get_user_by_id(user_id)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Générer nouveaux tokens
        tokens = generate_tokens(user_id, UserRole(user_data["role"]))
        
        return tokens
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

# =============================================================================
# DIAGNOSTIC APIs
# =============================================================================

@app.post("/api/v1/diagnose", response_model=DiagnoseResponse, tags=["diagnostic"])
@monitor_performance
def diagnose_plant_images(
    request: DiagnoseRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyser des images de plantes pour diagnostiquer les maladies."""
    try:
        start_time = time.time()
        diagnostic_id = _generate_diagnostic_id()
        
        # Traiter chaque image
        results = []
        overall_confidences = []
        
        for i, image_base64 in enumerate(request.images):
            try:
                # Décoder l'image base64
                image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
                image = Image.open(io.BytesIO(image_data))
                
                # Traiter l'image
                img_array = _process_image_for_yolo(image)
                model = get_model()
                predictions = model(img_array, conf=settings.model_confidence_threshold)
                
                # Analyser les résultats
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    if pred.boxes is not None and len(pred.boxes) > 0:
                        # Prendre la détection avec la plus haute confiance
                        best_detection = pred.boxes[0]
                        confidence = float(best_detection.conf[0])
                        class_id = int(best_detection.cls[0])
                        class_name = model.names[class_id]
                        
                        # Créer le résultat
                        disease = Disease(
                            id=f"{class_name.lower().replace(' ', '-')}",
                            name=class_name,
                            confidence=confidence * 100,
                            severity=SeverityLevel.HIGH if confidence > 0.8 else SeverityLevel.MEDIUM,
                            category="diseases"
                        )
                        
                        # Générer des recommandations basées sur la maladie
                        recommendations = _get_disease_recommendations(class_name)
                        
                        result = ImageResult(
                            imageIndex=i,
                            disease=disease,
                            symptoms=recommendations.get("symptoms", []),
                            solutions=recommendations.get("solutions", []),
                            prevention=recommendations.get("prevention", []),
                            treatmentUrgency=TreatmentUrgency.IMMEDIATE if confidence > 0.8 else TreatmentUrgency.URGENT,
                            affectedArea=0.75  # Estimation
                        )
                        
                        results.append(result)
                        overall_confidences.append(confidence * 100)
                    else:
                        # Aucune détection
                        disease = Disease(
                            id="healthy",
                            name="Plante saine",
                            confidence=95.0,
                            severity=SeverityLevel.LOW,
                            category="healthy"
                        )
                        
                        result = ImageResult(
                            imageIndex=i,
                            disease=disease,
                            symptoms=[],
                            solutions=["Continuer les soins habituels"],
                            prevention=["Maintenir de bonnes pratiques culturales"],
                            treatmentUrgency=TreatmentUrgency.MONITORING,
                            affectedArea=0.0
                        )
                        
                        results.append(result)
                        overall_confidences.append(95.0)
                else:
                    # Aucune prédiction
                    disease = Disease(
                        id="unknown",
                        name="Diagnostic incertain",
                        confidence=50.0,
                        severity=SeverityLevel.LOW,
                        category="unknown"
                    )
                    
                    result = ImageResult(
                        imageIndex=i,
                        disease=disease,
                        symptoms=[],
                        solutions=["Consulter un expert"],
                        prevention=[],
                        treatmentUrgency=TreatmentUrgency.MONITORING,
                        affectedArea=0.0
                    )
                    
                    results.append(result)
                    overall_confidences.append(50.0)
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Créer un résultat d'erreur
                disease = Disease(
                    id="error",
                    name="Erreur de traitement",
                    confidence=0.0,
                    severity=SeverityLevel.LOW,
                    category="error"
                )
                
                result = ImageResult(
                    imageIndex=i,
                    disease=disease,
                    symptoms=[],
                    solutions=["Vérifier la qualité de l'image"],
                    prevention=[],
                    treatmentUrgency=TreatmentUrgency.MONITORING,
                    affectedArea=0.0
                )
                
                results.append(result)
                overall_confidences.append(0.0)
        
        # Calculer la confiance globale
        overall_confidence = sum(overall_confidences) / len(overall_confidences) if overall_confidences else 0.0
        
        # Générer des recommandations globales
        recommendations = _generate_global_recommendations(results)
        
        # Sauvegarder le diagnostic
        diagnostic_data = {
            "id": diagnostic_id,
            "userId": current_user.id,
            "images": request.images,
            "results": [result.dict() for result in results],
            "location": request.location.dict(),
            "plantType": request.plantType,
            "status": "completed",
            "createdAt": datetime.utcnow()
        }
        db.save_diagnostic(diagnostic_data)
        
        processing_time = time.time() - start_time
        
        return DiagnoseResponse(
            success=True,
            diagnosticId=diagnostic_id,
            results=results,
            overallConfidence=overall_confidence,
            processingTime=processing_time,
            timestamp=datetime.utcnow(),
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Diagnostic error: {e}")
        raise HTTPException(status_code=500, detail="Diagnostic failed")

# =============================================================================
# DIAGNOSTICS MANAGEMENT APIs
# =============================================================================

@app.get("/api/v1/diagnostics", response_model=DiagnosticsListResponse, tags=["diagnostics"])
@monitor_performance
def get_diagnostics(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sortBy: str = Query("created_at", regex="^(created_at|updated_at)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: User = Depends(get_current_user)
):
    """Récupérer l'historique des diagnostics d'un utilisateur."""
    try:
        result = db.get_diagnostics(
            user_id=current_user.id,
            limit=limit,
            offset=offset,
            sort_by=sortBy,
            order=order
        )
        return DiagnosticsListResponse(**result)
    except Exception as e:
        logger.error(f"Get diagnostics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve diagnostics")

@app.get("/api/v1/diagnostics/{diagnostic_id}", tags=["diagnostics"])
@monitor_performance
def get_diagnostic_by_id(
    diagnostic_id: str,
    current_user: User = Depends(get_current_user)
):
    """Récupérer un diagnostic spécifique."""
    try:
        diagnostic = db.get_diagnostic_by_id(diagnostic_id)
        if not diagnostic:
            raise HTTPException(status_code=404, detail="Diagnostic not found")
        
        # Vérifier que l'utilisateur peut accéder à ce diagnostic
        if diagnostic["userId"] != current_user.id and current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {"success": True, "diagnostic": diagnostic}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get diagnostic error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve diagnostic")

@app.post("/api/v1/diagnostics/sync", tags=["diagnostics"])
@monitor_performance
def sync_diagnostics(
    sync_request: SyncRequest,
    current_user: User = Depends(get_current_user)
):
    """Synchroniser les diagnostics créés hors ligne."""
    try:
        synced_count = 0
        for diagnostic in sync_request.diagnostics:
            # Générer un nouvel ID pour éviter les conflits
            new_id = _generate_diagnostic_id()
            
            diagnostic_data = {
                "id": new_id,
                "userId": current_user.id,
                "images": diagnostic.images,
                "results": diagnostic.results,
                "location": diagnostic.location.dict(),
                "plantType": "unknown",
                "status": "completed",
                "createdAt": diagnostic.timestamp
            }
            
            db.save_diagnostic(diagnostic_data)
            synced_count += 1
        
        return {
            "success": True,
            "syncedCount": synced_count,
            "message": f"Successfully synced {synced_count} diagnostics"
        }
    except Exception as e:
        logger.error(f"Sync diagnostics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync diagnostics")

# =============================================================================
# WEATHER APIs
# =============================================================================

@app.get("/api/v1/weather", response_model=WeatherResponse, tags=["weather"])
@monitor_performance
def get_weather_data(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    includeForecast: bool = Query(True),
    current_user: User = Depends(get_current_user)
):
    """Récupérer les données météo agricoles."""
    try:
        return weather_service.get_weather_data(lat, lng, includeForecast)
    except Exception as e:
        logger.error(f"Weather data error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve weather data")

# =============================================================================
# DISEASES KNOWLEDGE BASE APIs
# =============================================================================

@app.get("/api/v1/diseases", response_model=DiseasesListResponse, tags=["diseases"])
@monitor_performance
def search_diseases(
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user)
):
    """Rechercher des maladies dans la base de connaissances."""
    try:
        result = diseases_service.search_diseases(search, category, limit, offset)
        return DiseasesListResponse(**result)
    except Exception as e:
        logger.error(f"Search diseases error: {e}")
        raise HTTPException(status_code=500, detail="Failed to search diseases")

@app.get("/api/v1/diseases/{disease_id}", tags=["diseases"])
@monitor_performance
def get_disease_by_id(
    disease_id: str,
    current_user: User = Depends(get_current_user)
):
    """Récupérer les détails d'une maladie spécifique."""
    try:
        disease = diseases_service.get_disease_by_id(disease_id)
        if not disease:
            raise HTTPException(status_code=404, detail="Disease not found")
        
        return {"success": True, "disease": disease}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get disease error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve disease")

# =============================================================================
# USER MANAGEMENT APIs
# =============================================================================

@app.get("/api/v1/users/profile", response_model=UserProfileResponse, tags=["users"])
@monitor_performance
def get_user_profile(current_user: User = Depends(get_current_user)):
    """Récupérer le profil de l'utilisateur connecté."""
    try:
        # Récupérer les statistiques de l'utilisateur
        user_stats = db.get_user_stats(current_user.id)
        
        profile = UserProfile(
            id=current_user.id,
            name=current_user.name,
            email=current_user.email,
            country=current_user.country,
            role=current_user.role,
            preferences=UserPreferences(**current_user.preferences) if current_user.preferences else UserPreferences(),
            stats=user_stats
        )
        
        return UserProfileResponse(success=True, user=profile)
    except Exception as e:
        logger.error(f"Get user profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")

# =============================================================================
# LEGACY APIs (pour compatibilité)
# =============================================================================

@app.get("/health", tags=["system"])
@monitor_performance
def health_check():
    """Vérifier l'état du service."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": settings.app_version,
        "model_loaded": _model is not None
    }

@app.get("/model/info", tags=["model"])
@monitor_performance
def get_model_info():
    """Informations sur le modèle chargé."""
    try:
        model = get_model()
        return {
            "model_path": str(_model_path),
            "model_exists": _model_path.exists(),
            "max_upload_bytes": settings.max_upload_bytes,
            "classes": list(model.names.values()) if model else [],
            "num_classes": len(model.names) if model else 0
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

@app.get("/classes", response_model=List[ClassItem], tags=["model"])
@monitor_performance
def get_classes():
    """Lister les classes du modèle."""
    try:
        model = get_model()
        classes = []
        for class_id, class_name in model.names.items():
            recommendations = _get_disease_recommendations(class_name)
            classes.append(ClassItem(
                name=class_name,
                recommendations=recommendations.get("solutions", [])
            ))
        return classes
    except Exception as e:
        logger.error(f"Get classes error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve classes")

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
@monitor_performance
def predict_single_image(
    file: UploadFile = File(...),
    user_id: str = "anonymous"
):
    """Prédire sur une image unique (API legacy)."""
    try:
        # Lire l'image
        image = _read_image_from_upload(file)
        img_array = _process_image_for_yolo(image)
        
        # Faire la prédiction
        model = get_model()
        predictions = model(img_array, conf=settings.model_confidence_threshold)
        
        # Traiter les résultats
        results = []
        detected_classes = []
        
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            if pred.boxes is not None:
                for box in pred.boxes:
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Coordonnées de la bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = [x1, y1, x2, y2]
                    
                    # Recommandations
                    recommendations = _get_disease_recommendations(class_name)
                    
                    results.append(Prediction(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        recommendations=recommendations.get("solutions", [])
                    ))
                    
                    if class_name not in detected_classes:
                        detected_classes.append(class_name)
        
        # Sauvegarder la prédiction
        db.save_prediction(
            user_id=user_id,
            image_filename=file.filename,
            image_width=image.width,
            image_height=image.height,
            predictions=[pred.dict() for pred in results],
            processing_time_ms=int(time.time() * 1000)
        )
        
        return PredictResponse(
            predictions=results,
            image_width=image.width,
            image_height=image.height,
            detected_classes=detected_classes,
            uncertain=len(results) == 0,
            quality={}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Endpoint legacy pour compatibilité avec les tests
@app.post("/predict-multi", tags=["inference", "legacy"])
@monitor_performance
def predict_multi_legacy():
    """Endpoint legacy pour prédiction multi-images."""
    raise HTTPException(
        status_code=410, 
        detail="This endpoint is deprecated. Use /api/v1/diagnose instead."
    )

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _get_disease_recommendations(class_name: str) -> Dict[str, List[str]]:
    """Générer des recommandations basées sur le nom de la classe."""
    # Base de connaissances simple
    recommendations_db = {
        "mildiou": {
            "symptoms": ["Taches brunes sur les feuilles", "Pourriture des fruits"],
            "solutions": ["Traitement au cuivre", "Améliorer l'aération"],
            "prevention": ["Rotation des cultures", "Éviter l'humidité excessive"]
        },
        "oïdium": {
            "symptoms": ["Poudre blanche sur les feuilles", "Feuilles qui jaunissent"],
            "solutions": ["Traitement au soufre", "Réduire l'humidité"],
            "prevention": ["Éviter l'humidité excessive", "Espacement des plants"]
        },
        "anthracnose": {
            "symptoms": ["Taches noires sur les fruits", "Pourriture"],
            "solutions": ["Supprimer les parties atteintes", "Traitement fongicide"],
            "prevention": ["Rotation des cultures", "Éviter l'humidité"]
        }
    }
    
    # Rechercher des correspondances partielles
    for disease, recs in recommendations_db.items():
        if disease.lower() in class_name.lower():
            return recs
    
    # Recommandations par défaut
    return {
        "symptoms": ["Symptômes à identifier"],
        "solutions": ["Consulter un expert", "Améliorer les conditions de culture"],
        "prevention": ["Maintenir de bonnes pratiques culturales"]
    }

def _generate_global_recommendations(results: List[ImageResult]) -> Recommendations:
    """Générer des recommandations globales basées sur tous les résultats."""
    if not results:
        return Recommendations(
            priority=SeverityLevel.LOW,
            nextSteps=["Aucune maladie détectée", "Continuer la surveillance"]
        )
    
    # Déterminer la priorité basée sur la sévérité
    max_severity = max(result.disease.severity for result in results)
    high_confidence_results = [r for r in results if r.disease.confidence > 80]
    
    if max_severity == SeverityLevel.HIGH or len(high_confidence_results) > 0:
        priority = SeverityLevel.HIGH
        next_steps = [
            "Traiter immédiatement les maladies détectées",
            "Isoler les plants atteints",
            "Surveiller l'évolution"
        ]
    elif max_severity == SeverityLevel.MEDIUM:
        priority = SeverityLevel.MEDIUM
        next_steps = [
            "Planifier un traitement préventif",
            "Améliorer les conditions de culture",
            "Surveiller régulièrement"
        ]
    else:
        priority = SeverityLevel.LOW
        next_steps = [
            "Continuer la surveillance",
            "Maintenir de bonnes pratiques"
        ]
    
    return Recommendations(priority=priority, nextSteps=next_steps)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)