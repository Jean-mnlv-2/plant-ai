from __future__ import annotations

import io
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
import hashlib
import hmac
import json
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import cv2


# -----------------------------------------------------------------------------
# Settings (must import first!)
# -----------------------------------------------------------------------------
from .settings import settings
from .admin import router as admin_router
from .database import db
from .metrics import performance_monitor, monitor_performance, calculate_prediction_metrics
from .dataset_manager import dataset_manager


# -----------------------------------------------------------------------------
# Logging setup (needs settings)
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
app = FastAPI(title=settings.app_name, version=settings.app_version)

# Enable CORS for PWA/mobile/web clients (production-ready configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Templates & static (admin UI)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
_model = None  # type: ignore[var-annotated]
_model_path = settings.model_file


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from ultralytics import YOLO  # import locally to speed cold startup
                except Exception as e:  # pragma: no cover
                    logger.exception("Ultralytics import failed")
                    raise HTTPException(status_code=500, detail=f"Ultralytics import failed: {e}")

                if not _model_path.exists():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model file not found at '{_model_path.as_posix()}'. Train the model first.",
                    )
                try:
                    logger.info("Loading YOLO model from %s", _model_path)
                    _model = YOLO(str(_model_path))
                except Exception as e:
                    logger.exception("Model load failed")
                    raise HTTPException(status_code=500, detail=f"Model load failed: {e}")
    return _model



# -----------------------------------------------------------------------------
# Minimal JWT (HS256) helpers
# -----------------------------------------------------------------------------
_JWT_SECRET_FILE = settings.jwt_secret_file
if _JWT_SECRET_FILE.exists():
    _JWT_SECRET = _JWT_SECRET_FILE.read_text(encoding="utf-8").strip()
else:
    _JWT_SECRET = os.getenv("PLANT_AI_JWT_SECRET", "dev-secret-change-me")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sign(msg: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return _b64url(sig)


def jwt_encode(payload: Dict[str, object], secret: str = _JWT_SECRET) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = _sign(signing_input, secret)
    return f"{header_b64}.{payload_b64}.{sig}"


def jwt_decode(token: str, secret: str = _JWT_SECRET) -> Dict[str, object]:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid token format")
    header_b64, payload_b64, sig = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = _sign(signing_input, secret)
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=401, detail="Invalid token signature")
    pad = "=" * (-len(payload_b64) % 4)
    try:
        payload_json = base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii")).decode("utf-8")
        payload = json.loads(payload_json)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and int(time.time()) > int(exp):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload  # type: ignore[return-value]


def get_current_user_id(authorization: Optional[str] = None) -> Optional[str]:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    payload = jwt_decode(token)
    sub = payload.get("sub")
    return str(sub) if sub is not None else None


def _analyze_single_image(image_data: bytes) -> Dict:
    """Analyse une seule image et retourne les résultats"""
    try:
        # Convertir bytes en PIL Image
        img = Image.open(io.BytesIO(image_data))
        np_img = _pil_to_numpy(img)
        
        # Run inference
        model = get_model()
        results = model.predict(source=[np_img], verbose=False)
        
        names: Dict[int, str] = getattr(model, "names", {})
        predictions: List[Prediction] = []
        
        if results:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.zeros((0, 4))
                conf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.zeros((0,))
                cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.zeros((0,))
                
                for i in range(xyxy.shape[0]):
                    bbox = xyxy[i].tolist()
                    confidence = float(conf[i]) if i < conf.shape[0] else 0.0
                    cls_id = int(cls[i]) if i < cls.shape[0] else 0
                    class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    recs = get_recommendations_for_class(class_name)
                    predictions.append(
                        Prediction(class_name=class_name, confidence=confidence, bbox=bbox, recommendations=recs)
                    )
        
        return {
            "image_width": img.width,
            "image_height": img.height,
            "predictions": [p.dict() for p in predictions],
            "num_detections": len(predictions),
            "diseases": list(set([p.class_name for p in predictions]))
        }
    except Exception as e:
        logger.error(f"Erreur analyse image: {e}")
        return {
            "image_width": 0,
            "image_height": 0,
            "predictions": [],
            "num_detections": 0,
            "diseases": [],
            "error": str(e)
        }


def _consolidate_diagnosis(image_results: List[Dict], plant_id: str) -> MultiImageDiagnosis:
    """Consolide les résultats de plusieurs images en un diagnostic complet"""
    
    # Collecter toutes les maladies détectées
    all_diseases = []
    all_predictions = []
    total_detections = 0
    
    for result in image_results:
        if "error" not in result:
            all_diseases.extend(result["diseases"])
            all_predictions.extend(result["predictions"])
            total_detections += result["num_detections"]
    
    # Calculer les scores de confiance par maladie
    disease_confidence = {}
    disease_recommendations = {}
    
    for pred in all_predictions:
        disease = pred["class_name"]
        confidence = pred["confidence"]
        
        if disease not in disease_confidence:
            disease_confidence[disease] = []
            disease_recommendations[disease] = set()
        
        disease_confidence[disease].append(confidence)
        disease_recommendations[disease].update(pred["recommendations"])
    
    # Calculer la confiance moyenne par maladie
    avg_confidence = {}
    for disease, confidences in disease_confidence.items():
        avg_confidence[disease] = sum(confidences) / len(confidences)
    
    # Déterminer le niveau de sévérité
    max_confidence = max(avg_confidence.values()) if avg_confidence else 0
    if max_confidence >= 0.8:
        severity = "élevé"
    elif max_confidence >= 0.6:
        severity = "moyen"
    else:
        severity = "faible"
    
    # Générer le résumé du diagnostic
    diseases_found = list(avg_confidence.keys())
    if not diseases_found:
        diagnostic_summary = "Aucune maladie détectée. La plante semble en bonne santé."
    elif len(diseases_found) == 1:
        disease = diseases_found[0]
        confidence = avg_confidence[disease]
        diagnostic_summary = f"Maladie détectée: {disease} (confiance: {confidence:.1%})"
    else:
        primary_disease = max(avg_confidence.items(), key=lambda x: x[1])
        diagnostic_summary = f"Maladies multiples détectées. Principale: {primary_disease[0]} ({primary_disease[1]:.1%})"
    
    # Consolider les recommandations
    all_recommendations = set()
    for recommendations in disease_recommendations.values():
        all_recommendations.update(recommendations)
    
    return MultiImageDiagnosis(
        plant_id=plant_id,
        images_analyzed=len(image_results),
        total_detections=total_detections,
        diseases_found=diseases_found,
        confidence_scores=avg_confidence,
        diagnostic_summary=diagnostic_summary,
        recommendations=list(all_recommendations),
        severity_level=severity,
        image_results=image_results
    )


def get_recommendations_for_class(class_name: str) -> List[str]:
    """Get agronomic recommendations for a detected class."""
    return settings.class_to_recommendations.get(
        class_name,
        [
            "Isoler la plante affectée",
            "Observer l'évolution 48-72h", 
            "Consulter un conseiller agronomique si aggravation",
        ],
    )


# -----------------------------------------------------------------------------
# Schemas
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


class MultiImageDiagnosis(BaseModel):
    """Diagnostic basé sur plusieurs images d'une même plante"""
    plant_id: str = Field(..., description="Identifiant unique de la plante")
    images_analyzed: int = Field(..., description="Nombre d'images analysées")
    total_detections: int = Field(..., description="Nombre total de détections")
    diseases_found: List[str] = Field(..., description="Maladies détectées")
    confidence_scores: Dict[str, float] = Field(..., description="Scores de confiance par maladie")
    diagnostic_summary: str = Field(..., description="Résumé du diagnostic")
    recommendations: List[str] = Field(..., description="Recommandations consolidées")
    severity_level: str = Field(..., description="Niveau de sévérité: faible/moyen/élevé")
    image_results: List[Dict] = Field(..., description="Résultats détaillés par image")


class MultiImageRequest(BaseModel):
    """Requête pour analyse multi-images"""
    plant_id: str = Field(..., description="Identifiant de la plante")
    images: List[bytes] = Field(..., description="Images en base64")
    analysis_type: str = Field(default="comprehensive", description="Type d'analyse")


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


# -----------------------------------------------------------------------------
# Database and performance monitoring (replaces in-memory storage)
# -----------------------------------------------------------------------------
# HISTORY is now handled by SQLite database


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
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Image must be RGB")
    return arr


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", tags=["system"], summary="Vérifier l'état du service")
@monitor_performance("health_check")
async def health() -> Dict[str, str]:
    """Vérification de santé du système avec métriques."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version
    }


@app.get("/classes", response_model=List[ClassItem], tags=["model"], summary="Lister les classes du modèle")
@monitor_performance("list_classes")
async def list_classes() -> List[ClassItem]:
    """Liste toutes les classes de maladies détectables avec leurs recommandations."""
    model = get_model()
    names: Dict[int, str] = getattr(model, "names", {})  # type: ignore[assignment]
    # Ultralytics names is usually id->name dict
    class_names = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names)
    result: List[ClassItem] = []
    for c in class_names:
        result.append(ClassItem(name=c, recommendations=get_recommendations_for_class(c)))
    return result


@app.post("/auth/login", tags=["auth"], summary="Connexion et génération de token")
@monitor_performance("auth_login")
async def auth_login(username: str, password: str) -> Dict[str, object]:
    """Authentification utilisateur avec génération de token JWT."""
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    now = int(time.time())
    token = jwt_encode({"sub": username, "iat": now, "exp": now + settings.jwt_ttl_seconds})
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.jwt_ttl_seconds}


@app.post("/predict-multi", response_model=MultiImageDiagnosis, tags=["inference"], summary="Diagnostic multi-images d'une plante")
@monitor_performance("predict_multi")
async def predict_multi_images(
    plant_id: str,
    images: list[UploadFile] = File(..., description="Jusqu'à 10 images de la même plante"),
    analysis_type: str = "comprehensive",
    user_id: Optional[str] = None,
    current_user: Optional[str] = Depends(get_current_user_id),
):
    """
    Analyse plusieurs images d'une même plante pour un diagnostic complet.
    
    - **plant_id**: Identifiant unique de la plante
    - **images**: Liste d'images (feuilles, tiges, fruits, etc.)
    - **analysis_type**: Type d'analyse (comprehensive, quick, detailed)
    
    Retourne un diagnostic consolidé avec:
    - Maladies détectées avec scores de confiance
    - Niveau de sévérité
    - Recommandations consolidées
    - Résumé du diagnostic
    """
    start_time = time.time()
    
    if len(images) < 1:
        raise HTTPException(status_code=400, detail="Au moins une image est requise")
    
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images par analyse")
    
    # Analyser chaque image
    image_results = []
    for i, image in enumerate(images):
        try:
            # Lire l'image
            image_data = await image.read()
            
            # Valider le type de contenu
            if image.content_type not in settings.allowed_content_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Type de fichier non supporté: {image.content_type}"
                )
            
            # Analyser l'image
            result = _analyze_single_image(image_data)
            result["image_index"] = i
            result["filename"] = image.filename or f"image_{i}.jpg"
            image_results.append(result)
            
        except Exception as e:
            logger.error(f"Erreur analyse image {i}: {e}")
            image_results.append({
                "image_index": i,
                "filename": image.filename or f"image_{i}.jpg",
                "error": str(e),
                "predictions": [],
                "num_detections": 0,
                "diseases": []
            })
    
    # Consolider le diagnostic
    diagnosis = _consolidate_diagnosis(image_results, plant_id)
    
    # Calculer les métriques de performance
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    # Sauvegarder en base de données
    effective_user = user_id or current_user or "anonymous"
    
    try:
        # Sauvegarder le diagnostic multi-images
        db.save_prediction(
            user_id=effective_user,
            image_filename=None,
            image_width=0,  # Multi-images
            image_height=0,
            predictions=[],  # Résultats consolidés dans diagnosis
            processing_time_ms=processing_time_ms,
            confidence_avg=sum(diagnosis.confidence_scores.values()) / len(diagnosis.confidence_scores) if diagnosis.confidence_scores else 0.0,
            num_detections=diagnosis.total_detections
        )
        
        # Tracker les métriques
        performance_monitor.track_prediction_metrics(
            user_id=effective_user,
            num_detections=diagnosis.total_detections,
            avg_confidence=sum(diagnosis.confidence_scores.values()) / len(diagnosis.confidence_scores) if diagnosis.confidence_scores else 0,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Erreur sauvegarde diagnostic multi-images: {e}")
    
    logger.info(f"Diagnostic multi-images terminé pour {plant_id}: {diagnosis.diseases_found}")
    return diagnosis


@app.post("/predict", response_model=PredictResponse, tags=["inference"], summary="Prédire sur une image unique")
@monitor_performance("predict")
async def predict(
    image: UploadFile = File(...),
    user_id: Optional[str] = None,
    current_user: Optional[str] = Depends(get_current_user_id),
):
    """Prédiction principale avec métriques de performance et sauvegarde en base."""
    start_time = time.time()
    
    # Read and validate image
    img = _read_image_from_upload(image)
    np_img = _pil_to_numpy(img)

    # Qualité d'image: flou (variance du Laplacien)
    quality: Dict[str, float] = {}
    try:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        lap_var = 0.0
    quality["blur_variance"] = lap_var

    # Run inference
    model = get_model()
    try:
        results = model.predict(source=[np_img], verbose=False)  # batch of one
    except Exception as e:  # pragma: no cover
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    names: Dict[int, str] = getattr(model, "names", {})  # type: ignore[assignment]
    predictions: List[Prediction] = []

    if results:
        r = results[0]
        # Boxes in xyxy format
        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.zeros((0, 4))
            conf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.zeros((0,))
            cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else np.zeros((0,))
            for i in range(xyxy.shape[0]):
                bbox = xyxy[i].tolist()
                confidence = float(conf[i]) if i < conf.shape[0] else 0.0
                cls_id = int(cls[i]) if i < cls.shape[0] else 0
                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                recs = get_recommendations_for_class(class_name)
                predictions.append(
                    Prediction(class_name=class_name, confidence=confidence, bbox=bbox, recommendations=recs)
                )

    # Calculer les métriques de performance
    processing_time_ms = int((time.time() - start_time) * 1000)
    prediction_metrics = calculate_prediction_metrics([p.dict() for p in predictions])
    
    # Incertitude: seuils + qualité
    max_conf = max([p.confidence for p in predictions], default=0.0)
    min_conf = min([p.confidence for p in predictions], default=1.0) if predictions else 1.0
    quality_ok = (lap_var >= settings.blur_variance_threshold) if settings.enable_quality_checks else True
    uncertain = False
    uncertain_reason = None
    if settings.enable_uncertainty_routing:
        if not predictions:
            uncertain = True
            uncertain_reason = "no_detection"
        elif max_conf < settings.confidence_threshold_low:
            uncertain = True
            uncertain_reason = "low_confidence"
        elif not quality_ok:
            uncertain = True
            uncertain_reason = "low_quality_blur"
    
    response = PredictResponse(
        predictions=predictions,
        image_width=img.width,
        image_height=img.height,
        detected_classes=[names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names),
        uncertain=uncertain,
        quality=quality,
    )

    # Sauvegarder en base de données
    effective_user = user_id or current_user or "anonymous"
    try:
        prediction_id = db.save_prediction(
            user_id=effective_user,
            image_filename=image.filename,
            image_width=img.width,
            image_height=img.height,
            predictions=[p.dict() for p in predictions],
            processing_time_ms=processing_time_ms,
            confidence_avg=prediction_metrics["avg_confidence"],
            num_detections=prediction_metrics["num_detections"]
        )
        
        # Mettre à jour les métriques en mémoire
        performance_monitor.track_prediction_metrics(
            user_id=effective_user,
            num_detections=prediction_metrics["num_detections"],
            avg_confidence=prediction_metrics["avg_confidence"],
            processing_time_ms=processing_time_ms
        )
        
        logger.info(f"Prediction saved with ID {prediction_id} for user {effective_user}")
        
    except Exception as e:
        logger.error(f"Failed to save prediction to database: {e}")
        # Ne pas faire échouer la prédiction si la sauvegarde échoue

    # Enregistrer cas incertain pour annotation ultérieure
    if uncertain:
        try:
            db.save_uncertain_case(
                user_id=effective_user,
                image_filename=image.filename,
                image_width=img.width,
                image_height=img.height,
                predictions=[p.dict() for p in predictions],
                reason=uncertain_reason or "uncertain",
                min_confidence=min_conf,
                max_confidence=max_conf,
            )
        except Exception as e:
            logger.error(f"Failed to save uncertain case: {e}")

    return response


class FeedbackItem(BaseModel):
    user_id: Optional[str] = None
    image_filename: Optional[str] = None
    original_class: Optional[str] = None
    corrected_class: str
    notes: Optional[str] = None
    prediction_id: Optional[int] = None


@app.post("/feedback", tags=["feedback"], summary="Soumettre un feedback de correction")
@monitor_performance("post_feedback")
async def post_feedback(item: FeedbackItem, current_user: Optional[str] = Depends(get_current_user_id)) -> Dict[str, Any]:
    """Enregistre un feedback utilisateur pour correction de classe et annotation."""
    effective_user = item.user_id or current_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required or provide user_id")
    try:
        feedback_id = db.save_feedback(
            user_id=effective_user,
            image_filename=item.image_filename,
            original_class=item.original_class,
            corrected_class=item.corrected_class,
            notes=item.notes,
            prediction_id=item.prediction_id,
        )
        return {"status": "ok", "feedback_id": feedback_id}
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@app.get("/uncertain", tags=["feedback"], summary="Lister les cas incertains")
@monitor_performance("list_uncertain")
async def list_uncertain(limit: int = 100) -> Dict[str, Any]:
    """Liste les cas incertains pour annotation."""
    try:
        rows = db.list_uncertain_cases(limit)
        return {"items": rows, "count": len(rows)}
    except Exception as e:
        logger.error(f"Failed to list uncertain cases: {e}")
        raise HTTPException(status_code=500, detail="Failed to list uncertain cases")


@app.get("/history", response_model=List[HistoryItem], tags=["history"], summary="Historique des prédictions d'un utilisateur")
@monitor_performance("get_history")
async def get_history(
    user_id: Optional[str] = None, 
    current_user: Optional[str] = Depends(get_current_user_id),
    limit: int = 100
) -> List[HistoryItem]:
    """Récupère l'historique des prédictions d'un utilisateur depuis la base de données."""
    effective_user = user_id or current_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required or provide user_id")
    
    try:
        history_data = db.get_user_history(effective_user, limit)
        history_items = []
        
        for row in history_data:
            predictions_data = json.loads(row["predictions_json"])
            predictions = [Prediction(**p) for p in predictions_data]
            
            history_item = HistoryItem(
                user_id=row["user_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                predictions=predictions
            )
            history_items.append(history_item)
        
        return history_items
        
    except Exception as e:
        logger.error(f"Failed to retrieve history for user {effective_user}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@app.post("/history", response_model=HistoryItem, tags=["history"], summary="Ajouter un item d'historique manuellement")
@monitor_performance("post_history")
async def post_history(item: HistoryPost, current_user: Optional[str] = Depends(get_current_user_id)) -> HistoryItem:
    """Sauvegarde manuellement un historique de prédiction."""
    effective_user = item.user_id or current_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required or provide user_id")
    
    try:
        # Sauvegarder en base de données
        prediction_metrics = calculate_prediction_metrics([p.dict() for p in item.predictions])
        
        prediction_id = db.save_prediction(
            user_id=effective_user,
            image_filename=None,
            image_width=0,
            image_height=0,
            predictions=[p.dict() for p in item.predictions],
            processing_time_ms=0,
            confidence_avg=prediction_metrics["avg_confidence"],
            num_detections=prediction_metrics["num_detections"]
        )
        
        history_item = HistoryItem(
            user_id=effective_user, 
            timestamp=datetime.utcnow(), 
            predictions=item.predictions
        )
        
        return history_item
        
    except Exception as e:
        logger.error(f"Failed to save history for user {effective_user}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save history")


# -----------------------------------------------------------------------------
# Nouveaux endpoints utiles
# -----------------------------------------------------------------------------

@app.get("/model/info", tags=["model"], summary="Informations sur le modèle")
@monitor_performance("model_info")
async def model_info() -> Dict[str, Any]:
    """Informations sur le modèle chargé."""
    try:
        model = get_model()
        names: Dict[int, str] = getattr(model, "names", {})
        class_names = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names)
        
        return {
            "model_path": str(settings.model_file),
            "model_exists": settings.model_file.exists(),
            "classes": class_names,
            "num_classes": len(class_names),
            "input_size": 640,
            "confidence_threshold": 0.5,
            "supported_formats": settings.allowed_content_types,
            "max_upload_size_mb": settings.max_upload_bytes // (1024 * 1024)
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.get("/stats/performance", tags=["stats"], summary="Statistiques de performance")
@monitor_performance("performance_stats")
async def performance_stats(hours: int = 24) -> Dict[str, Any]:
    """Statistiques de performance du système."""
    try:
        stats = db.get_performance_stats(hours)
        global_metrics = performance_monitor.get_global_metrics()
        
        return {
            "performance": stats,
            "global_metrics": global_metrics,
            "period_hours": hours
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance statistics")


@app.get("/stats/model-usage", tags=["stats"], summary="Statistiques d'usage du modèle")
@monitor_performance("model_usage_stats")
async def model_usage_stats(days: int = 7) -> Dict[str, Any]:
    """Statistiques d'utilisation du modèle."""
    try:
        stats = db.get_model_usage_stats(days)
        return stats
    except Exception as e:
        logger.error(f"Failed to get model usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model usage statistics")


@app.get("/stats/user/{user_id}", tags=["stats"], summary="Statistiques par utilisateur")
@monitor_performance("user_stats")
async def user_stats(user_id: str, current_user: Optional[str] = Depends(get_current_user_id)) -> Dict[str, Any]:
    """Statistiques spécifiques à un utilisateur."""
    effective_user = user_id or current_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Récupérer l'historique de l'utilisateur
        history_data = db.get_user_history(effective_user, limit=1000)
        
        if not history_data:
            return {
                "user_id": effective_user,
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "total_detections": 0,
                "avg_processing_time_ms": 0.0,
                "most_detected_classes": [],
                "recent_activity": []
            }
        
        # Calculer les statistiques
        total_predictions = len(history_data)
        total_detections = sum(row["num_detections"] for row in history_data)
        avg_confidence = sum(row["confidence_avg"] for row in history_data) / total_predictions
        avg_processing_time = sum(row["processing_time_ms"] for row in history_data) / total_predictions
        
        # Classes les plus détectées
        class_counts = {}
        for row in history_data:
            predictions_data = json.loads(row["predictions_json"])
            for pred in predictions_data:
                class_name = pred.get("class_name", "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        most_detected_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Activité récente (dernières 10 prédictions)
        recent_activity = []
        for row in history_data[:10]:
            recent_activity.append({
                "timestamp": row["timestamp"],
                "num_detections": row["num_detections"],
                "avg_confidence": row["confidence_avg"],
                "processing_time_ms": row["processing_time_ms"]
            })
        
        return {
            "user_id": effective_user,
            "total_predictions": total_predictions,
            "avg_confidence": round(avg_confidence, 3),
            "total_detections": total_detections,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "most_detected_classes": most_detected_classes,
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"Failed to get user stats for {effective_user}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user statistics")


@app.post("/admin/cleanup", tags=["admin"], summary="Nettoyage des anciennes données")
@monitor_performance("admin_cleanup")
async def admin_cleanup(days_to_keep: int = 30) -> Dict[str, str]:
    """Nettoyage des anciennes données (endpoint admin)."""
    try:
        db.cleanup_old_data(days_to_keep)
        return {
            "status": "success",
            "message": f"Cleaned up data older than {days_to_keep} days",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup old data")


@app.get("/admin/health-detailed", tags=["admin"], summary="Santé détaillée du système")
@monitor_performance("detailed_health")
async def detailed_health() -> Dict[str, Any]:
    """Vérification de santé détaillée du système."""
    try:
        # Vérifier la base de données
        db_status = "ok"
        try:
            db.get_performance_stats(1)  # Test simple
        except Exception:
            db_status = "error"
        
        # Vérifier le modèle
        model_status = "ok"
        try:
            model = get_model()
            if not settings.model_file.exists():
                model_status = "missing"
        except Exception:
            model_status = "error"
        
        # Statistiques rapides
        global_metrics = performance_monitor.get_global_metrics()
        
        return {
            "status": "ok" if db_status == "ok" and model_status == "ok" else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "database": db_status,
            "model": model_status,
            "model_path": str(settings.model_file),
            "metrics": global_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# -----------------------------------------------------------------------------
# Dataset Management Endpoints
# -----------------------------------------------------------------------------

@app.post("/admin/dataset/reset", tags=["admin", "dataset"], summary="Réinitialiser le dataset")
@monitor_performance("reset_dataset")
async def reset_dataset(current_user: Optional[str] = Depends(get_current_user_id)) -> Dict[str, Any]:
    """
    Réinitialise complètement le dataset.
    
    - Supprime tous les datasets existants (raw, cleaned, YOLO, COCO)
    - Nettoie la base de données des prédictions
    - Remet à zéro tous les répertoires de données
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        result = await dataset_manager.reset_dataset()
        logger.info(f"Dataset réinitialisé par {current_user}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation du dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la réinitialisation: {str(e)}")


@app.post("/admin/dataset/upload", tags=["admin", "dataset"], summary="Uploader un nouveau dataset")
@monitor_performance("upload_dataset")
async def upload_dataset(
    train_images: List[UploadFile] = File(..., description="Images d'entraînement"),
    test_images: List[UploadFile] = File(..., description="Images de test"),
    train_annotations: UploadFile = File(..., description="Annotations d'entraînement (CSV)"),
    test_annotations: UploadFile = File(..., description="Annotations de test (CSV)"),
    current_user: Optional[str] = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Upload un nouveau dataset avec images et annotations.
    
    - **train_images**: Images d'entraînement (JPEG/PNG)
    - **test_images**: Images de test (JPEG/PNG)
    - **train_annotations**: Fichier CSV des annotations d'entraînement
    - **test_annotations**: Fichier CSV des annotations de test
    
    Format CSV requis: filename,width,height,class,xmin,ymin,xmax,ymax
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if len(train_images) == 0 or len(test_images) == 0:
        raise HTTPException(status_code=400, detail="Au moins une image d'entraînement et une image de test sont requises")
    
    try:
        result = await dataset_manager.upload_dataset(
            train_images=train_images,
            test_images=test_images,
            train_annotations=train_annotations,
            test_annotations=test_annotations
        )
        logger.info(f"Dataset uploadé par {current_user}: {result['train_images']} train, {result['test_images']} test")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(e)}")


@app.post("/admin/dataset/process", tags=["admin", "dataset"], summary="Traiter le dataset uploadé")
@monitor_performance("process_dataset")
async def process_dataset(current_user: Optional[str] = Depends(get_current_user_id)) -> Dict[str, Any]:
    """
    Traite le dataset uploadé (nettoyage, conversion, validation).
    
    Étapes:
    1. Nettoyage des noms de fichiers
    2. Analyse du dataset
    3. Conversion vers format YOLO
    4. Conversion vers format COCO
    5. Vérification de l'intégrité
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        result = await dataset_manager.process_dataset()
        logger.info(f"Dataset traité par {current_user}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors du traitement du dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")


@app.post("/admin/dataset/train", tags=["admin", "dataset"], summary="Entraîner un nouveau modèle")
@monitor_performance("train_model")
async def train_model(
    epochs: int = 50,
    batch_size: int = 16,
    image_size: int = 640,
    current_user: Optional[str] = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """
    Entraîne un nouveau modèle avec le dataset traité.
    
    - **epochs**: Nombre d'époques d'entraînement (défaut: 50)
    - **batch_size**: Taille du batch (défaut: 16)
    - **image_size**: Taille des images (défaut: 640)
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if epochs < 1 or epochs > 1000:
        raise HTTPException(status_code=400, detail="Le nombre d'époques doit être entre 1 et 1000")
    
    if batch_size < 1 or batch_size > 64:
        raise HTTPException(status_code=400, detail="La taille du batch doit être entre 1 et 64")
    
    try:
        result = await dataset_manager.train_model(
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size
        )
        logger.info(f"Modèle entraîné par {current_user}: {epochs} époques, batch={batch_size}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement: {str(e)}")


@app.get("/admin/dataset/status", tags=["admin", "dataset"], summary="Statut du dataset")
@monitor_performance("dataset_status")
async def get_dataset_status(current_user: Optional[str] = Depends(get_current_user_id)) -> Dict[str, Any]:
    """
    Retourne le statut actuel du dataset.
    
    Inclut:
    - Données brutes (images et annotations)
    - Données nettoyées
    - Dataset YOLO formaté
    - Dataset COCO formaté
    - Modèle entraîné
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        status = await dataset_manager.get_dataset_status()
        return status
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du statut: {str(e)}")


# -----------------------------------------------------------------------------
# Notes
# - Database integration completed with SQLite
# - Performance monitoring implemented
# - New useful endpoints added
# - Dataset management endpoints added
# - All endpoints now have proper error handling and logging
# -----------------------------------------------------------------------------


