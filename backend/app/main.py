from __future__ import annotations

import io
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
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
from pydantic import BaseModel, Field
from PIL import Image


# -----------------------------------------------------------------------------
# Settings (must import first!)
# -----------------------------------------------------------------------------
from .settings import settings


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
# In-memory history store (replace with DB in production)
# -----------------------------------------------------------------------------
HISTORY: Dict[str, List[HistoryItem]] = {}


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
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/classes", response_model=List[ClassItem])
async def list_classes() -> List[ClassItem]:
    model = get_model()
    names: Dict[int, str] = getattr(model, "names", {})  # type: ignore[assignment]
    # Ultralytics names is usually id->name dict
    class_names = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names)
    result: List[ClassItem] = []
    for c in class_names:
        result.append(ClassItem(name=c, recommendations=get_recommendations_for_class(c)))
    return result


@app.post("/auth/login")
async def auth_login(username: str, password: str) -> Dict[str, object]:
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    now = int(time.time())
    token = jwt_encode({"sub": username, "iat": now, "exp": now + settings.jwt_ttl_seconds})
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.jwt_ttl_seconds}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    user_id: Optional[str] = None,
    current_user: Optional[str] = Depends(get_current_user_id),
) -> PredictResponse:
    # Read and validate image
    img = _read_image_from_upload(image)
    np_img = _pil_to_numpy(img)

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

    response = PredictResponse(
        predictions=predictions,
        image_width=img.width,
        image_height=img.height,
        detected_classes=[names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names),
    )

    # Optionally store history
    effective_user = user_id or current_user
    if effective_user:
        HISTORY.setdefault(effective_user, []).append(
            HistoryItem(user_id=effective_user, timestamp=datetime.utcnow(), predictions=predictions)
        )

    return response


@app.get("/history", response_model=List[HistoryItem])
async def get_history(user_id: Optional[str] = None, current_user: Optional[str] = Depends(get_current_user_id)) -> List[HistoryItem]:
    effective_user = user_id or current_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required or provide user_id")
    return HISTORY.get(effective_user, [])


@app.post("/history", response_model=HistoryItem)
async def post_history(item: HistoryPost, current_user: Optional[str] = Depends(get_current_user_id)) -> HistoryItem:
    effective_user = item.user_id or current_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required or provide user_id")
    history_item = HistoryItem(user_id=effective_user, timestamp=datetime.utcnow(), predictions=item.predictions)
    HISTORY.setdefault(effective_user, []).append(history_item)
    return history_item


# -----------------------------------------------------------------------------
# Notes
# - For production, replace in-memory HISTORY with a persistent database.
# - Secure endpoints with JWT (fastapi-jwt-auth or fastapi security utilities).
# - Add rate limiting, request size limits, and better input validation if needed.
# -----------------------------------------------------------------------------


