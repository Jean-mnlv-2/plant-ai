from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from .database import db
from .metrics import performance_monitor, monitor_performance
from .main import get_current_user_id
from .settings import settings


templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/admin", tags=["admin-ui"])


def _require_auth(request: Request):
    token = request.headers.get("Authorization")
    user = get_current_user_id(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


@router.get("/", response_class=HTMLResponse)
@monitor_performance("admin_dashboard_page")
async def admin_dashboard(request: Request, user: str = Depends(_require_auth)):
    stats = db.get_performance_stats(hours=24)
    model_usage = db.get_model_usage_stats(days=7)
    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "stats": stats,
            "model_usage": model_usage,
            "settings": settings,
        },
    )


@router.get("/uncertain", response_class=HTMLResponse)
@monitor_performance("admin_uncertain_page")
async def admin_uncertain(request: Request, page: int = 1, size: int = 20, user: str = Depends(_require_auth)):
    total = len(db.list_uncertain_cases(100000))
    start = max(0, (page - 1) * size)
    end = start + size
    items = db.list_uncertain_cases(end)[start:end]
    return templates.TemplateResponse(
        "admin/uncertain.html",
        {"request": request, "items": items, "page": page, "size": size, "total": total},
    )


@router.get("/feedback", response_class=HTMLResponse)
@monitor_performance("admin_feedback_page")
async def admin_feedback(request: Request, page: int = 1, size: int = 20, user: str = Depends(_require_auth)):
    total = len(db.get_all_history(100000))
    start = max(0, (page - 1) * size)
    end = start + size
    history = db.get_all_history(end)[start:end]
    return templates.TemplateResponse(
        "admin/feedback.html",
        {"request": request, "history": history, "page": page, "size": size, "total": total},
    )


@router.get("/stats", response_class=HTMLResponse)
@monitor_performance("admin_stats_page")
async def admin_stats(request: Request, user: str = Depends(_require_auth)):
    perf = db.get_performance_stats(hours=24)
    usage = db.get_model_usage_stats(days=7)
    return templates.TemplateResponse(
        "admin/stats.html",
        {"request": request, "performance": perf, "usage": usage},
    )


@router.get("/model", response_class=HTMLResponse)
@monitor_performance("admin_model_page")
async def admin_model(request: Request, user: str = Depends(_require_auth)):
    # lazy import to avoid loading model for page
    model_info = {
        "model_path": str(settings.model_file),
        "model_exists": settings.model_file.exists(),
    }
    return templates.TemplateResponse(
        "admin/model.html",
        {"request": request, "model": model_info, "settings": settings},
    )


@router.get("/users", response_class=HTMLResponse)
@monitor_performance("admin_users_page")
async def admin_users(request: Request, user: str = Depends(_require_auth)):
    users = db.list_users()
    return templates.TemplateResponse(
        "admin/users.html",
        {"request": request, "users": users},
    )


