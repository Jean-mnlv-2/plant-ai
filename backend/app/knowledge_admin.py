"""
Interface d'administration pour la base de connaissances agronomique.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import status

from .knowledge_models import *
from .database_postgres import postgres_db
from .auth import get_current_user, require_role, UserRole
from .models import User

logger = logging.getLogger(__name__)

# Router pour l'administration de la base de connaissances
router = APIRouter(prefix="/admin/knowledge", tags=["admin", "knowledge"])

# Templates pour l'interface d'administration
templates = Jinja2Templates(directory="templates")


# =============================================================================
# PAGES D'ADMINISTRATION
# =============================================================================

@router.get("/", response_class=HTMLResponse)
async def knowledge_admin_dashboard(
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Page principale d'administration de la base de connaissances."""
    try:
        # Récupérer les statistiques
        stats = await get_knowledge_statistics()
        
        # Récupérer les dernières fiches créées
        recent_entries, _ = await postgres_db.get_disease_entries(limit=10, offset=0)
        
        return templates.TemplateResponse("admin/knowledge_dashboard.html", {
            "request": request,
            "user": current_user,
            "stats": stats,
            "recent_entries": recent_entries
        })
    except Exception as e:
        logger.error(f"Error loading knowledge admin dashboard: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement du tableau de bord")


@router.get("/diseases", response_class=HTMLResponse)
async def manage_diseases(
    request: Request,
    page: int = Query(1, ge=1, description="Numéro de page"),
    search: Optional[str] = Query(None, description="Recherche"),
    culture_id: Optional[int] = Query(None, description="Filtrer par culture"),
    pathogen_type: Optional[str] = Query(None, description="Filtrer par type de pathogène"),
    severity: Optional[str] = Query(None, description="Filtrer par sévérité"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Page de gestion des fiches maladies."""
    try:
        limit = 20
        offset = (page - 1) * limit
        
        # Récupérer les fiches maladies
        entries, total = await postgres_db.get_disease_entries(
            limit=limit,
            offset=offset,
            search=search,
            culture_id=culture_id,
            pathogen_type=pathogen_type,
            severity=severity
        )
        
        # Récupérer les cultures et pathogènes pour les filtres
        cultures = await postgres_db.get_cultures()
        pathogens = await postgres_db.get_pathogens()
        
        # Calculer la pagination
        total_pages = (total + limit - 1) // limit
        
        return templates.TemplateResponse("admin/knowledge_diseases.html", {
            "request": request,
            "user": current_user,
            "entries": entries,
            "cultures": cultures,
            "pathogens": pathogens,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_entries": total,
                "has_prev": page > 1,
                "has_next": page < total_pages
            },
            "filters": {
                "search": search,
                "culture_id": culture_id,
                "pathogen_type": pathogen_type,
                "severity": severity
            }
        })
    except Exception as e:
        logger.error(f"Error loading diseases management page: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement de la page de gestion")


@router.get("/diseases/create", response_class=HTMLResponse)
async def create_disease_form(
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Formulaire de création d'une fiche maladie."""
    try:
        # Récupérer les cultures et pathogènes
        cultures = await postgres_db.get_cultures()
        pathogens = await postgres_db.get_pathogens()
        
        return templates.TemplateResponse("admin/knowledge_disease_form.html", {
            "request": request,
            "user": current_user,
            "cultures": cultures,
            "pathogens": pathogens,
            "is_edit": False,
            "entry": None
        })
    except Exception as e:
        logger.error(f"Error loading disease creation form: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement du formulaire")


@router.get("/diseases/{disease_id}/edit", response_class=HTMLResponse)
async def edit_disease_form(
    disease_id: int,
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Formulaire d'édition d'une fiche maladie."""
    try:
        # Récupérer la fiche maladie
        entry = await postgres_db.get_disease_entry_by_id(disease_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Fiche maladie non trouvée")
        
        # Récupérer les cultures et pathogènes
        cultures = await postgres_db.get_cultures()
        pathogens = await postgres_db.get_pathogens()
        
        return templates.TemplateResponse("admin/knowledge_disease_form.html", {
            "request": request,
            "user": current_user,
            "cultures": cultures,
            "pathogens": pathogens,
            "is_edit": True,
            "entry": entry
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading disease edit form: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement du formulaire d'édition")


@router.get("/cultures", response_class=HTMLResponse)
async def manage_cultures(
    request: Request,
    search: Optional[str] = Query(None, description="Recherche"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Page de gestion des cultures."""
    try:
        cultures = await postgres_db.get_cultures(search)
        
        return templates.TemplateResponse("admin/knowledge_cultures.html", {
            "request": request,
            "user": current_user,
            "cultures": cultures,
            "search": search
        })
    except Exception as e:
        logger.error(f"Error loading cultures management page: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement de la page des cultures")


@router.get("/pathogens", response_class=HTMLResponse)
async def manage_pathogens(
    request: Request,
    search: Optional[str] = Query(None, description="Recherche"),
    pathogen_type: Optional[str] = Query(None, description="Filtrer par type"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Page de gestion des pathogènes."""
    try:
        pathogens = await postgres_db.get_pathogens(search, pathogen_type)
        
        return templates.TemplateResponse("admin/knowledge_pathogens.html", {
            "request": request,
            "user": current_user,
            "pathogens": pathogens,
            "search": search,
            "pathogen_type": pathogen_type
        })
    except Exception as e:
        logger.error(f"Error loading pathogens management page: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement de la page des pathogènes")


@router.get("/statistics", response_class=HTMLResponse)
async def knowledge_statistics(
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Page des statistiques de la base de connaissances."""
    try:
        stats = await get_knowledge_statistics()
        
        return templates.TemplateResponse("admin/knowledge_statistics.html", {
            "request": request,
            "user": current_user,
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Error loading knowledge statistics page: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du chargement des statistiques")


# =============================================================================
# ACTIONS D'ADMINISTRATION
# =============================================================================

@router.post("/diseases/create")
async def create_disease_action(
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Action de création d'une fiche maladie."""
    try:
        form_data = await request.form()
        
        # Extraire les données du formulaire
        entry_data = extract_disease_form_data(form_data)
        
        # Créer la fiche maladie
        disease_id = await postgres_db.create_disease_entry(entry_data)
        
        return RedirectResponse(
            url=f"/admin/knowledge/diseases/{disease_id}/edit?created=true",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except Exception as e:
        logger.error(f"Error creating disease entry: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création de la fiche maladie")


@router.post("/diseases/{disease_id}/update")
async def update_disease_action(
    disease_id: int,
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Action de mise à jour d'une fiche maladie."""
    try:
        form_data = await request.form()
        
        # Extraire les données du formulaire
        entry_data = extract_disease_form_data(form_data)
        
        # Mettre à jour la fiche maladie
        success = await postgres_db.update_disease_entry(disease_id, entry_data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")
        
        return RedirectResponse(
            url=f"/admin/knowledge/diseases/{disease_id}/edit?updated=true",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating disease entry {disease_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour de la fiche maladie")


@router.post("/diseases/{disease_id}/delete")
async def delete_disease_action(
    disease_id: int,
    request: Request,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Action de suppression d'une fiche maladie."""
    try:
        form_data = await request.form()
        confirm = form_data.get("confirm") == "true"
        reason = form_data.get("reason", "")
        
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation requise pour la suppression"
            )
        
        # Supprimer la fiche maladie
        success = await postgres_db.delete_disease_entry(disease_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Erreur lors de la suppression")
        
        return RedirectResponse(
            url="/admin/knowledge/diseases?deleted=true",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting disease entry {disease_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de la fiche maladie")


@router.post("/cultures/create")
async def create_culture_action(
    name: str = Form(...),
    scientific_name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Action de création d'une culture."""
    try:
        culture_id = await postgres_db.create_culture(
            name=name,
            scientific_name=scientific_name,
            category=category,
            description=description
        )
        
        return RedirectResponse(
            url="/admin/knowledge/cultures?created=true",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except Exception as e:
        logger.error(f"Error creating culture: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création de la culture")


@router.post("/pathogens/create")
async def create_pathogen_action(
    scientific_name: str = Form(...),
    common_name: Optional[str] = Form(None),
    pathogen_type: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Action de création d'un pathogène."""
    try:
        pathogen_id = await postgres_db.create_pathogen(
            scientific_name=scientific_name,
            common_name=common_name,
            pathogen_type=pathogen_type,
            description=description
        )
        
        return RedirectResponse(
            url="/admin/knowledge/pathogens?created=true",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except Exception as e:
        logger.error(f"Error creating pathogen: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création du pathogène")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

async def get_knowledge_statistics() -> Dict[str, Any]:
    """Récupérer les statistiques de la base de connaissances."""
    try:
        # Récupérer les statistiques de base
        entries, total = await postgres_db.get_disease_entries(limit=1, offset=0)
        
        # TODO: Implémenter des requêtes plus complexes pour les statistiques
        return {
            "total_entries": total,
            "active_entries": total,
            "entries_by_culture": {},
            "entries_by_pathogen_type": {},
            "entries_by_severity": {},
            "recent_entries": 0
        }
    except Exception as e:
        logger.error(f"Error getting knowledge statistics: {e}")
        return {
            "total_entries": 0,
            "active_entries": 0,
            "entries_by_culture": {},
            "entries_by_pathogen_type": {},
            "entries_by_severity": {},
            "recent_entries": 0
        }


def extract_disease_form_data(form_data) -> Dict[str, Any]:
    """Extraire les données d'une fiche maladie depuis un formulaire."""
    try:
        # Données de base
        entry_data = {
            "culture_id": int(form_data.get("culture_id", 0)),
            "pathogen_id": int(form_data.get("pathogen_id", 0)),
            "scientific_name": form_data.get("scientific_name", ""),
            "common_name": form_data.get("common_name", ""),
            "severity_level": form_data.get("severity_level", "medium"),
            "treatment_urgency": form_data.get("treatment_urgency", "planned")
        }
        
        # Symptômes
        symptoms = []
        symptom_descriptions = form_data.getlist("symptom_description")
        symptom_severities = form_data.getlist("symptom_severity")
        symptom_parts = form_data.getlist("symptom_affected_part")
        
        for i, description in enumerate(symptom_descriptions):
            if description.strip():
                symptoms.append({
                    "description": description,
                    "severity": symptom_severities[i] if i < len(symptom_severities) else "medium",
                    "affected_part": symptom_parts[i] if i < len(symptom_parts) else None
                })
        
        entry_data["symptoms"] = symptoms
        
        # Images
        images = []
        image_urls = form_data.getlist("image_url")
        image_types = form_data.getlist("image_type")
        image_descriptions = form_data.getlist("image_description")
        image_primary = form_data.getlist("image_primary")
        
        for i, url in enumerate(image_urls):
            if url.strip():
                images.append({
                    "image_url": url,
                    "image_type": image_types[i] if i < len(image_types) else "symptom",
                    "description": image_descriptions[i] if i < len(image_descriptions) else None,
                    "is_primary": str(i) in image_primary
                })
        
        entry_data["images"] = images
        
        # Conditions
        if form_data.get("temperature_min") or form_data.get("temperature_max"):
            entry_data["conditions"] = {
                "temperature_min": float(form_data.get("temperature_min", 0)) if form_data.get("temperature_min") else None,
                "temperature_max": float(form_data.get("temperature_max", 0)) if form_data.get("temperature_max") else None,
                "humidity_min": int(form_data.get("humidity_min", 0)) if form_data.get("humidity_min") else None,
                "humidity_max": int(form_data.get("humidity_max", 0)) if form_data.get("humidity_max") else None,
                "soil_type": form_data.get("soil_type"),
                "seasonality": form_data.get("seasonality"),
                "climate_conditions": form_data.get("climate_conditions")
            }
        
        # Méthodes de lutte
        control_methods = []
        method_types = form_data.getlist("control_method_type")
        method_descriptions = form_data.getlist("control_method_description")
        method_effectiveness = form_data.getlist("control_method_effectiveness")
        method_costs = form_data.getlist("control_method_cost")
        
        for i, method_type in enumerate(method_types):
            if method_type and method_descriptions[i].strip():
                control_methods.append({
                    "method_type": method_type,
                    "description": method_descriptions[i],
                    "effectiveness": method_effectiveness[i] if i < len(method_effectiveness) else "medium",
                    "cost_level": method_costs[i] if i < len(method_costs) else "medium"
                })
        
        entry_data["control_methods"] = control_methods
        
        # Produits
        products = []
        product_names = form_data.getlist("product_name")
        product_ingredients = form_data.getlist("product_ingredient")
        product_dosages = form_data.getlist("product_dosage")
        product_methods = form_data.getlist("product_method")
        product_safety = form_data.getlist("product_safety")
        product_registration = form_data.getlist("product_registration")
        
        for i, name in enumerate(product_names):
            if name.strip():
                products.append({
                    "product_name": name,
                    "active_ingredient": product_ingredients[i] if i < len(product_ingredients) else None,
                    "dosage": product_dosages[i] if i < len(product_dosages) else None,
                    "application_method": product_methods[i] if i < len(product_methods) else None,
                    "safety_class": product_safety[i] if i < len(product_safety) else None,
                    "registration_number": product_registration[i] if i < len(product_registration) else None
                })
        
        entry_data["products"] = products
        
        # Précautions
        if form_data.get("risk_level"):
            entry_data["precautions"] = {
                "risk_level": form_data.get("risk_level"),
                "safety_period_days": int(form_data.get("safety_period_days", 0)) if form_data.get("safety_period_days") else None,
                "dosage_instructions": form_data.get("dosage_instructions"),
                "contraindications": form_data.get("contraindications"),
                "protective_equipment": form_data.get("protective_equipment")
            }
        
        # Prévention
        prevention = []
        prevention_types = form_data.getlist("prevention_type")
        prevention_descriptions = form_data.getlist("prevention_description")
        prevention_timing = form_data.getlist("prevention_timing")
        prevention_effectiveness = form_data.getlist("prevention_effectiveness")
        
        for i, prevention_type in enumerate(prevention_types):
            if prevention_type and prevention_descriptions[i].strip():
                prevention.append({
                    "practice_type": prevention_type,
                    "description": prevention_descriptions[i],
                    "timing": prevention_timing[i] if i < len(prevention_timing) else None,
                    "effectiveness": prevention_effectiveness[i] if i < len(prevention_effectiveness) else "medium"
                })
        
        entry_data["prevention"] = prevention
        
        # Régions
        regions = []
        region_names = form_data.getlist("region_name")
        region_countries = form_data.getlist("region_country")
        region_climates = form_data.getlist("region_climate")
        
        for i, name in enumerate(region_names):
            if name.strip():
                regions.append({
                    "region_name": name,
                    "country": region_countries[i] if i < len(region_countries) else None,
                    "climate_zone": region_climates[i] if i < len(region_climates) else None
                })
        
        entry_data["regions"] = regions
        
        return entry_data
    
    except Exception as e:
        logger.error(f"Error extracting disease form data: {e}")
        raise HTTPException(status_code=400, detail="Erreur dans les données du formulaire")
