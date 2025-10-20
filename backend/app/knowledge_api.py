"""
API endpoints pour la gestion de la base de connaissances agronomique.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import StreamingResponse
import io
import csv
import json

from .knowledge_models import *
from .database_postgres import postgres_db
from .auth import get_current_user, require_role, UserRole
from .models import User

logger = logging.getLogger(__name__)

# Router pour les APIs de base de connaissances
router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


# =============================================================================
# ENDPOINTS POUR LES FICHES MALADIES
# =============================================================================

@router.get("/diseases", response_model=DiseaseEntryListResponse)
async def get_disease_entries(
    limit: int = Query(20, ge=1, le=100, description="Nombre d'éléments par page"),
    offset: int = Query(0, ge=0, description="Décalage pour la pagination"),
    search: Optional[str] = Query(None, description="Recherche textuelle"),
    culture_id: Optional[int] = Query(None, description="Filtrer par culture"),
    pathogen_type: Optional[PathogenType] = Query(None, description="Filtrer par type de pathogène"),
    severity: Optional[SeverityLevel] = Query(None, description="Filtrer par sévérité"),
    urgency: Optional[TreatmentUrgency] = Query(None, description="Filtrer par urgence"),
    language: str = Query("fr", description="Langue pour les traductions"),
    current_user: User = Depends(get_current_user)
):
    """Récupérer la liste des fiches maladies avec filtres et pagination."""
    try:
        entries, total = await postgres_db.get_disease_entries(
            limit=limit,
            offset=offset,
            search=search,
            culture_id=culture_id,
            pathogen_type=pathogen_type,
            severity=severity,
            language=language
        )
        
        return DiseaseEntryListResponse(
            entries=entries,
            pagination={
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            },
            filters={
                "search": search,
                "culture_id": culture_id,
                "pathogen_type": pathogen_type,
                "severity": severity,
                "urgency": urgency,
                "language": language
            }
        )
    except Exception as e:
        logger.error(f"Error getting disease entries: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des fiches maladies")


@router.get("/diseases/{disease_id}", response_model=DiseaseEntryResponse)
async def get_disease_entry(
    disease_id: int,
    language: str = Query("fr", description="Langue pour les traductions"),
    current_user: User = Depends(get_current_user)
):
    """Récupérer une fiche maladie complète par ID."""
    try:
        entry = await postgres_db.get_disease_entry_by_id(disease_id, language)
        if not entry:
            raise HTTPException(status_code=404, detail="Fiche maladie non trouvée")
        
        return DiseaseEntryResponse(entry=entry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting disease entry {disease_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de la fiche maladie")


@router.post("/diseases", response_model=DiseaseEntryCreateResponse)
async def create_disease_entry(
    entry_data: DiseaseEntryCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Créer une nouvelle fiche maladie."""
    try:
        # Convertir le modèle Pydantic en dictionnaire
        entry_dict = entry_data.dict()
        entry_dict['created_by'] = current_user.id
        
        # Créer l'entrée dans la base de données
        disease_id = await postgres_db.create_disease_entry(entry_dict)
        
        return DiseaseEntryCreateResponse(
            entry_id=disease_id,
            message="Fiche maladie créée avec succès"
        )
    except Exception as e:
        logger.error(f"Error creating disease entry: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création de la fiche maladie")


@router.put("/diseases/{disease_id}", response_model=DiseaseEntryUpdateResponse)
async def update_disease_entry(
    disease_id: int,
    entry_data: DiseaseEntryUpdate,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Mettre à jour une fiche maladie."""
    try:
        # Vérifier que la fiche existe
        existing_entry = await postgres_db.get_disease_entry_by_id(disease_id)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="Fiche maladie non trouvée")
        
        # Convertir le modèle Pydantic en dictionnaire
        entry_dict = entry_data.dict(exclude_unset=True)
        
        # Mettre à jour l'entrée
        success = await postgres_db.update_disease_entry(disease_id, entry_dict)
        
        if not success:
            raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour")
        
        return DiseaseEntryUpdateResponse(
            message="Fiche maladie mise à jour avec succès"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating disease entry {disease_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour de la fiche maladie")


@router.delete("/diseases/{disease_id}", response_model=DiseaseEntryDeleteResponse)
async def delete_disease_entry(
    disease_id: int,
    confirm: bool = Query(False, description="Confirmation de suppression"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Supprimer une fiche maladie (soft delete)."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Confirmation requise pour la suppression"
            )
        
        # Vérifier que la fiche existe
        existing_entry = await postgres_db.get_disease_entry_by_id(disease_id)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="Fiche maladie non trouvée")
        
        # Supprimer l'entrée (soft delete)
        success = await postgres_db.delete_disease_entry(disease_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Erreur lors de la suppression")
        
        return DiseaseEntryDeleteResponse(
            message="Fiche maladie supprimée avec succès"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting disease entry {disease_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de la fiche maladie")


@router.delete("/diseases/{disease_id}/hard", response_model=DiseaseEntryDeleteResponse)
async def hard_delete_disease_entry(
    disease_id: int,
    confirm: bool = Query(False, description="Confirmation de suppression définitive"),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Supprimer définitivement une fiche maladie."""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Confirmation requise pour la suppression définitive"
            )
        
        # Vérifier que la fiche existe
        existing_entry = await postgres_db.get_disease_entry_by_id(disease_id)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="Fiche maladie non trouvée")
        
        # Supprimer définitivement l'entrée
        success = await postgres_db.hard_delete_disease_entry(disease_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Erreur lors de la suppression définitive")
        
        return DiseaseEntryDeleteResponse(
            message="Fiche maladie supprimée définitivement"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error hard deleting disease entry {disease_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression définitive")


# =============================================================================
# ENDPOINTS POUR LES CULTURES
# =============================================================================

@router.get("/cultures", response_model=CultureListResponse)
async def get_cultures(
    search: Optional[str] = Query(None, description="Recherche dans les cultures"),
    current_user: User = Depends(get_current_user)
):
    """Récupérer la liste des cultures."""
    try:
        cultures = await postgres_db.get_cultures(search)
        return CultureListResponse(cultures=cultures)
    except Exception as e:
        logger.error(f"Error getting cultures: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des cultures")


@router.post("/cultures", response_model=Dict[str, Any])
async def create_culture(
    culture_data: CultureCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Créer une nouvelle culture."""
    try:
        culture_id = await postgres_db.create_culture(
            name=culture_data.name,
            scientific_name=culture_data.scientific_name,
            category=culture_data.category,
            description=culture_data.description
        )
        
        return {
            "success": True,
            "culture_id": culture_id,
            "message": "Culture créée avec succès"
        }
    except Exception as e:
        logger.error(f"Error creating culture: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création de la culture")


# =============================================================================
# ENDPOINTS POUR LES PATHOGÈNES
# =============================================================================

@router.get("/pathogens", response_model=PathogenListResponse)
async def get_pathogens(
    search: Optional[str] = Query(None, description="Recherche dans les pathogènes"),
    pathogen_type: Optional[PathogenType] = Query(None, description="Filtrer par type"),
    current_user: User = Depends(get_current_user)
):
    """Récupérer la liste des pathogènes."""
    try:
        pathogens = await postgres_db.get_pathogens(search, pathogen_type)
        return PathogenListResponse(pathogens=pathogens)
    except Exception as e:
        logger.error(f"Error getting pathogens: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des pathogènes")


@router.post("/pathogens", response_model=Dict[str, Any])
async def create_pathogen(
    pathogen_data: PathogenCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Créer un nouveau pathogène."""
    try:
        pathogen_id = await postgres_db.create_pathogen(
            scientific_name=pathogen_data.scientific_name,
            common_name=pathogen_data.common_name,
            pathogen_type=pathogen_data.type,
            description=pathogen_data.description
        )
        
        return {
            "success": True,
            "pathogen_id": pathogen_id,
            "message": "Pathogène créé avec succès"
        }
    except Exception as e:
        logger.error(f"Error creating pathogen: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création du pathogène")


# =============================================================================
# ENDPOINTS POUR LES STATISTIQUES
# =============================================================================

@router.get("/statistics", response_model=KnowledgeBaseStats)
async def get_knowledge_base_statistics(
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Récupérer les statistiques de la base de connaissances."""
    try:
        # TODO: Implémenter la récupération des statistiques
        # Pour l'instant, retourner des données de démonstration
        statistics = DiseaseStatistics(
            total_entries=0,
            active_entries=0,
            entries_by_culture={},
            entries_by_pathogen_type={},
            entries_by_severity={},
            recent_entries=0
        )
        
        return KnowledgeBaseStats(
            statistics=statistics,
            last_updated=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des statistiques")


# =============================================================================
# ENDPOINTS POUR L'IMPORT/EXPORT
# =============================================================================

@router.post("/import", response_model=ImportResult)
async def import_disease_entries(
    entries: List[DiseaseEntryImport],
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Importer des fiches maladies depuis un fichier."""
    try:
        # TODO: Implémenter l'import
        return ImportResult(
            success=True,
            imported_count=0,
            errors=[],
            warnings=[]
        )
    except Exception as e:
        logger.error(f"Error importing disease entries: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'import")


@router.get("/export")
async def export_disease_entries(
    format: str = Query("json", description="Format d'export"),
    search: Optional[str] = Query(None, description="Filtrer par recherche"),
    culture_id: Optional[int] = Query(None, description="Filtrer par culture"),
    current_user: User = Depends(get_current_user)
):
    """Exporter les fiches maladies."""
    try:
        # Récupérer les données
        entries, total = await postgres_db.get_disease_entries(
            limit=10000,  # Limite pour l'export
            offset=0,
            search=search,
            culture_id=culture_id
        )
        
        if format == "csv":
            # Créer un CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # En-têtes
            writer.writerow([
                "ID", "Nom Scientifique", "Nom Commun", "Culture", "Pathogène",
                "Sévérité", "Urgence", "Créé le"
            ])
            
            # Données
            for entry in entries:
                writer.writerow([
                    entry.get('id'),
                    entry.get('scientific_name'),
                    entry.get('common_name'),
                    entry.get('culture_name'),
                    entry.get('pathogen_name'),
                    entry.get('severity_level'),
                    entry.get('treatment_urgency'),
                    entry.get('created_at')
                ])
            
            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=disease_entries.csv"}
            )
        
        else:  # JSON par défaut
            return {
                "success": True,
                "entries": entries,
                "total": total,
                "exported_at": datetime.utcnow().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error exporting disease entries: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'export")


# =============================================================================
# ENDPOINTS POUR LA RECHERCHE AVANCÉE
# =============================================================================

@router.post("/search", response_model=DiseaseEntryListResponse)
async def search_disease_entries(
    search_request: DiseaseEntrySearchRequest,
    current_user: User = Depends(get_current_user)
):
    """Recherche avancée dans les fiches maladies."""
    try:
        filters = search_request.filters
        entries, total = await postgres_db.get_disease_entries(
            limit=search_request.limit,
            offset=search_request.offset,
            search=filters.search,
            culture_id=filters.culture_id,
            pathogen_type=filters.pathogen_type,
            severity=filters.severity,
            language=filters.language
        )
        
        return DiseaseEntryListResponse(
            entries=entries,
            pagination={
                "total": total,
                "limit": search_request.limit,
                "offset": search_request.offset,
                "has_more": search_request.offset + search_request.limit < total
            },
            filters=filters.dict()
        )
    except Exception as e:
        logger.error(f"Error searching disease entries: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la recherche")


# =============================================================================
# ENDPOINTS POUR LA GESTION EN MASSE
# =============================================================================

@router.post("/bulk-delete", response_model=Dict[str, Any])
async def bulk_delete_disease_entries(
    delete_request: BulkDeleteRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Supprimer plusieurs fiches maladies en masse."""
    try:
        if not delete_request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirmation requise pour la suppression en masse"
            )
        
        deleted_count = 0
        errors = []
        
        for disease_id in delete_request.disease_ids:
            try:
                success = await postgres_db.delete_disease_entry(disease_id)
                if success:
                    deleted_count += 1
                else:
                    errors.append(f"Impossible de supprimer la fiche {disease_id}")
            except Exception as e:
                errors.append(f"Erreur lors de la suppression de la fiche {disease_id}: {str(e)}")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "total_requested": len(delete_request.disease_ids),
            "errors": errors,
            "message": f"{deleted_count} fiches supprimées avec succès"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk delete: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression en masse")
