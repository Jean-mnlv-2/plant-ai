"""
Service de base de connaissances des maladies pour l'API Plant-AI.
"""
from typing import Dict, List, Any, Optional
from .models import DiseaseDetail, SeverityLevel, TreatmentUrgency


class DiseasesService:
    """Service pour gérer la base de connaissances des maladies."""
    
    def __init__(self):
        self.diseases_db = self._initialize_diseases_database()
    
    def _initialize_diseases_database(self) -> Dict[str, DiseaseDetail]:
        """Initialiser la base de données des maladies."""
        diseases = {
            "mildiou-tomate": DiseaseDetail(
                id="mildiou-tomate",
                name="Mildiou de la tomate",
                category="diseases",
                symptoms=[
                    "Taches brunes sur les feuilles",
                    "Pourriture des fruits",
                    "Feuilles qui se dessèchent",
                    "Taches huileuses sur les feuilles"
                ],
                solutions=[
                    "Traitement au cuivre (bouillie bordelaise)",
                    "Améliorer l'aération des plants",
                    "Éviter l'arrosage sur les feuilles",
                    "Supprimer les parties atteintes"
                ],
                prevention=[
                    "Rotation des cultures",
                    "Espacement adéquat entre les plants",
                    "Paillage pour éviter les éclaboussures",
                    "Choisir des variétés résistantes"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/mildiou-tomate-1.jpg",
                    "https://api.plant-ai.com/diseases/mildiou-tomate-2.jpg"
                ],
                severity=SeverityLevel.HIGH,
                affectedPlants=["tomato", "potato", "pepper"],
                seasonality=["spring", "summer"],
                geographicDistribution=["europe", "north_america"],
                treatmentUrgency=TreatmentUrgency.IMMEDIATE,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "oïdium-tomate": DiseaseDetail(
                id="oïdium-tomate",
                name="Oïdium de la tomate",
                category="diseases",
                symptoms=[
                    "Poudre blanche sur les feuilles",
                    "Feuilles qui jaunissent",
                    "Croissance ralentie",
                    "Fruits déformés"
                ],
                solutions=[
                    "Traitement au soufre",
                    "Améliorer la circulation d'air",
                    "Réduire l'humidité",
                    "Traitement fongicide"
                ],
                prevention=[
                    "Éviter l'humidité excessive",
                    "Espacement des plants",
                    "Rotation des cultures",
                    "Variétés résistantes"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/oïdium-tomate-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "cucumber", "squash"],
                seasonality=["summer", "fall"],
                geographicDistribution=["europe", "north_america", "asia"],
                treatmentUrgency=TreatmentUrgency.URGENT,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "anthracnose-tomate": DiseaseDetail(
                id="anthracnose-tomate",
                name="Anthracnose de la tomate",
                category="diseases",
                symptoms=[
                    "Taches circulaires noires sur les fruits",
                    "Pourriture des fruits mûrs",
                    "Taches sur les feuilles",
                    "Fruits qui se décomposent"
                ],
                solutions=[
                    "Supprimer les fruits atteints",
                    "Traitement fongicide",
                    "Améliorer le drainage",
                    "Éviter l'humidité excessive"
                ],
                prevention=[
                    "Rotation des cultures",
                    "Éviter l'arrosage par aspersion",
                    "Paillage",
                    "Récolte précoce"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/anthracnose-tomate-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "pepper", "eggplant"],
                seasonality=["summer", "fall"],
                geographicDistribution=["europe", "north_america"],
                treatmentUrgency=TreatmentUrgency.URGENT,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "bactériose-tomate": DiseaseDetail(
                id="bactériose-tomate",
                name="Bactériose de la tomate",
                category="diseases",
                symptoms=[
                    "Taches noires sur les feuilles",
                    "Pourriture des tiges",
                    "Fruits qui pourrissent",
                    "Feuilles qui se flétrissent"
                ],
                solutions=[
                    "Supprimer les plants atteints",
                    "Désinfecter les outils",
                    "Améliorer le drainage",
                    "Traitement bactéricide"
                ],
                prevention=[
                    "Rotation des cultures",
                    "Éviter l'humidité excessive",
                    "Désinfection des semences",
                    "Variétés résistantes"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/bactériose-tomate-1.jpg"
                ],
                severity=SeverityLevel.HIGH,
                affectedPlants=["tomato", "pepper"],
                seasonality=["spring", "summer"],
                geographicDistribution=["europe", "north_america"],
                treatmentUrgency=TreatmentUrgency.IMMEDIATE,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "carence-azote": DiseaseDetail(
                id="carence-azote",
                name="Carence en azote",
                category="nutrient_deficiency",
                symptoms=[
                    "Feuilles jaunissent (chlorose)",
                    "Croissance ralentie",
                    "Feuilles plus petites",
                    "Tiges fines et fragiles"
                ],
                solutions=[
                    "Apport d'engrais azoté",
                    "Compost organique",
                    "Engrais liquide",
                    "Amélioration du sol"
                ],
                prevention=[
                    "Rotation avec légumineuses",
                    "Paillage organique",
                    "Compost régulier",
                    "Éviter la sur-fertilisation"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/carence-azote-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "lettuce", "spinach", "cabbage"],
                seasonality=["spring", "summer"],
                geographicDistribution=["global"],
                treatmentUrgency=TreatmentUrgency.PLANNED,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "carence-phosphore": DiseaseDetail(
                id="carence-phosphore",
                name="Carence en phosphore",
                category="nutrient_deficiency",
                symptoms=[
                    "Feuilles vert foncé avec taches violettes",
                    "Croissance très ralentie",
                    "Racines peu développées",
                    "Floraison retardée"
                ],
                solutions=[
                    "Apport de phosphate",
                    "Compost riche en phosphore",
                    "Engrais organique",
                    "Amélioration du pH du sol"
                ],
                prevention=[
                    "Analyse du sol régulière",
                    "Compost équilibré",
                    "Rotation des cultures",
                    "Éviter l'acidification"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/carence-phosphore-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "pepper", "eggplant", "potato"],
                seasonality=["spring", "summer"],
                geographicDistribution=["global"],
                treatmentUrgency=TreatmentUrgency.PLANNED,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "carence-potassium": DiseaseDetail(
                id="carence-potassium",
                name="Carence en potassium",
                category="nutrient_deficiency",
                symptoms=[
                    "Bords des feuilles qui brûlent",
                    "Feuilles qui s'enroulent",
                    "Fruits de petite taille",
                    "Résistance réduite aux maladies"
                ],
                solutions=[
                    "Apport de potasse",
                    "Cendre de bois",
                    "Compost de bananes",
                    "Engrais potassique"
                ],
                prevention=[
                    "Compost équilibré",
                    "Rotation avec cultures riches en K",
                    "Paillage organique",
                    "Éviter la lixiviation"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/carence-potassium-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "potato", "pepper", "cucumber"],
                seasonality=["spring", "summer"],
                geographicDistribution=["global"],
                treatmentUrgency=TreatmentUrgency.PLANNED,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "pucerons": DiseaseDetail(
                id="pucerons",
                name="Pucerons",
                category="pests",
                symptoms=[
                    "Insectes verts ou noirs sur les feuilles",
                    "Feuilles qui s'enroulent",
                    "Miellat sur les feuilles",
                    "Fourmis sur les plants"
                ],
                solutions=[
                    "Traitement insecticide naturel",
                    "Savon noir",
                    "Prédateurs naturels (coccinelles)",
                    "Jus de tabac"
                ],
                prevention=[
                    "Plantes répulsives (basilic, menthe)",
                    "Éviter la sur-fertilisation",
                    "Rotation des cultures",
                    "Surveillance régulière"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/pucerons-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "pepper", "eggplant", "cucumber"],
                seasonality=["spring", "summer", "fall"],
                geographicDistribution=["global"],
                treatmentUrgency=TreatmentUrgency.URGENT,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "aleurodes": DiseaseDetail(
                id="aleurodes",
                name="Aleurodes (mouches blanches)",
                category="pests",
                symptoms=[
                    "Petites mouches blanches sur les feuilles",
                    "Feuilles qui jaunissent",
                    "Miellat collant",
                    "Fumagine noire"
                ],
                solutions=[
                    "Pièges jaunes collants",
                    "Savon insecticide",
                    "Prédateurs naturels",
                    "Traitement à l'huile de neem"
                ],
                prevention=[
                    "Filets anti-insectes",
                    "Plantes répulsives",
                    "Éviter l'humidité excessive",
                    "Surveillance précoce"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/aleurodes-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "pepper", "eggplant", "cucumber"],
                seasonality=["summer", "fall"],
                geographicDistribution=["global"],
                treatmentUrgency=TreatmentUrgency.URGENT,
                lastUpdated="2024-01-10T10:00:00Z"
            ),
            "thrips": DiseaseDetail(
                id="thrips",
                name="Thrips",
                category="pests",
                symptoms=[
                    "Petites taches argentées sur les feuilles",
                    "Feuilles déformées",
                    "Fleurs qui ne s'ouvrent pas",
                    "Fruits déformés"
                ],
                solutions=[
                    "Traitement insecticide",
                    "Pièges bleus",
                    "Prédateurs naturels",
                    "Savon insecticide"
                ],
                prevention=[
                    "Filets anti-insectes",
                    "Éviter la sur-fertilisation",
                    "Rotation des cultures",
                    "Surveillance régulière"
                ],
                images=[
                    "https://api.plant-ai.com/diseases/thrips-1.jpg"
                ],
                severity=SeverityLevel.MEDIUM,
                affectedPlants=["tomato", "pepper", "eggplant", "onion"],
                seasonality=["spring", "summer"],
                geographicDistribution=["global"],
                treatmentUrgency=TreatmentUrgency.URGENT,
                lastUpdated="2024-01-10T10:00:00Z"
            )
        }
        return diseases
    
    def search_diseases(self, search: Optional[str] = None, category: Optional[str] = None, 
                       limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Rechercher des maladies dans la base de données."""
        results = list(self.diseases_db.values())
        
        # Filtrer par recherche textuelle
        if search:
            search_lower = search.lower()
            results = [
                disease for disease in results
                if (search_lower in disease.name.lower() or
                    search_lower in disease.id.lower() or
                    any(search_lower in symptom.lower() for symptom in disease.symptoms))
            ]
        
        # Filtrer par catégorie
        if category:
            results = [disease for disease in results if disease.category == category]
        
        # Pagination
        total = len(results)
        results = results[offset:offset + limit]
        
        return {
            "success": True,
            "diseases": results,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "hasMore": offset + limit < total
            }
        }
    
    def get_disease_by_id(self, disease_id: str) -> Optional[DiseaseDetail]:
        """Récupérer une maladie par son ID."""
        return self.diseases_db.get(disease_id)
    
    def get_diseases_by_plant(self, plant_type: str) -> List[DiseaseDetail]:
        """Récupérer les maladies affectant un type de plante."""
        return [
            disease for disease in self.diseases_db.values()
            if plant_type.lower() in [plant.lower() for plant in disease.affectedPlants]
        ]
    
    def get_diseases_by_severity(self, severity: SeverityLevel) -> List[DiseaseDetail]:
        """Récupérer les maladies par niveau de sévérité."""
        return [
            disease for disease in self.diseases_db.values()
            if disease.severity == severity
        ]
    
    def get_diseases_by_urgency(self, urgency: TreatmentUrgency) -> List[DiseaseDetail]:
        """Récupérer les maladies par urgence de traitement."""
        return [
            disease for disease in self.diseases_db.values()
            if disease.treatmentUrgency == urgency
        ]
    
    def get_categories(self) -> List[str]:
        """Récupérer toutes les catégories de maladies."""
        categories = set(disease.category for disease in self.diseases_db.values())
        return sorted(list(categories))
    
    def get_affected_plants(self) -> List[str]:
        """Récupérer toutes les plantes affectées."""
        plants = set()
        for disease in self.diseases_db.values():
            plants.update(disease.affectedPlants)
        return sorted(list(plants))


# Instance globale du service des maladies
diseases_service = DiseasesService()


