"""
Script d'initialisation de la base de données PostgreSQL pour Plant-AI.
"""

import asyncio
import sys
import os
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.app.database_postgres import postgres_db
from backend.app.settings import settings


async def init_database():
    """Initialiser la base de données PostgreSQL."""
    print("🌱 Initialisation de la base de données PostgreSQL pour Plant-AI")
    print("=" * 60)
    
    try:
        # Se connecter à PostgreSQL
        print("📡 Connexion à PostgreSQL...")
        await postgres_db.connect()
        print("✅ Connexion réussie")
        
        # Les tables sont créées automatiquement lors de la connexion
        print("✅ Base de données initialisée avec succès")
        
        # Afficher les informations de connexion
        print("\n📊 Informations de connexion:")
        print(f"   Host: {settings.postgres_host}")
        print(f"   Port: {settings.postgres_port}")
        print(f"   Database: {settings.postgres_database}")
        print(f"   User: {settings.postgres_user}")
        
        print("\n🎉 Base de données prête pour la base de connaissances agronomique !")
        print("   - Support de 50,000+ fiches maladies")
        print("   - Recherche full-text optimisée")
        print("   - Support multilingue")
        print("   - Interface d'administration complète")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        print("\n🔧 Vérifiez que PostgreSQL est installé et configuré correctement.")
        print("   - PostgreSQL doit être en cours d'exécution")
        print("   - La base de données doit exister")
        print("   - L'utilisateur doit avoir les permissions appropriées")
        sys.exit(1)
    
    finally:
        # Fermer la connexion
        await postgres_db.disconnect()


async def test_connection():
    """Tester la connexion à PostgreSQL."""
    print("🧪 Test de connexion PostgreSQL...")
    
    try:
        await postgres_db.connect()
        
        # Test simple
        async with postgres_db.pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                print("✅ Test de connexion réussi")
            else:
                print("❌ Test de connexion échoué")
                return False
        
        await postgres_db.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False


async def create_sample_data():
    """Créer des données d'exemple pour tester."""
    print("\n📝 Création de données d'exemple...")
    
    try:
        await postgres_db.connect()
        
        # Créer quelques cultures
        cultures = [
            ("Tomate", "Solanum lycopersicum", "Légume", "Plante potagère très cultivée"),
            ("Maïs", "Zea mays", "Céréale", "Céréale de base dans de nombreuses régions"),
            ("Blé", "Triticum aestivum", "Céréale", "Céréale la plus cultivée au monde"),
            ("Pomme", "Malus domestica", "Arbre fruitier", "Arbre fruitier très répandu"),
            ("Riz", "Oryza sativa", "Céréale", "Céréale de base en Asie")
        ]
        
        culture_ids = []
        for name, scientific, category, description in cultures:
            culture_id = await postgres_db.create_culture(name, scientific, category, description)
            culture_ids.append(culture_id)
            print(f"   ✅ Culture créée: {name}")
        
        # Créer quelques pathogènes
        pathogens = [
            ("Phytophthora infestans", "Mildiou", "fungus", "Champignon responsable du mildiou"),
            ("Puccinia graminis", "Rouille", "fungus", "Champignon responsable de la rouille"),
            ("Xanthomonas oryzae", "Bactériose", "bacteria", "Bactérie pathogène du riz"),
            ("Tobacco mosaic virus", "Virus de la mosaïque", "virus", "Virus très répandu"),
            ("Meloidogyne incognita", "Nématode", "parasite", "Nématode des racines")
        ]
        
        pathogen_ids = []
        for scientific, common, ptype, description in pathogens:
            pathogen_id = await postgres_db.create_pathogen(scientific, common, ptype, description)
            pathogen_ids.append(pathogen_id)
            print(f"   ✅ Pathogène créé: {scientific}")
        
        # Créer une fiche maladie d'exemple
        sample_disease = {
            "culture_id": culture_ids[0],  # Tomate
            "pathogen_id": pathogen_ids[0],  # Phytophthora infestans
            "scientific_name": "Mildiou de la tomate",
            "common_name": "Mildiou",
            "severity_level": "high",
            "treatment_urgency": "immediate",
            "symptoms": [
                {
                    "description": "Taches brunes sur les feuilles",
                    "severity": "high",
                    "affected_part": "feuilles"
                },
                {
                    "description": "Pourriture des fruits",
                    "severity": "high",
                    "affected_part": "fruits"
                }
            ],
            "images": [
                {
                    "image_url": "https://example.com/mildiou-tomate-1.jpg",
                    "image_type": "symptom",
                    "description": "Taches sur les feuilles",
                    "is_primary": True
                }
            ],
            "conditions": {
                "temperature_min": 15.0,
                "temperature_max": 25.0,
                "humidity_min": 80,
                "humidity_max": 100,
                "soil_type": "Humide",
                "seasonality": "Printemps, Été",
                "climate_conditions": "Humidité élevée et températures modérées"
            },
            "control_methods": [
                {
                    "method_type": "cultural",
                    "description": "Rotation des cultures",
                    "effectiveness": "high",
                    "cost_level": "low"
                },
                {
                    "method_type": "chemical",
                    "description": "Traitement fongicide au cuivre",
                    "effectiveness": "high",
                    "cost_level": "medium"
                }
            ],
            "products": [
                {
                    "product_name": "Bouillie bordelaise",
                    "active_ingredient": "Sulfate de cuivre",
                    "dosage": "2-3 kg/ha",
                    "application_method": "Pulvérisation foliaire",
                    "safety_class": "Classe 2",
                    "registration_number": "FR-123456"
                }
            ],
            "precautions": {
                "risk_level": "medium",
                "safety_period_days": 7,
                "dosage_instructions": "Respecter les doses recommandées",
                "contraindications": "Ne pas traiter par temps chaud",
                "protective_equipment": "Gants, masque, lunettes"
            },
            "prevention": [
                {
                    "practice_type": "cultural",
                    "description": "Éviter l'humidité excessive",
                    "timing": "Toute la saison",
                    "effectiveness": "high"
                }
            ],
            "regions": [
                {
                    "region_name": "Europe",
                    "country": "France",
                    "climate_zone": "Tempéré",
                    "latitude_min": 42.0,
                    "latitude_max": 51.0,
                    "longitude_min": -5.0,
                    "longitude_max": 8.0
                }
            ],
            "translations": [
                {
                    "language_code": "en",
                    "field_name": "common_name",
                    "translated_text": "Late blight"
                }
            ]
        }
        
        disease_id = await postgres_db.create_disease_entry(sample_disease)
        print(f"   ✅ Fiche maladie créée: {sample_disease['common_name']} (ID: {disease_id})")
        
        await postgres_db.disconnect()
        print("\n🎉 Données d'exemple créées avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des données d'exemple: {e}")
        await postgres_db.disconnect()


def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialisation PostgreSQL pour Plant-AI")
    parser.add_argument("--test", action="store_true", help="Tester seulement la connexion")
    parser.add_argument("--sample", action="store_true", help="Créer des données d'exemple")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_connection())
    elif args.sample:
        asyncio.run(create_sample_data())
    else:
        asyncio.run(init_database())


if __name__ == "__main__":
    main()
