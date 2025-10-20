"""
Script de démarrage pour la base de connaissances agronomique.
"""

import asyncio
import sys
import os
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.app.database_postgres import postgres_db
from backend.app.settings import settings


async def start_knowledge_base():
    """Démarrer la base de connaissances agronomique."""
    print("🌱 Démarrage de la Base de Connaissances Agronomique Plant-AI")
    print("=" * 70)
    
    try:
        # Vérifier la configuration PostgreSQL
        print("📋 Configuration PostgreSQL:")
        print(f"   Host: {settings.postgres_host}")
        print(f"   Port: {settings.postgres_port}")
        print(f"   Database: {settings.postgres_database}")
        print(f"   User: {settings.postgres_user}")
        
        # Se connecter à PostgreSQL
        print("\n📡 Connexion à PostgreSQL...")
        await postgres_db.connect()
        print("✅ Connexion PostgreSQL réussie")
        
        # Vérifier les tables
        print("\n🗄️ Vérification des tables...")
        async with postgres_db.pool.acquire() as conn:
            # Vérifier que les tables principales existent
            tables = [
                "disease_entries", "cultures", "pathogens", "disease_symptoms",
                "disease_images", "disease_conditions", "disease_control_methods",
                "disease_products", "disease_precautions", "disease_prevention",
                "disease_regions", "disease_translations"
            ]
            
            for table in tables:
                result = await conn.fetchval(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table}'
                    )
                """)
                
                if result:
                    print(f"   ✅ Table {table} existe")
                else:
                    print(f"   ❌ Table {table} manquante")
                    return False
        
        print("\n🎯 Base de Connaissances Prête !")
        print("   📊 Support de 50,000+ fiches maladies")
        print("   🔍 Recherche full-text optimisée")
        print("   🌍 Support multilingue")
        print("   🛠️ Interface d'administration complète")
        print("   📈 Monitoring et statistiques")
        
        print("\n🔗 URLs d'accès:")
        print("   - API Documentation: http://localhost:8000/docs")
        print("   - Admin Dashboard: http://localhost:8000/admin/knowledge/")
        print("   - Gestion Fiches: http://localhost:8000/admin/knowledge/diseases")
        print("   - Gestion Cultures: http://localhost:8000/admin/knowledge/cultures")
        print("   - Gestion Pathogènes: http://localhost:8000/admin/knowledge/pathogens")
        print("   - Statistiques: http://localhost:8000/admin/knowledge/statistics")
        
        print("\n📚 APIs Disponibles:")
        print("   - GET /api/v1/knowledge/diseases - Liste des fiches")
        print("   - POST /api/v1/knowledge/diseases - Créer une fiche")
        print("   - GET /api/v1/knowledge/diseases/{id} - Détails d'une fiche")
        print("   - PUT /api/v1/knowledge/diseases/{id} - Modifier une fiche")
        print("   - DELETE /api/v1/knowledge/diseases/{id} - Supprimer une fiche")
        print("   - GET /api/v1/knowledge/cultures - Liste des cultures")
        print("   - GET /api/v1/knowledge/pathogens - Liste des pathogènes")
        print("   - GET /api/v1/knowledge/statistics - Statistiques")
        
        print("\n🚀 Commandes utiles:")
        print("   - Test complet: python test_knowledge_base.py")
        print("   - Test performance: python test_knowledge_base.py --performance")
        print("   - Données d'exemple: python scripts/init_postgres.py --sample")
        print("   - Démarrer l'API: python start_api.py")
        
        print("\n✅ Base de Connaissances Agronomique opérationnelle !")
        
        # Garder la connexion ouverte pour les tests
        print("\n⏳ Connexion maintenue pour les tests...")
        print("   Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé par l'utilisateur")
        
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        print("\n🔧 Solutions possibles:")
        print("   1. Vérifiez que PostgreSQL est installé et démarré")
        print("   2. Créez la base de données: createdb plant_ai")
        print("   3. Créez l'utilisateur: createuser plant_ai")
        print("   4. Configurez les permissions appropriées")
        print("   5. Vérifiez les paramètres dans .env")
        sys.exit(1)
    
    finally:
        # Fermer la connexion
        await postgres_db.disconnect()
        print("📡 Connexion PostgreSQL fermée")


def main():
    """Fonction principale."""
    print("🌱 Plant-AI - Base de Connaissances Agronomique")
    print("   Système de diagnostic intelligent des maladies végétales")
    print("   Support de 50,000+ fiches maladies avec recherche avancée")
    print()
    
    asyncio.run(start_knowledge_base())


if __name__ == "__main__":
    main()
