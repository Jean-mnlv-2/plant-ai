"""
Script de démarrage pour l'API Plant-AI.
"""
import uvicorn
import os
import sys
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def main():
    """Démarrer l'API Plant-AI."""
    print("🚀 Starting Plant-AI API Server")
    print("=" * 50)
    
    # Vérifier les variables d'environnement
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found. Using default settings.")
        print("   Copy env.example to .env and configure your settings.")
    
    # Configuration du serveur
    host = os.getenv("PLANT_AI_HOST", "127.0.0.1")
    port = int(os.getenv("PLANT_AI_PORT", "8000"))
    reload = os.getenv("PLANT_AI_RELOAD", "true").lower() == "true"
    
    print(f"📍 Server will start at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Admin Dashboard: http://{host}:{port}/admin/")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print("=" * 50)
    
    try:
        # Démarrer le serveur
        uvicorn.run(
            "backend.app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


