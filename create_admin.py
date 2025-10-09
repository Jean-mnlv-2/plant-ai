"""
Script pour créer un utilisateur administrateur local.
"""
import sys
import os
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.app.auth import create_user, UserRole
from backend.app.database import db

def create_admin_user():
    """Créer un utilisateur administrateur."""
    print("Creation d'un utilisateur administrateur...")
    print("=" * 50)
    
    try:
        # Données de l'admin
        admin_data = {
            "name": "Admin Plant-AI",
            "email": "admin@plant-ai.local",
            "password": "Admin123!",
            "country": "France",
            "role": UserRole.ADMIN
        }
        
        # Créer l'utilisateur admin
        user = create_user(
            name=admin_data["name"],
            email=admin_data["email"],
            password=admin_data["password"],
            country=admin_data["country"],
            role=admin_data["role"]
        )
        
        print("SUCCESS: Utilisateur administrateur cree avec succes !")
        print(f"Email: {user.email}")
        print(f"Mot de passe: {admin_data['password']}")
        print(f"Role: {user.role}")
        print(f"ID: {user.id}")
        print("=" * 50)
        print("Vous pouvez maintenant vous connecter sur:")
        print("   - API: http://localhost:8000")
        print("   - Swagger: http://localhost:8000/docs")
        print("   - Admin Dashboard: http://localhost:8000/admin/")
        print("=" * 50)
        
        return user
        
    except Exception as e:
        if "already exists" in str(e):
            print("WARNING: L'utilisateur administrateur existe deja.")
            print("Email: admin@plant-ai.local")
            print("Mot de passe: Admin123!")
        else:
            print(f"ERROR: Erreur lors de la creation de l'admin: {e}")
        return None

def test_admin_login():
    """Tester la connexion de l'admin."""
    print("\nTest de connexion de l'admin...")
    
    try:
        from backend.app.auth import authenticate_user, generate_tokens
        
        # Authentifier l'admin
        user = authenticate_user("admin@plant-ai.local", "Admin123!")
        if user:
            tokens = generate_tokens(user.id, user.role)
            print("SUCCESS: Connexion admin reussie !")
            print(f"Token d'acces: {tokens.accessToken[:50]}...")
            return tokens.accessToken
        else:
            print("ERROR: Echec de la connexion admin")
            return None
    except Exception as e:
        print(f"ERROR: Erreur de connexion: {e}")
        return None

def main():
    """Fonction principale."""
    print("Plant-AI - Creation d'Administrateur Local")
    print("=" * 50)
    
    # Créer l'admin
    admin = create_admin_user()
    
    if admin:
        # Tester la connexion
        token = test_admin_login()
        
        if token:
            print("\nSUCCESS: Configuration terminee avec succes !")
            print("\nInformations de connexion:")
            print("   Email: admin@plant-ai.local")
            print("   Mot de passe: Admin123!")
            print("   Role: ADMIN")
            
            print("\nLiens utiles:")
            print("   - Swagger UI: http://localhost:8000/docs")
            print("   - ReDoc: http://localhost:8000/redoc")
            print("   - Admin Dashboard: http://localhost:8000/admin/")
            print("   - Health Check: http://localhost:8000/health")
            
            print("\nTest API avec cURL:")
            print(f'curl -X POST http://localhost:8000/api/v1/auth/login \\')
            print(f'  -H "Content-Type: application/json" \\')
            print(f'  -d \'{{"email": "admin@plant-ai.local", "password": "Admin123!"}}\'')
        else:
            print("\nWARNING: L'admin a ete cree mais la connexion a echoue.")
    else:
        print("\nERROR: Echec de la creation de l'administrateur.")

if __name__ == "__main__":
    main()
