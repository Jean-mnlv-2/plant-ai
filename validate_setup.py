"""
Script de validation pour vérifier que l'installation est correcte.
"""
import sys
import os
from pathlib import Path

def check_python_version():
    """Vérifier la version de Python."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8+ is required")
        return False
    print(f"SUCCESS: Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    """Vérifier les dépendances critiques."""
    print("\nChecking critical dependencies...")
    critical_deps = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'ultralytics',
        'PIL',
        'numpy',
        'sqlite3'
    ]
    
    missing_deps = []
    for dep in critical_deps:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'sqlite3':
                import sqlite3
            else:
                __import__(dep)
            print(f"  OK {dep}")
        except ImportError:
            print(f"  FAIL {dep} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nERROR: Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("SUCCESS: All critical dependencies are installed")
    return True

def check_file_structure():
    """Vérifier la structure des fichiers."""
    print("\nChecking file structure...")
    required_files = [
        "backend/app/main.py",
        "backend/app/settings.py",
        "backend/app/database.py",
        "backend/app/auth.py",
        "backend/app/models.py",
        "requirements.txt",
        "env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  OK {file_path}")
        else:
            print(f"  FAIL {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nERROR: Missing files: {', '.join(missing_files)}")
        return False
    
    print("SUCCESS: All required files are present")
    return True

def check_model_file():
    """Vérifier le fichier modèle."""
    print("\nChecking model file...")
    model_path = Path("models/yolov8_best.pt")
    
    if model_path.exists():
        print(f"  OK Model file exists: {model_path}")
        print(f"  OK Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print(f"  FAIL Model file missing: {model_path}")
        print("  Run: python create_test_model.py")
        return False

def check_environment():
    """Vérifier la configuration d'environnement."""
    print("\nChecking environment configuration...")
    
    # Vérifier si .env existe
    env_file = Path(".env")
    if env_file.exists():
        print("  OK .env file exists")
    else:
        print("  WARNING .env file missing - using defaults")
        print("  Run: cp env.example .env")
    
    # Vérifier les variables critiques
    jwt_secret = os.getenv("JWT_SECRET")
    if jwt_secret and jwt_secret != "your-secret-key-change-in-production":
        print("  OK JWT_SECRET is configured")
    else:
        print("  WARNING JWT_SECRET not configured or using default")
        print("  Set JWT_SECRET in .env file")
    
    return True

def check_database():
    """Vérifier la base de données."""
    print("\nChecking database...")
    
    try:
        import sqlite3
        db_path = Path("data/plant_ai.db")
        db_path.parent.mkdir(exist_ok=True)
        
        # Tester la connexion
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        
        print(f"  OK Database connection successful")
        print(f"  OK Database file: {db_path}")
        print(f"  OK Tables found: {len(tables)}")
        
        return True
    except Exception as e:
        print(f"  FAIL Database error: {e}")
        return False

def main():
    """Fonction principale de validation."""
    print("Plant-AI Setup Validation")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_file_structure,
        check_model_file,
        check_environment,
        check_database
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"ERROR in {check.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("SUCCESS: ALL CHECKS PASSED!")
        print("\nYour Plant-AI setup is ready!")
        print("\nNext steps:")
        print("1. python create_admin.py")
        print("2. python start_api.py")
        print("3. python quick_test.py")
    else:
        print(f"FAILED: {total - passed} check(s) failed")
        print("\nPlease fix the issues above before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
